/*
 * Copyright 2024-2026 Embabel Pty Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.embabel.agent.anthropic

import com.anthropic.core.RequestOptions
import com.anthropic.core.http.HttpClient
import com.anthropic.core.http.HttpMethod
import com.anthropic.core.http.HttpRequest
import com.anthropic.core.http.HttpRequestBody
import com.anthropic.core.http.HttpResponse
import com.anthropic.core.http.Interceptor
import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.databind.node.ArrayNode
import com.fasterxml.jackson.databind.node.JsonNodeFactory
import com.fasterxml.jackson.databind.node.ObjectNode
import org.slf4j.LoggerFactory
import java.security.MessageDigest
import java.util.Collections
import java.util.concurrent.CompletableFuture

/**
 * Sends each tool-calling assistant turn back to Claude exactly as Claude returned it.
 *
 * Spring AI rebuilds an assistant turn from its joined text, its thinking blocks and its tool
 * calls, always in that order, so a returned `[thinking, text, tool_use]` goes back as
 * `[text, thinking, tool_use]`. Claude Fable 5.1, Opus 5.5 and Sonnet 5.5 reject any edit to an
 * earlier turn with a 400 (platform.claude.com/docs/en/build-with-claude/preserved-thinking#prefix-check).
 * This interceptor records the content of each Messages response that calls tools and puts it
 * back in place of the rebuilt turn, matched on the turn's tool_use ids.
 *
 * A thinking block is also signed against the system prompt, the tools and the messages before
 * it, and embabel changes those mid-loop (dynamic tool injection, history transforms). Each
 * record keeps a digest of that prefix; a turn whose prefix no longer matches counts as unproven.
 *
 * A turn carrying thinking that it has no record of (evicted, streamed, or from before a
 * restart), or whose prefix changed, can't be proven unchanged, so its thinking blocks and every earlier one are left out instead: the API accepts
 * thinking dropped from the start of the history, at the cost of that reasoning.
 *
 * ponytail: remembers the last [MAX_TURNS] tool-calling turns per client, in memory and only from
 * non-streamed responses. Persist them alongside the conversation if long sessions lose reasoning.
 */
internal class AssistantTurnReplay : Interceptor {

    private val logger = LoggerFactory.getLogger(AssistantTurnReplay::class.java)

    /** What Claude returned, and a digest of the prefix it was produced under. */
    private class Turn(val content: ArrayNode, val prefix: String)

    private val turns: MutableMap<String, Turn> = Collections.synchronizedMap(
        object : LinkedHashMap<String, Turn>() {
            override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, Turn>) = size > MAX_TURNS
        }
    )

    override fun intercept(httpClient: HttpClient): HttpClient = object : HttpClient {
        override fun execute(request: HttpRequest, requestOptions: RequestOptions): HttpResponse {
            if (!isCreateMessage(request)) {
                return httpClient.execute(request, requestOptions)
            }
            val (sent, prefix) = replay(request)
            return record(httpClient.execute(sent, requestOptions), prefix)
        }

        override fun executeAsync(
            request: HttpRequest,
            requestOptions: RequestOptions,
        ): CompletableFuture<HttpResponse> =
            // Streamed responses aren't recorded, so this only strips thinking it can't prove.
            httpClient.executeAsync(if (isCreateMessage(request)) replay(request).first else request, requestOptions)

        override fun close() = httpClient.close()
    }

    private fun isCreateMessage(request: HttpRequest) =
        request.method == HttpMethod.POST && request.pathSegments == listOf("v1", "messages")

    /**
     * Returns the request to send, and the digest of its whole prefix, which is what the turn
     * Claude answers with is produced under.
     */
    private fun replay(request: HttpRequest): Pair<HttpRequest, String?> {
        val buffered = request.buffered()
        val body = buffered.body?.json(JsonNode::class.java) as? ObjectNode ?: return buffered to null
        val messages = body["messages"]?.filterIsInstance<ObjectNode>().orEmpty()
        var changed = false
        var lastUnproven = -1
        messages.forEachIndexed { index, message ->
            val content = message["content"] as? ArrayNode
            if (message["role"]?.asText() != "assistant" || content == null) {
                return@forEachIndexed
            }
            val ids = toolUseIds(content)
            val returned = ids.firstOrNull()?.let { turns[it] }?.takeIf { toolUseIds(it.content) == ids }
            if (returned != null && returned.content != content) {
                message.set<JsonNode>("content", returned.content.deepCopy())
                changed = true
            }
            // Unproven: no record (Spring AI's rebuild groups all text, then all thinking, then all
            // tool calls, so it may not match what Claude returned), or the system prompt, tools or
            // an earlier message changed since Claude produced the turn.
            val proven = returned != null && returned.prefix == digest(body, messages, index)
            if (!proven && (message["content"] as ArrayNode).any { it.isThinking() }) {
                lastUnproven = index
            }
        }
        // Dropping thinking from the start of the history is allowed; editing what precedes it is not.
        if (lastUnproven >= 0) {
            logger.warn(
                "Cannot prove an earlier assistant turn goes back to Claude as it was produced, so thinking up to that turn is left out"
            )
            messages.take(lastUnproven + 1).forEach { message ->
                (message["content"] as? ArrayNode)?.removeAll { it.isThinking() }
            }
            changed = true
        }
        val prefix = digest(body, messages, messages.size)
        if (!changed) {
            return buffered to prefix
        }
        return buffered.toBuilder().body(HttpRequestBody.ofJson(body)).build() to prefix
    }

    /**
     * Digest of what a thinking block is signed against: the system prompt, the tools and the
     * messages before it. Earlier thinking and `cache_control` markers are not part of it.
     */
    private fun digest(body: ObjectNode, messages: List<ObjectNode>, count: Int): String {
        val prefix = JsonNodeFactory.instance.objectNode()
        body["system"]?.let { prefix.set<JsonNode>("system", it.deepCopy()) }
        body["tools"]?.let { prefix.set<JsonNode>("tools", it.deepCopy()) }
        prefix.putArray("messages").addAll(messages.take(count).map { it.deepCopy() })
        // Markers sit on system blocks, tools and content blocks (tool results nest blocks of their
        // own). Anywhere else, such as inside a tool's input_schema, cache_control is the caller's data.
        val blocks = prefix["messages"].flatMap { it["content"] as? ArrayNode ?: emptyList() }
        (listOf(prefix["system"], prefix["tools"]).flatMap { it as? ArrayNode ?: emptyList() } +
            blocks + blocks.flatMap { it["content"] as? ArrayNode ?: emptyList() })
            .forEach { (it as? ObjectNode)?.remove("cache_control") }
        prefix["messages"].forEach { (it["content"] as? ArrayNode)?.removeAll { block -> block.isThinking() } }
        return MessageDigest.getInstance("SHA-256").digest(MAPPER.writeValueAsBytes(prefix))
            .joinToString("") { "%02x".format(it) }
    }

    private fun JsonNode.isThinking() = this["type"]?.asText() in THINKING_TYPES

    private fun record(response: HttpResponse, prefix: String?): HttpResponse {
        val isJson = response.headers().values("Content-Type").any { it.startsWith("application/json") }
        if (response.statusCode() != 200 || !isJson || prefix == null) {
            return response
        }
        val buffered = response.buffered()
        val content = runCatching { buffered.json(JsonNode::class.java)["content"] as? ArrayNode }.getOrNull()
        content?.let { toolUseIds(it).firstOrNull()?.let { id -> turns[id] = Turn(it, prefix) } }
        return buffered
    }

    private fun toolUseIds(content: ArrayNode): List<String> =
        content.filter { it["type"]?.asText() == "tool_use" }.mapNotNull { it["id"]?.asText() }

    companion object {
        const val MAX_TURNS = 1_000
        private val THINKING_TYPES = setOf("thinking", "redacted_thinking")
        private val MAPPER = ObjectMapper()
    }
}
