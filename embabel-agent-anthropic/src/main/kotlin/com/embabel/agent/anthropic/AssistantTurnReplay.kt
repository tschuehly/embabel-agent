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
import com.fasterxml.jackson.databind.node.ArrayNode
import com.fasterxml.jackson.databind.node.ObjectNode
import org.slf4j.LoggerFactory
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
 * A rebuilt turn it has no record of (evicted, streamed, or from before a restart) can't be
 * restored, so its thinking blocks and every earlier one are left out instead: the API accepts
 * thinking dropped from the start of the history, at the cost of that reasoning.
 *
 * ponytail: remembers the last [MAX_TURNS] tool-calling turns per client, in memory and only from
 * non-streamed responses. Persist them alongside the conversation if long sessions lose reasoning.
 */
internal class AssistantTurnReplay : Interceptor {

    private val logger = LoggerFactory.getLogger(AssistantTurnReplay::class.java)

    private val turns: MutableMap<String, ArrayNode> = Collections.synchronizedMap(
        object : LinkedHashMap<String, ArrayNode>() {
            override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, ArrayNode>) = size > MAX_TURNS
        }
    )

    override fun intercept(httpClient: HttpClient): HttpClient = object : HttpClient {
        override fun execute(request: HttpRequest, requestOptions: RequestOptions): HttpResponse {
            if (!isCreateMessage(request)) {
                return httpClient.execute(request, requestOptions)
            }
            return record(httpClient.execute(replay(request), requestOptions))
        }

        override fun executeAsync(
            request: HttpRequest,
            requestOptions: RequestOptions,
        ): CompletableFuture<HttpResponse> = httpClient.executeAsync(request, requestOptions)

        override fun close() = httpClient.close()
    }

    private fun isCreateMessage(request: HttpRequest) =
        request.method == HttpMethod.POST && request.pathSegments == listOf("v1", "messages")

    private fun replay(request: HttpRequest): HttpRequest {
        val buffered = request.buffered()
        val body = buffered.body?.json(JsonNode::class.java) ?: return buffered
        val assistantTurns = body["messages"]?.filterIsInstance<ObjectNode>()
            ?.filter { it["role"]?.asText() == "assistant" && it["content"] is ArrayNode }
            .orEmpty()
        var changed = false
        var lastUnreplayable = -1
        assistantTurns.forEachIndexed { index, message ->
            val content = message["content"] as ArrayNode
            val ids = toolUseIds(content)
            val returned = ids.firstOrNull()?.let { turns[it] }
            if (returned != null && toolUseIds(returned) == ids) {
                if (returned != content) {
                    message.set<JsonNode>("content", returned.deepCopy())
                    changed = true
                }
            } else if (mayBeReordered(content)) {
                lastUnreplayable = index
            }
        }
        // A turn not recorded here (evicted, streamed, or from before a restart) can't be restored.
        // Dropping thinking from the start of the history is allowed; editing it is not.
        if (lastUnreplayable >= 0) {
            logger.warn(
                "Cannot send an earlier assistant turn back as Claude returned it, so thinking up to that turn is left out"
            )
            assistantTurns.take(lastUnreplayable + 1).forEach { message ->
                (message["content"] as ArrayNode).removeAll { it["type"]?.asText() in THINKING_TYPES }
            }
            changed = true
        }
        if (!changed) {
            return buffered
        }
        return buffered.toBuilder().body(HttpRequestBody.ofJson(body)).build()
    }

    /**
     * Whether Spring AI's rebuild may differ from what Claude returned: the rebuild puts all text
     * ahead of all thinking, which matches Claude's order only when there is no text.
     */
    private fun mayBeReordered(content: ArrayNode): Boolean =
        content.any { it["type"]?.asText() in THINKING_TYPES } && content.any { it["type"]?.asText() == "text" }

    private fun record(response: HttpResponse): HttpResponse {
        val isJson = response.headers().values("Content-Type").any { it.startsWith("application/json") }
        if (response.statusCode() != 200 || !isJson) {
            return response
        }
        val buffered = response.buffered()
        val content = runCatching { buffered.json(JsonNode::class.java)["content"] as? ArrayNode }.getOrNull()
        content?.let { toolUseIds(it).firstOrNull()?.let { id -> turns[id] = it } }
        return buffered
    }

    private fun toolUseIds(content: ArrayNode): List<String> =
        content.filter { it["type"]?.asText() == "tool_use" }.mapNotNull { it["id"]?.asText() }

    companion object {
        const val MAX_TURNS = 1_000
        private val THINKING_TYPES = setOf("thinking", "redacted_thinking")
    }
}
