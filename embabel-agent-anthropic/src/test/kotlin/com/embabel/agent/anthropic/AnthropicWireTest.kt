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

import com.embabel.agent.api.models.AnthropicModels
import com.embabel.agent.spi.loop.streaming.LlmInferenceStreamEvent
import com.embabel.chat.AssistantMessageWithToolCalls
import com.embabel.chat.ToolResultMessage
import com.embabel.chat.UserMessage
import com.embabel.common.ai.model.LlmOptions
import com.embabel.common.ai.model.Thinking
import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.ObjectMapper
import com.sun.net.httpserver.HttpServer
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Nested
import org.junit.jupiter.api.Test
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ValueSource
import java.net.InetSocketAddress

/**
 * Drives a real [org.springframework.ai.anthropic.AnthropicChatModel] against a local stub of
 * the Messages API, so both the request body Anthropic receives and Spring AI's own parsing of
 * the response are exercised.
 */
class AnthropicWireTest {

    private lateinit var server: HttpServer
    private var responseContent = """[{"type":"text","text":"Hi"}]"""
    private lateinit var request: JsonNode

    @BeforeEach
    fun setUp() {
        server = HttpServer.create(InetSocketAddress(0), 0)
        server.createContext("/v1/messages") { exchange ->
            request = ObjectMapper().readTree(exchange.requestBody.use { it.readBytes() })
            val streamed = request["stream"]?.asBoolean() == true
            val body = (if (streamed) sse(responseContent) else """
                {"id":"msg_test","type":"message","role":"assistant","content":$responseContent,
                 "model":"test","stop_reason":"end_turn","stop_sequence":null,
                 "usage":{"input_tokens":5,"output_tokens":2}}
            """.trimIndent()).toByteArray()
            exchange.responseHeaders.set("Content-Type", if (streamed) "text/event-stream" else "application/json")
            exchange.sendResponseHeaders(200, body.size.toLong())
            exchange.responseBody.use { it.write(body) }
        }
        server.start()
    }

    @AfterEach
    fun tearDown() {
        server.stop(0)
    }

    /** The Messages streaming events that deliver [content]. */
    private fun sse(content: String): String {
        val mapper = ObjectMapper()
        fun event(type: String, data: String) = "event: $type\ndata: $data\n\n"
        return buildString {
            append(event("message_start", """{"type":"message_start","message":{"id":"msg_test","type":"message","role":"assistant","content":[],"model":"test","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":5,"output_tokens":1}}}"""))
            mapper.readTree(content).forEachIndexed { index, block ->
                val (start, delta) = when (block["type"].asText()) {
                    "thinking" -> """{"type":"thinking","thinking":"","signature":""}""" to
                        """{"type":"signature_delta","signature":${mapper.writeValueAsString(block["signature"].asText())}}"""
                    "tool_use" -> """{"type":"tool_use","id":"${block["id"].asText()}","name":"${block["name"].asText()}","input":{}}""" to
                        """{"type":"input_json_delta","partial_json":${mapper.writeValueAsString(block["input"].toString())}}"""
                    else -> """{"type":"text","text":""}""" to
                        """{"type":"text_delta","text":${mapper.writeValueAsString(block["text"].asText())}}"""
                }
                append(event("content_block_start", """{"type":"content_block_start","index":$index,"content_block":$start}"""))
                append(event("content_block_delta", """{"type":"content_block_delta","index":$index,"delta":$delta}"""))
                append(event("content_block_stop", """{"type":"content_block_stop","index":$index}"""))
            }
            append(event("message_delta", """{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":2}}"""))
            append(event("message_stop", """{"type":"message_stop"}"""))
        }
    }

    private fun blockTypes(content: JsonNode) = content.map { it["type"].asText() }

    private fun service(model: String) =
        AnthropicModelFactory(apiKey = "test-key", baseUrl = "http://localhost:${server.address.port}")
            .build(model)

    private fun send(model: String, options: LlmOptions = LlmOptions()) =
        service(model)
            .createMessageSender(options)
            .call(listOf(UserMessage("Reply READY")), emptyList())

    /**
     * Claude 5 thinks by default and, with display "omitted", returns an empty thinking block
     * before the answer. Spring AI turns each block into its own generation.
     */
    @Test
    fun `answer text survives an empty thinking block ahead of it`() {
        responseContent = """[{"type":"thinking","thinking":"","signature":"sig"},{"type":"text","text":"READY"}]"""

        val response = send(AnthropicModels.CLAUDE_OPUS_5_5)

        assertEquals("READY", response.textContent)
        assertEquals("READY", response.message.content)
    }

    @Test
    fun `summarized thinking does not leak into the answer`() {
        responseContent = """[{"type":"thinking","thinking":"Let me think.","signature":"sig"},{"type":"text","text":"READY"}]"""

        assertEquals("READY", send(AnthropicModels.CLAUDE_SONNET_4_6).textContent)
    }

    @Test
    fun `summarized thinking does not leak into a tool-calling turn`() {
        responseContent = """[{"type":"thinking","thinking":"Let me think.","signature":"sig"},
            {"type":"text","text":"Looking it up."},
            {"type":"tool_use","id":"toolu_1","name":"lookup","input":{}}]"""

        val response = send(AnthropicModels.CLAUDE_SONNET_4_6)

        assertEquals("Looking it up.", response.textContent)
        assertEquals("lookup", (response.message as AssistantMessageWithToolCalls).toolCalls.single().name)
    }

    /**
     * Claude Fable 5.1, Opus 5.5 and Sonnet 5.5 reject a request whose earlier assistant turn
     * differs from what they returned, reordered blocks included
     * (platform.claude.com/docs/en/build-with-claude/preserved-thinking#prefix-check).
     */
    @Test
    fun `a tool-calling turn goes back to Claude exactly as it was returned`() {
        val returned = """[{"type":"thinking","thinking":"","signature":"sig1"},
            {"type":"text","text":"Looking it up."},
            {"type":"thinking","thinking":"","signature":"sig2"},
            {"type":"tool_use","id":"toolu_1","name":"lookup","input":{"q":"x"}}]"""
        responseContent = returned
        val sender = service(AnthropicModels.CLAUDE_FABLE_5_1).createMessageSender(LlmOptions())
        val user = UserMessage("Look it up")

        val first = sender.call(listOf(user), emptyList())
        responseContent = """[{"type":"text","text":"Done"}]"""
        sender.call(listOf(user, first.message, ToolResultMessage("toolu_1", "lookup", "found")), emptyList())

        assertEquals("assistant", request["messages"][1]["role"].asText())
        assertEquals(ObjectMapper().readTree(returned), request["messages"][1]["content"])
    }

    /**
     * A turn this client never saw (another instance, a restart) can't be restored. Thinking
     * dropped from the start of the history is accepted where a reordered turn is not.
     */
    @Test
    fun `a tool-calling turn this client did not record goes back without its thinking`() {
        responseContent = """[{"type":"thinking","thinking":"","signature":"sig1"},
            {"type":"text","text":"Looking it up."},
            {"type":"tool_use","id":"toolu_1","name":"lookup","input":{}}]"""
        val user = UserMessage("Look it up")
        val first = service(AnthropicModels.CLAUDE_FABLE_5_1).createMessageSender(LlmOptions())
            .call(listOf(user), emptyList())

        responseContent = """[{"type":"text","text":"Done"}]"""
        service(AnthropicModels.CLAUDE_FABLE_5_1).createMessageSender(LlmOptions())
            .call(listOf(user, first.message, ToolResultMessage("toolu_1", "lookup", "found")), emptyList())

        assertEquals(
            listOf("text", "tool_use"),
            request["messages"][1]["content"].map { it["type"].asText() },
        )
    }

    /**
     * A streamed turn isn't recorded, so the Spring AI rebuild of it must not go back with its
     * thinking moved behind the text.
     */
    @Test
    fun `a streamed tool-calling turn goes back without its thinking rather than reordered`() {
        responseContent = """[{"type":"thinking","thinking":"","signature":"sig1"},
            {"type":"text","text":"Looking it up."},
            {"type":"tool_use","id":"toolu_1","name":"lookup","input":{"q":"x"}}]"""
        val streamer = service(AnthropicModels.CLAUDE_FABLE_5_1).createMessageStreamer(LlmOptions())
        val user = UserMessage("Look it up")

        val first = streamer.streamInference(listOf(user), emptyList())
            .ofType(LlmInferenceStreamEvent.Complete::class.java).blockLast()!!
        responseContent = """[{"type":"text","text":"Done"}]"""
        streamer.streamInference(
            listOf(user, first.message, ToolResultMessage("toolu_1", "lookup", "found")), emptyList(),
        ).blockLast()

        assertEquals(listOf("text", "tool_use"), blockTypes(request["messages"][1]["content"]))
    }

    /** Spring AI groups thinking ahead of tool calls, so even a turn without text can be reordered. */
    @Test
    fun `an unrecorded turn without text goes back without its thinking`() {
        responseContent = """[{"type":"thinking","thinking":"","signature":"sigA"},
            {"type":"tool_use","id":"toolu_a","name":"lookup","input":{}},
            {"type":"thinking","thinking":"","signature":"sigB"},
            {"type":"tool_use","id":"toolu_b","name":"lookup","input":{}}]"""
        val user = UserMessage("Look both up")
        val first = service(AnthropicModels.CLAUDE_FABLE_5_1).createMessageSender(LlmOptions())
            .call(listOf(user), emptyList())

        responseContent = """[{"type":"text","text":"Done"}]"""
        service(AnthropicModels.CLAUDE_FABLE_5_1).createMessageSender(LlmOptions()).call(
            listOf(
                user, first.message,
                ToolResultMessage("toolu_a", "lookup", "a"), ToolResultMessage("toolu_b", "lookup", "b"),
            ),
            emptyList(),
        )

        assertEquals(listOf("tool_use", "tool_use"), blockTypes(request["messages"][1]["content"]))
    }

    /**
     * Per-model behaviour from platform.claude.com/docs/en/build-with-claude/thinking
     * ("Configuring thinking" table and "Sampling parameters").
     */
    @Nested
    inner class ThinkingRequest {

        @ParameterizedTest
        @ValueSource(strings = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-opus-4-8", "claude-opus-4-7"])
        fun `a budget becomes adaptive thinking on models that reject budgets`(model: String) {
            send(model, LlmOptions().withThinking(Thinking.withTokenBudget(2000)))

            assertEquals("adaptive", request["thinking"]["type"].asText())
            assertFalse(request["thinking"].has("budget_tokens"))
        }

        @ParameterizedTest
        @ValueSource(strings = ["claude-sonnet-4-5", "claude-haiku-4-5", "claude-opus-4-6", "claude-sonnet-4-6", "claude-mythos-preview"])
        fun `a budget is sent as is to models that accept budgets`(model: String) {
            send(model, LlmOptions().withThinking(Thinking.withTokenBudget(2000)))

            assertEquals("enabled", request["thinking"]["type"].asText())
            assertEquals(2000, request["thinking"]["budget_tokens"].asInt())
        }

        @ParameterizedTest
        @ValueSource(strings = ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5", "claude-opus-5", "claude-sonnet-5"])
        fun `withoutThinking turns thinking off where the model allows it`(model: String) {
            send(model, LlmOptions().withoutThinking())

            assertEquals("disabled", request["thinking"]["type"].asText())
        }

        @ParameterizedTest
        @ValueSource(strings = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-mythos-preview"])
        fun `withoutThinking leaves thinking unset on models that reject disabled`(model: String) {
            send(model, LlmOptions().withoutThinking())

            assertNull(request["thinking"])
        }

        @Test
        fun `thinking extraction leaves the model's own thinking default alone`() {
            send("claude-opus-5", LlmOptions().withThinking(Thinking.withExtraction()))

            assertNull(request["thinking"])
        }

        @ParameterizedTest
        @ValueSource(strings = ["claude-opus-5-5", "claude-haiku-4-5"])
        fun `thinking is unset when the caller says nothing`(model: String) {
            send(model)

            assertNull(request["thinking"])
        }
    }

    @Nested
    inner class SamplingRequest {

        private val sampling = LlmOptions().withTemperature(0.3).withTopP(0.9).withTopK(40)

        @ParameterizedTest
        @ValueSource(strings = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-opus-4-8", "claude-mythos-preview"])
        fun `sampling parameters are dropped on models that reject them`(model: String) {
            send(model, sampling)

            assertNull(request["temperature"], "temperature")
            assertNull(request["top_p"], "top_p")
            assertNull(request["top_k"], "top_k")
        }

        @Test
        fun `sampling parameters reach models that accept them`() {
            send(AnthropicModels.CLAUDE_SONNET_4_6, sampling)

            assertEquals(0.3, request["temperature"].asDouble())
            assertEquals(0.9, request["top_p"].asDouble())
            assertEquals(40, request["top_k"].asInt())
        }
    }
}
