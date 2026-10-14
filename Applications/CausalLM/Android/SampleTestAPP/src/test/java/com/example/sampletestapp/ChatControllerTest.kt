// SPDX-License-Identifier: Apache-2.0
package com.example.sampletestapp

import com.example.quickdotai.BackendResult
import com.example.quickdotai.BackendType
import com.example.quickdotai.Capability
import com.example.quickdotai.ModelDescriptor
import com.example.quickdotai.ModelIds
import com.example.quickdotai.QuickAiError
import com.example.quickdotai.RuntimeKind
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asCoroutineDispatcher
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import java.util.concurrent.Executors

class ChatControllerTest {
    val npu = ModelDescriptor("fake-npu", "fake-npu", "Fake NPU",
        RuntimeKind.NATIVE, setOf(BackendType.NPU), setOf(Capability.STREAMING, Capability.MESSAGES_API))
    val qwen = ModelDescriptor(ModelIds.QWEN3_0_6B, "qwen3-0.6b", "Qwen3 0.6B",
        RuntimeKind.NATIVE, setOf(BackendType.CPU, BackendType.GPU), setOf(Capability.STREAMING, Capability.TOOL_USE))
    val litert = ModelDescriptor(ModelIds.GEMMA4, "gemma4", "Gemma4 (LiteRT)",
        RuntimeKind.LITERT, setOf(BackendType.GPU), setOf(Capability.MULTIMODAL, Capability.MESSAGES_API, Capability.STREAMING))
    val catalog = listOf(qwen, npu, litert)

    private val executor = Executors.newSingleThreadExecutor()
    val engines: MutableList<FakeQuickDotAI> = java.util.Collections.synchronizedList(mutableListOf())
    var nextEngine: () -> FakeQuickDotAI = { FakeQuickDotAI() }
    val existing = mutableSetOf("/m/fake-npu", "/m/qwen3-0.6b", "/m/gemma-4-E2B-it/gemma-4-E2B-it.litertlm")
    @Volatile var clockMs = 0L
    var publishInterval = 0L

    fun controller(): ChatController = ChatController(
        scope = CoroutineScope(Dispatchers.Unconfined),
        engineDispatcher = executor.asCoroutineDispatcher(),
        catalogProvider = { catalog },
        engineFactory = { _, _ -> nextEngine().also { engines += it } },
        nativeLibDir = "/data/app/lib",
        examples = Examples(),
        pathExists = { it in existing },
        now = { clockMs },
        publishIntervalMs = publishInterval,
    ).also { c -> await("catalog") { c.state.value.catalog.isNotEmpty() } }

    fun await(label: String, timeoutMs: Long = 5000, cond: () -> Boolean) {
        val end = System.currentTimeMillis() + timeoutMs
        while (!cond()) { if (System.currentTimeMillis() > end) fail("timeout: $label"); Thread.sleep(5) }
    }
    fun ChatController.awaitIdle() {
        await("idle") { !state.value.busy }
        executor.submit {}.get()                                   // queued engine work finished too
    }
    fun ChatController.selectNpu() { selectFamily("fake-npu"); setBasePath("/m") }

    @After fun tearDown() { executor.shutdownNow() }

    @Test
    fun catalogSelectsTheFirstDescriptorAndSeedsTheRawExample() {
        val c = controller()
        val s = c.state.value
        assertEquals(qwen, s.descriptor)
        assertEquals("", s.draft)
        assertTrue(s.rawJson.trimStart().startsWith("{"))             // qwen: full request object
        c.setDraft("keep me")
        c.selectFamily("fake-npu")
        assertEquals(npu, c.state.value.descriptor)
        assertEquals("keep me", c.state.value.draft)                   // a model pick does not wipe the draft
        assertTrue(c.state.value.rawJson.trimStart().startsWith("[")) // MESSAGES_API: messages array
        c.selectFamily("qwen3-0.6b")
        c.selectBackend(BackendType.GPU)
        assertEquals(BackendType.GPU, c.state.value.selection.backend)
    }

    @Test
    fun loadReportsAndKeepsEngineOnMissingPath() {
        val c = controller(); c.selectNpu()
        c.load(); c.awaitIdle()
        assertTrue(c.state.value.status, c.state.value.status.startsWith("Loaded fake-npu"))
        assertEquals(listOf("load:fake-npu"), engines.single().calls)
        c.setBasePath("/m/missing-model")
        c.load(); c.awaitIdle()
        assertEquals("Load failed: model folder not found: /m/missing-model/fake-npu", c.state.value.status)
        assertTrue(c.state.value.load is LoadState.Loaded)        // the old engine stayed
        assertFalse(engines.single().closed)
        assertEquals(1, engines.size)
    }

    @Test
    fun loadFailureReportsNativeDiagnostic() {
        nextEngine = { FakeQuickDotAI(loadResult = BackendResult.Err(QuickAiError.MODEL_LOAD_FAILED, "Failed to open file x")) }
        val c = controller(); c.selectNpu()
        c.load(); c.awaitIdle()
        assertEquals("Load failed: [MODEL_LOAD_FAILED] Failed to open file x", c.state.value.status)
        assertEquals(LoadState.Idle, c.state.value.load)
        assertTrue(engines.single().closed)
    }

    @Test
    fun reloadReplacesEngineAndUnloadReleasesIt() {
        val c = controller(); c.selectNpu()
        c.load(); c.awaitIdle()
        c.load(); c.awaitIdle()                                   // same request: no new engine
        assertEquals(1, engines.size)
        c.load(force = true); c.awaitIdle()
        assertEquals(2, engines.size)
        assertTrue(engines[0].closed)
        assertTrue(c.state.value.status.startsWith("Loaded "))
        c.unload(); c.awaitIdle()
        assertEquals("Unloaded.", c.state.value.status)
        assertEquals(LoadState.Idle, c.state.value.load)
        assertTrue(engines[1].calls.containsAll(listOf("unload", "close")))
        c.unload(); c.awaitIdle()
        assertEquals("Nothing to unload.", c.state.value.status)
    }

    @Test
    fun selectionChangeAfterLoadMakesNextLoadReplace() {
        val c = controller(); c.selectNpu()
        c.load(); c.awaitIdle()
        c.selectFamily("qwen3-0.6b")
        c.load(); c.awaitIdle()
        assertEquals(2, engines.size)
        assertEquals("load:qwen3-0.6b", engines[1].calls.first())
        assertTrue(engines[0].closed)
    }
}
