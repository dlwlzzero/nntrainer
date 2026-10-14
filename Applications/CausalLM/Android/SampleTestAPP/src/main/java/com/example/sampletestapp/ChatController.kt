// SPDX-License-Identifier: Apache-2.0
/**
 * @file  ChatController.kt
 * @brief All engine logic behind the Compose screens: one StateFlow, one
 *        engine at a time, every engine call on [engineDispatcher].
 *        Pure Kotlin so the JVM tests drive it with a fake engine.
 */
package com.example.sampletestapp

import com.example.quickdotai.BackendResult
import com.example.quickdotai.BackendType
import com.example.quickdotai.Capability
import com.example.quickdotai.LoadModelRequest
import com.example.quickdotai.ModelDescriptor
import com.example.quickdotai.QuantizationType
import com.example.quickdotai.QuickDotAI
import com.example.quickdotai.RuntimeKind
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

class ChatController(
    private val scope: CoroutineScope,
    private val engineDispatcher: CoroutineDispatcher,
    private val catalogProvider: suspend () -> List<ModelDescriptor>,
    private val engineFactory: (ModelDescriptor, String) -> QuickDotAI,
    private val nativeLibDir: String?,
    private val examples: Examples,
    private val pathExists: (String) -> Boolean = { File(it).exists() },
    private val now: () -> Long = System::currentTimeMillis,
    initialTheme: ThemeMode = ThemeMode.SYSTEM,
    private val publishIntervalMs: Long = 50,
) {
    private val _state = MutableStateFlow(UiState(theme = initialTheme))
    val state: StateFlow<UiState> = _state.asStateFlow()

    /** Written only on [engineDispatcher]; read from the main thread by cancel(). */
    @Volatile private var engine: QuickDotAI? = null
    @Volatile private var cancelRequested = false
    // Turn ids are LazyColumn keys: unique across threads, never minted inside update {}.
    private val nextTurnId = AtomicLong(1)
    private val runIndex = AtomicInteger(0)

    init {
        scope.launch {
            val catalog = withContext(engineDispatcher) { catalogProvider() } // dlopen off the main thread
            val initial = catalog.firstOrNull()
            update {
                copy(
                    catalog = catalog,
                    selection = initial?.let {
                        selection.copy(family = it.family, runtime = it.runtime, backend = it.backends.first())
                    } ?: selection,
                )
            }
            seedDefaults()
        }
    }

    // ----- selection -------------------------------------------------------

    fun selectFamily(family: String) {
        val s = state.value
        val runtime = s.runtimesFor(family).firstOrNull() ?: s.selection.runtime
        val backend = s.backendsFor(family, runtime).firstOrNull() ?: s.selection.backend
        applySelection(s.selection.copy(family = family, runtime = runtime, backend = backend, sdEnabled = false))
    }

    fun selectRuntime(runtime: RuntimeKind) {
        val s = state.value
        val backend = s.backendsFor(s.selection.family, runtime).firstOrNull() ?: s.selection.backend
        applySelection(s.selection.copy(runtime = runtime, backend = backend, sdEnabled = false))
    }

    fun selectBackend(backend: BackendType) =
        applySelection(state.value.selection.copy(backend = backend, sdEnabled = false))

    fun setSdEnabled(enabled: Boolean) = update { copy(selection = selection.copy(sdEnabled = enabled)) }
    fun setBasePath(path: String) = update { copy(selection = selection.copy(basePath = path)) }

    private fun applySelection(sel: ModelSelection) {
        update { copy(selection = sel) }
        seedDefaults()
    }

    /** Like the View sample: a model pick resets the Raw example and leaves the draft alone. */
    private fun seedDefaults() {
        update { copy(rawJson = examples.rawJsonFor(state.value.descriptor)) }
    }

    // ----- plain UI state --------------------------------------------------

    fun setDraft(text: String) = update { copy(draft = text) }
    fun addDraftImages(images: List<ByteArray>) = update { copy(draftImages = draftImages + images) }
    fun removeDraftImage(index: Int) = update { copy(draftImages = draftImages.filterIndexed { i, _ -> i != index }) }
    fun setRawJson(text: String) = update { copy(rawJson = text) }
    fun resetRawJson() = update { copy(rawJson = examples.rawJsonFor(descriptor)) }
    fun setScreen(screen: Screen) = update { copy(screen = screen) }
    fun setTheme(theme: ThemeMode) = update { copy(theme = theme) }
    fun splashDone() = update { copy(splashShown = true) }
    fun consumeNotice() = update { copy(notice = null) }
    fun notify(text: String) = update { copy(notice = text) }

    // ----- model lifecycle -------------------------------------------------

    fun load(force: Boolean = false) {
        scope.launch { loadInternal(buildRequest(), force) }
    }

    fun unload() {
        scope.launch {
            val e = engine ?: run { setStatus("Nothing to unload."); return@launch }
            if (state.value.streaming) { cancelRequested = true; e.cancel(); e.chatCancel() }
            setStatus("Unloading…")
            withContext(engineDispatcher) {
                runCatching { if (e.chatSessionId != null) e.closeChatSession() }
                runCatching { e.unload() }
                runCatching { e.close() }
                engine = null
            }
            update { copy(load = LoadState.Idle, status = "Unloaded.") }
        }
    }

    /** For the ViewModel's onCleared: hands the engine back so the caller can close it on its executor. */
    fun shutdown(): QuickDotAI? {
        val e = engine
        engine = null
        e?.cancel()
        return e
    }

    internal fun buildRequest(): LoadModelRequest {
        val s = state.value
        val d = s.descriptor
        val base = s.selection.basePath.trim().ifEmpty { DEFAULT_BASE_PATH }
        val id = if (s.selection.sdEnabled) (d?.sdVariantId ?: d?.id) else d?.id
        return LoadModelRequest(
            backend = s.selection.backend,
            modelId = id ?: s.selection.family,
            quantization = QuantizationType.W4A32,
            modelPath = d?.let { Examples.modelPath(it, base) },
            visionBackend = d?.takeIf { Capability.MULTIMODAL in it.capabilities }?.let { s.selection.backend },
            nativeLibDir = nativeLibDir,
            modelBasePath = base,
            useSpeculativeDecoding = s.selection.sdEnabled,
        )
    }

    /** Returns the engine for [req], loading or replacing as needed; null and a status on failure. */
    internal suspend fun loadInternal(req: LoadModelRequest, force: Boolean): QuickDotAI? {
        val d = state.value.descriptor ?: run { setStatus("Load failed: no model selected"); return null }
        val current = engine
        if (current != null && !force && state.value.loadedRequest == req) return current
        val required = Examples.requiredPath(d, req.modelBasePath ?: DEFAULT_BASE_PATH)
        if (!pathExists(required)) {
            // Checked before anything is torn down: a typo must not cost the loaded model.
            update { copy(status = "Load failed: model folder not found: $required", notice = "Model folder not found: $required") }
            return null
        }
        update { copy(load = LoadState.Loading(req.modelKey), status = "Loading ${req.modelKey}…") }
        return withContext(engineDispatcher) {
            current?.let { runCatching { it.close() } }
            engine = null
            val e = engineFactory(d, req.modelBasePath ?: DEFAULT_BASE_PATH)
            when (val r = e.load(req)) {
                is BackendResult.Ok -> {
                    engine = e
                    update {
                        copy(
                            load = LoadState.Loaded(req, e.kind, e.architecture),
                            status = "Loaded ${req.modelKey} (${e.kind}, arch=${e.architecture ?: "?"})",
                            turns = if (current != null) emptyList() else turns,
                        )
                    }
                    e
                }
                is BackendResult.Err -> {
                    runCatching { e.close() }
                    update { copy(load = LoadState.Idle, status = "Load failed: [${r.error.name}] ${r.message ?: ""}") }
                    null
                }
            }
        }
    }

    // ----- helpers ---------------------------------------------------------

    internal fun nextId(): Long = nextTurnId.getAndIncrement()
    internal fun setStatus(text: String) = update { copy(status = text) }
    private fun update(block: UiState.() -> UiState) = _state.update(block)
}
