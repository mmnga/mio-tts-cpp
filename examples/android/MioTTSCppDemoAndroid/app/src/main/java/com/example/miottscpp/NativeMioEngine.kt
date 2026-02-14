package com.example.miottscpp

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

data class MioReference(
    val key: String,
    val embeddingDim: Int,
)

class NativeMioEngine(context: Context) {
    private var handle: Long = 0L
    private val nativeLibDir: String = context.applicationInfo.nativeLibraryDir

    companion object {
        init {
            System.loadLibrary("mio-tts-android")
        }
    }

    fun initBackends() {
        nativeInitBackends(nativeLibDir)
    }

    fun create(
        llmModelPath: String?,
        vocoderModelPath: String,
        wavlmModelPath: String?,
        nGpuLayers: Int,
        nCtx: Int,
        nThreads: Int,
        flashAttn: Boolean,
    ) {
        close()

        val llm = llmModelPath?.trim()?.takeIf { it.isNotBlank() }
        val wavlm = wavlmModelPath?.trim()?.takeIf { it.isNotBlank() }
        val vocoder = vocoderModelPath.trim()

        require(vocoder.isNotBlank()) { "vocoder model path is required" }

        handle = nativeCreateEngine(llm, vocoder, wavlm, nGpuLayers, nCtx, nThreads, flashAttn)
        if (handle == 0L) {
            throw IllegalStateException(nativeGetGlobalError().ifBlank { "failed to create engine" })
        }
    }

    fun close() {
        if (handle != 0L) {
            nativeDestroyEngine(handle)
            handle = 0L
        }
    }

    fun setGenerationParams(
        nCtx: Int,
        topK: Int,
        topP: Float,
        temp: Float,
    ) {
        val h = requireHandle()
        nativeSetGenerationParams(h, nCtx, topK, topP, temp)?.let { throw IllegalStateException(it) }
    }

    fun unloadLLMRuntime() {
        val h = requireHandle()
        nativeUnloadLlmRuntime(h)?.let { throw IllegalStateException(it) }
    }

    fun addReferenceFromGguf(referenceKey: String, embeddingPath: String) {
        val h = requireHandle()
        nativeAddReferenceFromGguf(h, referenceKey, embeddingPath)?.let { throw IllegalStateException(it) }
    }

    fun removeReference(referenceKey: String) {
        val h = requireHandle()
        nativeRemoveReference(h, referenceKey)?.let { throw IllegalStateException(it) }
    }

    fun registerDefaultReferences(
        modelDirPath: String?,
        fallbackEmbeddingPath: String?,
    ): String? {
        val h = requireHandle()
        val modelDir = modelDirPath?.trim()?.takeIf { it.isNotBlank() }
        val fallbackPath = fallbackEmbeddingPath?.trim()?.takeIf { it.isNotBlank() }
        val preferred = nativeRegisterDefaultReferences(h, modelDir, fallbackPath)
        if (preferred == null) {
            val err = nativeGetLastError(h)
            if (err.isNotBlank()) {
                throw IllegalStateException(err)
            }
        }
        return preferred?.takeIf { it.isNotBlank() }
    }

    fun createReferenceFromAudio(
        referenceKey: String,
        audioPath: String,
        maxReferenceSeconds: Float,
        saveEmbeddingPath: String,
    ) {
        val h = requireHandle()
        nativeCreateReferenceFromAudio(h, referenceKey, audioPath, maxReferenceSeconds, saveEmbeddingPath)
            ?.let { throw IllegalStateException(it) }
    }

    fun listReferences(): List<MioReference> {
        val h = requireHandle()
        val json = nativeListReferencesJson(h)
            ?: throw IllegalStateException(nativeGetLastError(h).ifBlank { "failed to list references" })

        val arr = JSONArray(json)
        val refs = ArrayList<MioReference>(arr.length())
        for (i in 0 until arr.length()) {
            val obj: JSONObject = arr.getJSONObject(i)
            refs += MioReference(
                key = obj.getString("key"),
                embeddingDim = obj.optInt("embedding_dim", 0),
            )
        }
        return refs
    }

    fun synthesizeToWav(
        text: String,
        referenceKey: String,
        nPredict: Int,
        outputWavPath: String,
    ) {
        val h = requireHandle()
        nativeSynthesizeToWav(h, text, referenceKey, nPredict, outputWavPath)
            ?.let { throw IllegalStateException(it) }
    }

    fun synthesizeCodesToWav(
        codes: IntArray,
        referenceKey: String,
        outputWavPath: String,
    ) {
        val h = requireHandle()
        nativeSynthesizeCodesToWav(h, codes, referenceKey, outputWavPath)
            ?.let { throw IllegalStateException(it) }
    }

    private fun requireHandle(): Long {
        check(handle != 0L) { "engine is not initialized" }
        return handle
    }

    private external fun nativeInitBackends(nativeLibDir: String)

    private external fun nativeCreateEngine(
        llmModelPath: String?,
        vocoderModelPath: String,
        wavlmModelPath: String?,
        nGpuLayers: Int,
        nCtx: Int,
        nThreads: Int,
        flashAttn: Boolean,
    ): Long

    private external fun nativeDestroyEngine(handle: Long)

    private external fun nativeSetGenerationParams(
        handle: Long,
        nCtx: Int,
        topK: Int,
        topP: Float,
        temp: Float,
    ): String?

    private external fun nativeUnloadLlmRuntime(handle: Long): String?

    private external fun nativeAddReferenceFromGguf(
        handle: Long,
        referenceKey: String,
        embeddingPath: String,
    ): String?

    private external fun nativeRemoveReference(
        handle: Long,
        referenceKey: String,
    ): String?

    private external fun nativeRegisterDefaultReferences(
        handle: Long,
        modelDirPath: String?,
        fallbackEmbeddingPath: String?,
    ): String?

    private external fun nativeCreateReferenceFromAudio(
        handle: Long,
        referenceKey: String,
        audioPath: String,
        maxReferenceSeconds: Float,
        saveEmbeddingPath: String,
    ): String?

    private external fun nativeListReferencesJson(handle: Long): String?

    private external fun nativeSynthesizeToWav(
        handle: Long,
        text: String,
        referenceKey: String,
        nPredict: Int,
        outputWavPath: String,
    ): String?

    private external fun nativeSynthesizeCodesToWav(
        handle: Long,
        codes: IntArray,
        referenceKey: String,
        outputWavPath: String,
    ): String?

    private external fun nativeGetLastError(handle: Long): String

    private external fun nativeGetGlobalError(): String
}
