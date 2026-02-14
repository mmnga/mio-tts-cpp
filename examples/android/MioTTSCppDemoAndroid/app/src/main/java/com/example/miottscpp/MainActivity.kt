package com.example.miottscpp

import android.Manifest
import android.content.pm.PackageManager
import android.media.MediaPlayer
import android.net.Uri
import android.os.Bundle
import android.os.SystemClock
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.documentfile.provider.DocumentFile
import androidx.lifecycle.lifecycleScope
import com.google.android.material.switchmaterial.SwitchMaterial
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.nio.charset.StandardCharsets
import java.util.Locale

class MainActivity : AppCompatActivity() {
    private lateinit var engine: NativeMioEngine

    private lateinit var edtLlmModelPath: EditText
    private lateinit var edtVocoderModelPath: EditText
    private lateinit var edtWavlmModelPath: EditText
    private lateinit var edtSpeakerEmbPath: EditText
    private lateinit var btnLoadModels: Button

    private lateinit var switchExternalApi: SwitchMaterial
    private lateinit var edtApiBaseUrl: EditText

    private lateinit var spinnerReferences: Spinner
    private lateinit var btnRemoveReference: Button
    private lateinit var btnToggleRecording: Button
    private lateinit var btnPickAudio: Button
    private lateinit var btnPickEmbedding: Button

    private lateinit var edtPrompt: EditText
    private lateinit var edtNPredict: EditText
    private lateinit var edtCtxSize: EditText
    private lateinit var edtTopK: EditText
    private lateinit var edtTopP: EditText
    private lateinit var edtTemp: EditText
    private lateinit var btnGenerate: Button
    private lateinit var btnPlayAudio: Button

    private lateinit var txtGenerateTime: TextView
    private lateinit var txtStatus: TextView

    private lateinit var loadingOverlay: LinearLayout
    private lateinit var txtLoading: TextView

    private val references = mutableListOf<MioReference>()
    private lateinit var referencesAdapter: ArrayAdapter<String>

    private var isEngineLoading = false
    private var isBusy = false
    private var isGenerating = false
    private var isRecording = false
    private var engineLoaded = false

    private var wavRecorder: WavRecorder? = null
    private var mediaPlayer: MediaPlayer? = null
    private var latestOutputPath: String? = null
    private val defaultSpeakerKeys = listOf("en_female", "en_male", "jp_female", "jp_male")

    private val requestRecordPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                startRecordingInternal()
            } else {
                setStatus("録音権限が必要です。")
            }
        }

    private val pickAudioLauncher =
        registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
            if (uri == null) {
                return@registerForActivityResult
            }
            lifecycleScope.launch {
                runCatching {
                    val copied = copyUriToInternalFile(
                        uri = uri,
                        destinationDirName = "reference-audio",
                        prefix = "reference-import",
                        fallbackExt = "wav",
                    )
                    createReferenceFromAudioFile(copied)
                }.onFailure {
                    setStatus("音声ファイルの追加に失敗: ${it.message}")
                }
            }
        }

    private val pickEmbeddingLauncher =
        registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
            if (uri == null) {
                return@registerForActivityResult
            }
            lifecycleScope.launch {
                runCatching {
                    val copied = copyUriToInternalFile(
                        uri = uri,
                        destinationDirName = "reference-embeddings",
                        prefix = "embedding-import",
                        fallbackExt = "gguf",
                    )
                    val key = nextGeneratedReferenceKey()
                    withContext(Dispatchers.IO) {
                        engine.addReferenceFromGguf(key, copied.absolutePath)
                    }
                    reloadReferences(selectKey = key)
                    setStatus("$key を追加しました。")
                }.onFailure {
                    setStatus("emb.gguf の追加に失敗: ${it.message}")
                }
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        bindViews()

        referencesAdapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, mutableListOf<String>())
        referencesAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        spinnerReferences.adapter = referencesAdapter
        spinnerReferences.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                refreshUiState()
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {
                refreshUiState()
            }
        }

        engine = NativeMioEngine(applicationContext)
        engine.initBackends()

        btnLoadModels.setOnClickListener {
            lifecycleScope.launch {
                loadModels()
            }
        }

        btnToggleRecording.setOnClickListener {
            if (isRecording) {
                stopRecordingAndCreateReference()
            } else {
                startRecording()
            }
        }

        btnPickAudio.setOnClickListener {
            if (!canInteract()) {
                return@setOnClickListener
            }
            pickAudioLauncher.launch(arrayOf("audio/*"))
        }

        btnPickEmbedding.setOnClickListener {
            if (!canInteract()) {
                return@setOnClickListener
            }
            pickEmbeddingLauncher.launch(arrayOf("*/*"))
        }

        btnRemoveReference.setOnClickListener {
            lifecycleScope.launch {
                removeSelectedReference()
            }
        }

        btnGenerate.setOnClickListener {
            lifecycleScope.launch {
                generateSpeech()
            }
        }

        btnPlayAudio.setOnClickListener {
            playLatestAudio()
        }

        refreshUiState()
        setStatus("同梱モデルを確認中...")

        lifecycleScope.launch {
            initializeBundledModels()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        runCatching {
            wavRecorder?.stop()
        }
        wavRecorder = null
        isRecording = false

        mediaPlayer?.release()
        mediaPlayer = null

        runCatching {
            engine.close()
        }
    }

    private fun bindViews() {
        edtLlmModelPath = findViewById(R.id.edtLlmModelPath)
        edtVocoderModelPath = findViewById(R.id.edtVocoderModelPath)
        edtWavlmModelPath = findViewById(R.id.edtWavlmModelPath)
        edtSpeakerEmbPath = findViewById(R.id.edtSpeakerEmbPath)
        btnLoadModels = findViewById(R.id.btnLoadModels)

        switchExternalApi = findViewById(R.id.switchExternalApi)
        edtApiBaseUrl = findViewById(R.id.edtApiBaseUrl)

        spinnerReferences = findViewById(R.id.spinnerReferences)
        btnRemoveReference = findViewById(R.id.btnRemoveReference)
        btnToggleRecording = findViewById(R.id.btnToggleRecording)
        btnPickAudio = findViewById(R.id.btnPickAudio)
        btnPickEmbedding = findViewById(R.id.btnPickEmbedding)

        edtPrompt = findViewById(R.id.edtPrompt)
        edtNPredict = findViewById(R.id.edtNPredict)
        edtCtxSize = findViewById(R.id.edtCtxSize)
        edtTopK = findViewById(R.id.edtTopK)
        edtTopP = findViewById(R.id.edtTopP)
        edtTemp = findViewById(R.id.edtTemp)
        btnGenerate = findViewById(R.id.btnGenerate)
        btnPlayAudio = findViewById(R.id.btnPlayAudio)

        txtGenerateTime = findViewById(R.id.txtGenerateTime)
        txtStatus = findViewById(R.id.txtStatus)

        loadingOverlay = findViewById(R.id.loadingOverlay)
        txtLoading = findViewById(R.id.txtLoading)
    }

    private suspend fun loadModels() {
        if (!canInteract()) {
            return
        }

        val localLlmRequired = !switchExternalApi.isChecked
        val llmPath = edtLlmModelPath.text.toString().trim()
        val vocoderPath = edtVocoderModelPath.text.toString().trim()
        val wavlmPath = edtWavlmModelPath.text.toString().trim()
        val speakerPath = edtSpeakerEmbPath.text.toString().trim()

        if (vocoderPath.isBlank()) {
            setStatus("MioCodec model path は必須です。")
            showToast("MioCodec model path は必須です。")
            return
        }
        if (!File(vocoderPath).exists()) {
            setStatus("MioCodec model が見つかりません: $vocoderPath")
            showToast("MioCodec model が見つかりません。")
            return
        }
        if (localLlmRequired && llmPath.isBlank()) {
            setStatus("ローカル推論では LLM model path が必須です。")
            showToast("LLM model path が必要です。")
            return
        }
        if (localLlmRequired && !File(llmPath).exists()) {
            setStatus("LLM model が見つかりません: $llmPath")
            showToast("LLM model が見つかりません。")
            return
        }
        if (wavlmPath.isBlank()) {
            setStatus("WavLM model path は必須です。")
            showToast("WavLM model path は必須です。")
            return
        }
        if (!File(wavlmPath).exists()) {
            setStatus("WavLM model が見つかりません: $wavlmPath")
            showToast("WavLM model が見つかりません。")
            return
        }

        isEngineLoading = true
        txtLoading.text = "モデルロード中です..."
        refreshUiState()
        txtGenerateTime.text = ""

        runCatching {
            val nCtx = parseInt(edtCtxSize, 700)
            val topK = parseInt(edtTopK, 50)
            val topP = parseFloat(edtTopP, 1.0f)
            val temp = parseFloat(edtTemp, 0.8f)
            var preferredReferenceKey: String? = null

            withContext(Dispatchers.IO) {
                engine.create(
                    llmModelPath = llmPath.takeIf { it.isNotBlank() },
                    vocoderModelPath = vocoderPath,
                    wavlmModelPath = wavlmPath.takeIf { it.isNotBlank() },
                    nGpuLayers = -1,
                    nCtx = nCtx,
                    nThreads = 2,
                    flashAttn = false,
                )
                engine.setGenerationParams(
                    nCtx = nCtx,
                    topK = topK,
                    topP = topP,
                    temp = temp,
                )
                preferredReferenceKey = engine.registerDefaultReferences(
                    modelDirPath = File(vocoderPath).parentFile?.absolutePath,
                    fallbackEmbeddingPath = speakerPath.takeIf { it.isNotBlank() },
                )
            }

            engineLoaded = true
            reloadReferences(selectKey = preferredReferenceKey)
            setStatus("モデルをロードしました。")
            showToast("モデルをロードしました。")
        }.onFailure {
            engineLoaded = false
            setStatus("モデルロード失敗: ${it.message}")
            showToast("モデルロード失敗: ${it.message}")
        }

        isEngineLoading = false
        refreshUiState()
    }

    private suspend fun reloadReferences(selectKey: String? = null) {
        val refs = withContext(Dispatchers.IO) { engine.listReferences() }
            .sortedBy { it.key }

        references.clear()
        references.addAll(refs)

        val names = refs.map { "${it.key} (dim=${it.embeddingDim})" }
        referencesAdapter.clear()
        referencesAdapter.addAll(names)
        referencesAdapter.notifyDataSetChanged()

        if (refs.isNotEmpty()) {
            val target = selectKey ?: refs.first().key
            val idx = refs.indexOfFirst { it.key == target }.takeIf { it >= 0 } ?: 0
            spinnerReferences.setSelection(idx)
        }
    }

    private fun selectedReferenceKey(): String {
        val idx = spinnerReferences.selectedItemPosition
        if (idx < 0 || idx >= references.size) {
            return ""
        }
        return references[idx].key
    }

    private fun isPresetReference(referenceKey: String): Boolean {
        return defaultSpeakerKeys.contains(referenceKey)
    }

    private fun nextGeneratedReferenceKey(): String {
        var maxId = 0
        val regex = Regex("^added_speaker_(\\d+)$")
        references.forEach { ref ->
            val m = regex.find(ref.key) ?: return@forEach
            val id = m.groupValues[1].toIntOrNull() ?: return@forEach
            if (id > maxId) {
                maxId = id
            }
        }
        return "added_speaker_${maxId + 1}"
    }

    private fun startRecording() {
        if (!canInteract()) {
            return
        }

        val granted = ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED
        if (!granted) {
            requestRecordPermission.launch(Manifest.permission.RECORD_AUDIO)
            return
        }

        startRecordingInternal()
    }

    private fun startRecordingInternal() {
        if (!canInteract() || isRecording) {
            return
        }

        runCatching {
            engine.unloadLLMRuntime()
            val file = newReferenceAudioFile("reference-recorded", "wav")
            val recorder = WavRecorder(file)
            recorder.start()
            wavRecorder = recorder
            isRecording = true
            setStatus("録音中... もう一度押すと停止して話者を追加します。")
        }.onFailure {
            setStatus("録音開始に失敗: ${it.message}")
        }
        refreshUiState()
    }

    private fun stopRecordingAndCreateReference() {
        if (!isRecording) {
            return
        }

        val recorder = wavRecorder
        wavRecorder = null
        isRecording = false
        refreshUiState()

        lifecycleScope.launch {
            runCatching {
                val audioFile = recorder?.outputFile
                withContext(Dispatchers.IO) {
                    recorder?.stop()
                }
                if (audioFile == null || !audioFile.exists()) {
                    error("録音ファイルが見つかりません")
                }
                createReferenceFromAudioFile(audioFile)
            }.onFailure {
                setStatus("録音から話者追加に失敗: ${it.message}")
            }
        }
    }

    private suspend fun createReferenceFromAudioFile(audioFile: File) {
        if (!engineLoaded) {
            setStatus("先にモデルをロードしてください。")
            return
        }

        val key = nextGeneratedReferenceKey()
        val embeddingOut = referenceEmbeddingFile(key)

        isBusy = true
        refreshUiState()

        runCatching {
            withContext(Dispatchers.IO) {
                engine.unloadLLMRuntime()
                engine.createReferenceFromAudio(
                    referenceKey = key,
                    audioPath = audioFile.absolutePath,
                    maxReferenceSeconds = 20.0f,
                    saveEmbeddingPath = embeddingOut.absolutePath,
                )
            }
            reloadReferences(selectKey = key)
            setStatus("$key を作成して追加しました。")
        }.onFailure {
            setStatus("話者追加に失敗: ${it.message}")
        }

        isBusy = false
        refreshUiState()
    }

    private suspend fun removeSelectedReference() {
        if (!engineLoaded) {
            setStatus("先にモデルをロードしてください。")
            return
        }
        if (!canInteract()) {
            return
        }

        val referenceKey = selectedReferenceKey()
        if (referenceKey.isBlank()) {
            setStatus("削除する話者を選択してください。")
            return
        }
        if (isPresetReference(referenceKey)) {
            setStatus("既定の話者は削除できません。")
            return
        }

        isBusy = true
        refreshUiState()

        runCatching {
            withContext(Dispatchers.IO) {
                engine.removeReference(referenceKey)
            }
            val removedFile = removeSavedEmbeddingForReference(referenceKey)
            reloadReferences()
            if (removedFile) {
                setStatus("$referenceKey を削除しました（保存ファイルも削除）。")
            } else {
                setStatus("$referenceKey を削除しました。")
            }
        }.onFailure {
            setStatus("話者削除に失敗: ${it.message}")
        }

        isBusy = false
        refreshUiState()
    }

    private suspend fun generateSpeech() {
        if (!engineLoaded) {
            setStatus("先にモデルをロードしてください。")
            return
        }
        if (!canInteract()) {
            return
        }

        val prompt = edtPrompt.text.toString().trim()
        if (prompt.isBlank()) {
            setStatus("Prompt を入力してください。")
            return
        }

        val referenceKey = selectedReferenceKey()
        if (referenceKey.isBlank()) {
            setStatus("話者を選択してください。")
            return
        }

        isGenerating = true
        refreshUiState()
        setStatus("生成中...")

        val t0 = SystemClock.elapsedRealtime()

        runCatching {
            val nPredict = parseInt(edtNPredict, 200).coerceIn(1, 300)
            val nCtx = parseInt(edtCtxSize, 700)
            val topK = parseInt(edtTopK, 50)
            val topP = parseFloat(edtTopP, 1.0f)
            val temp = parseFloat(edtTemp, 0.8f)

            val outPath = outputWavFile().absolutePath

            withContext(Dispatchers.IO) {
                engine.setGenerationParams(nCtx = nCtx, topK = topK, topP = topP, temp = temp)

                if (switchExternalApi.isChecked) {
                    val baseUrl = edtApiBaseUrl.text.toString().trim().ifBlank { "http://10.0.2.2:8080/v1" }
                    val codes = fetchCodesViaExternalApi(
                        baseUrl = baseUrl,
                        apiKey = "dummy",
                        prompt = prompt,
                        nPredict = nPredict,
                        topK = topK,
                        topP = topP,
                        temp = temp,
                    )
                    engine.synthesizeCodesToWav(codes, referenceKey, outPath)
                } else {
                    engine.synthesizeToWav(
                        text = prompt,
                        referenceKey = referenceKey,
                        nPredict = nPredict,
                        outputWavPath = outPath,
                    )
                }
            }

            latestOutputPath = outPath
            btnPlayAudio.isEnabled = true
            val autoPlayed = playLatestAudio(showStatusMessage = false)

            val elapsedMs = SystemClock.elapsedRealtime() - t0
            txtGenerateTime.text = String.format(Locale.US, "生成時間: %.2f 秒", elapsedMs / 1000.0)
            if (autoPlayed) {
                setStatus("生成完了・自動再生中: $outPath")
            } else {
                setStatus("生成完了 (自動再生失敗): $outPath")
            }
        }.onFailure {
            val elapsedMs = SystemClock.elapsedRealtime() - t0
            txtGenerateTime.text = String.format(Locale.US, "失敗まで: %.2f 秒", elapsedMs / 1000.0)
            setStatus("生成失敗: ${it.message}")
        }

        isGenerating = false
        refreshUiState()
    }

    private fun playLatestAudio(showStatusMessage: Boolean = true): Boolean {
        val path = latestOutputPath ?: run {
            if (showStatusMessage) {
                setStatus("再生する音声がありません。")
            }
            return false
        }

        val file = File(path)
        if (!file.exists()) {
            if (showStatusMessage) {
                setStatus("音声ファイルが見つかりません: $path")
            }
            return false
        }

        return runCatching {
            mediaPlayer?.release()
            mediaPlayer = MediaPlayer().apply {
                setDataSource(path)
                prepare()
                start()
            }
            if (showStatusMessage) {
                setStatus("再生中...")
            }
            true
        }.onFailure {
            if (showStatusMessage) {
                setStatus("再生失敗: ${it.message}")
            }
        }.getOrElse { false }
    }

    private suspend fun fetchCodesViaExternalApi(
        baseUrl: String,
        apiKey: String,
        prompt: String,
        nPredict: Int,
        topK: Int,
        topP: Float,
        temp: Float,
    ): IntArray = withContext(Dispatchers.IO) {
        val root = baseUrl.trim().trimEnd('/')
        val url = URL("$root/chat/completions")

        val requestJson = JSONObject().apply {
            put("messages", JSONArray().put(JSONObject().apply {
                put("role", "user")
                put("content", prompt)
            }))
            put("max_tokens", nPredict)
            put("n_predict", nPredict)
            put("top_k", topK)
            put("top_p", topP)
            put("temperature", temp)
            put("stream", false)
        }.toString()

        val conn = (url.openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = 60_000
            readTimeout = 60_000
            doOutput = true
            setRequestProperty("Content-Type", "application/json")
            setRequestProperty("Authorization", "Bearer $apiKey")
        }

        try {
            conn.outputStream.use { os ->
                os.write(requestJson.toByteArray(StandardCharsets.UTF_8))
            }

            val status = conn.responseCode
            val stream = if (status in 200..299) conn.inputStream else conn.errorStream
            val body = stream?.bufferedReader()?.use { it.readText() }.orEmpty()

            if (status !in 200..299) {
                error("external API HTTP $status: ${body.take(300)}")
            }

            return@withContext parseCodesFromApiBody(body)
        } finally {
            conn.disconnect()
        }
    }

    private fun parseCodesFromApiBody(body: String): IntArray {
        runCatching {
            val obj = JSONObject(body)
            val codesValues = obj.optJSONArray("codes_values")
            if (codesValues != null) {
                return parseCodesFromJsonArray(codesValues)
            }
            val codes = obj.optJSONArray("codes")
            if (codes != null) {
                return parseCodesFromJsonArray(codes)
            }
            val audioCodes = obj.optJSONArray("audio_codes")
            if (audioCodes != null) {
                return parseCodesFromJsonArray(audioCodes)
            }

            val choices = obj.optJSONArray("choices")
            if (choices != null && choices.length() > 0) {
                val c0 = choices.optJSONObject(0)
                val messageContent = c0?.optJSONObject("message")?.optString("content")
                val text = c0?.optString("text")
                val src = when {
                    !messageContent.isNullOrBlank() -> messageContent
                    !text.isNullOrBlank() -> text
                    else -> null
                }
                if (!src.isNullOrBlank()) {
                    return parseCodesFromText(src)
                }
            }

            val text = obj.optString("text")
            if (text.isNotBlank()) {
                return parseCodesFromText(text)
            }
        }

        return parseCodesFromText(body)
    }

    private fun parseCodesFromJsonArray(arr: JSONArray): IntArray {
        val out = IntArray(arr.length())
        for (i in 0 until arr.length()) {
            val v = arr.get(i)
            out[i] = when (v) {
                is Number -> v.toInt()
                is String -> parseCodeToken(v)
                else -> error("invalid code at index $i")
            }
        }
        if (out.isEmpty()) {
            error("codes are empty")
        }
        return out
    }

    private fun parseCodeToken(token: String): Int {
        CODE_TOKEN_REGEX.find(token.trim())?.let {
            return it.groupValues[1].toInt()
        }
        return token.trim().toInt()
    }

    private fun parseCodesFromText(text: String): IntArray {
        val list = CODE_TOKEN_REGEX.findAll(text).map { it.groupValues[1].toInt() }.toList()
        if (list.isEmpty()) {
            error("external API response does not include Mio codes")
        }
        return list.toIntArray()
    }

    private fun parseInt(editText: EditText, fallback: Int): Int {
        return editText.text.toString().trim().toIntOrNull() ?: fallback
    }

    private fun parseFloat(editText: EditText, fallback: Float): Float {
        return editText.text.toString().trim().toFloatOrNull() ?: fallback
    }

    private fun canInteract(): Boolean {
        return !isEngineLoading && !isBusy && !isGenerating
    }

    private fun refreshUiState() {
        val loading = isEngineLoading || isBusy || isGenerating

        loadingOverlay.visibility = if (loading) LinearLayout.VISIBLE else LinearLayout.GONE
        if (isGenerating) {
            txtLoading.text = "生成中..."
        } else {
            txtLoading.text = "モデルロード中です..."
        }

        val canTap = !loading
        val selectedKey = selectedReferenceKey()
        val canRemoveSelected = selectedKey.isNotBlank() && !isPresetReference(selectedKey)
        btnLoadModels.isEnabled = canTap && !isRecording
        btnRemoveReference.isEnabled = canTap && !isRecording && canRemoveSelected
        btnPickAudio.isEnabled = canTap && !isRecording
        btnPickEmbedding.isEnabled = canTap && !isRecording
        btnGenerate.isEnabled = canTap && !isRecording && references.isNotEmpty()
        btnPlayAudio.isEnabled = canTap && latestOutputPath != null
        btnToggleRecording.isEnabled = canTap || isRecording
        btnToggleRecording.text = if (isRecording) "録音停止して話者追加" else "録音して話者追加"
    }

    private fun setStatus(message: String) {
        txtStatus.text = message
    }

    private fun showToast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }

    private fun newReferenceAudioFile(prefix: String, ext: String): File {
        val dir = File(filesDir, "reference-audio")
        dir.mkdirs()
        return File(dir, "$prefix-${System.currentTimeMillis()}.$ext")
    }

    private fun referenceEmbeddingFile(referenceKey: String): File {
        val dir = File(filesDir, "reference-embeddings")
        dir.mkdirs()
        return File(dir, "$referenceKey.emb.gguf")
    }

    private fun removeSavedEmbeddingForReference(referenceKey: String): Boolean {
        val file = referenceEmbeddingFile(referenceKey)
        if (!file.exists()) {
            return false
        }
        return runCatching { file.delete() }.getOrDefault(false)
    }

    private fun outputWavFile(): File {
        val dir = File(filesDir, "generated")
        dir.mkdirs()
        return File(dir, "mio-tts-${System.currentTimeMillis()}.wav")
    }

    private fun copyUriToInternalFile(
        uri: Uri,
        destinationDirName: String,
        prefix: String,
        fallbackExt: String,
    ): File {
        val sourceName = DocumentFile.fromSingleUri(this, uri)?.name.orEmpty()
        val ext = sourceName.substringAfterLast('.', fallbackExt).ifBlank { fallbackExt }
        val dstDir = File(filesDir, destinationDirName).apply { mkdirs() }
        val dst = File(dstDir, "$prefix-${System.currentTimeMillis()}.$ext")

        contentResolver.openInputStream(uri).use { input ->
            requireNotNull(input) { "failed to open input stream" }
            FileOutputStream(dst).use { out ->
                input.copyTo(out)
            }
        }

        return dst
    }

    private suspend fun initializeBundledModels() {
        isBusy = true
        txtLoading.text = "同梱モデルを準備中..."
        refreshUiState()

        runCatching {
            val localModelsDir = File(filesDir, "models").apply { mkdirs() }
            val bundledNames = withContext(Dispatchers.IO) {
                assets.list("models")?.toSet().orEmpty()
            }

            if (bundledNames.isEmpty()) {
                setStatus("同梱モデルが見つかりません。")
                showToast("同梱モデルが見つかりません。")
                return@runCatching
            }

            fun selectModelName(
                preferredName: String,
                fallbackPredicate: (String) -> Boolean,
            ): String? {
                if (bundledNames.contains(preferredName)) {
                    return preferredName
                }
                return bundledNames.sorted().firstOrNull(fallbackPredicate)
            }

            suspend fun materialize(name: String?): File? {
                if (name.isNullOrBlank()) {
                    return null
                }
                val outFile = File(localModelsDir, name)
                withContext(Dispatchers.IO) {
                    copyAssetToFileIfNeeded(assetPath = "models/$name", destination = outFile)
                }
                return outFile
            }

            val llmName = if (bundledNames.contains("MioTTS-0.1B-Q8_0.gguf")) {
                "MioTTS-0.1B-Q8_0.gguf"
            } else {
                null
            }
            val vocoderName = selectModelName("miocodec.gguf") {
                it.endsWith(".gguf") && it.contains("miocodec", ignoreCase = true)
            }
            val wavlmName = selectModelName("wavlm_base_plus_2l_f32.gguf") {
                it.endsWith(".gguf") && it.contains("wavlm", ignoreCase = true)
            }

            val llm = materialize(llmName)
            val vocoder = materialize(vocoderName)
            val wavlm = materialize(wavlmName)
            val preparedSpeakerFiles = mutableListOf<File>()
            for (key in defaultSpeakerKeys) {
                val fileName = "$key.emb.gguf"
                if (!bundledNames.contains(fileName)) {
                    continue
                }
                val prepared = materialize(fileName)
                if (prepared != null) {
                    preparedSpeakerFiles += prepared
                }
            }

            llm?.let { edtLlmModelPath.setText(it.absolutePath) }
            vocoder?.let { edtVocoderModelPath.setText(it.absolutePath) }
            wavlm?.let { edtWavlmModelPath.setText(it.absolutePath) }
            preparedSpeakerFiles.firstOrNull { it.name == "jp_female.emb.gguf" }
                ?.let { edtSpeakerEmbPath.setText(it.absolutePath) }
                ?: preparedSpeakerFiles.firstOrNull()?.let { edtSpeakerEmbPath.setText(it.absolutePath) }

            if (llm == null) {
                switchExternalApi.isChecked = true
            }

            val canAutoLoad = vocoder != null && wavlm != null && (switchExternalApi.isChecked || llm != null)
            if (canAutoLoad) {
                setStatus("同梱モデルを準備しました。モデルをロードします...")
            } else {
                setStatus("同梱モデルを準備しました。必要なモデルが不足しています。")
                showToast("同梱モデルの一部が不足しています。")
            }
        }.onFailure {
            setStatus("同梱モデルの準備に失敗: ${it.message}")
            showToast("同梱モデルの準備に失敗: ${it.message}")
        }

        isBusy = false
        refreshUiState()

        val canAutoLoadNow = edtVocoderModelPath.text.toString().isNotBlank() &&
            edtWavlmModelPath.text.toString().isNotBlank() &&
            (switchExternalApi.isChecked || edtLlmModelPath.text.toString().isNotBlank())
        if (canAutoLoadNow) {
            loadModels()
        }
    }

    private fun copyAssetToFileIfNeeded(assetPath: String, destination: File) {
        val expectedLength = runCatching {
            assets.openFd(assetPath).length
        }.getOrDefault(-1L)

        if (destination.exists()) {
            val hasValidLength = expectedLength > 0L && destination.length() == expectedLength
            val hasSomeContent = expectedLength <= 0L && destination.length() > 0L
            if (hasValidLength || hasSomeContent) {
                return
            }
        }

        assets.open(assetPath).use { input ->
            FileOutputStream(destination).use { out ->
                input.copyTo(out)
            }
        }
    }

    companion object {
        private val CODE_TOKEN_REGEX = Regex("<\\|s_(-?\\d+)\\|>")
    }
}
