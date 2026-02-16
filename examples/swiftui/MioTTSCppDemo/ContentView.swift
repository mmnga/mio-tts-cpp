import AVFoundation
import SwiftUI
import UniformTypeIdentifiers

@MainActor
final class MioTTSViewModel: NSObject, ObservableObject, AVAudioPlayerDelegate, AVAudioRecorderDelegate {
    struct GenerationParams {
        let nCtx: Int32
        let topK: Int32
        let topP: Float
        let temp: Float
    }

    @Published var isBusy = false
    @Published var isEngineLoading = true
    @Published var isEngineReady = false

    @Published var engineStatus = "モデルのロード中です..."
    @Published var createStatus = ""

    @Published var references: [MioReference] = []
    @Published var selectedReferenceKey = ""

    @Published var ctxSizeText = "700"
    @Published var topKText = "50"
    @Published var topPText = "1.0"
    @Published var tempText = "0.8"
    @Published var nPredict = 200
    @Published var useExternalLLMAPI = false
    @Published var externalLLMBaseURL = "http://localhost:8080/v1"
    @Published var externalLLMApiKey = "dummy"

    @Published var synthesisText = "こんにちわ、今日はいい天気ですね"
    @Published var synthStatus = "Ready"
    @Published var latestMetrics = ""
    @Published var latestGenerationMs = ""
    @Published var isSynthesizing = false

    @Published var isRecordingReference = false
    @Published var isPlaying = false

    private let engine = MioTTSLocalEngine()
    private let fileManager = FileManager.default

    private let bundledSpeakerKeys = ["en_female", "en_male", "jp_female", "jp_male"]
    private let autoKeyPrefix = "added_speaker_"
    // Keep reference generation bounded on mobile to avoid large transient memory spikes.
    private let maxReferenceSeconds: Float = 8.0

    private var recorder: AVAudioRecorder?
    private var player: AVAudioPlayer?

    var canRemoveSelectedReference: Bool {
        let key = selectedReferenceKey.trimmingCharacters(in: .whitespacesAndNewlines)
        return !key.isEmpty && !isBundledPresetReference(key)
    }

    override init() {
        super.init()

        do {
            _ = try referenceAudioDirectory()
            _ = try referenceEmbeddingDirectory()
        } catch {
            engineStatus = error.localizedDescription
            isEngineLoading = false
        }

        Task {
            await bootstrapOnLaunch()
        }
    }

    func bootstrapOnLaunch() async {
        isBusy = true
        isEngineLoading = true
        isEngineReady = false
        engineStatus = "モデルのロード中です..."

        defer {
            isBusy = false
            isEngineLoading = false
        }

        do {
            let llmURL = try requireBundledFile(named: "MioTTS-0.1B-Q8_0.gguf", subdirectories: ["Models", "Resources/Models"])
            let vocoderURL = try requireBundledFile(
                oneOf: ["miocodec.gguf", "miocodec-25hz_44khz.gguf", "miocodec-25hz.gguf"],
                subdirectories: ["Models", "Resources/Models"]
            )
            let wavlmURL = try requireBundledFile(named: "wavlm_base_plus_2l_f32.gguf", subdirectories: ["Models", "Resources/Models"])
            let fallbackSpeakerURL =
                findBundledFile(named: "jp_female.emb.gguf", subdirectories: ["Embeddings", "Resources/Embeddings"]) ??
                bundledSpeakerKeys
                    .compactMap { findBundledFile(named: "\($0).emb.gguf", subdirectories: ["Embeddings", "Resources/Embeddings"]) }
                    .first
            let bundledEmbeddingsDir = fallbackSpeakerURL?.deletingLastPathComponent().path

            let ctxSize = try parseInt32(ctxSizeText, name: "ctx-size", min: 256)

            try await engine.initialize(
                llmModelPath: llmURL.path,
                vocoderModelPath: vocoderURL.path,
                wavlmModelPath: wavlmURL.path,
                nGpuLayers: 0,
                nCtx: ctxSize,
                nThreads: 2,
                flashAttn: false
            )

            _ = try await applyGenerationParams()
            let preferredDefaultReference = try await engine.registerDefaultReferences(
                modelDirPath: bundledEmbeddingsDir,
                fallbackEmbeddingPath: fallbackSpeakerURL?.path
            )
            let restoredCount = await restorePersistedReferencesFromDisk()
            try await reloadReferencesFromEngine()

            if let preferredDefaultReference, references.contains(where: { $0.key == preferredDefaultReference }) {
                selectedReferenceKey = preferredDefaultReference
            } else if references.contains(where: { $0.key == "jp_female" }) {
                selectedReferenceKey = "jp_female"
            } else {
                selectedReferenceKey = references.first?.key ?? ""
            }

            isEngineReady = true
            engineStatus = "準備完了"
            if references.isEmpty {
                synthStatus = "既定話者が見つかりません。録音または音声ファイルで追加してください。"
            } else {
                synthStatus = "既定話者を読み込みました: \(references.map(\.key).joined(separator: ", "))"
            }
            if restoredCount > 0 {
                createStatus = "保存済み話者を \(restoredCount) 件読み込みました。"
            }
        } catch {
            engineStatus = error.localizedDescription
            synthStatus = error.localizedDescription
        }
    }

    func startReferenceRecording() async {
        guard isEngineReady else {
            createStatus = "モデル読み込み完了まで待ってください。"
            return
        }
        guard !isRecordingReference else { return }

        let granted = await requestRecordPermission()
        guard granted else {
            createStatus = "マイク権限がありません。"
            return
        }

        do {
            try await engine.unloadLLMRuntime()

            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker, .allowBluetooth])
            try session.setActive(true)

            let outputURL = try recordedAudioOutputURL()
            let settings: [String: Any] = [
                AVFormatIDKey: kAudioFormatLinearPCM,
                AVSampleRateKey: 16000,
                AVNumberOfChannelsKey: 1,
                AVLinearPCMBitDepthKey: 16,
                AVLinearPCMIsFloatKey: false,
                AVLinearPCMIsBigEndianKey: false
            ]

            let recorder = try AVAudioRecorder(url: outputURL, settings: settings)
            recorder.delegate = self

            guard recorder.prepareToRecord(), recorder.record(forDuration: TimeInterval(maxReferenceSeconds)) else {
                throw MioTTSLocalError.operationFailed("録音開始に失敗しました。")
            }

            self.recorder = recorder
            isRecordingReference = true
            createStatus = "録音中...（最大\(Int(maxReferenceSeconds))秒）停止すると自動で話者追加します。"
        } catch {
            isRecordingReference = false
            createStatus = error.localizedDescription
        }
    }

    func stopReferenceRecording() {
        guard isRecordingReference else { return }
        recorder?.stop()
    }

    func importAudioFileAndCreateReference(from sourceURL: URL) async {
        guard isEngineReady else {
            createStatus = "モデル読み込み完了まで待ってください。"
            return
        }

        do {
            let savedURL = try persistImportedAudioFile(from: sourceURL)
            createStatus = "音声を保存しました。話者を追加中..."
            await createReferenceFromAudio(savedURL)
        } catch {
            createStatus = error.localizedDescription
        }
    }

    func removeSelectedReference() async {
        guard isEngineReady else {
            createStatus = "モデル読み込み完了まで待ってください。"
            return
        }

        let key = selectedReferenceKey.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !key.isEmpty else {
            createStatus = "削除する話者を選択してください。"
            return
        }
        guard !isBundledPresetReference(key) else {
            createStatus = "既定の話者は削除できません。"
            return
        }

        isBusy = true
        defer { isBusy = false }

        do {
            try await engine.removeReference(referenceKey: key)
            let removedFile = try removeSavedEmbeddingIfExists(for: key)
            try await reloadReferencesFromEngine()

            if removedFile {
                createStatus = "\(key) を削除しました（保存ファイルも削除）。"
            } else {
                createStatus = "\(key) を削除しました。"
            }
            synthStatus = createStatus
        } catch {
            createStatus = error.localizedDescription
        }
    }

    func synthesizeAndPlay() async {
        guard isEngineReady else {
            synthStatus = "モデル読み込み中です。"
            return
        }

        let trimmed = synthesisText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            synthStatus = "テキストを入力してください。"
            return
        }

        guard !selectedReferenceKey.isEmpty else {
            synthStatus = "話者を選択してください。"
            return
        }

        isBusy = true
        isSynthesizing = true
        synthStatus = "音声生成中..."
        defer {
            isBusy = false
            isSynthesizing = false
        }

        let outURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("mio-tts-\(UUID().uuidString)")
            .appendingPathExtension("wav")

        let begin = Date()

        do {
            let generationParams = try await applyGenerationParams()

            let generationBegin = Date()
            let sourceLabel: String
            let generatedCodeCount: Int?

            if useExternalLLMAPI {
                let codes = try await engine.generateCodesFromExternalAPI(
                    baseURL: externalLLMBaseURL,
                    apiKey: externalLLMApiKey,
                    text: trimmed,
                    nPredict: Int32(nPredict),
                    topK: generationParams.topK,
                    topP: generationParams.topP,
                    temp: generationParams.temp
                )
                let generationMs = Date().timeIntervalSince(generationBegin) * 1000.0

                try await engine.synthesizeCodesToWav(
                    codes: codes,
                    referenceKey: selectedReferenceKey,
                    outputPath: outURL.path
                )

                try play(outURL)

                let elapsedMs = Date().timeIntervalSince(begin) * 1000.0
                latestGenerationMs = String(format: "%.0f", generationMs)
                sourceLabel = "api"
                generatedCodeCount = codes.count
                latestMetrics = "source=\(sourceLabel), codes=\(generatedCodeCount ?? 0), wav=\(outURL.lastPathComponent), generation=\(String(format: "%.0f", generationMs))ms, total=\(String(format: "%.0f", elapsedMs))ms"
                synthStatus = "生成完了（\(latestGenerationMs)ms, API）。再生中。"
                return
            } else {
                try await engine.synthesizeToWav(
                    text: trimmed,
                    referenceKey: selectedReferenceKey,
                    nPredict: Int32(nPredict),
                    outputPath: outURL.path
                )
                sourceLabel = "local"
                generatedCodeCount = nil
            }

            let generationMs = Date().timeIntervalSince(generationBegin) * 1000.0

            try play(outURL)

            let elapsedMs = Date().timeIntervalSince(begin) * 1000.0
            latestGenerationMs = String(format: "%.0f", generationMs)
            if let generatedCodeCount {
                latestMetrics = "source=\(sourceLabel), codes=\(generatedCodeCount), wav=\(outURL.lastPathComponent), generation=\(String(format: "%.0f", generationMs))ms, total=\(String(format: "%.0f", elapsedMs))ms"
            } else {
                latestMetrics = "source=\(sourceLabel), wav=\(outURL.lastPathComponent), generation=\(String(format: "%.0f", generationMs))ms, total=\(String(format: "%.0f", elapsedMs))ms"
            }
            synthStatus = "生成完了（\(latestGenerationMs)ms）。再生中。"
        } catch {
            synthStatus = error.localizedDescription
        }
    }

    func stopAudio() {
        player?.stop()
        isPlaying = false
    }

    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor in
            self.isPlaying = false
            if flag {
                self.synthStatus = "再生終了"
            }
        }
    }

    nonisolated func audioRecorderDidFinishRecording(_ recorder: AVAudioRecorder, successfully flag: Bool) {
        Task { @MainActor in
            self.isRecordingReference = false
            self.recorder = nil

            do {
                try AVAudioSession.sharedInstance().setActive(false, options: [.notifyOthersOnDeactivation])
            } catch {
                // Ignore deactivation failure.
            }

            guard flag else {
                self.createStatus = "録音に失敗しました。"
                return
            }

            self.createStatus = "録音完了。話者を追加中..."
            await self.createReferenceFromAudio(recorder.url)
        }
    }

    nonisolated func audioRecorderEncodeErrorDidOccur(_ recorder: AVAudioRecorder, error: Error?) {
        guard let error else { return }
        Task { @MainActor in
            self.isRecordingReference = false
            self.recorder = nil
            self.createStatus = error.localizedDescription
        }
    }

    private func createReferenceFromAudio(_ audioURL: URL) async {
        guard isEngineReady else {
            createStatus = "モデル読み込み完了まで待ってください。"
            return
        }

        isBusy = true
        defer { isBusy = false }

        do {
            try await engine.unloadLLMRuntime()

            let referenceKey = nextGeneratedReferenceKey()
            let embeddingURL = try embeddingOutputURL(for: referenceKey)

            try await engine.createReferenceFromAudio(
                referenceKey: referenceKey,
                audioPath: audioURL.path,
                maxReferenceSeconds: maxReferenceSeconds,
                saveEmbeddingPath: embeddingURL.path
            )

            try await reloadReferencesFromEngine()
            selectedReferenceKey = referenceKey
            createStatus = "\(referenceKey).emb.gguf を作成して追加しました。"
            synthStatus = "話者を追加しました: \(referenceKey)"
        } catch {
            createStatus = error.localizedDescription
        }
    }

    private func applyGenerationParams() async throws -> GenerationParams {
        let ctx = try parseInt32(ctxSizeText, name: "ctx-size", min: 256)
        let topK = try parseInt32(topKText, name: "top-k", min: 1)
        let topP = try parseFloat(topPText, name: "top-p", minExclusive: 0.0, maxInclusive: 1.0)
        let temp = try parseFloat(tempText, name: "temp", minExclusive: -0.0001, maxInclusive: 10.0)

        try await engine.setGenerationParams(nCtx: ctx, topK: topK, topP: topP, temp: temp)
        return GenerationParams(nCtx: ctx, topK: topK, topP: topP, temp: temp)
    }

    private func reloadReferencesFromEngine() async throws {
        let refs = try await engine.listReferences().sorted { $0.key < $1.key }
        references = refs

        if selectedReferenceKey.isEmpty || !refs.contains(where: { $0.key == selectedReferenceKey }) {
            selectedReferenceKey = refs.first?.key ?? ""
        }
    }

    private func play(_ wavURL: URL) throws {
        player?.stop()

        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playback, mode: .default, options: [])
        try session.setPreferredSampleRate(48000)
        try session.setActive(true)

        let player = try AVAudioPlayer(contentsOf: wavURL)
        player.delegate = self
        player.prepareToPlay()
        player.play()
        self.player = player
        isPlaying = true
    }

    private func requireBundledFile(named fileName: String, subdirectories: [String]) throws -> URL {
        if let url = findBundledFile(named: fileName, subdirectories: subdirectories) {
            return url
        }
        throw MioTTSLocalError.operationFailed("リソースが見つかりません: \(fileName)")
    }

    private func requireBundledFile(oneOf fileNames: [String], subdirectories: [String]) throws -> URL {
        for fileName in fileNames {
            if let url = try? requireBundledFile(named: fileName, subdirectories: subdirectories) {
                return url
            }
        }
        throw MioTTSLocalError.operationFailed("リソースが見つかりません: \(fileNames.joined(separator: ", "))")
    }

    private func findBundledFile(named fileName: String, subdirectories: [String]) -> URL? {
        for subdirectory in subdirectories {
            if let url = Bundle.main.url(forResource: fileName, withExtension: nil, subdirectory: subdirectory) {
                return url
            }
        }
        return Bundle.main.url(forResource: fileName, withExtension: nil)
    }

    private func parseInt32(_ raw: String, name: String, min: Int32) throws -> Int32 {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let value = Int32(trimmed), value >= min else {
            throw MioTTSLocalError.operationFailed("\(name) は \(min) 以上の整数にしてください。")
        }
        return value
    }

    private func parseFloat(_ raw: String, name: String, minExclusive: Float, maxInclusive: Float) throws -> Float {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let value = Float(trimmed), value > minExclusive, value <= maxInclusive else {
            throw MioTTSLocalError.operationFailed("\(name) の値が不正です。")
        }
        return value
    }

    private func requestRecordPermission() async -> Bool {
        if #available(iOS 17.0, *) {
            return await AVAudioApplication.requestRecordPermission()
        }

        return await withCheckedContinuation { continuation in
            AVAudioSession.sharedInstance().requestRecordPermission { allowed in
                continuation.resume(returning: allowed)
            }
        }
    }

    private func nextGeneratedReferenceKey() -> String {
        var maxIndex = 0

        for key in references.map(\.key) {
            guard key.hasPrefix(autoKeyPrefix) else { continue }
            let suffix = key.dropFirst(autoKeyPrefix.count)
            if let index = Int(suffix), index > maxIndex {
                maxIndex = index
            }
        }

        return "\(autoKeyPrefix)\(maxIndex + 1)"
    }

    private func isBundledPresetReference(_ key: String) -> Bool {
        bundledSpeakerKeys.contains(key)
    }

    private func persistImportedAudioFile(from sourceURL: URL) throws -> URL {
        let didAccess = sourceURL.startAccessingSecurityScopedResource()
        defer {
            if didAccess {
                sourceURL.stopAccessingSecurityScopedResource()
            }
        }

        let ext = sourceURL.pathExtension.isEmpty ? "wav" : sourceURL.pathExtension
        let destination = try referenceAudioDirectory()
            .appendingPathComponent("reference-import-\(timestampMillis())-\(shortUUID())")
            .appendingPathExtension(ext)

        if fileManager.fileExists(atPath: destination.path) {
            try fileManager.removeItem(at: destination)
        }

        try fileManager.copyItem(at: sourceURL, to: destination)
        return destination
    }

    private func recordedAudioOutputURL() throws -> URL {
        try referenceAudioDirectory()
            .appendingPathComponent("reference-recorded-\(timestampMillis())")
            .appendingPathExtension("wav")
    }

    private func embeddingOutputURL(for referenceKey: String) throws -> URL {
        let safeKey = sanitizeReferenceKey(referenceKey)
        return try referenceEmbeddingDirectory().appendingPathComponent("\(safeKey).emb.gguf")
    }

    private func removeSavedEmbeddingIfExists(for referenceKey: String) throws -> Bool {
        let path = try embeddingOutputURL(for: referenceKey)
        guard fileManager.fileExists(atPath: path.path) else {
            return false
        }
        try fileManager.removeItem(at: path)
        return true
    }

    private func restorePersistedReferencesFromDisk() async -> Int {
        let dir: URL
        do {
            dir = try referenceEmbeddingDirectory()
        } catch {
            return 0
        }

        let urls: [URL]
        do {
            urls = try fileManager.contentsOfDirectory(
                at: dir,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles]
            )
        } catch {
            return 0
        }

        var loaded = 0
        let sorted = urls.sorted { $0.lastPathComponent < $1.lastPathComponent }
        for url in sorted {
            guard url.pathExtension.lowercased() == "gguf" else { continue }
            let fileName = url.lastPathComponent
            guard fileName.lowercased().hasSuffix(".emb.gguf") else { continue }

            let rawKey = String(fileName.dropLast(".emb.gguf".count))
            let key = sanitizeReferenceKey(rawKey)
            guard !key.isEmpty else { continue }

            do {
                try await engine.addReference(referenceKey: key, embeddingPath: url.path)
                loaded += 1
            } catch {
                // Ignore corrupted/incompatible files and continue.
            }
        }

        return loaded
    }

    private func sanitizeReferenceKey(_ key: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "_-"))
        let transformed = key.unicodeScalars.map { allowed.contains($0) ? Character($0) : "_" }
        let safe = String(transformed)
        return safe.isEmpty ? "speaker" : safe
    }

    private func appSupportDirectory() throws -> URL {
        let base = try fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )

        let root = base.appendingPathComponent("MioTTSCppDemo", isDirectory: true)
        try ensureDirectory(root)
        return root
    }

    private func referenceAudioDirectory() throws -> URL {
        let dir = try appSupportDirectory().appendingPathComponent("reference-audio", isDirectory: true)
        try ensureDirectory(dir)
        return dir
    }

    private func referenceEmbeddingDirectory() throws -> URL {
        let dir = try appSupportDirectory().appendingPathComponent("reference-embeddings", isDirectory: true)
        try ensureDirectory(dir)
        return dir
    }

    private func ensureDirectory(_ url: URL) throws {
        var isDirectory: ObjCBool = false
        if fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory) {
            if !isDirectory.boolValue {
                throw MioTTSLocalError.operationFailed("ディレクトリではありません: \(url.path)")
            }
            return
        }

        try fileManager.createDirectory(at: url, withIntermediateDirectories: true)
    }

    private func timestampMillis() -> Int64 {
        Int64(Date().timeIntervalSince1970 * 1000)
    }

    private func shortUUID() -> String {
        String(UUID().uuidString.prefix(8))
    }
}

struct ContentView: View {
    private enum InputField: Hashable {
        case ctxSize
        case topK
        case topP
        case temp
        case synthesisText
        case externalLLMBaseURL
        case externalLLMApiKey
    }

    @StateObject private var vm = MioTTSViewModel()
    @FocusState private var focusedField: InputField?

    @State private var showAudioImporter = false

    private var loadingToastMessage: String? {
        if vm.isEngineLoading {
            return "モデル読み込み中..."
        }
        if vm.isSynthesizing {
            return "音声生成中..."
        }
        if vm.isBusy {
            return "処理中..."
        }
        return nil
    }

    var body: some View {
        NavigationStack {
            Form {
                Section("状態") {
                    if vm.isEngineLoading {
                        ProgressView("モデルのロード中です...")
                    } else if vm.isSynthesizing {
                        ProgressView("音声生成中...")
                    } else if vm.isBusy {
                        ProgressView("処理中...")
                    }

                    Text(vm.engineStatus)
                        .font(.footnote)

                    if !vm.createStatus.isEmpty {
                        Text(vm.createStatus)
                            .font(.footnote)
                    }
                }

                Section("話者") {
                    if vm.references.isEmpty {
                        Text("既定話者の読み込み待ち")
                            .foregroundStyle(.secondary)
                    } else {
                        Picker("voice", selection: $vm.selectedReferenceKey) {
                            ForEach(vm.references) { ref in
                                Text(ref.key).tag(ref.key)
                            }
                        }
                    }

                    Button {
                        focusedField = nil
                        if vm.isRecordingReference {
                            vm.stopReferenceRecording()
                        } else {
                            Task { await vm.startReferenceRecording() }
                        }
                    } label: {
                        HStack(spacing: 6) {
                            if vm.isRecordingReference {
                                Image(systemName: "stop.fill")
                                    .font(.caption)
                                Text("STOP")
                            } else {
                                Circle()
                                    .fill(.white)
                                    .frame(width: 10, height: 10)
                                Text("REC")
                            }
                        }
                        .fontWeight(.bold)
                        .frame(maxWidth: .infinity)
                    }
                    .disabled(vm.isEngineLoading || vm.isBusy)
                    .buttonStyle(.borderedProminent)
                    .tint(.red)
                    .controlSize(.large)

                    Button {
                        focusedField = nil
                        showAudioImporter = true
                    } label: {
                        Text("音声ファイルから話者追加")
                            .frame(maxWidth: .infinity)
                    }
                    .disabled(vm.isEngineLoading || vm.isBusy || vm.isRecordingReference)
                    .buttonStyle(.bordered)
                    .controlSize(.large)

                    Button {
                        focusedField = nil
                        Task { await vm.removeSelectedReference() }
                    } label: {
                        Text("選択話者を削除")
                            .frame(maxWidth: .infinity)
                    }
                    .disabled(vm.isEngineLoading || vm.isBusy || vm.isRecordingReference || !vm.canRemoveSelectedReference)
                    .buttonStyle(.bordered)
                    .controlSize(.large)
                    .tint(.red)

                    if !vm.selectedReferenceKey.isEmpty && !vm.canRemoveSelectedReference {
                        Text("既定話者（preset）は削除できません。")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                Section("LLM") {
                    Toggle("外部API", isOn: $vm.useExternalLLMAPI)

                    if vm.useExternalLLMAPI {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("BaseURL")
                                .font(.subheadline.weight(.semibold))
                            TextField("http://localhost:8080/v1", text: $vm.externalLLMBaseURL)
                                .textInputAutocapitalization(.never)
                                .autocorrectionDisabled(true)
                                .keyboardType(.URL)
                                .focused($focusedField, equals: .externalLLMBaseURL)
                            Text("llama-server(OpenAI互換)のURLです。既定は http://localhost:8080/v1 です。")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }

                        VStack(alignment: .leading, spacing: 6) {
                            Text("api-key")
                                .font(.subheadline.weight(.semibold))
                            TextField("dummy", text: $vm.externalLLMApiKey)
                                .textInputAutocapitalization(.never)
                                .autocorrectionDisabled(true)
                                .focused($focusedField, equals: .externalLLMApiKey)
                            Text("既定値は dummy です。必要に応じて変更してください。")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                }

                Section("生成パラメータ") {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("ctx-size")
                            .font(.subheadline.weight(.semibold))
                        TextField("700", text: $vm.ctxSizeText)
                            .keyboardType(.numberPad)
                            .focused($focusedField, equals: .ctxSize)
                        Text("LLMのコンテキスト長。長文対応には有利ですが、大きいほどメモリ使用量が増えます。")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    VStack(alignment: .leading, spacing: 6) {
                        Text("top-k")
                            .font(.subheadline.weight(.semibold))
                        TextField("50", text: $vm.topKText)
                            .keyboardType(.numberPad)
                            .focused($focusedField, equals: .topK)
                        Text("次トークン候補の上位K件だけから選びます。小さいほど安定し、大きいほど多様になります。")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    VStack(alignment: .leading, spacing: 6) {
                        Text("top-p")
                            .font(.subheadline.weight(.semibold))
                        TextField("1.0", text: $vm.topPText)
                            .keyboardType(.decimalPad)
                            .focused($focusedField, equals: .topP)
                        Text("累積確率がpに達する候補集合から選びます。小さいほど保守的な生成になります。")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    VStack(alignment: .leading, spacing: 6) {
                        Text("temp")
                            .font(.subheadline.weight(.semibold))
                        TextField("0.8", text: $vm.tempText)
                            .keyboardType(.decimalPad)
                            .focused($focusedField, equals: .temp)
                        Text("温度。0に近いほど決定的で、値を上げるほどランダム性が増えます。")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    VStack(alignment: .leading, spacing: 6) {
                        Stepper("n_predict: \(vm.nPredict)", value: $vm.nPredict, in: 16...700, step: 1)
                        Text("生成する音声トークン数。大きいほど長い音声になり、生成時間も増えます。")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                Section("音声生成") {
                    if vm.isSynthesizing {
                        ProgressView("音声生成中...")
                    }

                    TextEditor(text: $vm.synthesisText)
                        .frame(minHeight: 120)
                        .focused($focusedField, equals: .synthesisText)

                    VStack(spacing: 10) {
                        Button {
                            focusedField = nil
                            Task { await vm.synthesizeAndPlay() }
                        } label: {
                            Text("生成して再生")
                                .frame(maxWidth: .infinity)
                        }
                        .disabled(vm.isEngineLoading || vm.isBusy || vm.selectedReferenceKey.isEmpty)
                        .buttonStyle(.borderedProminent)
                        .controlSize(.large)

                        if vm.isPlaying {
                            Button {
                                vm.stopAudio()
                            } label: {
                                Text("停止")
                                    .frame(maxWidth: .infinity)
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.large)
                            .tint(.red)
                        }
                    }
                    .frame(maxWidth: .infinity)

                    Text(vm.synthStatus)
                        .font(.footnote)

                    if !vm.latestGenerationMs.isEmpty {
                        Text("生成時間: \(vm.latestGenerationMs) ms")
                            .font(.footnote)
                    }

                    if !vm.latestMetrics.isEmpty {
                        Text(vm.latestMetrics)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .navigationTitle("MioTTS-Cpp")
            .overlay(alignment: .top) {
                if let message = loadingToastMessage {
                    HStack(spacing: 8) {
                        ProgressView()
                            .controlSize(.small)
                        Text(message)
                            .font(.footnote.weight(.semibold))
                    }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(.ultraThinMaterial, in: Capsule())
                    .shadow(color: .black.opacity(0.12), radius: 8, y: 3)
                    .padding(.top, 8)
                    .transition(.move(edge: .top).combined(with: .opacity))
                }
            }
            .animation(.easeInOut(duration: 0.2), value: loadingToastMessage != nil)
            .fileImporter(
                isPresented: $showAudioImporter,
                allowedContentTypes: [.audio],
                allowsMultipleSelection: false
            ) { result in
                switch result {
                case .success(let urls):
                    guard let url = urls.first else { return }
                    Task {
                        await vm.importAudioFileAndCreateReference(from: url)
                    }
                case .failure(let error):
                    vm.createStatus = error.localizedDescription
                }
            }
        }
    }
}
