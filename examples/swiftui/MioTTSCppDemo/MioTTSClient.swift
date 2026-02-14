import Foundation

struct MioReference: Identifiable, Hashable {
    let key: String
    let embeddingDim: Int

    var id: String { key }
}

enum MioTTSLocalError: LocalizedError {
    case notInitialized
    case operationFailed(String)
    case invalidResponse

    var errorDescription: String? {
        switch self {
        case .notInitialized:
            return "Engine is not initialized."
        case .operationFailed(let message):
            return message
        case .invalidResponse:
            return "Engine returned an invalid response."
        }
    }
}

private struct MioReferenceDTO: Decodable {
    let key: String
    let embeddingDim: Int

    enum CodingKeys: String, CodingKey {
        case key
        case embeddingDim = "embedding_dim"
    }
}

private struct MioExternalLegacyCodesRequest: Encodable {
    let text: String
    let nPredict: Int32
    let topK: Int32
    let topP: Float
    let temp: Float
    let codesOnly: Bool

    enum CodingKeys: String, CodingKey {
        case text
        case nPredict = "n_predict"
        case topK = "top_k"
        case topP = "top_p"
        case temp
        case codesOnly = "codes_only"
    }
}

private struct MioExternalChatMessage: Encodable {
    let role: String
    let content: String
}

private struct MioExternalChatRequest: Encodable {
    let messages: [MioExternalChatMessage]
    let maxTokens: Int32
    let nPredict: Int32
    let topK: Int32
    let topP: Float
    let temperature: Float
    let stream: Bool

    enum CodingKeys: String, CodingKey {
        case messages
        case maxTokens = "max_tokens"
        case nPredict = "n_predict"
        case topK = "top_k"
        case topP = "top_p"
        case temperature
        case stream
    }
}

actor MioTTSLocalEngine {
    private var handle: UnsafeMutableRawPointer?

    deinit {
        if let handle {
            mio_swift_engine_destroy(handle)
        }
    }

    func initialize(
        llmModelPath: String,
        vocoderModelPath: String,
        wavlmModelPath: String?,
        nGpuLayers: Int32,
        nCtx: Int32,
        nThreads: Int32,
        flashAttn: Bool
    ) throws {
        close()

        var err = Self.makeErrorBuffer()
        let created: UnsafeMutableRawPointer? = err.withUnsafeMutableBufferPointer { errBuf in
            llmModelPath.withCString { llmC in
                vocoderModelPath.withCString { vocoderC in
                    if let wavlmModelPath, !wavlmModelPath.isEmpty {
                        return wavlmModelPath.withCString { wavlmC in
                            mio_swift_engine_create(
                                llmC,
                                vocoderC,
                                wavlmC,
                                nGpuLayers,
                                nCtx,
                                nThreads,
                                flashAttn,
                                errBuf.baseAddress,
                                errBuf.count
                            )
                        }
                    }

                    return mio_swift_engine_create(
                        llmC,
                        vocoderC,
                        nil,
                        nGpuLayers,
                        nCtx,
                        nThreads,
                        flashAttn,
                        errBuf.baseAddress,
                        errBuf.count
                    )
                }
            }
        }

        guard let created else {
            throw MioTTSLocalError.operationFailed(Self.message(from: err, fallback: "Failed to initialize Mio engine."))
        }

        handle = created
    }

    func close() {
        if let handle {
            mio_swift_engine_destroy(handle)
            self.handle = nil
        }
    }

    func setGenerationParams(
        nCtx: Int32,
        topK: Int32,
        topP: Float,
        temp: Float
    ) throws {
        guard let handle else {
            throw MioTTSLocalError.notInitialized
        }

        var err = Self.makeErrorBuffer()
        let ok = err.withUnsafeMutableBufferPointer { errBuf in
            mio_swift_engine_set_generation_params(
                handle,
                nCtx,
                topK,
                topP,
                temp,
                errBuf.baseAddress,
                errBuf.count
            )
        }

        guard ok else {
            throw MioTTSLocalError.operationFailed(Self.message(from: err, fallback: "Failed to update generation params."))
        }
    }

    func unloadLLMRuntime() throws {
        guard let handle else {
            throw MioTTSLocalError.notInitialized
        }

        var err = Self.makeErrorBuffer()
        let ok = err.withUnsafeMutableBufferPointer { errBuf in
            mio_swift_engine_unload_llm_runtime(
                handle,
                errBuf.baseAddress,
                errBuf.count
            )
        }

        guard ok else {
            throw MioTTSLocalError.operationFailed(Self.message(from: err, fallback: "Failed to unload LLM runtime."))
        }
    }

    func listReferences() throws -> [MioReference] {
        guard let handle else {
            throw MioTTSLocalError.notInitialized
        }

        var err = Self.makeErrorBuffer()
        var jsonPtr: UnsafeMutablePointer<CChar>?

        let ok = err.withUnsafeMutableBufferPointer { errBuf in
            mio_swift_engine_list_references_json(handle, &jsonPtr, errBuf.baseAddress, errBuf.count)
        }

        guard ok else {
            throw MioTTSLocalError.operationFailed(Self.message(from: err, fallback: "Failed to load references."))
        }

        guard let jsonPtr else {
            throw MioTTSLocalError.invalidResponse
        }
        defer {
            mio_swift_string_free(jsonPtr)
        }

        let json = String(cString: jsonPtr)
        guard let data = json.data(using: .utf8) else {
            throw MioTTSLocalError.invalidResponse
        }

        let decoded = try JSONDecoder().decode([MioReferenceDTO].self, from: data)
        return decoded.map { MioReference(key: $0.key, embeddingDim: $0.embeddingDim) }
    }

    func createReferenceFromAudio(
        referenceKey: String,
        audioPath: String,
        maxReferenceSeconds: Float,
        saveEmbeddingPath: String?
    ) throws {
        guard let handle else {
            throw MioTTSLocalError.notInitialized
        }

        var err = Self.makeErrorBuffer()
        let ok = err.withUnsafeMutableBufferPointer { errBuf in
            referenceKey.withCString { keyC in
                audioPath.withCString { audioC in
                    if let saveEmbeddingPath, !saveEmbeddingPath.isEmpty {
                        return saveEmbeddingPath.withCString { saveC in
                            mio_swift_engine_create_reference_from_audio(
                                handle,
                                keyC,
                                audioC,
                                maxReferenceSeconds,
                                saveC,
                                errBuf.baseAddress,
                                errBuf.count
                            )
                        }
                    }

                    return mio_swift_engine_create_reference_from_audio(
                        handle,
                        keyC,
                        audioC,
                        maxReferenceSeconds,
                        nil,
                        errBuf.baseAddress,
                        errBuf.count
                    )
                }
            }
        }

        guard ok else {
            throw MioTTSLocalError.operationFailed(Self.message(from: err, fallback: "Failed to create reference from audio."))
        }
    }

    func addReference(referenceKey: String, embeddingPath: String) throws {
        guard let handle else {
            throw MioTTSLocalError.notInitialized
        }

        var err = Self.makeErrorBuffer()
        let ok = err.withUnsafeMutableBufferPointer { errBuf in
            referenceKey.withCString { keyC in
                embeddingPath.withCString { embC in
                    mio_swift_engine_add_reference_from_gguf(
                        handle,
                        keyC,
                        embC,
                        errBuf.baseAddress,
                        errBuf.count
                    )
                }
            }
        }

        guard ok else {
            throw MioTTSLocalError.operationFailed(Self.message(from: err, fallback: "Failed to add reference."))
        }
    }

    func removeReference(referenceKey: String) throws {
        guard let handle else {
            throw MioTTSLocalError.notInitialized
        }

        var err = Self.makeErrorBuffer()
        let ok = err.withUnsafeMutableBufferPointer { errBuf in
            referenceKey.withCString { keyC in
                mio_swift_engine_remove_reference(
                    handle,
                    keyC,
                    errBuf.baseAddress,
                    errBuf.count
                )
            }
        }

        guard ok else {
            throw MioTTSLocalError.operationFailed(Self.message(from: err, fallback: "Failed to remove reference."))
        }
    }

    func registerDefaultReferences(modelDirPath: String?, fallbackEmbeddingPath: String?) throws -> String? {
        guard let handle else {
            throw MioTTSLocalError.notInitialized
        }

        var err = Self.makeErrorBuffer()
        var preferredPtr: UnsafeMutablePointer<CChar>?

        let modelDir = modelDirPath?.trimmingCharacters(in: .whitespacesAndNewlines)
        let fallbackPath = fallbackEmbeddingPath?.trimmingCharacters(in: .whitespacesAndNewlines)
        let modelDirCString: String? = (modelDir?.isEmpty == false) ? modelDir : nil
        let fallbackCString: String? = (fallbackPath?.isEmpty == false) ? fallbackPath : nil

        let ok = err.withUnsafeMutableBufferPointer { errBuf in
            if let modelDirCString {
                return modelDirCString.withCString { modelDirC in
                    if let fallbackCString {
                        return fallbackCString.withCString { fallbackC in
                            mio_swift_engine_register_default_references(
                                handle,
                                modelDirC,
                                fallbackC,
                                &preferredPtr,
                                errBuf.baseAddress,
                                errBuf.count
                            )
                        }
                    }

                    return mio_swift_engine_register_default_references(
                        handle,
                        modelDirC,
                        nil,
                        &preferredPtr,
                        errBuf.baseAddress,
                        errBuf.count
                    )
                }
            }

            if let fallbackCString {
                return fallbackCString.withCString { fallbackC in
                    mio_swift_engine_register_default_references(
                        handle,
                        nil,
                        fallbackC,
                        &preferredPtr,
                        errBuf.baseAddress,
                        errBuf.count
                    )
                }
            }

            return mio_swift_engine_register_default_references(
                handle,
                nil,
                nil,
                &preferredPtr,
                errBuf.baseAddress,
                errBuf.count
            )
        }

        guard ok else {
            throw MioTTSLocalError.operationFailed(Self.message(from: err, fallback: "Failed to register default references."))
        }

        guard let preferredPtr else {
            return nil
        }
        defer {
            mio_swift_string_free(preferredPtr)
        }

        let preferred = String(cString: preferredPtr)
        return preferred.isEmpty ? nil : preferred
    }

    func synthesizeToWav(
        text: String,
        referenceKey: String,
        nPredict: Int32,
        outputPath: String
    ) throws {
        guard let handle else {
            throw MioTTSLocalError.notInitialized
        }

        var err = Self.makeErrorBuffer()
        let ok = err.withUnsafeMutableBufferPointer { errBuf in
            text.withCString { textC in
                referenceKey.withCString { keyC in
                    outputPath.withCString { outC in
                        mio_swift_engine_synthesize_to_wav(
                            handle,
                            textC,
                            keyC,
                            nPredict,
                            outC,
                            errBuf.baseAddress,
                            errBuf.count
                        )
                    }
                }
            }
        }

        guard ok else {
            throw MioTTSLocalError.operationFailed(Self.message(from: err, fallback: "Synthesis failed."))
        }
    }

    func synthesizeCodesToWav(
        codes: [Int32],
        referenceKey: String,
        outputPath: String
    ) throws {
        guard let handle else {
            throw MioTTSLocalError.notInitialized
        }

        guard !codes.isEmpty else {
            throw MioTTSLocalError.operationFailed("External API returned empty codes.")
        }

        var err = Self.makeErrorBuffer()
        let ok = err.withUnsafeMutableBufferPointer { errBuf in
            codes.withUnsafeBufferPointer { codesBuf in
                referenceKey.withCString { keyC in
                    outputPath.withCString { outC in
                        mio_swift_engine_synthesize_codes_to_wav(
                            handle,
                            codesBuf.baseAddress,
                            codesBuf.count,
                            keyC,
                            outC,
                            errBuf.baseAddress,
                            errBuf.count
                        )
                    }
                }
            }
        }

        guard ok else {
            throw MioTTSLocalError.operationFailed(Self.message(from: err, fallback: "Synthesis from external codes failed."))
        }
    }

    func generateCodesFromExternalAPI(
        baseURL: String,
        apiKey: String,
        text: String,
        nPredict: Int32,
        topK: Int32,
        topP: Float,
        temp: Float
    ) async throws -> [Int32] {
        let trimmedBaseURL = baseURL.trimmingCharacters(in: .whitespacesAndNewlines)
        guard var normalizedURL = URL(string: trimmedBaseURL) else {
            throw MioTTSLocalError.operationFailed("API BaseURL が不正です。")
        }

        if normalizedURL.scheme == nil {
            guard let url = URL(string: "http://\(trimmedBaseURL)") else {
                throw MioTTSLocalError.operationFailed("API BaseURL が不正です。")
            }
            normalizedURL = url
        }

        var pathLower = normalizedURL.path.lowercased()
        while pathLower.count > 1 && pathLower.hasSuffix("/") {
            pathLower.removeLast()
        }

        let endpoint: URL
        if pathLower.hasSuffix("/v1/chat/completions") ||
            pathLower.hasSuffix("/v1/completions") ||
            pathLower.hasSuffix("/mio/tts") {
            endpoint = normalizedURL
        } else if pathLower.hasSuffix("/v1") {
            endpoint = normalizedURL.appendingPathComponent("chat/completions")
        } else {
            endpoint = normalizedURL.appendingPathComponent("v1/chat/completions")
        }

        let endpointPathLower = endpoint.path.lowercased()
        let useLegacyMioEndpoint = endpointPathLower.hasSuffix("/mio/tts")
        let useCompletionsEndpoint = endpointPathLower.hasSuffix("/v1/completions")

        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")

        let trimmedApiKey = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedApiKey.isEmpty {
            request.setValue("Bearer \(trimmedApiKey)", forHTTPHeaderField: "Authorization")
            request.setValue(trimmedApiKey, forHTTPHeaderField: "X-API-Key")
        }

        if useLegacyMioEndpoint {
            let payload = MioExternalLegacyCodesRequest(
                text: text,
                nPredict: nPredict,
                topK: topK,
                topP: topP,
                temp: temp,
                codesOnly: true
            )
            request.httpBody = try JSONEncoder().encode(payload)
        } else if useCompletionsEndpoint {
            let payload: [String: Any] = [
                "prompt": text,
                "max_tokens": nPredict,
                "n_predict": nPredict,
                "top_k": topK,
                "top_p": topP,
                "temperature": temp,
                "stream": false
            ]
            request.httpBody = try JSONSerialization.data(withJSONObject: payload)
        } else {
            let payload = MioExternalChatRequest(
                messages: [MioExternalChatMessage(role: "user", content: text)],
                maxTokens: nPredict,
                nPredict: nPredict,
                topK: topK,
                topP: topP,
                temperature: temp,
                stream: false
            )
            request.httpBody = try JSONEncoder().encode(payload)
        }

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw MioTTSLocalError.operationFailed("外部API応答が不正です。")
        }

        let jsonObject = try? JSONSerialization.jsonObject(with: data)
        let parsedCodes = Self.parseCodesFromResponse(jsonObject, data: data)

        if !(200...299).contains(httpResponse.statusCode) {
            if let message = Self.extractErrorMessage(from: jsonObject), !message.isEmpty {
                throw MioTTSLocalError.operationFailed("外部APIエラー: \(message)")
            }
            let bodyText = String(data: data, encoding: .utf8) ?? ""
            throw MioTTSLocalError.operationFailed("外部APIエラー (\(httpResponse.statusCode)): \(bodyText)")
        }

        if let codes = parsedCodes, !codes.isEmpty {
            return codes
        }

        throw MioTTSLocalError.operationFailed("外部API応答から音声コードを取得できませんでした。llama-serverの出力に <|s_...|> が含まれているか確認してください。")
    }

    private static func parseCodesFromResponse(_ jsonObject: Any?, data: Data) -> [Int32]? {
        if let object = jsonObject as? [String: Any] {
            if let values = parseCodesArray(object["codes_values"]) {
                return values
            }
            if let values = parseCodesArray(object["codes"]) {
                return values
            }
            if let values = parseCodesArray(object["audio_codes"]) {
                return values
            }
            if let text = extractText(from: object) {
                let values = extractCodesFromText(text)
                if !values.isEmpty {
                    return values
                }
            }
        }

        if let text = String(data: data, encoding: .utf8) {
            let values = extractCodesFromText(text)
            if !values.isEmpty {
                return values
            }
        }

        return nil
    }

    private static func parseCodesArray(_ raw: Any?) -> [Int32]? {
        guard let array = raw as? [Any] else {
            return nil
        }
        var out: [Int32] = []
        out.reserveCapacity(array.count)

        for value in array {
            if let i = value as? Int {
                out.append(Int32(i))
                continue
            }
            if let i32 = value as? Int32 {
                out.append(i32)
                continue
            }
            if let n = value as? NSNumber {
                out.append(n.int32Value)
                continue
            }
            if let s = value as? String {
                if let wrapped = parseWrappedCode(s) {
                    out.append(wrapped)
                    continue
                }
                let trimmed = s.trimmingCharacters(in: .whitespacesAndNewlines)
                if let parsed = Int32(trimmed) {
                    out.append(parsed)
                    continue
                }
            }
            return nil
        }

        return out.isEmpty ? nil : out
    }

    private static func parseWrappedCode(_ token: String) -> Int32? {
        let pattern = #"^<\|s_(-?\d+)\|>$"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else {
            return nil
        }
        let range = NSRange(token.startIndex..<token.endIndex, in: token)
        guard let match = regex.firstMatch(in: token, options: [], range: range) else {
            return nil
        }
        guard match.numberOfRanges >= 2 else {
            return nil
        }
        guard let codeRange = Range(match.range(at: 1), in: token) else {
            return nil
        }
        return Int32(String(token[codeRange]))
    }

    private static func extractText(from object: [String: Any]) -> String? {
        var parts: [String] = []

        appendTextContent(object["text"], to: &parts)
        appendTextContent(object["output_text"], to: &parts)

        if let choices = object["choices"] as? [Any],
            let firstChoice = choices.first as? [String: Any] {
            appendTextContent(firstChoice["text"], to: &parts)
            if let message = firstChoice["message"] as? [String: Any] {
                appendTextContent(message["content"], to: &parts)
            }
        }

        guard !parts.isEmpty else {
            return nil
        }
        return parts.joined(separator: "\n")
    }

    private static func appendTextContent(_ value: Any?, to parts: inout [String]) {
        guard let value else {
            return
        }

        if let text = value as? String {
            if !text.isEmpty {
                parts.append(text)
            }
            return
        }

        if let array = value as? [Any] {
            for item in array {
                if let text = item as? String, !text.isEmpty {
                    parts.append(text)
                } else if let object = item as? [String: Any], let text = object["text"] as? String, !text.isEmpty {
                    parts.append(text)
                }
            }
            return
        }

        if let object = value as? [String: Any], let text = object["text"] as? String, !text.isEmpty {
            parts.append(text)
        }
    }

    private static func extractCodesFromText(_ text: String) -> [Int32] {
        let pattern = #"<\|s_(-?\d+)\|>"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else {
            return []
        }

        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        let matches = regex.matches(in: text, options: [], range: range)
        if matches.isEmpty {
            return []
        }

        var out: [Int32] = []
        out.reserveCapacity(matches.count)
        for match in matches {
            guard match.numberOfRanges >= 2 else {
                continue
            }
            guard let codeRange = Range(match.range(at: 1), in: text) else {
                continue
            }
            if let value = Int32(String(text[codeRange])) {
                out.append(value)
            }
        }
        return out
    }

    private static func extractErrorMessage(from jsonObject: Any?) -> String? {
        guard let object = jsonObject as? [String: Any] else {
            return nil
        }
        if let errorObject = object["error"] as? [String: Any],
            let message = errorObject["message"] as? String {
            return message
        }
        if let errorText = object["error"] as? String {
            return errorText
        }
        return nil
    }

    private static func makeErrorBuffer() -> [CChar] {
        Array(repeating: 0, count: 2048)
    }

    private static func message(from err: [CChar], fallback: String) -> String {
        let msg = String(cString: err)
        if msg.isEmpty {
            return fallback
        }
        return msg
    }
}
