#include "llama.h"
#include "mio-tts-lib.h"

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

static int resolve_threads(int n_threads) {
    return n_threads > 0 ? n_threads : 2;
}

static bool parse_code_token(const std::string & raw, int32_t & out) {
    std::string token = raw;
    while (!token.empty() && std::ispunct((unsigned char) token.front()) && token.front() != '<' && token.front() != '-') {
        token.erase(token.begin());
    }
    while (!token.empty() && std::ispunct((unsigned char) token.back()) && token.back() != '>' && !std::isdigit((unsigned char) token.back())) {
        token.pop_back();
    }
    if (token.empty()) {
        return false;
    }

    if (token.rfind("<|s_", 0) == 0 && token.size() > 6 && token.substr(token.size() - 2) == "|>") {
        token = token.substr(4, token.size() - 6);
    }

    try {
        size_t used = 0;
        long long v = std::stoll(token, &used, 10);
        if (used != token.size()) {
            return false;
        }
        if (v < 0 || v > 12799) {
            return false;
        }
        out = (int32_t) v;
        return true;
    } catch (...) {
        return false;
    }
}

static bool parse_codes(const std::string & input, std::vector<int32_t> & out, std::string & err) {
    out.clear();
    std::string token;
    token.reserve(16);

    auto flush = [&]() -> bool {
        if (token.empty()) {
            return true;
        }
        int32_t code = -1;
        if (!parse_code_token(token, code)) {
            err = "invalid code token: " + token;
            return false;
        }
        out.push_back(code);
        token.clear();
        return true;
    };

    for (char ch : input) {
        if (std::isspace((unsigned char) ch) || ch == ',') {
            if (!flush()) {
                return false;
            }
        } else {
            token.push_back(ch);
        }
    }
    if (!flush()) {
        return false;
    }
    if (out.empty()) {
        err = "codes are empty";
        return false;
    }
    return true;
}

static bool tokenize_text(
        const llama_vocab * vocab,
        const std::string & text,
        bool add_special,
        bool parse_special,
        std::vector<llama_token> & out_tokens,
        std::string & err) {
    const int n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), nullptr, 0, add_special, parse_special);
    if (n >= 0) {
        err = "unexpected tokenization size response";
        return false;
    }

    const int needed = -n;
    if (needed <= 0) {
        err = "tokenization produced empty sequence";
        return false;
    }

    out_tokens.resize((size_t) needed);
    const int got = llama_tokenize(
            vocab,
            text.c_str(),
            (int32_t) text.size(),
            out_tokens.data(),
            (int32_t) out_tokens.size(),
            add_special,
            parse_special);
    if (got < 0) {
        err = "tokenization failed";
        return false;
    }
    out_tokens.resize((size_t) got);
    return !out_tokens.empty();
}

static llama_sampler * make_sampler(int top_k, float top_p, float temp) {
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(64, 1.0f, 0.0f, 0.0f));

    if (top_k > 0) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
    }
    if (top_p > 0.0f && top_p < 1.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
    }
    if (temp <= 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(0));
    }

    return smpl;
}

} // namespace

class MioTTSCppCore {
public:
    static constexpr int kMaxEmbeddingFloats = 8192;

    MioTTSCppCore() = default;
    ~MioTTSCppCore() {
        cleanup();
    }

    bool init(
            const std::string & vocoder_model_path,
            const std::string & llm_model_path,
            int n_threads,
            int n_ctx) {
        cleanup();
        last_error_.clear();
        audio_.clear();
        last_codes_.clear();
        embedding_.clear();
        sample_rate_ = 0;
        n_threads_ = resolve_threads(n_threads);
        n_ctx_ = std::max(256, n_ctx);
        vocoder_model_path_ = vocoder_model_path;

        char err[1024] = {0};
        mio_ = mio_tts_init_from_file(vocoder_model_path.c_str(), nullptr, err, sizeof(err));
        if (mio_ == nullptr) {
            last_error_ = err[0] ? err : "failed to init mio";
            return false;
        }

        if (!mio_tts_context_set_backend_device(mio_, "CPU", err, sizeof(err))) {
            last_error_ = err[0] ? err : "failed to set CPU backend";
            cleanup();
            return false;
        }

        if (!llm_model_path.empty()) {
            if (!llama_inited_) {
                llama_backend_init();
                llama_inited_ = true;
            }

            llama_model_params mparams = llama_model_default_params();
            mparams.n_gpu_layers = 0; // LLM always on CPU in WASM (WebGPU lacks SSM op support)
            mparams.use_mmap = false;
            mparams.use_mlock = false;
            llm_model_ = llama_model_load_from_file(llm_model_path.c_str(), mparams);
            if (llm_model_ == nullptr) {
                last_error_ = "failed to load llm model";
                cleanup();
                return false;
            }

            const llama_vocab * vocab = llama_model_get_vocab(llm_model_);
            vocab_map_ = mio_tts_vocab_map_init(vocab, err, sizeof(err));
            if (vocab_map_ == nullptr) {
                last_error_ = err[0] ? err : "failed to init vocab map";
                cleanup();
                return false;
            }
        }

        return true;
    }

    bool hasLLM() const {
        return llm_model_ != nullptr && vocab_map_ != nullptr;
    }

    bool loadEmbedding(const std::string & embedding_path) {
        last_error_.clear();
        if (mio_ == nullptr) {
            last_error_ = "engine is not initialized";
            return false;
        }

        float * emb = nullptr;
        size_t n_emb = 0;
        char err[512] = {0};
        if (!mio_tts_embedding_load_gguf(embedding_path.c_str(), &emb, &n_emb, err, sizeof(err))) {
            last_error_ = err[0] ? err : "failed to load embedding";
            return false;
        }
        embedding_.assign(emb, emb + n_emb);
        mio_tts_embedding_free(emb);
        return true;
    }

    bool setEmbeddingFromF32(const emscripten::val & embedding_f32) {
        last_error_.clear();
        if (embedding_f32.isNull() || embedding_f32.isUndefined()) {
            last_error_ = "embedding array is null";
            return false;
        }

        const int n = embedding_f32["length"].as<int>();
        if (n <= 0) {
            last_error_ = "embedding array is empty";
            return false;
        }
        if (n > kMaxEmbeddingFloats) {
            last_error_ = "embedding array is too large";
            return false;
        }

        std::vector<float> tmp((size_t) n);
        emscripten::val dst = emscripten::val(emscripten::typed_memory_view(tmp.size(), tmp.data()));
        dst.call<void>("set", embedding_f32);
        embedding_.swap(tmp);
        return true;
    }

    emscripten::val embeddingF32() const {
        return emscripten::val(emscripten::typed_memory_view(embedding_.size(), embedding_.data()));
    }

    bool createReferenceFromAudio(
            const std::string & audio_path,
            const std::string & wavlm_model_path,
            float max_reference_seconds,
            int n_threads) {
        last_error_.clear();

        if (mio_ == nullptr) {
            last_error_ = "engine is not initialized";
            return false;
        }
        if (vocoder_model_path_.empty()) {
            last_error_ = "vocoder model path is empty";
            return false;
        }
        if (audio_path.empty()) {
            last_error_ = "audio path is empty";
            return false;
        }
        if (wavlm_model_path.empty()) {
            last_error_ = "wavlm model path is empty";
            return false;
        }

        char err[1024] = {0};
        mio_tts_context * ref_ctx = mio_tts_init_from_file(
                vocoder_model_path_.c_str(),
                wavlm_model_path.c_str(),
                err,
                sizeof(err));
        if (ref_ctx == nullptr) {
            last_error_ = err[0] ? err : "failed to init temporary reference context";
            return false;
        }

        auto free_ref_ctx = [&]() {
            if (ref_ctx != nullptr) {
                mio_tts_free(ref_ctx);
                ref_ctx = nullptr;
            }
        };

        if (!mio_tts_context_set_backend_device(ref_ctx, "CPU", err, sizeof(err))) {
            last_error_ = err[0] ? err : "failed to set reference backend";
            free_ref_ctx();
            return false;
        }

        mio_tts_params mp = mio_tts_default_params();
        mp.n_threads = resolve_threads(n_threads > 0 ? n_threads : n_threads_);
        mp.max_reference_seconds = max_reference_seconds > 0.0f ? max_reference_seconds : 20.0f;
        mp.wavlm_flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        mp.miocodec_flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

        float * emb = nullptr;
        size_t n_emb = 0;
        if (!mio_tts_reference_to_embedding(
                    ref_ctx,
                    audio_path.c_str(),
                    mp,
                    &emb,
                    &n_emb,
                    err,
                    sizeof(err))) {
            last_error_ = err[0] ? err : "failed to create reference embedding";
            free_ref_ctx();
            return false;
        }

        embedding_.assign(emb, emb + n_emb);
        mio_tts_embedding_free(emb);
        free_ref_ctx();
        return true;
    }

    bool synthesizeFromCodes(const std::string & codes_text, int n_threads) {
        std::vector<int32_t> codes;
        std::string parse_err;
        if (!parse_codes(codes_text, codes, parse_err)) {
            last_error_ = parse_err;
            return false;
        }
        last_codes_ = codes;
        return synthesizeCodes(codes, n_threads);
    }

    bool synthesizeFromText(
            const std::string & text,
            int n_predict,
            int top_k,
            float top_p,
            float temp,
            int n_threads,
            int n_ctx) {
        last_error_.clear();
        if (!hasLLM()) {
            last_error_ = "llm is not loaded";
            return false;
        }
        if (text.empty()) {
            last_error_ = "text is empty";
            return false;
        }

        std::vector<int32_t> codes;
        if (!generateCodesFromText(text, n_predict, top_k, top_p, temp, n_threads, n_ctx, codes)) {
            return false;
        }
        last_codes_ = codes;
        return synthesizeCodes(codes, n_threads);
    }

    emscripten::val audioF32() const {
        return emscripten::val(emscripten::typed_memory_view(audio_.size(), audio_.data()));
    }

    emscripten::val lastCodesI32() const {
        return emscripten::val(emscripten::typed_memory_view(last_codes_.size(), last_codes_.data()));
    }

    int sampleRate() const {
        return sample_rate_;
    }

    std::string lastError() const {
        return last_error_;
    }

private:
    bool generateCodesFromText(
            const std::string & text,
            int n_predict,
            int top_k,
            float top_p,
            float temp,
            int n_threads,
            int n_ctx,
            std::vector<int32_t> & out_codes) {
        const llama_vocab * vocab = llama_model_get_vocab(llm_model_);
        const std::string prompt_chat =
                std::string("<|im_start|>user\n") + text + "<|im_end|>\n<|im_start|>assistant\n";

        std::vector<llama_token> prompt_tokens;
        if (!tokenize_text(vocab, prompt_chat, false, true, prompt_tokens, last_error_)) {
            return false;
        }

        const int use_predict = std::max(1, n_predict);
        const int use_ctx = std::max(256, n_ctx > 0 ? n_ctx : n_ctx_);
        const int use_threads = resolve_threads(n_threads > 0 ? n_threads : n_threads_);

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = (uint32_t) std::max<int32_t>(
                use_ctx,
                (int32_t) prompt_tokens.size() + use_predict + 32);
        cparams.n_batch = (uint32_t) std::max<int32_t>((int32_t) prompt_tokens.size(), 512);
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cparams.n_threads = use_threads;
        cparams.n_threads_batch = use_threads;

        llama_context * ctx = llama_init_from_model(llm_model_, cparams);
        if (ctx == nullptr) {
            last_error_ = "failed to create llm context";
            return false;
        }

        llama_sampler * sampler = make_sampler(top_k, top_p, temp);
        std::vector<llama_token> generated;
        generated.reserve((size_t) use_predict);

        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), (int32_t) prompt_tokens.size());
        if (llama_decode(ctx, batch) != 0) {
            llama_sampler_free(sampler);
            llama_free(ctx);
            last_error_ = "llama_decode failed on prompt";
            return false;
        }

        for (int32_t i = 0; i < use_predict; ++i) {
            llama_token tok = llama_sampler_sample(sampler, ctx, -1);
            llama_sampler_accept(sampler, tok);
            generated.push_back(tok);

            if (llama_vocab_is_eog(vocab, tok)) {
                break;
            }

            batch = llama_batch_get_one(&tok, 1);
            if (llama_decode(ctx, batch) != 0) {
                llama_sampler_free(sampler);
                llama_free(ctx);
                last_error_ = "llama_decode failed during generation";
                return false;
            }
        }

        llama_sampler_free(sampler);
        llama_free(ctx);

        int32_t * raw_codes = nullptr;
        size_t n_codes = 0;
        char err[512] = {0};
        if (!mio_tts_tokens_to_codes(
                    vocab_map_,
                    generated.data(),
                    generated.size(),
                    &raw_codes,
                    &n_codes,
                    err,
                    sizeof(err))) {
            last_error_ = err[0] ? err : "failed to convert tokens to codes";
            return false;
        }

        out_codes.assign(raw_codes, raw_codes + n_codes);
        mio_tts_codes_free(raw_codes);
        return true;
    }

    bool synthesizeCodes(const std::vector<int32_t> & codes, int n_threads) {
        last_error_.clear();
        audio_.clear();
        sample_rate_ = 0;

        if (mio_ == nullptr) {
            last_error_ = "engine is not initialized";
            return false;
        }
        if (embedding_.empty()) {
            last_error_ = "embedding is not loaded";
            return false;
        }
        if (codes.empty()) {
            last_error_ = "codes are empty";
            return false;
        }

        mio_tts_params p = mio_tts_default_params();
        p.n_threads = resolve_threads(n_threads > 0 ? n_threads : n_threads_);
        p.miocodec_flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        p.wavlm_flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

        float * out_audio = nullptr;
        size_t n_audio = 0;
        int32_t out_sr = 0;
        char err[512] = {0};
        if (!mio_tts_synthesize(
                    mio_,
                    codes.data(),
                    codes.size(),
                    embedding_.data(),
                    embedding_.size(),
                    p,
                    &out_audio,
                    &n_audio,
                    &out_sr,
                    err,
                    sizeof(err))) {
            last_error_ = err[0] ? err : "synthesis failed";
            return false;
        }

        audio_.assign(out_audio, out_audio + n_audio);
        sample_rate_ = out_sr;
        mio_tts_audio_free(out_audio);
        return true;
    }

    void cleanup() {
        if (vocab_map_ != nullptr) {
            mio_tts_vocab_map_free(vocab_map_);
            vocab_map_ = nullptr;
        }
        if (llm_model_ != nullptr) {
            llama_model_free(llm_model_);
            llm_model_ = nullptr;
        }
        if (mio_ != nullptr) {
            mio_tts_free(mio_);
            mio_ = nullptr;
        }
        if (llama_inited_) {
            llama_backend_free();
            llama_inited_ = false;
        }
    }

    mio_tts_context * mio_ = nullptr;
    llama_model * llm_model_ = nullptr;
    mio_tts_vocab_map * vocab_map_ = nullptr;
    bool llama_inited_ = false;

    std::vector<float> embedding_;
    std::vector<float> audio_;
    std::vector<int32_t> last_codes_;

    int sample_rate_ = 0;
    int n_threads_ = 2;
    int n_ctx_ = 700;
    std::string vocoder_model_path_;
    std::string last_error_;
};

EMSCRIPTEN_BINDINGS(miottscpp_core) {
    emscripten::class_<MioTTSCppCore>("MioTTSCppCore")
            .constructor<>()
            .function("init", &MioTTSCppCore::init)
            .function("hasLLM", &MioTTSCppCore::hasLLM)
            .function("loadEmbedding", &MioTTSCppCore::loadEmbedding)
            .function("setEmbeddingFromF32", &MioTTSCppCore::setEmbeddingFromF32)
            .function("embeddingF32", &MioTTSCppCore::embeddingF32)
            .function("createReferenceFromAudio", &MioTTSCppCore::createReferenceFromAudio)
            .function("synthesizeFromCodes", &MioTTSCppCore::synthesizeFromCodes)
            .function("synthesizeFromText", &MioTTSCppCore::synthesizeFromText)
            .function("audioF32", &MioTTSCppCore::audioF32)
            .function("lastCodesI32", &MioTTSCppCore::lastCodesI32)
            .function("sampleRate", &MioTTSCppCore::sampleRate)
            .function("lastError", &MioTTSCppCore::lastError);
}
