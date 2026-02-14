#pragma once

#if __has_include(<llama/llama.h>)
#include <llama/llama.h>
#else
#include "llama.h"
#endif

#if __has_include(<mio-tts-lib.h>)
#include <mio-tts-lib.h>
#else
#include "mio-tts-lib.h"
#endif

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cstdio>

#if defined(__APPLE__)
#include <TargetConditionals.h>
#include <malloc/malloc.h>
#endif

#if defined(__ANDROID__)
#include <malloc.h>
#endif

namespace mio_tts_mobile {

static constexpr uint64_t k_mobile_decode_workspace_limit_bytes = 2ull * 1024ull * 1024ull * 1024ull;

struct mobile_engine {
    llama_model * llm_model = nullptr;
    mio_tts_context * mio = nullptr;
    mio_tts_vocab_map * vmap = nullptr;

    int32_t n_ctx = 4096;
    int32_t n_threads = 2;
    int32_t top_k = 50;
    float top_p = 1.0f;
    float temp = 0.8f;
    float repeat_penalty = 1.0f;
    uint32_t seed = 0;
    llama_flash_attn_type flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    std::string llm_model_path;
    int32_t llm_n_gpu_layers = 0;
    bool llm_unload_after_generation = false;
    std::string vocoder_model_path;
    std::string wavlm_model_path;
    bool mio_has_wavlm = false;

    std::unordered_map<std::string, std::vector<float>> references;
    std::string last_error;
    std::mutex mutex;

    ~mobile_engine() {
        if (vmap != nullptr) {
            mio_tts_vocab_map_free(vmap);
            vmap = nullptr;
        }
        if (mio != nullptr) {
            mio_tts_free(mio);
            mio = nullptr;
        }
        if (llm_model != nullptr) {
            llama_model_free(llm_model);
            llm_model = nullptr;
        }
    }
};

inline int resolve_threads(int32_t n_threads) {
    if (n_threads > 0) {
        return (int) n_threads;
    }
    const unsigned hc = std::thread::hardware_concurrency();
    return (int) (hc > 0 ? hc : 1);
}

inline void release_memory_pressure() {
#if defined(__APPLE__)
    (void) malloc_zone_pressure_relief(nullptr, 0);
#elif defined(__ANDROID__)
    (void) mallopt(M_PURGE, 0);
#endif
}

inline void free_llm_runtime(mobile_engine * engine) {
    if (engine == nullptr) {
        return;
    }
    bool released = false;
    if (engine->vmap != nullptr) {
        mio_tts_vocab_map_free(engine->vmap);
        engine->vmap = nullptr;
        released = true;
    }
    if (engine->llm_model != nullptr) {
        llama_model_free(engine->llm_model);
        engine->llm_model = nullptr;
        released = true;
    }
    if (released) {
        release_memory_pressure();
    }
}

inline bool unload_llm_runtime(
        mobile_engine * engine,
        std::string & err) {
    if (engine == nullptr) {
        err = "engine is null";
        return false;
    }

    std::lock_guard<std::mutex> lock(engine->mutex);
    free_llm_runtime(engine);
    engine->last_error.clear();
    err.clear();
    return true;
}

inline bool ensure_llm_runtime(mobile_engine * engine, bool & loaded_now, std::string & err) {
    loaded_now = false;
    if (engine == nullptr) {
        err = "engine is null";
        return false;
    }

    if ((engine->llm_model != nullptr) != (engine->vmap != nullptr)) {
        free_llm_runtime(engine);
    }
    if (engine->llm_model != nullptr && engine->vmap != nullptr) {
        err.clear();
        return true;
    }
    if (engine->llm_model_path.empty()) {
        err = "LLM model is not loaded; use external API mode or load llm model";
        return false;
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = std::max<int32_t>(0, engine->llm_n_gpu_layers);

    engine->llm_model = llama_model_load_from_file(engine->llm_model_path.c_str(), mparams);
    if (engine->llm_model == nullptr) {
        err = std::string("failed to load LLM model: ") + engine->llm_model_path;
        return false;
    }

    char c_err[1024] = {0};
    const llama_vocab * vocab = llama_model_get_vocab(engine->llm_model);
    engine->vmap = mio_tts_vocab_map_init(vocab, c_err, sizeof(c_err));
    if (engine->vmap == nullptr) {
        llama_model_free(engine->llm_model);
        engine->llm_model = nullptr;
        err = std::string("mio_tts_vocab_map_init failed: ") + c_err;
        return false;
    }

    loaded_now = true;
    err.clear();
    return true;
}

inline bool validate_readable_file(const char * path, std::string & err) {
    if (path == nullptr || path[0] == '\0') {
        err = "empty path";
        return false;
    }

    errno = 0;
    std::FILE * fp = std::fopen(path, "rb");
    if (fp == nullptr) {
        err = std::strerror(errno);
        return false;
    }

    bool ok = true;
    if (std::fseek(fp, 0, SEEK_END) != 0) {
        ok = false;
        err = "failed to seek file";
    } else {
        const long file_size = std::ftell(fp);
        if (file_size <= 0) {
            ok = false;
            err = "file is empty";
        }
    }

    std::fclose(fp);
    return ok;
}

inline bool validate_readable_file(const std::string & path, std::string & err) {
    return validate_readable_file(path.c_str(), err);
}

inline bool tokenize_text(
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

inline llama_sampler * make_sampler(const mobile_engine & e) {
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(64, e.repeat_penalty, 0.0f, 0.0f));

    if (e.top_k > 0) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(e.top_k));
    }
    if (e.top_p > 0.0f && e.top_p < 1.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(e.top_p, 1));
    }

    if (e.temp <= 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(e.temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(e.seed));
    }

    return smpl;
}

inline bool generate_audio_tokens(
        const mobile_engine & e,
        const std::string & text,
        int32_t n_predict,
        bool low_memory_batch,
        std::vector<llama_token> & generated,
        std::string & err) {
    if (e.llm_model == nullptr || e.vmap == nullptr) {
        err = "LLM model is not initialized";
        return false;
    }

    const llama_vocab * vocab = llama_model_get_vocab(e.llm_model);

    const std::string prompt_chat =
        std::string("<|im_start|>user\n") + text + "<|im_end|>\n<|im_start|>assistant\n";

    std::vector<llama_token> prompt_tokens;
    if (!tokenize_text(vocab, prompt_chat, false, true, prompt_tokens, err)) {
        return false;
    }

    const int32_t n_predict_safe = std::max<int32_t>(1, n_predict);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = (uint32_t) std::max<int32_t>(e.n_ctx, (int32_t) prompt_tokens.size() + n_predict_safe + 32);

    if (low_memory_batch) {
        const int32_t mobile_batch = std::max<int32_t>(32, std::min<int32_t>(128, (int32_t) cparams.n_ctx));
        cparams.n_batch = (uint32_t) mobile_batch;
        cparams.n_ubatch = cparams.n_batch;
    } else {
        cparams.n_batch = (uint32_t) std::max<int32_t>((int32_t) prompt_tokens.size(), 512);
    }

    cparams.flash_attn_type = e.flash_attn_type;
    cparams.n_threads = resolve_threads(e.n_threads);
    cparams.n_threads_batch = cparams.n_threads;

    llama_context * ctx = llama_init_from_model(e.llm_model, cparams);
    if (ctx == nullptr) {
        err = "failed to create llama context";
        return false;
    }

    llama_sampler * sampler = make_sampler(e);

    generated.clear();

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), (int32_t) prompt_tokens.size());
    if (llama_decode(ctx, batch) != 0) {
        llama_sampler_free(sampler);
        llama_free(ctx);
        release_memory_pressure();
        err = "llama_decode failed on prompt";
        return false;
    }

    for (int32_t i = 0; i < n_predict_safe; ++i) {
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
            release_memory_pressure();
            err = "llama_decode failed during generation";
            return false;
        }
    }

    llama_sampler_free(sampler);
    llama_free(ctx);
    release_memory_pressure();
    return true;
}

inline std::string json_escape(const std::string & in) {
    std::string out;
    out.reserve(in.size() + 8);

    for (unsigned char c : in) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned) c);
                    out += buf;
                } else {
                    out.push_back((char) c);
                }
        }
    }

    return out;
}

inline bool ends_with(const std::string & value, const std::string & suffix) {
    if (value.size() < suffix.size()) {
        return false;
    }
    return value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

inline std::string join_path(const std::string & base_dir, const std::string & name) {
    if (base_dir.empty()) {
        return name;
    }

    const char tail = base_dir.back();
    if (tail == '/' || tail == '\\') {
        return base_dir + name;
    }

    return base_dir + "/" + name;
}

inline std::string fallback_reference_key_from_path(const std::string & path) {
    if (path.empty()) {
        return "jp_female";
    }

    const size_t slash = path.find_last_of("/\\");
    std::string file_name = (slash == std::string::npos) ? path : path.substr(slash + 1);

    if (ends_with(file_name, ".emb.gguf")) {
        file_name.resize(file_name.size() - std::string(".emb.gguf").size());
    } else if (ends_with(file_name, ".gguf")) {
        file_name.resize(file_name.size() - std::string(".gguf").size());
    }

    return file_name.empty() ? "jp_female" : file_name;
}

inline std::string select_preferred_default_reference(const std::vector<std::string> & keys) {
    if (keys.empty()) {
        return "";
    }

    for (const auto & key : keys) {
        if (key == "jp_female") {
            return "jp_female";
        }
    }

    static const char * k_default_order[] = {"en_female", "en_male", "jp_female", "jp_male"};
    for (const char * candidate : k_default_order) {
        for (const auto & key : keys) {
            if (key == candidate) {
                return key;
            }
        }
    }

    return keys.front();
}

inline bool set_generation_params(
        mobile_engine * engine,
        int32_t n_ctx,
        int32_t top_k,
        float top_p,
        float temp,
        std::string & err) {
    if (engine == nullptr) {
        err = "engine is null";
        return false;
    }
    if (n_ctx < 256) {
        err = "n_ctx must be >= 256";
        return false;
    }
    if (top_k < 1) {
        err = "top_k must be >= 1";
        return false;
    }
    if (!(top_p > 0.0f && top_p <= 1.0f)) {
        err = "top_p must be > 0 and <= 1";
        return false;
    }
    if (temp < 0.0f) {
        err = "temp must be >= 0";
        return false;
    }

    std::lock_guard<std::mutex> lock(engine->mutex);
    engine->n_ctx = n_ctx;
    engine->top_k = top_k;
    engine->top_p = top_p;
    engine->temp = temp;
    engine->last_error.clear();
    err.clear();
    return true;
}

inline bool add_reference_from_gguf(
        mobile_engine * engine,
        const char * reference_key,
        const char * embedding_path,
        std::string & err) {
    if (engine == nullptr) {
        err = "engine is null";
        return false;
    }
    if (reference_key == nullptr || reference_key[0] == '\0') {
        err = "reference_key is required";
        return false;
    }
    if (embedding_path == nullptr || embedding_path[0] == '\0') {
        err = "embedding_path is required";
        return false;
    }

    std::lock_guard<std::mutex> lock(engine->mutex);

    float * embedding = nullptr;
    size_t n_embedding = 0;
    char c_err[1024] = {0};

    if (!mio_tts_embedding_load_gguf(embedding_path, &embedding, &n_embedding, c_err, sizeof(c_err))) {
        err = std::string("mio_tts_embedding_load_gguf failed: ") + c_err;
        engine->last_error = err;
        return false;
    }

    std::vector<float> copied(embedding, embedding + n_embedding);
    engine->references[reference_key] = std::move(copied);
    mio_tts_embedding_free(embedding);

    engine->last_error.clear();
    err.clear();
    return true;
}

inline bool remove_reference(
        mobile_engine * engine,
        const char * reference_key,
        std::string & err) {
    if (engine == nullptr) {
        err = "engine is null";
        return false;
    }
    if (reference_key == nullptr || reference_key[0] == '\0') {
        err = "reference_key is required";
        return false;
    }

    std::lock_guard<std::mutex> lock(engine->mutex);

    const auto it = engine->references.find(reference_key);
    if (it == engine->references.end()) {
        err = std::string("reference not found: ") + reference_key;
        engine->last_error = err;
        return false;
    }

    engine->references.erase(it);
    engine->last_error.clear();
    err.clear();
    return true;
}

inline bool create_reference_from_audio(
        mobile_engine * engine,
        const char * reference_key,
        const char * audio_path,
        float max_reference_seconds,
        const char * embedding_out_path,
        std::string & err) {
    if (engine == nullptr) {
        err = "engine is null";
        return false;
    }
    if (reference_key == nullptr || reference_key[0] == '\0') {
        err = "reference_key is required";
        return false;
    }
    if (audio_path == nullptr || audio_path[0] == '\0') {
        err = "audio_path is required";
        return false;
    }

    std::lock_guard<std::mutex> lock(engine->mutex);

    // Reference generation does not need LLM runtime.
    // Drop it proactively to reduce peak memory on mobile.
    free_llm_runtime(engine);

    mio_tts_context * reference_ctx = engine->mio;
    bool owns_reference_ctx = false;
#if (defined(__APPLE__) && TARGET_OS_IPHONE) || defined(__ANDROID__)
    const bool unload_primary_ctx_for_reference = !engine->mio_has_wavlm;
#else
    const bool unload_primary_ctx_for_reference = false;
#endif
    bool primary_ctx_unloaded = false;

    auto restore_primary_ctx_if_needed = [&](std::string & restore_err) -> bool {
        restore_err.clear();
        if (!primary_ctx_unloaded) {
            return true;
        }

        // Encourage allocator to return freed pages before allocating the
        // new synthesis context, reducing peak RSS on mobile.
        release_memory_pressure();

        char init_err[1024] = {0};
        mio_tts_context * restored = mio_tts_init_from_file(
                engine->vocoder_model_path.c_str(),
                nullptr,
                init_err,
                sizeof(init_err));
        if (restored == nullptr) {
            restore_err = std::string("failed to restore synthesis context: ") + init_err;
            return false;
        }

#if (defined(__APPLE__) && TARGET_OS_IPHONE) || defined(__ANDROID__)
        // MioCodec decode uses ggml_interpolate(BILINEAR) → GGML_OP_UPSCALE;
        // ggml-metal only supports NEAREST.  Keep on CPU.
        if (!mio_tts_context_set_backend_device(restored, "CPU", init_err, sizeof(init_err))) {
            mio_tts_free(restored);
            restore_err = std::string("failed to restore synthesis backend to CPU: ") + init_err;
            return false;
        }
#endif

        engine->mio = restored;
        primary_ctx_unloaded = false;
        return true;
    };

    if (!engine->mio_has_wavlm) {
        if (engine->vocoder_model_path.empty() || engine->wavlm_model_path.empty()) {
            err = "WavLM model is not configured for reference generation";
            engine->last_error = err;
            return false;
        }

        if (unload_primary_ctx_for_reference && engine->mio != nullptr) {
            mio_tts_free(engine->mio);
            engine->mio = nullptr;
            reference_ctx = nullptr;
            primary_ctx_unloaded = true;
            release_memory_pressure();
        }

        char init_err[1024] = {0};
        reference_ctx = mio_tts_init_from_file(
                engine->vocoder_model_path.c_str(),
                engine->wavlm_model_path.c_str(),
                init_err,
                sizeof(init_err));
        if (reference_ctx == nullptr) {
            err = std::string("failed to initialize reference context: ") + init_err;
            std::string restore_err;
            if (!restore_primary_ctx_if_needed(restore_err)) {
                err += " | " + restore_err;
            }
            engine->last_error = err;
            return false;
        }
        owns_reference_ctx = true;

        // Reference generation is infrequent; force CPU backend for this
        // temporary context to avoid long-lived GPU allocator/residency memory.
        if (!mio_tts_context_set_backend_device(reference_ctx, "CPU", init_err, sizeof(init_err))) {
            mio_tts_free(reference_ctx);
            release_memory_pressure();
            err = std::string("mio_tts_context_set_backend_device(CPU) failed: ") + init_err;
            std::string restore_err;
            if (!restore_primary_ctx_if_needed(restore_err)) {
                err += " | " + restore_err;
            }
            engine->last_error = err;
            return false;
        }
    }

    mio_tts_params mp = mio_tts_default_params();
    mp.n_threads = engine->n_threads;
    mp.max_reference_seconds = std::max<float>(0.1f, max_reference_seconds);
    mp.wavlm_flash_attn_type = engine->flash_attn_type;
    mp.miocodec_flash_attn_type = engine->flash_attn_type;

    float * embedding = nullptr;
    size_t n_embedding = 0;
    char c_err[1024] = {0};

    if (!mio_tts_reserve_reference_workspace(reference_ctx, mp.max_reference_seconds, c_err, sizeof(c_err))) {
        if (owns_reference_ctx) {
            mio_tts_free(reference_ctx);
            release_memory_pressure();
        }
        err = std::string("mio_tts_reserve_reference_workspace failed: ") + c_err;
        std::string restore_err;
        if (!restore_primary_ctx_if_needed(restore_err)) {
            err += " | " + restore_err;
        }
        engine->last_error = err;
        return false;
    }

    const bool ok = mio_tts_reference_to_embedding(
            reference_ctx,
            audio_path,
            mp,
            &embedding,
            &n_embedding,
            c_err,
            sizeof(c_err));

    if (owns_reference_ctx) {
        mio_tts_free(reference_ctx);
        release_memory_pressure();
    }

    std::string restore_err;
    const bool restored_ok = restore_primary_ctx_if_needed(restore_err);

    if (!ok) {
        err = std::string("mio_tts_reference_to_embedding failed: ") + c_err;
        if (!restored_ok) {
            err += " | " + restore_err;
        }
        engine->last_error = err;
        return false;
    }

    if (!restored_ok) {
        mio_tts_embedding_free(embedding);
        release_memory_pressure();
        err = restore_err;
        engine->last_error = err;
        return false;
    }

    // After reference generation, keep runtime in "synthesis-ready" state:
    // WavLM context is already released above, then proactively load LLM.
    if (!engine->llm_model_path.empty() && !engine->llm_unload_after_generation) {
        bool loaded_now = false;
        std::string llm_load_err;
        if (!ensure_llm_runtime(engine, loaded_now, llm_load_err)) {
            mio_tts_embedding_free(embedding);
            release_memory_pressure();
            err = std::string("failed to load LLM after reference generation: ") + llm_load_err;
            engine->last_error = err;
            return false;
        }
        (void) loaded_now;
    }

    if (embedding_out_path != nullptr && embedding_out_path[0] != '\0') {
        if (!mio_tts_embedding_save_gguf(embedding_out_path, embedding, n_embedding, c_err, sizeof(c_err))) {
            mio_tts_embedding_free(embedding);
            release_memory_pressure();
            err = std::string("mio_tts_embedding_save_gguf failed: ") + c_err;
            engine->last_error = err;
            return false;
        }
    }

    std::vector<float> copied(embedding, embedding + n_embedding);
    engine->references[reference_key] = std::move(copied);
    mio_tts_embedding_free(embedding);
    release_memory_pressure();

    engine->last_error.clear();
    err.clear();
    return true;
}

inline bool list_references_json(
        mobile_engine * engine,
        std::string & json_out,
        std::string & err) {
    if (engine == nullptr) {
        err = "engine is null";
        return false;
    }

    std::lock_guard<std::mutex> lock(engine->mutex);

    std::vector<std::pair<std::string, int32_t>> items;
    items.reserve(engine->references.size());

    for (const auto & kv : engine->references) {
        items.emplace_back(kv.first, (int32_t) kv.second.size());
    }

    std::sort(items.begin(), items.end(), [](const auto & a, const auto & b) {
        return a.first < b.first;
    });

    std::string json;
    json.push_back('[');
    for (size_t i = 0; i < items.size(); ++i) {
        if (i > 0) {
            json.push_back(',');
        }
        json += "{\"key\":\"";
        json += json_escape(items[i].first);
        json += "\",\"embedding_dim\":";
        json += std::to_string(items[i].second);
        json.push_back('}');
    }
    json.push_back(']');

    json_out = std::move(json);
    engine->last_error.clear();
    err.clear();
    return true;
}

inline bool synthesize_codes_to_wav(
        mobile_engine * engine,
        const int32_t * codes,
        size_t n_codes,
        const char * reference_key,
        const char * output_wav_path,
        std::string & err) {
    if (engine == nullptr) {
        err = "engine is null";
        return false;
    }
    if (codes == nullptr || n_codes == 0) {
        err = "codes are required";
        return false;
    }
    if (reference_key == nullptr || reference_key[0] == '\0') {
        err = "reference_key is required";
        return false;
    }
    if (output_wav_path == nullptr || output_wav_path[0] == '\0') {
        err = "output_wav_path is required";
        return false;
    }

    std::lock_guard<std::mutex> lock(engine->mutex);

    const auto it = engine->references.find(reference_key);
    if (it == engine->references.end()) {
        err = std::string("reference not found: ") + reference_key;
        engine->last_error = err;
        return false;
    }

    float * audio = nullptr;
    size_t n_audio = 0;
    int32_t sample_rate = 0;
    char c_err[1024] = {0};

    mio_tts_params sp = mio_tts_default_params();
    sp.n_threads = engine->n_threads;
    sp.wavlm_flash_attn_type = engine->flash_attn_type;
    sp.miocodec_flash_attn_type = engine->flash_attn_type;

    const int32_t reserve_codes = (n_codes > (size_t) std::numeric_limits<int32_t>::max())
            ? std::numeric_limits<int32_t>::max()
            : (int32_t) n_codes;

    uint64_t estimated_bytes = 0;
    if (mio_tts_estimate_workspace_bytes(engine->mio, reserve_codes, &estimated_bytes, c_err, sizeof(c_err))) {
        std::fprintf(
                stderr,
                "mio-mobile: synthesize_codes n_codes=%zu reserve_codes=%d est_workspace=%.2f GiB threads=%d\n",
                n_codes,
                reserve_codes,
                (double) estimated_bytes / (1024.0 * 1024.0 * 1024.0),
                engine->n_threads);
        if (estimated_bytes > k_mobile_decode_workspace_limit_bytes) {
            err = "estimated decode workspace is too large for mobile; reduce n_predict / codes";
            engine->last_error = err;
            return false;
        }
    }

    if (!mio_tts_reserve_workspace(engine->mio, reserve_codes, c_err, sizeof(c_err))) {
        release_memory_pressure();
        err = std::string("mio_tts_reserve_workspace failed: ") + c_err;
        engine->last_error = err;
        return false;
    }

    const bool synth_ok = mio_tts_synthesize(
            engine->mio,
            codes,
            n_codes,
            it->second.data(),
            it->second.size(),
            sp,
            &audio,
            &n_audio,
            &sample_rate,
            c_err,
            sizeof(c_err));

    if (!synth_ok) {
        release_memory_pressure();
        err = std::string("mio_tts_synthesize failed: ") + c_err;
        engine->last_error = err;
        return false;
    }

    if (!mio_tts_audio_save_wav16(output_wav_path, audio, n_audio, sample_rate, c_err, sizeof(c_err))) {
        mio_tts_audio_free(audio);
        release_memory_pressure();
        err = std::string("mio_tts_audio_save_wav16 failed: ") + c_err;
        engine->last_error = err;
        return false;
    }

    mio_tts_audio_free(audio);
    release_memory_pressure();

    engine->last_error.clear();
    err.clear();
    return true;
}

inline bool synthesize_text_to_wav(
        mobile_engine * engine,
        const char * text,
        const char * reference_key,
        int32_t n_predict,
        const char * output_wav_path,
        bool low_memory_batch,
        std::string & err) {
    if (engine == nullptr) {
        err = "engine is null";
        return false;
    }
    if (text == nullptr || text[0] == '\0') {
        err = "text is required";
        return false;
    }
    if (reference_key == nullptr || reference_key[0] == '\0') {
        err = "reference_key is required";
        return false;
    }
    if (output_wav_path == nullptr || output_wav_path[0] == '\0') {
        err = "output_wav_path is required";
        return false;
    }

    std::lock_guard<std::mutex> lock(engine->mutex);

    bool loaded_llm_now = false;
    std::string load_llm_err;
    if (!ensure_llm_runtime(engine, loaded_llm_now, load_llm_err)) {
        err = load_llm_err;
        engine->last_error = err;
        return false;
    }

    auto maybe_unload_llm = [&]() {
        if (engine->llm_unload_after_generation) {
            free_llm_runtime(engine);
        } else {
            (void) loaded_llm_now;
        }
    };

    const auto it = engine->references.find(reference_key);
    if (it == engine->references.end()) {
        maybe_unload_llm();
        err = std::string("reference not found: ") + reference_key;
        engine->last_error = err;
        return false;
    }

    std::vector<llama_token> generated;
    std::string gen_err;
    if (!generate_audio_tokens(*engine, text, n_predict, low_memory_batch, generated, gen_err)) {
        maybe_unload_llm();
        err = std::string("token generation failed: ") + gen_err;
        engine->last_error = err;
        return false;
    }

    int32_t * codes = nullptr;
    size_t n_codes = 0;
    char c_err[1024] = {0};

    if (!mio_tts_tokens_to_codes(engine->vmap, generated.data(), generated.size(), &codes, &n_codes, c_err, sizeof(c_err))) {
        maybe_unload_llm();
        release_memory_pressure();
        err = std::string("mio_tts_tokens_to_codes failed: ") + c_err;
        engine->last_error = err;
        return false;
    }

    // Keep footprint low on mobile by dropping local LLM weights before Mio synthesis.
    maybe_unload_llm();

    float * audio = nullptr;
    size_t n_audio = 0;
    int32_t sample_rate = 0;

    mio_tts_params sp = mio_tts_default_params();
    sp.n_threads = engine->n_threads;
    sp.wavlm_flash_attn_type = engine->flash_attn_type;
    sp.miocodec_flash_attn_type = engine->flash_attn_type;

    const int32_t reserve_codes = (n_codes > (size_t) std::numeric_limits<int32_t>::max())
            ? std::numeric_limits<int32_t>::max()
            : (int32_t) n_codes;

    uint64_t estimated_bytes = 0;
    if (mio_tts_estimate_workspace_bytes(engine->mio, reserve_codes, &estimated_bytes, c_err, sizeof(c_err))) {
        std::fprintf(
                stderr,
                "mio-mobile: synthesize_text n_predict=%d generated=%zu n_codes=%zu reserve_codes=%d est_workspace=%.2f GiB threads=%d\n",
                n_predict,
                generated.size(),
                n_codes,
                reserve_codes,
                (double) estimated_bytes / (1024.0 * 1024.0 * 1024.0),
                engine->n_threads);
        if (estimated_bytes > k_mobile_decode_workspace_limit_bytes) {
            mio_tts_codes_free(codes);
            release_memory_pressure();
            err = "estimated decode workspace is too large for mobile; reduce n_predict";
            engine->last_error = err;
            return false;
        }
    }

    if (!mio_tts_reserve_workspace(engine->mio, reserve_codes, c_err, sizeof(c_err))) {
        mio_tts_codes_free(codes);
        release_memory_pressure();
        err = std::string("mio_tts_reserve_workspace failed: ") + c_err;
        engine->last_error = err;
        return false;
    }

    const bool synth_ok = mio_tts_synthesize(
            engine->mio,
            codes,
            n_codes,
            it->second.data(),
            it->second.size(),
            sp,
            &audio,
            &n_audio,
            &sample_rate,
            c_err,
            sizeof(c_err));

    mio_tts_codes_free(codes);

    if (!synth_ok) {
        release_memory_pressure();
        err = std::string("mio_tts_synthesize failed: ") + c_err;
        engine->last_error = err;
        return false;
    }

    if (!mio_tts_audio_save_wav16(output_wav_path, audio, n_audio, sample_rate, c_err, sizeof(c_err))) {
        mio_tts_audio_free(audio);
        release_memory_pressure();
        err = std::string("mio_tts_audio_save_wav16 failed: ") + c_err;
        engine->last_error = err;
        return false;
    }

    mio_tts_audio_free(audio);
    release_memory_pressure();

    engine->last_error.clear();
    err.clear();
    return true;
}

inline bool register_default_references(
        mobile_engine * engine,
        const char * model_dir,
        const char * fallback_embedding_path,
        std::string & preferred_key_out,
        std::string & err) {
    preferred_key_out.clear();

    if (engine == nullptr) {
        err = "engine is null";
        return false;
    }

    std::vector<std::string> added_keys;
    static const char * k_default_keys[] = {"en_female", "en_male", "jp_female", "jp_male"};

    if (model_dir != nullptr && model_dir[0] != '\0') {
        const std::string model_dir_s(model_dir);
        for (const char * key : k_default_keys) {
            const std::string emb_path = join_path(model_dir_s, std::string(key) + ".emb.gguf");
            std::string check_err;
            if (!validate_readable_file(emb_path, check_err)) {
                continue;
            }

            std::string add_err;
            if (!add_reference_from_gguf(engine, key, emb_path.c_str(), add_err)) {
                err = add_err;
                return false;
            }

            added_keys.emplace_back(key);
        }
    }

    if (added_keys.empty() && fallback_embedding_path != nullptr && fallback_embedding_path[0] != '\0') {
        std::string check_err;
        if (validate_readable_file(fallback_embedding_path, check_err)) {
            const std::string fallback_key = fallback_reference_key_from_path(fallback_embedding_path);
            std::string add_err;
            if (!add_reference_from_gguf(engine, fallback_key.c_str(), fallback_embedding_path, add_err)) {
                err = add_err;
                return false;
            }
            added_keys.emplace_back(fallback_key);
        }
    }

    preferred_key_out = select_preferred_default_reference(added_keys);
    err.clear();
    return true;
}

} // namespace mio_tts_mobile
