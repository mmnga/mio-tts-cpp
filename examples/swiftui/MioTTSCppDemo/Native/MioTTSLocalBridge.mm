#include "MioTTSLocalBridge.h"

#include <llama/llama.h>
#include "mio-tts-lib.h"
#include "../../../../src/mio-tts-mobile-shared.hpp"

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <string>

namespace {

using swift_mio_engine = mio_tts_mobile::mobile_engine;

static void set_err(char * err, size_t err_size, const std::string & msg) {
    if (err != nullptr && err_size > 0) {
        std::snprintf(err, err_size, "%s", msg.c_str());
    }
}

} // namespace

void * mio_swift_engine_create(
        const char * llm_model,
        const char * vocoder_model,
        const char * wavlm_model,
        int32_t n_gpu_layers,
        int32_t n_ctx,
        int32_t n_threads,
        bool flash_attn,
        char * err,
        size_t err_size) {
    set_err(err, err_size, "");

    if (llm_model == nullptr || llm_model[0] == '\0') {
        set_err(err, err_size, "llm_model is required");
        return nullptr;
    }
    if (vocoder_model == nullptr || vocoder_model[0] == '\0') {
        set_err(err, err_size, "vocoder_model is required");
        return nullptr;
    }
    if (wavlm_model != nullptr && wavlm_model[0] != '\0') {
        std::string check_err;
        if (!mio_tts_mobile::validate_readable_file(wavlm_model, check_err)) {
            set_err(err, err_size, std::string("wavlm_model is not readable: ") + wavlm_model + " (" + check_err + ")");
            return nullptr;
        }
    }

    {
        std::string check_err;
        if (!mio_tts_mobile::validate_readable_file(llm_model, check_err)) {
            set_err(err, err_size, std::string("llm_model is not readable: ") + llm_model + " (" + check_err + ")");
            return nullptr;
        }
        if (!mio_tts_mobile::validate_readable_file(vocoder_model, check_err)) {
            set_err(err, err_size, std::string("vocoder_model is not readable: ") + vocoder_model + " (" + check_err + ")");
            return nullptr;
        }
    }

#if defined(__APPLE__) && TARGET_OS_SIMULATOR
    // iOS Simulator's Metal debug device does not support residency sets.
    // Force CPU backend for MioCodec/WavLM paths to avoid runtime assert.
    setenv("GGML_METAL_NO_RESIDENCY", "1", 1);
#endif

    llama_backend_init();
#if defined(__APPLE__) && TARGET_OS_IPHONE
    // On iOS, backend plugins are not used; dynamic scan can throw in sandboxed environments.
#else
    try {
        ggml_backend_load_all();
    } catch (const std::exception & e) {
        set_err(err, err_size, std::string("ggml_backend_load_all failed: ") + e.what());
        return nullptr;
    } catch (...) {
        set_err(err, err_size, "ggml_backend_load_all failed with unknown exception");
        return nullptr;
    }
#endif

    std::unique_ptr<swift_mio_engine> engine(new (std::nothrow) swift_mio_engine());
    if (!engine) {
        set_err(err, err_size, "out of memory");
        return nullptr;
    }

    engine->n_ctx = std::max<int32_t>(n_ctx, 256);
    engine->n_threads = (n_threads > 0) ? n_threads : 2;
    engine->llm_model_path = llm_model != nullptr ? llm_model : "";
    engine->llm_n_gpu_layers = std::max<int32_t>(0, n_gpu_layers);
    engine->vocoder_model_path = vocoder_model != nullptr ? vocoder_model : "";
    engine->wavlm_model_path = (wavlm_model != nullptr && wavlm_model[0] != '\0') ? wavlm_model : "";
    engine->mio_has_wavlm = false;
#if defined(__APPLE__) && TARGET_OS_IPHONE
    engine->flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    engine->llm_unload_after_generation = false;
    const bool eager_load_llm = false;
#else
    engine->flash_attn_type = flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    engine->llm_unload_after_generation = false;
    const bool eager_load_llm = true;
#endif

    char c_err[1024] = {0};
    if (eager_load_llm) {
        llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = engine->llm_n_gpu_layers;

        engine->llm_model = llama_model_load_from_file(llm_model, mparams);
        if (engine->llm_model == nullptr) {
            set_err(err, err_size, std::string("failed to load LLM model: ") + llm_model);
            return nullptr;
        }

        const llama_vocab * vocab = llama_model_get_vocab(engine->llm_model);
        engine->vmap = mio_tts_vocab_map_init(vocab, c_err, sizeof(c_err));
        if (engine->vmap == nullptr) {
            set_err(err, err_size, std::string("mio_tts_vocab_map_init failed: ") + c_err);
            return nullptr;
        }
    }

    // Load WavLM together with the decoder from the start so that
    // reference-generation does NOT need to destroy and recreate the
    // entire context.  The old approach (mio_has_wavlm=false) caused
    // ~1.5 GB of alloc/free churn every reference cycle, leading to
    // severe heap fragmentation and monotonic RSS growth on mobile.
    const char * mio_wavlm_model =
            (!engine->wavlm_model_path.empty()) ? engine->wavlm_model_path.c_str() : nullptr;

    engine->mio = mio_tts_init_from_file(
            vocoder_model,
            mio_wavlm_model,
            c_err,
            sizeof(c_err));

    if (engine->mio == nullptr) {
        set_err(err, err_size, std::string("mio_tts_init_from_file failed: ") + c_err);
        return nullptr;
    }

    engine->mio_has_wavlm = (mio_wavlm_model != nullptr);

#if defined(__APPLE__) && TARGET_OS_IPHONE
    // MioCodec decode uses ggml_interpolate(BILINEAR) which maps to
    // GGML_OP_UPSCALE; ggml-metal only supports NEAREST mode.
    // Keep MioCodec/WavLM on CPU until Metal gains bilinear upscale.
    if (!mio_tts_context_set_backend_device(engine->mio, "CPU", c_err, sizeof(c_err))) {
        set_err(err, err_size, std::string("mio_tts_context_set_backend_device(CPU) failed: ") + c_err);
        return nullptr;
    }
#endif

    return engine.release();
}

void mio_swift_engine_destroy(void * handle) {
    delete reinterpret_cast<swift_mio_engine *>(handle);
}

bool mio_swift_engine_set_generation_params(
        void * handle,
        int32_t n_ctx,
        int32_t top_k,
        float top_p,
        float temp,
        char * err,
        size_t err_size) {
    set_err(err, err_size, "");

    std::string op_err;
    if (!mio_tts_mobile::set_generation_params(
                reinterpret_cast<swift_mio_engine *>(handle),
                n_ctx,
                top_k,
                top_p,
                temp,
                op_err)) {
        set_err(err, err_size, op_err);
        return false;
    }

    return true;
}

bool mio_swift_engine_unload_llm_runtime(
        void * handle,
        char * err,
        size_t err_size) {
    set_err(err, err_size, "");

    std::string op_err;
    if (!mio_tts_mobile::unload_llm_runtime(
                reinterpret_cast<swift_mio_engine *>(handle),
                op_err)) {
        set_err(err, err_size, op_err);
        return false;
    }

    return true;
}

bool mio_swift_engine_create_reference_from_audio(
        void * handle,
        const char * reference_key,
        const char * audio_path,
        float max_reference_seconds,
        const char * embedding_out_path,
        char * err,
        size_t err_size) {
    set_err(err, err_size, "");

    std::string op_err;
    if (!mio_tts_mobile::create_reference_from_audio(
                reinterpret_cast<swift_mio_engine *>(handle),
                reference_key,
                audio_path,
                max_reference_seconds,
                embedding_out_path,
                op_err)) {
        set_err(err, err_size, op_err);
        return false;
    }

    return true;
}

bool mio_swift_engine_add_reference_from_gguf(
        void * handle,
        const char * reference_key,
        const char * embedding_path,
        char * err,
        size_t err_size) {
    set_err(err, err_size, "");

    std::string op_err;
    if (!mio_tts_mobile::add_reference_from_gguf(
                reinterpret_cast<swift_mio_engine *>(handle),
                reference_key,
                embedding_path,
                op_err)) {
        set_err(err, err_size, op_err);
        return false;
    }

    return true;
}

bool mio_swift_engine_remove_reference(
        void * handle,
        const char * reference_key,
        char * err,
        size_t err_size) {
    set_err(err, err_size, "");

    std::string op_err;
    if (!mio_tts_mobile::remove_reference(
                reinterpret_cast<swift_mio_engine *>(handle),
                reference_key,
                op_err)) {
        set_err(err, err_size, op_err);
        return false;
    }

    return true;
}

bool mio_swift_engine_register_default_references(
        void * handle,
        const char * model_dir,
        const char * fallback_embedding_path,
        char ** preferred_key_out,
        char * err,
        size_t err_size) {
    set_err(err, err_size, "");

    if (preferred_key_out != nullptr) {
        *preferred_key_out = nullptr;
    }

    std::string preferred_key;
    std::string op_err;
    if (!mio_tts_mobile::register_default_references(
                reinterpret_cast<swift_mio_engine *>(handle),
                model_dir,
                fallback_embedding_path,
                preferred_key,
                op_err)) {
        set_err(err, err_size, op_err);
        return false;
    }

    if (preferred_key_out != nullptr && !preferred_key.empty()) {
        char * out = reinterpret_cast<char *>(std::malloc(preferred_key.size() + 1));
        if (out == nullptr) {
            set_err(err, err_size, "out of memory");
            return false;
        }
        std::memcpy(out, preferred_key.c_str(), preferred_key.size() + 1);
        *preferred_key_out = out;
    }

    return true;
}

bool mio_swift_engine_list_references_json(
        void * handle,
        char ** json_out,
        char * err,
        size_t err_size) {
    set_err(err, err_size, "");

    if (json_out == nullptr) {
        set_err(err, err_size, "json_out is null");
        return false;
    }
    *json_out = nullptr;

    std::string json;
    std::string op_err;
    if (!mio_tts_mobile::list_references_json(
                reinterpret_cast<swift_mio_engine *>(handle),
                json,
                op_err)) {
        set_err(err, err_size, op_err);
        return false;
    }

    char * out = reinterpret_cast<char *>(std::malloc(json.size() + 1));
    if (out == nullptr) {
        set_err(err, err_size, "out of memory");
        return false;
    }

    std::memcpy(out, json.c_str(), json.size() + 1);
    *json_out = out;
    return true;
}

bool mio_swift_engine_synthesize_to_wav(
        void * handle,
        const char * text,
        const char * reference_key,
        int32_t n_predict,
        const char * output_wav_path,
        char * err,
        size_t err_size) {
    set_err(err, err_size, "");

    std::string op_err;
    if (!mio_tts_mobile::synthesize_text_to_wav(
                reinterpret_cast<swift_mio_engine *>(handle),
                text,
                reference_key,
                n_predict,
                output_wav_path,
                true,
                op_err)) {
        set_err(err, err_size, op_err);
        return false;
    }

    return true;
}

bool mio_swift_engine_synthesize_codes_to_wav(
        void * handle,
        const int32_t * codes,
        size_t n_codes,
        const char * reference_key,
        const char * output_wav_path,
        char * err,
        size_t err_size) {
    set_err(err, err_size, "");

    std::string op_err;
    if (!mio_tts_mobile::synthesize_codes_to_wav(
                reinterpret_cast<swift_mio_engine *>(handle),
                codes,
                n_codes,
                reference_key,
                output_wav_path,
                op_err)) {
        set_err(err, err_size, op_err);
        return false;
    }

    return true;
}

void mio_swift_string_free(char * str) {
    std::free(str);
}
