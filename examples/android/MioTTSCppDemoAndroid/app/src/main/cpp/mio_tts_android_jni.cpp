#include <jni.h>
#include <android/log.h>

#include "llama.h"
#include "mio-tts-lib.h"
#include "ggml-backend.h"
#include "mio-tts-mobile-shared.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <vector>

namespace {

static constexpr const char * TAG = "MioTTS-Android";

using android_mio_engine = mio_tts_mobile::mobile_engine;

static std::mutex g_global_error_mutex;
static std::string g_global_error;

static void set_global_error(const std::string & msg) {
    std::lock_guard<std::mutex> lock(g_global_error_mutex);
    g_global_error = msg;
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s", msg.c_str());
}

static std::string get_global_error() {
    std::lock_guard<std::mutex> lock(g_global_error_mutex);
    return g_global_error;
}

static std::string jstring_to_std(JNIEnv * env, jstring value) {
    if (value == nullptr) {
        return "";
    }
    const char * ptr = env->GetStringUTFChars(value, nullptr);
    if (ptr == nullptr) {
        return "";
    }
    std::string out(ptr);
    env->ReleaseStringUTFChars(value, ptr);
    return out;
}

static jstring make_jstring(JNIEnv * env, const std::string & value) {
    return env->NewStringUTF(value.c_str());
}

static void set_engine_error(android_mio_engine * engine, const std::string & msg) {
    if (engine != nullptr) {
        engine->last_error = msg;
    }
    __android_log_print(ANDROID_LOG_ERROR, TAG, "%s", msg.c_str());
}

static jstring return_error(JNIEnv * env, android_mio_engine * engine, const std::string & msg) {
    set_engine_error(engine, msg);
    return make_jstring(env, msg);
}

static android_mio_engine * to_engine(jlong handle) {
    return reinterpret_cast<android_mio_engine *>(handle);
}

} // namespace

extern "C" JNIEXPORT void JNICALL
Java_com_example_miottscpp_NativeMioEngine_nativeInitBackends(
        JNIEnv * env,
        jobject,
        jstring nativeLibDir) {
    const std::string path = jstring_to_std(env, nativeLibDir);

    try {
        llama_backend_init();
        if (!path.empty()) {
            ggml_backend_load_all_from_path(path.c_str());
        } else {
            ggml_backend_load_all();
        }
        set_global_error("");
    } catch (const std::exception & e) {
        set_global_error(std::string("backend init failed: ") + e.what());
    } catch (...) {
        set_global_error("backend init failed (unknown error)");
    }
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_miottscpp_NativeMioEngine_nativeCreateEngine(
        JNIEnv * env,
        jobject,
        jstring jLlmModelPath,
        jstring jVocoderModelPath,
        jstring jWavlmModelPath,
        jint nGpuLayers,
        jint nCtx,
        jint nThreads,
        jboolean flashAttn) {
    (void) flashAttn;

    const std::string llm_model = jstring_to_std(env, jLlmModelPath);
    const std::string vocoder_model = jstring_to_std(env, jVocoderModelPath);
    const std::string wavlm_model = jstring_to_std(env, jWavlmModelPath);

    if (vocoder_model.empty()) {
        set_global_error("vocoder_model is required");
        return 0;
    }

    std::string check_err;
    if (!mio_tts_mobile::validate_readable_file(vocoder_model, check_err)) {
        set_global_error("vocoder_model is not readable: " + vocoder_model + " (" + check_err + ")");
        return 0;
    }
    if (!llm_model.empty() && !mio_tts_mobile::validate_readable_file(llm_model, check_err)) {
        set_global_error("llm_model is not readable: " + llm_model + " (" + check_err + ")");
        return 0;
    }
    if (!wavlm_model.empty() && !mio_tts_mobile::validate_readable_file(wavlm_model, check_err)) {
        set_global_error("wavlm_model is not readable: " + wavlm_model + " (" + check_err + ")");
        return 0;
    }

    std::unique_ptr<android_mio_engine> engine(new (std::nothrow) android_mio_engine());
    if (!engine) {
        set_global_error("out of memory");
        return 0;
    }

    engine->n_ctx = std::max<int32_t>((int32_t) nCtx, 256);
    engine->n_threads = (nThreads > 0) ? (int32_t) nThreads : 2;
    // Mobile app policy: keep flash-attn disabled for stability and audio quality.
    engine->flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    engine->llm_model_path = llm_model;
    engine->llm_n_gpu_layers = std::max<int32_t>(0, (int32_t) nGpuLayers);
    engine->llm_unload_after_generation = false;
    engine->vocoder_model_path = vocoder_model;
    engine->wavlm_model_path = wavlm_model;
    engine->mio_has_wavlm = false;

    // Load WavLM together with the decoder from the start so that
    // reference-generation does NOT need to destroy and recreate the
    // entire context.  The old approach (mio_has_wavlm=false) caused
    // ~1.5 GB of alloc/free churn every reference cycle, leading to
    // severe heap fragmentation and monotonic RSS growth on mobile.
    const char * mio_wavlm_ptr =
            wavlm_model.empty() ? nullptr : wavlm_model.c_str();

    char c_err[1024] = {0};
    engine->mio = mio_tts_init_from_file(
            vocoder_model.c_str(),
            mio_wavlm_ptr,
            c_err,
            sizeof(c_err));

    if (engine->mio == nullptr) {
        set_global_error(std::string("mio_tts_init_from_file failed: ") + c_err);
        return 0;
    }

    engine->mio_has_wavlm = (mio_wavlm_ptr != nullptr);

    // Mobile app policy: force MioCodec/WavLM path to CPU for runtime stability.
    if (!mio_tts_context_set_backend_device(engine->mio, "CPU", c_err, sizeof(c_err))) {
        set_global_error(std::string("mio_tts_context_set_backend_device(CPU) failed: ") + c_err);
        return 0;
    }

    set_global_error("");
    return reinterpret_cast<jlong>(engine.release());
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_miottscpp_NativeMioEngine_nativeDestroyEngine(
        JNIEnv *,
        jobject,
        jlong handle) {
    delete to_engine(handle);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_miottscpp_NativeMioEngine_nativeSetGenerationParams(
        JNIEnv * env,
        jobject,
        jlong handle,
        jint nCtx,
        jint topK,
        jfloat topP,
        jfloat temp) {
    android_mio_engine * engine = to_engine(handle);
    std::string op_err;
    if (!mio_tts_mobile::set_generation_params(engine, nCtx, topK, topP, temp, op_err)) {
        return return_error(env, engine, op_err);
    }
    return nullptr;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_miottscpp_NativeMioEngine_nativeUnloadLlmRuntime(
        JNIEnv * env,
        jobject,
        jlong handle) {
    android_mio_engine * engine = to_engine(handle);
    std::string op_err;
    if (!mio_tts_mobile::unload_llm_runtime(engine, op_err)) {
        return return_error(env, engine, op_err);
    }
    return nullptr;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_miottscpp_NativeMioEngine_nativeAddReferenceFromGguf(
        JNIEnv * env,
        jobject,
        jlong handle,
        jstring jReferenceKey,
        jstring jEmbeddingPath) {
    android_mio_engine * engine = to_engine(handle);

    const std::string reference_key = jstring_to_std(env, jReferenceKey);
    const std::string embedding_path = jstring_to_std(env, jEmbeddingPath);

    std::string op_err;
    if (!mio_tts_mobile::add_reference_from_gguf(
                engine,
                reference_key.c_str(),
                embedding_path.c_str(),
                op_err)) {
        return return_error(env, engine, op_err);
    }

    return nullptr;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_miottscpp_NativeMioEngine_nativeRemoveReference(
        JNIEnv * env,
        jobject,
        jlong handle,
        jstring jReferenceKey) {
    android_mio_engine * engine = to_engine(handle);

    const std::string reference_key = jstring_to_std(env, jReferenceKey);

    std::string op_err;
    if (!mio_tts_mobile::remove_reference(
                engine,
                reference_key.c_str(),
                op_err)) {
        return return_error(env, engine, op_err);
    }

    return nullptr;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_miottscpp_NativeMioEngine_nativeRegisterDefaultReferences(
        JNIEnv * env,
        jobject,
        jlong handle,
        jstring jModelDirPath,
        jstring jFallbackEmbeddingPath) {
    android_mio_engine * engine = to_engine(handle);
    if (engine == nullptr) {
        return nullptr;
    }

    const std::string model_dir = jstring_to_std(env, jModelDirPath);
    const std::string fallback_embedding_path = jstring_to_std(env, jFallbackEmbeddingPath);

    std::string preferred_key;
    std::string op_err;
    if (!mio_tts_mobile::register_default_references(
                engine,
                model_dir.empty() ? nullptr : model_dir.c_str(),
                fallback_embedding_path.empty() ? nullptr : fallback_embedding_path.c_str(),
                preferred_key,
                op_err)) {
        set_engine_error(engine, op_err);
        return nullptr;
    }

    return make_jstring(env, preferred_key);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_miottscpp_NativeMioEngine_nativeCreateReferenceFromAudio(
        JNIEnv * env,
        jobject,
        jlong handle,
        jstring jReferenceKey,
        jstring jAudioPath,
        jfloat maxReferenceSeconds,
        jstring jSaveEmbeddingPath) {
    android_mio_engine * engine = to_engine(handle);

    const std::string reference_key = jstring_to_std(env, jReferenceKey);
    const std::string audio_path = jstring_to_std(env, jAudioPath);
    const std::string save_embedding_path = jstring_to_std(env, jSaveEmbeddingPath);

    std::string op_err;
    if (!mio_tts_mobile::create_reference_from_audio(
                engine,
                reference_key.c_str(),
                audio_path.c_str(),
                maxReferenceSeconds,
                save_embedding_path.empty() ? nullptr : save_embedding_path.c_str(),
                op_err)) {
        return return_error(env, engine, op_err);
    }

    return nullptr;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_miottscpp_NativeMioEngine_nativeListReferencesJson(
        JNIEnv * env,
        jobject,
        jlong handle) {
    android_mio_engine * engine = to_engine(handle);

    std::string json;
    std::string op_err;
    if (!mio_tts_mobile::list_references_json(engine, json, op_err)) {
        if (engine != nullptr) {
            set_engine_error(engine, op_err);
        }
        return nullptr;
    }

    return make_jstring(env, json);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_miottscpp_NativeMioEngine_nativeSynthesizeToWav(
        JNIEnv * env,
        jobject,
        jlong handle,
        jstring jText,
        jstring jReferenceKey,
        jint nPredict,
        jstring jOutputWavPath) {
    android_mio_engine * engine = to_engine(handle);

    const std::string text = jstring_to_std(env, jText);
    const std::string reference_key = jstring_to_std(env, jReferenceKey);
    const std::string output_wav_path = jstring_to_std(env, jOutputWavPath);

    std::string op_err;
    if (!mio_tts_mobile::synthesize_text_to_wav(
                engine,
                text.c_str(),
                reference_key.c_str(),
                (int32_t) nPredict,
                output_wav_path.c_str(),
                true,
                op_err)) {
        return return_error(env, engine, op_err);
    }

    return nullptr;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_miottscpp_NativeMioEngine_nativeSynthesizeCodesToWav(
        JNIEnv * env,
        jobject,
        jlong handle,
        jintArray jCodes,
        jstring jReferenceKey,
        jstring jOutputWavPath) {
    android_mio_engine * engine = to_engine(handle);

    const std::string reference_key = jstring_to_std(env, jReferenceKey);
    const std::string output_wav_path = jstring_to_std(env, jOutputWavPath);

    if (jCodes == nullptr) {
        return return_error(env, engine, "codes are required");
    }

    const jsize n = env->GetArrayLength(jCodes);
    if (n <= 0) {
        return return_error(env, engine, "codes are empty");
    }

    std::vector<int32_t> codes((size_t) n);
    env->GetIntArrayRegion(jCodes, 0, n, reinterpret_cast<jint *>(codes.data()));

    std::string op_err;
    if (!mio_tts_mobile::synthesize_codes_to_wav(
                engine,
                codes.data(),
                codes.size(),
                reference_key.c_str(),
                output_wav_path.c_str(),
                op_err)) {
        return return_error(env, engine, op_err);
    }

    return nullptr;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_miottscpp_NativeMioEngine_nativeGetLastError(
        JNIEnv * env,
        jobject,
        jlong handle) {
    android_mio_engine * engine = to_engine(handle);
    if (engine == nullptr) {
        return make_jstring(env, "engine is null");
    }

    std::lock_guard<std::mutex> lock(engine->mutex);
    return make_jstring(env, engine->last_error);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_miottscpp_NativeMioEngine_nativeGetGlobalError(
        JNIEnv * env,
        jobject) {
    return make_jstring(env, get_global_error());
}
