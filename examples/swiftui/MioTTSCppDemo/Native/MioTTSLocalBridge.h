#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void * mio_swift_engine_create(
        const char * llm_model,
        const char * vocoder_model,
        const char * wavlm_model,
        int32_t n_gpu_layers,
        int32_t n_ctx,
        int32_t n_threads,
        bool flash_attn,
        char * err,
        size_t err_size);

void mio_swift_engine_destroy(void * handle);

bool mio_swift_engine_set_generation_params(
        void * handle,
        int32_t n_ctx,
        int32_t top_k,
        float top_p,
        float temp,
        char * err,
        size_t err_size);

bool mio_swift_engine_unload_llm_runtime(
        void * handle,
        char * err,
        size_t err_size);

bool mio_swift_engine_create_reference_from_audio(
        void * handle,
        const char * reference_key,
        const char * audio_path,
        float max_reference_seconds,
        const char * embedding_out_path,
        char * err,
        size_t err_size);

bool mio_swift_engine_add_reference_from_gguf(
        void * handle,
        const char * reference_key,
        const char * embedding_path,
        char * err,
        size_t err_size);

bool mio_swift_engine_remove_reference(
        void * handle,
        const char * reference_key,
        char * err,
        size_t err_size);

bool mio_swift_engine_register_default_references(
        void * handle,
        const char * model_dir,
        const char * fallback_embedding_path,
        char ** preferred_key_out,
        char * err,
        size_t err_size);

bool mio_swift_engine_list_references_json(
        void * handle,
        char ** json_out,
        char * err,
        size_t err_size);

bool mio_swift_engine_synthesize_to_wav(
        void * handle,
        const char * text,
        const char * reference_key,
        int32_t n_predict,
        const char * output_wav_path,
        char * err,
        size_t err_size);

bool mio_swift_engine_synthesize_codes_to_wav(
        void * handle,
        const int32_t * codes,
        size_t n_codes,
        const char * reference_key,
        const char * output_wav_path,
        char * err,
        size_t err_size);

void mio_swift_string_free(char * str);

#ifdef __cplusplus
}
#endif
