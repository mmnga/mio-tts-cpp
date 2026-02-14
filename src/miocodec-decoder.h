#pragma once

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-graph-copy-host.h"

#include <string>
#include <unordered_map>
#include <vector>

struct miocodec_decoder_params {
    int model_type        = 0; // 0 = wave(stft), 1 = mel
    int sample_rate       = 24000;
    int n_fft             = 1920;
    int hop_length        = 480;
    int n_mels            = 0;
    int samples_per_token = 960;

    int prenet_layers  = 6;
    int prenet_dim     = 768;
    int prenet_heads   = 12;
    int prenet_ff      = 2048;
    int prenet_window  = 65;

    int decoder_layers = 8;
    int decoder_dim    = 512;
    int decoder_heads  = 8;
    int decoder_ff     = 1536;
    int decoder_window = 65;
    int decoder_adanorm_dim = 128;

    int resnet_blocks      = 2;
    int resnet_groups      = 32;
    float rope_theta       = 10000.0f;
    float norm_eps         = 1e-5f;
    float group_norm_eps   = 1e-6f;

    bool dynamic_global = false;
    bool has_vocoder = false;
    int mel_postnet_layers = 0;
    int mel_postnet_kernel_size = 0;

    int global_encoder_input_channels = 768;
    int global_encoder_output_channels = 128;
    int global_encoder_dim = 384;
    int global_encoder_intermediate_dim = 1152;
    int global_encoder_layers = 4;
};

class miocodec_decoder {
public:
    miocodec_decoder();
    ~miocodec_decoder();

    miocodec_decoder(const miocodec_decoder &) = delete;
    miocodec_decoder & operator=(const miocodec_decoder &) = delete;

    bool load(const std::string & path, std::string & err);
    bool is_loaded() const;
    bool has_dynamic_global() const;
    bool has_global_encoder() const;
    bool output_is_mel() const;
    bool has_vocoder() const;
    bool set_backend_device(const std::string & backend_name, std::string & err);

    const miocodec_decoder_params & params() const;
    size_t estimate_decode_workspace_bytes(int n_tokens) const;
    void reserve_decode_workspace(int n_tokens) const;
    size_t estimate_embedding_workspace_bytes(int n_frames) const;
    void reserve_embedding_workspace(int n_frames) const;

    bool decode_tokens(
            const std::vector<int32_t> & tokens,
            int n_threads,
            llama_flash_attn_type flash_attn_type,
            std::vector<float> & spec_out,
            int & n_frames,
            std::string & err) const;
    bool decode_tokens(
            const std::vector<int32_t> & tokens,
            const std::vector<float> & global_embedding,
            int n_threads,
            llama_flash_attn_type flash_attn_type,
            std::vector<float> & spec_out,
            int & n_frames,
            std::string & err) const;
    bool decode_mel_to_audio(
            const std::vector<float> & mel,
            int n_frames,
            int n_threads,
            std::vector<float> & audio_out,
            std::string & err) const;

    // ssl_features: [global_encoder_input_channels, n_frames]
    bool encode_global_embedding_from_ssl(
            const std::vector<float> & ssl_features,
            int n_frames,
            int n_threads,
            std::vector<float> & global_embedding_out,
            std::string & err) const;

private:
    struct gguf_context * ctx_gguf_    = nullptr;
    struct ggml_context * ctx_weights_ = nullptr;

    struct transformer_block {
        ggml_tensor * attn_norm_w = nullptr;
        ggml_tensor * attn_norm_b = nullptr;
        ggml_tensor * attn_cond_w = nullptr;
        ggml_tensor * attn_cond_b = nullptr;
        ggml_tensor * wq = nullptr;
        ggml_tensor * wk = nullptr;
        ggml_tensor * wv = nullptr;
        ggml_tensor * wo = nullptr;
        ggml_tensor * ffn_norm_w = nullptr;
        ggml_tensor * ffn_norm_b = nullptr;
        ggml_tensor * ffn_cond_w = nullptr;
        ggml_tensor * ffn_cond_b = nullptr;
        ggml_tensor * w1 = nullptr;
        ggml_tensor * w2 = nullptr;
        ggml_tensor * w3 = nullptr;
    };

    struct resnet_block {
        ggml_tensor * norm1_w = nullptr;
        ggml_tensor * norm1_b = nullptr;
        ggml_tensor * conv1_w = nullptr;
        ggml_tensor * conv1_b = nullptr;
        ggml_tensor * norm2_w = nullptr;
        ggml_tensor * norm2_b = nullptr;
        ggml_tensor * conv2_w = nullptr;
        ggml_tensor * conv2_b = nullptr;
    };

    struct wave_upsampler_stage {
        ggml_tensor * up_w = nullptr;
        ggml_tensor * up_b = nullptr;
        ggml_tensor * snake_alpha = nullptr;
        ggml_tensor * snake_beta = nullptr;
        resnet_block resblk = {};
    };

    miocodec_decoder_params hp_ = {};

    ggml_tensor * tok_embd_ = nullptr;

    std::vector<transformer_block> prenet_;
    ggml_tensor * prenet_norm_w_ = nullptr;
    ggml_tensor * prenet_norm_b_ = nullptr;
    ggml_tensor * prenet_out_w_  = nullptr;
    ggml_tensor * prenet_out_b_  = nullptr;

    ggml_tensor * upsample_w_ = nullptr;
    ggml_tensor * upsample_b_ = nullptr;

    std::vector<resnet_block> prior_;
    std::vector<resnet_block> post_;

    std::vector<transformer_block> decoder_;
    ggml_tensor * decoder_norm_w_ = nullptr;
    ggml_tensor * decoder_norm_b_ = nullptr;
    ggml_tensor * decoder_norm_cond_w_ = nullptr;
    ggml_tensor * decoder_norm_cond_b_ = nullptr;

    ggml_tensor * istft_out_w_ = nullptr;
    ggml_tensor * istft_out_b_ = nullptr;

    std::vector<wave_upsampler_stage> wave_upsampler_;
    std::vector<int> wave_upsampler_factors_;
    std::vector<int> wave_upsampler_kernel_sizes_;
    int wave_upsampler_total_factor_ = 1;
    ggml_tensor * wave_upsampler_out_proj_w_ = nullptr;
    ggml_tensor * wave_upsampler_out_proj_b_ = nullptr;
    ggml_tensor * wave_upsampler_out_snake_alpha_ = nullptr;
    ggml_tensor * wave_upsampler_out_snake_beta_ = nullptr;

    struct mel_postnet_layer {
        ggml_tensor * conv_w = nullptr;
        ggml_tensor * conv_b = nullptr;
        ggml_tensor * norm_w = nullptr;
        ggml_tensor * norm_b = nullptr;
    };
    std::vector<mel_postnet_layer> mel_postnet_;

    struct vocoder_conv {
        ggml_tensor * weight = nullptr; // [k, in, out]
        ggml_tensor * bias   = nullptr; // [out]
    };
    struct vocoder_activation {
        ggml_tensor * alpha = nullptr;       // [C]
        ggml_tensor * beta = nullptr;        // [C]
        ggml_tensor * up_filter = nullptr;   // [K,1,1]
        ggml_tensor * down_filter = nullptr; // [K,1,1]
    };
    struct vocoder_resblock {
        vocoder_conv convs1[3];
        vocoder_conv convs2[3];
        vocoder_activation acts[6];
    };
    struct vocoder_upsample {
        vocoder_conv after;
        vocoder_conv noise;
    };

    int vocoder_num_upsamples_ = 0;
    int vocoder_num_kernels_ = 0;
    std::vector<int> vocoder_upsample_rates_;
    vocoder_conv vocoder_conv_pre_;
    vocoder_conv vocoder_conv_post_;
    std::vector<vocoder_upsample> vocoder_ups_;
    std::vector<vocoder_resblock> vocoder_resblocks_;
    vocoder_activation vocoder_activation_post_;

    struct global_encoder_block {
        ggml_tensor * dwconv_w = nullptr;
        ggml_tensor * dwconv_b = nullptr;
        ggml_tensor * norm_w = nullptr;
        ggml_tensor * norm_b = nullptr;
        ggml_tensor * pw1_w = nullptr;
        ggml_tensor * pw1_b = nullptr;
        ggml_tensor * pw2_w = nullptr;
        ggml_tensor * pw2_b = nullptr;
        ggml_tensor * gamma = nullptr;
    };

    ggml_tensor * ge_embed_w_ = nullptr;
    ggml_tensor * ge_embed_b_ = nullptr;
    ggml_tensor * ge_norm_w_ = nullptr;
    ggml_tensor * ge_norm_b_ = nullptr;
    ggml_tensor * ge_final_norm_w_ = nullptr;
    ggml_tensor * ge_final_norm_b_ = nullptr;
    std::vector<global_encoder_block> ge_blocks_;

    ggml_tensor * ge_pool_attn0_w_ = nullptr;
    ggml_tensor * ge_pool_attn0_b_ = nullptr;
    ggml_tensor * ge_pool_attn2_w_ = nullptr;
    ggml_tensor * ge_pool_attn2_b_ = nullptr;
    ggml_tensor * ge_pool_proj_w_ = nullptr;
    ggml_tensor * ge_pool_proj_b_ = nullptr;
    ggml_tensor * ge_pool_norm_w_ = nullptr;
    ggml_tensor * ge_pool_norm_b_ = nullptr;

    mutable ggml_backend_t backend_gpu_ = nullptr;
    std::string backend_device_name_;

    struct gpu_graph_cache_entry {
        mio_ggml_graph_copy copy;
        int64_t n0 = 0;
        int64_t n1 = 0;
    };

    mutable std::unordered_map<uint64_t, gpu_graph_cache_entry> cache_decode_;
    mutable std::vector<uint64_t> cache_decode_lru_;
    mutable std::unordered_map<uint64_t, gpu_graph_cache_entry> cache_embed_;
    mutable std::vector<uint64_t> cache_embed_lru_;
    static constexpr size_t k_gpu_graph_cache_limit = 1;

    void cache_trim(
            std::unordered_map<uint64_t, gpu_graph_cache_entry> & cache,
            std::vector<uint64_t> & lru,
            uint64_t keep_key) const;
    void cache_clear(
            std::unordered_map<uint64_t, gpu_graph_cache_entry> & cache,
            std::vector<uint64_t> & lru) const;

    bool get_u32_kv(const char * key, uint32_t & out) const;
    bool get_f32_kv(const char * key, float & out) const;
};
