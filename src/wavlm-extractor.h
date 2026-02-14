#pragma once

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-graph-copy-host.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct wavlm_extractor_params {
    int sample_rate = 16000;
    int n_layers = 2;
    int n_heads = 12;
    int head_dim = 64;
    int embed_dim = 768;
    int num_buckets = 320;
    int max_distance = 800;
    float norm_eps = 1e-5f;

    int conv_kernel[7] = {10, 3, 3, 3, 3, 2, 2};
    int conv_stride[7] = {5, 2, 2, 2, 2, 2, 2};
};

class wavlm_extractor {
public:
    wavlm_extractor();
    ~wavlm_extractor();

    wavlm_extractor(const wavlm_extractor &) = delete;
    wavlm_extractor & operator=(const wavlm_extractor &) = delete;

    bool load(const std::string & path, std::string & err);
    bool is_loaded() const;
    bool set_backend_device(const std::string & backend_name, std::string & err);

    const wavlm_extractor_params & params() const;
    int estimate_ssl_frames(int source_sample_rate, float max_reference_seconds) const;
    size_t estimate_workspace_bytes(int source_sample_rate, float max_reference_seconds) const;
    void reserve_workspace(int source_sample_rate, float max_reference_seconds) const;

    // Output is [embed_dim, n_frames] contiguous float data.
    bool extract_ssl_features(
            const std::string & audio_path,
            int source_sample_rate,
            float max_reference_seconds,
            int n_threads,
            llama_flash_attn_type flash_attn_type,
            std::vector<float> & ssl_features_out,
            int & n_frames,
            std::string & err) const;

private:
    struct gguf_context * ctx_gguf_    = nullptr;
    struct ggml_context * ctx_weights_ = nullptr;

    struct wavlm_layer {
        ggml_tensor * in_proj_w = nullptr;
        ggml_tensor * in_proj_b = nullptr;
        ggml_tensor * out_proj_w = nullptr;
        ggml_tensor * out_proj_b = nullptr;

        ggml_tensor * gru_w = nullptr;
        ggml_tensor * gru_b = nullptr;
        ggml_tensor * gru_const = nullptr;

        ggml_tensor * norm1_w = nullptr;
        ggml_tensor * norm1_b = nullptr;
        ggml_tensor * ffn_w1 = nullptr;
        ggml_tensor * ffn_b1 = nullptr;
        ggml_tensor * ffn_w2 = nullptr;
        ggml_tensor * ffn_b2 = nullptr;
        ggml_tensor * norm2_w = nullptr;
        ggml_tensor * norm2_b = nullptr;
    };

    wavlm_extractor_params hp_ = {};

    ggml_tensor * conv0_norm_w_ = nullptr;
    ggml_tensor * conv0_norm_b_ = nullptr;
    ggml_tensor * conv_w_[7] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};

    ggml_tensor * proj_norm_w_ = nullptr;
    ggml_tensor * proj_norm_b_ = nullptr;
    ggml_tensor * proj_w_ = nullptr;
    ggml_tensor * proj_b_ = nullptr;

    ggml_tensor * pos_conv_w_ = nullptr;
    ggml_tensor * pos_conv_b_ = nullptr;

    ggml_tensor * transformer_norm_w_ = nullptr;
    ggml_tensor * transformer_norm_b_ = nullptr;

    ggml_tensor * rel_embed_ = nullptr; // first layer relative bias embedding [num_buckets, n_heads]
    std::vector<wavlm_layer> layers_;

    mutable ggml_backend_t backend_gpu_ = nullptr;
    std::string backend_device_name_;

    struct gpu_graph_cache_entry {
        mio_ggml_graph_copy copy;
        int64_t n0 = 0;
        int64_t n1 = 0;
    };

    mutable std::unordered_map<uint64_t, gpu_graph_cache_entry> cache_gpu_;
    mutable std::vector<uint64_t> cache_gpu_lru_;
    static constexpr size_t k_gpu_graph_cache_limit = 1;

    void cache_trim(uint64_t keep_key) const;
    void cache_clear() const;

    bool get_u32_kv(const char * key, uint32_t & out) const;
    bool get_f32_kv(const char * key, float & out) const;
};
