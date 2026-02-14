#include "miocodec-decoder.h"
#include "ggml-graph-copy-host.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace {

static constexpr float k_pi = 3.14159265358979323846f;

static void set_backend_threads(ggml_backend_t backend, int n_threads) {
    if (backend == nullptr || n_threads <= 0) {
        return;
    }

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (reg == nullptr) {
        return;
    }

    auto fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (fn != nullptr) {
        fn(backend, n_threads);
    }
}

static bool backend_name_is_metal(const char * name) {
    return name != nullptr && std::strncmp(name, "MTL", 3) == 0;
}

static bool backend_name_is_cpu(const char * name) {
    if (name == nullptr || name[0] == '\0') {
        return false;
    }
    const unsigned char c0 = (unsigned char) name[0];
    const unsigned char c1 = (unsigned char) name[1];
    const unsigned char c2 = (unsigned char) name[2];
    return std::toupper(c0) == 'C' &&
           std::toupper(c1) == 'P' &&
           std::toupper(c2) == 'U' &&
           name[3] == '\0';
}

static bool backend_name_is_cpu(const std::string & name) {
    return backend_name_is_cpu(name.c_str());
}

static ggml_backend_t init_gpu_backend(const std::string & preferred_backend_name) {
    ggml_backend_t backend_gpu = nullptr;
    if (!preferred_backend_name.empty()) {
        backend_gpu = ggml_backend_init_by_name(preferred_backend_name.c_str(), nullptr);
        if (backend_gpu != nullptr) {
            return backend_gpu;
        }
    }
    const char * env_backend_name = std::getenv("MIO_TTS_BACKEND_DEVICE");
    const bool has_env_backend_name = env_backend_name != nullptr && env_backend_name[0] != '\0';
    if (env_backend_name != nullptr && env_backend_name[0] != '\0') {
        backend_gpu = ggml_backend_init_by_name(env_backend_name, nullptr);
    }
    if (backend_gpu == nullptr) {
        backend_gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    }
    if (backend_gpu == nullptr) {
        backend_gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
    }

    return backend_gpu;
}

static void log_backend_once(const char * tag, ggml_backend_t backend) {
    static bool logged = false;
    if (logged) {
        return;
    }
    logged = true;
    if (backend == nullptr) {
        std::fprintf(stderr, "mio: %s backend = CPU (GPU backend unavailable)\n", tag);
        return;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    std::fprintf(
            stderr,
            "mio: %s backend = %s (%s)\n",
            tag,
            ggml_backend_name(backend),
            dev ? ggml_backend_dev_description(dev) : "unknown device");
}

static inline uint64_t hash_mix_u64(uint64_t h, uint64_t v) {
    // 64-bit mix based on boost::hash_combine.
    h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    return h;
}

static ggml_tensor * repeat_cuda_safe(ggml_context * ctx, ggml_tensor * src, ggml_tensor * like) {
    if (src == nullptr || like == nullptr) {
        return ggml_repeat(ctx, src, like);
    }

    if (like->type == GGML_TYPE_F32 || like->type == GGML_TYPE_F16) {
        if (src->type != like->type) {
            src = ggml_cast(ctx, src, like->type);
        }
    } else if (src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_F16) {
        src = ggml_cast(ctx, src, GGML_TYPE_F16);
    }

    if (!ggml_is_contiguous(src)) {
        src = ggml_cont(ctx, src);
    }

    return ggml_repeat(ctx, src, like);
}

static ggml_tensor * linear(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    if (b != nullptr) {
        y = ggml_add(ctx, y, repeat_cuda_safe(ctx, b, y));
    }
    return y;
}

static ggml_tensor * layer_norm(
        ggml_context * ctx,
        ggml_tensor * x,
        ggml_tensor * w,
        ggml_tensor * b,
        float eps) {
    ggml_tensor * y = ggml_norm(ctx, x, eps);
    if (w != nullptr) {
        y = ggml_mul(ctx, y, repeat_cuda_safe(ctx, w, y));
    }
    if (b != nullptr) {
        y = ggml_add(ctx, y, repeat_cuda_safe(ctx, b, y));
    }
    return y;
}

} // namespace

miocodec_decoder::miocodec_decoder() = default;

miocodec_decoder::~miocodec_decoder() {
    cache_clear(cache_decode_, cache_decode_lru_);
    cache_clear(cache_embed_, cache_embed_lru_);
    if (backend_gpu_ != nullptr) {
        ggml_backend_free(backend_gpu_);
        backend_gpu_ = nullptr;
    }
    if (ctx_weights_ != nullptr) {
        ggml_free(ctx_weights_);
        ctx_weights_ = nullptr;
    }
    if (ctx_gguf_ != nullptr) {
        gguf_free(ctx_gguf_);
        ctx_gguf_ = nullptr;
    }
}

bool miocodec_decoder::is_loaded() const {
    return ctx_gguf_ != nullptr && ctx_weights_ != nullptr;
}

bool miocodec_decoder::has_dynamic_global() const {
    return hp_.dynamic_global;
}

bool miocodec_decoder::has_global_encoder() const {
    return ge_embed_w_ != nullptr &&
           ge_embed_b_ != nullptr &&
           ge_norm_w_ != nullptr &&
           ge_norm_b_ != nullptr &&
           ge_final_norm_w_ != nullptr &&
           ge_final_norm_b_ != nullptr &&
           ge_pool_attn0_w_ != nullptr &&
           ge_pool_attn0_b_ != nullptr &&
           ge_pool_attn2_w_ != nullptr &&
           ge_pool_attn2_b_ != nullptr &&
           ge_pool_proj_w_ != nullptr &&
           ge_pool_proj_b_ != nullptr &&
           ge_pool_norm_w_ != nullptr &&
           ge_pool_norm_b_ != nullptr &&
           !ge_blocks_.empty();
}

bool miocodec_decoder::output_is_mel() const {
    return hp_.model_type == 1;
}

bool miocodec_decoder::has_vocoder() const {
    return hp_.has_vocoder;
}

bool miocodec_decoder::set_backend_device(const std::string & backend_name, std::string & err) {
    ggml_backend_t backend_next = nullptr;
    if (!backend_name.empty()) {
        if (!backend_name_is_cpu(backend_name)) {
            backend_next = ggml_backend_init_by_name(backend_name.c_str(), nullptr);
            if (backend_next == nullptr) {
                err = "failed to initialize backend device: " + backend_name;
                return false;
            }
        }
    } else {
        backend_next = init_gpu_backend(backend_name);
    }

    cache_clear(cache_decode_, cache_decode_lru_);
    cache_clear(cache_embed_, cache_embed_lru_);

    if (backend_gpu_ != nullptr) {
        ggml_backend_free(backend_gpu_);
        backend_gpu_ = nullptr;
    }

    backend_gpu_ = backend_next;
    backend_device_name_ = backend_name;
    return true;
}

const miocodec_decoder_params & miocodec_decoder::params() const {
    return hp_;
}

size_t miocodec_decoder::estimate_decode_workspace_bytes(int n_tokens) const {
    const int n_tok = std::max(1, n_tokens);
    const int target_audio_len = n_tok * hp_.samples_per_token;
    const int target_frames = std::max(1, target_audio_len / std::max(1, hp_.hop_length));

    // Dimension-based peak estimate for no_alloc=true + ggml_gallocr.
    // Unlike the old no_alloc=false arena model, layers reuse buffer space,
    // so we estimate one-layer peaks + core extras rather than summing all layers.

    // Prenet: operates on n_tok sequence length.
    const size_t p_dim  = (size_t) std::max(1, hp_.prenet_dim);
    const size_t p_ff   = (size_t) std::max(1, hp_.prenet_ff);
    const size_t p_heads = (size_t) std::max(1, hp_.prenet_heads);
    const size_t p_dim_bytes  = (size_t) n_tok * p_dim * sizeof(float);
    const size_t p_ff_bytes   = (size_t) n_tok * p_ff * sizeof(float);
    const size_t p_attn_bytes = (size_t) n_tok * (size_t) n_tok * p_heads * sizeof(float);
    // One-layer peak (attention + FFN + residual/normalization intermediates).
    const size_t prenet_peak = 10 * p_dim_bytes + 2 * p_ff_bytes + 2 * p_attn_bytes;

    // Decoder: operates on target_frames sequence length.
    const size_t d_dim  = (size_t) std::max(1, hp_.decoder_dim);
    const size_t d_ff   = (size_t) std::max(1, hp_.decoder_ff);
    const size_t d_heads = (size_t) std::max(1, hp_.decoder_heads);
    const size_t d_dim_bytes  = (size_t) target_frames * d_dim * sizeof(float);
    const size_t d_ff_bytes   = (size_t) target_frames * d_ff * sizeof(float);
    const size_t d_attn_bytes = (size_t) target_frames * (size_t) target_frames * d_heads * sizeof(float);
    // One-layer peak (adaln + attention + FFN).
    const size_t decoder_peak = 14 * d_dim_bytes + 3 * d_ff_bytes + 2 * d_attn_bytes;

    size_t estimate = prenet_peak + decoder_peak;

    // Additional: vocoder/resnet intermediates, upsample, masks, iSTFT output.
    estimate += 20 * d_dim_bytes;

    // ggml context/graph overhead + alignment padding.
    estimate += 64ull * 1024ull * 1024ull;

    // Safety margin for backend-dependent planner variations.
    estimate += estimate / 2;

    // Final slack for edge cases.
    estimate += 32ull * 1024ull * 1024ull;

    return estimate;
}

void miocodec_decoder::reserve_decode_workspace(int n_tokens) const {
    // no_alloc=true + ggml_gallocr path no longer relies on persistent arena storage.
    (void) n_tokens;
}

size_t miocodec_decoder::estimate_embedding_workspace_bytes(int n_frames) const {
    const int frames = std::max(1, n_frames);

    // Global encoder: ConvNeXt blocks operating on [dim, frames].
    const size_t ge_in   = (size_t) std::max(1, hp_.global_encoder_input_channels);
    const size_t ge_dim  = (size_t) std::max(1, hp_.global_encoder_dim);
    const size_t ge_ff   = (size_t) std::max(1, hp_.global_encoder_intermediate_dim);

    const size_t ssl_bytes = ge_in * (size_t) frames * sizeof(float);
    const size_t dim_bytes = ge_dim * (size_t) frames * sizeof(float);
    const size_t ff_bytes  = ge_ff * (size_t) frames * sizeof(float);

    // Input + embed conv + ggml_cast: ~6 ssl-sized tensors.
    // Per ConvNeXt block peak: ~14 dim-sized + 2 ff-sized tensors
    //   (includes layer_norm, linear, repeat_cuda_safe, ggml_cont intermediates).
    // Attentive stats pooling + final norm peak: ~6 dim-sized tensors.
    size_t estimate = 6 * ssl_bytes
                    + (14 * dim_bytes + 2 * ff_bytes)
                    + 6 * dim_bytes;

    // ggml overhead (tensor descriptors, graph nodes, etc.).
    estimate += 32ull * 1024ull * 1024ull;

    // Safety margin for backend-dependent graph expansion and alignment.
    estimate += estimate / 2;

    // Final slack.
    estimate += 16ull * 1024ull * 1024ull;

    return estimate;
}

void miocodec_decoder::reserve_embedding_workspace(int n_frames) const {
    // no_alloc=true + ggml_gallocr path no longer relies on persistent arena storage.
    (void) n_frames;
}

void miocodec_decoder::cache_trim(
        std::unordered_map<uint64_t, gpu_graph_cache_entry> & cache,
        std::vector<uint64_t> & lru,
        uint64_t keep_key) const {
    while (cache.size() > k_gpu_graph_cache_limit && !lru.empty()) {
        const uint64_t k = lru.front();
        lru.erase(lru.begin());
        if (k == keep_key) {
            lru.push_back(k);
            continue;
        }
        auto it = cache.find(k);
        if (it == cache.end()) {
            continue;
        }
        mio_ggml_graph_copy_free(it->second.copy);
        cache.erase(it);
    }
}

void miocodec_decoder::cache_clear(
        std::unordered_map<uint64_t, gpu_graph_cache_entry> & cache,
        std::vector<uint64_t> & lru) const {
    for (auto & kv : cache) {
        mio_ggml_graph_copy_free(kv.second.copy);
    }
    cache.clear();
    lru.clear();
}

bool miocodec_decoder::get_u32_kv(const char * key, uint32_t & out) const {
    if (ctx_gguf_ == nullptr) {
        return false;
    }
    const int key_id = gguf_find_key(ctx_gguf_, key);
    if (key_id < 0) {
        return false;
    }

    switch (gguf_get_kv_type(ctx_gguf_, key_id)) {
        case GGUF_TYPE_UINT8:  out = gguf_get_val_u8 (ctx_gguf_, key_id); return true;
        case GGUF_TYPE_UINT16: out = gguf_get_val_u16(ctx_gguf_, key_id); return true;
        case GGUF_TYPE_UINT32: out = gguf_get_val_u32(ctx_gguf_, key_id); return true;
        case GGUF_TYPE_INT8:   out = (uint32_t) gguf_get_val_i8 (ctx_gguf_, key_id); return true;
        case GGUF_TYPE_INT16:  out = (uint32_t) gguf_get_val_i16(ctx_gguf_, key_id); return true;
        case GGUF_TYPE_INT32:  out = (uint32_t) gguf_get_val_i32(ctx_gguf_, key_id); return true;
        default: return false;
    }
}

bool miocodec_decoder::get_f32_kv(const char * key, float & out) const {
    if (ctx_gguf_ == nullptr) {
        return false;
    }
    const int key_id = gguf_find_key(ctx_gguf_, key);
    if (key_id < 0) {
        return false;
    }

    switch (gguf_get_kv_type(ctx_gguf_, key_id)) {
        case GGUF_TYPE_FLOAT32: out = gguf_get_val_f32(ctx_gguf_, key_id); return true;
        case GGUF_TYPE_FLOAT64: out = (float) gguf_get_val_f64(ctx_gguf_, key_id); return true;
        default: return false;
    }
}

bool miocodec_decoder::load(const std::string & path, std::string & err) {
    cache_clear(cache_decode_, cache_decode_lru_);
    cache_clear(cache_embed_, cache_embed_lru_);

    if (ctx_weights_ != nullptr) {
        ggml_free(ctx_weights_);
        ctx_weights_ = nullptr;
    }
    if (ctx_gguf_ != nullptr) {
        gguf_free(ctx_gguf_);
        ctx_gguf_ = nullptr;
    }

    tok_embd_ = nullptr;
    prenet_.clear();
    prior_.clear();
    decoder_.clear();
    post_.clear();
    ge_blocks_.clear();
    mel_postnet_.clear();
    vocoder_num_upsamples_ = 0;
    vocoder_num_kernels_ = 0;
    vocoder_upsample_rates_.clear();
    vocoder_conv_pre_ = {};
    vocoder_conv_post_ = {};
    vocoder_ups_.clear();
    vocoder_resblocks_.clear();
    vocoder_activation_post_ = {};
    wave_upsampler_.clear();
    wave_upsampler_factors_.clear();
    wave_upsampler_kernel_sizes_.clear();
    wave_upsampler_total_factor_ = 1;
    wave_upsampler_out_proj_w_ = nullptr;
    wave_upsampler_out_proj_b_ = nullptr;
    wave_upsampler_out_snake_alpha_ = nullptr;
    wave_upsampler_out_snake_beta_ = nullptr;
    decoder_norm_w_ = nullptr;
    decoder_norm_b_ = nullptr;
    decoder_norm_cond_w_ = nullptr;
    decoder_norm_cond_b_ = nullptr;
    ge_embed_w_ = nullptr;
    ge_embed_b_ = nullptr;
    ge_norm_w_ = nullptr;
    ge_norm_b_ = nullptr;
    ge_final_norm_w_ = nullptr;
    ge_final_norm_b_ = nullptr;
    ge_pool_attn0_w_ = nullptr;
    ge_pool_attn0_b_ = nullptr;
    ge_pool_attn2_w_ = nullptr;
    ge_pool_attn2_b_ = nullptr;
    ge_pool_proj_w_ = nullptr;
    ge_pool_proj_b_ = nullptr;
    ge_pool_norm_w_ = nullptr;
    ge_pool_norm_b_ = nullptr;

    gguf_init_params params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ &ctx_weights_,
    };

    ctx_gguf_ = gguf_init_from_file(path.c_str(), params);
    if (ctx_gguf_ == nullptr || ctx_weights_ == nullptr) {
        err = "failed to load GGUF";
        return false;
    }

    uint32_t u32 = 0;
    float f32 = 0.0f;
    if (get_u32_kv("miocodec.model_type", u32))      hp_.model_type = (int) u32;
    if (get_u32_kv("miocodec.sample_rate", u32))     hp_.sample_rate = (int) u32;
    if (get_u32_kv("miocodec.n_fft", u32))           hp_.n_fft = (int) u32;
    if (get_u32_kv("miocodec.hop_length", u32))      hp_.hop_length = (int) u32;
    if (get_u32_kv("miocodec.n_mels", u32))          hp_.n_mels = (int) u32;
    if (get_u32_kv("miocodec.samples_per_token", u32)) hp_.samples_per_token = (int) u32;

    if (get_u32_kv("miocodec.prenet_layers", u32))   hp_.prenet_layers = (int) u32;
    if (get_u32_kv("miocodec.prenet_dim", u32))      hp_.prenet_dim = (int) u32;
    if (get_u32_kv("miocodec.prenet_heads", u32))    hp_.prenet_heads = (int) u32;
    if (get_u32_kv("miocodec.prenet_ff", u32))       hp_.prenet_ff = (int) u32;
    if (get_u32_kv("miocodec.prenet_window", u32))   hp_.prenet_window = (int) u32;

    if (get_u32_kv("miocodec.decoder_layers", u32))  hp_.decoder_layers = (int) u32;
    if (get_u32_kv("miocodec.decoder_dim", u32))     hp_.decoder_dim = (int) u32;
    if (get_u32_kv("miocodec.decoder_heads", u32))   hp_.decoder_heads = (int) u32;
    if (get_u32_kv("miocodec.decoder_ff", u32))      hp_.decoder_ff = (int) u32;
    if (get_u32_kv("miocodec.decoder_window", u32))  hp_.decoder_window = (int) u32;
    if (get_u32_kv("miocodec.decoder_adanorm_dim", u32)) hp_.decoder_adanorm_dim = (int) u32;

    if (get_u32_kv("miocodec.resnet_blocks", u32))   hp_.resnet_blocks = (int) u32;
    if (get_u32_kv("miocodec.resnet_groups", u32))   hp_.resnet_groups = (int) u32;
    int wave_upsampler_layers = 0;
    if (get_u32_kv("miocodec.wave_upsampler_layers", u32)) wave_upsampler_layers = (int) u32;
    if (get_u32_kv("miocodec.dynamic_global", u32))  hp_.dynamic_global = (u32 != 0);
    if (get_u32_kv("miocodec.has_vocoder", u32))     hp_.has_vocoder = (u32 != 0);
    if (get_u32_kv("miocodec.mel_postnet_layers", u32)) hp_.mel_postnet_layers = (int) u32;
    if (get_u32_kv("miocodec.mel_postnet_kernel_size", u32)) hp_.mel_postnet_kernel_size = (int) u32;

    if (get_u32_kv("miocodec.global_encoder.input_channels", u32)) hp_.global_encoder_input_channels = (int) u32;
    if (get_u32_kv("miocodec.global_encoder.output_channels", u32)) hp_.global_encoder_output_channels = (int) u32;
    if (get_u32_kv("miocodec.global_encoder.dim", u32)) hp_.global_encoder_dim = (int) u32;
    if (get_u32_kv("miocodec.global_encoder.intermediate_dim", u32)) hp_.global_encoder_intermediate_dim = (int) u32;
    if (get_u32_kv("miocodec.global_encoder.num_layers", u32)) hp_.global_encoder_layers = (int) u32;

    if (get_f32_kv("miocodec.rope_theta", f32))      hp_.rope_theta = f32;
    if (get_f32_kv("miocodec.norm_eps", f32))        hp_.norm_eps = f32;
    if (get_f32_kv("miocodec.group_norm_eps", f32))  hp_.group_norm_eps = f32;

    auto require_tensor = [&](const std::string & name) -> ggml_tensor * {
        ggml_tensor * t = ggml_get_tensor(ctx_weights_, name.c_str());
        if (t == nullptr) {
            err = "missing tensor: " + name;
        }
        return t;
    };

    tok_embd_ = require_tensor("token_embd");
    if (tok_embd_ == nullptr) {
        return false;
    }

    prenet_.resize(hp_.prenet_layers);
    for (int i = 0; i < hp_.prenet_layers; ++i) {
        auto & blk = prenet_[i];
        blk.attn_norm_w = require_tensor("wave_prenet.blk." + std::to_string(i) + ".attn_norm.weight");
        blk.attn_norm_b = require_tensor("wave_prenet.blk." + std::to_string(i) + ".attn_norm.bias");
        blk.wq          = require_tensor("wave_prenet.blk." + std::to_string(i) + ".attn_q.weight");
        blk.wk          = require_tensor("wave_prenet.blk." + std::to_string(i) + ".attn_k.weight");
        blk.wv          = require_tensor("wave_prenet.blk." + std::to_string(i) + ".attn_v.weight");
        blk.wo          = require_tensor("wave_prenet.blk." + std::to_string(i) + ".attn_output.weight");
        blk.ffn_norm_w  = require_tensor("wave_prenet.blk." + std::to_string(i) + ".ffn_norm.weight");
        blk.ffn_norm_b  = require_tensor("wave_prenet.blk." + std::to_string(i) + ".ffn_norm.bias");
        blk.w1          = require_tensor("wave_prenet.blk." + std::to_string(i) + ".ffn_gate.weight");
        blk.w2          = require_tensor("wave_prenet.blk." + std::to_string(i) + ".ffn_down.weight");
        blk.w3          = require_tensor("wave_prenet.blk." + std::to_string(i) + ".ffn_up.weight");
        if (err.size() > 0) {
            return false;
        }
    }

    prenet_norm_w_ = require_tensor("wave_prenet.norm.weight");
    prenet_norm_b_ = require_tensor("wave_prenet.norm.bias");
    prenet_out_w_  = require_tensor("wave_prenet.output.weight");
    prenet_out_b_  = require_tensor("wave_prenet.output.bias");
    upsample_w_    = require_tensor("wave_upsample.weight");
    upsample_b_    = require_tensor("wave_upsample.bias");
    if (err.size() > 0) {
        return false;
    }

    if (hp_.model_type == 0) {
        prior_.resize(hp_.resnet_blocks);
        post_.resize(hp_.resnet_blocks);
        for (int i = 0; i < hp_.resnet_blocks; ++i) {
            auto & p = prior_[i];
            p.norm1_w = require_tensor("wave_prior." + std::to_string(i) + ".norm1.weight");
            p.norm1_b = require_tensor("wave_prior." + std::to_string(i) + ".norm1.bias");
            p.conv1_w = require_tensor("wave_prior." + std::to_string(i) + ".conv1.weight");
            p.conv1_b = require_tensor("wave_prior." + std::to_string(i) + ".conv1.bias");
            p.norm2_w = require_tensor("wave_prior." + std::to_string(i) + ".norm2.weight");
            p.norm2_b = require_tensor("wave_prior." + std::to_string(i) + ".norm2.bias");
            p.conv2_w = require_tensor("wave_prior." + std::to_string(i) + ".conv2.weight");
            p.conv2_b = require_tensor("wave_prior." + std::to_string(i) + ".conv2.bias");

            auto & q = post_[i];
            q.norm1_w = require_tensor("wave_post." + std::to_string(i) + ".norm1.weight");
            q.norm1_b = require_tensor("wave_post." + std::to_string(i) + ".norm1.bias");
            q.conv1_w = require_tensor("wave_post." + std::to_string(i) + ".conv1.weight");
            q.conv1_b = require_tensor("wave_post." + std::to_string(i) + ".conv1.bias");
            q.norm2_w = require_tensor("wave_post." + std::to_string(i) + ".norm2.weight");
            q.norm2_b = require_tensor("wave_post." + std::to_string(i) + ".norm2.bias");
            q.conv2_w = require_tensor("wave_post." + std::to_string(i) + ".conv2.weight");
            q.conv2_b = require_tensor("wave_post." + std::to_string(i) + ".conv2.bias");

            if (err.size() > 0) {
                return false;
            }
        }
    }

    if (wave_upsampler_layers > 0) {
        if (hp_.model_type != 0) {
            err = "wave upsampler is supported only for wave-mode models";
            return false;
        }

        ggml_tensor * factors_t = require_tensor("miocodec.wave_upsampler.factors");
        ggml_tensor * kernels_t = require_tensor("miocodec.wave_upsampler.kernel_sizes");
        if (err.size() > 0) {
            return false;
        }
        if (factors_t->type != GGML_TYPE_I32 || kernels_t->type != GGML_TYPE_I32) {
            err = "wave upsampler factors/kernel sizes must be i32 tensors";
            return false;
        }
        if ((int) factors_t->ne[0] != wave_upsampler_layers || (int) kernels_t->ne[0] != wave_upsampler_layers) {
            err = "wave upsampler tensor size mismatch";
            return false;
        }

        wave_upsampler_factors_.resize((size_t) wave_upsampler_layers);
        wave_upsampler_kernel_sizes_.resize((size_t) wave_upsampler_layers);
        std::memcpy(
                wave_upsampler_factors_.data(),
                factors_t->data,
                wave_upsampler_factors_.size() * sizeof(int32_t));
        std::memcpy(
                wave_upsampler_kernel_sizes_.data(),
                kernels_t->data,
                wave_upsampler_kernel_sizes_.size() * sizeof(int32_t));

        wave_upsampler_.resize((size_t) wave_upsampler_layers);
        wave_upsampler_total_factor_ = 1;
        for (int i = 0; i < wave_upsampler_layers; ++i) {
            const int factor = wave_upsampler_factors_[(size_t) i];
            const int kernel = wave_upsampler_kernel_sizes_[(size_t) i];
            if (factor <= 0 || kernel <= 0) {
                err = "wave upsampler factor/kernel must be positive";
                return false;
            }
            wave_upsampler_total_factor_ *= factor;

            auto & st = wave_upsampler_[(size_t) i];
            st.up_w = require_tensor("wave_upsampler.up." + std::to_string(i) + ".weight");
            st.up_b = require_tensor("wave_upsampler.up." + std::to_string(i) + ".bias");
            st.snake_alpha = require_tensor("wave_upsampler.snake." + std::to_string(i) + ".alpha");
            st.snake_beta = require_tensor("wave_upsampler.snake." + std::to_string(i) + ".beta");
            st.resblk.norm1_w = require_tensor("wave_upsampler.resblk." + std::to_string(i) + ".norm1.weight");
            st.resblk.norm1_b = require_tensor("wave_upsampler.resblk." + std::to_string(i) + ".norm1.bias");
            st.resblk.conv1_w = require_tensor("wave_upsampler.resblk." + std::to_string(i) + ".conv1.weight");
            st.resblk.conv1_b = require_tensor("wave_upsampler.resblk." + std::to_string(i) + ".conv1.bias");
            st.resblk.norm2_w = require_tensor("wave_upsampler.resblk." + std::to_string(i) + ".norm2.weight");
            st.resblk.norm2_b = require_tensor("wave_upsampler.resblk." + std::to_string(i) + ".norm2.bias");
            st.resblk.conv2_w = require_tensor("wave_upsampler.resblk." + std::to_string(i) + ".conv2.weight");
            st.resblk.conv2_b = require_tensor("wave_upsampler.resblk." + std::to_string(i) + ".conv2.bias");
            if (err.size() > 0) {
                return false;
            }
        }

        wave_upsampler_out_proj_w_ = require_tensor("wave_upsampler.out_proj.weight");
        wave_upsampler_out_proj_b_ = require_tensor("wave_upsampler.out_proj.bias");
        wave_upsampler_out_snake_alpha_ = require_tensor("wave_upsampler.out_snake.alpha");
        wave_upsampler_out_snake_beta_ = require_tensor("wave_upsampler.out_snake.beta");
        if (err.size() > 0) {
            return false;
        }
    }

    decoder_.resize(hp_.decoder_layers);
    for (int i = 0; i < hp_.decoder_layers; ++i) {
        auto & blk = decoder_[i];
        blk.attn_norm_w = ggml_get_tensor(ctx_weights_, ("wave_decoder.blk." + std::to_string(i) + ".attn_norm.weight").c_str());
        blk.attn_norm_b = ggml_get_tensor(ctx_weights_, ("wave_decoder.blk." + std::to_string(i) + ".attn_norm.bias").c_str());
        blk.attn_cond_w = ggml_get_tensor(ctx_weights_, ("wave_decoder.blk." + std::to_string(i) + ".attn_cond.weight").c_str());
        blk.attn_cond_b = ggml_get_tensor(ctx_weights_, ("wave_decoder.blk." + std::to_string(i) + ".attn_cond.bias").c_str());
        blk.wq          = require_tensor("wave_decoder.blk." + std::to_string(i) + ".attn_q.weight");
        blk.wk          = require_tensor("wave_decoder.blk." + std::to_string(i) + ".attn_k.weight");
        blk.wv          = require_tensor("wave_decoder.blk." + std::to_string(i) + ".attn_v.weight");
        blk.wo          = require_tensor("wave_decoder.blk." + std::to_string(i) + ".attn_output.weight");
        blk.ffn_norm_w  = ggml_get_tensor(ctx_weights_, ("wave_decoder.blk." + std::to_string(i) + ".ffn_norm.weight").c_str());
        blk.ffn_norm_b  = ggml_get_tensor(ctx_weights_, ("wave_decoder.blk." + std::to_string(i) + ".ffn_norm.bias").c_str());
        blk.ffn_cond_w  = ggml_get_tensor(ctx_weights_, ("wave_decoder.blk." + std::to_string(i) + ".ffn_cond.weight").c_str());
        blk.ffn_cond_b  = ggml_get_tensor(ctx_weights_, ("wave_decoder.blk." + std::to_string(i) + ".ffn_cond.bias").c_str());
        blk.w1          = require_tensor("wave_decoder.blk." + std::to_string(i) + ".ffn_gate.weight");
        blk.w2          = require_tensor("wave_decoder.blk." + std::to_string(i) + ".ffn_down.weight");
        blk.w3          = require_tensor("wave_decoder.blk." + std::to_string(i) + ".ffn_up.weight");
        if (err.size() > 0) {
            return false;
        }

        if (hp_.dynamic_global) {
            if (blk.attn_cond_w == nullptr || blk.attn_cond_b == nullptr ||
                blk.ffn_cond_w == nullptr || blk.ffn_cond_b == nullptr) {
                err = "dynamic_global=1 but decoder conditioning tensors are missing";
                return false;
            }
        } else {
            if (blk.attn_norm_w == nullptr || blk.attn_norm_b == nullptr ||
                blk.ffn_norm_w == nullptr || blk.ffn_norm_b == nullptr) {
                err = "static decoder tensors are missing";
                return false;
            }
        }
    }

    decoder_norm_w_ = ggml_get_tensor(ctx_weights_, "wave_decoder.norm.weight");
    decoder_norm_b_ = ggml_get_tensor(ctx_weights_, "wave_decoder.norm.bias");
    decoder_norm_cond_w_ = ggml_get_tensor(ctx_weights_, "wave_decoder.norm_cond.weight");
    decoder_norm_cond_b_ = ggml_get_tensor(ctx_weights_, "wave_decoder.norm_cond.bias");
    istft_out_w_    = require_tensor("istft_head.out.weight");
    istft_out_b_    = require_tensor("istft_head.out.bias");
    if (err.size() > 0) {
        return false;
    }

    if (hp_.model_type == 1) {
        mel_postnet_.resize(std::max(0, hp_.mel_postnet_layers));
        for (int i = 0; i < hp_.mel_postnet_layers; ++i) {
            auto & l = mel_postnet_[i];
            l.conv_w = require_tensor("mel_postnet." + std::to_string(i) + ".conv.weight");
            l.conv_b = require_tensor("mel_postnet." + std::to_string(i) + ".conv.bias");
            l.norm_w = require_tensor("mel_postnet." + std::to_string(i) + ".norm.weight");
            l.norm_b = require_tensor("mel_postnet." + std::to_string(i) + ".norm.bias");
            if (err.size() > 0) {
                return false;
            }
        }
    }

    if (hp_.dynamic_global) {
        if (decoder_norm_cond_w_ == nullptr || decoder_norm_cond_b_ == nullptr) {
            err = "dynamic_global=1 but decoder norm conditioning tensors are missing";
            return false;
        }
    } else {
        if (decoder_norm_w_ == nullptr || decoder_norm_b_ == nullptr) {
            err = "static decoder norm tensors are missing";
            return false;
        }
    }

    // Optional global encoder tensors for runtime reference-audio conditioning.
    ge_embed_w_ = ggml_get_tensor(ctx_weights_, "global_encoder.backbone.embed.weight");
    ge_embed_b_ = ggml_get_tensor(ctx_weights_, "global_encoder.backbone.embed.bias");
    ge_norm_w_ = ggml_get_tensor(ctx_weights_, "global_encoder.backbone.norm.weight");
    ge_norm_b_ = ggml_get_tensor(ctx_weights_, "global_encoder.backbone.norm.bias");
    ge_final_norm_w_ = ggml_get_tensor(ctx_weights_, "global_encoder.backbone.final_norm.weight");
    ge_final_norm_b_ = ggml_get_tensor(ctx_weights_, "global_encoder.backbone.final_norm.bias");

    ge_pool_attn0_w_ = ggml_get_tensor(ctx_weights_, "global_encoder.pool.attn0.weight");
    ge_pool_attn0_b_ = ggml_get_tensor(ctx_weights_, "global_encoder.pool.attn0.bias");
    ge_pool_attn2_w_ = ggml_get_tensor(ctx_weights_, "global_encoder.pool.attn2.weight");
    ge_pool_attn2_b_ = ggml_get_tensor(ctx_weights_, "global_encoder.pool.attn2.bias");
    ge_pool_proj_w_ = ggml_get_tensor(ctx_weights_, "global_encoder.pool.proj.weight");
    ge_pool_proj_b_ = ggml_get_tensor(ctx_weights_, "global_encoder.pool.proj.bias");
    ge_pool_norm_w_ = ggml_get_tensor(ctx_weights_, "global_encoder.pool.norm.weight");
    ge_pool_norm_b_ = ggml_get_tensor(ctx_weights_, "global_encoder.pool.norm.bias");

    if (ge_embed_w_ != nullptr) {
        ge_blocks_.resize(hp_.global_encoder_layers);
        for (int i = 0; i < hp_.global_encoder_layers; ++i) {
            auto & b = ge_blocks_[i];
            b.dwconv_w = ggml_get_tensor(ctx_weights_, ("global_encoder.backbone.blk." + std::to_string(i) + ".dwconv.weight").c_str());
            b.dwconv_b = ggml_get_tensor(ctx_weights_, ("global_encoder.backbone.blk." + std::to_string(i) + ".dwconv.bias").c_str());
            b.norm_w = ggml_get_tensor(ctx_weights_, ("global_encoder.backbone.blk." + std::to_string(i) + ".norm.weight").c_str());
            b.norm_b = ggml_get_tensor(ctx_weights_, ("global_encoder.backbone.blk." + std::to_string(i) + ".norm.bias").c_str());
            b.pw1_w = ggml_get_tensor(ctx_weights_, ("global_encoder.backbone.blk." + std::to_string(i) + ".pw1.weight").c_str());
            b.pw1_b = ggml_get_tensor(ctx_weights_, ("global_encoder.backbone.blk." + std::to_string(i) + ".pw1.bias").c_str());
            b.pw2_w = ggml_get_tensor(ctx_weights_, ("global_encoder.backbone.blk." + std::to_string(i) + ".pw2.weight").c_str());
            b.pw2_b = ggml_get_tensor(ctx_weights_, ("global_encoder.backbone.blk." + std::to_string(i) + ".pw2.bias").c_str());
            b.gamma = ggml_get_tensor(ctx_weights_, ("global_encoder.backbone.blk." + std::to_string(i) + ".gamma").c_str());
        }
    }

    if (hp_.has_vocoder) {
        if (get_u32_kv("miovocoder.num_upsamples", u32)) {
            vocoder_num_upsamples_ = (int) u32;
        }
        if (get_u32_kv("miovocoder.num_kernels", u32)) {
            vocoder_num_kernels_ = (int) u32;
        }
        ggml_tensor * rates = require_tensor("miovocoder.upsample_rates");
        if (err.size() > 0) {
            return false;
        }
        if (rates->type != GGML_TYPE_I32) {
            err = "miovocoder.upsample_rates must be i32";
            return false;
        }
        vocoder_upsample_rates_.resize((size_t) rates->ne[0]);
        std::memcpy(vocoder_upsample_rates_.data(), rates->data, vocoder_upsample_rates_.size() * sizeof(int32_t));
        if (vocoder_num_upsamples_ <= 0) {
            vocoder_num_upsamples_ = (int) vocoder_upsample_rates_.size();
        }
        if ((int) vocoder_upsample_rates_.size() != vocoder_num_upsamples_) {
            err = "miovocoder upsample_rates size mismatch";
            return false;
        }

        vocoder_conv_pre_.weight = require_tensor("vocoder.conv_pre.weight");
        vocoder_conv_pre_.bias   = require_tensor("vocoder.conv_pre.bias");
        vocoder_conv_post_.weight = require_tensor("vocoder.conv_post.weight");
        vocoder_conv_post_.bias = nullptr;
        if (err.size() > 0) {
            return false;
        }

        vocoder_ups_.resize(vocoder_num_upsamples_);
        for (int i = 0; i < vocoder_num_upsamples_; ++i) {
            auto & u = vocoder_ups_[i];
            u.after.weight = require_tensor("vocoder.ups." + std::to_string(i) + ".after.weight");
            u.after.bias   = require_tensor("vocoder.ups." + std::to_string(i) + ".after.bias");
            u.noise.weight = require_tensor("vocoder.ups." + std::to_string(i) + ".noise.weight");
            u.noise.bias   = require_tensor("vocoder.ups." + std::to_string(i) + ".noise.bias");
            if (err.size() > 0) {
                return false;
            }
        }

        const int n_resblocks = std::max(0, vocoder_num_upsamples_ * vocoder_num_kernels_);
        vocoder_resblocks_.resize(n_resblocks);
        for (int r = 0; r < n_resblocks; ++r) {
            auto & rb = vocoder_resblocks_[r];
            for (int j = 0; j < 3; ++j) {
                rb.convs1[j].weight = require_tensor("vocoder.resblocks." + std::to_string(r) + ".convs1." + std::to_string(j) + ".weight");
                rb.convs1[j].bias   = require_tensor("vocoder.resblocks." + std::to_string(r) + ".convs1." + std::to_string(j) + ".bias");
                rb.convs2[j].weight = require_tensor("vocoder.resblocks." + std::to_string(r) + ".convs2." + std::to_string(j) + ".weight");
                rb.convs2[j].bias   = require_tensor("vocoder.resblocks." + std::to_string(r) + ".convs2." + std::to_string(j) + ".bias");
            }
            for (int a = 0; a < 6; ++a) {
                rb.acts[a].alpha = require_tensor("vocoder.resblocks." + std::to_string(r) + ".acts." + std::to_string(a) + ".alpha");
                rb.acts[a].beta = require_tensor("vocoder.resblocks." + std::to_string(r) + ".acts." + std::to_string(a) + ".beta");
                rb.acts[a].up_filter = require_tensor("vocoder.resblocks." + std::to_string(r) + ".acts." + std::to_string(a) + ".up_filter");
                rb.acts[a].down_filter = require_tensor("vocoder.resblocks." + std::to_string(r) + ".acts." + std::to_string(a) + ".down_filter");
            }
            if (err.size() > 0) {
                return false;
            }
        }

        vocoder_activation_post_.alpha = require_tensor("vocoder.activation_post.alpha");
        vocoder_activation_post_.beta = require_tensor("vocoder.activation_post.beta");
        vocoder_activation_post_.up_filter = require_tensor("vocoder.activation_post.up_filter");
        vocoder_activation_post_.down_filter = require_tensor("vocoder.activation_post.down_filter");
        if (err.size() > 0) {
            return false;
        }
    }

    return true;
}

bool miocodec_decoder::encode_global_embedding_from_ssl(
        const std::vector<float> & ssl_features,
        int n_frames,
        int n_threads,
        std::vector<float> & global_embedding_out,
        std::string & err) const {
    if (!is_loaded()) {
        err = "decoder not loaded";
        return false;
    }
    if (!has_global_encoder()) {
        err = "global encoder tensors are not available in this GGUF";
        return false;
    }
    if (n_frames <= 0) {
        err = "invalid ssl frame length";
        return false;
    }
    if ((int) ssl_features.size() != hp_.global_encoder_input_channels * n_frames) {
        err = "ssl feature shape mismatch";
        return false;
    }

    const size_t n_tensors_est = 4096;
    const size_t ctx_size = n_tensors_est * ggml_tensor_overhead()
                          + ggml_graph_overhead_custom(4096, false)
                          + 64ull * 1024ull;
    ggml_init_params params = {
        /*.mem_size   = */ ctx_size,
        /*.mem_buffer = */ nullptr,
        /*.no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        err = "ggml_init failed";
        return false;
    }

    ggml_tensor * in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp_.global_encoder_input_channels, n_frames);
    ggml_set_name(in, "miocodec.ssl_in");
    ggml_set_input(in);

    // Input [C, T] -> [T, C, 1] for conv path.
    ggml_tensor * x = ggml_cont(ctx, ggml_transpose(ctx, in)); // [T, C]
    x = ggml_reshape_3d(ctx, x, x->ne[0], x->ne[1], 1);

    // Embed conv + pre-norm.
    ggml_tensor * ew = ge_embed_w_->type == GGML_TYPE_F16 ? ge_embed_w_ : ggml_cast(ctx, ge_embed_w_, GGML_TYPE_F16);
    x = ggml_conv_1d_ph(ctx, ew, x, 1, 1);
    ggml_tensor * eb = ggml_reshape_3d(ctx, ge_embed_b_, 1, x->ne[1], 1);
    x = ggml_add(ctx, x, repeat_cuda_safe(ctx, eb, x));

    ggml_tensor * x2 = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1]); // [T, dim]
    ggml_tensor * xt = ggml_cont(ctx, ggml_transpose(ctx, x2));      // [dim, T]
    xt = layer_norm(ctx, xt, ge_norm_w_, ge_norm_b_, 1e-6f);
    x = ggml_cont(ctx, ggml_transpose(ctx, xt));                      // [T, dim]
    x = ggml_reshape_3d(ctx, x, x->ne[0], x->ne[1], 1);

    // ConvNeXt blocks.
    for (int i = 0; i < hp_.global_encoder_layers; ++i) {
        const auto & b = ge_blocks_[i];
        ggml_tensor * residual = x;

        ggml_tensor * dw = b.dwconv_w->type == GGML_TYPE_F16 ? b.dwconv_w : ggml_cast(ctx, b.dwconv_w, GGML_TYPE_F16);
        ggml_tensor * y = ggml_conv_1d_dw_ph(ctx, dw, x, 1, 1);
        ggml_tensor * db = ggml_reshape_3d(ctx, b.dwconv_b, 1, y->ne[1], 1);
        y = ggml_add(ctx, y, repeat_cuda_safe(ctx, db, y));

        ggml_tensor * y2 = ggml_reshape_2d(ctx, y, y->ne[0], y->ne[1]); // [T,dim]
        ggml_tensor * yt = ggml_cont(ctx, ggml_transpose(ctx, y2));      // [dim,T]
        yt = layer_norm(ctx, yt, b.norm_w, b.norm_b, 1e-6f);
        yt = linear(ctx, yt, b.pw1_w, b.pw1_b);
        yt = ggml_gelu(ctx, yt);
        yt = linear(ctx, yt, b.pw2_w, b.pw2_b);
        ggml_tensor * gamma = ggml_reshape_2d(ctx, b.gamma, b.gamma->ne[0], 1);
        yt = ggml_mul(ctx, yt, repeat_cuda_safe(ctx, gamma, yt));
        y = ggml_cont(ctx, ggml_transpose(ctx, yt));                    // [T,dim]
        y = ggml_reshape_3d(ctx, y, y->ne[0], y->ne[1], 1);

        x = ggml_add(ctx, residual, y);
    }

    // Final layer norm.
    x2 = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1]); // [T,dim]
    xt = ggml_cont(ctx, ggml_transpose(ctx, x2));      // [dim,T]
    xt = layer_norm(ctx, xt, ge_final_norm_w_, ge_final_norm_b_, 1e-6f);

    // Attentive stats pooling.
    x = ggml_cont(ctx, ggml_transpose(ctx, xt)); // [T,dim]
    x = ggml_reshape_3d(ctx, x, x->ne[0], x->ne[1], 1);

    ggml_tensor * attn0_w = ge_pool_attn0_w_->type == GGML_TYPE_F16 ? ge_pool_attn0_w_ : ggml_cast(ctx, ge_pool_attn0_w_, GGML_TYPE_F16);
    ggml_tensor * a = ggml_conv_1d_ph(ctx, attn0_w, x, 1, 0);
    ggml_tensor * a0b = ggml_reshape_3d(ctx, ge_pool_attn0_b_, 1, a->ne[1], 1);
    a = ggml_add(ctx, a, repeat_cuda_safe(ctx, a0b, a));
    a = ggml_tanh(ctx, a);
    ggml_tensor * attn2_w = ge_pool_attn2_w_->type == GGML_TYPE_F16 ? ge_pool_attn2_w_ : ggml_cast(ctx, ge_pool_attn2_w_, GGML_TYPE_F16);
    a = ggml_conv_1d_ph(ctx, attn2_w, a, 1, 0);
    ggml_tensor * a2b = ggml_reshape_3d(ctx, ge_pool_attn2_b_, 1, a->ne[1], 1);
    a = ggml_add(ctx, a, repeat_cuda_safe(ctx, a2b, a));

    ggml_tensor * a2 = ggml_reshape_2d(ctx, a, a->ne[0], a->ne[1]); // [T,dim]
    ggml_tensor * alpha = ggml_soft_max(ctx, a2);                   // softmax over T

    ggml_tensor * x_pool = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1]); // [T,dim]
    ggml_tensor * mean = ggml_sum_rows(ctx, ggml_mul(ctx, alpha, x_pool)); // [1,dim]
    ggml_tensor * m2 = ggml_sum_rows(ctx, ggml_mul(ctx, alpha, ggml_sqr(ctx, x_pool))); // [1,dim]
    ggml_tensor * var = ggml_sub(ctx, m2, ggml_sqr(ctx, mean));
    var = ggml_clamp(ctx, var, 1e-4f, 1e4f);
    ggml_tensor * std = ggml_sqrt(ctx, var);

    ggml_tensor * mean_t = ggml_cont(ctx, ggml_transpose(ctx, mean)); // [dim,1]
    ggml_tensor * std_t = ggml_cont(ctx, ggml_transpose(ctx, std));   // [dim,1]
    ggml_tensor * stat = ggml_concat(ctx, mean_t, std_t, 0);          // [2*dim,1]

    ggml_tensor * out = linear(ctx, stat, ge_pool_proj_w_, ge_pool_proj_b_); // [out_dim,1]
    out = layer_norm(ctx, out, ge_pool_norm_w_, ge_pool_norm_b_, 1e-5f);
    out = ggml_cont(ctx, out);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
    ggml_set_name(out, "miocodec.global_embedding_out");
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        err = "ggml_gallocr_alloc_graph failed";
        return false;
    }

    std::fprintf(stderr, "mio: global-embedding gallocr buffer=%.1f MiB n_frames=%d\n",
                 (double) ggml_gallocr_get_buffer_size(galloc, 0) / (1024.0 * 1024.0), n_frames);

    std::memcpy(in->data, ssl_features.data(), ssl_features.size() * sizeof(float));

    const uint64_t cache_key_embed = hash_mix_u64(0x7a3f5bbf34c2d871ull, (uint64_t) n_frames);

    bool graph_ok = false;
    if (backend_gpu_ == nullptr && !backend_name_is_cpu(backend_device_name_)) {
        backend_gpu_ = init_gpu_backend(backend_device_name_);
    }
    ggml_backend_t backend_gpu = backend_gpu_;
    log_backend_once("global-embedding", backend_gpu);
    if (backend_gpu != nullptr) {
        set_backend_threads(backend_gpu, std::max(1, n_threads));
        auto touch_lru = [&](uint64_t key) {
            auto itl = std::find(cache_embed_lru_.begin(), cache_embed_lru_.end(), key);
            if (itl != cache_embed_lru_.end()) {
                cache_embed_lru_.erase(itl);
            }
            cache_embed_lru_.push_back(key);
            cache_trim(cache_embed_, cache_embed_lru_, key);
        };

        auto it = cache_embed_.find(cache_key_embed);
        if (it == cache_embed_.end()) {
            gpu_graph_cache_entry entry;
            std::string copy_err;
            if (mio_ggml_backend_graph_copy_from_host(backend_gpu, gf, entry.copy, copy_err)) {
                entry.n0 = out->ne[0];
                entry.n1 = out->ne[1];
                auto ins = cache_embed_.emplace(cache_key_embed, std::move(entry));
                it = ins.first;
                touch_lru(cache_key_embed);
            } else {
                err = "mio_ggml_backend_graph_copy_from_host failed: " + copy_err;
            }
        } else {
            touch_lru(cache_key_embed);
        }

        if (it != cache_embed_.end()) {
            ggml_tensor * in_gpu = mio_ggml_graph_copy_get_tensor(it->second.copy, "miocodec.ssl_in");
            if (in_gpu == nullptr || in_gpu->type != GGML_TYPE_F32 ||
                ggml_nbytes(in_gpu) != ssl_features.size() * sizeof(float)) {
                err = "cached GPU graph input mismatch for global embedding";
                mio_ggml_graph_copy_free(it->second.copy);
                cache_embed_.erase(it);
            } else {
                ggml_backend_tensor_set(in_gpu, ssl_features.data(), 0, ssl_features.size() * sizeof(float));
                const ggml_status st = ggml_backend_graph_compute(backend_gpu, it->second.copy.graph);
                if (st == GGML_STATUS_SUCCESS) {
                    ggml_tensor * out_gpu = mio_ggml_graph_copy_get_tensor(it->second.copy, "miocodec.global_embedding_out");
                    if (out_gpu != nullptr && out_gpu->type == GGML_TYPE_F32) {
                        global_embedding_out.resize((size_t) ggml_nelements(out_gpu));
                        ggml_backend_tensor_get(out_gpu, global_embedding_out.data(), 0, global_embedding_out.size() * sizeof(float));
                        graph_ok = true;
                    } else {
                        err = "failed to get GPU output tensor";
                    }
                } else {
                    err = "ggml_backend_graph_compute failed on GPU backend";
                }
            }
        }
    }

    if (!graph_ok) {
        static bool logged_fallback = false;
        if (!logged_fallback) {
            logged_fallback = true;
            std::fprintf(stderr, "mio: global-embedding fallback to CPU: %s\n", err.empty() ? "unknown" : err.c_str());
        }
        err.clear();
        ggml_cplan cplan = ggml_graph_plan(gf, std::max(1, n_threads), nullptr);
        std::vector<uint8_t> work_data;
        if (cplan.work_size > 0) {
            work_data.resize(cplan.work_size);
            cplan.work_data = work_data.data();
        }

        const ggml_status status = ggml_graph_compute(gf, &cplan);
        if (status != GGML_STATUS_SUCCESS) {
            err = "ggml_graph_compute failed";
            ggml_gallocr_free(galloc);
            ggml_free(ctx);
            return false;
        }

        global_embedding_out.resize((size_t) out->ne[0]);
        std::memcpy(global_embedding_out.data(), out->data, global_embedding_out.size() * sizeof(float));
    }

    auto has_non_finite = [](const std::vector<float> & v) -> bool {
        for (float x : v) {
            if (!std::isfinite(x)) {
                return true;
            }
        }
        return false;
    };

    if (has_non_finite(global_embedding_out)) {
        const int target_dim = std::max(1, hp_.decoder_adanorm_dim);
        global_embedding_out.assign((size_t) target_dim, 0.0f);
        err = "global encoder produced non-finite values; used zero fallback embedding";
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return true;
}

bool miocodec_decoder::decode_tokens(
        const std::vector<int32_t> & tokens,
        int n_threads,
        llama_flash_attn_type flash_attn_type,
        std::vector<float> & spec_out,
        int & n_frames,
        std::string & err) const {
    static const std::vector<float> empty_embedding;
    return decode_tokens(tokens, empty_embedding, n_threads, flash_attn_type, spec_out, n_frames, err);
}

bool miocodec_decoder::decode_tokens(
        const std::vector<int32_t> & tokens,
        const std::vector<float> & global_embedding,
        int n_threads,
        llama_flash_attn_type flash_attn_type,
        std::vector<float> & spec_out,
        int & n_frames,
        std::string & err) const {
    if (!is_loaded()) {
        err = "decoder not loaded";
        return false;
    }
    if (tokens.empty()) {
        err = "empty token sequence";
        return false;
    }

    const int n_tokens = (int) tokens.size();
    const int target_audio_len = n_tokens * hp_.samples_per_token;
    const int stft_target_frames = std::max(1, target_audio_len / std::max(1, hp_.hop_length));
    const int decoder_target_frames = std::max(
            1,
            wave_upsampler_total_factor_ > 1
                ? stft_target_frames / wave_upsampler_total_factor_
                : stft_target_frames);
    const bool use_flash_attn = flash_attn_type == LLAMA_FLASH_ATTN_TYPE_ENABLED;

    const size_t n_tensors_est = 16384;
    const size_t ctx_size = n_tensors_est * ggml_tensor_overhead()
                          + ggml_graph_overhead_custom(8192, false)
                          + 64ull * 1024ull;
    ggml_init_params params = {
        /*.mem_size   = */ ctx_size,
        /*.mem_buffer = */ nullptr,
        /*.no_alloc   = */ true,
    };

    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        err = "ggml_init failed";
        return false;
    }

    struct pending_pos_fill {
        ggml_tensor * tensor = nullptr;
        int n = 0;
    };
    struct pending_mask_fill {
        ggml_tensor * tensor = nullptr;
        int n = 0;
        int window = 0;
        bool fp16 = false;
    };
    std::vector<pending_pos_fill> pending_pos;
    std::vector<pending_mask_fill> pending_masks;

    auto make_pos = [&](int n) -> ggml_tensor * {
        ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
        ggml_set_input(pos);
        pending_pos.push_back({pos, n});
        return pos;
    };

    auto make_local_mask = [&](int n, int window) -> ggml_tensor * {
        const bool fp16 = use_flash_attn;
        ggml_tensor * mask = ggml_new_tensor_2d(ctx, fp16 ? GGML_TYPE_F16 : GGML_TYPE_F32, n, n);
        ggml_set_input(mask);
        pending_masks.push_back({mask, n, window, fp16});
        return mask;
    };

    ggml_tensor * scalar_one = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_input(scalar_one);
    ggml_tensor * scalar_eps = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_input(scalar_eps);

    if (!hp_.dynamic_global && !global_embedding.empty()) {
        err = "global embedding was provided but decoder is static (dynamic_global=0)";
        ggml_free(ctx);
        return false;
    }

    std::vector<float> cond_vec;
    ggml_tensor * decoder_condition = nullptr;
    ggml_tensor * decoder_condition_act = nullptr;
    if (hp_.dynamic_global) {
        if (!global_embedding.empty()) {
            if ((int) global_embedding.size() != hp_.decoder_adanorm_dim) {
                err = "global embedding size mismatch";
                ggml_free(ctx);
                return false;
            }
            cond_vec = global_embedding;
        } else {
            cond_vec.assign((size_t) hp_.decoder_adanorm_dim, 0.0f);
        }

        decoder_condition = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp_.decoder_adanorm_dim, 1);
        ggml_set_input(decoder_condition);
        ggml_set_name(decoder_condition, "miocodec.decoder_condition");
        // Match PyTorch AdaLNZero: condition_proj is SiLU -> Linear.
        decoder_condition_act = ggml_silu(ctx, decoder_condition);
    }

    auto adaln_apply = [&](ggml_tensor * x_norm, ggml_tensor * shift, ggml_tensor * scale) -> ggml_tensor * {
        ggml_tensor * one_plus_scale = ggml_add1(ctx, scale, scalar_one);
        ggml_tensor * y = ggml_mul(ctx, x_norm, repeat_cuda_safe(ctx, one_plus_scale, x_norm));
        y = ggml_add(ctx, y, repeat_cuda_safe(ctx, shift, y));
        return y;
    };

    auto transformer_block_eval = [&](ggml_tensor * x,
                                      const transformer_block & blk,
                                      int n_heads,
                                      ggml_tensor * pos,
                                      ggml_tensor * mask,
                                      ggml_tensor * cond) -> ggml_tensor * {
        const int dim = (int) x->ne[0];
        const int seq = (int) x->ne[1];
        const int head_dim = dim / n_heads;
        const float scale = 1.0f / std::sqrt((float) head_dim);

        ggml_tensor * attn_gate = nullptr;
        ggml_tensor * xn = nullptr;
        if (cond != nullptr) {
            ggml_tensor * p = linear(ctx, cond, blk.attn_cond_w, blk.attn_cond_b); // [3*dim, 1]
            const int64_t elem = (int64_t) p->nb[0];
            ggml_tensor * shift = ggml_view_2d(ctx, p, dim, 1, p->nb[1], 0);
            ggml_tensor * scale_ada = ggml_view_2d(ctx, p, dim, 1, p->nb[1], (int64_t) dim * elem);
            attn_gate = ggml_view_2d(ctx, p, dim, 1, p->nb[1], (int64_t) 2 * (int64_t) dim * elem);
            xn = layer_norm(ctx, x, nullptr, nullptr, hp_.norm_eps);
            xn = adaln_apply(xn, shift, scale_ada);
        } else {
            xn = layer_norm(ctx, x, blk.attn_norm_w, blk.attn_norm_b, hp_.norm_eps);
        }

        ggml_tensor * q = linear(ctx, xn, blk.wq, nullptr);
        ggml_tensor * k = linear(ctx, xn, blk.wk, nullptr);
        ggml_tensor * v = linear(ctx, xn, blk.wv, nullptr);

        q = ggml_reshape_3d(ctx, q, head_dim, n_heads, seq);
        k = ggml_reshape_3d(ctx, k, head_dim, n_heads, seq);
        v = ggml_reshape_3d(ctx, v, head_dim, n_heads, seq);

        q = ggml_rope_ext(ctx, q, pos, nullptr, head_dim, 0, 0, hp_.rope_theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        k = ggml_rope_ext(ctx, k, pos, nullptr, head_dim, 0, 0, hp_.rope_theta, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

        q = ggml_reshape_4d(ctx, q, head_dim, n_heads, seq, 1);
        k = ggml_reshape_4d(ctx, k, head_dim, n_heads, seq, 1);
        v = ggml_reshape_4d(ctx, v, head_dim, n_heads, seq, 1);

        q = ggml_permute(ctx, q, 0, 2, 1, 3);
        k = ggml_permute(ctx, k, 0, 2, 1, 3);
        v = ggml_permute(ctx, v, 0, 2, 1, 3);

        q = ggml_cont(ctx, q);
        k = ggml_cont(ctx, k);
        v = ggml_cont(ctx, v);

        ggml_tensor * att = nullptr;
        if (use_flash_attn) {
            if (k->type == GGML_TYPE_F32) {
                k = ggml_cast(ctx, k, GGML_TYPE_F16);
            }
            if (v->type == GGML_TYPE_F32) {
                v = ggml_cast(ctx, v, GGML_TYPE_F16);
            }

            att = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(att, GGML_PREC_F32);
            att = ggml_cont(ctx, att);
        } else {
            ggml_tensor * kq = ggml_mul_mat(ctx, k, q);
            ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
            kq = ggml_soft_max_ext(ctx, kq, mask, scale, 0.0f);
            ggml_tensor * v_t = ggml_cont(ctx, ggml_transpose(ctx, v));
            ggml_tensor * kqv = ggml_mul_mat(ctx, v_t, kq);
            att = ggml_permute(ctx, kqv, 0, 2, 1, 3);
            att = ggml_cont(ctx, att);
        }

        att = ggml_reshape_2d(ctx, att, dim, seq);

        ggml_tensor * out = linear(ctx, att, blk.wo, nullptr);
        if (attn_gate != nullptr) {
            out = ggml_mul(ctx, out, repeat_cuda_safe(ctx, attn_gate, out));
        }
        ggml_tensor * h = ggml_add(ctx, x, out);

        ggml_tensor * ffn_gate = nullptr;
        ggml_tensor * fn = nullptr;
        if (cond != nullptr) {
            ggml_tensor * p = linear(ctx, cond, blk.ffn_cond_w, blk.ffn_cond_b); // [3*dim, 1]
            const int64_t elem = (int64_t) p->nb[0];
            ggml_tensor * shift = ggml_view_2d(ctx, p, dim, 1, p->nb[1], 0);
            ggml_tensor * scale_ada = ggml_view_2d(ctx, p, dim, 1, p->nb[1], (int64_t) dim * elem);
            ffn_gate = ggml_view_2d(ctx, p, dim, 1, p->nb[1], (int64_t) 2 * (int64_t) dim * elem);
            fn = layer_norm(ctx, h, nullptr, nullptr, hp_.norm_eps);
            fn = adaln_apply(fn, shift, scale_ada);
        } else {
            fn = layer_norm(ctx, h, blk.ffn_norm_w, blk.ffn_norm_b, hp_.norm_eps);
        }

        ggml_tensor * ff1 = linear(ctx, fn, blk.w1, nullptr);
        ggml_tensor * ff3 = linear(ctx, fn, blk.w3, nullptr);
        ggml_tensor * ff = ggml_mul(ctx, ggml_silu(ctx, ff1), ff3);
        ff = linear(ctx, ff, blk.w2, nullptr);
        if (ffn_gate != nullptr) {
            ff = ggml_mul(ctx, ff, repeat_cuda_safe(ctx, ffn_gate, ff));
        }

        return ggml_add(ctx, h, ff);
    };

    auto resnet_block_eval = [&](ggml_tensor * x, const resnet_block & blk, int n_groups) -> ggml_tensor * {
        // Use [width, channels, batch] for conv path, and reshape to [width, 1, channels, batch]
        // for group norm because ggml_group_norm treats ne[2] as the channel axis.
        ggml_tensor * x3 = ggml_reshape_3d(ctx, x, x->ne[0], x->ne[1], x->ne[2]);
        auto group_norm_channels = [&](ggml_tensor * t3) -> ggml_tensor * {
            ggml_tensor * t4 = ggml_reshape_4d(ctx, t3, t3->ne[0], 1, t3->ne[1], t3->ne[2]);
            ggml_tensor * n4 = ggml_group_norm(ctx, t4, std::max(1, n_groups), hp_.group_norm_eps);
            return ggml_reshape_3d(ctx, n4, t3->ne[0], t3->ne[1], t3->ne[2]);
        };

        ggml_tensor * y = group_norm_channels(x3);
        ggml_tensor * n1w = ggml_reshape_3d(ctx, blk.norm1_w, 1, y->ne[1], 1);
        ggml_tensor * n1b = ggml_reshape_3d(ctx, blk.norm1_b, 1, y->ne[1], 1);
        y = ggml_mul(ctx, y, repeat_cuda_safe(ctx, n1w, y));
        y = ggml_add(ctx, y, repeat_cuda_safe(ctx, n1b, y));
        y = ggml_silu(ctx, y);
        ggml_tensor * conv1_w_f16 = blk.conv1_w->type == GGML_TYPE_F16
            ? blk.conv1_w
            : ggml_cast(ctx, blk.conv1_w, GGML_TYPE_F16);
        y = ggml_conv_1d_ph(ctx, conv1_w_f16, y, 1, 1);
        ggml_tensor * c1b = ggml_reshape_3d(ctx, blk.conv1_b, 1, y->ne[1], 1);
        y = ggml_add(ctx, y, repeat_cuda_safe(ctx, c1b, y));

        y = group_norm_channels(y);
        ggml_tensor * n2w = ggml_reshape_3d(ctx, blk.norm2_w, 1, y->ne[1], 1);
        ggml_tensor * n2b = ggml_reshape_3d(ctx, blk.norm2_b, 1, y->ne[1], 1);
        y = ggml_mul(ctx, y, repeat_cuda_safe(ctx, n2w, y));
        y = ggml_add(ctx, y, repeat_cuda_safe(ctx, n2b, y));
        y = ggml_silu(ctx, y);
        ggml_tensor * conv2_w_f16 = blk.conv2_w->type == GGML_TYPE_F16
            ? blk.conv2_w
            : ggml_cast(ctx, blk.conv2_w, GGML_TYPE_F16);
        y = ggml_conv_1d_ph(ctx, conv2_w_f16, y, 1, 1);
        ggml_tensor * c2b = ggml_reshape_3d(ctx, blk.conv2_b, 1, y->ne[1], 1);
        y = ggml_add(ctx, y, repeat_cuda_safe(ctx, c2b, y));

        return ggml_add(ctx, y, x3);
    };

    auto snake_beta_apply = [&](ggml_tensor * x_cl, ggml_tensor * alpha, ggml_tensor * beta) -> ggml_tensor * {
        ggml_tensor * a = ggml_reshape_2d(ctx, alpha, alpha->ne[0], 1);
        ggml_tensor * b = ggml_reshape_2d(ctx, beta, beta->ne[0], 1);
        a = ggml_exp(ctx, a);
        b = ggml_exp(ctx, b);

        ggml_tensor * ax = ggml_mul(ctx, x_cl, repeat_cuda_safe(ctx, a, x_cl));
        ggml_tensor * sin2 = ggml_sqr(ctx, ggml_sin(ctx, ax));
        ggml_tensor * denom = ggml_add1(ctx, b, scalar_eps);
        ggml_tensor * scaled = ggml_div(ctx, sin2, repeat_cuda_safe(ctx, denom, sin2));
        return ggml_add(ctx, x_cl, scaled);
    };

    auto choose_num_groups = [&](int channels) -> int {
        int g = std::max(1, std::min(hp_.resnet_groups, channels));
        while (g > 1 && (channels % g) != 0) {
            --g;
        }
        return std::max(1, g);
    };

    ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_tokens, "miocodec.inp_tokens");
    ggml_set_input(inp_tokens);

    ggml_tensor * pos_prenet = make_pos(n_tokens);
    ggml_tensor * mask_prenet = make_local_mask(n_tokens, hp_.prenet_window);

    ggml_tensor * x = ggml_get_rows(ctx, tok_embd_, inp_tokens); // [prenet_dim, n_tokens]
    for (int i = 0; i < hp_.prenet_layers; ++i) {
        x = transformer_block_eval(x, prenet_[i], hp_.prenet_heads, pos_prenet, mask_prenet, nullptr);
    }
    x = layer_norm(ctx, x, prenet_norm_w_, prenet_norm_b_, hp_.norm_eps);
    x = linear(ctx, x, prenet_out_w_, prenet_out_b_); // [decoder_dim, n_tokens]

    // ConvTranspose + interpolation path uses width-first layout [width, channels].
    ggml_tensor * y = ggml_cont(ctx, ggml_transpose(ctx, x)); // [n_tokens, decoder_dim]
    y = ggml_reshape_4d(ctx, y, y->ne[0], y->ne[1], 1, 1);
    y = ggml_conv_transpose_1d(ctx, upsample_w_, y, 2, 0, 1);
    ggml_tensor * up_b = ggml_reshape_2d(ctx, upsample_b_, 1, y->ne[1]);
    y = ggml_add(ctx, y, repeat_cuda_safe(ctx, up_b, y));
    y = ggml_interpolate(ctx, y, decoder_target_frames, y->ne[1], y->ne[2], y->ne[3], GGML_SCALE_MODE_BILINEAR);
    y = ggml_reshape_3d(ctx, y, y->ne[0], y->ne[1], y->ne[2]);

    if (hp_.model_type == 0) {
        for (int i = 0; i < hp_.resnet_blocks; ++i) {
            y = resnet_block_eval(y, prior_[i], choose_num_groups((int) y->ne[1]));
        }
    }

    ggml_tensor * pos_decoder = make_pos(decoder_target_frames);
    ggml_tensor * mask_decoder = make_local_mask(decoder_target_frames, hp_.decoder_window);

    ggml_tensor * y2 = ggml_reshape_2d(ctx, y, y->ne[0], y->ne[1]);
    x = ggml_cont(ctx, ggml_transpose(ctx, y2)); // [decoder_dim, target_frames]
    for (int i = 0; i < hp_.decoder_layers; ++i) {
            x = transformer_block_eval(x, decoder_[i], hp_.decoder_heads, pos_decoder, mask_decoder, decoder_condition_act);
        }
    if (decoder_condition != nullptr) {
        const int dim = (int) x->ne[0];
        ggml_tensor * p = linear(ctx, decoder_condition_act, decoder_norm_cond_w_, decoder_norm_cond_b_); // [2*dim, 1]
        const int64_t elem = (int64_t) p->nb[0];
        ggml_tensor * shift = ggml_view_2d(ctx, p, dim, 1, p->nb[1], 0);
        ggml_tensor * scale_ada = ggml_view_2d(ctx, p, dim, 1, p->nb[1], (int64_t) dim * elem);
        x = layer_norm(ctx, x, nullptr, nullptr, hp_.norm_eps);
        x = adaln_apply(x, shift, scale_ada);
    } else {
        x = layer_norm(ctx, x, decoder_norm_w_, decoder_norm_b_, hp_.norm_eps);
    }

    ggml_tensor * spec = nullptr;
    if (hp_.model_type == 0) {
        y = ggml_cont(ctx, ggml_transpose(ctx, x)); // [target_frames, decoder_dim]
        y = ggml_reshape_3d(ctx, y, y->ne[0], y->ne[1], 1);
        for (int i = 0; i < hp_.resnet_blocks; ++i) {
            y = resnet_block_eval(y, post_[i], choose_num_groups((int) y->ne[1]));
        }

        if (!wave_upsampler_.empty()) {
            for (size_t i = 0; i < wave_upsampler_.size(); ++i) {
                const auto & st = wave_upsampler_[i];
                const int stride = wave_upsampler_factors_[i];
                const int kernel = wave_upsampler_kernel_sizes_[i];
                const int pad = std::max(0, (kernel - stride) / 2);

                ggml_tensor * y4 = ggml_reshape_4d(ctx, y, y->ne[0], y->ne[1], y->ne[2], 1);
                ggml_tensor * up_w_f16 = st.up_w->type == GGML_TYPE_F16
                    ? st.up_w
                    : ggml_cast(ctx, st.up_w, GGML_TYPE_F16);
                y4 = ggml_conv_transpose_1d(ctx, up_w_f16, y4, stride, 0, 1);
                ggml_tensor * ub = ggml_reshape_2d(ctx, st.up_b, 1, y4->ne[1]);
                y4 = ggml_add(ctx, y4, repeat_cuda_safe(ctx, ub, y4));
                ggml_tensor * y3_up = ggml_reshape_3d(ctx, y4, y4->ne[0], y4->ne[1], y4->ne[2]);
                if (pad > 0) {
                    const int64_t out_w = y3_up->ne[0] - 2 * (int64_t) pad;
                    if (out_w <= 0) {
                        err = "wave upsampler crop became non-positive";
                        ggml_free(ctx);
                        return false;
                    }
                    y3_up = ggml_view_3d(
                            ctx,
                            y3_up,
                            out_w,
                            y3_up->ne[1],
                            y3_up->ne[2],
                            y3_up->nb[1],
                            y3_up->nb[2],
                            (size_t) pad * y3_up->nb[0]);
                    y3_up = ggml_cont(ctx, y3_up);
                }

                ggml_tensor * y2_up = ggml_reshape_2d(ctx, y3_up, y3_up->ne[0], y3_up->ne[1]); // [L, C]
                ggml_tensor * y_cl = ggml_cont(ctx, ggml_transpose(ctx, y2_up));       // [C, L]
                y_cl = snake_beta_apply(y_cl, st.snake_alpha, st.snake_beta);
                y2_up = ggml_cont(ctx, ggml_transpose(ctx, y_cl));                      // [L, C]
                y = ggml_reshape_3d(ctx, y2_up, y2_up->ne[0], y2_up->ne[1], 1);
                y = resnet_block_eval(y, st.resblk, choose_num_groups((int) y->ne[1]));
            }

            ggml_tensor * y2_proj = ggml_reshape_2d(ctx, y, y->ne[0], y->ne[1]);     // [L, C]
            ggml_tensor * y_cl_proj = ggml_cont(ctx, ggml_transpose(ctx, y2_proj));  // [C, L]
            x = linear(ctx, y_cl_proj, wave_upsampler_out_proj_w_, wave_upsampler_out_proj_b_);
            x = snake_beta_apply(x, wave_upsampler_out_snake_alpha_, wave_upsampler_out_snake_beta_);
        } else {
            y2 = ggml_reshape_2d(ctx, y, y->ne[0], y->ne[1]);
            x = ggml_cont(ctx, ggml_transpose(ctx, y2)); // [decoder_dim, target_frames]
        }
        spec = linear(ctx, x, istft_out_w_, istft_out_b_); // [n_fft + 2, target_frames]
        spec = ggml_cont(ctx, spec);
    } else {
        // mel path: direct decoder output projection + mel postnet
        spec = linear(ctx, x, istft_out_w_, istft_out_b_); // [n_mels, target_frames]
        spec = ggml_cont(ctx, spec);
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, false);
    ggml_set_name(spec, "miocodec.spec_out");
    ggml_build_forward_expand(gf, spec);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        err = "ggml_gallocr_alloc_graph failed";
        return false;
    }

    std::fprintf(stderr, "mio: decode gallocr buffer=%.1f MiB n_tokens=%d frames=%d\n",
                 (double) ggml_gallocr_get_buffer_size(galloc, 0) / (1024.0 * 1024.0),
                 n_tokens, decoder_target_frames);

    if (scalar_one->data != nullptr) {
        ((float *) scalar_one->data)[0] = 1.0f;
    }
    if (scalar_eps->data != nullptr) {
        ((float *) scalar_eps->data)[0] = 1e-9f;
    }
    std::memcpy(inp_tokens->data, tokens.data(), (size_t) n_tokens * sizeof(int32_t));
    if (decoder_condition != nullptr) {
        std::memcpy(decoder_condition->data, cond_vec.data(), cond_vec.size() * sizeof(float));
    }
    for (size_t pi = 0; pi < pending_pos.size(); ++pi) {
        const auto & p = pending_pos[pi];
        auto * data = (int32_t *) p.tensor->data;
        for (int i = 0; i < p.n; ++i) {
            data[i] = i;
        }
    }
    for (size_t mi = 0; mi < pending_masks.size(); ++mi) {
        const auto & m = pending_masks[mi];
        const int half = std::max(0, m.window / 2);
        if (m.fp16) {
            auto * data = (ggml_fp16_t *) m.tensor->data;
            const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
            const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
            for (int q = 0; q < m.n; ++q) {
                for (int k = 0; k < m.n; ++k) {
                    const bool allow = std::abs(k - q) <= half;
                    data[q * m.n + k] = allow ? zero : neg_inf;
                }
            }
        } else {
            auto * data = (float *) m.tensor->data;
            const float neg_inf = -INFINITY;
            const float zero = 0.0f;
            for (int q = 0; q < m.n; ++q) {
                for (int k = 0; k < m.n; ++k) {
                    const bool allow = std::abs(k - q) <= half;
                    data[q * m.n + k] = allow ? zero : neg_inf;
                }
            }
        }
    }

    const int n_bins = (int) spec->ne[0];
    n_frames = (int) spec->ne[1];

    uint64_t cache_key_decode = 0x4fd16a9f5c2b8d13ull;
    cache_key_decode = hash_mix_u64(cache_key_decode, (uint64_t) n_tokens);
    cache_key_decode = hash_mix_u64(cache_key_decode, (uint64_t) decoder_target_frames);
    cache_key_decode = hash_mix_u64(cache_key_decode, (uint64_t) stft_target_frames);
    cache_key_decode = hash_mix_u64(cache_key_decode, (uint64_t) wave_upsampler_total_factor_);
    cache_key_decode = hash_mix_u64(cache_key_decode, (uint64_t) (use_flash_attn ? 1 : 0));

    bool graph_ok = false;
    if (backend_gpu_ == nullptr && !backend_name_is_cpu(backend_device_name_)) {
        backend_gpu_ = init_gpu_backend(backend_device_name_);
    }
    ggml_backend_t backend_gpu = backend_gpu_;
    // Metal does not support GGML_OP_UPSCALE with BILINEAR mode
    // (used by ggml_interpolate in the decode graph).  Skip GPU for Metal.
    if (backend_gpu != nullptr) {
        const char * bname = ggml_backend_name(backend_gpu);
        if (bname != nullptr && std::strncmp(bname, "MTL", 3) == 0) {
            backend_gpu = nullptr;
        }
    }
    log_backend_once("decode", backend_gpu);
    if (backend_gpu != nullptr) {
        set_backend_threads(backend_gpu, std::max(1, n_threads));
        auto touch_lru = [&](uint64_t key) {
            auto itl = std::find(cache_decode_lru_.begin(), cache_decode_lru_.end(), key);
            if (itl != cache_decode_lru_.end()) {
                cache_decode_lru_.erase(itl);
            }
            cache_decode_lru_.push_back(key);
            cache_trim(cache_decode_, cache_decode_lru_, key);
        };

        auto it = cache_decode_.find(cache_key_decode);
        if (it == cache_decode_.end()) {
            gpu_graph_cache_entry entry;
            std::string copy_err;
            if (mio_ggml_backend_graph_copy_from_host(backend_gpu, gf, entry.copy, copy_err)) {
                entry.n0 = spec->ne[0];
                entry.n1 = spec->ne[1];
                auto ins = cache_decode_.emplace(cache_key_decode, std::move(entry));
                it = ins.first;
                touch_lru(cache_key_decode);
            } else {
                err = "mio_ggml_backend_graph_copy_from_host failed: " + copy_err;
            }
        } else {
            touch_lru(cache_key_decode);
        }

        if (it != cache_decode_.end()) {
            ggml_tensor * inp_gpu = mio_ggml_graph_copy_get_tensor(it->second.copy, "miocodec.inp_tokens");
            if (inp_gpu == nullptr || inp_gpu->type != GGML_TYPE_I32 ||
                ggml_nbytes(inp_gpu) != (size_t) n_tokens * sizeof(int32_t)) {
                err = "cached GPU graph input mismatch for decode tokens";
                mio_ggml_graph_copy_free(it->second.copy);
                cache_decode_.erase(it);
            } else {
                ggml_backend_tensor_set(inp_gpu, tokens.data(), 0, (size_t) n_tokens * sizeof(int32_t));

                if (hp_.dynamic_global) {
                    ggml_tensor * cond_gpu = mio_ggml_graph_copy_get_tensor(it->second.copy, "miocodec.decoder_condition");
                    if (cond_gpu == nullptr || cond_gpu->type != GGML_TYPE_F32 ||
                        ggml_nbytes(cond_gpu) != cond_vec.size() * sizeof(float)) {
                        err = "cached GPU graph input mismatch for decode embedding";
                        mio_ggml_graph_copy_free(it->second.copy);
                        cache_decode_.erase(it);
                        it = cache_decode_.end();
                    } else {
                        ggml_backend_tensor_set(cond_gpu, cond_vec.data(), 0, cond_vec.size() * sizeof(float));
                    }
                }

                if (it != cache_decode_.end()) {
                    const ggml_status st = ggml_backend_graph_compute(backend_gpu, it->second.copy.graph);
                    if (st == GGML_STATUS_SUCCESS) {
                        ggml_tensor * spec_gpu = mio_ggml_graph_copy_get_tensor(it->second.copy, "miocodec.spec_out");
                        if (spec_gpu != nullptr && spec_gpu->type == GGML_TYPE_F32) {
                            spec_out.resize((size_t) ggml_nelements(spec_gpu));
                            ggml_backend_tensor_get(spec_gpu, spec_out.data(), 0, spec_out.size() * sizeof(float));
                            graph_ok = true;
                        } else {
                            err = "failed to get GPU output tensor";
                        }
                    } else {
                        err = "ggml_backend_graph_compute failed on GPU backend";
                    }
                }
            }
        }
    }

    if (!graph_ok) {
        static bool logged_fallback = false;
        if (!logged_fallback) {
            logged_fallback = true;
            std::fprintf(stderr, "mio: decode fallback to CPU: %s\n", err.empty() ? "unknown" : err.c_str());
        }
        err.clear();
        ggml_cplan cplan = ggml_graph_plan(gf, std::max(1, n_threads), nullptr);
        std::vector<uint8_t> work_data;
        if (cplan.work_size > 0) {
            work_data.resize(cplan.work_size);
            cplan.work_data = work_data.data();
        }

        const ggml_status status = ggml_graph_compute(gf, &cplan);
        if (status != GGML_STATUS_SUCCESS) {
            err = "ggml_graph_compute failed";
            ggml_gallocr_free(galloc);
            ggml_free(ctx);
            return false;
        }

        if (spec->type != GGML_TYPE_F32) {
            err = "unexpected spec tensor type";
            ggml_gallocr_free(galloc);
            ggml_free(ctx);
            return false;
        }

        spec_out.resize((size_t) n_bins * (size_t) n_frames);
        std::memcpy(spec_out.data(), spec->data, spec_out.size() * sizeof(float));
    }

    if (spec_out.size() != (size_t) n_bins * (size_t) n_frames) {
        err = "unexpected spec tensor size";
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        return false;
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return true;
}

namespace {

struct fmap {
    int c = 0;
    int t = 0;
    std::vector<float> v;

    fmap() = default;
    fmap(int channels, int steps) : c(channels), t(steps), v((size_t) channels * (size_t) steps, 0.0f) {}

    float & at(int ch, int ti) { return v[(size_t) ch * (size_t) t + (size_t) ti]; }
    float at(int ch, int ti) const { return v[(size_t) ch * (size_t) t + (size_t) ti]; }
};

static bool tensor_is_f32_1d(const ggml_tensor * t) {
    return t != nullptr && t->type == GGML_TYPE_F32 && t->ne[0] > 0;
}

static bool tensor_is_f32_conv1d(const ggml_tensor * t) {
    return t != nullptr && t->type == GGML_TYPE_F32 && t->ne[0] > 0 && t->ne[1] > 0 && t->ne[2] > 0;
}

static inline float sinc1(float x) {
    if (std::abs(x) < 1e-12f) {
        return 1.0f;
    }
    return std::sin(x) / x;
}

static std::vector<float> make_hann_window(int n) {
    std::vector<float> w((size_t) std::max(0, n), 0.0f);
    if (n <= 1) {
        if (n == 1) {
            w[0] = 1.0f;
        }
        return w;
    }
    for (int i = 0; i < n; ++i) {
        w[(size_t) i] = 0.5f - 0.5f * std::cos(2.0f * k_pi * (float) i / (float) (n - 1));
    }
    return w;
}

static std::vector<float> make_julius_lowpass_kernel(float cutoff, float zeros = 8.0f) {
    if (cutoff <= 0.0f) {
        return {0.0f};
    }
    const int half_size = std::max(1, (int) (zeros / cutoff / 2.0f));
    const int kernel = 2 * half_size + 1;
    std::vector<float> window = make_hann_window(kernel);
    std::vector<float> filt((size_t) kernel, 0.0f);
    double sum = 0.0;
    for (int i = 0; i < kernel; ++i) {
        const int t = i - half_size;
        const float x = 2.0f * cutoff * k_pi * (float) t;
        const float v = 2.0f * cutoff * window[(size_t) i] * sinc1(x);
        filt[(size_t) i] = v;
        sum += v;
    }
    if (std::abs(sum) > 1e-12) {
        const float inv = (float) (1.0 / sum);
        for (float & f : filt) {
            f *= inv;
        }
    }
    return filt;
}

static fmap pad_replicate(const fmap & x, int left, int right) {
    const int out_t = x.t + left + right;
    fmap y(x.c, out_t);
    for (int ch = 0; ch < x.c; ++ch) {
        for (int t = 0; t < out_t; ++t) {
            int src = t - left;
            if (src < 0) {
                src = 0;
            } else if (src >= x.t) {
                src = x.t - 1;
            }
            y.at(ch, t) = x.at(ch, src);
        }
    }
    return y;
}

static fmap conv1d_same(
        const fmap & x,
        const ggml_tensor * w,
        const ggml_tensor * b,
        int dilation,
        int padding) {
    const int k = (int) w->ne[0];
    const int in_ch = (int) w->ne[1];
    const int out_ch = (int) w->ne[2];
    fmap y(out_ch, x.t);
    const auto * ww = (const float *) w->data;
    const auto * bb = b != nullptr ? (const float *) b->data : nullptr;

    for (int oc = 0; oc < out_ch; ++oc) {
        for (int t = 0; t < x.t; ++t) {
            float sum = bb != nullptr ? bb[oc] : 0.0f;
            for (int ic = 0; ic < in_ch; ++ic) {
                for (int kk = 0; kk < k; ++kk) {
                    const int ti = t + kk * dilation - padding;
                    if (ti < 0 || ti >= x.t) {
                        continue;
                    }
                    const size_t wi = (size_t) kk + (size_t) k * ((size_t) ic + (size_t) in_ch * (size_t) oc);
                    sum += x.at(ic, ti) * ww[wi];
                }
            }
            y.at(oc, t) = sum;
        }
    }
    return y;
}

static fmap conv1d_1x1(const fmap & x, const ggml_tensor * w, const ggml_tensor * b) {
    const int in_ch = (int) w->ne[1];
    const int out_ch = (int) w->ne[2];
    fmap y(out_ch, x.t);
    const auto * ww = (const float *) w->data;
    const auto * bb = b != nullptr ? (const float *) b->data : nullptr;

    for (int oc = 0; oc < out_ch; ++oc) {
        for (int t = 0; t < x.t; ++t) {
            float sum = bb != nullptr ? bb[oc] : 0.0f;
            for (int ic = 0; ic < in_ch; ++ic) {
                const size_t wi = (size_t) ic + (size_t) in_ch * (size_t) oc;
                sum += x.at(ic, t) * ww[wi];
            }
            y.at(oc, t) = sum;
        }
    }
    return y;
}

static fmap layer_norm_channels(
        const fmap & x,
        const ggml_tensor * w,
        const ggml_tensor * b,
        float eps) {
    fmap y(x.c, x.t);
    const bool has_w = tensor_is_f32_1d(w) && (int) w->ne[0] == x.c;
    const bool has_b = tensor_is_f32_1d(b) && (int) b->ne[0] == x.c;
    const auto * ww = has_w ? (const float *) w->data : nullptr;
    const auto * bb = has_b ? (const float *) b->data : nullptr;

    for (int t = 0; t < x.t; ++t) {
        float mean = 0.0f;
        for (int ch = 0; ch < x.c; ++ch) {
            mean += x.at(ch, t);
        }
        mean /= (float) std::max(1, x.c);

        float var = 0.0f;
        for (int ch = 0; ch < x.c; ++ch) {
            const float d = x.at(ch, t) - mean;
            var += d * d;
        }
        var /= (float) std::max(1, x.c);
        const float inv_std = 1.0f / std::sqrt(var + eps);

        for (int ch = 0; ch < x.c; ++ch) {
            float v = (x.at(ch, t) - mean) * inv_std;
            if (ww != nullptr) {
                v *= ww[ch];
            }
            if (bb != nullptr) {
                v += bb[ch];
            }
            y.at(ch, t) = v;
        }
    }
    return y;
}

static void tanh_inplace(fmap & x) {
    for (float & v : x.v) {
        v = std::tanh(v);
    }
}

static fmap conv1d_depthwise(const fmap & x, const std::vector<float> & filt, int stride) {
    const int k = (int) filt.size();
    const int out_t = (x.t - k) / stride + 1;
    fmap y(x.c, std::max(0, out_t));
    for (int ch = 0; ch < x.c; ++ch) {
        for (int t = 0; t < out_t; ++t) {
            float sum = 0.0f;
            const int base = t * stride;
            for (int kk = 0; kk < k; ++kk) {
                sum += x.at(ch, base + kk) * filt[(size_t) kk];
            }
            y.at(ch, t) = sum;
        }
    }
    return y;
}

static fmap lowpass_filter(const fmap & x, float cutoff, int stride) {
    const std::vector<float> filt = make_julius_lowpass_kernel(cutoff, 8.0f);
    if (filt.size() == 1 && filt[0] == 0.0f) {
        fmap z(x.c, (x.t + stride - 1) / stride);
        return z;
    }
    const int half = (int) filt.size() / 2;
    fmap xp = pad_replicate(x, half, half);
    return conv1d_depthwise(xp, filt, stride);
}

static fmap highpass_filter(const fmap & x, float cutoff) {
    fmap low = lowpass_filter(x, cutoff, 1);
    fmap y(x.c, x.t);
    for (int ch = 0; ch < x.c; ++ch) {
        for (int t = 0; t < x.t; ++t) {
            y.at(ch, t) = x.at(ch, t) - low.at(ch, t);
        }
    }
    return y;
}

static fmap upsample_activation(const fmap & x, const ggml_tensor * filter) {
    const int k = (int) filter->ne[0];
    const auto * f = (const float *) filter->data;
    const int ratio = 2;
    const int pad = k / ratio - 1;
    const int pad_left = pad * ratio + (k - ratio) / 2;
    const int pad_right = pad * ratio + (k - ratio + 1) / 2;

    fmap xp = pad_replicate(x, pad, pad);
    const int out_t_full = (xp.t - 1) * ratio + k;
    fmap y(x.c, out_t_full);
    for (int ch = 0; ch < x.c; ++ch) {
        for (int t = 0; t < xp.t; ++t) {
            const float v = xp.at(ch, t) * (float) ratio;
            const int o0 = t * ratio;
            for (int kk = 0; kk < k; ++kk) {
                y.at(ch, o0 + kk) += v * f[kk];
            }
        }
    }

    const int out_t = std::max(0, out_t_full - pad_left - pad_right);
    fmap z(x.c, out_t);
    for (int ch = 0; ch < x.c; ++ch) {
        for (int t = 0; t < out_t; ++t) {
            z.at(ch, t) = y.at(ch, t + pad_left);
        }
    }
    return z;
}

static fmap downsample_activation(const fmap & x, const ggml_tensor * filter) {
    const int k = (int) filter->ne[0];
    const auto * ff = (const float *) filter->data;
    std::vector<float> filt((size_t) k, 0.0f);
    for (int i = 0; i < k; ++i) {
        filt[(size_t) i] = ff[i];
    }
    const bool even = (k % 2 == 0);
    const int pad_left = k / 2 - (even ? 1 : 0);
    const int pad_right = k / 2;
    fmap xp = pad_replicate(x, pad_left, pad_right);
    return conv1d_depthwise(xp, filt, 2);
}

static fmap adaa_snake_beta(const fmap & x, const ggml_tensor * alpha_t, const ggml_tensor * beta_t) {
    const auto * alpha = (const float *) alpha_t->data;
    const auto * beta = (const float *) beta_t->data;
    fmap y(x.c, x.t);
    for (int ch = 0; ch < x.c; ++ch) {
        const float a = std::exp(alpha[ch]);
        const float b = std::exp(beta[ch]);
        const float inv = 1.0f / (2.0f * (b + 1e-9f));
        for (int t = 0; t < x.t; ++t) {
            const float cur = x.at(ch, t);
            const float prev = t > 0 ? x.at(ch, t - 1) : 0.0f;
            const float x_delta = cur - prev;
            const float x_sum = cur + prev;
            const float term = std::cos(a * x_sum) * sinc1(a * x_delta);
            y.at(ch, t) = x_sum * 0.5f + inv * (1.0f - term);
        }
    }
    return y;
}

static fmap activation1d(
        const fmap & x,
        const ggml_tensor * alpha,
        const ggml_tensor * beta,
        const ggml_tensor * up_filter,
        const ggml_tensor * down_filter) {
    fmap y = upsample_activation(x, up_filter);
    y = adaa_snake_beta(y, alpha, beta);
    y = downsample_activation(y, down_filter);
    return y;
}

} // namespace

bool miocodec_decoder::decode_mel_to_audio(
        const std::vector<float> & mel,
        int n_frames,
        int /*n_threads*/,
        std::vector<float> & audio_out,
        std::string & err) const {
    if (!is_loaded()) {
        err = "decoder not loaded";
        return false;
    }
    if (hp_.model_type != 1) {
        err = "decode_mel_to_audio is available only for mel-mode models";
        return false;
    }
    if (!hp_.has_vocoder) {
        err = "mel-mode model has no bundled MioVocoder tensors";
        return false;
    }
    if (!tensor_is_f32_conv1d(vocoder_conv_pre_.weight) || !tensor_is_f32_conv1d(vocoder_conv_post_.weight)) {
        err = "invalid vocoder conv tensors";
        return false;
    }
    if (n_frames <= 0 || (int) mel.size() != hp_.n_mels * n_frames) {
        err = "mel shape mismatch";
        return false;
    }

    fmap m(hp_.n_mels, n_frames);
    for (int t = 0; t < n_frames; ++t) {
        for (int c = 0; c < hp_.n_mels; ++c) {
            m.at(c, t) = mel[(size_t) c + (size_t) hp_.n_mels * (size_t) t];
        }
    }

    if (!mel_postnet_.empty()) {
        fmap r = m;
        for (size_t i = 0; i < mel_postnet_.size(); ++i) {
            const auto & l = mel_postnet_[i];
            const int k = (int) l.conv_w->ne[0];
            const int pad = std::max(0, (k - 1) / 2);
            r = conv1d_same(r, l.conv_w, l.conv_b, 1, pad);
            r = layer_norm_channels(r, l.norm_w, l.norm_b, hp_.norm_eps);
            if (i + 1 < mel_postnet_.size()) {
                tanh_inplace(r);
            }
        }
        if (r.c != m.c || r.t != m.t) {
            err = "mel postnet shape mismatch";
            return false;
        }
        for (size_t p = 0; p < m.v.size(); ++p) {
            m.v[p] += r.v[p];
        }
    }

    fmap x = conv1d_same(m, vocoder_conv_pre_.weight, vocoder_conv_pre_.bias, 1, 3);
    fmap x0 = x;

    std::vector<int> upps(vocoder_num_upsamples_, 1);
    int run = 1;
    for (int i = 0; i < vocoder_num_upsamples_; ++i) {
        run *= vocoder_upsample_rates_[(size_t) i];
        upps[(size_t) i] = run;
    }

    for (int i = 0; i < vocoder_num_upsamples_; ++i) {
        const int scale = vocoder_upsample_rates_[(size_t) i];
        const int upp = upps[(size_t) i];
        const auto & up = vocoder_ups_[(size_t) i];

        fmap y0(x0.c, x0.t * upp);
        for (int ch = 0; ch < x0.c; ++ch) {
            for (int t = 0; t < x0.t; ++t) {
                y0.at(ch, t * upp) = x0.at(ch, t);
            }
        }
        y0 = conv1d_same(y0, up.noise.weight, up.noise.bias, 1, 3);
        y0 = highpass_filter(y0, 0.5f / (float) scale);

        fmap y(x.c, x.t * scale);
        for (int ch = 0; ch < x.c; ++ch) {
            for (int t = 0; t < x.t; ++t) {
                y.at(ch, t * scale) = x.at(ch, t);
            }
        }
        y = lowpass_filter(y, 0.5f / (float) scale, 1);

        if (y.c != y0.c || y.t != y0.t) {
            err = "vocoder resample shape mismatch";
            return false;
        }
        for (size_t p = 0; p < y.v.size(); ++p) {
            y.v[p] += y0.v[p];
        }
        x = conv1d_1x1(y, up.after.weight, up.after.bias);

        fmap xs(x.c, x.t);
        for (int j = 0; j < vocoder_num_kernels_; ++j) {
            const auto & rb = vocoder_resblocks_[(size_t) (i * vocoder_num_kernels_ + j)];
            fmap r = x;
            static const int dilations[3] = {1, 3, 5};
            for (int k = 0; k < 3; ++k) {
                const fmap r_in = r;
                r = activation1d(
                        r,
                        rb.acts[2 * k + 0].alpha,
                        rb.acts[2 * k + 0].beta,
                        rb.acts[2 * k + 0].up_filter,
                        rb.acts[2 * k + 0].down_filter);
                const int k1 = (int) rb.convs1[k].weight->ne[0];
                r = conv1d_same(r, rb.convs1[k].weight, rb.convs1[k].bias, dilations[k], (k1 * dilations[k] - dilations[k]) / 2);
                r = activation1d(
                        r,
                        rb.acts[2 * k + 1].alpha,
                        rb.acts[2 * k + 1].beta,
                        rb.acts[2 * k + 1].up_filter,
                        rb.acts[2 * k + 1].down_filter);
                const int k2 = (int) rb.convs2[k].weight->ne[0];
                r = conv1d_same(r, rb.convs2[k].weight, rb.convs2[k].bias, 1, (k2 - 1) / 2);
                for (size_t p = 0; p < r.v.size(); ++p) {
                    r.v[p] += r_in.v[p];
                }
            }
            for (size_t p = 0; p < xs.v.size(); ++p) {
                xs.v[p] += r.v[p];
            }
        }
        const float inv_k = 1.0f / (float) std::max(1, vocoder_num_kernels_);
        for (float & v : xs.v) {
            v *= inv_k;
        }
        x = std::move(xs);
    }

    x = activation1d(
            x,
            vocoder_activation_post_.alpha,
            vocoder_activation_post_.beta,
            vocoder_activation_post_.up_filter,
            vocoder_activation_post_.down_filter);
    x = conv1d_same(x, vocoder_conv_post_.weight, nullptr, 1, 3);

    if (x.c != 1) {
        err = "vocoder output channel mismatch";
        return false;
    }
    audio_out.resize((size_t) x.t);
    for (int t = 0; t < x.t; ++t) {
        const float s = std::clamp(x.at(0, t), -1.0f, 1.0f);
        audio_out[(size_t) t] = s;
    }
    return true;
}
