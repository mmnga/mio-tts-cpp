#include "wavlm-extractor.h"
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
#include <numeric>

#define MINIAUDIO_IMPLEMENTATION
#define MA_NO_ENCODING
#define MA_NO_DEVICE_IO
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define MA_NO_GENERATION
#define MA_API static
#include "miniaudio/miniaudio.h"

namespace {

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

static ggml_backend_t init_gpu_backend(const std::string & preferred_backend_name) {
    ggml_backend_t backend_gpu = nullptr;
    if (!preferred_backend_name.empty()) {
        backend_gpu = ggml_backend_init_by_name(preferred_backend_name.c_str(), nullptr);
        if (backend_gpu != nullptr) {
            return backend_gpu;
        }
    }
    const char * env_backend_name = std::getenv("MIO_TTS_BACKEND_DEVICE");
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

static void log_backend_once(ggml_backend_t backend) {
    static bool logged = false;
    if (logged) {
        return;
    }
    logged = true;
    if (backend == nullptr) {
        std::fprintf(stderr, "mio: wavlm backend = CPU (GPU backend unavailable)\n");
        return;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    std::fprintf(
            stderr,
            "mio: wavlm backend = %s (%s)\n",
            ggml_backend_name(backend),
            dev ? ggml_backend_dev_description(dev) : "unknown device");
}

static inline uint64_t hash_mix_u64(uint64_t h, uint64_t v) {
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

static bool decode_audio_file_f32_mono(
        const std::string & path,
        int target_sample_rate,
        size_t max_frames,
        std::vector<float> & out,
        std::string & err) {
    ma_decoder_config decoder_config = ma_decoder_config_init(ma_format_f32, 1, target_sample_rate);
    ma_decoder decoder;
    ma_result result = ma_decoder_init_file(path.c_str(), &decoder_config, &decoder);
    if (result != MA_SUCCESS) {
        err = "ma_decoder_init_file failed";
        return false;
    }

    ma_uint64 frame_count = 0;
    result = ma_decoder_get_length_in_pcm_frames(&decoder, &frame_count);
    if (result != MA_SUCCESS || frame_count == 0) {
        ma_decoder_uninit(&decoder);
        err = "ma_decoder_get_length_in_pcm_frames failed";
        return false;
    }

    ma_uint64 frames_to_read = frame_count;
    if (max_frames > 0) {
        frames_to_read = std::min<ma_uint64>(frame_count, (ma_uint64) max_frames);
    }
    if (frames_to_read == 0) {
        ma_decoder_uninit(&decoder);
        err = "reference audio decode frame limit is zero";
        return false;
    }
    if (frames_to_read > (ma_uint64) std::numeric_limits<size_t>::max()) {
        ma_decoder_uninit(&decoder);
        err = "reference audio is too large";
        return false;
    }

    out.resize((size_t) frames_to_read);
    ma_uint64 frames_read = 0;
    result = ma_decoder_read_pcm_frames(&decoder, out.data(), frames_to_read, &frames_read);
    ma_decoder_uninit(&decoder);
    if (result != MA_SUCCESS || frames_read == 0) {
        err = "ma_decoder_read_pcm_frames failed";
        return false;
    }

    if (frames_read < frames_to_read) {
        out.resize((size_t) frames_read);
    }
    return true;
}

static void normalize_audio(std::vector<float> & audio) {
    float max_abs = 0.0f;
    for (float v : audio) {
        max_abs = std::max(max_abs, std::fabs(v));
    }
    max_abs += 1e-8f;
    if (max_abs > 0.0f) {
        for (float & v : audio) {
            v /= max_abs;
        }
    }
}

static std::vector<float> resample_linear(
        const std::vector<float> & input,
        int sample_rate_in,
        int sample_rate_out) {
    if (sample_rate_in == sample_rate_out || input.empty()) {
        return input;
    }

    const double ratio = (double) sample_rate_out / (double) sample_rate_in;
    const int64_t n_out = std::max<int64_t>(1, (int64_t) std::llround((double) input.size() * ratio));
    std::vector<float> output((size_t) n_out);

    for (int64_t i = 0; i < n_out; ++i) {
        const double pos = (double) i / ratio;
        const int64_t i0 = (int64_t) std::floor(pos);
        const int64_t i1 = std::min<int64_t>((int64_t) input.size() - 1, i0 + 1);
        const double t = pos - (double) i0;
        output[(size_t) i] = (float) ((1.0 - t) * (double) input[(size_t) std::max<int64_t>(0, i0)] +
                                      t * (double) input[(size_t) i1]);
    }

    return output;
}

static int64_t wavlm_conv_stack_out_len(int64_t n_in, const wavlm_extractor_params & hp) {
    int64_t n = std::max<int64_t>(1, n_in);
    for (int i = 0; i < 7; ++i) {
        const int64_t k = std::max<int64_t>(1, hp.conv_kernel[i]);
        const int64_t s = std::max<int64_t>(1, hp.conv_stride[i]);
        if (n < k) {
            n = 1;
            break;
        }
        n = (n - k) / s + 1;
        n = std::max<int64_t>(1, n);
    }
    return n;
}

static int wavlm_relative_position_bucket(int relative_pos, int num_buckets, int max_distance) {
    int relative_buckets = 0;

    const int half = num_buckets / 2;
    if (relative_pos > 0) {
        relative_buckets += half;
    }
    int n = std::abs(relative_pos);

    const int max_exact = half / 2;
    if (n < max_exact) {
        relative_buckets += n;
        return relative_buckets;
    }

    const float n_f = (float) n;
    const float max_exact_f = (float) max_exact;
    const float denom = std::log((float) max_distance / max_exact_f);
    int val = max_exact + (int) std::floor(std::log(n_f / max_exact_f) / denom * (half - max_exact));
    val = std::min(val, half - 1);
    relative_buckets += val;
    return relative_buckets;
}

} // namespace

wavlm_extractor::wavlm_extractor() = default;

wavlm_extractor::~wavlm_extractor() {
    cache_clear();
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

bool wavlm_extractor::is_loaded() const {
    return ctx_gguf_ != nullptr && ctx_weights_ != nullptr;
}

bool wavlm_extractor::set_backend_device(const std::string & backend_name, std::string & err) {
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

    cache_clear();
    if (backend_gpu_ != nullptr) {
        ggml_backend_free(backend_gpu_);
        backend_gpu_ = nullptr;
    }
    backend_gpu_ = backend_next;
    backend_device_name_ = backend_name;
    return true;
}

const wavlm_extractor_params & wavlm_extractor::params() const {
    return hp_;
}

int wavlm_extractor::estimate_ssl_frames(int source_sample_rate, float max_reference_seconds) const {
    if (source_sample_rate <= 0) {
        return 1;
    }
    const float sec = max_reference_seconds > 0.0f ? max_reference_seconds : 20.0f;
    const int64_t n_src = std::max<int64_t>(1, (int64_t) std::llround((double) source_sample_rate * (double) sec));
    const double ratio = (double) hp_.sample_rate / (double) source_sample_rate;
    const int64_t n_wav = std::max<int64_t>(1, (int64_t) std::llround((double) n_src * ratio));
    return (int) std::max<int64_t>(1, wavlm_conv_stack_out_len(n_wav, hp_));
}

size_t wavlm_extractor::estimate_workspace_bytes(int source_sample_rate, float max_reference_seconds) const {
    // With ggml_gallocr, the actual buffer is allocated dynamically with reuse.
    // This estimate is used only for reporting purposes (server startup log).
    // The gallocr approach dramatically reduces peak memory compared to no_alloc=false.
    const int frames = estimate_ssl_frames(source_sample_rate, max_reference_seconds);
    const int64_t seq_est = std::max<int64_t>(1, (int64_t) frames) + 16;
    const float sec = max_reference_seconds > 0.0f ? max_reference_seconds : 20.0f;
    const int src_sr = std::max(1, source_sample_rate);
    const int64_t n_src = std::max<int64_t>(1, (int64_t) std::llround((double) src_sr * (double) sec));
    const double ratio = (double) hp_.sample_rate / (double) src_sr;
    const int64_t n_wav = std::max<int64_t>(1, (int64_t) std::llround((double) n_src * ratio));

    const size_t embed = (size_t) std::max(1, hp_.embed_dim);
    const size_t heads = (size_t) std::max(1, hp_.n_heads);
    const size_t n_layers = (size_t) std::max(1, hp_.n_layers);
    const size_t seq = (size_t) seq_est;

    // Conv layer 0 dominates peak: output + norm intermediates ~ 8 * OL0 * 512 * 4.
    const int64_t ol0 = std::max<int64_t>(1, (n_wav - hp_.conv_kernel[0]) / hp_.conv_stride[0] + 1);
    size_t estimate = (size_t) ol0 * 512 * 4 * 8;

    // Positional conv im2col: seq * 768 * 128 * 2 (F16).
    estimate += seq * embed * 128 * 2;

    // Transformer layers: attention scores are the peak.
    estimate += n_layers * (20 * seq * embed * 4 + 4 * seq * seq * heads * 4);

    // Overhead.
    estimate += 32ull * 1024ull * 1024ull;

    return estimate;
}

void wavlm_extractor::reserve_workspace(int source_sample_rate, float max_reference_seconds) const {
    // With ggml_gallocr, the workspace vector is no longer used as the main arena.
    // This function is kept for API compatibility but is now a no-op.
    (void) source_sample_rate;
    (void) max_reference_seconds;
}

void wavlm_extractor::cache_trim(uint64_t keep_key) const {
    while (cache_gpu_.size() > k_gpu_graph_cache_limit && !cache_gpu_lru_.empty()) {
        const uint64_t k = cache_gpu_lru_.front();
        cache_gpu_lru_.erase(cache_gpu_lru_.begin());
        if (k == keep_key) {
            cache_gpu_lru_.push_back(k);
            continue;
        }
        auto it = cache_gpu_.find(k);
        if (it == cache_gpu_.end()) {
            continue;
        }
        mio_ggml_graph_copy_free(it->second.copy);
        cache_gpu_.erase(it);
    }
}

void wavlm_extractor::cache_clear() const {
    for (auto & kv : cache_gpu_) {
        mio_ggml_graph_copy_free(kv.second.copy);
    }
    cache_gpu_.clear();
    cache_gpu_lru_.clear();
}

bool wavlm_extractor::get_u32_kv(const char * key, uint32_t & out) const {
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

bool wavlm_extractor::get_f32_kv(const char * key, float & out) const {
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

bool wavlm_extractor::load(const std::string & path, std::string & err) {
    cache_clear();

    if (ctx_weights_ != nullptr) {
        ggml_free(ctx_weights_);
        ctx_weights_ = nullptr;
    }
    if (ctx_gguf_ != nullptr) {
        gguf_free(ctx_gguf_);
        ctx_gguf_ = nullptr;
    }

    layers_.clear();
    rel_embed_ = nullptr;

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
    if (get_u32_kv("wavlm.sample_rate", u32)) hp_.sample_rate = (int) u32;
    if (get_u32_kv("wavlm.n_layers", u32)) hp_.n_layers = (int) u32;
    if (get_u32_kv("wavlm.n_heads", u32)) hp_.n_heads = (int) u32;
    if (get_u32_kv("wavlm.head_dim", u32)) hp_.head_dim = (int) u32;
    if (get_u32_kv("wavlm.embed_dim", u32)) hp_.embed_dim = (int) u32;
    if (get_u32_kv("wavlm.num_buckets", u32)) hp_.num_buckets = (int) u32;
    if (get_u32_kv("wavlm.max_distance", u32)) hp_.max_distance = (int) u32;
    if (get_f32_kv("wavlm.layer_norm_eps", f32)) hp_.norm_eps = f32;
    for (int i = 0; i < 7; ++i) {
        if (get_u32_kv(("wavlm.feat.conv" + std::to_string(i) + ".kernel").c_str(), u32)) {
            hp_.conv_kernel[i] = (int) u32;
        }
        if (get_u32_kv(("wavlm.feat.conv" + std::to_string(i) + ".stride").c_str(), u32)) {
            hp_.conv_stride[i] = (int) u32;
        }
    }

    auto require_tensor = [&](const std::string & name) -> ggml_tensor * {
        ggml_tensor * t = ggml_get_tensor(ctx_weights_, name.c_str());
        if (t == nullptr) {
            err = "missing tensor: " + name;
        }
        return t;
    };

    conv0_norm_w_ = require_tensor("wavlm.feat.conv0.norm.weight");
    conv0_norm_b_ = require_tensor("wavlm.feat.conv0.norm.bias");
    for (int i = 0; i < 7; ++i) {
        conv_w_[i] = require_tensor("wavlm.feat.conv" + std::to_string(i) + ".weight");
    }

    proj_norm_w_ = require_tensor("wavlm.proj.norm.weight");
    proj_norm_b_ = require_tensor("wavlm.proj.norm.bias");
    proj_w_ = require_tensor("wavlm.proj.weight");
    proj_b_ = require_tensor("wavlm.proj.bias");

    pos_conv_w_ = require_tensor("wavlm.pos_conv.weight");
    pos_conv_b_ = require_tensor("wavlm.pos_conv.bias");

    transformer_norm_w_ = require_tensor("wavlm.transformer.norm.weight");
    transformer_norm_b_ = require_tensor("wavlm.transformer.norm.bias");

    rel_embed_ = require_tensor("wavlm.layer.0.attn.rel_embed.weight");

    layers_.resize(hp_.n_layers);
    for (int i = 0; i < hp_.n_layers; ++i) {
        auto & l = layers_[i];
        l.in_proj_w = require_tensor("wavlm.layer." + std::to_string(i) + ".attn.in_proj.weight");
        l.in_proj_b = require_tensor("wavlm.layer." + std::to_string(i) + ".attn.in_proj.bias");
        l.out_proj_w = require_tensor("wavlm.layer." + std::to_string(i) + ".attn.out_proj.weight");
        l.out_proj_b = require_tensor("wavlm.layer." + std::to_string(i) + ".attn.out_proj.bias");
        l.gru_w = require_tensor("wavlm.layer." + std::to_string(i) + ".attn.gru.weight");
        l.gru_b = require_tensor("wavlm.layer." + std::to_string(i) + ".attn.gru.bias");
        l.gru_const = require_tensor("wavlm.layer." + std::to_string(i) + ".attn.gru_const");
        l.norm1_w = require_tensor("wavlm.layer." + std::to_string(i) + ".norm1.weight");
        l.norm1_b = require_tensor("wavlm.layer." + std::to_string(i) + ".norm1.bias");
        l.ffn_w1 = require_tensor("wavlm.layer." + std::to_string(i) + ".ffn.w1.weight");
        l.ffn_b1 = require_tensor("wavlm.layer." + std::to_string(i) + ".ffn.w1.bias");
        l.ffn_w2 = require_tensor("wavlm.layer." + std::to_string(i) + ".ffn.w2.weight");
        l.ffn_b2 = require_tensor("wavlm.layer." + std::to_string(i) + ".ffn.w2.bias");
        l.norm2_w = require_tensor("wavlm.layer." + std::to_string(i) + ".norm2.weight");
        l.norm2_b = require_tensor("wavlm.layer." + std::to_string(i) + ".norm2.bias");
        if (!err.empty()) {
            return false;
        }
    }

    return err.empty();
}

bool wavlm_extractor::extract_ssl_features(
        const std::string & audio_path,
        int source_sample_rate,
        float max_reference_seconds,
        int n_threads,
        llama_flash_attn_type flash_attn_type,
        std::vector<float> & ssl_features_out,
        int & n_frames,
        std::string & err) const {
    if (!is_loaded()) {
        err = "wavlm extractor not loaded";
        return false;
    }

    if (source_sample_rate <= 0) {
        err = "invalid source sample rate";
        return false;
    }

    size_t max_src_samples = 0;
    if (max_reference_seconds > 0.0f) {
        max_src_samples = (size_t) std::max<int64_t>(
                1,
                (int64_t) std::llround((double) source_sample_rate * (double) max_reference_seconds));
    }

    std::vector<float> wav_src;
    if (!decode_audio_file_f32_mono(audio_path, source_sample_rate, max_src_samples, wav_src, err)) {
        err = "failed to decode reference audio: " + err;
        return false;
    }

    if (max_src_samples > 0 && wav_src.size() > max_src_samples) {
        wav_src.resize(max_src_samples);
    }

    if (!wav_src.empty() && max_src_samples > 0 && wav_src.capacity() > max_src_samples * 2) {
        // Large source files can leave oversized temporary capacity behind.
        // Compact early to keep reference-generation RSS stable on mobile.
        std::vector<float>(wav_src).swap(wav_src);
    }

    if (wav_src.empty()) {
        err = "reference audio is empty";
        return false;
    }

    normalize_audio(wav_src);

    // Resample directly to WavLM sample rate.
    // The previous reverse-conv padding path could dramatically over-estimate input
    // length and trigger impossible memory allocations for medium-length references.
    std::vector<float> wav16k = resample_linear(wav_src, source_sample_rate, hp_.sample_rate);
    {
        std::vector<float> tmp;
        wav_src.swap(tmp);
    }
    if (wav16k.empty()) {
        err = "resampling produced empty waveform";
        return false;
    }

    if (backend_gpu_ == nullptr && !backend_name_is_cpu(backend_device_name_)) {
        backend_gpu_ = init_gpu_backend(backend_device_name_);
    }
    ggml_backend_t backend_gpu = backend_gpu_;
    const char * backend_name = backend_gpu ? ggml_backend_name(backend_gpu) : nullptr;

    const bool flash_attn_requested = flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED;
    const bool backend_is_cuda = backend_name != nullptr && std::strncmp(backend_name, "CUDA", 4) == 0;
    // WavLM gated relative bias uses per-head masks; CUDA flash-attn currently only
    // supports masks with ne[2] == 1 and aborts otherwise.
    const bool use_flash_attn = flash_attn_requested && !(backend_is_cuda && hp_.n_heads > 1);
    if (flash_attn_requested && !use_flash_attn) {
        static bool logged_flash_fallback = false;
        if (!logged_flash_fallback) {
            logged_flash_fallback = true;
            std::fprintf(
                    stderr,
                    "mio: wavlm flash-attn disabled on CUDA for multi-head mask (n_heads=%d); using standard attention\n",
                    hp_.n_heads);
        }
    }

    const int64_t n_wav = (int64_t) wav16k.size();
    const int64_t seq_est = wavlm_conv_stack_out_len(n_wav, hp_) + 16;
    (void) seq_est;

    // Use no_alloc=true with ggml_gallocr for automatic buffer planning.
    // This avoids the fragile manual workspace estimation that caused OOM with
    // no_alloc=false (the conv stack alone creates ~1.3 GB of intermediates for
    // 20s audio when allocated linearly without reuse).
    const size_t n_tensors_est = 1024;
    const size_t ctx_size = n_tensors_est * ggml_tensor_overhead()
                          + ggml_graph_overhead_custom(4096, false)
                          + 64ull * 1024ull;
    ggml_init_params gparams = {
        /*.mem_size   = */ ctx_size,
        /*.mem_buffer = */ nullptr,
        /*.no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(gparams);
    if (ctx == nullptr) {
        err = "ggml_init failed";
        return false;
    }

    // Create input tensors (data filled after gallocr allocation).
    ggml_tensor * scalar_neg1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_tensor * scalar_two = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name(scalar_neg1, "scalar_neg1");
    ggml_set_name(scalar_two, "scalar_two");
    ggml_set_input(scalar_neg1);
    ggml_set_input(scalar_two);

    ggml_tensor * wav = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_wav);
    ggml_set_name(wav, "wavlm.wav_in");
    ggml_set_input(wav);
    ggml_tensor * x = ggml_reshape_3d(ctx, wav, n_wav, 1, 1); // [width, channels, batch]

    // Feature extractor conv stack.
    for (int i = 0; i < 7; ++i) {
        ggml_tensor * w = conv_w_[i]->type == GGML_TYPE_F16 ? conv_w_[i] : ggml_cast(ctx, conv_w_[i], GGML_TYPE_F16);
        x = ggml_conv_1d(ctx, w, x, hp_.conv_stride[i], 0, 1);

        if (i == 0) {
            // GroupNorm(num_groups=channels) -> normalize each channel across time.
            ggml_tensor * x4 = ggml_reshape_4d(ctx, x, x->ne[0], 1, x->ne[1], x->ne[2]);
            x4 = ggml_group_norm(ctx, x4, (int) x->ne[1], hp_.norm_eps);
            x = ggml_reshape_3d(ctx, x4, x->ne[0], x->ne[1], x->ne[2]);

            ggml_tensor * nw = ggml_reshape_3d(ctx, conv0_norm_w_, 1, x->ne[1], 1);
            ggml_tensor * nb = ggml_reshape_3d(ctx, conv0_norm_b_, 1, x->ne[1], 1);
            x = ggml_mul(ctx, x, repeat_cuda_safe(ctx, nw, x));
            x = ggml_add(ctx, x, repeat_cuda_safe(ctx, nb, x));
        }

        x = ggml_gelu(ctx, x);
    }

    // [width, channels, 1] -> [channels, width]
    ggml_tensor * x2 = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1]);
    x = ggml_cont(ctx, ggml_transpose(ctx, x2)); // [512, seq]
    x = layer_norm(ctx, x, proj_norm_w_, proj_norm_b_, hp_.norm_eps);
    x = linear(ctx, x, proj_w_, proj_b_); // [768, seq]

    // Positional conv (kernel=128, padding=64, groups=16), then GELU.
    ggml_tensor * xw = ggml_cont(ctx, ggml_transpose(ctx, x)); // [seq, 768]
    xw = ggml_reshape_3d(ctx, xw, xw->ne[0], xw->ne[1], 1);

    const int pos_groups = 16;
    const int pos_in_total = (int) xw->ne[1];
    const int pos_out_total = (int) pos_conv_w_->ne[2];
    if (pos_in_total % pos_groups != 0 || pos_out_total % pos_groups != 0) {
        ggml_free(ctx);
        err = "unexpected grouped positional-conv shape";
        return false;
    }

    const int in_per_group = pos_in_total / pos_groups;
    const int out_per_group = pos_out_total / pos_groups;
    ggml_tensor * pos = nullptr;
    for (int g = 0; g < pos_groups; ++g) {
        ggml_tensor * xg = ggml_view_3d(
                ctx,
                xw,
                xw->ne[0], in_per_group, 1,
                xw->nb[1], xw->nb[2],
                (int64_t) g * (int64_t) in_per_group * xw->nb[1]);
        ggml_tensor * wg = ggml_view_3d(
                ctx,
                pos_conv_w_,
                pos_conv_w_->ne[0], pos_conv_w_->ne[1], out_per_group,
                pos_conv_w_->nb[1], pos_conv_w_->nb[2],
                (int64_t) g * (int64_t) out_per_group * pos_conv_w_->nb[2]);
        xg = ggml_cont(ctx, xg);
        wg = ggml_cont(ctx, wg);
        if (wg->type != GGML_TYPE_F16) {
            wg = ggml_cast(ctx, wg, GGML_TYPE_F16);
        }
        ggml_tensor * pg = ggml_conv_1d(ctx, wg, xg, 1, 64, 1);
        pos = pos == nullptr ? pg : ggml_concat(ctx, pos, pg, 1);
    }

    ggml_tensor * pb = ggml_reshape_3d(ctx, pos_conv_b_, 1, pos->ne[1], 1);
    pos = ggml_add(ctx, pos, repeat_cuda_safe(ctx, pb, pos));
    if (pos->ne[0] > xw->ne[0]) {
        pos = ggml_view_3d(ctx, pos, xw->ne[0], pos->ne[1], pos->ne[2], pos->nb[1], pos->nb[2], 0);
        pos = ggml_cont(ctx, pos); // CUDA gelu requires contiguous input
    }
    pos = ggml_gelu(ctx, pos);
    ggml_tensor * pos2 = ggml_reshape_2d(ctx, pos, pos->ne[0], pos->ne[1]);
    pos2 = ggml_cont(ctx, ggml_transpose(ctx, pos2)); // [768, seq]

    x = ggml_add(ctx, x, pos2);
    x = layer_norm(ctx, x, transformer_norm_w_, transformer_norm_b_, hp_.norm_eps);
    ggml_tensor * ssl_pre = ggml_cont(ctx, x); // [embed_dim, seq] before Transformer stack

    const int seq = (int) x->ne[1];
    if (seq <= 0) {
        ggml_free(ctx);
        err = "unexpected empty SSL frame sequence";
        return false;
    }

    // Build raw relative position bias [seq, seq, heads, 1] from layer-0 embedding table.
    // Data is filled after gallocr allocation (see below).
    ggml_tensor * raw_bias = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, seq, seq, hp_.n_heads, 1);
    ggml_set_name(raw_bias, "raw_bias");
    ggml_set_input(raw_bias);

    const bool rel_layout_buckets_heads = rel_embed_->ne[0] == hp_.num_buckets && rel_embed_->ne[1] == hp_.n_heads;
    const bool rel_layout_heads_buckets = rel_embed_->ne[0] == hp_.n_heads && rel_embed_->ne[1] == hp_.num_buckets;
    if (!rel_layout_buckets_heads && !rel_layout_heads_buckets) {
        ggml_free(ctx);
        err = "unexpected wavlm rel_embed tensor shape";
        return false;
    }

    auto wavlm_layer_eval = [&](ggml_tensor * xin, const wavlm_layer & l) -> ggml_tensor * {
        const int dim = (int) xin->ne[0];
        const int seq_len = (int) xin->ne[1];
        const int n_heads = hp_.n_heads;
        const int head_dim = hp_.head_dim;
        const float scale = 1.0f / std::sqrt((float) head_dim);

        // Build gated relative bias mask:
        // gate from query input (before QKV projection), shape [1, seq, heads, 1]
        ggml_tensor * qh = ggml_reshape_3d(ctx, xin, head_dim, n_heads, seq_len);            // [64,12,seq]
        qh = ggml_reshape_2d(ctx, qh, head_dim, n_heads * seq_len);                           // [64,12*seq]
        ggml_tensor * g = linear(ctx, qh, l.gru_w, l.gru_b);                                  // [8,12*seq]
        const int64_t g_elem = (int64_t) g->nb[0];
        ggml_tensor * g0 = ggml_view_2d(ctx, g, 4, g->ne[1], g->nb[1], 0);
        ggml_tensor * g1 = ggml_view_2d(ctx, g, 4, g->ne[1], g->nb[1], 4 * g_elem);
        // CUDA SUM_ROWS requires a contiguous source tensor.
        g0 = ggml_sum_rows(ctx, ggml_cont(ctx, g0));                                           // [1,12*seq]
        g1 = ggml_sum_rows(ctx, ggml_cont(ctx, g1));                                           // [1,12*seq]
        g0 = ggml_sigmoid(ctx, g0);
        g1 = ggml_sigmoid(ctx, g1);
        g0 = ggml_cont(ctx, g0);
        g1 = ggml_cont(ctx, g1);
        g0 = ggml_reshape_4d(ctx, g0, 1, seq_len, n_heads, 1);
        g1 = ggml_reshape_4d(ctx, g1, 1, seq_len, n_heads, 1);
        ggml_tensor * gc = ggml_reshape_4d(ctx, l.gru_const, 1, 1, n_heads, 1);
        ggml_tensor * gate = ggml_mul(ctx, g1, repeat_cuda_safe(ctx, gc, g1));
        gate = ggml_add(ctx, gate, repeat_cuda_safe(ctx, scalar_neg1, gate));
        gate = ggml_mul(ctx, g0, gate);
        gate = ggml_add(ctx, gate, repeat_cuda_safe(ctx, scalar_two, gate));

        ggml_tensor * mask = ggml_mul(ctx, raw_bias, repeat_cuda_safe(ctx, gate, raw_bias));
        // ggml_flash_attn_ext CPU path is numerically safer with fp16 mask rows.
        // Using fp32 mask here can produce non-finite activations on some builds.
        if (use_flash_attn && mask->type == GGML_TYPE_F32) {
            mask = ggml_cast(ctx, mask, GGML_TYPE_F16);
        }

        // QKV attention
        ggml_tensor * qkv = linear(ctx, xin, l.in_proj_w, l.in_proj_b);                       // [3*dim, seq]
        qkv = ggml_cont(ctx, qkv);
        const int64_t qkv_elem = (int64_t) qkv->nb[0];
        ggml_tensor * q = ggml_view_2d(ctx, qkv, dim, seq_len, qkv->nb[1], 0);
        ggml_tensor * k = ggml_view_2d(ctx, qkv, dim, seq_len, qkv->nb[1], (int64_t) dim * qkv_elem);
        ggml_tensor * v = ggml_view_2d(ctx, qkv, dim, seq_len, qkv->nb[1], (int64_t) 2 * (int64_t) dim * qkv_elem);

        q = ggml_cont(ctx, q);
        k = ggml_cont(ctx, k);
        v = ggml_cont(ctx, v);
        q = ggml_reshape_4d(ctx, q, head_dim, n_heads, seq_len, 1);
        k = ggml_reshape_4d(ctx, k, head_dim, n_heads, seq_len, 1);
        v = ggml_reshape_4d(ctx, v, head_dim, n_heads, seq_len, 1);
        q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

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

        att = ggml_reshape_2d(ctx, att, dim, seq_len);
        ggml_tensor * att_out = linear(ctx, att, l.out_proj_w, l.out_proj_b);

        ggml_tensor * h = ggml_add(ctx, xin, att_out);
        ggml_tensor * n1 = layer_norm(ctx, h, l.norm1_w, l.norm1_b, hp_.norm_eps);
        ggml_tensor * ff = linear(ctx, n1, l.ffn_w1, l.ffn_b1);
        ff = ggml_gelu(ctx, ff);
        ff = linear(ctx, ff, l.ffn_w2, l.ffn_b2);

        // Match torchaudio WavLM (layer_norm_first = false):
        // h = x + attn(x); n1 = LN(h); out = LN2(h + FFN(n1))
        return layer_norm(ctx, ggml_add(ctx, h, ff), l.norm2_w, l.norm2_b, hp_.norm_eps);
    };

    ggml_tensor * ssl_sum = nullptr;
    for (int i = 0; i < hp_.n_layers; ++i) {
        x = wavlm_layer_eval(x, layers_[i]);
        if (ssl_sum == nullptr) {
            ssl_sum = x;
        } else {
            ssl_sum = ggml_add(ctx, ssl_sum, x);
        }
    }

    const float inv_layers = 1.0f / (float) std::max(1, hp_.n_layers);
    ggml_tensor * ssl = ggml_scale(ctx, ssl_sum, inv_layers); // [embed_dim, seq]
    ssl = ggml_cont(ctx, ssl);

    ggml_set_name(ssl, "wavlm.ssl");
    ggml_set_name(ssl_pre, "wavlm.ssl_pre");
    ggml_set_output(ssl);
    ggml_set_output(ssl_pre);

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(gf, ssl);
    ggml_build_forward_expand(gf, ssl_pre);

    // Allocate tensor data with ggml_gallocr (automatic buffer reuse).
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        err = "ggml_gallocr_alloc_graph failed";
        return false;
    }

    std::fprintf(stderr, "mio: wavlm gallocr buffer=%.1f MiB n_wav=%lld seq=%d\n",
                 (double) ggml_gallocr_get_buffer_size(galloc, 0) / (1024.0 * 1024.0),
                 (long long) n_wav, seq);

    // Fill input tensor data (deferred from graph construction due to no_alloc=true).
    ((float *) scalar_neg1->data)[0] = -1.0f;
    ((float *) scalar_two->data)[0] = 2.0f;
    std::memcpy(wav->data, wav16k.data(), wav16k.size() * sizeof(float));

    // Fill raw relative position bias.
    {
        auto * rb = (float *) raw_bias->data;
        const auto * re = (const float *) rel_embed_->data;
        for (int q = 0; q < seq; ++q) {
            for (int k = 0; k < seq; ++k) {
                const int bucket = wavlm_relative_position_bucket(k - q, hp_.num_buckets, hp_.max_distance);
                for (int h = 0; h < hp_.n_heads; ++h) {
                    float v = 0.0f;
                    if (rel_layout_buckets_heads) {
                        v = re[bucket + rel_embed_->ne[0] * h];
                    } else {
                        v = re[h + rel_embed_->ne[0] * bucket];
                    }
                    rb[k + seq * q + (size_t) seq * (size_t) seq * (size_t) h] = v;
                }
            }
        }
    }

    std::vector<float> ssl_pre_features;
    n_frames = (int) ssl->ne[1];

    uint64_t cache_key = 0x91a4e72bc53df109ull;
    cache_key = hash_mix_u64(cache_key, (uint64_t) n_wav);
    cache_key = hash_mix_u64(cache_key, (uint64_t) (use_flash_attn ? 1 : 0));

    bool graph_ok = false;
    log_backend_once(backend_gpu);
    if (backend_gpu != nullptr) {
        set_backend_threads(backend_gpu, std::max(1, n_threads));
        auto touch_lru = [&](uint64_t key) {
            auto itl = std::find(cache_gpu_lru_.begin(), cache_gpu_lru_.end(), key);
            if (itl != cache_gpu_lru_.end()) {
                cache_gpu_lru_.erase(itl);
            }
            cache_gpu_lru_.push_back(key);
            cache_trim(key);
        };

        auto it = cache_gpu_.find(cache_key);
        if (it == cache_gpu_.end()) {
            gpu_graph_cache_entry entry;
            std::string copy_err;
            if (mio_ggml_backend_graph_copy_from_host(backend_gpu, gf, entry.copy, copy_err)) {
                entry.n0 = ssl->ne[0];
                entry.n1 = ssl->ne[1];
                auto ins = cache_gpu_.emplace(cache_key, std::move(entry));
                it = ins.first;
                touch_lru(cache_key);
            } else {
                err = "mio_ggml_backend_graph_copy_from_host failed: " + copy_err;
            }
        } else {
            touch_lru(cache_key);
        }

        if (it != cache_gpu_.end()) {
            ggml_tensor * wav_gpu = mio_ggml_graph_copy_get_tensor(it->second.copy, "wavlm.wav_in");
            if (wav_gpu == nullptr || wav_gpu->type != GGML_TYPE_F32 ||
                ggml_nbytes(wav_gpu) != wav16k.size() * sizeof(float)) {
                err = "cached GPU graph input mismatch for wavlm input";
                mio_ggml_graph_copy_free(it->second.copy);
                cache_gpu_.erase(it);
            } else {
                ggml_backend_tensor_set(wav_gpu, wav16k.data(), 0, wav16k.size() * sizeof(float));
                const ggml_status st = ggml_backend_graph_compute(backend_gpu, it->second.copy.graph);
                if (st == GGML_STATUS_SUCCESS) {
                    ggml_tensor * ssl_gpu = mio_ggml_graph_copy_get_tensor(it->second.copy, "wavlm.ssl");
                    ggml_tensor * ssl_pre_gpu = mio_ggml_graph_copy_get_tensor(it->second.copy, "wavlm.ssl_pre");
                    if (ssl_gpu != nullptr && ssl_pre_gpu != nullptr &&
                        ssl_gpu->type == GGML_TYPE_F32 && ssl_pre_gpu->type == GGML_TYPE_F32) {
                        ssl_features_out.resize((size_t) ggml_nelements(ssl_gpu));
                        ggml_backend_tensor_get(ssl_gpu, ssl_features_out.data(), 0, ssl_features_out.size() * sizeof(float));
                        ssl_pre_features.resize((size_t) ggml_nelements(ssl_pre_gpu));
                        ggml_backend_tensor_get(ssl_pre_gpu, ssl_pre_features.data(), 0, ssl_pre_features.size() * sizeof(float));
                        graph_ok = true;
                    } else {
                        err = "failed to get GPU output tensors";
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
            std::fprintf(stderr, "mio: wavlm fallback to CPU: %s\n", err.empty() ? "unknown" : err.c_str());
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

        if (ssl->type != GGML_TYPE_F32 || ssl_pre->type != GGML_TYPE_F32) {
            err = "unexpected ssl tensor type";
            ggml_gallocr_free(galloc);
            ggml_free(ctx);
            return false;
        }

        ssl_features_out.resize((size_t) ssl->ne[0] * (size_t) ssl->ne[1]);
        std::memcpy(ssl_features_out.data(), ssl->data, ssl_features_out.size() * sizeof(float));
        ssl_pre_features.resize((size_t) ssl_pre->ne[0] * (size_t) ssl_pre->ne[1]);
        std::memcpy(ssl_pre_features.data(), ssl_pre->data, ssl_pre_features.size() * sizeof(float));
    }

    bool has_non_finite = false;
    for (float x : ssl_features_out) {
        if (!std::isfinite(x)) {
            has_non_finite = true;
            break;
        }
    }

    if (has_non_finite) {
        bool pre_finite = !ssl_pre_features.empty();
        for (size_t i = 0; pre_finite && i < ssl_pre_features.size(); ++i) {
            if (!std::isfinite(ssl_pre_features[i])) {
                pre_finite = false;
                break;
            }
        }

        if (pre_finite) {
            n_frames = (int) ssl_pre->ne[1];
            ssl_features_out = std::move(ssl_pre_features);
            err = "wavlm transformer path produced non-finite features; used pre-transformer fallback";
        }

        if (!pre_finite) {
        const int embed = std::max(1, hp_.embed_dim);
        const int hop = 320;
        const int win = 400;
        const int total = (int) wav16k.size();
        const int fb_frames = std::max(1, (total + hop - 1) / hop);

        ssl_features_out.assign((size_t) embed * (size_t) fb_frames, 0.0f);
        for (int t = 0; t < fb_frames; ++t) {
            const int s0 = std::min(total - 1, t * hop);
            const int s1 = std::min(total, s0 + win);

            double sum_abs = 0.0;
            double sum_sq = 0.0;
            double peak = 0.0;
            const int n = std::max(1, s1 - s0);
            for (int i = s0; i < s1; ++i) {
                const double a = std::abs((double) wav16k[(size_t) i]);
                sum_abs += a;
                sum_sq += a * a;
                peak = std::max(peak, a);
            }

            const double mean_abs = sum_abs / (double) n;
            const double rms = std::sqrt(sum_sq / (double) n);
            for (int d = 0; d < embed; ++d) {
                const double alpha = 0.15 + 0.85 * ((double) (d % 31) / 30.0);
                const double beta = (double) ((d % 7) + 1) / 7.0;
                double v = (alpha * rms + (1.0 - alpha) * mean_abs) * beta;
                v += 0.05 * peak * ((d & 1) ? 1.0 : -1.0);
                ssl_features_out[(size_t) d + (size_t) embed * (size_t) t] = (float) v;
            }
        }

        n_frames = fb_frames;
        err = "wavlm extractor produced non-finite features; used audio-stat fallback";
        }
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return true;
}
