#define _USE_MATH_DEFINES

#include "mio-tts-lib.h"

#include "ggml.h"
#include "gguf.h"
#include "miocodec-decoder.h"
#include "wavlm-extractor.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

static constexpr const char * k_mio_embedding_tensor_name = "mio.global_embedding";
static constexpr int32_t k_mio_code_min = 0;
static constexpr int32_t k_mio_code_max = 12799;

struct mio_tts_model_shared {
    std::string vocoder_path;
    std::string wavlm_path;
    miocodec_decoder decoder;
    wavlm_extractor wavlm;
    mutable std::mutex runtime_mutex;
};

struct mio_tts_context_impl {
    std::shared_ptr<mio_tts_model_shared> model;
};

struct mio_tts_vocab_map_impl {
    std::unordered_map<llama_token, int32_t> token_to_code;
};

struct wav_header {
    char riff[4] = {'R', 'I', 'F', 'F'};
    uint32_t chunk_size = 0;
    char wave[4] = {'W', 'A', 'V', 'E'};
    char fmt[4] = {'f', 'm', 't', ' '};
    uint32_t fmt_chunk_size = 16;
    uint16_t audio_format = 1;
    uint16_t num_channels = 1;
    uint32_t sample_rate = 0;
    uint32_t byte_rate = 0;
    uint16_t block_align = 0;
    uint16_t bits_per_sample = 16;
    char data[4] = {'d', 'a', 't', 'a'};
    uint32_t data_size = 0;
};

static void set_err(char * err, size_t err_size, const std::string & msg) {
    if (err != nullptr && err_size > 0) {
        std::snprintf(err, err_size, "%s", msg.c_str());
    }
}

static int resolve_threads(int n_threads) {
    if (n_threads > 0) {
        return n_threads;
    }
    const unsigned hc = std::thread::hardware_concurrency();
    return (int) (hc > 0 ? hc : 1);
}

static std::string strip_ascii_punct(std::string s) {
    while (!s.empty() && std::ispunct((unsigned char) s.front()) && s.front() != '<' && s.front() != '-') {
        s.erase(s.begin());
    }
    while (!s.empty() && std::ispunct((unsigned char) s.back()) && s.back() != '>' && !std::isdigit((unsigned char) s.back())) {
        s.pop_back();
    }
    return s;
}

static bool parse_mio_code_token(const std::string & raw, int32_t & out) {
    std::string token = strip_ascii_punct(raw);
    if (token.empty()) {
        return false;
    }

    if (token.rfind("<|s_", 0) == 0 && token.size() > 6 && token.substr(token.size() - 2) == "|>") {
        token = token.substr(4, token.size() - 6);
    }

    try {
        size_t used = 0;
        const long long v = std::stoll(token, &used, 10);
        if (used != token.size()) {
            return false;
        }
        if (v < (long long) std::numeric_limits<int32_t>::min() || v > (long long) std::numeric_limits<int32_t>::max()) {
            return false;
        }
        out = (int32_t) v;
        return true;
    } catch (...) {
        return false;
    }
}

static bool parse_mio_codec_piece(const std::string & piece, int32_t & code) {
    if (piece.size() < 6) {
        return false;
    }
    if (piece.compare(0, 4, "<|s_") != 0) {
        return false;
    }
    if (piece[piece.size() - 2] != '|' || piece.back() != '>') {
        return false;
    }

    const std::string number_str = piece.substr(4, piece.size() - 6);
    if (number_str.empty()) {
        return false;
    }

    char * end = nullptr;
    const long value = std::strtol(number_str.c_str(), &end, 10);
    if (end == nullptr || *end != '\0' || value < 0 || value > std::numeric_limits<int32_t>::max()) {
        return false;
    }

    code = (int32_t) value;
    return true;
}

static bool token_to_piece(const llama_vocab * vocab, llama_token token, std::string & out) {
    out.clear();

    int cap = 32;
    for (int i = 0; i < 8; ++i) {
        std::vector<char> buf((size_t) cap);
        const int n = llama_token_to_piece(vocab, token, buf.data(), (int32_t) buf.size(), 0, true);
        if (n >= 0) {
            out.assign(buf.data(), (size_t) n);
            return true;
        }
        cap = std::max(cap * 2, -n + 1);
    }

    return false;
}

static std::optional<llama_token> try_get_single_token_id(const llama_vocab * vocab, const std::string & token_str) {
    const int n = llama_tokenize(vocab, token_str.c_str(), (int32_t) token_str.size(), nullptr, 0, false, true);
    if (n >= 0) {
        return std::nullopt;
    }
    const int needed = -n;
    if (needed != 1) {
        return std::nullopt;
    }

    llama_token tok = LLAMA_TOKEN_NULL;
    if (llama_tokenize(vocab, token_str.c_str(), (int32_t) token_str.size(), &tok, 1, false, true) != 1) {
        return std::nullopt;
    }

    std::string piece;
    if (!token_to_piece(vocab, tok, piece) || piece != token_str) {
        return std::nullopt;
    }

    return tok;
}

static bool build_mio_audio_token_map(
        const llama_vocab * vocab,
        std::unordered_map<llama_token, int32_t> & out,
        std::string & err) {
    out.clear();

    const auto mio_start = try_get_single_token_id(vocab, "<|s_0|>");
    const auto mio_end   = try_get_single_token_id(vocab, "<|s_12799|>");
    if (!mio_start.has_value() || !mio_end.has_value()) {
        err = "MioTTS audio token range not found (<|s_0|>..<|s_12799|>)";
        return false;
    }

    if (*mio_start > *mio_end) {
        err = "invalid MioTTS token range";
        return false;
    }

    for (llama_token token_id = *mio_start; token_id <= *mio_end; ++token_id) {
        std::string piece;
        if (!token_to_piece(vocab, token_id, piece)) {
            continue;
        }

        int32_t code = -1;
        if (!parse_mio_codec_piece(piece, code)) {
            continue;
        }

        if (code < k_mio_code_min || code > k_mio_code_max) {
            continue;
        }

        out[token_id] = code;
    }

    if (out.empty()) {
        err = "failed to build MioTTS token->code map";
        return false;
    }

    return true;
}

static bool save_codes_text(const char * path, const int32_t * codes, size_t n_codes, std::string & err) {
    if (path == nullptr || path[0] == '\0') {
        err = "codes output path is empty";
        return false;
    }
    if (codes == nullptr || n_codes == 0) {
        err = "codes are empty";
        return false;
    }

    std::ofstream file(path, std::ios::trunc);
    if (!file) {
        err = "failed to open codes output file";
        return false;
    }

    for (size_t i = 0; i < n_codes; ++i) {
        file << codes[i] << "\n";
    }

    if (!file.good()) {
        err = "failed while writing codes output file";
        return false;
    }

    return true;
}

static bool load_codes_text(const char * path, std::vector<int32_t> & out, std::string & err) {
    if (path == nullptr || path[0] == '\0') {
        err = "codes input path is empty";
        return false;
    }

    std::ifstream file(path);
    if (!file) {
        err = "failed to open codes input file";
        return false;
    }

    out.clear();
    std::string tok;
    while (file >> tok) {
        int32_t code = 0;
        if (!parse_mio_code_token(tok, code)) {
            err = "failed to parse code token: " + tok;
            return false;
        }
        if (code < k_mio_code_min || code > k_mio_code_max) {
            err = "code id out of range in input file";
            return false;
        }
        out.push_back(code);
    }

    if (out.empty()) {
        err = "codes input file is empty";
        return false;
    }

    return true;
}

static bool save_embedding_gguf_vec(const char * path, const std::vector<float> & emb, std::string & err) {
    if (path == nullptr || path[0] == '\0') {
        err = "embedding output path is empty";
        return false;
    }
    if (emb.empty()) {
        err = "embedding is empty";
        return false;
    }
    if (emb.size() > (size_t) std::numeric_limits<int64_t>::max()) {
        err = "embedding size is too large";
        return false;
    }

    struct gguf_context * ctx_gguf = gguf_init_empty();
    if (ctx_gguf == nullptr) {
        err = "failed to initialize GGUF context";
        return false;
    }

    gguf_set_val_str(ctx_gguf, "general.architecture", "mio-embedding");
    gguf_set_val_str(ctx_gguf, "general.type", "embedding");
    gguf_set_val_u32(ctx_gguf, "mio.embedding.dim", (uint32_t) emb.size());

    struct ggml_init_params ggml_params = {
        /*.mem_size   =*/ std::max<size_t>(1u << 20, ggml_tensor_overhead() * 4 + emb.size() * sizeof(float) + 1024),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx_data = ggml_init(ggml_params);
    if (ctx_data == nullptr) {
        gguf_free(ctx_gguf);
        err = "failed to initialize GGML context for embedding tensor";
        return false;
    }

    struct ggml_tensor * tensor = ggml_new_tensor_1d(ctx_data, GGML_TYPE_F32, (int64_t) emb.size());
    if (tensor == nullptr || tensor->data == nullptr) {
        ggml_free(ctx_data);
        gguf_free(ctx_gguf);
        err = "failed to create embedding tensor";
        return false;
    }

    ggml_set_name(tensor, k_mio_embedding_tensor_name);
    std::memcpy(tensor->data, emb.data(), emb.size() * sizeof(float));

    gguf_add_tensor(ctx_gguf, tensor);
    const bool ok = gguf_write_to_file(ctx_gguf, path, false);

    ggml_free(ctx_data);
    gguf_free(ctx_gguf);

    if (!ok) {
        err = "failed to write embedding GGUF";
        return false;
    }

    return true;
}

static bool load_embedding_gguf_vec(const char * path, std::vector<float> & emb, std::string & err) {
    if (path == nullptr || path[0] == '\0') {
        err = "embedding input path is empty";
        return false;
    }

    struct ggml_context * ctx_data = nullptr;
    struct gguf_init_params gguf_params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ &ctx_data,
    };
    struct gguf_context * ctx_gguf = gguf_init_from_file(path, gguf_params);
    if (ctx_gguf == nullptr) {
        err = "failed to open embedding GGUF";
        return false;
    }

    auto cleanup = [&]() {
        if (ctx_data != nullptr) {
            ggml_free(ctx_data);
            ctx_data = nullptr;
        }
        gguf_free(ctx_gguf);
        ctx_gguf = nullptr;
    };

    int64_t tensor_id = gguf_find_tensor(ctx_gguf, k_mio_embedding_tensor_name);
    if (tensor_id < 0) {
        const int64_t n_tensors = gguf_get_n_tensors(ctx_gguf);
        if (n_tensors == 1) {
            tensor_id = 0;
        } else {
            cleanup();
            err = "embedding GGUF is missing tensor 'mio.global_embedding'";
            return false;
        }
    }

    const char * tensor_name = gguf_get_tensor_name(ctx_gguf, tensor_id);
    struct ggml_tensor * tensor = ctx_data != nullptr ? ggml_get_tensor(ctx_data, tensor_name) : nullptr;
    if (tensor == nullptr || tensor->data == nullptr) {
        cleanup();
        err = "failed to load embedding tensor data from GGUF";
        return false;
    }

    if (tensor->type != GGML_TYPE_F32) {
        cleanup();
        err = "embedding tensor must be f32";
        return false;
    }

    const int64_t n = ggml_nelements(tensor);
    if (n <= 0) {
        cleanup();
        err = "embedding tensor is empty";
        return false;
    }

    emb.assign((size_t) n, 0.0f);
    std::memcpy(emb.data(), tensor->data, emb.size() * sizeof(float));

    cleanup();
    return true;
}

static void fill_hann_window(int length, bool periodic, float * output) {
    int offset = periodic ? 0 : -1;
    for (int i = 0; i < length; i++) {
        output[i] = 0.5f * (1.0f - cosf((2.0f * (float) M_PI * (float) i) / (float) (length + offset)));
    }
}

// ---------------------------------------------------------------------------
// Row-column FFT-based IRFFT.
//
// For n = 1920 = 15 * 128 the naive O(n * n_freq) DFT required a 14 MB
// twiddle table and ~1.8 M multiply-adds per frame.  The row-column
// decomposition factors n into an odd part N1 (=15) and a power-of-2 part
// N2 (=128) and computes:
//   1.  128 naive DFTs of size 15  (column DFTs)         ~28 K ops
//   2.  twiddle-factor multiply                           ~2 K ops
//   3.  15 radix-2 FFTs of size 128  (row FFTs)          ~13 K ops
// Total twiddle storage: ~20 KB  (vs 14 MB before).
// ---------------------------------------------------------------------------

struct irfft_plan {
    int n       = 0;   // FFT size  (e.g. 1920)
    int n_freq  = 0;   // n/2 + 1   (e.g. 961)
    int n1      = 0;   // odd factor (e.g. 15)
    int n2      = 0;   // power-of-2 factor (e.g. 128)
    int log2_n2 = 0;   // log2(n2)  (e.g. 7)

    // DFT-N1 twiddle matrix (positive-exponent / IDFT direction).
    // tw_n1[2*(k*n1+j) + 0/1] = cos/sin(+2 pi k j / n1)
    std::vector<float> tw_n1;   // 2 * n1 * n1

    // Row-column twiddle factors.
    // tw_rc[2*(row*n2+col) + 0/1] = cos/sin(+2 pi row col / n)
    std::vector<float> tw_rc;   // 2 * n1 * n2

    // Radix-2 butterfly twiddles (positive exponent).
    // Stage s: half = 2^s, entry at tw_fft[2*(half-1+j) + 0/1].
    std::vector<float> tw_fft;  // 2 * (n2 - 1)

    // Per-call workspace size requirement: 2*n + 2*n1 floats.
    int work_floats = 0;
};

static irfft_plan make_irfft_plan(int n) {
    irfft_plan plan;
    if (n <= 0) {
        return plan;
    }

    plan.n      = n;
    plan.n_freq = n / 2 + 1;

    // Factor n = n1 * n2, n2 = largest power-of-2 factor.
    plan.n2      = 1;
    plan.log2_n2 = 0;
    int tmp = n;
    while ((tmp & 1) == 0) {
        plan.n2 <<= 1;
        plan.log2_n2++;
        tmp >>= 1;
    }
    plan.n1 = tmp;
    plan.work_floats = 2 * n + 2 * plan.n1;

    // DFT-N1 twiddle matrix.
    plan.tw_n1.resize((size_t)(2 * plan.n1 * plan.n1));
    for (int k = 0; k < plan.n1; ++k) {
        for (int j = 0; j < plan.n1; ++j) {
            const float angle = 2.0f * (float) M_PI * (float) (k * j) / (float) plan.n1;
            plan.tw_n1[(size_t)(2 * (k * plan.n1 + j) + 0)] = std::cos(angle);
            plan.tw_n1[(size_t)(2 * (k * plan.n1 + j) + 1)] = std::sin(angle);
        }
    }

    // Row-column twiddle factors.
    plan.tw_rc.resize((size_t)(2 * plan.n1 * plan.n2));
    for (int row = 0; row < plan.n1; ++row) {
        for (int col = 0; col < plan.n2; ++col) {
            const float angle = 2.0f * (float) M_PI * (float) (row * col) / (float) plan.n;
            plan.tw_rc[(size_t)(2 * (row * plan.n2 + col) + 0)] = std::cos(angle);
            plan.tw_rc[(size_t)(2 * (row * plan.n2 + col) + 1)] = std::sin(angle);
        }
    }

    // Radix-2 FFT twiddles.
    if (plan.n2 > 1) {
        plan.tw_fft.resize((size_t)(2 * (plan.n2 - 1)));
        for (int s = 0; s < plan.log2_n2; ++s) {
            const int half   = 1 << s;
            const int stride = half << 1;
            const int off    = half - 1;
            for (int j = 0; j < half; ++j) {
                const float angle = 2.0f * (float) M_PI * (float) j / (float) stride;
                plan.tw_fft[(size_t)(2 * (off + j) + 0)] = std::cos(angle);
                plan.tw_fft[(size_t)(2 * (off + j) + 1)] = std::sin(angle);
            }
        }
    }

    return plan;
}

// work must point to at least plan.work_floats floats.
static void irfft_with_plan(
        const irfft_plan & plan,
        const float * inp_cplx,
        float * out_real,
        float * work) {
    const int N  = plan.n;
    const int N1 = plan.n1;
    const int N2 = plan.n2;

    // --- Zero-pad n_freq complex values to N complex values. ---------------
    std::memcpy(work, inp_cplx, (size_t)(2 * plan.n_freq) * sizeof(float));
    std::memset(work + 2 * plan.n_freq, 0, (size_t)(2 * (N - plan.n_freq)) * sizeof(float));

    // Data layout in work: row n1, col n2 → work[2*(n1*N2 + n2) + 0/1].
    // N1 rows of N2 complex elements (row-major).

    // --- Step 1: Column DFTs of size N1. -----------------------------------
    // T[k1][n2] = Σ_{j=0}^{N1-1} x[j*N2+n2] · tw_n1[k1·N1+j]
    if (N1 > 1) {
        float * col = work + 2 * N;  // scratch for one column (2*N1 floats)

        for (int n2 = 0; n2 < N2; ++n2) {
            // Gather column n2.
            for (int j = 0; j < N1; ++j) {
                col[2 * j + 0] = work[2 * (j * N2 + n2) + 0];
                col[2 * j + 1] = work[2 * (j * N2 + n2) + 1];
            }
            // DFT-N1 and scatter back.
            for (int k1 = 0; k1 < N1; ++k1) {
                float re = 0.0f, im = 0.0f;
                const float * tw = plan.tw_n1.data() + 2 * k1 * N1;
                for (int j = 0; j < N1; ++j) {
                    const float wr = tw[2 * j + 0];
                    const float wi = tw[2 * j + 1];
                    re += col[2 * j + 0] * wr - col[2 * j + 1] * wi;
                    im += col[2 * j + 0] * wi + col[2 * j + 1] * wr;
                }
                work[2 * (k1 * N2 + n2) + 0] = re;
                work[2 * (k1 * N2 + n2) + 1] = im;
            }
        }
    }

    // --- Step 2: Twiddle-factor multiply. ----------------------------------
    for (int k1 = 0; k1 < N1; ++k1) {
        const float * tw = plan.tw_rc.data() + 2 * k1 * N2;
        float * row = work + 2 * k1 * N2;
        for (int n2 = 0; n2 < N2; ++n2) {
            const float xr = row[2 * n2 + 0];
            const float xi = row[2 * n2 + 1];
            const float wr = tw[2 * n2 + 0];
            const float wi = tw[2 * n2 + 1];
            row[2 * n2 + 0] = xr * wr - xi * wi;
            row[2 * n2 + 1] = xr * wi + xi * wr;
        }
    }

    // --- Step 3: Row FFTs of size N2 (in-place radix-2). -------------------
    if (plan.log2_n2 > 0) {
        for (int k1 = 0; k1 < N1; ++k1) {
            float * row = work + 2 * k1 * N2;

            // Bit-reversal permutation.
            for (int i = 1, j = 0; i < N2; ++i) {
                int bit = N2 >> 1;
                while (j & bit) { j ^= bit; bit >>= 1; }
                j ^= bit;
                if (i < j) {
                    std::swap(row[2 * i + 0], row[2 * j + 0]);
                    std::swap(row[2 * i + 1], row[2 * j + 1]);
                }
            }

            // Butterfly stages.
            for (int s = 0; s < plan.log2_n2; ++s) {
                const int half   = 1 << s;
                const int stride = half << 1;
                const int tw_off = half - 1;
                for (int g = 0; g < N2; g += stride) {
                    for (int j = 0; j < half; ++j) {
                        const float wr = plan.tw_fft[2 * (tw_off + j) + 0];
                        const float wi = plan.tw_fft[2 * (tw_off + j) + 1];
                        const int u = g + j;
                        const int v = u + half;
                        const float tr = row[2 * v + 0] * wr - row[2 * v + 1] * wi;
                        const float ti = row[2 * v + 0] * wi + row[2 * v + 1] * wr;
                        row[2 * v + 0] = row[2 * u + 0] - tr;
                        row[2 * v + 1] = row[2 * u + 1] - ti;
                        row[2 * u + 0] += tr;
                        row[2 * u + 1] += ti;
                    }
                }
            }
        }
    }

    // --- Step 4: Extract real parts with Cooley-Tukey output mapping. ------
    // DFT output index k = k1 + N1*k2  maps to  work[k1*N2 + k2].
    const float scale = 1.0f / (float) plan.n_freq;
    for (int k1 = 0; k1 < N1; ++k1) {
        const float * row = work + 2 * k1 * N2;
        for (int k2 = 0; k2 < N2; ++k2) {
            out_real[k1 + N1 * k2] = row[2 * k2] * scale;
        }
    }
}

static std::vector<float> embd_to_audio(
        const float * embd,
        int n_codes,
        int n_embd,
        int n_threads,
        int n_fft,
        int n_hop) {
    if (embd == nullptr || n_codes <= 0 || n_embd <= 0 || n_fft <= 0 || n_hop <= 0) {
        return {};
    }
    if ((n_embd % 2) != 0 || n_embd != n_fft + 2) {
        return {};
    }

    const int n_win = n_fft;
    const int n_pad = (n_win - n_hop) / 2;
    const int n_out = (n_codes - 1) * n_hop + n_win;
    const int n_freq = n_embd / 2;

    std::vector<float> hann((size_t) n_fft);
    fill_hann_window((int) hann.size(), true, hann.data());
    std::vector<float> hann2((size_t) n_fft);
    for (int j = 0; j < n_fft; ++j) {
        hann2[(size_t) j] = hann[(size_t) j] * hann[(size_t) j];
    }

    std::vector<float> env_ola((size_t) n_out, 0.0f);
    for (int l = 0; l < n_codes; ++l) {
        const int out_off = l * n_hop;
        const int max_j = std::max(0, std::min(n_fft, n_out - out_off));
        for (int j = 0; j < max_j; ++j) {
            env_ola[(size_t) (out_off + j)] += hann2[(size_t) j];
        }
    }

    irfft_plan plan = make_irfft_plan(n_fft);
    if (plan.n != n_fft || plan.n_freq != n_freq) {
        return {};
    }

#if defined(__EMSCRIPTEN__)
    // Browser runtime: keep this path single-threaded to avoid requiring
    // SharedArrayBuffer / pthread-enabled hosting.
    std::vector<float> audio_ola((size_t) n_out, 0.0f);
    std::vector<float> frame_cplx((size_t) n_embd, 0.0f);
    std::vector<float> frame_time((size_t) n_fft, 0.0f);
    std::vector<float> fft_work((size_t) plan.work_floats, 0.0f);

    for (int l = 0; l < n_codes; ++l) {
        const float * frame = embd + (size_t) l * (size_t) n_embd;
        for (int k = 0; k < n_freq; ++k) {
            float mag = std::exp(frame[(size_t) k]);
            if (mag > 1e2f) {
                mag = 1e2f;
            }
            const float phi = frame[(size_t) (k + n_freq)];
            frame_cplx[(size_t) 2 * (size_t) k + 0] = mag * std::cos(phi);
            frame_cplx[(size_t) 2 * (size_t) k + 1] = mag * std::sin(phi);
        }

        irfft_with_plan(plan, frame_cplx.data(), frame_time.data(), fft_work.data());

        const int out_off = l * n_hop;
        const int max_j = std::max(0, std::min(n_fft, n_out - out_off));
        for (int j = 0; j < max_j; ++j) {
            audio_ola[(size_t) (out_off + j)] += frame_time[(size_t) j] * hann[(size_t) j];
        }
    }
#else
    n_threads = std::max(1, std::min(n_threads, n_codes));
    std::vector<std::vector<float>> partials((size_t) n_threads, std::vector<float>((size_t) n_out, 0.0f));
    std::vector<std::thread> workers((size_t) n_threads);

    for (int ti = 0; ti < n_threads; ++ti) {
        workers[(size_t) ti] = std::thread([&, ti]() {
            std::vector<float> frame_cplx((size_t) n_embd, 0.0f);
            std::vector<float> frame_time((size_t) n_fft, 0.0f);
            std::vector<float> fft_work((size_t) plan.work_floats, 0.0f);
            std::vector<float> & out = partials[(size_t) ti];

            for (int l = ti; l < n_codes; l += n_threads) {
                const float * frame = embd + (size_t) l * (size_t) n_embd;

                for (int k = 0; k < n_freq; ++k) {
                    float mag = std::exp(frame[(size_t) k]);
                    if (mag > 1e2f) {
                        mag = 1e2f;
                    }
                    const float phi = frame[(size_t) (k + n_freq)];
                    frame_cplx[(size_t) 2 * (size_t) k + 0] = mag * std::cos(phi);
                    frame_cplx[(size_t) 2 * (size_t) k + 1] = mag * std::sin(phi);
                }

                irfft_with_plan(plan, frame_cplx.data(), frame_time.data(), fft_work.data());

                const int out_off = l * n_hop;
                const int max_j = std::max(0, std::min(n_fft, n_out - out_off));
                for (int j = 0; j < max_j; ++j) {
                    out[(size_t) (out_off + j)] += frame_time[(size_t) j] * hann[(size_t) j];
                }
            }
        });
    }

    for (int ti = 0; ti < n_threads; ++ti) {
        workers[(size_t) ti].join();
    }

    std::vector<float> audio_ola((size_t) n_out, 0.0f);
    for (int ti = 0; ti < n_threads; ++ti) {
        const std::vector<float> & part = partials[(size_t) ti];
        for (int i = 0; i < n_out; ++i) {
            audio_ola[(size_t) i] += part[(size_t) i];
        }
    }
#endif

    const int crop_begin = std::max(0, n_pad);
    const int crop_end = std::max(crop_begin, n_out - n_pad);
    std::vector<float> audio((size_t) (crop_end - crop_begin), 0.0f);

    for (int i = crop_begin; i < crop_end; ++i) {
        float y = audio_ola[(size_t) i];
        const float env = env_ola[(size_t) i];
        if (env > 1e-12f) {
            y /= env;
        }
        audio[(size_t) (i - crop_begin)] = y;
    }

    return audio;
}

static bool save_wav16(const char * fname, const float * data, size_t n, int sample_rate, std::string & err) {
    if (fname == nullptr || fname[0] == '\0') {
        err = "output wav path is empty";
        return false;
    }

    std::ofstream file(fname, std::ios::binary);
    if (!file) {
        err = "failed to open output wav file";
        return false;
    }

    wav_header header;
    header.sample_rate = (uint32_t) sample_rate;
    header.byte_rate = header.sample_rate * header.num_channels * (header.bits_per_sample / 8);
    header.block_align = header.num_channels * (header.bits_per_sample / 8);
    header.data_size = (uint32_t) (n * (header.bits_per_sample / 8));
    header.chunk_size = 36 + header.data_size;

    file.write(reinterpret_cast<const char *>(&header), sizeof(header));

    for (size_t i = 0; i < n; ++i) {
        const float x = std::clamp(data[i], -1.0f, 1.0f);
        const int16_t pcm = (int16_t) std::lrintf(x * 32767.0f);
        file.write(reinterpret_cast<const char *>(&pcm), sizeof(pcm));
    }

    if (!file.good()) {
        err = "failed while writing output wav file";
        return false;
    }

    return true;
}

} // namespace

extern "C" {

struct mio_tts_context {
    mio_tts_context_impl impl;
};

struct mio_tts_vocab_map {
    mio_tts_vocab_map_impl impl;
};

struct mio_tts_params mio_tts_default_params(void) {
    struct mio_tts_params p;
    p.n_threads = 0;
    p.max_reference_seconds = 20.0f;
    p.wavlm_flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    p.miocodec_flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    return p;
}

struct mio_tts_context * mio_tts_init_from_file(
        const char * model_vocoder,
        const char * wavlm_model,
        char * err,
        size_t err_size) {
    if (model_vocoder == nullptr || model_vocoder[0] == '\0') {
        set_err(err, err_size, "model_vocoder path is empty");
        return nullptr;
    }

    auto model = std::make_shared<mio_tts_model_shared>();
    model->vocoder_path = model_vocoder;

    std::string load_err;
    if (!model->decoder.load(model->vocoder_path, load_err)) {
        set_err(err, err_size, "failed to load MioCodec GGUF: " + load_err);
        return nullptr;
    }

    if (wavlm_model != nullptr && wavlm_model[0] != '\0') {
        model->wavlm_path = wavlm_model;
        std::string wavlm_err;
        if (!model->wavlm.load(model->wavlm_path, wavlm_err)) {
            set_err(err, err_size, "failed to load WavLM GGUF: " + wavlm_err);
            return nullptr;
        }
    }

    auto * ctx = new mio_tts_context();
    ctx->impl.model = std::move(model);
    return ctx;
}

struct mio_tts_context * mio_tts_context_clone(
        const struct mio_tts_context * ctx,
        char * err,
        size_t err_size) {
    if (ctx == nullptr || !ctx->impl.model) {
        set_err(err, err_size, "source context is null");
        return nullptr;
    }

    auto * out = new mio_tts_context();
    out->impl.model = ctx->impl.model;
    return out;
}

void mio_tts_free(struct mio_tts_context * ctx) {
    delete ctx;
}

bool mio_tts_context_is_dynamic_global(const struct mio_tts_context * ctx) {
    return ctx != nullptr && ctx->impl.model && ctx->impl.model->decoder.has_dynamic_global();
}

int32_t mio_tts_context_sample_rate(const struct mio_tts_context * ctx) {
    if (ctx == nullptr || !ctx->impl.model) {
        return 0;
    }
    return ctx->impl.model->decoder.params().sample_rate;
}

int32_t mio_tts_context_samples_per_token(const struct mio_tts_context * ctx) {
    if (ctx == nullptr || !ctx->impl.model) {
        return 0;
    }
    return ctx->impl.model->decoder.params().samples_per_token;
}

bool mio_tts_context_set_backend_device(
        struct mio_tts_context * ctx,
        const char * backend_name,
        char * err,
        size_t err_size) {
    if (ctx == nullptr || !ctx->impl.model) {
        set_err(err, err_size, "context is null");
        return false;
    }

    const std::string name = backend_name != nullptr ? backend_name : "";
    std::lock_guard<std::mutex> lock(ctx->impl.model->runtime_mutex);

    std::string backend_err;
    if (!ctx->impl.model->decoder.set_backend_device(name, backend_err)) {
        set_err(err, err_size, "decoder backend setup failed: " + backend_err);
        return false;
    }

    if (!ctx->impl.model->wavlm_path.empty()) {
        backend_err.clear();
        if (!ctx->impl.model->wavlm.set_backend_device(name, backend_err)) {
            set_err(err, err_size, "wavlm backend setup failed: " + backend_err);
            return false;
        }
    }

    return true;
}

struct mio_tts_vocab_map * mio_tts_vocab_map_init(
        const struct llama_vocab * vocab,
        char * err,
        size_t err_size) {
    if (vocab == nullptr) {
        set_err(err, err_size, "vocab is null");
        return nullptr;
    }

    auto * map = new mio_tts_vocab_map();
    std::string map_err;
    if (!build_mio_audio_token_map(vocab, map->impl.token_to_code, map_err)) {
        set_err(err, err_size, map_err);
        delete map;
        return nullptr;
    }

    return map;
}

void mio_tts_vocab_map_free(struct mio_tts_vocab_map * map) {
    delete map;
}

bool mio_tts_tokens_to_codes(
        const struct mio_tts_vocab_map * map,
        const llama_token * tokens,
        size_t n_tokens,
        int32_t ** codes_out,
        size_t * n_codes_out,
        char * err,
        size_t err_size) {
    if (map == nullptr) {
        set_err(err, err_size, "vocab map is null");
        return false;
    }
    if (tokens == nullptr || n_tokens == 0) {
        set_err(err, err_size, "input tokens are empty");
        return false;
    }
    if (codes_out == nullptr || n_codes_out == nullptr) {
        set_err(err, err_size, "output pointers are null");
        return false;
    }

    std::vector<int32_t> codes;
    codes.reserve(n_tokens);

    for (size_t i = 0; i < n_tokens; ++i) {
        auto it = map->impl.token_to_code.find(tokens[i]);
        if (it != map->impl.token_to_code.end()) {
            codes.push_back(it->second);
        }
    }

    if (codes.empty()) {
        set_err(err, err_size, "no Mio audio codes were found in token sequence");
        return false;
    }

    int32_t * out = (int32_t *) std::malloc(codes.size() * sizeof(int32_t));
    if (out == nullptr) {
        set_err(err, err_size, "failed to allocate output codes");
        return false;
    }

    std::memcpy(out, codes.data(), codes.size() * sizeof(int32_t));
    *codes_out = out;
    *n_codes_out = codes.size();
    return true;
}

bool mio_tts_token_to_code(
        const struct mio_tts_vocab_map * map,
        llama_token token,
        int32_t * code_out) {
    if (map == nullptr || code_out == nullptr) {
        return false;
    }
    auto it = map->impl.token_to_code.find(token);
    if (it == map->impl.token_to_code.end()) {
        return false;
    }
    *code_out = it->second;
    return true;
}

bool mio_tts_codes_load(
        const char * path,
        int32_t ** codes_out,
        size_t * n_codes_out,
        char * err,
        size_t err_size) {
    if (codes_out == nullptr || n_codes_out == nullptr) {
        set_err(err, err_size, "output pointers are null");
        return false;
    }

    std::vector<int32_t> codes;
    std::string load_err;
    if (!load_codes_text(path, codes, load_err)) {
        set_err(err, err_size, load_err);
        return false;
    }

    int32_t * out = (int32_t *) std::malloc(codes.size() * sizeof(int32_t));
    if (out == nullptr) {
        set_err(err, err_size, "failed to allocate output codes");
        return false;
    }

    std::memcpy(out, codes.data(), codes.size() * sizeof(int32_t));
    *codes_out = out;
    *n_codes_out = codes.size();
    return true;
}

bool mio_tts_codes_save(
        const char * path,
        const int32_t * codes,
        size_t n_codes,
        char * err,
        size_t err_size) {
    std::string save_err;
    if (!save_codes_text(path, codes, n_codes, save_err)) {
        set_err(err, err_size, save_err);
        return false;
    }
    return true;
}

void mio_tts_codes_free(int32_t * codes) {
    std::free(codes);
}

bool mio_tts_reference_to_embedding(
        struct mio_tts_context * ctx,
        const char * reference_audio,
        struct mio_tts_params params,
        float ** embedding_out,
        size_t * n_embedding_out,
        char * err,
        size_t err_size) {
    if (ctx == nullptr || !ctx->impl.model) {
        set_err(err, err_size, "context is null");
        return false;
    }
    if (reference_audio == nullptr || reference_audio[0] == '\0') {
        set_err(err, err_size, "reference_audio path is empty");
        return false;
    }
    if (embedding_out == nullptr || n_embedding_out == nullptr) {
        set_err(err, err_size, "output pointers are null");
        return false;
    }

    std::lock_guard<std::mutex> lock(ctx->impl.model->runtime_mutex);

    if (!ctx->impl.model->decoder.has_dynamic_global()) {
        set_err(err, err_size, "reference embedding requires dynamic-global MioCodec");
        return false;
    }
    if (!ctx->impl.model->decoder.has_global_encoder()) {
        set_err(err, err_size, "reference embedding requires global_encoder tensors in MioCodec GGUF");
        return false;
    }

    if (ctx->impl.model->wavlm_path.empty()) {
        set_err(err, err_size, "WavLM model is not loaded");
        return false;
    }

    std::vector<float> ssl_features;
    int ssl_frames = 0;
    std::string ref_err;

    const int n_threads = resolve_threads(params.n_threads);
    if (!ctx->impl.model->wavlm.extract_ssl_features(
                reference_audio,
                ctx->impl.model->decoder.params().sample_rate,
                params.max_reference_seconds,
                n_threads,
                params.wavlm_flash_attn_type,
                ssl_features,
                ssl_frames,
                ref_err)) {
        set_err(err, err_size, "failed to extract WavLM SSL features: " + ref_err);
        return false;
    }

    std::vector<float> emb;
    ref_err.clear();
    if (!ctx->impl.model->decoder.encode_global_embedding_from_ssl(
                ssl_features,
                ssl_frames,
                n_threads,
                emb,
                ref_err)) {
        set_err(err, err_size, "failed to encode global embedding: " + ref_err);
        return false;
    }

    float * out = (float *) std::malloc(emb.size() * sizeof(float));
    if (out == nullptr) {
        set_err(err, err_size, "failed to allocate output embedding");
        return false;
    }

    std::memcpy(out, emb.data(), emb.size() * sizeof(float));
    *embedding_out = out;
    *n_embedding_out = emb.size();
    return true;
}

bool mio_tts_embedding_load_gguf(
        const char * path,
        float ** embedding_out,
        size_t * n_embedding_out,
        char * err,
        size_t err_size) {
    if (embedding_out == nullptr || n_embedding_out == nullptr) {
        set_err(err, err_size, "output pointers are null");
        return false;
    }

    std::vector<float> emb;
    std::string load_err;
    if (!load_embedding_gguf_vec(path, emb, load_err)) {
        set_err(err, err_size, load_err);
        return false;
    }

    float * out = (float *) std::malloc(emb.size() * sizeof(float));
    if (out == nullptr) {
        set_err(err, err_size, "failed to allocate output embedding");
        return false;
    }

    std::memcpy(out, emb.data(), emb.size() * sizeof(float));
    *embedding_out = out;
    *n_embedding_out = emb.size();
    return true;
}

bool mio_tts_embedding_save_gguf(
        const char * path,
        const float * embedding,
        size_t n_embedding,
        char * err,
        size_t err_size) {
    if (embedding == nullptr || n_embedding == 0) {
        set_err(err, err_size, "embedding is empty");
        return false;
    }

    std::vector<float> emb(embedding, embedding + n_embedding);
    std::string save_err;
    if (!save_embedding_gguf_vec(path, emb, save_err)) {
        set_err(err, err_size, save_err);
        return false;
    }

    return true;
}

void mio_tts_embedding_free(float * embedding) {
    std::free(embedding);
}

bool mio_tts_synthesize(
        struct mio_tts_context * ctx,
        const int32_t * codes,
        size_t n_codes,
        const float * embedding,
        size_t n_embedding,
        struct mio_tts_params params,
        float ** audio_out,
        size_t * n_audio_out,
        int32_t * sample_rate_out,
        char * err,
        size_t err_size) {
    if (ctx == nullptr || !ctx->impl.model) {
        set_err(err, err_size, "context is null");
        return false;
    }
    if (codes == nullptr || n_codes == 0) {
        set_err(err, err_size, "codes are empty");
        return false;
    }
    if (audio_out == nullptr || n_audio_out == nullptr || sample_rate_out == nullptr) {
        set_err(err, err_size, "output pointers are null");
        return false;
    }

    std::lock_guard<std::mutex> lock(ctx->impl.model->runtime_mutex);

    std::vector<int32_t> codes_vec(codes, codes + n_codes);
    for (int32_t c : codes_vec) {
        if (c < k_mio_code_min || c > k_mio_code_max) {
            set_err(err, err_size, "code id out of range");
            return false;
        }
    }

    const bool has_embedding = embedding != nullptr && n_embedding > 0;
    if (ctx->impl.model->decoder.has_dynamic_global() && !has_embedding) {
        set_err(err, err_size, "dynamic-global MioCodec requires embedding");
        return false;
    }
    if (!ctx->impl.model->decoder.has_dynamic_global() && has_embedding) {
        set_err(err, err_size, "static MioCodec does not accept external embedding");
        return false;
    }

    std::vector<float> emb_vec;
    if (has_embedding) {
        emb_vec.assign(embedding, embedding + n_embedding);
        if ((int) emb_vec.size() != ctx->impl.model->decoder.params().decoder_adanorm_dim) {
            set_err(err, err_size, "embedding dimension mismatch");
            return false;
        }
    }

    const int n_threads = resolve_threads(params.n_threads);

    std::vector<float> spec;
    int n_frames = 0;
    std::string dec_err;

    const auto t0 = std::chrono::steady_clock::now();

    bool ok = false;
    if (has_embedding) {
        ok = ctx->impl.model->decoder.decode_tokens(
                codes_vec,
                emb_vec,
                n_threads,
                params.miocodec_flash_attn_type,
                spec,
                n_frames,
                dec_err);
    } else {
        ok = ctx->impl.model->decoder.decode_tokens(
                codes_vec,
                n_threads,
                params.miocodec_flash_attn_type,
                spec,
                n_frames,
                dec_err);
    }

    const auto t1 = std::chrono::steady_clock::now();

    if (!ok) {
        set_err(err, err_size, "MioCodec decode failed: " + dec_err);
        return false;
    }

    if (n_frames <= 0 || spec.empty()) {
        set_err(err, err_size, "MioCodec produced empty spectrogram");
        return false;
    }

    std::vector<float> audio;
    if (ctx->impl.model->decoder.output_is_mel()) {
        std::string voc_err;
        if (!ctx->impl.model->decoder.decode_mel_to_audio(spec, n_frames, n_threads, audio, voc_err)) {
            set_err(err, err_size, "MioVocoder decode failed: " + voc_err);
            return false;
        }
    } else {
        const auto & hp = ctx->impl.model->decoder.params();
        audio = embd_to_audio(spec.data(), n_frames, hp.n_fft + 2, n_threads, hp.n_fft, hp.hop_length);
    }

    const auto t2 = std::chrono::steady_clock::now();
    const double decode_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double istft_ms  = std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::fprintf(stderr, "synth breakdown: decode=%.1fms istft=%.1fms n_codes=%zu n_frames=%d\n",
                 decode_ms, istft_ms, n_codes, n_frames);

    if (audio.empty()) {
        set_err(err, err_size, "audio is empty");
        return false;
    }

    double peak = 0.0;
    for (float x : audio) {
        if (std::isfinite(x)) {
            peak = std::max(peak, (double) std::fabs(x));
        }
    }
    if (peak > 0.98) {
        const float gain = (float) (0.95 / peak);
        for (float & x : audio) {
            x *= gain;
        }
    }

    float * out = (float *) std::malloc(audio.size() * sizeof(float));
    if (out == nullptr) {
        set_err(err, err_size, "failed to allocate output audio");
        return false;
    }

    std::memcpy(out, audio.data(), audio.size() * sizeof(float));
    *audio_out = out;
    *n_audio_out = audio.size();
    *sample_rate_out = ctx->impl.model->decoder.params().sample_rate;
    return true;
}

bool mio_tts_reserve_workspace(
        struct mio_tts_context * ctx,
        int32_t n_codes,
        char * err,
        size_t err_size) {
    if (ctx == nullptr || !ctx->impl.model) {
        set_err(err, err_size, "context is null");
        return false;
    }

    const int n = std::max<int32_t>(1, n_codes);
    try {
        std::lock_guard<std::mutex> lock(ctx->impl.model->runtime_mutex);
        ctx->impl.model->decoder.reserve_decode_workspace(n);
    } catch (const std::bad_alloc &) {
        set_err(err, err_size, "failed to reserve decode workspace");
        return false;
    }

    return true;
}

bool mio_tts_estimate_workspace_bytes(
        struct mio_tts_context * ctx,
        int32_t n_codes,
        uint64_t * out_bytes,
        char * err,
        size_t err_size) {
    if (ctx == nullptr || !ctx->impl.model) {
        set_err(err, err_size, "context is null");
        return false;
    }
    if (out_bytes == nullptr) {
        set_err(err, err_size, "out_bytes is null");
        return false;
    }

    const int n = std::max<int32_t>(1, n_codes);
    std::lock_guard<std::mutex> lock(ctx->impl.model->runtime_mutex);
    const size_t est = ctx->impl.model->decoder.estimate_decode_workspace_bytes(n);
    *out_bytes = (uint64_t) est;
    return true;
}

bool mio_tts_reserve_reference_workspace(
        struct mio_tts_context * ctx,
        float max_reference_seconds,
        char * err,
        size_t err_size) {
    if (ctx == nullptr || !ctx->impl.model) {
        set_err(err, err_size, "context is null");
        return false;
    }
    if (ctx->impl.model->wavlm_path.empty()) {
        set_err(err, err_size, "WavLM model is not loaded");
        return false;
    }
    if (!ctx->impl.model->decoder.has_global_encoder()) {
        set_err(err, err_size, "global encoder tensors are not available in this GGUF");
        return false;
    }

    const int source_sr = std::max(1, ctx->impl.model->decoder.params().sample_rate);
    const float ref_sec = max_reference_seconds > 0.0f ? max_reference_seconds : 20.0f;
    try {
        std::lock_guard<std::mutex> lock(ctx->impl.model->runtime_mutex);
        const int ssl_frames = ctx->impl.model->wavlm.estimate_ssl_frames(source_sr, ref_sec);
        ctx->impl.model->wavlm.reserve_workspace(source_sr, ref_sec);
        ctx->impl.model->decoder.reserve_embedding_workspace(ssl_frames);
    } catch (const std::bad_alloc &) {
        set_err(err, err_size, "failed to reserve reference workspace");
        return false;
    }
    return true;
}

bool mio_tts_estimate_reference_workspace_bytes(
        struct mio_tts_context * ctx,
        float max_reference_seconds,
        uint64_t * out_bytes,
        char * err,
        size_t err_size) {
    if (ctx == nullptr || !ctx->impl.model) {
        set_err(err, err_size, "context is null");
        return false;
    }
    if (out_bytes == nullptr) {
        set_err(err, err_size, "out_bytes is null");
        return false;
    }
    if (ctx->impl.model->wavlm_path.empty()) {
        set_err(err, err_size, "WavLM model is not loaded");
        return false;
    }
    if (!ctx->impl.model->decoder.has_global_encoder()) {
        set_err(err, err_size, "global encoder tensors are not available in this GGUF");
        return false;
    }

    const int source_sr = std::max(1, ctx->impl.model->decoder.params().sample_rate);
    const float ref_sec = max_reference_seconds > 0.0f ? max_reference_seconds : 20.0f;
    std::lock_guard<std::mutex> lock(ctx->impl.model->runtime_mutex);
    const int ssl_frames = ctx->impl.model->wavlm.estimate_ssl_frames(source_sr, ref_sec);
    const size_t wavlm_bytes = ctx->impl.model->wavlm.estimate_workspace_bytes(source_sr, ref_sec);
    const size_t embed_bytes = ctx->impl.model->decoder.estimate_embedding_workspace_bytes(ssl_frames);
    *out_bytes = (uint64_t) (wavlm_bytes + embed_bytes);
    return true;
}

bool mio_tts_audio_save_wav16(
        const char * path,
        const float * audio,
        size_t n_audio,
        int32_t sample_rate,
        char * err,
        size_t err_size) {
    if (audio == nullptr || n_audio == 0) {
        set_err(err, err_size, "audio is empty");
        return false;
    }

    std::string save_err;
    if (!save_wav16(path, audio, n_audio, sample_rate, save_err)) {
        set_err(err, err_size, save_err);
        return false;
    }

    return true;
}

void mio_tts_audio_free(float * audio) {
    std::free(audio);
}

} // extern "C"
