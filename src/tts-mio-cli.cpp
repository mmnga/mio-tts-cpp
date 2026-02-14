#include "llama.h"
#include "mio-tts-lib.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <regex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using json = nlohmann::ordered_json;

enum llm_api_mode {
    LLM_API_MODE_OPENAI_CHAT = 0,
    LLM_API_MODE_GENERIC = 1,
};

struct cli_params {
    std::string model;
    std::string model_vocoder;
    std::string wavlm_model;

    std::string prompt;
    std::string prompt_file;
    std::string output = "output.wav";

    std::string inline_codes_text;
    std::string codes_in;
    std::string codes_out;
    std::string embedding_in;
    std::string embedding_out;
    std::string embedding_default_in;
    std::string reference_audio;
    std::string reference_dir;
    std::string remove_reference_key;

    int32_t n_gpu_layers = -1;
    int32_t n_threads = 2;
    int32_t n_ctx = 700;
    int32_t n_predict = 400;
    llama_flash_attn_type flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;

    int32_t top_k = 50;
    float top_p = 1.0f;
    float temp = 0.8f;
    float repeat_penalty = 1.0f;
    uint32_t seed = 0;

    std::string llm_api_url;
    std::string llm_api_key;
    std::string llm_api_model;
    std::vector<std::pair<std::string, std::string>> llm_api_headers;
    int32_t llm_api_timeout_sec = 120;
    llm_api_mode llm_api_mode_value = LLM_API_MODE_OPENAI_CHAT;

    float max_reference_seconds = 20.0f;

    bool codes_only = false;
    bool embedding_only = false;
    bool show_help = false;
    std::vector<int32_t> inline_codes;
};

static void print_usage(const char * argv0) {
    std::fprintf(stderr,
        "Usage:\n"
        "  %s -mv MIOCODEC [options]\n\n"
        "Required:\n"
        "  -mv, --model-vocoder FNAME      MioCodec GGUF (always required)\n"
        "  -m, --model FNAME               MioTTS LLM GGUF (required with --prompt unless --llm-api-url)\n\n"
        "Optional external LLM API:\n"
        "  --llm-api-url URL               external LLM API endpoint\n"
        "  --llm-api-key STR               bearer token for --llm-api-url\n"
        "  --llm-api-model STR             model name for external API requests\n"
        "  --llm-api-headers JSON          extra request headers as JSON object\n"
        "  --llm-api-timeout N             external API timeout seconds (default: 120)\n"
        "  --llm-api-mode STR              openai-chat | generic (default: openai-chat)\n\n"
        "Generation:\n"
        "  -p, --prompt TEXT               input text\n"
        "  --prompt-file FNAME             input text file\n"
        "  -o, --output FNAME              output wav (default: output.wav)\n"
        "  -n, --n-predict N               max generated tokens (default: 400)\n"
        "  --temp F                        temperature (default: 0.8)\n"
        "  --top-p F                       top-p (default: 1.0)\n"
        "  --top-k N                       top-k (default: 50)\n"
        "  --repeat-penalty F              repeat penalty (default: 1.0)\n"
        "  --seed N                        sampler seed (default: 0)\n"
        "  --threads N                     threads for llama + mio decode (default: 2, auto if 0)\n"
        "  --ctx-size N                    llama context size (default: 700)\n"
        "  --n-gpu-layers N, -ngl N        GPU layers for LLM\n\n"
        "  -fa, --flash-attn [on|off|auto] flash attention mode (default: auto)\n\n"
        "Mio cache/reference:\n"
        "  --tts-mio-codes TEXT            inline codes (CSV / whitespace / <|s_123|>)\n"
        "  --tts-mio-codes-in FNAME\n"
        "  --tts-mio-codes-out FNAME\n"
        "  --tts-mio-codes-only\n"
        "  --tts-reference-audio FNAME\n"
        "  --tts-wavlm-model FNAME\n"
        "  --tts-max-reference-seconds F\n"
        "  --tts-reference-dir DIR\n"
        "  --tts-remove-reference-key STR\n"
        "  --tts-mio-embedding-in FNAME\n"
        "  -emb, --tts-mio-default-embedding-in FNAME\n"
        "  --tts-mio-embedding-out FNAME\n"
        "  --tts-mio-embedding-only\n\n"
        "Other:\n"
        "  -h, --help                      show this help\n",
        argv0);
}

static bool parse_i32(const char * s, int32_t & out) {
    if (s == nullptr) {
        return false;
    }
    char * end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (end == nullptr || *end != '\0') {
        return false;
    }
    out = (int32_t) v;
    return true;
}

static bool parse_u32(const char * s, uint32_t & out) {
    if (s == nullptr) {
        return false;
    }
    char * end = nullptr;
    const unsigned long v = std::strtoul(s, &end, 10);
    if (end == nullptr || *end != '\0') {
        return false;
    }
    out = (uint32_t) v;
    return true;
}

static bool parse_f32(const char * s, float & out) {
    if (s == nullptr) {
        return false;
    }
    char * end = nullptr;
    const float v = std::strtof(s, &end);
    if (end == nullptr || *end != '\0') {
        return false;
    }
    out = v;
    return true;
}

static bool parse_flash_attn_type(const char * s, llama_flash_attn_type & out) {
    if (s == nullptr) {
        return false;
    }

    std::string v(s);
    std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) {
        return (char) std::tolower(c);
    });

    if (v == "on" || v == "1" || v == "true" || v == "enabled") {
        out = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        return true;
    }
    if (v == "off" || v == "0" || v == "false" || v == "disabled") {
        out = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        return true;
    }
    if (v == "auto") {
        out = LLAMA_FLASH_ATTN_TYPE_AUTO;
        return true;
    }
    return false;
}

static bool parse_llm_api_mode(const char * s, llm_api_mode & out) {
    if (s == nullptr) {
        return false;
    }

    std::string v(s);
    std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) {
        return (char) std::tolower(c);
    });

    if (v == "openai-chat" || v == "openai" || v == "chat") {
        out = LLM_API_MODE_OPENAI_CHAT;
        return true;
    }
    if (v == "generic" || v == "raw") {
        out = LLM_API_MODE_GENERIC;
        return true;
    }
    return false;
}

static bool is_valid_reference_key(const std::string & key) {
    if (key.empty() || key.size() > 128) {
        return false;
    }
    for (char c : key) {
        const bool ok = (c >= 'a' && c <= 'z') ||
                        (c >= 'A' && c <= 'Z') ||
                        (c >= '0' && c <= '9') ||
                        c == '_' || c == '-' || c == '.';
        if (!ok) {
            return false;
        }
    }
    return true;
}

static std::string make_reference_added_output_path(const std::string & output_dir, const std::string & reference_key) {
    std::filesystem::path dir(output_dir.empty() ? "." : output_dir);
    return (dir / (reference_key + ".emb.gguf")).string();
}

static bool ieq_ascii(std::string a, std::string b) {
    if (a.size() != b.size()) {
        return false;
    }
    std::transform(a.begin(), a.end(), a.begin(), [](unsigned char c) {
        return (char) std::tolower(c);
    });
    std::transform(b.begin(), b.end(), b.begin(), [](unsigned char c) {
        return (char) std::tolower(c);
    });
    return a == b;
}

static std::string trim_copy(const std::string & in) {
    size_t b = 0;
    while (b < in.size() && std::isspace((unsigned char) in[b])) {
        ++b;
    }
    size_t e = in.size();
    while (e > b && std::isspace((unsigned char) in[e - 1])) {
        --e;
    }
    return in.substr(b, e - b);
}

static bool load_text_file(const std::string & path, std::string & out, std::string & err) {
    std::ifstream file(path);
    if (!file) {
        err = "failed to open prompt file: " + path;
        return false;
    }
    out.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (!file.good() && !file.eof()) {
        err = "failed to read prompt file: " + path;
        return false;
    }
    return true;
}

static bool parse_codes_array(const json & j, std::vector<int32_t> & out, std::string & err) {
    if (!j.is_array()) {
        err = "codes must be an array";
        return false;
    }

    static const std::regex wrapped_re(R"(<\|s_(-?\d+)\|>)");
    out.clear();
    out.reserve(j.size());

    for (size_t i = 0; i < j.size(); ++i) {
        const auto & v = j[i];
        if (v.is_number_integer()) {
            out.push_back(v.get<int32_t>());
            continue;
        }
        if (v.is_string()) {
            const std::string s = v.get<std::string>();
            std::smatch m;
            if (std::regex_match(s, m, wrapped_re)) {
                out.push_back((int32_t) std::strtol(m[1].str().c_str(), nullptr, 10));
                continue;
            }
            char * end = nullptr;
            long n = std::strtol(s.c_str(), &end, 10);
            if (end != nullptr && *end == '\0') {
                out.push_back((int32_t) n);
                continue;
            }
        }
        err = "invalid code at index " + std::to_string(i);
        return false;
    }

    return true;
}

static bool extract_codes_from_text(const std::string & text, std::vector<int32_t> & out) {
    static const std::regex token_re(R"(<\|s_(-?\d+)\|>)");
    out.clear();
    for (std::sregex_iterator it(text.begin(), text.end(), token_re), end; it != end; ++it) {
        const long v = std::strtol((*it)[1].str().c_str(), nullptr, 10);
        out.push_back((int32_t) v);
    }
    return !out.empty();
}

static bool parse_inline_codes_arg(const std::string & raw, std::vector<int32_t> & out, std::string & err) {
    out.clear();
    const std::string s = trim_copy(raw);
    if (s.empty()) {
        err = "--tts-mio-codes is empty";
        return false;
    }

    if (extract_codes_from_text(s, out)) {
        return true;
    }

    if (!s.empty() && s.front() == '[') {
        try {
            const json j = json::parse(s);
            if (parse_codes_array(j, out, err)) {
                if (out.empty()) {
                    err = "--tts-mio-codes JSON array is empty";
                    return false;
                }
                return true;
            }
            err = "--tts-mio-codes JSON parse error: " + err;
            return false;
        } catch (const std::exception & e) {
            err = std::string("--tts-mio-codes invalid JSON: ") + e.what();
            return false;
        }
    }

    std::string norm = s;
    for (char & c : norm) {
        if (c == ',') {
            c = ' ';
        }
    }

    size_t pos = 0;
    while (pos < norm.size()) {
        while (pos < norm.size() && std::isspace((unsigned char) norm[pos])) {
            ++pos;
        }
        if (pos >= norm.size()) {
            break;
        }
        size_t end = pos;
        while (end < norm.size() && !std::isspace((unsigned char) norm[end])) {
            ++end;
        }
        const std::string token = norm.substr(pos, end - pos);
        char * endp = nullptr;
        const long v = std::strtol(token.c_str(), &endp, 10);
        if (endp == nullptr || *endp != '\0') {
            err = "invalid code token in --tts-mio-codes: " + token;
            return false;
        }
        out.push_back((int32_t) v);
        pos = end;
    }

    if (out.empty()) {
        err = "--tts-mio-codes did not contain any codes";
        return false;
    }
    return true;
}

static bool parse_llm_api_headers_json(
        const std::string & raw,
        std::vector<std::pair<std::string, std::string>> & out,
        std::string & err) {
    try {
        const json j = json::parse(raw);
        if (!j.is_object()) {
            err = "--llm-api-headers must be a JSON object";
            return false;
        }
        out.clear();
        out.reserve(j.size());
        for (auto it = j.begin(); it != j.end(); ++it) {
            const std::string key = trim_copy(it.key());
            if (key.empty()) {
                err = "--llm-api-headers contains empty header name";
                return false;
            }

            std::string value;
            if (it.value().is_string()) {
                value = it.value().get<std::string>();
            } else if (it.value().is_number_integer()) {
                value = std::to_string(it.value().get<long long>());
            } else if (it.value().is_number_unsigned()) {
                value = std::to_string(it.value().get<unsigned long long>());
            } else if (it.value().is_number_float()) {
                value = std::to_string(it.value().get<double>());
            } else if (it.value().is_boolean()) {
                value = it.value().get<bool>() ? "true" : "false";
            } else if (it.value().is_null()) {
                value = "";
            } else {
                err = "--llm-api-headers values must be scalar (string/number/bool/null)";
                return false;
            }

            out.emplace_back(key, value);
        }
        return true;
    } catch (const std::exception & e) {
        err = std::string("invalid --llm-api-headers JSON: ") + e.what();
        return false;
    }
}

static void headers_upsert_ci(httplib::Headers & headers, const std::string & key, const std::string & value) {
    for (auto it = headers.begin(); it != headers.end();) {
        if (ieq_ascii(it->first, key)) {
            it = headers.erase(it);
        } else {
            ++it;
        }
    }
    headers.emplace(key, value);
}

static bool headers_find_ci(const httplib::Headers & headers, const std::string & key, std::string * value_out = nullptr) {
    for (const auto & kv : headers) {
        if (ieq_ascii(kv.first, key)) {
            if (value_out != nullptr) {
                *value_out = kv.second;
            }
            return true;
        }
    }
    return false;
}

static std::string truncate_text(const std::string & s, size_t max_len = 240) {
    if (s.size() <= max_len) {
        return s;
    }
    return s.substr(0, max_len) + "...";
}

struct parsed_llm_api_url {
    bool https = false;
    std::string host;
    int32_t port = 0;
    std::string path = "/";
};

static bool parse_llm_api_url(const std::string & raw, parsed_llm_api_url & out, std::string & err) {
    static const std::regex re(R"(^(https?)://([^/:?#]+)(?::([0-9]+))?([^?#]*)?(\?[^#]*)?$)", std::regex::icase);
    std::smatch m;
    if (!std::regex_match(raw, m, re)) {
        err = "invalid --llm-api-url: " + raw;
        return false;
    }

    std::string scheme = m[1].str();
    std::transform(scheme.begin(), scheme.end(), scheme.begin(), [](unsigned char c) {
        return (char) std::tolower(c);
    });
    if (scheme != "http" && scheme != "https") {
        err = "unsupported URL scheme for --llm-api-url: " + scheme;
        return false;
    }

    out.https = scheme == "https";
    out.host = m[2].str();
    out.port = out.https ? 443 : 80;
    if (m[3].matched && !m[3].str().empty()) {
        int32_t p = 0;
        if (!parse_i32(m[3].str().c_str(), p) || p < 1 || p > 65535) {
            err = "invalid port in --llm-api-url";
            return false;
        }
        out.port = p;
    }
    out.path = m[4].matched ? m[4].str() : "/";
    if (out.path.empty()) {
        out.path = "/";
    }
    if (m[5].matched) {
        out.path += m[5].str();
    }

    return true;
}

static bool extract_text_from_llm_api_json(const json & j, std::string & out) {
    out.clear();

    auto append_content = [&](const json & content) {
        if (content.is_string()) {
            if (!out.empty()) {
                out.push_back('\n');
            }
            out += content.get<std::string>();
            return;
        }
        if (content.is_array()) {
            for (const auto & item : content) {
                if (item.is_string()) {
                    if (!out.empty()) {
                        out.push_back('\n');
                    }
                    out += item.get<std::string>();
                } else if (item.is_object()) {
                    const auto it_text = item.find("text");
                    if (it_text != item.end() && it_text->is_string()) {
                        if (!out.empty()) {
                            out.push_back('\n');
                        }
                        out += it_text->get<std::string>();
                    }
                }
            }
        }
    };

    const auto it_text = j.find("text");
    if (it_text != j.end()) {
        append_content(*it_text);
    }
    const auto it_output_text = j.find("output_text");
    if (it_output_text != j.end()) {
        append_content(*it_output_text);
    }

    const auto it_choices = j.find("choices");
    if (it_choices != j.end() && it_choices->is_array() && !it_choices->empty()) {
        const auto & c0 = (*it_choices)[0];
        const auto it_choice_text = c0.find("text");
        if (it_choice_text != c0.end()) {
            append_content(*it_choice_text);
        }
        const auto it_msg = c0.find("message");
        if (it_msg != c0.end() && it_msg->is_object()) {
            const auto it_content = it_msg->find("content");
            if (it_content != it_msg->end()) {
                append_content(*it_content);
            }
        }
    }

    return !out.empty();
}

static bool parse_codes_from_llm_api_response(const json & rsp, std::vector<int32_t> & out_codes, std::string & err) {
    out_codes.clear();

    auto it_codes = rsp.find("codes");
    if (it_codes != rsp.end() && !it_codes->is_null()) {
        if (!parse_codes_array(*it_codes, out_codes, err)) {
            err = "invalid 'codes' in LLM API response: " + err;
            return false;
        }
        if (out_codes.empty()) {
            err = "LLM API response contains empty 'codes'";
            return false;
        }
        return true;
    }

    it_codes = rsp.find("codes_values");
    if (it_codes != rsp.end() && !it_codes->is_null()) {
        if (!parse_codes_array(*it_codes, out_codes, err)) {
            err = "invalid 'codes_values' in LLM API response: " + err;
            return false;
        }
        if (out_codes.empty()) {
            err = "LLM API response contains empty 'codes_values'";
            return false;
        }
        return true;
    }

    it_codes = rsp.find("audio_codes");
    if (it_codes != rsp.end() && !it_codes->is_null()) {
        if (!parse_codes_array(*it_codes, out_codes, err)) {
            err = "invalid 'audio_codes' in LLM API response: " + err;
            return false;
        }
        if (out_codes.empty()) {
            err = "LLM API response contains empty 'audio_codes'";
            return false;
        }
        return true;
    }

    std::string text;
    if (extract_text_from_llm_api_json(rsp, text) && extract_codes_from_text(text, out_codes)) {
        return true;
    }

    err = "LLM API response did not include codes (expected 'codes' / 'codes_values' / text with <|s_...|>)";
    return false;
}

static bool llm_api_enabled(const cli_params & p) {
    return !p.llm_api_url.empty();
}

static bool generate_audio_codes_external(
        const cli_params & p,
        std::vector<int32_t> & out_codes,
        std::string & err) {
    if (!llm_api_enabled(p)) {
        err = "external LLM API is not configured";
        return false;
    }

    parsed_llm_api_url endpoint;
    if (!parse_llm_api_url(p.llm_api_url, endpoint, err)) {
        return false;
    }

    json body;
    if (p.llm_api_mode_value == LLM_API_MODE_OPENAI_CHAT) {
        if (!p.llm_api_model.empty()) {
            body["model"] = p.llm_api_model;
        }
        body["messages"] = json::array({
            json {
                {"role", "user"},
                {"content", p.prompt},
            }
        });
        body["max_tokens"] = p.n_predict;
        body["temperature"] = p.temp;
        body["top_p"] = p.top_p;
        body["stream"] = false;
    } else {
        body["text"] = p.prompt;
        body["prompt"] = p.prompt;
        body["n_predict"] = p.n_predict;
        body["temperature"] = p.temp;
        body["top_p"] = p.top_p;
        body["top_k"] = p.top_k;
        body["repeat_penalty"] = p.repeat_penalty;
        body["seed"] = p.seed;
        if (!p.llm_api_model.empty()) {
            body["model"] = p.llm_api_model;
        }
    }

    httplib::Headers headers;
    headers_upsert_ci(headers, "Content-Type", "application/json");
    for (const auto & kv : p.llm_api_headers) {
        headers_upsert_ci(headers, kv.first, kv.second);
    }
    if (!p.llm_api_key.empty() && !headers_find_ci(headers, "Authorization")) {
        headers_upsert_ci(headers, "Authorization", "Bearer " + p.llm_api_key);
    }
    std::string content_type = "application/json";
    headers_find_ci(headers, "Content-Type", &content_type);
    for (auto it = headers.begin(); it != headers.end();) {
        if (ieq_ascii(it->first, "Content-Type")) {
            it = headers.erase(it);
        } else {
            ++it;
        }
    }

    const std::string payload = body.dump();
    httplib::Result res;

    if (endpoint.https) {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
        httplib::SSLClient cli(endpoint.host, endpoint.port);
        cli.set_follow_location(true);
        cli.set_connection_timeout(p.llm_api_timeout_sec, 0);
        cli.set_read_timeout(p.llm_api_timeout_sec, 0);
        cli.set_write_timeout(p.llm_api_timeout_sec, 0);
        res = cli.Post(endpoint.path.c_str(), headers, payload, content_type.c_str());
#else
        err = "https URL requires CPPHTTPLIB_OPENSSL_SUPPORT";
        return false;
#endif
    } else {
        httplib::Client cli(endpoint.host, endpoint.port);
        cli.set_follow_location(true);
        cli.set_connection_timeout(p.llm_api_timeout_sec, 0);
        cli.set_read_timeout(p.llm_api_timeout_sec, 0);
        cli.set_write_timeout(p.llm_api_timeout_sec, 0);
        res = cli.Post(endpoint.path.c_str(), headers, payload, content_type.c_str());
    }

    if (!res) {
        err = "LLM API request failed: " + httplib::to_string(res.error());
        return false;
    }

    if (res->status < 200 || res->status >= 300) {
        err = "LLM API HTTP " + std::to_string(res->status) + ": " + truncate_text(res->body);
        return false;
    }

    json rsp;
    try {
        rsp = json::parse(res->body.empty() ? "{}" : res->body);
    } catch (const std::exception & e) {
        if (extract_codes_from_text(res->body, out_codes)) {
            return true;
        }
        err = std::string("LLM API returned non-JSON response: ") + e.what() + " body=" + truncate_text(res->body);
        return false;
    }

    return parse_codes_from_llm_api_response(rsp, out_codes, err);
}

static bool needs_value(int i, int argc) {
    return i + 1 < argc;
}

static bool parse_args(int argc, char ** argv, cli_params & p) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            p.show_help = true;
        } else if (arg == "-m" || arg == "--model") {
            if (!needs_value(i, argc)) return false;
            p.model = argv[++i];
        } else if (arg == "-mv" || arg == "--model-vocoder") {
            if (!needs_value(i, argc)) return false;
            p.model_vocoder = argv[++i];
        } else if (arg == "-p" || arg == "--prompt") {
            if (!needs_value(i, argc)) return false;
            p.prompt = argv[++i];
        } else if (arg == "--prompt-file") {
            if (!needs_value(i, argc)) return false;
            p.prompt_file = argv[++i];
        } else if (arg == "-o" || arg == "--output") {
            if (!needs_value(i, argc)) return false;
            p.output = argv[++i];
        } else if (arg == "-n" || arg == "--n-predict") {
            if (!needs_value(i, argc) || !parse_i32(argv[++i], p.n_predict)) return false;
        } else if (arg == "--temp") {
            if (!needs_value(i, argc) || !parse_f32(argv[++i], p.temp)) return false;
        } else if (arg == "--top-p") {
            if (!needs_value(i, argc) || !parse_f32(argv[++i], p.top_p)) return false;
        } else if (arg == "--top-k") {
            if (!needs_value(i, argc) || !parse_i32(argv[++i], p.top_k)) return false;
        } else if (arg == "--repeat-penalty") {
            if (!needs_value(i, argc) || !parse_f32(argv[++i], p.repeat_penalty)) return false;
        } else if (arg == "--seed") {
            if (!needs_value(i, argc) || !parse_u32(argv[++i], p.seed)) return false;
        } else if (arg == "--threads") {
            if (!needs_value(i, argc) || !parse_i32(argv[++i], p.n_threads)) return false;
        } else if (arg == "--ctx-size") {
            if (!needs_value(i, argc) || !parse_i32(argv[++i], p.n_ctx)) return false;
        } else if (arg == "-ngl" || arg == "--n-gpu-layers") {
            if (!needs_value(i, argc) || !parse_i32(argv[++i], p.n_gpu_layers)) return false;
        } else if (arg == "-fa" || arg == "--flash-attn") {
            if (!needs_value(i, argc) || !parse_flash_attn_type(argv[++i], p.flash_attn_type)) return false;

        } else if (arg == "--llm-api-url") {
            if (!needs_value(i, argc)) return false;
            p.llm_api_url = argv[++i];
        } else if (arg == "--llm-api-key") {
            if (!needs_value(i, argc)) return false;
            p.llm_api_key = argv[++i];
        } else if (arg == "--llm-api-model") {
            if (!needs_value(i, argc)) return false;
            p.llm_api_model = argv[++i];
        } else if (arg == "--llm-api-headers") {
            if (!needs_value(i, argc)) return false;
            std::string perr;
            if (!parse_llm_api_headers_json(argv[++i], p.llm_api_headers, perr)) {
                std::fprintf(stderr, "%s\n", perr.c_str());
                return false;
            }
        } else if (arg == "--llm-api-timeout") {
            if (!needs_value(i, argc) || !parse_i32(argv[++i], p.llm_api_timeout_sec)) return false;
        } else if (arg == "--llm-api-mode") {
            if (!needs_value(i, argc) || !parse_llm_api_mode(argv[++i], p.llm_api_mode_value)) return false;

        } else if (arg == "--tts-mio-codes") {
            if (!needs_value(i, argc)) return false;
            p.inline_codes_text = argv[++i];
        } else if (arg == "--tts-wavlm-model") {
            if (!needs_value(i, argc)) return false;
            p.wavlm_model = argv[++i];
        } else if (arg == "--tts-reference-audio") {
            if (!needs_value(i, argc)) return false;
            p.reference_audio = argv[++i];
        } else if (arg == "--tts-max-reference-seconds") {
            if (!needs_value(i, argc) || !parse_f32(argv[++i], p.max_reference_seconds)) return false;
        } else if (arg == "--tts-reference-dir") {
            if (!needs_value(i, argc)) return false;
            p.reference_dir = argv[++i];
        } else if (arg == "--tts-remove-reference-key") {
            if (!needs_value(i, argc)) return false;
            p.remove_reference_key = argv[++i];
        } else if (arg == "--tts-mio-embedding-in") {
            if (!needs_value(i, argc)) return false;
            p.embedding_in = argv[++i];
        } else if (arg == "--tts-mio-default-embedding-in" || arg == "-emb") {
            if (!needs_value(i, argc)) return false;
            p.embedding_default_in = argv[++i];
        } else if (arg == "--tts-mio-embedding-out") {
            if (!needs_value(i, argc)) return false;
            p.embedding_out = argv[++i];
        } else if (arg == "--tts-mio-codes-in") {
            if (!needs_value(i, argc)) return false;
            p.codes_in = argv[++i];
        } else if (arg == "--tts-mio-codes-out") {
            if (!needs_value(i, argc)) return false;
            p.codes_out = argv[++i];
        } else if (arg == "--tts-mio-codes-only") {
            p.codes_only = true;
        } else if (arg == "--tts-mio-embedding-only") {
            p.embedding_only = true;
        } else {
            return false;
        }
    }

    if (p.show_help) {
        return true;
    }

    if (!p.remove_reference_key.empty()) {
        if (p.reference_dir.empty()) {
            std::fprintf(stderr, "--tts-remove-reference-key requires --tts-reference-dir\n");
            return false;
        }
        if (!is_valid_reference_key(p.remove_reference_key)) {
            std::fprintf(stderr, "--tts-remove-reference-key is invalid\n");
            return false;
        }
        return true;
    }

    if (p.model_vocoder.empty()) {
        return false;
    }

    if (!p.prompt.empty() && !p.prompt_file.empty()) {
        std::fprintf(stderr, "--prompt and --prompt-file cannot be used together\n");
        return false;
    }

    if (!p.prompt_file.empty()) {
        std::string perr;
        if (!load_text_file(p.prompt_file, p.prompt, perr)) {
            std::fprintf(stderr, "%s\n", perr.c_str());
            return false;
        }
        p.prompt = trim_copy(p.prompt);
    }

    if (p.llm_api_timeout_sec < 1) {
        std::fprintf(stderr, "--llm-api-timeout must be >= 1\n");
        return false;
    }

    if (p.llm_api_url.empty()) {
        const char * v = std::getenv("MIO_TTS_LLM_API_URL");
        if (v != nullptr && v[0] != '\0') {
            p.llm_api_url = v;
        }
    }
    if (p.llm_api_key.empty()) {
        const char * v = std::getenv("MIO_TTS_LLM_API_KEY");
        if (v != nullptr && v[0] != '\0') {
            p.llm_api_key = v;
        }
    }
    if (p.llm_api_model.empty()) {
        const char * v = std::getenv("MIO_TTS_LLM_API_MODEL");
        if (v != nullptr && v[0] != '\0') {
            p.llm_api_model = v;
        }
    }
    if (p.llm_api_headers.empty()) {
        const char * v = std::getenv("MIO_TTS_LLM_API_HEADERS");
        if (v != nullptr && v[0] != '\0') {
            std::string perr;
            if (!parse_llm_api_headers_json(v, p.llm_api_headers, perr)) {
                std::fprintf(stderr, "%s\n", perr.c_str());
                return false;
            }
        }
    }

    if (!p.inline_codes_text.empty()) {
        std::string perr;
        if (!parse_inline_codes_arg(p.inline_codes_text, p.inline_codes, perr)) {
            std::fprintf(stderr, "%s\n", perr.c_str());
            return false;
        }
    }

    if (llm_api_enabled(p) && !p.model.empty()) {
        std::fprintf(stderr, "info: --llm-api-url is set; ignoring local --model\n");
        p.model.clear();
    }

    const bool need_codes = !p.embedding_only || p.codes_only || !p.codes_out.empty();
    if (need_codes && p.codes_in.empty() && p.inline_codes.empty() && p.prompt.empty()) {
        std::fprintf(stderr, "either --prompt/--prompt-file, --tts-mio-codes, or --tts-mio-codes-in is required\n");
        return false;
    }

    if (!p.prompt.empty() && p.model.empty() && !llm_api_enabled(p)) {
        std::fprintf(stderr, "--prompt requires --model or --llm-api-url\n");
        return false;
    }

    if (p.embedding_only && p.reference_audio.empty() && p.embedding_in.empty() && p.embedding_default_in.empty()) {
        std::fprintf(stderr, "--tts-mio-embedding-only requires --tts-reference-audio or --tts-mio-embedding-in or --tts-mio-default-embedding-in\n");
        return false;
    }

    return true;
}

static int resolve_threads(int n_threads) {
    if (n_threads > 0) {
        return n_threads;
    }
    const unsigned hc = std::thread::hardware_concurrency();
    return (int) (hc > 0 ? hc : 1);
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

static llama_sampler * make_sampler(const cli_params & p) {
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(64, p.repeat_penalty, 0.0f, 0.0f));

    if (p.top_k > 0) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(p.top_k));
    }
    if (p.top_p > 0.0f && p.top_p < 1.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(p.top_p, 1));
    }

    if (p.temp <= 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(p.temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(p.seed));
    }

    return smpl;
}

static bool generate_audio_tokens(
        const cli_params & p,
        llama_model * model,
        std::vector<llama_token> & generated,
        std::string & err) {
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const std::string prompt_chat =
        std::string("<|im_start|>user\n") + p.prompt + "<|im_end|>\n<|im_start|>assistant\n";

    std::vector<llama_token> prompt_tokens;
    if (!tokenize_text(vocab, prompt_chat, false, true, prompt_tokens, err)) {
        return false;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = (uint32_t) std::max<int32_t>(p.n_ctx, (int32_t) prompt_tokens.size() + p.n_predict + 32);
    cparams.n_batch = (uint32_t) std::max<int32_t>((int32_t) prompt_tokens.size(), 512);
    cparams.flash_attn_type = p.flash_attn_type;
    cparams.n_threads = resolve_threads(p.n_threads);
    cparams.n_threads_batch = cparams.n_threads;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        err = "failed to create llama context";
        return false;
    }

    llama_sampler * sampler = make_sampler(p);

    generated.clear();

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), (int32_t) prompt_tokens.size());
    if (llama_decode(ctx, batch) != 0) {
        llama_sampler_free(sampler);
        llama_free(ctx);
        err = "llama_decode failed on prompt";
        return false;
    }

    for (int32_t i = 0; i < p.n_predict; ++i) {
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
            err = "llama_decode failed during generation";
            return false;
        }
    }

    llama_sampler_free(sampler);
    llama_free(ctx);
    return true;
}

int main(int argc, char ** argv) {
    cli_params p;
    if (!parse_args(argc, argv, p)) {
        print_usage(argv[0]);
        return 1;
    }
    if (p.show_help) {
        print_usage(argv[0]);
        return 0;
    }

    if (!p.remove_reference_key.empty()) {
        const std::string path = make_reference_added_output_path(p.reference_dir, p.remove_reference_key);
        std::error_code ec;
        const bool removed = std::filesystem::remove(path, ec);
        if (ec) {
            std::fprintf(stderr, "failed to remove reference key '%s': %s\n", p.remove_reference_key.c_str(), ec.message().c_str());
            return 1;
        }
        if (!removed) {
            std::fprintf(stderr, "reference key not found: %s (%s)\n", p.remove_reference_key.c_str(), path.c_str());
            return 1;
        }
        std::fprintf(stderr, "removed reference key '%s': %s\n", p.remove_reference_key.c_str(), path.c_str());
        return 0;
    }

    llama_backend_init();
    ggml_backend_load_all();

    char err[1024] = {0};

    mio_tts_context * mio = mio_tts_init_from_file(
            p.model_vocoder.c_str(),
            p.wavlm_model.empty() ? nullptr : p.wavlm_model.c_str(),
            err,
            sizeof(err));
    if (mio == nullptr) {
        std::fprintf(stderr, "mio_tts_init_from_file failed: %s\n", err);
        return 1;
    }

    const bool need_codes = !p.embedding_only || p.codes_only || !p.codes_out.empty();

    int32_t * codes = nullptr;
    size_t n_codes = 0;

    if (need_codes) {
        if (!p.inline_codes.empty()) {
            codes = (int32_t *) std::malloc(p.inline_codes.size() * sizeof(int32_t));
            if (codes == nullptr) {
                std::fprintf(stderr, "failed to allocate codes buffer\n");
                mio_tts_free(mio);
                return 1;
            }
            std::memcpy(codes, p.inline_codes.data(), p.inline_codes.size() * sizeof(int32_t));
            n_codes = p.inline_codes.size();
        } else if (!p.codes_in.empty()) {
            if (!mio_tts_codes_load(p.codes_in.c_str(), &codes, &n_codes, err, sizeof(err))) {
                std::fprintf(stderr, "mio_tts_codes_load failed: %s\n", err);
                mio_tts_free(mio);
                return 1;
            }
        } else {
            if (llm_api_enabled(p)) {
                std::vector<int32_t> ext_codes;
                std::string gen_err;
                if (!generate_audio_codes_external(p, ext_codes, gen_err)) {
                    std::fprintf(stderr, "token generation failed: %s\n", gen_err.c_str());
                    mio_tts_free(mio);
                    return 1;
                }
                if (ext_codes.empty()) {
                    std::fprintf(stderr, "token generation failed: external LLM API returned empty codes\n");
                    mio_tts_free(mio);
                    return 1;
                }

                codes = (int32_t *) std::malloc(ext_codes.size() * sizeof(int32_t));
                if (codes == nullptr) {
                    std::fprintf(stderr, "failed to allocate codes buffer\n");
                    mio_tts_free(mio);
                    return 1;
                }
                std::memcpy(codes, ext_codes.data(), ext_codes.size() * sizeof(int32_t));
                n_codes = ext_codes.size();
            } else {
                llama_model_params mparams = llama_model_default_params();
                mparams.n_gpu_layers = p.n_gpu_layers;

                llama_model * model = llama_model_load_from_file(p.model.c_str(), mparams);
                if (model == nullptr) {
                    std::fprintf(stderr, "failed to load LLM model: %s\n", p.model.c_str());
                    mio_tts_free(mio);
                    return 1;
                }

                const llama_vocab * vocab = llama_model_get_vocab(model);
                mio_tts_vocab_map * vmap = mio_tts_vocab_map_init(vocab, err, sizeof(err));
                if (vmap == nullptr) {
                    std::fprintf(stderr, "mio_tts_vocab_map_init failed: %s\n", err);
                    llama_model_free(model);
                    mio_tts_free(mio);
                    return 1;
                }

                std::vector<llama_token> generated;
                std::string gen_err;
                if (!generate_audio_tokens(p, model, generated, gen_err)) {
                    std::fprintf(stderr, "token generation failed: %s\n", gen_err.c_str());
                    mio_tts_vocab_map_free(vmap);
                    llama_model_free(model);
                    mio_tts_free(mio);
                    return 1;
                }

                if (!mio_tts_tokens_to_codes(vmap, generated.data(), generated.size(), &codes, &n_codes, err, sizeof(err))) {
                    std::fprintf(stderr, "mio_tts_tokens_to_codes failed: %s\n", err);
                    mio_tts_vocab_map_free(vmap);
                    llama_model_free(model);
                    mio_tts_free(mio);
                    return 1;
                }

                mio_tts_vocab_map_free(vmap);
                llama_model_free(model);
            }
        }

        std::fprintf(stderr, "codes audio size: %zu\n", n_codes);
    }

    if (!p.codes_out.empty()) {
        if (codes == nullptr || n_codes == 0) {
            std::fprintf(stderr, "--tts-mio-codes-out was provided but no codes are available\n");
            mio_tts_free(mio);
            return 1;
        }
        if (!mio_tts_codes_save(p.codes_out.c_str(), codes, n_codes, err, sizeof(err))) {
            std::fprintf(stderr, "mio_tts_codes_save failed: %s\n", err);
            mio_tts_codes_free(codes);
            mio_tts_free(mio);
            return 1;
        }
        std::fprintf(stderr, "wrote codes: %s\n", p.codes_out.c_str());
    }

    float * embedding = nullptr;
    size_t n_embedding = 0;

    if (!p.embedding_in.empty()) {
        if (!mio_tts_embedding_load_gguf(p.embedding_in.c_str(), &embedding, &n_embedding, err, sizeof(err))) {
            std::fprintf(stderr, "mio_tts_embedding_load_gguf failed: %s\n", err);
            mio_tts_codes_free(codes);
            mio_tts_free(mio);
            return 1;
        }
    } else if (!p.reference_audio.empty()) {
        mio_tts_params mparams = mio_tts_default_params();
        mparams.n_threads = p.n_threads;
        mparams.max_reference_seconds = p.max_reference_seconds;
        mparams.wavlm_flash_attn_type = p.flash_attn_type;
        mparams.miocodec_flash_attn_type = p.flash_attn_type;

        if (!mio_tts_reference_to_embedding(
                    mio,
                    p.reference_audio.c_str(),
                    mparams,
                    &embedding,
                    &n_embedding,
                    err,
                    sizeof(err))) {
            std::fprintf(stderr, "mio_tts_reference_to_embedding failed: %s\n", err);
            mio_tts_codes_free(codes);
            mio_tts_free(mio);
            return 1;
        }
    } else if (!p.embedding_default_in.empty() && mio_tts_context_is_dynamic_global(mio)) {
        if (!mio_tts_embedding_load_gguf(p.embedding_default_in.c_str(), &embedding, &n_embedding, err, sizeof(err))) {
            std::fprintf(stderr, "mio_tts_embedding_load_gguf (default) failed: %s\n", err);
            mio_tts_codes_free(codes);
            mio_tts_free(mio);
            return 1;
        }
    }

    if (!p.embedding_out.empty()) {
        if (embedding == nullptr || n_embedding == 0) {
            std::fprintf(stderr, "warning: --tts-mio-embedding-out was provided but no embedding is available\n");
        } else if (!mio_tts_embedding_save_gguf(p.embedding_out.c_str(), embedding, n_embedding, err, sizeof(err))) {
            std::fprintf(stderr, "mio_tts_embedding_save_gguf failed: %s\n", err);
            mio_tts_embedding_free(embedding);
            mio_tts_codes_free(codes);
            mio_tts_free(mio);
            return 1;
        } else {
            std::fprintf(stderr, "wrote embedding: %s\n", p.embedding_out.c_str());
        }
    }

    if (p.codes_only || p.embedding_only) {
        if (p.codes_only && p.embedding_only) {
            std::fprintf(stderr, "skip synthesis: --tts-mio-codes-only + --tts-mio-embedding-only\n");
        } else if (p.codes_only) {
            std::fprintf(stderr, "skip synthesis: --tts-mio-codes-only\n");
        } else {
            std::fprintf(stderr, "skip synthesis: --tts-mio-embedding-only\n");
        }

        mio_tts_embedding_free(embedding);
        mio_tts_codes_free(codes);
        mio_tts_free(mio);
        return 0;
    }

    if (codes == nullptr || n_codes == 0) {
        std::fprintf(stderr, "synthesis requires codes (--prompt/--prompt-file, --tts-mio-codes, or --tts-mio-codes-in)\n");
        mio_tts_embedding_free(embedding);
        mio_tts_codes_free(codes);
        mio_tts_free(mio);
        return 1;
    }

    float * audio = nullptr;
    size_t n_audio = 0;
    int32_t sample_rate = 0;

    mio_tts_params synth_params = mio_tts_default_params();
    synth_params.n_threads = p.n_threads;
    synth_params.max_reference_seconds = p.max_reference_seconds;
    synth_params.wavlm_flash_attn_type = p.flash_attn_type;
    synth_params.miocodec_flash_attn_type = p.flash_attn_type;

    if (!mio_tts_synthesize(
                mio,
                codes,
                n_codes,
                embedding,
                n_embedding,
                synth_params,
                &audio,
                &n_audio,
                &sample_rate,
                err,
                sizeof(err))) {
        std::fprintf(stderr, "mio_tts_synthesize failed: %s\n", err);
        mio_tts_embedding_free(embedding);
        mio_tts_codes_free(codes);
        mio_tts_free(mio);
        return 1;
    }

    if (!mio_tts_audio_save_wav16(p.output.c_str(), audio, n_audio, sample_rate, err, sizeof(err))) {
        std::fprintf(stderr, "mio_tts_audio_save_wav16 failed: %s\n", err);
        mio_tts_audio_free(audio);
        mio_tts_embedding_free(embedding);
        mio_tts_codes_free(codes);
        mio_tts_free(mio);
        return 1;
    }

    std::fprintf(stderr, "audio written to file '%s'\n", p.output.c_str());

    mio_tts_audio_free(audio);
    mio_tts_embedding_free(embedding);
    mio_tts_codes_free(codes);
    mio_tts_free(mio);

    return 0;
}
