#include "llama.h"
#include "mio-tts-lib.h"
#include "ggml-backend.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

using json = nlohmann::ordered_json;

enum llm_api_mode {
    LLM_API_MODE_OPENAI_CHAT = 0,
    LLM_API_MODE_GENERIC = 1,
};

static const char * k_web_index_html = R"MIOWEB(<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>mio-tts-server</title>
  <link rel="stylesheet" href="/mio-ui.css">
</head>
<body>
  <div class="wrap">
    <h1>mio-tts-server</h1>
    <p class="sub">Text to speech + reference voice cache</p>

    <section class="card">
      <h2>Reference Cache</h2>
      <div class="row">
        <button id="btn-refresh" type="button">Refresh List</button>
        <span id="ref-count" class="muted">0 references</span>
      </div>
      <div class="row">
        <label for="ref-select">Reference key</label>
        <select id="ref-select"></select>
      </div>
      <div class="row">
        <button id="btn-del-ref" type="button">Delete Selected</button>
      </div>
      <div id="del-ref-status" class="status"></div>
      <div class="hint">Use selected key for synthesis. You can create or upload references below.</div>
    </section>

    <section class="card">
      <h2>Create Reference From Audio</h2>
      <div class="grid">
        <label for="gen-ref-key">key</label>
        <input id="gen-ref-key" type="text" placeholder="jp_female">
        <label for="gen-ref-audio">audio file</label>
        <input id="gen-ref-audio" type="file" accept="audio/*">
      </div>
      <div class="row">
        <button id="btn-gen-ref" type="button">Generate Reference</button>
        <a id="gen-ref-download" class="hidden" download>Download GGUF</a>
      </div>
      <div id="gen-ref-status" class="status"></div>
    </section>

    <section class="card">
      <h2>Add Reference (GGUF)</h2>
      <div class="grid">
        <label for="add-ref-key">key</label>
        <input id="add-ref-key" type="text" placeholder="jp_female">
        <label for="add-ref-file">gguf file</label>
        <input id="add-ref-file" type="file" accept=".gguf,application/octet-stream">
      </div>
      <div class="row">
        <button id="btn-add-ref" type="button">Add Reference</button>
      </div>
      <div id="add-ref-status" class="status"></div>
    </section>

    <section class="card">
      <h2>Synthesis</h2>
      <label for="tts-text">Text</label>
      <textarea id="tts-text" rows="3">こんにちわ、今日はいい天気ですね。</textarea>
      <div class="grid">
        <label for="tts-reference-key">reference_key</label>
        <input id="tts-reference-key" type="text" placeholder="jp_female">
      </div>
      <div class="params">
        <label>temperature <input id="p-temp" type="number" step="0.01" value="0.8"></label>
        <label>top_p <input id="p-top-p" type="number" step="0.01" value="1.0"></label>
        <label>top_k <input id="p-top-k" type="number" step="1" value="50"></label>
        <label>repeat_penalty <input id="p-repeat" type="number" step="0.01" value="1.0"></label>
        <label>n_predict <input id="p-n-predict" type="number" step="1" value="700"></label>
      </div>
      <div class="row">
        <label class="checkline"><input id="tts-stream-read" type="checkbox" checked>streaming download mode</label>
        <label class="checkline"><input id="tts-stream-tokens" type="checkbox">SSE token streaming</label>
      </div>
      <pre id="tts-tokens" class="hidden" style="max-height:120px;overflow-y:auto;font-size:11px;background:#111;color:#0f0;padding:4px;margin:4px 0"></pre>
      <div class="row">
        <button id="btn-tts" type="button">Generate Speech</button>
        <span id="tts-meta" class="muted"></span>
      </div>
      <div id="tts-status" class="status"></div>
      <audio id="audio-out" controls></audio>
      <div class="row">
        <a id="audio-download" class="hidden" download>Download WAV</a>
      </div>
    </section>
  </div>
  <script src="/mio-ui.js"></script>
</body>
</html>
)MIOWEB";

static const char * k_web_style_css = R"MIOCSS(:root {
  --bg: #f4f6f8;
  --card: #ffffff;
  --ink: #1e293b;
  --muted: #64748b;
  --line: #cbd5e1;
  --accent: #0f766e;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  color: var(--ink);
  background: linear-gradient(160deg, #ecfeff 0%, var(--bg) 50%, #f8fafc 100%);
}
.wrap {
  max-width: 920px;
  margin: 24px auto 64px;
  padding: 0 16px;
}
h1 { margin: 0 0 4px; }
h2 { margin: 0 0 12px; font-size: 1.05rem; }
.sub { margin: 0 0 18px; color: var(--muted); }
.card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 14px;
  margin: 12px 0;
  box-shadow: 0 8px 28px rgba(2, 6, 23, 0.04);
}
label { font-size: 0.9rem; color: #334155; }
input, textarea, select, button {
  font: inherit;
}
input:not([type="checkbox"]):not([type="radio"]), textarea, select {
  width: 100%;
  margin-top: 4px;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 8px 10px;
  background: #fff;
}
textarea { resize: vertical; min-height: 90px; }
.grid {
  display: grid;
  grid-template-columns: 130px 1fr;
  gap: 10px 12px;
  align-items: center;
}
.params {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 10px;
  margin: 10px 0;
}
.params label {
  display: block;
  font-size: 0.82rem;
}
.row {
  display: flex;
  gap: 10px;
  align-items: center;
  margin-top: 10px;
  flex-wrap: wrap;
}
.checkline {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-size: 0.9rem;
}
.checkline input[type="checkbox"] {
  margin: 0;
}
button {
  border: 0;
  border-radius: 8px;
  background: var(--accent);
  color: #fff;
  padding: 8px 12px;
  cursor: pointer;
  font-weight: 600;
}
button:disabled { opacity: 0.55; cursor: default; }
.status {
  margin-top: 8px;
  font-size: 0.92rem;
  color: #0f172a;
  white-space: pre-wrap;
}
.hint, .muted { color: var(--muted); font-size: 0.86rem; }
.hidden { display: none; }
audio { width: 100%; margin-top: 12px; }
)MIOCSS";

static const char * k_web_app_js = R"MIOJS((() => {
  const el = (id) => document.getElementById(id);
  const ui = {
    refSelect: el("ref-select"),
    refCount: el("ref-count"),
    btnRefresh: el("btn-refresh"),
    btnDelRef: el("btn-del-ref"),
    delRefStatus: el("del-ref-status"),
    genRefKey: el("gen-ref-key"),
    genRefAudio: el("gen-ref-audio"),
    btnGenRef: el("btn-gen-ref"),
    genRefStatus: el("gen-ref-status"),
    genRefDownload: el("gen-ref-download"),
    addRefKey: el("add-ref-key"),
    addRefFile: el("add-ref-file"),
    btnAddRef: el("btn-add-ref"),
    addRefStatus: el("add-ref-status"),
    text: el("tts-text"),
    ttsReferenceKey: el("tts-reference-key"),
    pTemp: el("p-temp"),
    pTopP: el("p-top-p"),
    pTopK: el("p-top-k"),
    pRepeat: el("p-repeat"),
    pNPredict: el("p-n-predict"),
    streamRead: el("tts-stream-read"),
    streamTokens: el("tts-stream-tokens"),
    ttsTokens: el("tts-tokens"),
    btnTts: el("btn-tts"),
    ttsStatus: el("tts-status"),
    ttsMeta: el("tts-meta"),
    audioOut: el("audio-out"),
    audioDownload: el("audio-download"),
  };

  let currentAudioUrl = "";
  let currentRefGgufUrl = "";

  function setStatus(node, message, isError = false) {
    node.textContent = message;
    node.style.color = isError ? "#b91c1c" : "#0f172a";
  }

  async function parseApiError(response) {
    const ct = response.headers.get("content-type") || "";
    if (ct.includes("application/json")) {
      const body = await response.json().catch(() => ({}));
      return body?.error?.message || JSON.stringify(body);
    }
    return await response.text();
  }

  async function refreshReferences(selected = "") {
    const res = await fetch("/mio/references");
    if (!res.ok) {
      const msg = await parseApiError(res);
      throw new Error(msg || `HTTP ${res.status}`);
    }
    const body = await res.json();
    const refs = Array.isArray(body.references) ? body.references : [];
    refs.sort((a, b) => String(a.key).localeCompare(String(b.key)));

    ui.refSelect.innerHTML = "";
    for (const r of refs) {
      const opt = document.createElement("option");
      opt.value = r.key;
      opt.textContent = `${r.key} (dim=${r.embedding_dim})`;
      ui.refSelect.appendChild(opt);
    }
    if (selected) {
      ui.refSelect.value = selected;
    }
    if (!ui.ttsReferenceKey.value.trim() || selected) {
      ui.ttsReferenceKey.value = ui.refSelect.value || "";
    }
    ui.refCount.textContent = `${refs.length} references`;
  }

  async function onGenerateReference() {
    const key = ui.genRefKey.value.trim();
    const file = ui.genRefAudio.files?.[0];
    if (!key) throw new Error("key is required");
    if (!file) throw new Error("audio file is required");

    const fd = new FormData();
    fd.append("reference_key", key);
    fd.append("audio", file);

    const res = await fetch("/mio/generate_reference", { method: "POST", body: fd });
    if (!res.ok) {
      const msg = await parseApiError(res);
      throw new Error(msg || `HTTP ${res.status}`);
    }

    const blob = await res.blob();
    if (currentRefGgufUrl) {
      URL.revokeObjectURL(currentRefGgufUrl);
    }
    currentRefGgufUrl = URL.createObjectURL(blob);
    ui.genRefDownload.href = currentRefGgufUrl;
    ui.genRefDownload.download = `${key}.emb.gguf`;
    ui.genRefDownload.classList.remove("hidden");

    await refreshReferences(key);
    setStatus(ui.genRefStatus, `reference generated: key=${key}, size=${blob.size} bytes`);
  }

  async function onAddReference() {
    const key = ui.addRefKey.value.trim();
    const file = ui.addRefFile.files?.[0];
    if (!key) throw new Error("key is required");
    if (!file) throw new Error("gguf file is required");

    const fd = new FormData();
    fd.append("key", key);
    fd.append("file", file);

    const res = await fetch("/mio/add_reference", { method: "POST", body: fd });
    if (!res.ok) {
      const msg = await parseApiError(res);
      throw new Error(msg || `HTTP ${res.status}`);
    }
    const body = await res.json();
    await refreshReferences(key);
    setStatus(
      ui.addRefStatus,
      `added key=${body.reference_key}, dim=${body.embedding_dim}, replaced=${body.replaced ? "true" : "false"}`
    );
  }

  async function onDeleteReference() {
    const key = (ui.refSelect.value || ui.ttsReferenceKey.value || "").trim();
    if (!key) throw new Error("reference_key is required");

    const res = await fetch("/mio/delete_reference", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reference_key: key }),
    });
    if (!res.ok) {
      const msg = await parseApiError(res);
      throw new Error(msg || `HTTP ${res.status}`);
    }

    const body = await res.json();
    await refreshReferences();
    setStatus(
      ui.delRefStatus,
      `deleted key=${body.reference_key}, cache=${body.reference_cache}, saved_file_removed=${body.removed_saved_file ? "true" : "false"}`
    );
  }

  function parseSSE(text) {
    const events = [];
    const blocks = text.split("\n\n");
    for (const block of blocks) {
      if (!block.trim()) continue;
      let event = "", data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7);
        else if (line.startsWith("data: ")) data = line.slice(6);
      }
      if (event || data) events.push({ event, data });
    }
    return events;
  }

  async function onSynthesizeSSE(payload, referenceKey) {
    payload.stream_tokens = true;
    ui.ttsTokens.classList.remove("hidden");
    ui.ttsTokens.textContent = "";

    const res = await fetch("/mio/tts/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const msg = await parseApiError(res);
      throw new Error(msg || `HTTP ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let tokenCount = 0;
    let codeCount = 0;
    let audioB64 = null;
    let audioMeta = null;
    let ttftMs = 0;
    const t0 = performance.now();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const parts = buffer.split("\n\n");
      buffer = parts.pop();

      for (const block of parts) {
        if (!block.trim()) continue;
        let event = "", data = "";
        for (const line of block.split("\n")) {
          if (line.startsWith("event: ")) event = line.slice(7);
          else if (line.startsWith("data: ")) data = line.slice(6);
        }

        if (event === "token") {
          tokenCount++;
          if (tokenCount === 1) ttftMs = performance.now() - t0;
          try {
            const tok = JSON.parse(data);
            if (tok.code !== undefined) codeCount++;
            const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
            ui.ttsTokens.textContent += (tok.code !== undefined ? `[${tok.code}]` : ".") +
              (tokenCount % 20 === 0 ? ` (${tokenCount}tok ${elapsed}s)\n` : "");
            ui.ttsTokens.scrollTop = ui.ttsTokens.scrollHeight;
          } catch {}
          setStatus(ui.ttsStatus, `generating tokens: ${tokenCount} (${codeCount} codes) TTFT=${ttftMs.toFixed(0)}ms`);
        } else if (event === "generation_complete") {
          try {
            const info = JSON.parse(data);
            setStatus(ui.ttsStatus, `LLM done: ${info.n_tokens} tokens, ${info.n_codes} codes, ${info.llm_ms?.toFixed(1)}ms. Synthesizing...`);
          } catch {}
        } else if (event === "audio_meta") {
          try { audioMeta = JSON.parse(data); } catch {}
        } else if (event === "audio_data") {
          audioB64 = data;
          setStatus(ui.ttsStatus, `audio received: ${(performance.now() - t0).toFixed(0)}ms`);
        } else if (event === "error") {
          try {
            const err = JSON.parse(data);
            throw new Error(err.error || "SSE error");
          } catch (e) { throw e; }
        }
      }
    }

    if (!audioB64) throw new Error("no audio received");

    const raw = atob(audioB64);
    const bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
    const blob = new Blob([bytes], { type: "audio/wav" });

    if (currentAudioUrl) URL.revokeObjectURL(currentAudioUrl);
    currentAudioUrl = URL.createObjectURL(blob);
    ui.audioOut.src = currentAudioUrl;
    ui.audioDownload.href = currentAudioUrl;
    ui.audioDownload.download = `tts-${Date.now()}.wav`;
    ui.audioDownload.classList.remove("hidden");

    const audioArrivalMs = performance.now() - t0;
    const sr = audioMeta?.sample_rate || "-";
    const na = audioMeta?.n_audio || "-";
    ui.ttsMeta.textContent = `ref=${referenceKey}, sr=${sr}, samples=${na}, synth=${audioMeta?.synth_ms?.toFixed(0) || "-"}ms`;

    await ui.audioOut.play().catch(() => {});
    setStatus(ui.ttsStatus, `TTFT=${ttftMs.toFixed(0)}ms, audio=${audioArrivalMs.toFixed(0)}ms (server: llm=${audioMeta?.total_ms ? (audioMeta.total_ms - (audioMeta?.synth_ms||0)).toFixed(0) : "-"}+synth=${audioMeta?.synth_ms?.toFixed(0) || "-"}ms), wav=${blob.size}B`);
  }

  async function onSynthesize() {
    const text = ui.text.value.trim();
    if (!text) throw new Error("text is required");
    const referenceKey = ui.ttsReferenceKey.value.trim() || ui.refSelect.value || "";
    if (!referenceKey) throw new Error("reference_key is required. Add/select a reference first.");

    const payload = {
      text,
      reference_key: referenceKey,
      temp: Number(ui.pTemp.value),
      top_p: Number(ui.pTopP.value),
      top_k: Number(ui.pTopK.value),
      repeat_penalty: Number(ui.pRepeat.value),
      n_predict: Number(ui.pNPredict.value),
    };

    if (ui.streamTokens.checked) {
      return onSynthesizeSSE(payload, referenceKey);
    }

    ui.ttsTokens.classList.add("hidden");
    const t0 = performance.now();
    const res = await fetch("/mio/tts/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const msg = await parseApiError(res);
      throw new Error(msg || `HTTP ${res.status}`);
    }

    let blob;
    let streamedBytes = 0;
    if (ui.streamRead.checked && res.body) {
      const reader = res.body.getReader();
      const chunks = [];
      let received = 0;
      let lastTick = 0;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (value && value.byteLength > 0) {
          chunks.push(value);
          received += value.byteLength;
          const now = performance.now();
          if (now - lastTick > 120) {
            setStatus(ui.ttsStatus, `streaming... ${Math.round(received / 1024)} KiB`);
            lastTick = now;
          }
        }
      }
      streamedBytes = received;
      blob = new Blob(chunks, { type: "audio/wav" });
    } else {
      blob = await res.blob();
      streamedBytes = blob.size;
    }
    const audioMs = performance.now() - t0;
    if (currentAudioUrl) {
      URL.revokeObjectURL(currentAudioUrl);
    }
    currentAudioUrl = URL.createObjectURL(blob);
    ui.audioOut.src = currentAudioUrl;
    ui.audioDownload.href = currentAudioUrl;
    ui.audioDownload.download = `tts-${Date.now()}.wav`;
    ui.audioDownload.classList.remove("hidden");

    const sr = res.headers.get("X-Sample-Rate") || "-";
    const n = res.headers.get("X-Audio-Samples") || "-";
    const slot = res.headers.get("X-Slot") || "-";
    const rk = res.headers.get("X-Reference-Key") || referenceKey || "-";
    ui.ttsMeta.textContent = `ref=${rk}, slot=${slot}, sr=${sr}, samples=${n}`;

    await ui.audioOut.play().catch(() => {});
    setStatus(ui.ttsStatus, `audio=${audioMs.toFixed(0)}ms, wav=${blob.size}B (streamed=${streamedBytes}B)`);
  }

  async function guarded(button, action, statusNode) {
    button.disabled = true;
    try {
      await action();
    } catch (e) {
      setStatus(statusNode, String(e?.message || e), true);
    } finally {
      button.disabled = false;
    }
  }

  ui.btnRefresh.addEventListener("click", () => {
    guarded(ui.btnRefresh, () => refreshReferences(ui.refSelect.value), ui.genRefStatus);
  });
  ui.btnGenRef.addEventListener("click", () => {
    guarded(ui.btnGenRef, onGenerateReference, ui.genRefStatus);
  });
  ui.btnAddRef.addEventListener("click", () => {
    guarded(ui.btnAddRef, onAddReference, ui.addRefStatus);
  });
  ui.btnDelRef.addEventListener("click", () => {
    guarded(ui.btnDelRef, onDeleteReference, ui.delRefStatus);
  });
  ui.refSelect.addEventListener("change", () => {
    ui.ttsReferenceKey.value = ui.refSelect.value || "";
  });
  ui.btnTts.addEventListener("click", () => {
    guarded(ui.btnTts, onSynthesize, ui.ttsStatus);
  });

  refreshReferences().catch((e) => {
    setStatus(ui.genRefStatus, String(e?.message || e), true);
  });
})();
)MIOJS";

struct server_config {
    struct reference_file_config {
        std::string key;
        std::string path;
    };

    std::string host = "127.0.0.1";
    int32_t port = 18089;

    std::string model;
    std::string model_vocoder;
    std::string wavlm_model;
    std::string embedding_default_in;
    std::string output_dir = "/tmp";
    std::string reference_added_output_dir;

    int32_t n_gpu_layers = -1;
    int32_t n_threads = 2;
    int32_t n_ctx = 700;
    int32_t n_predict = 700;
    bool llm_shared_context = true;
    int32_t n_parallel = 1;
    int32_t n_parallel_reference_generation = 0; // 0 means "use n_parallel"
    llama_flash_attn_type flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    std::vector<std::string> mio_backend_devices;
    std::string llm_api_url;
    std::string llm_api_key;
    std::string llm_api_model;
    std::vector<std::pair<std::string, std::string>> llm_api_headers;
    int32_t llm_api_timeout_sec = 120;
    llm_api_mode llm_api_mode_value = LLM_API_MODE_OPENAI_CHAT;

    int32_t top_k = 50;
    float top_p = 1.0f;
    float temp = 0.8f;
    float repeat_penalty = 1.0f;
    uint32_t seed = 0;

    float max_reference_seconds = 20.0f;

    std::vector<reference_file_config> reference_files;
};

struct request_params {
    std::string text;
    std::string output_file;
    std::string codes_in;
    std::string codes_out;
    std::string embedding_in;
    std::string embedding_default_in;
    std::string embedding_out;
    std::string reference_key;
    std::string reference_audio;

    int32_t n_threads = 2;
    int32_t n_ctx = 700;
    int32_t n_predict = 700;
    llama_flash_attn_type flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    int32_t top_k = 50;
    float top_p = 1.0f;
    float temp = 0.8f;
    float repeat_penalty = 1.0f;
    uint32_t seed = 0;

    float max_reference_seconds = 20.0f;

    bool codes_only = false;
    bool embedding_only = false;
    bool stream_tokens = false;

    std::vector<int32_t> inline_codes;
};

struct server_state {
    server_config cfg;
    mio_tts_context * mio = nullptr;
    llama_model * llm = nullptr;
    llama_context * llm_ctx = nullptr;
    const llama_vocab * vocab = nullptr;
    mio_tts_vocab_map * vmap = nullptr;
    bool own_mio = true;
    bool own_llm = true;
    bool own_llm_ctx = true;
    bool own_vmap = true;
    bool llm_warmed = false;
    std::string mio_backend_device;
    size_t mio_runtime_owner_slot = 0;

    ~server_state() {
        if (own_vmap && vmap != nullptr) {
            mio_tts_vocab_map_free(vmap);
            vmap = nullptr;
        }
        if (own_llm_ctx && llm_ctx != nullptr) {
            llama_free(llm_ctx);
            llm_ctx = nullptr;
        }
        if (own_llm && llm != nullptr) {
            llama_model_free(llm);
            llm = nullptr;
        }
        if (own_mio && mio != nullptr) {
            mio_tts_free(mio);
            mio = nullptr;
        }
    }
};

struct reference_cache {
    std::mutex mtx;
    std::unordered_map<std::string, std::vector<float>> embedding_by_key;
};

static void print_usage(const char * argv0) {
    std::fprintf(stderr,
        "Usage:\n"
        "  %s -mv MIOCODEC [options]\n\n"
        "Required:\n"
        "  -mv, --model-vocoder FNAME      MioCodec GGUF\n\n"
        "Optional model:\n"
        "  -m, --model FNAME               MioTTS LLM GGUF (required when request uses text)\n"
        "  --llm-api-url URL               external LLM API endpoint (used instead of local --model)\n"
        "  --llm-api-key STR               bearer token for --llm-api-url\n"
        "  --llm-api-model STR             model name for external API requests\n"
        "  --llm-api-headers JSON          extra request headers as JSON object\n"
        "  --llm-api-timeout N             external API timeout seconds (default: 120)\n"
        "  --llm-api-mode STR              openai-chat | generic (default: openai-chat)\n"
        "  --tts-wavlm-model FNAME         WavLM GGUF (required for reference generation)\n"
        "  -emb, --tts-mio-default-embedding-in FNAME  default embedding GGUF for dynamic codec\n"
        "  --reference-file-json JSON      preload reference embedding GGUF\n"
        "                                  accepts object or array\n"
        "                                  e.g. '{\"key\":\"jp_female\",\"path\":\"/path/jp_female.emb.gguf\"}'\n"
        "                                  e.g. '[{\"key\":\"a\",\"path\":\"/path/a.emb.gguf\"},{\"key\":\"b\",\"path\":\"/path/b.emb.gguf\"}]'\n\n"
        "Server:\n"
        "  --host STR                      bind host (default: 127.0.0.1)\n"
        "  --port N                        bind port (default: 18089)\n"
        "  --output-dir DIR                default output directory (default: /tmp)\n\n"
        "  --reference-added-output-dir DIR  save added/generated reference GGUF files\n\n"
        "Parallelism & GPU:\n"
        "  --parallel N, -np N             synthesis worker slots (default: 1)\n"
        "  --llm-shared-context on|off     share one llama context across slots (default: on)\n"
        "  --parallel-reference-generation N  reference generation slots (default: --parallel)\n"
        "  --mio-backend-devices LIST      comma-separated GPU names (e.g. CUDA0,CUDA1)\n"
        "  -ngl N, --n-gpu-layers N        GPU layers for LLM (default: -1, all)\n"
        "  -fa, --flash-attn [on|off|auto] flash attention mode (default: auto)\n\n"
        "Generation:\n"
        "  --threads N                     threads per worker (default: 2, auto if 0)\n"
        "  --ctx-size N                    max context size (default: 700)\n"
        "  -n N, --n-predict N             max generated tokens (default: 700)\n"
        "  --temp F                        temperature (default: 0.8)\n"
        "  --top-p F                       top-p (default: 1.0)\n"
        "  --top-k N                       top-k (default: 50)\n"
        "  --repeat-penalty F              repeat penalty (default: 1.0)\n"
        "  --seed N                        seed (default: 0)\n"
        "  --tts-max-reference-seconds F   max reference audio seconds (default: 20.0)\n",
        argv0);
}

static int64_t now_ms() {
    const auto now = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
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

static bool save_binary_file(const std::string & path, const std::string & data, std::string & err) {
    std::filesystem::path p(path);
    if (p.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(p.parent_path(), ec);
    }

    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        err = "failed to open file for write: " + path;
        return false;
    }
    file.write(data.data(), (std::streamsize) data.size());
    if (!file.good()) {
        err = "failed to write file: " + path;
        return false;
    }
    return true;
}

static bool load_binary_file(const std::string & path, std::vector<char> & out, std::string & err) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        err = "failed to open file for read: " + path;
        return false;
    }
    file.seekg(0, std::ios::end);
    const auto end_pos = file.tellg();
    if (end_pos < 0) {
        err = "failed to seek file: " + path;
        return false;
    }
    out.resize((size_t) end_pos);
    file.seekg(0, std::ios::beg);
    if (!out.empty()) {
        file.read(out.data(), (std::streamsize) out.size());
        if (!file.good()) {
            err = "failed to read file: " + path;
            return false;
        }
    }
    return true;
}

static std::string make_upload_audio_path(const std::string & output_dir, const std::string & suffix) {
    std::filesystem::path dir(output_dir.empty() ? "/tmp" : output_dir);
    return (dir / ("mio-ref-upload-" + std::to_string(now_ms()) + suffix)).string();
}

static std::string make_embedding_temp_path(const std::string & output_dir, const std::string & reference_key) {
    std::filesystem::path dir(output_dir.empty() ? "/tmp" : output_dir);
    return (dir / ("mio-ref-" + reference_key + "-" + std::to_string(now_ms()) + ".gguf")).string();
}

static std::string make_reference_added_output_path(const std::string & output_dir, const std::string & reference_key) {
    std::filesystem::path dir(output_dir.empty() ? "/tmp" : output_dir);
    return (dir / (reference_key + ".emb.gguf")).string();
}

static bool save_added_reference_embedding(
        const server_config & cfg,
        const std::string & reference_key,
        const float * embedding,
        size_t n_embedding,
        std::string & saved_path,
        std::string & err) {
    saved_path.clear();

    if (cfg.reference_added_output_dir.empty()) {
        return true;
    }
    if (embedding == nullptr || n_embedding == 0) {
        err = "embedding is empty";
        return false;
    }

    const std::string out_path = make_reference_added_output_path(cfg.reference_added_output_dir, reference_key);
    {
        std::filesystem::path p(out_path);
        if (p.has_parent_path()) {
            std::error_code ec;
            std::filesystem::create_directories(p.parent_path(), ec);
            if (ec) {
                err = "failed to create --reference-added-output-dir: " + p.parent_path().string();
                return false;
            }
        }
    }

    char c_err[1024] = {0};
    if (!mio_tts_embedding_save_gguf(out_path.c_str(), embedding, n_embedding, c_err, sizeof(c_err))) {
        err = std::string("mio_tts_embedding_save_gguf failed for --reference-added-output-dir: ") + c_err;
        return false;
    }

    saved_path = out_path;
    return true;
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

static bool parse_csv_list(const std::string & raw, std::vector<std::string> & out) {
    out.clear();
    size_t start = 0;
    while (start <= raw.size()) {
        const size_t comma = raw.find(',', start);
        const size_t end = comma == std::string::npos ? raw.size() : comma;
        const std::string token = trim_copy(raw.substr(start, end - start));
        if (!token.empty()) {
            out.push_back(token);
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return !out.empty();
}

static std::vector<std::string> detect_gpu_backend_names() {
    std::vector<std::string> out;
    const size_t n_dev = ggml_backend_dev_count();
    out.reserve(n_dev);
    int32_t skipped_metal = 0;
    for (size_t i = 0; i < n_dev; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (dev == nullptr) {
            continue;
        }
        const auto type = ggml_backend_dev_type(dev);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }
        const char * name = ggml_backend_dev_name(dev);
        if (name != nullptr && name[0] != '\0') {
            // ggml-metal does not currently support the UPSCALE op used by MioCodec
            // wave upsampler paths. Avoid selecting MTL* automatically to prevent
            // hard aborts in synthesis workers on Apple platforms.
            if (std::strncmp(name, "MTL", 3) == 0) {
                ++skipped_metal;
                continue;
            }
            out.emplace_back(name);
        }
    }
    if (skipped_metal > 0) {
        std::fprintf(stderr,
                "info: skipped %d Metal backend(s) for mio runtime auto-selection "
                "(unsupported UPSCALE op); falling back to CPU unless "
                "--mio-backend-devices is specified\n",
                skipped_metal);
    }
    return out;
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

static bool parse_on_off_bool(const char * s, bool & out) {
    if (s == nullptr) {
        return false;
    }
    std::string v(s);
    std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    if (v == "on" || v == "true" || v == "1" || v == "yes") {
        out = true;
        return true;
    }
    if (v == "off" || v == "false" || v == "0" || v == "no") {
        out = false;
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

static const char * llm_api_mode_to_cstr(llm_api_mode v) {
    switch (v) {
        case LLM_API_MODE_OPENAI_CHAT: return "openai-chat";
        case LLM_API_MODE_GENERIC:     return "generic";
        default:                       return "unknown";
    }
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

static bool parse_llm_api_headers_json(
        const std::string & raw,
        std::vector<std::pair<std::string, std::string>> & out,
        std::string & err) {
    auto trim_local = [](const std::string & in) -> std::string {
        size_t b = 0;
        while (b < in.size() && std::isspace((unsigned char) in[b])) {
            ++b;
        }
        size_t e = in.size();
        while (e > b && std::isspace((unsigned char) in[e - 1])) {
            --e;
        }
        return in.substr(b, e - b);
    };

    try {
        const json j = json::parse(raw);
        if (!j.is_object()) {
            err = "--llm-api-headers must be a JSON object";
            return false;
        }
        out.clear();
        out.reserve(j.size());
        for (auto it = j.begin(); it != j.end(); ++it) {
            const std::string key = trim_local(it.key());
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

static const char * flash_attn_type_to_cstr(llama_flash_attn_type v) {
    switch (v) {
        case LLAMA_FLASH_ATTN_TYPE_ENABLED:  return "on";
        case LLAMA_FLASH_ATTN_TYPE_DISABLED: return "off";
        case LLAMA_FLASH_ATTN_TYPE_AUTO:     return "auto";
        default:                             return "unknown";
    }
}

static bool parse_reference_file_json_entry(
        const json & j,
        const std::string & arg_name,
        server_config::reference_file_config & out,
        std::string & err) {
    if (!j.is_object()) {
        err = arg_name + " entry must be a JSON object";
        return false;
    }

    const auto it_key = j.find("key");
    const auto it_path = j.find("path");
    if (it_key == j.end() || !it_key->is_string()) {
        err = arg_name + " requires string field 'key'";
        return false;
    }
    if (it_path == j.end() || !it_path->is_string()) {
        err = arg_name + " requires string field 'path'";
        return false;
    }

    out.key = it_key->get<std::string>();
    out.path = it_path->get<std::string>();
    if (!is_valid_reference_key(out.key)) {
        err = arg_name + " key is invalid: " + out.key;
        return false;
    }
    if (out.path.empty()) {
        err = arg_name + " path is empty";
        return false;
    }
    return true;
}

static bool parse_reference_file_json_arg(
        const std::string & raw,
        const std::string & option_name,
        std::vector<server_config::reference_file_config> & out,
        std::string & err) {
    try {
        const json j = json::parse(raw);

        if (j.is_object()) {
            server_config::reference_file_config rcfg;
            if (!parse_reference_file_json_entry(j, option_name, rcfg, err)) {
                return false;
            }
            out.push_back(std::move(rcfg));
            return true;
        }

        if (j.is_array()) {
            for (size_t idx = 0; idx < j.size(); ++idx) {
                server_config::reference_file_config rcfg;
                if (!parse_reference_file_json_entry(j[idx], option_name + "[" + std::to_string(idx) + "]", rcfg, err)) {
                    return false;
                }
                out.push_back(std::move(rcfg));
            }
            return true;
        }

        err = option_name + " must be a JSON object or array";
        return false;
    } catch (const std::exception & e) {
        err = std::string("invalid ") + option_name + " JSON: " + e.what();
        return false;
    }
}

static bool needs_value(int i, int argc) {
    return i + 1 < argc;
}

static bool parse_args(int argc, char ** argv, server_config & cfg) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-m" || arg == "--model") {
            if (!needs_value(i, argc)) return false;
            cfg.model = argv[++i];
        } else if (arg == "--llm-api-url") {
            if (!needs_value(i, argc)) return false;
            cfg.llm_api_url = argv[++i];
        } else if (arg == "--llm-api-key") {
            if (!needs_value(i, argc)) return false;
            cfg.llm_api_key = argv[++i];
        } else if (arg == "--llm-api-model") {
            if (!needs_value(i, argc)) return false;
            cfg.llm_api_model = argv[++i];
        } else if (arg == "--llm-api-headers") {
            if (!needs_value(i, argc)) return false;
            std::string perr;
            if (!parse_llm_api_headers_json(argv[++i], cfg.llm_api_headers, perr)) {
                std::fprintf(stderr, "%s\n", perr.c_str());
                return false;
            }
        } else if (arg == "--llm-api-timeout") {
            if (!needs_value(i, argc) || !parse_i32(argv[++i], cfg.llm_api_timeout_sec)) return false;
        } else if (arg == "--llm-api-mode") {
            if (!needs_value(i, argc) || !parse_llm_api_mode(argv[++i], cfg.llm_api_mode_value)) return false;
        } else if (arg == "-mv" || arg == "--model-vocoder") {
            if (!needs_value(i, argc)) return false;
            cfg.model_vocoder = argv[++i];
        } else if (arg == "--tts-wavlm-model") {
            if (!needs_value(i, argc)) return false;
            cfg.wavlm_model = argv[++i];
        } else if (arg == "--tts-mio-default-embedding-in" || arg == "-emb") {
            if (!needs_value(i, argc)) return false;
            cfg.embedding_default_in = argv[++i];
        } else if (arg == "--reference-file-json" || arg == "--reference-file") {
            if (!needs_value(i, argc)) return false;
            const std::string option_name = arg == "--reference-file-json" ? "--reference-file-json" : "--reference-file";
            std::vector<server_config::reference_file_config> refs;
            std::string perr;
            if (!parse_reference_file_json_arg(argv[++i], option_name, refs, perr)) {
                std::fprintf(stderr, "%s\n", perr.c_str());
                return false;
            }
            cfg.reference_files.insert(cfg.reference_files.end(),
                                       std::make_move_iterator(refs.begin()),
                                       std::make_move_iterator(refs.end()));
        } else if (arg == "--host") {
            if (!needs_value(i, argc)) return false;
            cfg.host = argv[++i];
        } else if (arg == "--port") {
            if (!needs_value(i, argc) || !parse_i32(argv[++i], cfg.port)) return false;
        } else if (arg == "--output-dir") {
            if (!needs_value(i, argc)) return false;
            cfg.output_dir = argv[++i];
        } else if (arg == "--reference-added-output-dir") {
            if (!needs_value(i, argc)) return false;
            cfg.reference_added_output_dir = argv[++i];
        } else if (arg == "--threads") {
            if (!needs_value(i, argc) || !parse_i32(argv[++i], cfg.n_threads)) return false;
        } else if (arg == "--ctx-size") {
            if (!needs_value(i, argc) || !parse_i32(argv[++i], cfg.n_ctx)) return false;
        } else if (arg == "-ngl" || arg == "--n-gpu-layers") {
            if (!needs_value(i, argc) || !parse_i32(argv[++i], cfg.n_gpu_layers)) return false;
        } else if (arg == "-fa" || arg == "--flash-attn") {
            if (!needs_value(i, argc) || !parse_flash_attn_type(argv[++i], cfg.flash_attn_type)) return false;
        } else if (arg == "--parallel" || arg == "-np") {
            if (!needs_value(i, argc) || !parse_i32(argv[++i], cfg.n_parallel)) return false;
        } else if (arg == "--llm-shared-context") {
            if (!needs_value(i, argc) || !parse_on_off_bool(argv[++i], cfg.llm_shared_context)) return false;
        } else if (arg == "--parallel-reference-generation") {
            if (!needs_value(i, argc) || !parse_i32(argv[++i], cfg.n_parallel_reference_generation)) return false;
        } else if (arg == "--mio-backend-devices") {
            if (!needs_value(i, argc) || !parse_csv_list(argv[++i], cfg.mio_backend_devices)) return false;
        } else if (arg == "-n" || arg == "--n-predict") {
            if (!needs_value(i, argc) || !parse_i32(argv[++i], cfg.n_predict)) return false;
        } else if (arg == "--temp") {
            if (!needs_value(i, argc) || !parse_f32(argv[++i], cfg.temp)) return false;
        } else if (arg == "--top-p") {
            if (!needs_value(i, argc) || !parse_f32(argv[++i], cfg.top_p)) return false;
        } else if (arg == "--top-k") {
            if (!needs_value(i, argc) || !parse_i32(argv[++i], cfg.top_k)) return false;
        } else if (arg == "--repeat-penalty") {
            if (!needs_value(i, argc) || !parse_f32(argv[++i], cfg.repeat_penalty)) return false;
        } else if (arg == "--seed") {
            if (!needs_value(i, argc) || !parse_u32(argv[++i], cfg.seed)) return false;
        } else if (arg == "--tts-max-reference-seconds") {
            if (!needs_value(i, argc) || !parse_f32(argv[++i], cfg.max_reference_seconds)) return false;
        } else if (arg == "-h" || arg == "--help") {
            return false;
        } else {
            return false;
        }
    }

    if (cfg.n_parallel < 1) {
        return false;
    }
    if (cfg.n_parallel_reference_generation <= 0) {
        cfg.n_parallel_reference_generation = cfg.n_parallel;
    }
    if (cfg.llm_api_timeout_sec < 1) {
        return false;
    }

    if (cfg.llm_api_url.empty()) {
        const char * v = std::getenv("MIO_TTS_LLM_API_URL");
        if (v != nullptr && v[0] != '\0') {
            cfg.llm_api_url = v;
        }
    }
    if (cfg.llm_api_key.empty()) {
        const char * v = std::getenv("MIO_TTS_LLM_API_KEY");
        if (v != nullptr && v[0] != '\0') {
            cfg.llm_api_key = v;
        }
    }
    if (cfg.llm_api_model.empty()) {
        const char * v = std::getenv("MIO_TTS_LLM_API_MODEL");
        if (v != nullptr && v[0] != '\0') {
            cfg.llm_api_model = v;
        }
    }
    if (cfg.llm_api_headers.empty()) {
        const char * v = std::getenv("MIO_TTS_LLM_API_HEADERS");
        if (v != nullptr && v[0] != '\0') {
            std::string perr;
            if (!parse_llm_api_headers_json(v, cfg.llm_api_headers, perr)) {
                std::fprintf(stderr, "%s\n", perr.c_str());
                return false;
            }
        }
    }

    if (!cfg.llm_api_url.empty() && !cfg.model.empty()) {
        std::fprintf(stderr, "info: --llm-api-url is set; ignoring local --model\n");
        cfg.model.clear();
    }

    return !cfg.model_vocoder.empty();
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

static llama_sampler * make_sampler(const request_params & p) {
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
        const request_params & p,
        llama_context * ctx,
        std::vector<llama_token> & generated,
        std::string & err) {
    if (ctx == nullptr) {
        err = "LLM context is not initialized";
        return false;
    }

    const llama_model * model = llama_get_model(ctx);
    if (model == nullptr) {
        err = "LLM model is not available from context";
        return false;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (vocab == nullptr) {
        err = "model has no vocab";
        return false;
    }

    // Keep prompt formatting identical to llama-tts-mio to avoid quality drift.
    const std::string prompt_chat =
            std::string("<|im_start|>user\n") + p.text + "<|im_end|>\n<|im_start|>assistant\n";

    std::vector<llama_token> prompt_tokens;
    if (!tokenize_text(vocab, prompt_chat, false, true, prompt_tokens, err)) {
        return false;
    }

    if (prompt_tokens.empty()) {
        err = "empty prompt tokenization";
        return false;
    }

    const int32_t required_ctx = std::max<int32_t>(p.n_ctx, (int32_t) prompt_tokens.size() + p.n_predict + 32);
    const int32_t available_ctx = (int32_t) llama_n_ctx(ctx);
    if (required_ctx > available_ctx) {
        err = "request requires n_ctx=" + std::to_string(required_ctx) +
              " but slot context has n_ctx=" + std::to_string(available_ctx) +
              " (restart server with larger --ctx-size)";
        return false;
    }

    const int32_t n_threads = resolve_threads(p.n_threads);
    llama_set_n_threads(ctx, n_threads, n_threads);
    llama_memory_clear(llama_get_memory(ctx), false);

    const int32_t n_batch = std::max<int32_t>(1, (int32_t) llama_n_batch(ctx));

    llama_sampler * sampler = make_sampler(p);

    generated.clear();
    for (int32_t pos = 0; pos < (int32_t) prompt_tokens.size(); pos += n_batch) {
        const int32_t n_chunk = std::min<int32_t>(n_batch, (int32_t) prompt_tokens.size() - pos);
        llama_batch batch = llama_batch_get_one(prompt_tokens.data() + pos, n_chunk);
        if (llama_decode(ctx, batch) != 0) {
            llama_sampler_free(sampler);
            err = "llama_decode failed on prompt";
            return false;
        }
    }

    for (int32_t i = 0; i < p.n_predict; ++i) {
        llama_token tok = llama_sampler_sample(sampler, ctx, -1);
        llama_sampler_accept(sampler, tok);
        generated.push_back(tok);

        if (llama_vocab_is_eog(vocab, tok)) {
            break;
        }

        llama_batch batch = llama_batch_get_one(&tok, 1);
        if (llama_decode(ctx, batch) != 0) {
            llama_sampler_free(sampler);
            err = "llama_decode failed during generation";
            return false;
        }
    }

    llama_sampler_free(sampler);
    return true;
}

// Streaming variant: calls on_token(tok, index) after each generated token.
// Return false from the callback to abort generation (e.g. client disconnect).
using token_callback_t = std::function<bool(llama_token tok, int32_t index, bool is_eog)>;

static bool generate_audio_tokens_streaming(
        const request_params & p,
        llama_context * ctx,
        std::vector<llama_token> & generated,
        const token_callback_t & on_token,
        std::string & err) {
    if (ctx == nullptr) {
        err = "LLM context is not initialized";
        return false;
    }

    const llama_model * model = llama_get_model(ctx);
    if (model == nullptr) {
        err = "LLM model is not available from context";
        return false;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    if (vocab == nullptr) {
        err = "model has no vocab";
        return false;
    }

    const std::string prompt_chat =
            std::string("<|im_start|>user\n") + p.text + "<|im_end|>\n<|im_start|>assistant\n";

    std::vector<llama_token> prompt_tokens;
    if (!tokenize_text(vocab, prompt_chat, false, true, prompt_tokens, err)) {
        return false;
    }

    if (prompt_tokens.empty()) {
        err = "empty prompt tokenization";
        return false;
    }

    const int32_t required_ctx = std::max<int32_t>(p.n_ctx, (int32_t) prompt_tokens.size() + p.n_predict + 32);
    const int32_t available_ctx = (int32_t) llama_n_ctx(ctx);
    if (required_ctx > available_ctx) {
        err = "request requires n_ctx=" + std::to_string(required_ctx) +
              " but slot context has n_ctx=" + std::to_string(available_ctx) +
              " (restart server with larger --ctx-size)";
        return false;
    }

    const int32_t n_threads = resolve_threads(p.n_threads);
    llama_set_n_threads(ctx, n_threads, n_threads);
    llama_memory_clear(llama_get_memory(ctx), false);

    const int32_t n_batch = std::max<int32_t>(1, (int32_t) llama_n_batch(ctx));

    llama_sampler * sampler = make_sampler(p);

    generated.clear();
    for (int32_t pos = 0; pos < (int32_t) prompt_tokens.size(); pos += n_batch) {
        const int32_t n_chunk = std::min<int32_t>(n_batch, (int32_t) prompt_tokens.size() - pos);
        llama_batch batch = llama_batch_get_one(prompt_tokens.data() + pos, n_chunk);
        if (llama_decode(ctx, batch) != 0) {
            llama_sampler_free(sampler);
            err = "llama_decode failed on prompt";
            return false;
        }
    }

    for (int32_t i = 0; i < p.n_predict; ++i) {
        llama_token tok = llama_sampler_sample(sampler, ctx, -1);
        llama_sampler_accept(sampler, tok);
        generated.push_back(tok);

        const bool is_eog = llama_vocab_is_eog(vocab, tok);

        if (on_token && !on_token(tok, i, is_eog)) {
            llama_sampler_free(sampler);
            err = "generation aborted by callback";
            return false;
        }

        if (is_eog) {
            break;
        }

        llama_batch batch = llama_batch_get_one(&tok, 1);
        if (llama_decode(ctx, batch) != 0) {
            llama_sampler_free(sampler);
            err = "llama_decode failed during generation";
            return false;
        }
    }

    llama_sampler_free(sampler);
    return true;
}

static bool get_json_string(const json & j, const char * key, std::string & out) {
    auto it = j.find(key);
    if (it == j.end() || it->is_null()) {
        return false;
    }
    if (!it->is_string()) {
        throw std::runtime_error(std::string("field '") + key + "' must be string");
    }
    out = it->get<std::string>();
    return true;
}

template<typename T>
static bool get_json_number(const json & j, const char * key, T & out) {
    auto it = j.find(key);
    if (it == j.end() || it->is_null()) {
        return false;
    }
    if (!it->is_number()) {
        throw std::runtime_error(std::string("field '") + key + "' must be number");
    }
    out = it->get<T>();
    return true;
}

static bool get_json_bool(const json & j, const char * key, bool & out) {
    auto it = j.find(key);
    if (it == j.end() || it->is_null()) {
        return false;
    }
    if (!it->is_boolean()) {
        throw std::runtime_error(std::string("field '") + key + "' must be bool");
    }
    out = it->get<bool>();
    return true;
}

static bool parse_codes_array(const json & j, std::vector<int32_t> & out, std::string & err) {
    if (!j.is_array()) {
        err = "'codes' must be an array";
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

static bool llm_api_enabled(const server_config & cfg) {
    return !cfg.llm_api_url.empty();
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

static bool extract_codes_from_text(const std::string & text, std::vector<int32_t> & out) {
    static const std::regex token_re(R"(<\|s_(-?\d+)\|>)");
    out.clear();
    for (std::sregex_iterator it(text.begin(), text.end(), token_re), end; it != end; ++it) {
        const long v = std::strtol((*it)[1].str().c_str(), nullptr, 10);
        out.push_back((int32_t) v);
    }
    return !out.empty();
}

static std::string truncate_text(const std::string & s, size_t max_len = 240) {
    if (s.size() <= max_len) {
        return s;
    }
    return s.substr(0, max_len) + "...";
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

    err = "LLM API response did not include codes (expected 'codes' or text with <|s_...|>)";
    return false;
}

static bool generate_audio_codes_external(
        const request_params & req,
        const server_config & cfg,
        std::vector<int32_t> & out_codes,
        std::string & err) {
    if (!llm_api_enabled(cfg)) {
        err = "external LLM API is not configured";
        return false;
    }

    parsed_llm_api_url endpoint;
    if (!parse_llm_api_url(cfg.llm_api_url, endpoint, err)) {
        return false;
    }

    json body;
    if (cfg.llm_api_mode_value == LLM_API_MODE_OPENAI_CHAT) {
        if (!cfg.llm_api_model.empty()) {
            body["model"] = cfg.llm_api_model;
        }
        body["messages"] = json::array({
            json {
                {"role", "user"},
                {"content", req.text},
            }
        });
        body["max_tokens"] = req.n_predict;
        body["temperature"] = req.temp;
        body["top_p"] = req.top_p;
        body["stream"] = false;
    } else {
        body["text"] = req.text;
        body["prompt"] = req.text;
        body["n_predict"] = req.n_predict;
        body["temperature"] = req.temp;
        body["top_p"] = req.top_p;
        body["top_k"] = req.top_k;
        body["repeat_penalty"] = req.repeat_penalty;
        body["seed"] = req.seed;
        if (!cfg.llm_api_model.empty()) {
            body["model"] = cfg.llm_api_model;
        }
    }

    httplib::Headers headers;
    headers_upsert_ci(headers, "Content-Type", "application/json");
    for (const auto & kv : cfg.llm_api_headers) {
        headers_upsert_ci(headers, kv.first, kv.second);
    }
    if (!cfg.llm_api_key.empty() && !headers_find_ci(headers, "Authorization")) {
        headers_upsert_ci(headers, "Authorization", "Bearer " + cfg.llm_api_key);
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
        cli.set_connection_timeout(cfg.llm_api_timeout_sec, 0);
        cli.set_read_timeout(cfg.llm_api_timeout_sec, 0);
        cli.set_write_timeout(cfg.llm_api_timeout_sec, 0);
        res = cli.Post(endpoint.path.c_str(), headers, payload, content_type.c_str());
#else
        err = "https URL requires CPPHTTPLIB_OPENSSL_SUPPORT";
        return false;
#endif
    } else {
        httplib::Client cli(endpoint.host, endpoint.port);
        cli.set_follow_location(true);
        cli.set_connection_timeout(cfg.llm_api_timeout_sec, 0);
        cli.set_read_timeout(cfg.llm_api_timeout_sec, 0);
        cli.set_write_timeout(cfg.llm_api_timeout_sec, 0);
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

static std::string make_output_path(const std::string & output_dir) {
    const auto now = std::chrono::system_clock::now();
    const auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    std::filesystem::path dir(output_dir.empty() ? "/tmp" : output_dir);
    return (dir / ("tts-mio-server-" + std::to_string(ts) + ".wav")).string();
}

static double ms_since(
        const std::chrono::steady_clock::time_point & t0,
        const std::chrono::steady_clock::time_point & t1) {
    return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();
}

static std::string base64_encode(const uint8_t * data, size_t len) {
    static const char table[] =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string result;
    result.reserve(((len + 2) / 3) * 4);
    for (size_t i = 0; i < len; i += 3) {
        const uint32_t a = data[i];
        const uint32_t b = (i + 1 < len) ? data[i + 1] : 0;
        const uint32_t c = (i + 2 < len) ? data[i + 2] : 0;
        const uint32_t triple = (a << 16) | (b << 8) | c;
        result += table[(triple >> 18) & 0x3F];
        result += table[(triple >> 12) & 0x3F];
        result += (i + 1 < len) ? table[(triple >> 6) & 0x3F] : '=';
        result += (i + 2 < len) ? table[triple & 0x3F] : '=';
    }
    return result;
}

static bool sse_write_event(httplib::DataSink & sink, const char * event, const std::string & data) {
    std::string msg;
    msg.reserve(16 + std::strlen(event) + data.size());
    msg += "event: ";
    msg += event;
    msg += "\ndata: ";
    msg += data;
    msg += "\n\n";
    return sink.write(msg.data(), msg.size());
}

static void build_wav_header(uint8_t * p, uint32_t sample_rate, uint32_t pcm_bytes) {
    const uint32_t chunk_size = 36 + pcm_bytes;
    const uint32_t byte_rate = sample_rate * 2;
    const uint16_t block_align = 2;
    const uint16_t bits = 16;
    const uint16_t channels = 1;
    const uint16_t fmt = 1;
    const uint32_t fmt_size = 16;
    std::memcpy(p, "RIFF", 4); p += 4;
    std::memcpy(p, &chunk_size, 4); p += 4;
    std::memcpy(p, "WAVE", 4); p += 4;
    std::memcpy(p, "fmt ", 4); p += 4;
    std::memcpy(p, &fmt_size, 4); p += 4;
    std::memcpy(p, &fmt, 2); p += 2;
    std::memcpy(p, &channels, 2); p += 2;
    std::memcpy(p, &sample_rate, 4); p += 4;
    std::memcpy(p, &byte_rate, 4); p += 4;
    std::memcpy(p, &block_align, 2); p += 2;
    std::memcpy(p, &bits, 2); p += 2;
    std::memcpy(p, "data", 4); p += 4;
    std::memcpy(p, &pcm_bytes, 4);
}

static bool parse_request_json(
        const json & body,
        const server_config & cfg,
        request_params & req,
        std::string & err) {
    req.n_threads = cfg.n_threads;
    req.n_ctx = cfg.n_ctx;
    req.n_predict = cfg.n_predict;
    req.flash_attn_type = cfg.flash_attn_type;
    req.top_k = cfg.top_k;
    req.top_p = cfg.top_p;
    req.temp = cfg.temp;
    req.repeat_penalty = cfg.repeat_penalty;
    req.seed = cfg.seed;
    req.max_reference_seconds = cfg.max_reference_seconds;

    try {
        get_json_string(body, "text", req.text);
        if (req.text.empty()) {
            get_json_string(body, "prompt", req.text);
        }
        if (req.text.empty()) {
            get_json_string(body, "input", req.text);
        }

        get_json_string(body, "output_file", req.output_file);
        get_json_string(body, "codes_in", req.codes_in);
        get_json_string(body, "codes_out", req.codes_out);
        get_json_string(body, "embedding_in", req.embedding_in);
        get_json_string(body, "default_embedding_in", req.embedding_default_in);
        if (req.embedding_default_in.empty()) {
            get_json_string(body, "tts_mio_default_embedding_in", req.embedding_default_in);
        }
        get_json_string(body, "embedding_out", req.embedding_out);
        get_json_string(body, "reference_key", req.reference_key);
        if (req.reference_key.empty()) {
            get_json_string(body, "tts_reference_key", req.reference_key);
        }
        if (req.reference_key.empty()) {
            get_json_string(body, "key", req.reference_key);
        }
        get_json_string(body, "reference_audio", req.reference_audio);
        if (req.reference_audio.empty()) {
            get_json_string(body, "tts_reference_audio", req.reference_audio);
        }

        get_json_number(body, "threads", req.n_threads);
        get_json_number(body, "n_ctx", req.n_ctx);
        get_json_number(body, "n_predict", req.n_predict);
        get_json_number(body, "top_k", req.top_k);
        get_json_number(body, "top_p", req.top_p);
        get_json_number(body, "temp", req.temp);
        get_json_number(body, "repeat_penalty", req.repeat_penalty);
        get_json_number(body, "seed", req.seed);
        get_json_number(body, "max_reference_seconds", req.max_reference_seconds);

        get_json_bool(body, "codes_only", req.codes_only);
        get_json_bool(body, "embedding_only", req.embedding_only);
        get_json_bool(body, "stream_tokens", req.stream_tokens);
    } catch (const std::exception & e) {
        err = e.what();
        return false;
    }

    if (req.stream_tokens && llm_api_enabled(cfg)) {
        err = "stream_tokens is not supported when external LLM API mode is enabled";
        return false;
    }

    auto it_codes = body.find("codes");
    if (it_codes != body.end() && !it_codes->is_null()) {
        if (!parse_codes_array(*it_codes, req.inline_codes, err)) {
            return false;
        }
    }

    const bool wants_synthesis = !req.codes_only && !req.embedding_only;

    if (wants_synthesis && req.reference_key.empty()) {
        err = "synthesis requires reference_key";
        return false;
    }

    if (req.embedding_only &&
        req.reference_key.empty() &&
        req.reference_audio.empty() &&
        req.embedding_in.empty() &&
        req.embedding_default_in.empty() &&
        cfg.embedding_default_in.empty()) {
        err = "embedding_only requires reference_key or reference_audio or embedding_in or default_embedding_in";
        return false;
    }

    if (!req.reference_key.empty() && !is_valid_reference_key(req.reference_key)) {
        err = "reference_key is invalid";
        return false;
    }

    if (req.n_ctx < 1) {
        err = "n_ctx must be >= 1";
        return false;
    }
    if (req.n_ctx > cfg.n_ctx) {
        err = "n_ctx exceeds preallocated slot context (" + std::to_string(req.n_ctx) + " > " +
              std::to_string(cfg.n_ctx) + "), restart server with larger --ctx-size";
        return false;
    }
    if (req.n_predict < 1) {
        err = "n_predict must be >= 1";
        return false;
    }
    req.n_predict = std::min(req.n_predict, cfg.n_predict);
    req.n_predict = std::min(req.n_predict, cfg.n_ctx);

    return true;
}

static bool run_tts_request(
        server_state & st,
        const request_params & req,
        std::mutex * llm_gen_mutex,
        reference_cache & ref_cache,
        json & out,
        std::string & err,
        float ** raw_audio_out = nullptr,
        size_t * raw_n_audio_out = nullptr,
        int32_t * raw_sample_rate_out = nullptr) {
    const bool need_codes = !req.embedding_only || req.codes_only || !req.codes_out.empty();

    int32_t * codes_heap = nullptr;
    size_t n_codes_heap = 0;
    const int32_t * codes_ptr = nullptr;
    size_t n_codes = 0;

    auto cleanup_codes = [&]() {
        if (codes_heap != nullptr) {
            mio_tts_codes_free(codes_heap);
            codes_heap = nullptr;
            n_codes_heap = 0;
        }
    };

    if (need_codes) {
        if (!req.inline_codes.empty()) {
            codes_ptr = req.inline_codes.data();
            n_codes = req.inline_codes.size();
        } else if (!req.codes_in.empty()) {
            char c_err[1024] = {0};
            if (!mio_tts_codes_load(req.codes_in.c_str(), &codes_heap, &n_codes_heap, c_err, sizeof(c_err))) {
                err = std::string("mio_tts_codes_load failed: ") + c_err;
                cleanup_codes();
                return false;
            }
            codes_ptr = codes_heap;
            n_codes = n_codes_heap;
        } else if (!req.text.empty()) {
            const auto t_llm_begin = std::chrono::steady_clock::now();
            std::string gen_err;

            if (llm_api_enabled(st.cfg)) {
                std::vector<int32_t> ext_codes;
                if (!generate_audio_codes_external(req, st.cfg, ext_codes, gen_err)) {
                    err = std::string("token generation failed: ") + gen_err;
                    cleanup_codes();
                    return false;
                }
                if (ext_codes.empty()) {
                    err = "token generation failed: external LLM API returned empty codes";
                    cleanup_codes();
                    return false;
                }

                codes_heap = (int32_t *) std::malloc(ext_codes.size() * sizeof(int32_t));
                if (codes_heap == nullptr) {
                    err = "failed to allocate codes buffer";
                    cleanup_codes();
                    return false;
                }
                std::memcpy(codes_heap, ext_codes.data(), ext_codes.size() * sizeof(int32_t));
                n_codes_heap = ext_codes.size();
                codes_ptr = codes_heap;
                n_codes = n_codes_heap;
            } else {
                if (st.llm_ctx == nullptr || st.vmap == nullptr) {
                    err = "text generation requested but LLM model is not loaded";
                    cleanup_codes();
                    return false;
                }

                std::vector<llama_token> generated;
                if (llm_gen_mutex != nullptr) {
                    std::lock_guard<std::mutex> lock(*llm_gen_mutex);
                    if (!generate_audio_tokens(req, st.llm_ctx, generated, gen_err)) {
                        err = std::string("token generation failed: ") + gen_err;
                        cleanup_codes();
                        return false;
                    }
                } else if (!generate_audio_tokens(req, st.llm_ctx, generated, gen_err)) {
                    err = std::string("token generation failed: ") + gen_err;
                    cleanup_codes();
                    return false;
                }

                char c_err[1024] = {0};
                if (!mio_tts_tokens_to_codes(st.vmap, generated.data(), generated.size(), &codes_heap, &n_codes_heap, c_err, sizeof(c_err))) {
                    err = std::string("mio_tts_tokens_to_codes failed: ") + c_err;
                    cleanup_codes();
                    return false;
                }
                codes_ptr = codes_heap;
                n_codes = n_codes_heap;
            }

            if (codes_ptr == nullptr || n_codes == 0) {
                err = "token generation produced no audio codes";
                cleanup_codes();
                return false;
            }

            const auto t_llm_end = std::chrono::steady_clock::now();
            out["llm_ms"] = ms_since(t_llm_begin, t_llm_end);
        } else {
            err = "either text/prompt, codes, or codes_in is required";
            cleanup_codes();
            return false;
        }
    }

    if (!req.codes_out.empty()) {
        if (codes_ptr == nullptr || n_codes == 0) {
            err = "--codes_out requested but no codes available";
            cleanup_codes();
            return false;
        }
        char c_err[1024] = {0};
        if (!mio_tts_codes_save(req.codes_out.c_str(), codes_ptr, n_codes, c_err, sizeof(c_err))) {
            err = std::string("mio_tts_codes_save failed: ") + c_err;
            cleanup_codes();
            return false;
        }
    }

    float * embedding = nullptr;
    size_t n_embedding = 0;
    std::vector<float> embedding_cached_copy;
    bool embedding_needs_free = false;
    auto cleanup_embedding = [&]() {
        if (embedding_needs_free && embedding != nullptr) {
            mio_tts_embedding_free(embedding);
        }
        embedding = nullptr;
        n_embedding = 0;
        embedding_needs_free = false;
        embedding_cached_copy.clear();
    };

    if (!req.embedding_in.empty()) {
        char c_err[1024] = {0};
        if (!mio_tts_embedding_load_gguf(req.embedding_in.c_str(), &embedding, &n_embedding, c_err, sizeof(c_err))) {
            err = std::string("mio_tts_embedding_load_gguf failed: ") + c_err;
            cleanup_embedding();
            cleanup_codes();
            return false;
        }
        embedding_needs_free = true;
    } else if (!req.reference_key.empty()) {
        std::lock_guard<std::mutex> lock(ref_cache.mtx);
        const auto it = ref_cache.embedding_by_key.find(req.reference_key);
        if (it == ref_cache.embedding_by_key.end() || it->second.empty()) {
            err = "reference_key not found: " + req.reference_key;
            cleanup_embedding();
            cleanup_codes();
            return false;
        }
        embedding_cached_copy = it->second;
        embedding = embedding_cached_copy.data();
        n_embedding = embedding_cached_copy.size();
        embedding_needs_free = false;
    } else if (!req.reference_audio.empty()) {
        err = "reference_audio is not supported in synthesis requests. use /mio/generate_reference then reference_key";
        cleanup_embedding();
        cleanup_codes();
        return false;
    } else {
        const std::string & default_emb = !req.embedding_default_in.empty() ? req.embedding_default_in : st.cfg.embedding_default_in;
        if (!default_emb.empty() && mio_tts_context_is_dynamic_global(st.mio)) {
            char c_err[1024] = {0};
            if (!mio_tts_embedding_load_gguf(default_emb.c_str(), &embedding, &n_embedding, c_err, sizeof(c_err))) {
                err = std::string("mio_tts_embedding_load_gguf (default) failed: ") + c_err;
                cleanup_embedding();
                cleanup_codes();
                return false;
            }
            embedding_needs_free = true;
        }
    }

    if (!req.embedding_out.empty()) {
        if (embedding == nullptr || n_embedding == 0) {
            err = "--embedding_out requested but no embedding available";
            cleanup_embedding();
            cleanup_codes();
            return false;
        }
        char c_err[1024] = {0};
        if (!mio_tts_embedding_save_gguf(req.embedding_out.c_str(), embedding, n_embedding, c_err, sizeof(c_err))) {
            err = std::string("mio_tts_embedding_save_gguf failed: ") + c_err;
            cleanup_embedding();
            cleanup_codes();
            return false;
        }
    }

    out["codes"] = n_codes;
    out["embedding_dim"] = n_embedding;
    out["codes_out"] = req.codes_out;
    out["embedding_out"] = req.embedding_out;
    out["reference_key"] = req.reference_key;
    out["key"] = req.reference_key;

    if (req.codes_only || req.embedding_only) {
        if (req.codes_only && codes_ptr != nullptr && n_codes > 0) {
            std::vector<int32_t> code_values(codes_ptr, codes_ptr + n_codes);
            out["codes_values"] = code_values;
        }
        out["ok"] = true;
        out["mode"] = req.codes_only && req.embedding_only ? "codes+embedding-only"
                                                            : req.codes_only ? "codes-only"
                                                                             : "embedding-only";
        cleanup_embedding();
        cleanup_codes();
        return true;
    }

    if (codes_ptr == nullptr || n_codes == 0) {
        err = "synthesis requires codes";
        cleanup_embedding();
        cleanup_codes();
        return false;
    }

    float * audio = nullptr;
    size_t n_audio = 0;
    int32_t sample_rate = 0;
    auto cleanup_audio = [&]() {
        if (audio != nullptr) {
            mio_tts_audio_free(audio);
            audio = nullptr;
            n_audio = 0;
        }
    };

    mio_tts_params sp = mio_tts_default_params();
    sp.n_threads = req.n_threads;
    sp.max_reference_seconds = req.max_reference_seconds;
    sp.wavlm_flash_attn_type = req.flash_attn_type;
    sp.miocodec_flash_attn_type = req.flash_attn_type;

    const auto t_synth_begin = std::chrono::steady_clock::now();
    char c_err[1024] = {0};
    if (!mio_tts_synthesize(st.mio, codes_ptr, n_codes, embedding, n_embedding, sp, &audio, &n_audio, &sample_rate, c_err, sizeof(c_err))) {
        err = std::string("mio_tts_synthesize failed: ") + c_err;
        cleanup_audio();
        cleanup_embedding();
        cleanup_codes();
        return false;
    }
    const auto t_synth_end = std::chrono::steady_clock::now();
    out["synth_ms"] = ms_since(t_synth_begin, t_synth_end);

    // If caller wants raw audio (streaming mode), transfer ownership and skip file write.
    if (raw_audio_out != nullptr) {
        *raw_audio_out = audio;
        if (raw_n_audio_out != nullptr) *raw_n_audio_out = n_audio;
        if (raw_sample_rate_out != nullptr) *raw_sample_rate_out = sample_rate;
        audio = nullptr; // prevent cleanup_audio from freeing
        n_audio = 0;

        out["ok"] = true;
        out["mode"] = "synthesis";
        out["sample_rate"] = sample_rate;
        out["n_audio"] = *raw_n_audio_out;
        out["duration_sec"] = sample_rate > 0 ? (double) *raw_n_audio_out / (double) sample_rate : 0.0;

        cleanup_audio();
        cleanup_embedding();
        cleanup_codes();
        return true;
    }

    std::string output_file = req.output_file.empty() ? make_output_path(st.cfg.output_dir) : req.output_file;
    std::filesystem::path out_path(output_file);
    if (out_path.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(out_path.parent_path(), ec);
    }

    std::memset(c_err, 0, sizeof(c_err));
    if (!mio_tts_audio_save_wav16(output_file.c_str(), audio, n_audio, sample_rate, c_err, sizeof(c_err))) {
        err = std::string("mio_tts_audio_save_wav16 failed: ") + c_err;
        cleanup_audio();
        cleanup_embedding();
        cleanup_codes();
        return false;
    }

    out["ok"] = true;
    out["mode"] = "synthesis";
    out["output_file"] = output_file;
    out["sample_rate"] = sample_rate;
    out["n_audio"] = n_audio;
    out["duration_sec"] = sample_rate > 0 ? (double) n_audio / (double) sample_rate : 0.0;

    cleanup_audio();
    cleanup_embedding();
    cleanup_codes();
    return true;
}

static json make_error_json(const std::string & msg, int code = 400) {
    return json {
        {"ok", false},
        {"error", {
            {"message", msg},
            {"code", code},
        }},
    };
}

static bool init_server_state(server_state & st, const server_config & cfg, std::string & err) {
    st.cfg = cfg;
    st.own_mio = true;
    st.own_llm = true;
    st.own_vmap = true;

    char c_err[1024] = {0};
    const char * wavlm_path = cfg.wavlm_model.empty() ? nullptr : cfg.wavlm_model.c_str();
    st.mio = mio_tts_init_from_file(
            cfg.model_vocoder.c_str(),
            wavlm_path,
            c_err,
            sizeof(c_err));
    if (st.mio == nullptr) {
        err = std::string("mio_tts_init_from_file failed: ") + c_err;
        return false;
    }

    if (!cfg.model.empty()) {
        llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = cfg.n_gpu_layers;
        mparams.use_mmap = true;
        st.llm = llama_model_load_from_file(cfg.model.c_str(), mparams);
        if (st.llm == nullptr) {
            err = std::string("failed to load model: ") + cfg.model;
            return false;
        }

        st.vocab = llama_model_get_vocab(st.llm);
        std::memset(c_err, 0, sizeof(c_err));
        st.vmap = mio_tts_vocab_map_init(st.vocab, c_err, sizeof(c_err));
        if (st.vmap == nullptr) {
            err = std::string("mio_tts_vocab_map_init failed: ") + c_err;
            return false;
        }
    }

    return true;
}

static bool init_worker_mio_context(
        server_state & st,
        const server_config & cfg,
        const std::string & backend_device,
        std::string & err) {
    st.cfg = cfg;
    st.own_mio = true;

    char c_err[1024] = {0};
    const char * wavlm_path = cfg.wavlm_model.empty() ? nullptr : cfg.wavlm_model.c_str();
    st.mio = mio_tts_init_from_file(
            cfg.model_vocoder.c_str(),
            wavlm_path,
            c_err,
            sizeof(c_err));
    if (st.mio == nullptr) {
        err = std::string("mio_tts_init_from_file failed: ") + c_err;
        return false;
    }

    if (!mio_tts_context_set_backend_device(
                st.mio,
                backend_device.empty() ? nullptr : backend_device.c_str(),
                c_err,
                sizeof(c_err))) {
        err = std::string("mio_tts_context_set_backend_device failed: ") + c_err;
        return false;
    }

    st.mio_backend_device = backend_device;
    return true;
}

static bool init_worker_llm_context(server_state & st, std::string & err) {
    if (st.llm == nullptr) {
        return true;
    }

    if (st.llm_ctx != nullptr) {
        return true;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = (uint32_t) std::max<int32_t>(1, st.cfg.n_ctx);
    // Auto-calculate batch sizes: use n_ctx for prompt prefill throughput,
    // capped to a reasonable upper bound to limit per-slot memory.
    cparams.n_batch = std::min<uint32_t>(cparams.n_ctx, 512u);
    cparams.n_ubatch = std::min<uint32_t>(cparams.n_batch, 512u);
    cparams.n_seq_max = 1;
    cparams.flash_attn_type = st.cfg.flash_attn_type;
    cparams.n_threads = resolve_threads(st.cfg.n_threads);
    cparams.n_threads_batch = cparams.n_threads;
    cparams.no_perf = true;

    st.llm_ctx = llama_init_from_model(st.llm, cparams);
    if (st.llm_ctx == nullptr) {
        err = "failed to create llama context for worker slot";
        return false;
    }

    return true;
}

static bool warmup_worker_llm(server_state & st, std::string & err) {
    if (st.llm_ctx == nullptr || st.vocab == nullptr) {
        return true;
    }
    if (st.llm_warmed) {
        return true;
    }

    llama_token tok = llama_vocab_bos(st.vocab);
    if (tok < 0) {
        tok = llama_vocab_eos(st.vocab);
    }
    if (tok < 0) {
        tok = 0;
    }

    llama_memory_clear(llama_get_memory(st.llm_ctx), false);
    llama_batch batch = llama_batch_get_one(&tok, 1);
    if (llama_decode(st.llm_ctx, batch) != 0) {
        err = "llm warmup decode failed";
        return false;
    }
    llama_memory_clear(llama_get_memory(st.llm_ctx), false);
    st.llm_warmed = true;
    return true;
}

static bool ensure_worker_llm_ready(server_state & st, std::string & err) {
    if (st.llm == nullptr) {
        return true;
    }
    if (!init_worker_llm_context(st, err)) {
        return false;
    }
    if (!warmup_worker_llm(st, err)) {
        return false;
    }
    return true;
}

static bool preload_reference_files(
        const server_config & cfg,
        reference_cache & ref_cache,
        std::string & err) {
    for (const auto & rf : cfg.reference_files) {
        float * embedding = nullptr;
        size_t n_embedding = 0;
        char c_err[1024] = {0};

        if (!mio_tts_embedding_load_gguf(rf.path.c_str(), &embedding, &n_embedding, c_err, sizeof(c_err))) {
            err = "failed to load --reference-file-json key='" + rf.key + "' path='" + rf.path + "': " + c_err;
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(ref_cache.mtx);
            ref_cache.embedding_by_key[rf.key] = std::vector<float>(embedding, embedding + n_embedding);
        }
        mio_tts_embedding_free(embedding);
    }
    return true;
}

// warmup_worker_mio removed — warmup is deferred to first request.

static void llama_log_callback_server(ggml_log_level level, const char * text, void * /* user_data */) {
    // Keep server logs concise and avoid expensive debug-log floods during
    // startup. Detailed profiling can still be done with dedicated debug builds.
    if (level >= GGML_LOG_LEVEL_WARN) {
        std::fputs(text, stderr);
    }
}

int main(int argc, char ** argv) {
    server_config cfg;
    if (!parse_args(argc, argv, cfg)) {
        print_usage(argv[0]);
        return 1;
    }

    // Disable CUDA pinned memory by default (can cause issues with multi-context).
    // CUDA graphs are left enabled (default) to speed up per-token LLM decode.
    setenv("GGML_CUDA_NO_PINNED", "1", 0);

    llama_log_set(llama_log_callback_server, nullptr);

    llama_backend_init();
    ggml_backend_load_all();

    std::vector<std::unique_ptr<server_state>> workers;
    workers.reserve((size_t) cfg.n_parallel);
    std::vector<std::string> backend_pool = cfg.mio_backend_devices;
    if (backend_pool.empty()) {
        backend_pool = detect_gpu_backend_names();
    }
    if (backend_pool.empty()) {
        backend_pool.emplace_back("");
    }

    std::vector<std::string> slot_backend((size_t) cfg.n_parallel);
    for (int32_t i = 0; i < cfg.n_parallel; ++i) {
        slot_backend[(size_t) i] = backend_pool[(size_t) i % backend_pool.size()];
    }

    auto backend_key = [](const std::string & name) -> std::string {
        return name.empty() ? std::string("__auto__") : name;
    };

    std::unordered_map<std::string, size_t> runtime_owner_slot_by_backend;
    runtime_owner_slot_by_backend.reserve(backend_pool.size());

    {
        auto worker0 = std::make_unique<server_state>();
        std::string init_err;
        if (!init_server_state(*worker0, cfg, init_err)) {
            std::fprintf(stderr, "worker[0] init failed: %s\n", init_err.c_str());
            return 1;
        }

        char c_err[1024] = {0};
        const std::string & device = slot_backend[0];
        if (!mio_tts_context_set_backend_device(
                    worker0->mio,
                    device.empty() ? nullptr : device.c_str(),
                    c_err,
                    sizeof(c_err))) {
            std::fprintf(stderr, "worker[0] backend assign failed (%s): %s\n",
                    device.empty() ? "auto" : device.c_str(), c_err);
            return 1;
        }
        worker0->mio_backend_device = device;
        worker0->mio_runtime_owner_slot = 0;
        runtime_owner_slot_by_backend[backend_key(device)] = 0;

        if (!init_worker_llm_context(*worker0, init_err)) {
            std::fprintf(stderr, "worker[0] llm context init failed: %s\n", init_err.c_str());
            return 1;
        }
        if (!warmup_worker_llm(*worker0, init_err)) {
            std::fprintf(stderr, "worker[0] llm warmup failed: %s\n", init_err.c_str());
            return 1;
        }
        workers.push_back(std::move(worker0));
    }

    for (int32_t i = 1; i < cfg.n_parallel; ++i) {
        auto worker = std::make_unique<server_state>();
        worker->cfg = cfg;
        worker->own_mio = true;
        worker->own_llm = false;
        worker->own_vmap = false;

        const std::string & device = slot_backend[(size_t) i];
        const std::string key = backend_key(device);
        const auto it_owner = runtime_owner_slot_by_backend.find(key);

        if (it_owner != runtime_owner_slot_by_backend.end()) {
            const size_t owner_slot = it_owner->second;
            char c_err[1024] = {0};
            worker->mio = mio_tts_context_clone(
                    workers[owner_slot]->mio,
                    c_err,
                    sizeof(c_err));
            if (worker->mio == nullptr) {
                std::fprintf(stderr, "worker[%d] mio clone failed (owner slot %zu): %s\n", i, owner_slot, c_err);
                return 1;
            }
            worker->mio_backend_device = workers[owner_slot]->mio_backend_device;
            worker->mio_runtime_owner_slot = owner_slot;
        } else {
            std::string init_err;
            if (!init_worker_mio_context(*worker, cfg, device, init_err)) {
                std::fprintf(stderr, "worker[%d] mio init failed (%s): %s\n",
                        i, device.empty() ? "auto" : device.c_str(), init_err.c_str());
                return 1;
            }
            worker->mio_runtime_owner_slot = (size_t) i;
            runtime_owner_slot_by_backend[key] = (size_t) i;
        }

        worker->llm = workers[0]->llm;
        worker->vocab = workers[0]->vocab;
        worker->vmap = workers[0]->vmap;

        if (cfg.llm_shared_context) {
            // Share one llama_context across all synthesis workers (serialized by
            // llm_gen_mutex). One context keeps VRAM usage constant regardless of
            // --parallel, which is critical on single-GPU setups.
            worker->own_llm_ctx = false;
            worker->llm_ctx = workers[0]->llm_ctx;
            worker->llm_warmed = workers[0]->llm_warmed;
        } else {
            // Per-slot llama contexts remove global generation serialization.
            // This improves parallel throughput at the cost of additional VRAM.
            worker->own_llm_ctx = true;
            worker->llm_ctx = nullptr;
            worker->llm_warmed = false;
            std::string init_err;
            if (!init_worker_llm_context(*worker, init_err)) {
                std::fprintf(stderr, "worker[%d] llm context init failed: %s\n", i, init_err.c_str());
                return 1;
            }
            if (!warmup_worker_llm(*worker, init_err)) {
                std::fprintf(stderr, "worker[%d] llm warmup failed: %s\n", i, init_err.c_str());
                return 1;
            }
        }

        workers.push_back(std::move(worker));
    }

    std::vector<size_t> runtime_owner_slots;
    runtime_owner_slots.reserve(runtime_owner_slot_by_backend.size());
    for (const auto & kv : runtime_owner_slot_by_backend) {
        runtime_owner_slots.push_back(kv.second);
    }
    std::sort(runtime_owner_slots.begin(), runtime_owner_slots.end());

    // Reserve a modest workspace at startup to trigger VRAM allocation.
    // The workspace grows on demand for larger requests. Full pre-allocation
    // of max n_predict can exceed GPU memory on smaller devices.
    const int32_t reserve_codes = 16;

    {
        uint64_t ws_bytes = 0;
        char c_err[1024] = {0};
        if (mio_tts_estimate_workspace_bytes(workers[runtime_owner_slots[0]]->mio, reserve_codes, &ws_bytes, c_err, sizeof(c_err))) {
            const double per_slot_mib = (double) ws_bytes / (1024.0 * 1024.0);
            const double total_mib = per_slot_mib * (double) runtime_owner_slots.size();
            std::fprintf(stderr,
                    "mio decode workspace estimate: %.1f MiB per runtime (reserve_codes=%d), total %.1f MiB for %zu runtime(s)\n",
                    per_slot_mib, reserve_codes, total_mib, runtime_owner_slots.size());
        } else {
            std::fprintf(stderr, "workspace estimate unavailable: %s\n", c_err);
        }
    }

    {
        for (size_t owner_slot : runtime_owner_slots) {
            char c_err[1024] = {0};
            if (!mio_tts_reserve_workspace(workers[owner_slot]->mio, reserve_codes, c_err, sizeof(c_err))) {
                std::fprintf(stderr, "worker[%zu] reserve workspace failed (%s): %s\n",
                        owner_slot,
                        workers[owner_slot]->mio_backend_device.empty() ? "auto" : workers[owner_slot]->mio_backend_device.c_str(),
                        c_err);
                return 1;
            }
        }
    }

    // Warmup is deferred to the first actual request to avoid OOM at startup.
    std::fprintf(stderr, "mio warmup: deferred to first request\n");

    // Reference workers are lazy-initialized on first generate_reference request
    // to avoid loading MioCodec + WavLM models (~480 MB) at startup, which can
    // trigger OOM on memory-constrained systems.
    std::vector<std::unique_ptr<server_state>> reference_workers;
    std::vector<size_t> reference_runtime_owner_slots;
    std::vector<bool> ref_slot_busy; // resized on lazy init
    const bool reference_gen_enabled = !cfg.wavlm_model.empty() && cfg.n_parallel_reference_generation > 0;
    std::once_flag reference_init_flag;
    std::atomic<bool> reference_init_done {false};
    std::string reference_init_error;

    auto lazy_init_reference_workers = [&]() -> bool {
        std::call_once(reference_init_flag, [&]() {
            const int32_t n_ref_slots = cfg.n_parallel_reference_generation;
            reference_workers.reserve((size_t) n_ref_slots);

            std::vector<std::string> ref_slot_backend((size_t) n_ref_slots);
            for (int32_t i = 0; i < n_ref_slots; ++i) {
                ref_slot_backend[(size_t) i] = backend_pool[(size_t) i % backend_pool.size()];
            }

            std::unordered_map<std::string, size_t> ref_owner_by_backend;
            ref_owner_by_backend.reserve(backend_pool.size());

            for (int32_t i = 0; i < n_ref_slots; ++i) {
                const std::string & device = ref_slot_backend[(size_t) i];
                const std::string key = backend_key(device);

                auto worker = std::make_unique<server_state>();
                worker->cfg = cfg;
                worker->cfg.model.clear();
                worker->own_mio = true;
                worker->own_llm = false;
                worker->own_vmap = false;

                const auto it_owner = ref_owner_by_backend.find(key);
                if (it_owner != ref_owner_by_backend.end()) {
                    const size_t owner_slot = it_owner->second;
                    char c_err[1024] = {0};
                    worker->mio = mio_tts_context_clone(
                            reference_workers[owner_slot]->mio,
                            c_err,
                            sizeof(c_err));
                    if (worker->mio == nullptr) {
                        reference_init_error = std::string("reference-worker clone failed: ") + c_err;
                        return;
                    }
                    worker->mio_backend_device = reference_workers[owner_slot]->mio_backend_device;
                    worker->mio_runtime_owner_slot = owner_slot;
                } else {
                    std::string init_err;
                    if (!init_worker_mio_context(*worker, worker->cfg, device, init_err)) {
                        reference_init_error = "reference-worker init failed (" +
                            (device.empty() ? std::string("auto") : device) + "): " + init_err;
                        return;
                    }
                    worker->mio_runtime_owner_slot = (size_t) i;
                    ref_owner_by_backend[key] = (size_t) i;
                }

                reference_workers.push_back(std::move(worker));
            }

            reference_runtime_owner_slots.reserve(ref_owner_by_backend.size());
            for (const auto & kv : ref_owner_by_backend) {
                reference_runtime_owner_slots.push_back(kv.second);
            }
            std::sort(reference_runtime_owner_slots.begin(), reference_runtime_owner_slots.end());

            // Reserve reference workspace (WavLM + global encoder) for each runtime owner.
            for (const size_t owner_slot : reference_runtime_owner_slots) {
                char c_err[1024] = {0};
                if (!mio_tts_reserve_reference_workspace(
                        reference_workers[owner_slot]->mio,
                        cfg.max_reference_seconds,
                        c_err, sizeof(c_err))) {
                    reference_init_error = std::string("reference workspace reserve failed: ") + c_err;
                    return;
                }
            }

            // Update ref slot tracking
            ref_slot_busy.assign(reference_workers.size(), false);

            std::fprintf(stderr, "reference workers lazy-initialized: %zu slot(s)\n", reference_workers.size());
            reference_init_done.store(true);
        });
        return reference_init_done.load();
    };

    if (reference_gen_enabled) {
        std::fprintf(stderr, "reference generation: %d slot(s) (lazy, will load on first request)\n",
                cfg.n_parallel_reference_generation);
    }

    if (!cfg.model.empty()) {
        const int32_t slot_n_batch = workers[0]->llm_ctx ? (int32_t) llama_n_batch(workers[0]->llm_ctx) : 0;
        std::fprintf(stderr, "loaded 1 LLM model + %d context(s), Mio runtime groups=%zu\n",
                cfg.n_parallel, runtime_owner_slots.size());
        if (cfg.llm_shared_context) {
            std::fprintf(stderr,
                    "preallocated 1 shared llama context (used by %d synthesis slot(s)): n_ctx=%d n_batch=%d flash_attn=%s\n",
                    cfg.n_parallel,
                    cfg.n_ctx,
                    slot_n_batch,
                    flash_attn_type_to_cstr(cfg.flash_attn_type));
        } else {
            std::fprintf(stderr,
                    "preallocated %d llama context(s), one per synthesis slot: n_ctx=%d n_batch=%d flash_attn=%s\n",
                    cfg.n_parallel,
                    cfg.n_ctx,
                    slot_n_batch,
                    flash_attn_type_to_cstr(cfg.flash_attn_type));
        }
    } else if (llm_api_enabled(cfg)) {
        std::fprintf(stderr, "local LLM disabled; using external LLM API: %s\n", cfg.llm_api_url.c_str());
        std::fprintf(stderr, "external LLM API mode: %s timeout=%ds model=%s\n",
                llm_api_mode_to_cstr(cfg.llm_api_mode_value),
                cfg.llm_api_timeout_sec,
                cfg.llm_api_model.empty() ? "-" : cfg.llm_api_model.c_str());
        std::fprintf(stderr, "loaded %d synthesis slot(s), Mio runtime groups=%zu\n", cfg.n_parallel, runtime_owner_slots.size());
    } else {
        std::fprintf(stderr, "warning: --model is not set, text generation requests are disabled\n");
        std::fprintf(stderr, "loaded %d synthesis slot(s), Mio runtime groups=%zu\n", cfg.n_parallel, runtime_owner_slots.size());
    }

    std::fprintf(stderr,
            "reference generation: %s\n",
            reference_gen_enabled ? "enabled (lazy-init on first request)" : "disabled (no --tts-wavlm-model)");
    if (reference_gen_enabled) {
        std::fprintf(stderr,
                "reference workspace reserve seconds: %.2f\n",
                cfg.max_reference_seconds);
    }
    if (!cfg.reference_added_output_dir.empty()) {
        std::fprintf(stderr, "reference added output dir: %s\n", cfg.reference_added_output_dir.c_str());
    }

    {
        std::unordered_map<std::string, int32_t> by_backend;
        for (const auto & w : workers) {
            const std::string key = w->mio_backend_device.empty() ? "auto" : w->mio_backend_device;
            by_backend[key] += 1;
        }
        std::vector<std::pair<std::string, int32_t>> rows;
        rows.reserve(by_backend.size());
        for (const auto & kv : by_backend) {
            rows.push_back(kv);
        }
        std::sort(rows.begin(), rows.end(), [](const auto & a, const auto & b) {
            return a.first < b.first;
        });
        std::fprintf(stderr, "mio synthesis slot backend assignment:");
        for (const auto & kv : rows) {
            std::fprintf(stderr, " %s=%d", kv.first.c_str(), kv.second);
        }
        std::fprintf(stderr, "\n");
    }
    // Reference slot backend assignment will be logged on lazy init.

    std::mutex slot_mtx;
    std::condition_variable slot_cv;
    std::vector<bool> slot_busy((size_t) cfg.n_parallel, false);
    size_t rr_slot = 0;
    std::atomic<int32_t> inflight {0};
    std::mutex llm_gen_mutex;
    std::mutex * llm_gen_mutex_ptr = cfg.llm_shared_context ? &llm_gen_mutex : nullptr;

    std::mutex ref_slot_mtx;
    std::condition_variable ref_slot_cv;
    // ref_slot_busy declared earlier, resized on lazy init
    size_t ref_rr_slot = 0;
    std::atomic<int32_t> ref_gen_inflight {0};
    reference_cache ref_cache;

    if (!cfg.reference_files.empty()) {
        std::string load_err;
        if (!preload_reference_files(cfg, ref_cache, load_err)) {
            std::fprintf(stderr, "%s\n", load_err.c_str());
            return 1;
        }
        std::fprintf(stderr, "preloaded %zu reference embedding(s)\n", cfg.reference_files.size());

        // Do not warm up every preloaded reference embedding.
        // That creates multiple decode graph-cache variants and can inflate
        // startup memory when many references are configured.
    }

    auto get_reference_count = [&]() -> int32_t {
        std::lock_guard<std::mutex> lock(ref_cache.mtx);
        return (int32_t) ref_cache.embedding_by_key.size();
    };

    auto acquire_slot = [&]() -> size_t {
        size_t slot_idx = 0;
        {
            std::unique_lock<std::mutex> lock(slot_mtx);
            slot_cv.wait(lock, [&]() {
                return std::any_of(slot_busy.begin(), slot_busy.end(), [](bool busy) { return !busy; });
            });
            for (size_t k = 0; k < slot_busy.size(); ++k) {
                const size_t idx = (rr_slot + k) % slot_busy.size();
                if (!slot_busy[idx]) {
                    slot_busy[idx] = true;
                    rr_slot = (idx + 1) % slot_busy.size();
                    slot_idx = idx;
                    break;
                }
            }
        }
        inflight.fetch_add(1);
        return slot_idx;
    };

    auto release_slot = [&](size_t slot_idx) {
        {
            std::lock_guard<std::mutex> lock(slot_mtx);
            slot_busy[slot_idx] = false;
        }
        inflight.fetch_sub(1);
        slot_cv.notify_one();
    };

    auto acquire_ref_slot = [&]() -> size_t {
        size_t slot_idx = 0;
        std::unique_lock<std::mutex> lock(ref_slot_mtx);
        ref_slot_cv.wait(lock, [&]() {
            return std::any_of(ref_slot_busy.begin(), ref_slot_busy.end(), [](bool busy) { return !busy; });
        });
        for (size_t k = 0; k < ref_slot_busy.size(); ++k) {
            const size_t idx = (ref_rr_slot + k) % ref_slot_busy.size();
            if (!ref_slot_busy[idx]) {
                ref_slot_busy[idx] = true;
                ref_rr_slot = (idx + 1) % ref_slot_busy.size();
                slot_idx = idx;
                break;
            }
        }
        ref_gen_inflight.fetch_add(1);
        return slot_idx;
    };

    auto release_ref_slot = [&](size_t slot_idx) {
        {
            std::lock_guard<std::mutex> lock(ref_slot_mtx);
            ref_slot_busy[slot_idx] = false;
        }
        ref_gen_inflight.fetch_sub(1);
        ref_slot_cv.notify_one();
    };

    httplib::Server server;
    server.set_default_headers({{"Server", "mio-tts-server"}});

    server.set_pre_routing_handler([](const httplib::Request & req, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin", req.get_header_value("Origin"));
        if (req.method == "OPTIONS") {
            res.set_header("Access-Control-Allow-Credentials", "true");
            res.set_header("Access-Control-Allow-Methods", "GET, POST");
            res.set_header("Access-Control-Allow-Headers", "*");
            res.set_content("", "text/plain");
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    server.Get("/health", [&](const httplib::Request &, httplib::Response & res) {
        const int32_t ref_gen_now = ref_gen_inflight.load();
        json j = {
            {"status", "ok"},
            {"parallel", cfg.n_parallel},
            {"parallel_reference_generation", cfg.n_parallel_reference_generation},
            {"reference_generation_enabled", reference_gen_enabled},
            {"reference_generation_initialized", reference_init_done.load()},
            {"inflight", inflight.load()},
            {"reference_generation_inflight", ref_gen_now},
            {"reference_cache", get_reference_count()},
            {"external_llm_enabled", llm_api_enabled(cfg)},
            {"external_llm_mode", llm_api_mode_to_cstr(cfg.llm_api_mode_value)},
            {"llm_shared_context", cfg.llm_shared_context},
        };
        res.set_content(j.dump(), "application/json; charset=utf-8");
    });

    server.Get("/mio/health", [&](const httplib::Request &, httplib::Response & res) {
        const int32_t ref_gen_now = ref_gen_inflight.load();
        json j = {
            {"status", "ok"},
            {"parallel", cfg.n_parallel},
            {"parallel_reference_generation", cfg.n_parallel_reference_generation},
            {"reference_generation_enabled", reference_gen_enabled},
            {"reference_generation_initialized", reference_init_done.load()},
            {"inflight", inflight.load()},
            {"reference_generation_inflight", ref_gen_now},
            {"reference_cache", get_reference_count()},
            {"external_llm_enabled", llm_api_enabled(cfg)},
            {"external_llm_mode", llm_api_mode_to_cstr(cfg.llm_api_mode_value)},
            {"llm_shared_context", cfg.llm_shared_context},
        };
        res.set_content(j.dump(), "application/json; charset=utf-8");
    });

    auto reference_list_handler = [&](const httplib::Request &, httplib::Response & res) {
        std::vector<std::pair<std::string, size_t>> refs;
        {
            std::lock_guard<std::mutex> lock(ref_cache.mtx);
            refs.reserve(ref_cache.embedding_by_key.size());
            for (const auto & kv : ref_cache.embedding_by_key) {
                refs.push_back({kv.first, kv.second.size()});
            }
        }
        std::sort(refs.begin(), refs.end(), [](const auto & a, const auto & b) {
            return a.first < b.first;
        });

        json arr = json::array();
        for (const auto & it : refs) {
            arr.push_back({
                {"key", it.first},
                {"embedding_dim", it.second},
            });
        }
        json j = {
            {"ok", true},
            {"count", refs.size()},
            {"references", std::move(arr)},
        };
        res.set_content(j.dump(), "application/json; charset=utf-8");
    };

    server.Get("/mio/references", reference_list_handler);
    server.Get("/v1/audio/references", reference_list_handler);

    server.Get("/", [&](const httplib::Request &, httplib::Response & res) {
        res.set_header("Cache-Control", "no-store, no-cache, must-revalidate");
        res.set_header("Pragma", "no-cache");
        res.set_header("Expires", "0");
        res.set_content(k_web_index_html, "text/html; charset=utf-8");
    });
    server.Get("/mio-ui.css", [&](const httplib::Request &, httplib::Response & res) {
        res.set_header("Cache-Control", "no-store, no-cache, must-revalidate");
        res.set_header("Pragma", "no-cache");
        res.set_header("Expires", "0");
        res.set_content(k_web_style_css, "text/css; charset=utf-8");
    });
    server.Get("/mio-ui.js", [&](const httplib::Request &, httplib::Response & res) {
        res.set_header("Cache-Control", "no-store, no-cache, must-revalidate");
        res.set_header("Pragma", "no-cache");
        res.set_header("Expires", "0");
        res.set_content(k_web_app_js, "application/javascript; charset=utf-8");
    });
    server.Get("/favicon.ico", [&](const httplib::Request &, httplib::Response & res) {
        res.status = 204;
        res.set_content("", "text/plain; charset=utf-8");
    });

    auto generate_reference_handler = [&](const httplib::Request & req, httplib::Response & res) {
        const auto t_req_begin = std::chrono::steady_clock::now();
        if (!reference_gen_enabled) {
            res.status = 400;
            res.set_content(make_error_json("server requires --tts-wavlm-model for reference generation", 400).dump(), "application/json; charset=utf-8");
            return;
        }
        // Lazy-init reference workers on first request.
        if (!lazy_init_reference_workers()) {
            res.status = 500;
            res.set_content(make_error_json("reference worker init failed: " + reference_init_error, 500).dump(), "application/json; charset=utf-8");
            return;
        }
        if (reference_workers.empty()) {
            res.status = 500;
            res.set_content(make_error_json("reference generation runtime is unavailable", 500).dump(), "application/json; charset=utf-8");
            return;
        }

        std::string reference_key;
        std::string reference_audio;
        float max_reference_seconds = cfg.max_reference_seconds;
        std::string upload_audio_path;

        if (req.is_multipart_form_data()) {
            if (req.form.has_field("reference_key")) {
                reference_key = req.form.get_field("reference_key");
            }
            if (req.form.has_field("reference_audio")) {
                reference_audio = req.form.get_field("reference_audio");
            }
            if (req.form.has_field("max_reference_seconds")) {
                const std::string value = req.form.get_field("max_reference_seconds");
                if (!value.empty() && !parse_f32(value.c_str(), max_reference_seconds)) {
                    res.status = 400;
                    res.set_content(make_error_json("invalid max_reference_seconds", 400).dump(), "application/json; charset=utf-8");
                    return;
                }
            }
            if (req.form.has_file("audio")) {
                const auto file = req.form.get_file("audio");
                std::string suffix = ".wav";
                if (!file.filename.empty()) {
                    const std::string ext = std::filesystem::path(file.filename).extension().string();
                    if (!ext.empty() && ext.size() <= 8) {
                        suffix = ext;
                    }
                }
                upload_audio_path = make_upload_audio_path(cfg.output_dir, suffix);
                std::string io_err;
                if (!save_binary_file(upload_audio_path, file.content, io_err)) {
                    res.status = 400;
                    res.set_content(make_error_json(io_err, 400).dump(), "application/json; charset=utf-8");
                    return;
                }
                reference_audio = upload_audio_path;
            }
        } else {
            json body;
            try {
                body = json::parse(req.body.empty() ? "{}" : req.body);
            } catch (const std::exception & e) {
                res.status = 400;
                res.set_content(make_error_json(std::string("invalid JSON: ") + e.what(), 400).dump(), "application/json; charset=utf-8");
                return;
            }

            try {
                get_json_string(body, "reference_key", reference_key);
                get_json_string(body, "reference_audio", reference_audio);
                if (reference_audio.empty()) {
                    get_json_string(body, "tts_reference_audio", reference_audio);
                }
                get_json_number(body, "max_reference_seconds", max_reference_seconds);
            } catch (const std::exception & e) {
                res.status = 400;
                res.set_content(make_error_json(e.what(), 400).dump(), "application/json; charset=utf-8");
                return;
            }
        }

        if (!is_valid_reference_key(reference_key)) {
            if (!upload_audio_path.empty()) {
                std::error_code ec;
                std::filesystem::remove(upload_audio_path, ec);
            }
            res.status = 400;
            res.set_content(make_error_json("reference_key is invalid", 400).dump(), "application/json; charset=utf-8");
            return;
        }

        if (reference_audio.empty()) {
            if (!upload_audio_path.empty()) {
                std::error_code ec;
                std::filesystem::remove(upload_audio_path, ec);
            }
            res.status = 400;
            res.set_content(make_error_json("reference_audio or multipart file 'audio' is required", 400).dump(), "application/json; charset=utf-8");
            return;
        }

        const auto t_wait_begin = std::chrono::steady_clock::now();
        const size_t slot_idx = acquire_ref_slot();
        const auto t_gen_begin = std::chrono::steady_clock::now();
        float * embedding = nullptr;
        size_t n_embedding = 0;
        char c_err[1024] = {0};

        mio_tts_params mp = mio_tts_default_params();
        mp.n_threads = cfg.n_threads;
        mp.max_reference_seconds = max_reference_seconds;
        mp.wavlm_flash_attn_type = cfg.flash_attn_type;
        mp.miocodec_flash_attn_type = cfg.flash_attn_type;
        const bool ok = mio_tts_reference_to_embedding(
                reference_workers[slot_idx]->mio,
                reference_audio.c_str(),
                mp,
                &embedding,
                &n_embedding,
                c_err,
                sizeof(c_err));
        release_ref_slot(slot_idx);

        const auto t_end = std::chrono::steady_clock::now();
        const double wait_ms = ms_since(t_wait_begin, t_gen_begin);
        const double gen_ms = ms_since(t_gen_begin, t_end);
        const double total_ms = ms_since(t_req_begin, t_end);
        if (!ok) {
            std::fprintf(
                    stderr,
                    "generate_reference: slot=%zu ok=false wait_ms=%.2f gen_ms=%.2f total_ms=%.2f key=%s err=%s\n",
                    slot_idx,
                    wait_ms,
                    gen_ms,
                    total_ms,
                    reference_key.empty() ? "-" : reference_key.c_str(),
                    c_err);
        } else {
            std::fprintf(
                    stderr,
                    "generate_reference: slot=%zu ok=true wait_ms=%.2f gen_ms=%.2f total_ms=%.2f key=%s\n",
                    slot_idx,
                    wait_ms,
                    gen_ms,
                    total_ms,
                    reference_key.empty() ? "-" : reference_key.c_str());
        }

        if (!ok) {
            if (!upload_audio_path.empty()) {
                std::error_code ec;
                std::filesystem::remove(upload_audio_path, ec);
            }
            res.status = 400;
            res.set_content(make_error_json(std::string("mio_tts_reference_to_embedding failed: ") + c_err, 400).dump(), "application/json; charset=utf-8");
            return;
        }

        std::string added_saved_path;
        std::string added_save_err;
        if (!save_added_reference_embedding(cfg, reference_key, embedding, n_embedding, added_saved_path, added_save_err)) {
            mio_tts_embedding_free(embedding);
            if (!upload_audio_path.empty()) {
                std::error_code ec;
                std::filesystem::remove(upload_audio_path, ec);
            }
            res.status = 500;
            res.set_content(make_error_json(added_save_err, 500).dump(), "application/json; charset=utf-8");
            return;
        }

        const std::string embedding_path = make_embedding_temp_path(cfg.output_dir, reference_key);
        std::memset(c_err, 0, sizeof(c_err));
        if (!mio_tts_embedding_save_gguf(embedding_path.c_str(), embedding, n_embedding, c_err, sizeof(c_err))) {
            mio_tts_embedding_free(embedding);
            if (!upload_audio_path.empty()) {
                std::error_code ec;
                std::filesystem::remove(upload_audio_path, ec);
            }
            res.status = 400;
            res.set_content(make_error_json(std::string("mio_tts_embedding_save_gguf failed: ") + c_err, 400).dump(), "application/json; charset=utf-8");
            return;
        }

        std::vector<char> payload;
        std::string io_err;
        if (!load_binary_file(embedding_path, payload, io_err)) {
            mio_tts_embedding_free(embedding);
            std::error_code ec1;
            std::filesystem::remove(embedding_path, ec1);
            if (!upload_audio_path.empty()) {
                std::error_code ec2;
                std::filesystem::remove(upload_audio_path, ec2);
            }
            res.status = 400;
            res.set_content(make_error_json(io_err, 400).dump(), "application/json; charset=utf-8");
            return;
        }

        {
            std::lock_guard<std::mutex> lock(ref_cache.mtx);
            ref_cache.embedding_by_key[reference_key] = std::vector<float>(embedding, embedding + n_embedding);
        }

        mio_tts_embedding_free(embedding);
        {
            std::error_code ec;
            std::filesystem::remove(embedding_path, ec);
        }
        if (!upload_audio_path.empty()) {
            std::error_code ec;
            std::filesystem::remove(upload_audio_path, ec);
        }

        res.status = 200;
        res.set_header("Content-Disposition", ("attachment; filename=\"" + reference_key + ".emb.gguf\"").c_str());
        res.set_header("X-Reference-Key", reference_key.c_str());
        res.set_header("X-Embedding-Dim", std::to_string(n_embedding).c_str());
        if (!added_saved_path.empty()) {
            res.set_header("X-Reference-Saved-Path", added_saved_path.c_str());
        }
        res.set_content(payload.data(), payload.size(), "application/octet-stream");
    };

    auto add_reference_handler = [&](const httplib::Request & req, httplib::Response & res) {
        std::string reference_key;
        std::string embedding_path;
        std::string upload_embedding_path;

        if (req.is_multipart_form_data()) {
            if (req.form.has_field("reference_key")) {
                reference_key = req.form.get_field("reference_key");
            }
            if (reference_key.empty() && req.form.has_field("key")) {
                reference_key = req.form.get_field("key");
            }
            if (req.form.has_field("path")) {
                embedding_path = req.form.get_field("path");
            }
            if (embedding_path.empty() && req.form.has_field("file")) {
                embedding_path = req.form.get_field("file");
            }

            httplib::FormData file;
            bool has_upload = false;
            if (req.form.has_file("file")) {
                file = req.form.get_file("file");
                has_upload = true;
            } else if (req.form.has_file("embedding")) {
                file = req.form.get_file("embedding");
                has_upload = true;
            }

            if (has_upload) {
                std::string suffix = ".gguf";
                if (!file.filename.empty()) {
                    const std::string ext = std::filesystem::path(file.filename).extension().string();
                    if (!ext.empty() && ext.size() <= 16) {
                        suffix = ext;
                    }
                }
                upload_embedding_path = make_upload_audio_path(cfg.output_dir, suffix);
                std::string io_err;
                if (!save_binary_file(upload_embedding_path, file.content, io_err)) {
                    res.status = 400;
                    res.set_content(make_error_json(io_err, 400).dump(), "application/json; charset=utf-8");
                    return;
                }
                embedding_path = upload_embedding_path;
            }
        } else {
            json body;
            try {
                body = json::parse(req.body.empty() ? "{}" : req.body);
            } catch (const std::exception & e) {
                res.status = 400;
                res.set_content(make_error_json(std::string("invalid JSON: ") + e.what(), 400).dump(), "application/json; charset=utf-8");
                return;
            }

            try {
                get_json_string(body, "reference_key", reference_key);
                if (reference_key.empty()) {
                    get_json_string(body, "key", reference_key);
                }
                get_json_string(body, "path", embedding_path);
                if (embedding_path.empty()) {
                    get_json_string(body, "file", embedding_path);
                }
                if (embedding_path.empty()) {
                    get_json_string(body, "embedding_in", embedding_path);
                }
            } catch (const std::exception & e) {
                res.status = 400;
                res.set_content(make_error_json(e.what(), 400).dump(), "application/json; charset=utf-8");
                return;
            }
        }

        if (!is_valid_reference_key(reference_key)) {
            if (!upload_embedding_path.empty()) {
                std::error_code ec;
                std::filesystem::remove(upload_embedding_path, ec);
            }
            res.status = 400;
            res.set_content(make_error_json("reference_key (or key) is invalid", 400).dump(), "application/json; charset=utf-8");
            return;
        }

        if (embedding_path.empty()) {
            if (!upload_embedding_path.empty()) {
                std::error_code ec;
                std::filesystem::remove(upload_embedding_path, ec);
            }
            res.status = 400;
            res.set_content(make_error_json("file/path (gguf) is required", 400).dump(), "application/json; charset=utf-8");
            return;
        }

        float * embedding = nullptr;
        size_t n_embedding = 0;
        char c_err[1024] = {0};
        if (!mio_tts_embedding_load_gguf(embedding_path.c_str(), &embedding, &n_embedding, c_err, sizeof(c_err))) {
            if (!upload_embedding_path.empty()) {
                std::error_code ec;
                std::filesystem::remove(upload_embedding_path, ec);
            }
            res.status = 400;
            res.set_content(make_error_json(std::string("mio_tts_embedding_load_gguf failed: ") + c_err, 400).dump(), "application/json; charset=utf-8");
            return;
        }

        bool replaced = false;
        std::vector<float> emb_vec(embedding, embedding + n_embedding);
        std::string added_saved_path;
        std::string added_save_err;
        if (!save_added_reference_embedding(cfg, reference_key, emb_vec.data(), emb_vec.size(), added_saved_path, added_save_err)) {
            mio_tts_embedding_free(embedding);
            if (!upload_embedding_path.empty()) {
                std::error_code ec;
                std::filesystem::remove(upload_embedding_path, ec);
            }
            res.status = 500;
            res.set_content(make_error_json(added_save_err, 500).dump(), "application/json; charset=utf-8");
            return;
        }
        {
            std::lock_guard<std::mutex> lock(ref_cache.mtx);
            replaced = ref_cache.embedding_by_key.find(reference_key) != ref_cache.embedding_by_key.end();
            ref_cache.embedding_by_key[reference_key] = emb_vec;
        }
        mio_tts_embedding_free(embedding);

        // Skip per-reference warmup to avoid accumulating cache variants.

        if (!upload_embedding_path.empty()) {
            std::error_code ec;
            std::filesystem::remove(upload_embedding_path, ec);
        }

        json out = {
            {"ok", true},
            {"mode", "add-reference"},
            {"reference_key", reference_key},
            {"embedding_dim", n_embedding},
            {"replaced", replaced},
            {"reference_cache", get_reference_count()},
            {"saved_path", added_saved_path},
        };
        res.status = 200;
        res.set_content(out.dump(), "application/json; charset=utf-8");
    };

    auto delete_reference_handler = [&](const httplib::Request & req, httplib::Response & res) {
        std::string reference_key;
        if (req.is_multipart_form_data()) {
            if (req.form.has_field("reference_key")) {
                reference_key = req.form.get_field("reference_key");
            }
            if (reference_key.empty() && req.form.has_field("key")) {
                reference_key = req.form.get_field("key");
            }
        } else {
            json body;
            try {
                body = json::parse(req.body.empty() ? "{}" : req.body);
            } catch (const std::exception & e) {
                res.status = 400;
                res.set_content(make_error_json(std::string("invalid JSON: ") + e.what(), 400).dump(), "application/json; charset=utf-8");
                return;
            }
            try {
                get_json_string(body, "reference_key", reference_key);
                if (reference_key.empty()) {
                    get_json_string(body, "key", reference_key);
                }
            } catch (const std::exception & e) {
                res.status = 400;
                res.set_content(make_error_json(e.what(), 400).dump(), "application/json; charset=utf-8");
                return;
            }
        }

        if (!is_valid_reference_key(reference_key)) {
            res.status = 400;
            res.set_content(make_error_json("reference_key (or key) is invalid", 400).dump(), "application/json; charset=utf-8");
            return;
        }

        bool removed = false;
        {
            std::lock_guard<std::mutex> lock(ref_cache.mtx);
            auto it = ref_cache.embedding_by_key.find(reference_key);
            if (it != ref_cache.embedding_by_key.end()) {
                ref_cache.embedding_by_key.erase(it);
                removed = true;
            }
        }
        if (!removed) {
            res.status = 404;
            res.set_content(make_error_json("reference_key not found: " + reference_key, 404).dump(), "application/json; charset=utf-8");
            return;
        }

        bool removed_saved_file = false;
        std::string saved_path;
        std::string file_warning;
        if (!cfg.reference_added_output_dir.empty()) {
            saved_path = make_reference_added_output_path(cfg.reference_added_output_dir, reference_key);
            std::error_code ec;
            removed_saved_file = std::filesystem::remove(saved_path, ec);
            if (ec) {
                file_warning = "failed to remove saved embedding: " + ec.message();
            }
        }

        json out = {
            {"ok", true},
            {"mode", "delete-reference"},
            {"reference_key", reference_key},
            {"removed", true},
            {"removed_saved_file", removed_saved_file},
            {"saved_path", saved_path},
            {"reference_cache", get_reference_count()},
        };
        if (!file_warning.empty()) {
            out["warning"] = file_warning;
        }

        res.status = 200;
        res.set_content(out.dump(), "application/json; charset=utf-8");
    };

    auto tts_handler = [&](const httplib::Request & req, httplib::Response & res) {
        const auto t_req_begin = std::chrono::steady_clock::now();
        json body;
        try {
            body = json::parse(req.body.empty() ? "{}" : req.body);
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(make_error_json(std::string("invalid JSON: ") + e.what(), 400).dump(), "application/json; charset=utf-8");
            return;
        }

        request_params rp;
        std::string parse_err;
        if (!parse_request_json(body, cfg, rp, parse_err)) {
            res.status = 400;
            res.set_content(make_error_json(parse_err, 400).dump(), "application/json; charset=utf-8");
            return;
        }

        const auto t_wait_begin = std::chrono::steady_clock::now();
        const size_t slot_idx = acquire_slot();
        const auto t_gen_begin = std::chrono::steady_clock::now();

        json out;
        std::string run_err;
        if (!run_tts_request(*workers[slot_idx], rp, llm_gen_mutex_ptr, ref_cache, out, run_err)) {
            const auto t_end = std::chrono::steady_clock::now();
            const double wait_ms = ms_since(t_wait_begin, t_gen_begin);
            const double gen_ms = ms_since(t_gen_begin, t_end);
            const double total_ms = ms_since(t_req_begin, t_end);
            std::fprintf(
                    stderr,
                    "generate: path=%s slot=%zu ok=false wait_ms=%.2f gen_ms=%.2f total_ms=%.2f n_predict=%d ref=%s err=%s\n",
                    req.path.c_str(),
                    slot_idx,
                    wait_ms,
                    gen_ms,
                    total_ms,
                    rp.n_predict,
                    rp.reference_key.empty() ? "-" : rp.reference_key.c_str(),
                    run_err.c_str());
            release_slot(slot_idx);
            res.status = 400;
            res.set_content(make_error_json(run_err, 400).dump(), "application/json; charset=utf-8");
            return;
        }

        release_slot(slot_idx);
        const auto t_end = std::chrono::steady_clock::now();
        const double wait_ms = ms_since(t_wait_begin, t_gen_begin);
        const double gen_ms = ms_since(t_gen_begin, t_end);
        const double total_ms = ms_since(t_req_begin, t_end);
        const double llm_ms = out.contains("llm_ms") ? out["llm_ms"].get<double>() : 0.0;
        const double synth_ms = out.contains("synth_ms") ? out["synth_ms"].get<double>() : 0.0;
        const std::string mode = out.contains("mode") && out["mode"].is_string() ? out["mode"].get<std::string>() : "unknown";
        const size_t n_codes = out.contains("codes") ? out["codes"].get<size_t>() : 0;
        std::fprintf(
                stderr,
                "generate: path=%s slot=%zu ok=true wait_ms=%.2f llm_ms=%.2f synth_ms=%.2f total_ms=%.2f n_predict=%d n_codes=%zu ref=%s mode=%s\n",
                req.path.c_str(),
                slot_idx,
                wait_ms,
                llm_ms,
                synth_ms,
                total_ms,
                rp.n_predict,
                n_codes,
                rp.reference_key.empty() ? "-" : rp.reference_key.c_str(),
                mode.c_str());

        res.status = 200;
        out["slot"] = (int32_t) slot_idx;
        res.set_content(out.dump(), "application/json; charset=utf-8");
    };

    auto tts_stream_handler = [&](const httplib::Request & req, httplib::Response & res) {
        const auto t_req_begin = std::chrono::steady_clock::now();
        json body;
        try {
            body = json::parse(req.body.empty() ? "{}" : req.body);
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(make_error_json(std::string("invalid JSON: ") + e.what(), 400).dump(), "application/json; charset=utf-8");
            return;
        }

        request_params rp;
        std::string parse_err;
        if (!parse_request_json(body, cfg, rp, parse_err)) {
            res.status = 400;
            res.set_content(make_error_json(parse_err, 400).dump(), "application/json; charset=utf-8");
            return;
        }

        // --- SSE streaming mode (stream_tokens=true) ---
        if (rp.stream_tokens) {
            if (rp.text.empty()) {
                res.status = 400;
                res.set_content(make_error_json("stream_tokens requires text input", 400).dump(), "application/json; charset=utf-8");
                return;
            }

            // Capture pointers to main-scope objects (safe: main scope outlives response).
            auto * p_workers        = &workers;
            auto * p_llm_gen_mutex  = llm_gen_mutex_ptr;
            auto * p_ref_cache      = &ref_cache;
            auto fn_acquire_slot    = acquire_slot;
            auto fn_release_slot    = release_slot;
            auto req_path           = req.path;

            res.set_header("Cache-Control", "no-cache");
            res.set_header("X-Accel-Buffering", "no");

            res.set_chunked_content_provider(
                "text/event-stream; charset=utf-8",
                [p_workers, p_llm_gen_mutex, p_ref_cache,
                 fn_acquire_slot, fn_release_slot,
                 rp = std::move(rp),
                 req_path = std::move(req_path),
                 t_req_begin,
                 first = true]
                (size_t, httplib::DataSink & sink) mutable -> bool {
                    if (!first) return false;
                    first = false;

                    auto & workers_ref = *p_workers;

                    const auto t_wait_begin = std::chrono::steady_clock::now();
                    const size_t slot_idx = fn_acquire_slot();
                    const auto t_gen_begin = std::chrono::steady_clock::now();

                    auto & st = *workers_ref[slot_idx];

                    // Resolve embedding from reference cache.
                    std::vector<float> embedding_copy;
                    {
                        std::lock_guard<std::mutex> lock(p_ref_cache->mtx);
                        auto it = p_ref_cache->embedding_by_key.find(rp.reference_key);
                        if (it == p_ref_cache->embedding_by_key.end() || it->second.empty()) {
                            sse_write_event(sink, "error",
                                    json({{"error", "reference_key not found: " + rp.reference_key}}).dump());
                            fn_release_slot(slot_idx);
                            sink.done();
                            return true;
                        }
                        embedding_copy = it->second;
                    }

                    // LLM token generation with SSE streaming.
                    std::vector<llama_token> generated;
                    const auto t_llm_begin = std::chrono::steady_clock::now();
                    std::string gen_err;
                    bool ok = false;
                    if (p_llm_gen_mutex != nullptr) {
                        std::lock_guard<std::mutex> lock(*p_llm_gen_mutex);
                        ok = generate_audio_tokens_streaming(rp, st.llm_ctx, generated,
                            [&](llama_token tok, int32_t index, bool is_eog) -> bool {
                                int32_t code = -1;
                                bool has_code = mio_tts_token_to_code(st.vmap, tok, &code);
                                json ev;
                                ev["id"] = tok;
                                ev["i"]  = index;
                                if (has_code) ev["code"] = code;
                                if (is_eog)   ev["eog"]  = true;
                                return sse_write_event(sink, "token", ev.dump());
                            }, gen_err);
                    } else {
                        ok = generate_audio_tokens_streaming(rp, st.llm_ctx, generated,
                            [&](llama_token tok, int32_t index, bool is_eog) -> bool {
                                int32_t code = -1;
                                bool has_code = mio_tts_token_to_code(st.vmap, tok, &code);
                                json ev;
                                ev["id"] = tok;
                                ev["i"]  = index;
                                if (has_code) ev["code"] = code;
                                if (is_eog)   ev["eog"]  = true;
                                return sse_write_event(sink, "token", ev.dump());
                            }, gen_err);
                    }
                    if (!ok) {
                        sse_write_event(sink, "error", json({{"error", gen_err}}).dump());
                        fn_release_slot(slot_idx);
                        sink.done();
                        return true;
                    }
                    const auto t_llm_end = std::chrono::steady_clock::now();
                    const double llm_ms = ms_since(t_llm_begin, t_llm_end);

                    // Convert tokens to codes.
                    int32_t * codes_heap = nullptr;
                    size_t n_codes = 0;
                    char c_err[1024] = {0};
                    if (!mio_tts_tokens_to_codes(st.vmap, generated.data(), generated.size(),
                                                 &codes_heap, &n_codes, c_err, sizeof(c_err))) {
                        sse_write_event(sink, "error", json({{"error", std::string(c_err)}}).dump());
                        fn_release_slot(slot_idx);
                        sink.done();
                        return true;
                    }

                    // Send generation_complete event.
                    sse_write_event(sink, "generation_complete", json({
                        {"n_tokens", (int) generated.size()},
                        {"n_codes",  (int) n_codes},
                        {"llm_ms",   llm_ms}
                    }).dump());

                    // Synthesize codes → audio.
                    const auto t_synth_begin = std::chrono::steady_clock::now();
                    mio_tts_params sp = mio_tts_default_params();
                    sp.n_threads = rp.n_threads;
                    sp.miocodec_flash_attn_type = rp.flash_attn_type;

                    float * audio = nullptr;
                    size_t n_audio = 0;
                    int32_t sample_rate = 0;
                    if (!mio_tts_synthesize(st.mio, codes_heap, n_codes,
                                            embedding_copy.data(), embedding_copy.size(),
                                            sp, &audio, &n_audio, &sample_rate,
                                            c_err, sizeof(c_err))) {
                        mio_tts_codes_free(codes_heap);
                        sse_write_event(sink, "error", json({{"error", std::string(c_err)}}).dump());
                        fn_release_slot(slot_idx);
                        sink.done();
                        return true;
                    }
                    mio_tts_codes_free(codes_heap);
                    const auto t_synth_end = std::chrono::steady_clock::now();
                    const double synth_ms = ms_since(t_synth_begin, t_synth_end);

                    // Build WAV in memory.
                    const size_t pcm_bytes = n_audio * sizeof(int16_t);
                    std::vector<uint8_t> wav_buf(44 + pcm_bytes);
                    build_wav_header(wav_buf.data(), (uint32_t) sample_rate, (uint32_t) pcm_bytes);

                    int16_t * pcm = reinterpret_cast<int16_t *>(wav_buf.data() + 44);
                    for (size_t i = 0; i < n_audio; ++i) {
                        const float x = std::clamp(audio[i], -1.0f, 1.0f);
                        pcm[i] = (int16_t) std::lrintf(x * 32767.0f);
                    }
                    mio_tts_audio_free(audio);

                    // Send audio metadata event.
                    const auto t_end = std::chrono::steady_clock::now();
                    const double wait_ms   = ms_since(t_wait_begin, t_gen_begin);
                    const double total_ms  = ms_since(t_req_begin, t_end);
                    sse_write_event(sink, "audio_meta", json({
                        {"sample_rate", sample_rate},
                        {"n_audio",     (int64_t) n_audio},
                        {"synth_ms",    synth_ms},
                        {"total_ms",    total_ms},
                        {"wav_size",    (int64_t) wav_buf.size()}
                    }).dump());

                    // Send base64-encoded WAV.
                    std::string b64 = base64_encode(wav_buf.data(), wav_buf.size());
                    sse_write_event(sink, "audio_data", b64);

                    // Log.
                    std::fprintf(stderr,
                        "generate: path=%s slot=%zu ok=true wait_ms=%.2f llm_ms=%.2f synth_ms=%.2f total_ms=%.2f "
                        "n_predict=%d n_codes=%zu ref=%s mode=sse_stream\n",
                        req_path.c_str(), slot_idx, wait_ms, llm_ms, synth_ms, total_ms,
                        rp.n_predict, n_codes,
                        rp.reference_key.empty() ? "-" : rp.reference_key.c_str());

                    fn_release_slot(slot_idx);
                    sink.done();
                    return true;
                });
            return;
        }

        // --- Standard mode (stream_tokens=false): binary WAV response ---
        const auto t_wait_begin = std::chrono::steady_clock::now();
        const size_t slot_idx = acquire_slot();
        const auto t_gen_begin = std::chrono::steady_clock::now();

        float * raw_audio = nullptr;
        size_t raw_n_audio = 0;
        int32_t raw_sample_rate = 0;
        json out;
        std::string run_err;
        if (!run_tts_request(*workers[slot_idx], rp, llm_gen_mutex_ptr, ref_cache, out, run_err,
                             &raw_audio, &raw_n_audio, &raw_sample_rate)) {
            const auto t_end = std::chrono::steady_clock::now();
            const double wait_ms = ms_since(t_wait_begin, t_gen_begin);
            const double gen_ms = ms_since(t_gen_begin, t_end);
            const double total_ms = ms_since(t_req_begin, t_end);
            std::fprintf(
                    stderr,
                    "generate: path=%s slot=%zu ok=false wait_ms=%.2f gen_ms=%.2f total_ms=%.2f n_predict=%d ref=%s err=%s\n",
                    req.path.c_str(),
                    slot_idx,
                    wait_ms,
                    gen_ms,
                    total_ms,
                    rp.n_predict,
                    rp.reference_key.empty() ? "-" : rp.reference_key.c_str(),
                    run_err.c_str());
            release_slot(slot_idx);
            res.status = 400;
            res.set_content(make_error_json(run_err, 400).dump(), "application/json; charset=utf-8");
            return;
        }
        release_slot(slot_idx);

        const auto t_end = std::chrono::steady_clock::now();
        const double wait_ms = ms_since(t_wait_begin, t_gen_begin);
        const double gen_ms = ms_since(t_gen_begin, t_end);
        const double total_ms = ms_since(t_req_begin, t_end);
        const double llm_ms = out.contains("llm_ms") ? out["llm_ms"].get<double>() : 0.0;
        const double synth_ms = out.contains("synth_ms") ? out["synth_ms"].get<double>() : 0.0;
        const std::string mode = out.contains("mode") && out["mode"].is_string() ? out["mode"].get<std::string>() : "unknown";
        const size_t n_codes = out.contains("codes") ? out["codes"].get<size_t>() : 0;
        std::fprintf(
                stderr,
                "generate: path=%s slot=%zu ok=true wait_ms=%.2f llm_ms=%.2f synth_ms=%.2f total_ms=%.2f n_predict=%d n_codes=%zu ref=%s mode=%s\n",
                req.path.c_str(),
                slot_idx,
                wait_ms,
                llm_ms,
                synth_ms,
                total_ms,
                rp.n_predict,
                n_codes,
                rp.reference_key.empty() ? "-" : rp.reference_key.c_str(),
                mode.c_str());

        if (raw_audio == nullptr || raw_n_audio == 0) {
            res.status = 500;
            res.set_content(make_error_json("synthesis produced no audio", 500).dump(), "application/json; charset=utf-8");
            return;
        }

        const size_t pcm_bytes = raw_n_audio * sizeof(int16_t);
        auto wav_buf = std::make_shared<std::vector<uint8_t>>(44 + pcm_bytes);
        build_wav_header(wav_buf->data(), (uint32_t) raw_sample_rate, (uint32_t) pcm_bytes);

        int16_t * pcm = reinterpret_cast<int16_t *>(wav_buf->data() + 44);
        for (size_t i = 0; i < raw_n_audio; ++i) {
            const float x = std::clamp(raw_audio[i], -1.0f, 1.0f);
            pcm[i] = (int16_t) std::lrintf(x * 32767.0f);
        }
        mio_tts_audio_free(raw_audio);

        res.status = 200;
        res.set_header("X-Slot", std::to_string(slot_idx));
        if (raw_sample_rate > 0) {
            res.set_header("X-Sample-Rate", std::to_string(raw_sample_rate));
        }
        if (raw_n_audio > 0) {
            res.set_header("X-Audio-Samples", std::to_string(raw_n_audio));
        }
        const std::string ref_key = out.contains("reference_key") && out["reference_key"].is_string()
                ? out["reference_key"].get<std::string>() : "";
        if (!ref_key.empty()) {
            res.set_header("X-Reference-Key", ref_key);
        }

        res.set_chunked_content_provider(
                "audio/wav",
                [wav_buf, offset = size_t(0)](size_t, httplib::DataSink & sink) mutable {
                    constexpr size_t CHUNK = 64 * 1024;
                    if (offset >= wav_buf->size()) {
                        sink.done();
                        return true;
                    }
                    const size_t n = std::min(CHUNK, wav_buf->size() - offset);
                    if (!sink.write(reinterpret_cast<const char *>(wav_buf->data() + offset), n)) {
                        return false;
                    }
                    offset += n;
                    return true;
                });
    };

    if (reference_gen_enabled) {
        server.Post("/mio/generate_reference", generate_reference_handler);
        server.Post("/v1/audio/generate_reference", generate_reference_handler);
    }
    server.Post("/mio/add_reference", add_reference_handler);
    server.Post("/v1/audio/add_reference", add_reference_handler);
    server.Post("/mio/delete_reference", delete_reference_handler);
    server.Post("/mio/remove_reference", delete_reference_handler);
    server.Post("/v1/audio/delete_reference", delete_reference_handler);
    server.Post("/v1/audio/remove_reference", delete_reference_handler);
    server.Post("/mio/tts", tts_handler);
    server.Post("/mio/tts/stream", tts_stream_handler);
    server.Post("/v1/audio/speech", tts_handler);
    server.Post("/v1/audio/speech/stream", tts_stream_handler);

    std::fprintf(stderr, "mio-tts-server listening on http://%s:%d\n", cfg.host.c_str(), cfg.port);
    if (!server.listen(cfg.host, cfg.port)) {
        std::fprintf(stderr, "failed to listen on %s:%d\n", cfg.host.c_str(), cfg.port);
        return 1;
    }

    return 0;
}
