import createModule from "./miottscpp_core.js";

const ui = {
  embedding: document.getElementById("embedding"),
  text: document.getElementById("text"),
  nThreads: document.getElementById("n-threads"),
  ctxSize: document.getElementById("ctx-size"),
  nPredict: document.getElementById("n-predict"),
  topK: document.getElementById("top-k"),
  topP: document.getElementById("top-p"),
  temp: document.getElementById("temp"),
  useExternalApi: document.getElementById("use-external-api"),
  apiSettings: document.getElementById("api-settings"),
  apiBase: document.getElementById("api-base"),
  apiKey: document.getElementById("api-key"),
  apiModel: document.getElementById("api-model"),
  btnGenerate: document.getElementById("btn-generate"),
  btnStop: document.getElementById("btn-stop"),
  status: document.getElementById("status"),
  metrics: document.getElementById("metrics"),
  player: document.getElementById("player"),
  btnLoadWavlm: document.getElementById("btn-load-wavlm"),
  btnDeleteReference: document.getElementById("btn-delete-reference"),
  btnRecordStart: document.getElementById("btn-record-start"),
  btnRecordStop: document.getElementById("btn-record-stop"),
  refFile: document.getElementById("ref-file"),
  btnAddFile: document.getElementById("btn-add-file"),
  refStatus: document.getElementById("ref-status"),
};

const VOCODER_MODEL_URL = new URL("./public/model/miocodec.gguf", import.meta.url).toString();
const LLM_MODEL_URL = new URL("./public/model/MioTTS-0.1B-Q8_0.gguf", import.meta.url).toString();
const WAVLM_MODEL_URL = new URL("./public/model/wavlm_base_plus_2l_f32.gguf", import.meta.url).toString();
const WAVLM_MODEL_FS_PATH = "/models/wavlm_base_plus_2l_f32.gguf";
const EMBEDDINGS_BASE_URL = new URL("./public/embeddings/", import.meta.url);
const EMBEDDINGS_INDEX_URL = new URL("./public/embeddings/index.json", import.meta.url).toString();
const STORAGE_KEY_LOCAL_REFERENCES = "miottscpp.local_references.v1";
const PRESET_EMBEDDING_CANDIDATES = [
  "en_female.emb.gguf",
  "en_male.emb.gguf",
  "ja_female.emb.gguf",
  "ja_male.emb.gguf",
  "jp_female.emb.gguf",
  "jp_male.emb.gguf",
];
const CODE_TOKEN_RE = /<\|s_(-?\d+)\|>/g;
const MAX_LOCAL_EMBEDDING_BASE64_CHARS = 512 * 1024;
const MAX_EMBEDDING_FLOATS = 8192;

let moduleRef = null;
let engine = null;
let currentAbortController = null;
let loadedEmbeddingPath = "";
let localLLMReady = false;
let playerObjectUrl = "";
let isBusy = false;

let presetEmbeddings = [];
let localReferences = [];
const localReferenceCache = new Map();

let mediaRecorder = null;
let mediaRecorderChunks = [];
let mediaRecorderStream = null;
let wavlmLoaded = false;
let wavlmLoadingPromise = null;

function setStatus(message) {
  ui.status.textContent = message;
}

function setRefStatus(message) {
  ui.refStatus.textContent = message;
}

function isRecordingActive() {
  return mediaRecorder != null && mediaRecorder.state === "recording";
}

function updateRecordingButtons() {
  const recording = isRecordingActive();
  ui.btnRecordStart.disabled = isBusy || recording;
  ui.btnRecordStop.disabled = isBusy || !recording;
}

function setBusy(busy) {
  isBusy = busy;
  ui.btnGenerate.disabled = busy;
  ui.embedding.disabled = busy;
  ui.text.disabled = busy;
  ui.nThreads.disabled = busy;
  ui.ctxSize.disabled = busy;
  ui.nPredict.disabled = busy;
  ui.topK.disabled = busy;
  ui.topP.disabled = busy;
  ui.temp.disabled = busy;
  ui.useExternalApi.disabled = busy;
  ui.apiBase.disabled = busy;
  ui.apiKey.disabled = busy;
  ui.apiModel.disabled = busy;
  ui.btnLoadWavlm.disabled = busy;
  ui.btnDeleteReference.disabled = busy;
  ui.btnAddFile.disabled = busy;
  ui.refFile.disabled = busy;
  updateRecordingButtons();
}

async function fetchBytes(url, signal) {
  const res = await fetch(url, {
    signal,
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(`fetch failed: ${url} (${res.status})`);
  }
  return new Uint8Array(await res.arrayBuffer());
}

async function fetchFileToFS(url, fsPath, signal) {
  const res = await fetch(url, {
    signal,
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(`fetch failed: ${url} (${res.status})`);
  }

  const slash = fsPath.lastIndexOf("/");
  if (slash > 0) {
    ensureDir(fsPath.substring(0, slash));
  }

  try {
    moduleRef.FS.unlink(fsPath);
  } catch (_e) {
    // ignore
  }

  if (!res.body || typeof res.body.getReader !== "function") {
    const bytes = new Uint8Array(await res.arrayBuffer());
    moduleRef.FS.writeFile(fsPath, bytes);
    return;
  }

  const stream = moduleRef.FS.open(fsPath, "w+");
  let position = 0;
  try {
    const reader = res.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!value || value.length === 0) continue;
      moduleRef.FS.write(stream, value, 0, value.length, position);
      position += value.length;
    }
  } finally {
    moduleRef.FS.close(stream);
  }
}

function ensureDir(path) {
  try {
    moduleRef.FS.mkdir(path);
  } catch (_e) {
    // already exists
  }
}

function writeFileToFS(path, bytes) {
  const slash = path.lastIndexOf("/");
  if (slash > 0) {
    ensureDir(path.substring(0, slash));
  }
  try {
    moduleRef.FS.unlink(path);
  } catch (_e) {
    // ignore
  }
  moduleRef.FS.writeFile(path, bytes);
}

function unlinkFileIfExists(path) {
  try {
    moduleRef.FS.unlink(path);
  } catch (_e) {
    // ignore
  }
}

function parseThreads() {
  const n = Number(ui.nThreads.value);
  if (!Number.isFinite(n)) return 2;
  return Math.max(1, Math.floor(n));
}

function parseIntInput(el, fallback, min = null, max = null) {
  const v = Number(el.value);
  if (!Number.isFinite(v)) return fallback;
  let i = Math.floor(v);
  if (min != null) {
    i = Math.max(min, i);
  }
  if (max != null) {
    i = Math.min(max, i);
  }
  return i;
}

function parseFloatInput(el, fallback, min = null, max = null) {
  const v = Number(el.value);
  if (!Number.isFinite(v)) return fallback;
  let out = v;
  if (min != null) out = Math.max(min, out);
  if (max != null) out = Math.min(max, out);
  return out;
}

function stopPlayback() {
  ui.player.pause();
  ui.player.currentTime = 0;
}

async function playFloatAudio(float32Array, sampleRate) {
  stopPlayback();
  const wav = floatToWavBlob(float32Array, sampleRate);
  if (playerObjectUrl) {
    URL.revokeObjectURL(playerObjectUrl);
    playerObjectUrl = "";
  }
  playerObjectUrl = URL.createObjectURL(wav);
  ui.player.src = playerObjectUrl;
  ui.player.currentTime = 0;
  void ui.player.play().catch(() => {});
}

function preparePlaybackAudio(input) {
  const audio = new Float32Array(input.length);
  let peak = 0;
  let sumSq = 0;
  let nonFinite = 0;

  for (let i = 0; i < input.length; i += 1) {
    const raw = input[i];
    const x = Number.isFinite(raw) ? raw : 0;
    if (!Number.isFinite(raw)) nonFinite += 1;
    audio[i] = x;
    const a = Math.abs(x);
    if (a > peak) peak = a;
    sumSq += x * x;
  }

  const rms = audio.length > 0 ? Math.sqrt(sumSq / audio.length) : 0;
  let gain = 1;
  // Avoid over-amplifying low-level artifacts; only apply gentle gain in a safe band.
  if (peak >= 1e-4 && rms >= 1e-5 && peak < 0.08) {
    gain = Math.min(6, 0.35 / peak);
    for (let i = 0; i < audio.length; i += 1) {
      audio[i] *= gain;
    }
  }

  return {
    audio,
    peak,
    rms,
    gain,
    nonFinite,
  };
}

function isInvalidAudio(prepared) {
  const n = prepared.audio.length;
  if (n <= 0) return true;
  const allNonFinite = prepared.nonFinite === n;
  const mostlyNonFinite = prepared.nonFinite > n * 0.5;
  const effectivelySilent = prepared.peak < 1e-8 && prepared.rms < 1e-9;
  return allNonFinite || (mostlyNonFinite && effectivelySilent);
}

function floatToWavBlob(audio, sampleRate) {
  const bytesPerSample = 2;
  const blockAlign = bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = audio.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeString = (offset, s) => {
    for (let i = 0; i < s.length; i += 1) {
      view.setUint8(offset + i, s.charCodeAt(i));
    }
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < audio.length; i += 1) {
    const x = Math.max(-1, Math.min(1, audio[i]));
    const v = x < 0 ? x * 0x8000 : x * 0x7fff;
    view.setInt16(offset, v, true);
    offset += 2;
  }
  return new Blob([buffer], { type: "audio/wav" });
}

function mixToMono(audioBuffer) {
  const channels = audioBuffer.numberOfChannels;
  const length = audioBuffer.length;
  const mono = new Float32Array(length);
  for (let ch = 0; ch < channels; ch += 1) {
    const data = audioBuffer.getChannelData(ch);
    for (let i = 0; i < length; i += 1) {
      mono[i] += data[i];
    }
  }
  if (channels > 1) {
    for (let i = 0; i < length; i += 1) {
      mono[i] /= channels;
    }
  }
  return mono;
}

async function blobToWavBytes(blob) {
  const arrayBuffer = await blob.arrayBuffer();
  const decodeCtx = new AudioContext();
  try {
    const decoded = await decodeCtx.decodeAudioData(arrayBuffer.slice(0));
    const mono = mixToMono(decoded);
    const wav = floatToWavBlob(mono, decoded.sampleRate);
    return new Uint8Array(await wav.arrayBuffer());
  } finally {
    await decodeCtx.close();
  }
}

function uint8ToBase64(bytes) {
  const chunkSize = 0x8000;
  let binary = "";
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize);
    binary += String.fromCharCode(...chunk);
  }
  return btoa(binary);
}

function base64ToUint8(base64) {
  const binary = atob(base64);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    out[i] = binary.charCodeAt(i);
  }
  return out;
}

function float32ToBase64(arr) {
  if (!(arr instanceof Float32Array)) {
    throw new Error("embedding data is not Float32Array");
  }
  if (arr.length <= 0 || arr.length > MAX_EMBEDDING_FLOATS) {
    throw new Error("embedding size is invalid");
  }
  const bytes = new Uint8Array(arr.buffer.slice(arr.byteOffset, arr.byteOffset + arr.byteLength));
  return uint8ToBase64(bytes);
}

function isLikelyBase64Text(text) {
  return /^[A-Za-z0-9+/=]+$/.test(text);
}

function base64ToFloat32(base64) {
  if (!base64 || base64.length > MAX_LOCAL_EMBEDDING_BASE64_CHARS || !isLikelyBase64Text(base64)) {
    throw new Error("invalid embedding text in localStorage");
  }
  const bytes = base64ToUint8(base64);
  if (bytes.byteLength % 4 !== 0) {
    throw new Error("invalid embedding bytes in localStorage");
  }
  if (bytes.byteLength === 0 || bytes.byteLength / 4 > MAX_EMBEDDING_FLOATS) {
    throw new Error("invalid embedding length in localStorage");
  }
  const copied = new Uint8Array(bytes.byteLength);
  copied.set(bytes);
  return new Float32Array(copied.buffer);
}

function loadLocalReferencesFromStorage() {
  localReferences = [];
  localReferenceCache.clear();
  let changed = false;

  try {
    const raw = localStorage.getItem(STORAGE_KEY_LOCAL_REFERENCES);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return;

    for (const item of parsed) {
      if (!item || typeof item !== "object") continue;
      const key = String(item.key || "").trim();
      const embeddingBase64 = String(item.embeddingBase64 || "").trim();
      if (!key || !embeddingBase64) {
        changed = true;
        continue;
      }
      if (
        embeddingBase64.length > MAX_LOCAL_EMBEDDING_BASE64_CHARS ||
        !isLikelyBase64Text(embeddingBase64)
      ) {
        changed = true;
        continue;
      }
      localReferences.push({ key, embeddingBase64 });
    }

    if (changed) {
      saveLocalReferencesToStorage();
    }
  } catch (_e) {
    // ignore
  }
}

function saveLocalReferencesToStorage() {
  try {
    localStorage.setItem(STORAGE_KEY_LOCAL_REFERENCES, JSON.stringify(localReferences));
  } catch (_e) {
    // ignore quota / private mode errors
  }
}

function nextLocalReferenceKey() {
  let maxId = 0;
  for (const ref of localReferences) {
    const m = /^added_speaker_(\d+)$/.exec(ref.key);
    if (!m) continue;
    const n = Number(m[1]);
    if (Number.isFinite(n)) {
      maxId = Math.max(maxId, n);
    }
  }
  return `added_speaker_${maxId + 1}`;
}

function removeLocalReference(key) {
  const before = localReferences.length;
  localReferences = localReferences.filter((ref) => ref.key !== key);
  localReferenceCache.delete(key);
  if (localReferences.length !== before) {
    saveLocalReferencesToStorage();
  }
}

function rebuildReferenceSelect(preferredValue = "") {
  const prev = preferredValue || ui.embedding.value;
  ui.embedding.innerHTML = "";

  for (const name of presetEmbeddings) {
    const opt = document.createElement("option");
    opt.value = `preset:${name}`;
    opt.textContent = name;
    ui.embedding.appendChild(opt);
  }

  for (const ref of localReferences) {
    const opt = document.createElement("option");
    opt.value = `local:${ref.key}`;
    opt.textContent = `${ref.key} (local)`;
    ui.embedding.appendChild(opt);
  }

  if (ui.embedding.options.length === 0) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "no references found";
    ui.embedding.appendChild(opt);
  }

  if (prev) {
    const found = Array.from(ui.embedding.options).find((o) => o.value === prev);
    if (found) {
      ui.embedding.value = prev;
      return;
    }
  }

  ui.embedding.selectedIndex = 0;
}

async function discoverPresetEmbeddings() {
  let candidates = PRESET_EMBEDDING_CANDIDATES;
  try {
    const res = await fetch(EMBEDDINGS_INDEX_URL, { cache: "no-store" });
    if (res.ok) {
      const body = await res.json();
      if (Array.isArray(body) && body.length > 0) {
        candidates = body.map((v) => String(v));
      }
    }
  } catch (_e) {
    // optional file
  }

  const normalized = [];
  for (const name of candidates) {
    const fileName = String(name || "").trim();
    if (!fileName) continue;
    if (!fileName.endsWith(".emb.gguf")) continue;
    if (!normalized.includes(fileName)) {
      normalized.push(fileName);
    }
  }
  presetEmbeddings = normalized;
}

function embeddingAliasCandidates(fileName) {
  const out = [fileName];
  if (fileName.includes("ja_")) {
    out.push(fileName.replace("ja_", "jp_"));
  }
  if (fileName.includes("jp_")) {
    out.push(fileName.replace("jp_", "ja_"));
  }
  return out.filter((v, i, arr) => arr.indexOf(v) === i);
}

async function loadEmbeddingToEngine(fileName) {
  let lastErr = "";
  for (const candidate of embeddingAliasCandidates(fileName)) {
    const url = new URL(candidate, EMBEDDINGS_BASE_URL).toString();
    const fsPath = `/embeddings/${candidate}`;
    try {
      if (loadedEmbeddingPath !== fsPath) {
        const bytes = await fetchBytes(url);
        writeFileToFS(fsPath, bytes);
        loadedEmbeddingPath = fsPath;
      }

      const ok = await engine.loadEmbedding(fsPath);
      if (!ok) {
        lastErr = await engine.lastError();
        continue;
      }
      return;
    } catch (e) {
      lastErr = String(e?.message ?? e);
    }
  }
  throw new Error(lastErr || `failed to load embedding: ${fileName}`);
}

function getLocalReferenceEmbedding(key) {
  if (localReferenceCache.has(key)) {
    return localReferenceCache.get(key);
  }
  const ref = localReferences.find((v) => v.key === key);
  if (!ref) {
    throw new Error(`local reference not found: ${key}`);
  }
  let arr;
  try {
    arr = base64ToFloat32(ref.embeddingBase64);
  } catch (_e) {
    removeLocalReference(key);
    rebuildReferenceSelect();
    throw new Error(`local reference '${key}' is broken and was removed`);
  }
  localReferenceCache.set(key, arr);
  return arr;
}

async function applySelectedReferenceToEngine() {
  const value = ui.embedding.value;
  if (!value) {
    throw new Error("reference がありません。");
  }

  if (value.startsWith("preset:")) {
    const name = value.slice("preset:".length);
    await loadEmbeddingToEngine(name);
    return;
  }

  if (value.startsWith("local:")) {
    const key = value.slice("local:".length);
    const emb = getLocalReferenceEmbedding(key);
    const ok = await engine.setEmbeddingFromF32(emb);
    if (!ok) {
      throw new Error(await engine.lastError());
    }
    loadedEmbeddingPath = "";
    return;
  }

  throw new Error(`unknown reference selection: ${value}`);
}

function parseCodeToken(raw) {
  const token = String(raw ?? "").trim();
  const wrapped = /^<\|s_(-?\d+)\|>$/.exec(token);
  if (wrapped) {
    return Number(wrapped[1]);
  }
  if (!/^[-+]?\d+$/.test(token)) {
    throw new Error(`invalid code token: ${token}`);
  }
  return Number(token);
}

function parseCodesArray(raw) {
  if (!Array.isArray(raw)) return null;
  const out = [];
  for (const value of raw) {
    if (typeof value === "number" && Number.isFinite(value)) {
      out.push(Math.trunc(value));
      continue;
    }
    if (typeof value === "string") {
      out.push(parseCodeToken(value));
      continue;
    }
    return null;
  }
  return out.length > 0 ? out : null;
}

function extractTextFromApiJson(obj) {
  const parts = [];

  const appendContent = (value) => {
    if (typeof value === "string" && value.length > 0) {
      parts.push(value);
      return;
    }
    if (Array.isArray(value)) {
      for (const item of value) {
        if (typeof item === "string") {
          parts.push(item);
        } else if (item && typeof item === "object") {
          if (typeof item.text === "string") {
            parts.push(item.text);
          } else if (typeof item.content === "string") {
            parts.push(item.content);
          }
        }
      }
      return;
    }
    if (value && typeof value === "object") {
      if (typeof value.text === "string") {
        parts.push(value.text);
      } else if (typeof value.content === "string") {
        parts.push(value.content);
      }
    }
  };

  appendContent(obj.text);
  appendContent(obj.output_text);

  if (Array.isArray(obj.choices)) {
    for (const choice of obj.choices) {
      if (!choice || typeof choice !== "object") continue;
      appendContent(choice.text);
      if (choice.message && typeof choice.message === "object") {
        appendContent(choice.message.content);
      }
    }
  }

  if (Array.isArray(obj.output)) {
    for (const item of obj.output) {
      if (!item || typeof item !== "object") continue;
      appendContent(item.content);
      appendContent(item.text);
    }
  }

  return parts.join("\n");
}

function extractCodesFromText(text) {
  CODE_TOKEN_RE.lastIndex = 0;
  const out = [];
  for (;;) {
    const m = CODE_TOKEN_RE.exec(text);
    if (!m) break;
    out.push(Number(m[1]));
  }
  return out;
}

function parseCodesFromApiBody(bodyText) {
  try {
    const obj = JSON.parse(bodyText);
    const direct =
      parseCodesArray(obj.codes_values) ||
      parseCodesArray(obj.codes) ||
      parseCodesArray(obj.audio_codes);
    if (direct) {
      return direct;
    }
    const text = extractTextFromApiJson(obj);
    if (text) {
      const fromText = extractCodesFromText(text);
      if (fromText.length > 0) {
        return fromText;
      }
    }
  } catch (_e) {
    // non-json fallback below
  }

  const fallback = extractCodesFromText(bodyText);
  if (fallback.length > 0) {
    return fallback;
  }
  throw new Error("external API response does not include Mio codes");
}

function normalizeApiEndpoint(rawBase) {
  const trimmed = rawBase.trim();
  if (!trimmed) {
    throw new Error("API Base URL is empty");
  }

  let url;
  try {
    url = new URL(trimmed);
  } catch (_e) {
    url = new URL(`http://${trimmed}`);
  }

  const normalizedPath = url.pathname.replace(/\/+$/, "").toLowerCase();
  if (
    normalizedPath.endsWith("/v1/chat/completions") ||
    normalizedPath.endsWith("/v1/completions") ||
    normalizedPath.endsWith("/mio/tts")
  ) {
    return url.toString();
  }
  if (normalizedPath.endsWith("/v1")) {
    return `${url.toString().replace(/\/+$/, "")}/chat/completions`;
  }
  return `${url.toString().replace(/\/+$/, "")}/v1/chat/completions`;
}

async function fetchCodesViaExternalApi({
  baseUrl,
  apiKey,
  model,
  text,
  nPredict,
  topK,
  topP,
  temp,
  signal,
}) {
  const endpoint = normalizeApiEndpoint(baseUrl);
  const endpointUrl = new URL(endpoint);
  const endpointPath = endpointUrl.pathname.toLowerCase();
  const useLegacyEndpoint = endpointPath.endsWith("/mio/tts");
  const useCompletionsEndpoint = endpointPath.endsWith("/v1/completions");

  let payload;
  if (useLegacyEndpoint) {
    payload = {
      text,
      n_predict: nPredict,
      top_k: topK,
      top_p: topP,
      temp,
      codes_only: true,
      model: model || undefined,
    };
  } else if (useCompletionsEndpoint) {
    payload = {
      prompt: text,
      max_tokens: nPredict,
      n_predict: nPredict,
      top_k: topK,
      top_p: topP,
      temperature: temp,
      stream: false,
      model: model || undefined,
    };
  } else {
    payload = {
      messages: [{ role: "user", content: text }],
      max_tokens: nPredict,
      n_predict: nPredict,
      top_k: topK,
      top_p: topP,
      temperature: temp,
      stream: false,
      model: model || undefined,
    };
  }

  const headers = {
    "Content-Type": "application/json",
    Accept: "application/json",
  };
  const trimmedKey = apiKey.trim();
  if (trimmedKey) {
    headers.Authorization = `Bearer ${trimmedKey}`;
    headers["X-API-Key"] = trimmedKey;
  }

  const res = await fetch(endpoint, {
    method: "POST",
    headers,
    body: JSON.stringify(payload),
    signal,
  });

  const bodyText = await res.text();
  if (!res.ok) {
    const shortBody = bodyText.length > 320 ? `${bodyText.slice(0, 320)}...` : bodyText;
    throw new Error(`external API HTTP ${res.status}: ${shortBody}`);
  }
  return parseCodesFromApiBody(bodyText);
}

function updateExternalApiVisibility() {
  ui.apiSettings.style.display = ui.useExternalApi.checked ? "grid" : "none";
}

function getParams() {
  return {
    nThreads: parseThreads(),
    nCtx: parseIntInput(ui.ctxSize, 700, 256),
    nPredict: parseIntInput(ui.nPredict, 200, 1, 300),
    topK: parseIntInput(ui.topK, 50, 1),
    topP: parseFloatInput(ui.topP, 1.0, 0.01, 1.0),
    temp: parseFloatInput(ui.temp, 0.8, 0.01, 3.0),
  };
}

function abortGeneration() {
  if (currentAbortController) {
    currentAbortController.abort();
    currentAbortController = null;
  }
}

async function ensureWavlmLoaded(signal) {
  if (wavlmLoaded) {
    return;
  }
  if (wavlmLoadingPromise) {
    await wavlmLoadingPromise;
    return;
  }

  wavlmLoadingPromise = (async () => {
    setRefStatus(`loading model: ${WAVLM_MODEL_URL}`);
    await fetchFileToFS(WAVLM_MODEL_URL, WAVLM_MODEL_FS_PATH, signal);
    wavlmLoaded = true;
  })();

  try {
    await wavlmLoadingPromise;
  } finally {
    wavlmLoadingPromise = null;
  }
}

async function loadWavlmModelManually() {
  if (!engine || isBusy) return;

  abortGeneration();
  currentAbortController = new AbortController();
  setBusy(true);
  try {
    await ensureWavlmLoaded(currentAbortController.signal);
    setRefStatus("WavLM loaded");
  } catch (error) {
    if (error?.name === "AbortError") {
      setRefStatus("stopped");
      return;
    }
    setRefStatus(String(error?.message ?? error));
  } finally {
    currentAbortController = null;
    setBusy(false);
  }
}

function deleteSelectedReference() {
  const value = ui.embedding.value || "";
  if (!value) {
    setRefStatus("reference がありません。");
    return;
  }
  if (value.startsWith("preset:")) {
    setRefStatus("preset reference は削除できません。");
    return;
  }
  if (!value.startsWith("local:")) {
    setRefStatus("local reference を選択してください。");
    return;
  }

  const key = value.slice("local:".length);
  removeLocalReference(key);
  rebuildReferenceSelect();
  setRefStatus(`deleted reference: ${key}`);
}

async function createReferenceFromAudioPath(audioPath) {
  const p = getParams();

  await ensureWavlmLoaded(currentAbortController?.signal);

  setRefStatus("creating reference embedding...");
  const ok = await engine.createReferenceFromAudio(audioPath, WAVLM_MODEL_FS_PATH, 20.0, p.nThreads);
  if (!ok) {
    throw new Error(await engine.lastError());
  }

  const embView = await engine.embeddingF32();
  const emb = new Float32Array(embView.length);
  emb.set(embView);

  const key = nextLocalReferenceKey();
  localReferenceCache.set(key, emb);
  localReferences.push({
    key,
    embeddingBase64: float32ToBase64(emb),
  });
  saveLocalReferencesToStorage();
  rebuildReferenceSelect(`local:${key}`);

  setRefStatus(`added reference: ${key}`);
  setStatus(`reference added: ${key}`);
}

async function createReferenceFromBlob(blob) {
  if (!engine) return;

  abortGeneration();
  currentAbortController = new AbortController();
  setBusy(true);

  const audioPath = `/tmp/ref-${Date.now()}.wav`;
  try {
    setRefStatus("decoding audio...");
    const wavBytes = await blobToWavBytes(blob);
    writeFileToFS(audioPath, wavBytes);
    await createReferenceFromAudioPath(audioPath);
  } catch (error) {
    if (error?.name === "AbortError") {
      setRefStatus("stopped");
      return;
    }
    setRefStatus(String(error?.message ?? error));
  } finally {
    unlinkFileIfExists(audioPath);
    currentAbortController = null;
    setBusy(false);
  }
}

function cleanupRecordingResources() {
  if (mediaRecorderStream) {
    for (const track of mediaRecorderStream.getTracks()) {
      track.stop();
    }
    mediaRecorderStream = null;
  }
  mediaRecorder = null;
  mediaRecorderChunks = [];
  updateRecordingButtons();
}

async function startRecording() {
  if (isBusy) return;

  if (!navigator.mediaDevices?.getUserMedia) {
    setRefStatus("getUserMedia is not supported in this browser");
    return;
  }
  if (typeof MediaRecorder === "undefined") {
    setRefStatus("MediaRecorder is not supported in this browser");
    return;
  }

  cleanupRecordingResources();

  try {
    mediaRecorderStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    const mimeCandidates = [
      "audio/webm;codecs=opus",
      "audio/webm",
      "audio/mp4",
    ];
    let mimeType = "";
    for (const m of mimeCandidates) {
      if (MediaRecorder.isTypeSupported(m)) {
        mimeType = m;
        break;
      }
    }

    mediaRecorder = mimeType
      ? new MediaRecorder(mediaRecorderStream, { mimeType })
      : new MediaRecorder(mediaRecorderStream);

    mediaRecorderChunks = [];
    mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        mediaRecorderChunks.push(event.data);
      }
    };

    mediaRecorder.start(200);
    updateRecordingButtons();
    setRefStatus("recording...");
  } catch (error) {
    cleanupRecordingResources();
    setRefStatus(`recording start failed: ${String(error?.message ?? error)}`);
  }
}

async function stopRecordingAndAddReference() {
  if (!mediaRecorder || mediaRecorder.state !== "recording") {
    return;
  }

  try {
    const done = new Promise((resolve, reject) => {
      mediaRecorder.onstop = () => resolve();
      mediaRecorder.onerror = (event) => {
        reject(event?.error || new Error("recording failed"));
      };
    });

    mediaRecorder.stop();
    await done;

    const blob = new Blob(mediaRecorderChunks, {
      type: mediaRecorder.mimeType || "audio/webm",
    });

    cleanupRecordingResources();
    await createReferenceFromBlob(blob);
  } catch (error) {
    cleanupRecordingResources();
    setRefStatus(`recording stop failed: ${String(error?.message ?? error)}`);
  }
}

async function addReferenceFromFileInput() {
  const file = ui.refFile.files && ui.refFile.files[0];
  if (!file) {
    setRefStatus("audio file を選択してください。");
    return;
  }

  await createReferenceFromBlob(file);
  ui.refFile.value = "";
}

async function generate() {
  if (!engine) return;
  if (isRecordingActive()) {
    setStatus("録音中です。先に Stop & Add を押してください。");
    return;
  }

  const text = ui.text.value.trim();
  if (!text) {
    setStatus("text を入力してください。");
    return;
  }

  const p = getParams();
  const useExternalApi = ui.useExternalApi.checked;

  if (!useExternalApi && !localLLMReady) {
    setStatus("ローカルLLMが未ロードです。外部APIをONにするか、public/model/MioTTS-0.1B-Q8_0.gguf を配置してください。");
    return;
  }

  abortGeneration();
  currentAbortController = new AbortController();
  setBusy(true);
  setStatus("generating...");
  const t0 = performance.now();
  try {
    await applySelectedReferenceToEngine();

    let codesCount = 0;
    let usedCodesText = "";
    let ok = false;
    if (useExternalApi) {
      const codes = await fetchCodesViaExternalApi({
        baseUrl: ui.apiBase.value,
        apiKey: ui.apiKey.value,
        model: ui.apiModel.value.trim(),
        text,
        nPredict: p.nPredict,
        topK: p.topK,
        topP: p.topP,
        temp: p.temp,
        signal: currentAbortController.signal,
      });
      codesCount = codes.length;
      usedCodesText = codes.join(" ");
      ok = await engine.synthesizeFromCodes(usedCodesText, p.nThreads);
    } else {
      ok = await engine.synthesizeFromText(
        text,
        p.nPredict,
        p.topK,
        p.topP,
        p.temp,
        p.nThreads,
        p.nCtx,
      );
      const codesView = await engine.lastCodesI32();
      codesCount = codesView.length;
      usedCodesText = Array.from(codesView).join(" ");
    }

    if (!ok) {
      throw new Error(await engine.lastError());
    }

    let sr = await engine.sampleRate();
    if (!Number.isFinite(sr) || sr <= 0) {
      sr = 44100;
    }
    const view = await engine.audioF32();
    const pcm = new Float32Array(view.length);
    pcm.set(view);
    const prepared = preparePlaybackAudio(pcm);
    await playFloatAudio(prepared.audio, sr);

    const t1 = performance.now();
    const elapsedMs = Math.max(1, t1 - t0);
    const elapsedSec = elapsedMs / 1000;
    const audioSec = sr > 0 ? prepared.audio.length / sr : 0;
    const realtimeX = audioSec > 0 ? audioSec / elapsedSec : 0;
    const codesPerSec = codesCount > 0 ? codesCount / elapsedSec : 0;

    ui.metrics.textContent = [
      `mode=${useExternalApi ? "external-api" : "local-llm"}`,
      `codes=${codesCount}`,
      `samples=${prepared.audio.length}`,
      `sample_rate=${sr}`,
      `peak=${prepared.peak.toFixed(6)}`,
      `rms=${prepared.rms.toFixed(6)}`,
      `gain=${prepared.gain.toFixed(2)}`,
      `non_finite=${prepared.nonFinite}`,
      `elapsed=${elapsedMs.toFixed(0)}ms`,
      `audio_sec=${audioSec.toFixed(2)}s`,
      `speed=${realtimeX.toFixed(2)}x realtime`,
      `codes_per_sec=${codesPerSec.toFixed(1)}`,
      `params: n_threads=${p.nThreads}, ctx=${p.nCtx}, n_predict=${p.nPredict}, top_k=${p.topK}, top_p=${p.topP.toFixed(2)}, temp=${p.temp.toFixed(2)}`,
    ].join(" | ");

    setStatus("done");
  } catch (error) {
    if (error?.name === "AbortError") {
      setStatus("stopped");
      return;
    }
    setStatus(String(error?.message ?? error));
  } finally {
    currentAbortController = null;
    setBusy(false);
  }
}

async function boot() {
  setStatus("loading wasm...");
  moduleRef = await createModule();
  engine = new moduleRef.MioTTSCppCore();

  ensureDir("/models");
  ensureDir("/embeddings");
  ensureDir("/tmp");

  setStatus(`loading model: ${VOCODER_MODEL_URL}`);
  const vocoderPath = "/models/miocodec.gguf";
  await fetchFileToFS(VOCODER_MODEL_URL, vocoderPath);

  let llmPath = "";
  let llmFetchError = "";
  try {
    setStatus(`loading model: ${LLM_MODEL_URL}`);
    llmPath = "/models/MioTTS-0.1B-Q8_0.gguf";
    await fetchFileToFS(LLM_MODEL_URL, llmPath);
  } catch (e) {
    llmFetchError = String(e?.message ?? e);
    console.warn(`llm model unavailable at ${LLM_MODEL_URL}:`, llmFetchError);
    llmPath = "";
  }

  const ok = await engine.init(
    vocoderPath,
    llmPath,
    parseThreads(),
    parseIntInput(ui.ctxSize, 700, 256),
  );

  if (!ok) {
    throw new Error(await engine.lastError());
  }

  localLLMReady = await engine.hasLLM();

  await discoverPresetEmbeddings();
  loadLocalReferencesFromStorage();
  rebuildReferenceSelect();
  updateRecordingButtons();

  setRefStatus("ready");
  let readyMessage = "";
  if (localLLMReady) {
    readyMessage = "ready (CPU, local LLM + external API)";
  } else {
    readyMessage = "ready (CPU, external API only)";
  }
  if (llmFetchError) {
    readyMessage += ` | LLM load failed: ${llmFetchError}`;
  }
  setStatus(readyMessage);
}

ui.btnGenerate.addEventListener("click", () => {
  generate().catch((e) => setStatus(String(e?.message ?? e)));
});

ui.btnStop.addEventListener("click", () => {
  abortGeneration();
  stopPlayback();
  setStatus("stopped");
});

ui.btnLoadWavlm.addEventListener("click", () => {
  loadWavlmModelManually().catch((e) => setRefStatus(String(e?.message ?? e)));
});

ui.btnDeleteReference.addEventListener("click", () => {
  deleteSelectedReference();
});

ui.btnRecordStart.addEventListener("click", () => {
  startRecording().catch((e) => setRefStatus(String(e?.message ?? e)));
});

ui.btnRecordStop.addEventListener("click", () => {
  stopRecordingAndAddReference().catch((e) => setRefStatus(String(e?.message ?? e)));
});

ui.btnAddFile.addEventListener("click", () => {
  addReferenceFromFileInput().catch((e) => setRefStatus(String(e?.message ?? e)));
});

ui.useExternalApi.addEventListener("change", updateExternalApiVisibility);

updateExternalApiVisibility();
setBusy(false);

if (typeof window !== "undefined") {
  window.__miottsResolvedAssetUrls = {
    vocoder: VOCODER_MODEL_URL,
    llm: LLM_MODEL_URL,
    wavlm: WAVLM_MODEL_URL,
    embeddingsIndex: EMBEDDINGS_INDEX_URL,
  };
}

boot().catch((error) => {
  setStatus(String(error?.message ?? error));
});
