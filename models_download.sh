#!/bin/bash
set -euo pipefail

HF_REPO="${MIO_TTS_MODELS_REPO:-mmnga-o/miotts-cpp-gguf}"
HF_BASE_URL="https://huggingface.co/${HF_REPO}/resolve/main"
WAVLM_REPO="${MIO_TTS_WAVLM_REPO:-mmnga-o/wavlm-base-plus-gguf}"
WAVLM_BASE_URL="https://huggingface.co/${WAVLM_REPO}/resolve/main"

mkdir -p models

download_with_available_tool() {
  local url="$1"
  local dst="$2"

  if command -v wget >/dev/null 2>&1; then
    wget -O "${dst}" "${url}"
    return 0
  fi

  if command -v curl >/dev/null 2>&1; then
    curl -fL "${url}" -o "${dst}"
    return 0
  fi

  echo "error: neither wget nor curl is available" >&2
  return 1
}

download() {
  local name="$1"
  local url_base="${2:-$HF_BASE_URL}"
  local dst="models/${name}"
  local url="${url_base}/${name}"

  if [[ -s "${dst}" ]]; then
    echo "skip (already exists): ${dst}"
    return 0
  fi

  echo "download: ${url} -> ${dst}"
  download_with_available_tool "${url}" "${dst}"
}

echo "Downloading models from: ${HF_REPO}"

echo "[1/8] download LLM"
download "MioTTS-0.1B-Q8_0.gguf"

echo "[2/8] download MioCodec"
download "miocodec.gguf"

echo "[3/8] download MioCodec 24khz"
download "miocodec-24khz.gguf"

echo "[4/8] download MioCodec 44.1khz"
download "miocodec-25hz-44k-v2.gguf"

echo "[5/8] download WavLM"
download "wavlm_base_plus_2l_f32.gguf" "${WAVLM_BASE_URL}"

echo "[6/8] download en_female"
download "en_female.emb.gguf"

echo "[7/8] download en_male"
download "en_male.emb.gguf"

echo "[7/8] download jp_female"
download "jp_female.emb.gguf"

echo "[8/8] download jp_male"
download "jp_male.emb.gguf"
