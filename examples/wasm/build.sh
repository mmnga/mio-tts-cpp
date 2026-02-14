#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LLAMA_DIR="${ROOT_DIR}/llama.cpp"
REPO_MODELS_DIR="${ROOT_DIR}/models"
PUBLIC_MODEL_DIR="${SCRIPT_DIR}/public/model"
PUBLIC_EMBEDDINGS_DIR="${SCRIPT_DIR}/public/embeddings"
BUILD_DIR="${SCRIPT_DIR}/.build-wasm"
OUT_JS="${SCRIPT_DIR}/miottscpp_core.js"

WEBGPU_MODE="${WASM_WEBGPU:-off}" # on | off
WEBGPU_ENABLED=0
WEBGPU_SHADER_DIR="${LLAMA_DIR}/ggml/src/ggml-webgpu/wgsl-shaders"
WEBGPU_GENERATED_DIR="${BUILD_DIR}/generated-webgpu"
WEBGPU_SHADER_HEADER="${WEBGPU_GENERATED_DIR}/ggml-wgsl-shaders.hpp"
WASM_ASSERTIONS="${WASM_ASSERTIONS:-1}" # 1 to include assertion details on abort

if ! command -v em++ >/dev/null 2>&1; then
  echo "em++ not found. activate emsdk first."
  exit 1
fi
if ! command -v emcc >/dev/null 2>&1; then
  echo "emcc not found. activate emsdk first."
  exit 1
fi

mkdir -p "${BUILD_DIR}"
rm -f "${BUILD_DIR}"/*.o
mkdir -p "${PUBLIC_MODEL_DIR}" "${PUBLIC_EMBEDDINGS_DIR}"

copy_assets_if_missing() {
  local missing=0

  local llm_src=""
  for cand in MioTTS-0.1B-Q8_0.gguf MioTTS-0.1B-Q4_0.gguf MioTTS-0.1B.gguf; do
    if [[ -f "${REPO_MODELS_DIR}/${cand}" ]]; then
      llm_src="${REPO_MODELS_DIR}/${cand}"
      break
    fi
  done
  if [[ -n "${llm_src}" ]]; then
    cp -f "${llm_src}" "${PUBLIC_MODEL_DIR}/MioTTS-0.1B-Q8_0.gguf"
    echo "synced llm: ${llm_src} -> ${PUBLIC_MODEL_DIR}/MioTTS-0.1B-Q8_0.gguf"
  else
    echo "error: llm model not found in ${REPO_MODELS_DIR}" >&2
    missing=1
  fi

  local model_src=""
  for cand in miocodec.gguf miocodec-25hz_44khz.gguf miocodec-25hz.gguf; do
    if [[ -f "${REPO_MODELS_DIR}/${cand}" ]]; then
      model_src="${REPO_MODELS_DIR}/${cand}"
      break
    fi
  done
  if [[ -n "${model_src}" ]]; then
    cp -f "${model_src}" "${PUBLIC_MODEL_DIR}/miocodec.gguf"
    echo "synced model: ${model_src} -> ${PUBLIC_MODEL_DIR}/miocodec.gguf"
  else
    echo "error: miocodec model not found in ${REPO_MODELS_DIR}" >&2
    missing=1
  fi

  local wavlm_src=""
  for cand in wavlm_base_plus_2l_f32.gguf wavlm_base_plus.gguf wavlm.gguf; do
    if [[ -f "${REPO_MODELS_DIR}/${cand}" ]]; then
      wavlm_src="${REPO_MODELS_DIR}/${cand}"
      break
    fi
  done
  if [[ -n "${wavlm_src}" ]]; then
    cp -f "${wavlm_src}" "${PUBLIC_MODEL_DIR}/wavlm_base_plus_2l_f32.gguf"
    echo "synced wavlm: ${wavlm_src} -> ${PUBLIC_MODEL_DIR}/wavlm_base_plus_2l_f32.gguf"
  else
    echo "error: wavlm model not found in ${REPO_MODELS_DIR}" >&2
    missing=1
  fi

  shopt -s nullglob
  find "${PUBLIC_EMBEDDINGS_DIR}" -maxdepth 1 -type f -name '*.emb.gguf' -delete
  local emb_sources=("${REPO_MODELS_DIR}"/*.emb.gguf)
  for src in "${emb_sources[@]}"; do
    local name
    name="$(basename "${src}")"
    local dst="${PUBLIC_EMBEDDINGS_DIR}/${name}"
    cp -f "${src}" "${dst}"
    echo "synced embedding: ${src} -> ${dst}"
  done
  if (( ${#emb_sources[@]} == 0 )); then
    echo "error: no *.emb.gguf found in ${REPO_MODELS_DIR}" >&2
    missing=1
  fi
  shopt -u nullglob

  shopt -s nullglob
  local emb_files=("${PUBLIC_EMBEDDINGS_DIR}"/*.emb.gguf)
  if (( ${#emb_files[@]} > 0 )); then
    {
      echo "["
      local i=0
      local last=$(( ${#emb_files[@]} - 1 ))
      for path in "${emb_files[@]}"; do
        local file_name
        file_name="$(basename "${path}")"
        if (( i < last )); then
          printf '  "%s",\n' "${file_name}"
        else
          printf '  "%s"\n' "${file_name}"
        fi
        i=$((i + 1))
      done
      echo "]"
    } > "${PUBLIC_EMBEDDINGS_DIR}/index.json"
  fi
  shopt -u nullglob

  if (( missing != 0 )); then
    echo "error: required assets are missing. place models under ${REPO_MODELS_DIR} and re-run build.sh" >&2
    exit 1
  fi
}

prepare_webgpu_header() {
  mkdir -p "${WEBGPU_GENERATED_DIR}"
  if ! command -v python3 >/dev/null 2>&1; then
    echo "warning: python3 not found. WebGPU build disabled." >&2
    return 1
  fi
  python3 "${LLAMA_DIR}/ggml/src/ggml-webgpu/wgsl-shaders/embed_wgsl.py" \
    --input_dir "${WEBGPU_SHADER_DIR}" \
    --output_file "${WEBGPU_SHADER_HEADER}"
}

probe_webgpu_port() {
  local probe_cpp="${BUILD_DIR}/webgpu_probe.cpp"
  local probe_js="${BUILD_DIR}/webgpu_probe.js"
  cat > "${probe_cpp}" <<'PROBE'
int main() { return 0; }
PROBE

  if em++ "${probe_cpp}" --use-port=emdawnwebgpu -sASYNCIFY -fexceptions -o "${probe_js}" >/dev/null 2>&1; then
    rm -f "${probe_cpp}" "${probe_js}" "${probe_js%.js}.wasm"
    return 0
  fi

  rm -f "${probe_cpp}" "${probe_js}" "${probe_js%.js}.wasm"
  return 1
}

copy_assets_if_missing

if [[ "${WEBGPU_MODE}" != "off" ]]; then
  if prepare_webgpu_header && probe_webgpu_port; then
    WEBGPU_ENABLED=1
    echo "WebGPU build: enabled"
  else
    if [[ "${WEBGPU_MODE}" == "on" ]]; then
      echo "error: WASM_WEBGPU=on but emdawnwebgpu/WebGPU build prerequisites are missing." >&2
      exit 1
    fi
    WEBGPU_ENABLED=0
    echo "WebGPU build: disabled (falling back to CPU-only)"
  fi
else
  echo "WebGPU build: disabled by WASM_WEBGPU=off"
fi

COMMON_FLAGS=(
  -O3
  -I"${ROOT_DIR}/src"
  -I"${LLAMA_DIR}/include"
  -I"${LLAMA_DIR}/ggml/include"
  -I"${LLAMA_DIR}/ggml/src"
  -I"${LLAMA_DIR}/ggml/src/ggml-cpu"
  -I"${LLAMA_DIR}"
  -I"${LLAMA_DIR}/vendor"
  -DGGML_USE_CPU
  -DGGML_USE_CPU_AARCH64=0
  -DGGML_USE_CPU_X86=0
  -DGGML_VERSION=\"wasm\"
  -DGGML_COMMIT=\"wasm\"
  -D_XOPEN_SOURCE=700
  -D_GNU_SOURCE
  -DM_PI=3.14159265358979323846
)

if (( WEBGPU_ENABLED == 1 )); then
  COMMON_FLAGS+=(
    -DGGML_USE_WEBGPU
    -I"${WEBGPU_GENERATED_DIR}"
    -fexceptions
    --use-port=emdawnwebgpu
  )
fi

C_SOURCES=(
  "${LLAMA_DIR}/ggml/src/ggml.c"
  "${LLAMA_DIR}/ggml/src/ggml-alloc.c"
  "${LLAMA_DIR}/ggml/src/ggml-quants.c"
  "${LLAMA_DIR}/ggml/src/ggml-cpu/ggml-cpu.c"
  "${LLAMA_DIR}/ggml/src/ggml-cpu/quants.c"
  "${LLAMA_DIR}/ggml/src/ggml-cpu/arch/wasm/quants.c"
)

CPP_SOURCES=(
  "${SCRIPT_DIR}/wasm/miottscpp_core.cpp"
  "${ROOT_DIR}/src/mio-tts-lib.cpp"
  "${ROOT_DIR}/src/miocodec-decoder.cpp"
  "${ROOT_DIR}/src/wavlm-extractor.cpp"
  "${LLAMA_DIR}/ggml/src/ggml.cpp"
  "${LLAMA_DIR}/ggml/src/ggml-backend.cpp"
  "${LLAMA_DIR}/ggml/src/ggml-backend-reg.cpp"
  "${LLAMA_DIR}/ggml/src/ggml-opt.cpp"
  "${LLAMA_DIR}/ggml/src/ggml-threading.cpp"
  "${LLAMA_DIR}/ggml/src/gguf.cpp"
  "${LLAMA_DIR}/ggml/src/ggml-cpu/ggml-cpu.cpp"
  "${LLAMA_DIR}/ggml/src/ggml-cpu/repack.cpp"
  "${LLAMA_DIR}/ggml/src/ggml-cpu/hbm.cpp"
  "${LLAMA_DIR}/ggml/src/ggml-cpu/traits.cpp"
  "${LLAMA_DIR}/ggml/src/ggml-cpu/binary-ops.cpp"
  "${LLAMA_DIR}/ggml/src/ggml-cpu/unary-ops.cpp"
  "${LLAMA_DIR}/ggml/src/ggml-cpu/vec.cpp"
  "${LLAMA_DIR}/ggml/src/ggml-cpu/ops.cpp"
)

if (( WEBGPU_ENABLED == 1 )); then
  CPP_SOURCES+=("${LLAMA_DIR}/ggml/src/ggml-webgpu/ggml-webgpu.cpp")
fi

while IFS= read -r src; do
  CPP_SOURCES+=("${src}")
done < <(find "${LLAMA_DIR}/src" -maxdepth 1 -name '*.cpp' | sort)

while IFS= read -r src; do
  CPP_SOURCES+=("${src}")
done < <(find "${LLAMA_DIR}/src/models" -maxdepth 1 -name '*.cpp' | sort)

OBJECTS=()

for src in "${C_SOURCES[@]}"; do
  obj_name="$(echo "${src}" | sed 's#^/##; s#[/ ]#_#g').o"
  obj="${BUILD_DIR}/${obj_name}"
  emcc "${COMMON_FLAGS[@]}" -std=c11 -c "${src}" -o "${obj}"
  OBJECTS+=("${obj}")
done

for src in "${CPP_SOURCES[@]}"; do
  obj_name="$(echo "${src}" | sed 's#^/##; s#[/ ]#_#g').o"
  obj="${BUILD_DIR}/${obj_name}"
  em++ "${COMMON_FLAGS[@]}" -std=c++17 -c "${src}" -o "${obj}"
  OBJECTS+=("${obj}")
done

LINK_FLAGS=()
if (( WEBGPU_ENABLED == 1 )); then
  LINK_FLAGS+=(--use-port=emdawnwebgpu -sASYNCIFY -fexceptions)
fi

if (( ${#LINK_FLAGS[@]} > 0 )); then
  em++ \
    -O3 \
    --bind \
    "${OBJECTS[@]}" \
    "${LINK_FLAGS[@]}" \
    -s WASM=1 \
    -s MODULARIZE=1 \
    -s EXPORT_ES6=1 \
    -s ENVIRONMENT=web \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s INITIAL_MEMORY=268435456 \
    -s MAXIMUM_MEMORY=2147483648 \
    -s EXPORTED_RUNTIME_METHODS='["FS"]' \
    -s ERROR_ON_UNDEFINED_SYMBOLS=1 \
    $( [[ "${WASM_ASSERTIONS}" == "1" ]] && echo "-s ASSERTIONS=2 -s STACK_OVERFLOW_CHECK=2" ) \
    -o "${OUT_JS}"
else
  em++ \
    -O3 \
    --bind \
    "${OBJECTS[@]}" \
    -s WASM=1 \
    -s MODULARIZE=1 \
    -s EXPORT_ES6=1 \
    -s ENVIRONMENT=web \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s INITIAL_MEMORY=268435456 \
    -s MAXIMUM_MEMORY=2147483648 \
    -s EXPORTED_RUNTIME_METHODS='["FS"]' \
    -s ERROR_ON_UNDEFINED_SYMBOLS=1 \
    $( [[ "${WASM_ASSERTIONS}" == "1" ]] && echo "-s ASSERTIONS=2 -s STACK_OVERFLOW_CHECK=2" ) \
    -o "${OUT_JS}"
fi

echo "Built:"
echo "  ${SCRIPT_DIR}/miottscpp_core.js"
echo "  ${SCRIPT_DIR}/miottscpp_core.wasm"
