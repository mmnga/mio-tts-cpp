#!/usr/bin/env bash
set -euo pipefail

resolve_path() {
  local p="${1:-}"

  if command -v realpath >/dev/null 2>&1; then
    if realpath -m . >/dev/null 2>&1; then
      realpath -m "${p}"
      return
    fi
    if realpath "${p}" >/dev/null 2>&1; then
      realpath "${p}"
      return
    fi
  fi

  if command -v python3 >/dev/null 2>&1; then
    python3 - "${p}" <<'PY'
import os
import sys

p = sys.argv[1] if len(sys.argv) > 1 else ""
print(os.path.abspath(os.path.expanduser(p)))
PY
    return
  fi

  if [[ -d "${p}" ]]; then
    (cd "${p}" && pwd -P)
    return
  fi

  local dir
  dir="$(dirname -- "${p}")"
  local base
  base="$(basename -- "${p}")"
  if [[ -d "${dir}" ]]; then
    echo "$(cd "${dir}" && pwd -P)/${base}"
  else
    echo "${p}"
  fi
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}" && pwd)"
SCRIPT_DIR_REAL="$(resolve_path "${SCRIPT_DIR}")"

BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-}"
DEFAULT_LLAMA_CPP_SOURCE_DIR="${ROOT_DIR}/llama.cpp"

if [[ -z "${LLAMA_CPP_SOURCE_DIR:-}" ]]; then
  LLAMA_CPP_SOURCE_DIR="${DEFAULT_LLAMA_CPP_SOURCE_DIR}"
elif [[ ! -d "${LLAMA_CPP_SOURCE_DIR}" && -d "${DEFAULT_LLAMA_CPP_SOURCE_DIR}" ]]; then
  echo "warning: LLAMA_CPP_SOURCE_DIR does not exist: ${LLAMA_CPP_SOURCE_DIR}" >&2
  echo "warning: falling back to ${DEFAULT_LLAMA_CPP_SOURCE_DIR}" >&2
  LLAMA_CPP_SOURCE_DIR="${DEFAULT_LLAMA_CPP_SOURCE_DIR}"
fi

if [[ ! -d "${LLAMA_CPP_SOURCE_DIR}" ]]; then
  echo "error: LLAMA_CPP_SOURCE_DIR does not exist: ${LLAMA_CPP_SOURCE_DIR}" >&2
  echo "hint: initialize the llama.cpp submodule:" >&2
  echo "hint: git submodule update --init llama.cpp" >&2
  exit 1
fi

BUILD_DIR_REAL="$(resolve_path "${BUILD_DIR}")"
LLAMA_CPP_SOURCE_DIR_REAL="$(resolve_path "${LLAMA_CPP_SOURCE_DIR}")"
GGML_CUDA_OPT="${GGML_CUDA:-}"
GGML_CUDA_NO_VMM_OPT="${GGML_CUDA_NO_VMM:-}"
GGML_CUDA_GRAPHS_OPT="${GGML_CUDA_GRAPHS:-}"
MAX_AUTO_JOBS_CUDA="${MAX_AUTO_JOBS_CUDA:-8}"
MAX_AUTO_JOBS_CPU="${MAX_AUTO_JOBS_CPU:-16}"

if [[ -z "${GGML_CUDA_OPT}" ]]; then
  CUDA_SWITCH="${CUDA:-}"
  CUDA_SWITCH_LC="$(printf '%s' "${CUDA_SWITCH}" | tr '[:upper:]' '[:lower:]')"
  case "${CUDA_SWITCH_LC}" in
    1|on|true|yes)
      GGML_CUDA_OPT="ON"
      ;;
    0|off|false|no)
      GGML_CUDA_OPT="OFF"
      ;;
  esac
fi

# For CUDA builds, default to disabling CUDA VMM to avoid massive VIRT reservations
# when many contexts are preallocated (parallel slots). Override with GGML_CUDA_NO_VMM=OFF.
if [[ -n "${GGML_CUDA_OPT}" && "${GGML_CUDA_OPT}" == "ON" && -z "${GGML_CUDA_NO_VMM_OPT}" ]]; then
  GGML_CUDA_NO_VMM_OPT="ON"
fi

# For CUDA builds, default to disabling CUDA Graphs for server stability with
# high parallel slot counts. Override with GGML_CUDA_GRAPHS=ON.
if [[ -n "${GGML_CUDA_OPT}" && "${GGML_CUDA_OPT}" == "ON" && -z "${GGML_CUDA_GRAPHS_OPT}" ]]; then
  GGML_CUDA_GRAPHS_OPT="OFF"
fi

CACHE_FILE="${BUILD_DIR}/CMakeCache.txt"
if [[ -f "${CACHE_FILE}" ]]; then
  CACHE_SOURCE_DIR="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "${CACHE_FILE}" | head -n 1)"
  CACHE_BUILD_DIR="$(sed -n 's/^CMAKE_CACHEFILE_DIR:INTERNAL=//p' "${CACHE_FILE}" | head -n 1)"
  CACHE_SOURCE_DIR_REAL="$(resolve_path "${CACHE_SOURCE_DIR:-}")"
  CACHE_BUILD_DIR_REAL="$(resolve_path "${CACHE_BUILD_DIR:-}")"

  if [[ "${CACHE_SOURCE_DIR_REAL}" != "${SCRIPT_DIR_REAL}" || "${CACHE_BUILD_DIR_REAL}" != "${BUILD_DIR_REAL}" ]]; then
    echo "==> stale CMake cache detected; recreating build directory"
    echo "  cached source: ${CACHE_SOURCE_DIR_REAL}"
    echo "  current source: ${SCRIPT_DIR_REAL}"
    echo "  cached build : ${CACHE_BUILD_DIR_REAL}"
    echo "  current build: ${BUILD_DIR_REAL}"
    rm -rf "${BUILD_DIR}"
  fi
fi

if [[ $# -gt 0 ]]; then
  TARGETS=("$@")
else
  TARGETS=("mio-tts-lib" "llama-tts-mio" "mio-tts-server" "llama-server")
fi

if [[ -z "${JOBS}" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    JOBS="$(nproc)"
    if [[ -n "${GGML_CUDA_OPT}" && "${GGML_CUDA_OPT}" == "ON" ]]; then
      if [[ "${JOBS}" -gt "${MAX_AUTO_JOBS_CUDA}" ]]; then
        JOBS="${MAX_AUTO_JOBS_CUDA}"
      fi
    else
      if [[ "${JOBS}" -gt "${MAX_AUTO_JOBS_CPU}" ]]; then
        JOBS="${MAX_AUTO_JOBS_CPU}"
      fi
    fi
  else
    JOBS="4"
  fi
fi

echo "==> configure"
echo "  source : ${SCRIPT_DIR_REAL}"
echo "  build  : ${BUILD_DIR_REAL}"
echo "  llama  : ${LLAMA_CPP_SOURCE_DIR_REAL}"
echo "  type   : ${BUILD_TYPE}"
if [[ -n "${GGML_CUDA_OPT}" ]]; then
  echo "  ggml_cuda: ${GGML_CUDA_OPT}"
fi
if [[ -n "${GGML_CUDA_NO_VMM_OPT}" ]]; then
  echo "  ggml_cuda_no_vmm: ${GGML_CUDA_NO_VMM_OPT}"
fi
if [[ -n "${GGML_CUDA_GRAPHS_OPT}" ]]; then
  echo "  ggml_cuda_graphs: ${GGML_CUDA_GRAPHS_OPT}"
fi
echo "  jobs  : ${JOBS}"

CMAKE_ARGS=(
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
  -DLLAMA_CPP_SOURCE_DIR="${LLAMA_CPP_SOURCE_DIR}"
)
if [[ -n "${GGML_CUDA_OPT}" ]]; then
  CMAKE_ARGS+=(-DGGML_CUDA="${GGML_CUDA_OPT}")
fi
if [[ -n "${GGML_CUDA_NO_VMM_OPT}" ]]; then
  CMAKE_ARGS+=(-DGGML_CUDA_NO_VMM="${GGML_CUDA_NO_VMM_OPT}")
fi
if [[ -n "${GGML_CUDA_GRAPHS_OPT}" ]]; then
  CMAKE_ARGS+=(-DGGML_CUDA_GRAPHS="${GGML_CUDA_GRAPHS_OPT}")
fi

cmake -S "${SCRIPT_DIR}" \
      -B "${BUILD_DIR}" \
      "${CMAKE_ARGS[@]}"

echo "==> build (jobs=${JOBS})"
echo "  targets: ${TARGETS[*]}"
cmake --build "${BUILD_DIR}" -j "${JOBS}" --target "${TARGETS[@]}"

echo "==> done"
echo "  binaries:"
for name in "${TARGETS[@]}"; do
  case "${name}" in
    mio-tts-server|llama-tts-mio)
      if [[ -x "${BUILD_DIR}/${name}" ]]; then
        echo "   - ${BUILD_DIR}/${name}"
      fi
      ;;
    llama-server)
      if [[ -x "${BUILD_DIR}/bin/llama-server" ]]; then
        echo "   - ${BUILD_DIR}/bin/llama-server"
      elif [[ -x "${BUILD_DIR}/${name}" ]]; then
        echo "   - ${BUILD_DIR}/${name}"
      fi
      ;;
  esac
done
