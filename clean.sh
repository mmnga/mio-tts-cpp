#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

remove_path() {
  local p="$1"
  if [[ -e "${p}" ]]; then
    rm -rf "${p}"
    echo "[removed] ${p}"
  fi
}

is_git_tracked() {
  local p="$1"
  if ! command -v git >/dev/null 2>&1; then
    return 1
  fi
  git ls-files --error-unmatch -- "${p}" >/dev/null 2>&1
}

remove_downloaded_models_only() {
  local model_dir="models"
  if [[ ! -d "${model_dir}" ]]; then
    return
  fi

  # Keep default tracked files; remove only downloadable artifacts that are not tracked.
  local downloaded_files=(
    "MioTTS-0.1B-Q8_0.gguf"
    "miocodec.gguf"
    "wavlm_base_plus_2l_f32.gguf"
    "en_female.emb.gguf"
    "en_male.emb.gguf"
    "jp_female.emb.gguf"
    "jp_male.emb.gguf"
  )

  for name in "${downloaded_files[@]}"; do
    local p="${model_dir}/${name}"
    if [[ -f "${p}" ]]; then
      if is_git_tracked "${p}"; then
        echo "[kept] ${p} (tracked default)"
      else
        rm -f "${p}"
        echo "[removed] ${p}"
      fi
    fi
  done
}

echo "[clean] removing generated artifacts..."

# Core generated artifacts
remove_path "llama.cpp"
remove_path "build"
remove_downloaded_models_only
remove_path ".venv"
remove_path ".cache"
remove_path ".tmp"

# Legacy build dirs that may exist in older setups
shopt -s nullglob
legacy_build_dirs=(build-*)
if (( ${#legacy_build_dirs[@]} > 0 )); then
  rm -rf "${legacy_build_dirs[@]}"
  for p in "${legacy_build_dirs[@]}"; do
    echo "[removed] ${p}"
  done
fi
shopt -u nullglob

# CMake in-source safety cleanup
remove_path "CMakeCache.txt"
remove_path "CMakeFiles"
remove_path "cmake_install.cmake"
remove_path "Makefile"
remove_path "compile_commands.json"

# SwiftUI sample generated caches
remove_path "examples/swiftui/build"
remove_path "examples/swiftui/MioTTSCppDemo.xcodeproj/xcuserdata"
remove_path "examples/swiftui/MioTTSCppDemo.xcodeproj/project.xcworkspace/xcuserdata"

# Android sample generated artifacts
remove_path "examples/android/MioTTSCppDemoAndroid/.gradle"
remove_path "examples/android/MioTTSCppDemoAndroid/.idea"
remove_path "examples/android/MioTTSCppDemoAndroid/build"
remove_path "examples/android/MioTTSCppDemoAndroid/local.properties"
remove_path "examples/android/MioTTSCppDemoAndroid/app/.cxx"
remove_path "examples/android/MioTTSCppDemoAndroid/app/build"

# WASM sample generated artifacts
remove_path "examples/wasm/.build-wasm"
if [[ -d "examples/wasm/public/model" ]]; then
  find "examples/wasm/public/model" -mindepth 1 -maxdepth 1 -type f ! -name ".gitkeep" -delete
  echo "[cleaned] examples/wasm/public/model (kept .gitkeep)"
fi
if [[ -d "examples/wasm/public/embeddings" ]]; then
  find "examples/wasm/public/embeddings" -mindepth 1 -maxdepth 1 -type f ! -name ".gitkeep" ! -name "index.json.example" -delete
  echo "[cleaned] examples/wasm/public/embeddings (kept .gitkeep, index.json.example)"
fi

# Runtime leftovers in repository root
shopt -s nullglob
for f in ./*.log ./core ./core.*; do
  rm -f "${f}"
  echo "[removed] ${f#./}"
done
shopt -u nullglob

echo "[done] clean completed"
