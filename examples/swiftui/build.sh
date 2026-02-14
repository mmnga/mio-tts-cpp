#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LLAMA_CPP_DIR="${LLAMA_CPP_SOURCE_DIR:-$REPO_ROOT/llama.cpp}"
APP_RES_DIR="$REPO_ROOT/examples/swiftui/MioTTSCppDemo/Resources"
MODEL_RES_DIR="$APP_RES_DIR/Models"
EMBED_RES_DIR="$APP_RES_DIR/Embeddings"

LLM_MODEL_SOURCE="${LLM_MODEL_SOURCE:-$REPO_ROOT/models/MioTTS-0.1B-Q8_0.gguf}"
MIOCODEC_MODEL_SOURCE="${MIOCODEC_MODEL_SOURCE:-$REPO_ROOT/models/miocodec.gguf}"
WAVLM_MODEL_SOURCE="${WAVLM_MODEL_SOURCE:-$REPO_ROOT/models/wavlm_base_plus_2l_f32.gguf}"
EMBED_MODEL_SOURCE_DIR="${EMBED_MODEL_SOURCE_DIR:-$REPO_ROOT/models}"
DEFAULT_EMBED_FILES=(
  "en_female.emb.gguf"
  "en_male.emb.gguf"
  "jp_female.emb.gguf"
  "jp_male.emb.gguf"
)

LLM_MODEL_DEST="$MODEL_RES_DIR/MioTTS-0.1B-Q8_0.gguf"
MIOCODEC_MODEL_DEST="$MODEL_RES_DIR/miocodec.gguf"
WAVLM_MODEL_DEST="$MODEL_RES_DIR/wavlm_base_plus_2l_f32.gguf"
IOS_BUILD_ROOT="$REPO_ROOT/build/ios"
BUILD_SIM="$IOS_BUILD_ROOT/sim"
BUILD_DEV="$IOS_BUILD_ROOT/device"
APPLE_OUT_DIR="$IOS_BUILD_ROOT/apple"
LLAMA_XCF_JOBS="${LLAMA_XCF_JOBS:-${JOBS:-}}"

if [[ -z "$LLAMA_XCF_JOBS" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    LLAMA_XCF_JOBS="$(nproc)"
  elif command -v sysctl >/dev/null 2>&1; then
    LLAMA_XCF_JOBS="$(sysctl -n hw.ncpu 2>/dev/null || true)"
  fi
fi

if [[ -z "$LLAMA_XCF_JOBS" || ! "$LLAMA_XCF_JOBS" =~ ^[0-9]+$ || "$LLAMA_XCF_JOBS" -lt 1 ]]; then
  LLAMA_XCF_JOBS=4
fi

copy_required() {
  local src="$1"
  local dst="$2"
  if [[ ! -f "$src" ]]; then
    echo "[error] required file not found: $src" >&2
    exit 1
  fi
  cp -f "$src" "$dst"
}

if [[ ! -f "$LLAMA_CPP_DIR/CMakeLists.txt" ]]; then
  echo "[error] LLAMA_CPP_SOURCE_DIR is invalid: $LLAMA_CPP_DIR" >&2
  exit 1
fi

if [[ ! -x "$LLAMA_CPP_DIR/build-xcframework.sh" ]]; then
  echo "[error] llama.cpp build-xcframework.sh not found: $LLAMA_CPP_DIR/build-xcframework.sh" >&2
  exit 1
fi

mkdir -p "$MODEL_RES_DIR" "$EMBED_RES_DIR"
echo "[1/4] Preparing bundled resources for MioTTSCppDemo"
copy_required "$LLM_MODEL_SOURCE" "$LLM_MODEL_DEST"
copy_required "$MIOCODEC_MODEL_SOURCE" "$MIOCODEC_MODEL_DEST"
copy_required "$WAVLM_MODEL_SOURCE" "$WAVLM_MODEL_DEST"
for emb_file in "${DEFAULT_EMBED_FILES[@]}"; do
  copy_required "$EMBED_MODEL_SOURCE_DIR/$emb_file" "$EMBED_RES_DIR/$emb_file"
done

echo "[2/4] Building llama.xcframework (jobs=$LLAMA_XCF_JOBS)"
(
  cd "$LLAMA_CPP_DIR"
  CMAKE_BUILD_PARALLEL_LEVEL="$LLAMA_XCF_JOBS" ./build-xcframework.sh
)

LLAMA_XCF_SRC="$LLAMA_CPP_DIR/build-apple/llama.xcframework"
LLAMA_XCF_DST="$APPLE_OUT_DIR/llama.xcframework"

if [[ ! -d "$LLAMA_XCF_SRC" ]]; then
  echo "[error] llama.xcframework not found: $LLAMA_XCF_SRC" >&2
  exit 1
fi

mkdir -p "$APPLE_OUT_DIR"
rm -rf "$LLAMA_XCF_DST"
cp -R "$LLAMA_XCF_SRC" "$LLAMA_XCF_DST"

IOS_SIM_ARCHS="${IOS_SIM_ARCHS:-arm64;x86_64}"
IOS_DEVICE_ARCHS="${IOS_DEVICE_ARCHS:-arm64}"
IOS_DEPLOYMENT_TARGET="${IOS_DEPLOYMENT_TARGET:-16.0}"
NO_SIGN_XCODE_ARGS=(
  -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED=NO
  -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED=NO
)


echo "[3/4] Building mio-tts-lib (iphonesimulator)"
cmake -S "$REPO_ROOT" -B "$BUILD_SIM" -G Xcode \
  -DCMAKE_SYSTEM_NAME=iOS \
  -DCMAKE_OSX_SYSROOT=iphonesimulator \
  -DCMAKE_OSX_ARCHITECTURES="$IOS_SIM_ARCHS" \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="$IOS_DEPLOYMENT_TARGET" \
  -DMIO_TTS_BUILD_SERVER=OFF \
  -DMIO_TTS_BUILD_LLAMA_SERVER=OFF \
  "${NO_SIGN_XCODE_ARGS[@]}" \
  -DLLAMA_CPP_SOURCE_DIR="$LLAMA_CPP_DIR"

cmake --build "$BUILD_SIM" --config Release --target mio-tts-lib


echo "[3/4] Building mio-tts-lib (iphoneos)"
cmake -S "$REPO_ROOT" -B "$BUILD_DEV" -G Xcode \
  -DCMAKE_SYSTEM_NAME=iOS \
  -DCMAKE_OSX_SYSROOT=iphoneos \
  -DCMAKE_OSX_ARCHITECTURES="$IOS_DEVICE_ARCHS" \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="$IOS_DEPLOYMENT_TARGET" \
  -DMIO_TTS_BUILD_SERVER=OFF \
  -DMIO_TTS_BUILD_LLAMA_SERVER=OFF \
  "${NO_SIGN_XCODE_ARGS[@]}" \
  -DLLAMA_CPP_SOURCE_DIR="$LLAMA_CPP_DIR"

cmake --build "$BUILD_DEV" --config Release --target mio-tts-lib

SIM_LIB="$(find "$BUILD_SIM" -path '*/Release-iphonesimulator/libmio-tts-lib.a' | head -n 1)"
DEV_LIB="$(find "$BUILD_DEV" -path '*/Release-iphoneos/libmio-tts-lib.a' | head -n 1)"

if [[ -z "$SIM_LIB" || ! -f "$SIM_LIB" ]]; then
  echo "[error] simulator static lib not found under $BUILD_SIM" >&2
  exit 1
fi
if [[ -z "$DEV_LIB" || ! -f "$DEV_LIB" ]]; then
  echo "[error] device static lib not found under $BUILD_DEV" >&2
  exit 1
fi

mkdir -p "$APPLE_OUT_DIR"
OUT_XCF="$APPLE_OUT_DIR/mio_tts.xcframework"
rm -rf "$OUT_XCF"

echo "[4/4] Creating mio_tts.xcframework"
xcodebuild -create-xcframework \
  -library "$SIM_LIB" -headers "$REPO_ROOT/src" \
  -library "$DEV_LIB" -headers "$REPO_ROOT/src" \
  -output "$OUT_XCF"

echo "[done] Created frameworks:"
echo "  $LLAMA_XCF_DST"
echo "  $OUT_XCF"
echo "[done] Bundled resources:"
echo "  $LLM_MODEL_DEST"
echo "  $MIOCODEC_MODEL_DEST"
echo "  $WAVLM_MODEL_DEST"
for emb_file in "${DEFAULT_EMBED_FILES[@]}"; do
  echo "  $EMBED_RES_DIR/$emb_file"
done
