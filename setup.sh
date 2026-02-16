#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f .gitmodules ] && grep -q llama.cpp .gitmodules 2>/dev/null; then
  echo "Initializing submodule..."
  git submodule update --init llama.cpp
elif [ ! -d llama.cpp ] || [ -z "$(ls -A llama.cpp 2>/dev/null)" ]; then
  echo "Cloning llama.cpp..."
  rm -rf llama.cpp
  git clone https://github.com/ggml-org/llama.cpp.git llama.cpp
fi

echo "Downloading models..."
./models_download.sh

echo "Building..."
./build.sh

echo "Done."
