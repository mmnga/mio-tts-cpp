#!/bin/bash
if [ ! -d llama.cpp ]; then
  echo "Cloning llama.cpp..."
  git clone https://github.com/ggml-org/llama.cpp.git llama.cpp
fi

echo "Downloading models..."
./models_download.sh

echo "Building..."
./build.sh

echo "Done."
