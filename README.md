# MioTTS-Cpp

C++ inference for [MioTTS](https://huggingface.co/Aratako/MioTTS-0.1B) using [llama.cpp](https://github.com/ggml-org/llama.cpp) and the [GGML](https://github.com/ggml-org/ggml) tensor library.

Runs the full TTS pipeline in C++17: LLM token generation, WavLM speaker embedding extraction, MioCodec vocoder decoding, and iSTFT audio synthesis — without Python or PyTorch at inference time.

## Features

- Full text-to-speech pipeline in C++17 with GGML backend
- Voice cloning from reference audio (WavLM feature extraction + MioCodec global embedding)
- HTTP server with parallel worker slots, reference cache, and streaming support
- CLI for batch processing and scripting
- Mobile support: iOS (SwiftUI) and Android (JNI) sample apps
- Browser support: WASM sample (no server required)
- Sampled decoding (temperature, top-k, top-p, repetition penalty)
- GGUF model format
- External LLM API support (OpenAI-compatible endpoints)

## Prerequisites

- C++17 compiler (Clang or GCC)
- CMake 3.14+
- `git`

Optional:

- `uv` + Python 3.10+ (model conversion only)
- `wget` or `curl` (model download script)

## Quickstart (macOS / Linux)

```bash
# Clone
git clone --recurse-submodules https://github.com/mmnga/mio-tts-cpp.git
cd mio-tts-cpp

# Setup (download models + build in one step)
./setup.sh

# Or run individually:
#   ./models_download.sh   # download models only
#   ./build.sh             # build only

# Synthesize speech
./build/llama-tts-mio \
  -m models/MioTTS-0.1B-Q8_0.gguf \
  -mv models/miocodec.gguf \
  -emb models/en_female.emb.gguf \
  -p "Hello, how are you today?" \
  -o out.wav
```

Expected model artifacts after `models_download.sh`:

| File | Description |
|------|-------------|
| `models/MioTTS-0.1B-Q8_0.gguf` | MioTTS LLM (Q8_0) |
| `models/miocodec.gguf` | MioCodec vocoder |
| `models/wavlm_base_plus_2l_f32.gguf` | WavLM Base+ (2-layer, F32) |
| `models/*.emb.gguf` | Preset speaker embeddings |

## Build

```bash
git clone --recurse-submodules https://github.com/mmnga/mio-tts-cpp.git
cd mio-tts-cpp
./build.sh
```

If you already cloned and `llama.cpp/` is empty:

```bash
git submodule update --init
```

Outputs:

| Binary | Description |
|--------|-------------|
| `build/llama-tts-mio` | CLI |
| `build/mio-tts-server` | HTTP server |

### Build options

`build.sh` respects the following environment variables:

```bash
GGML_CUDA=ON ./build.sh          # Enable CUDA
BUILD_TYPE=Debug ./build.sh       # Debug build
JOBS=8 ./build.sh                 # Parallel jobs
```

## Usage

### CLI

```bash
# Basic synthesis (local LLM)
./build/llama-tts-mio \
  -m models/MioTTS-0.1B-Q8_0.gguf \
  -mv models/miocodec.gguf \
  -emb models/jp_female.emb.gguf \
  -p "こんにちは、今日はいい天気ですね。" \
  -o out.wav

# Voice cloning from reference audio
./build/llama-tts-mio \
  -m models/MioTTS-0.1B-Q8_0.gguf \
  -mv models/miocodec.gguf \
  --tts-wavlm-model models/wavlm_base_plus_2l_f32.gguf \
  --tts-reference-audio reference.wav \
  -p "This sentence uses the cloned voice." \
  -o cloned.wav

# Using external LLM API (vocoder-only local)
./build/llama-tts-mio \
  -mv models/miocodec.gguf \
  -emb models/en_female.emb.gguf \
  --llm-api-url http://localhost:8080/v1 \
  -p "Hello from an external LLM." \
  -o out.wav
```

#### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `-m, --model FNAME` | MioTTS LLM GGUF | (required unless `--llm-api-url`) |
| `-mv, --model-vocoder FNAME` | MioCodec GGUF | (required) |
| `-p, --prompt TEXT` | Input text | |
| `--prompt-file FNAME` | Input text file | |
| `-o, --output FNAME` | Output WAV path | `output.wav` |
| `-n, --n-predict N` | Max generated tokens | `400` |
| `--temp F` | Temperature | `0.8` |
| `--top-p F` | Top-p | `1.0` |
| `--top-k N` | Top-k | `50` |
| `--repeat-penalty F` | Repetition penalty | `1.0` |
| `--seed N` | Sampler seed | `0` |
| `--threads N` | Threads (0 = auto) | `2` |
| `-ngl N` | GPU layers for LLM | `-1` (all) |
| `-fa [on\|off\|auto]` | Flash attention | `auto` |
| `--tts-wavlm-model FNAME` | WavLM GGUF (for reference audio) | |
| `--tts-reference-audio FNAME` | Reference audio file | |
| `-emb, --tts-mio-default-embedding-in FNAME` | Default speaker embedding GGUF | |
| `--llm-api-url URL` | External LLM API endpoint | |

### iOS (SwiftUI)

```bash
./examples/swiftui/build.sh
open examples/swiftui/MioTTSCppDemo.xcodeproj
```

See `examples/swiftui/README.md` for details.

### Android

Open `examples/android/MioTTSCppDemoAndroid` in Android Studio.

See `examples/android/README.md` for details.

### Server

```bash
./build/mio-tts-server \
  -m models/MioTTS-0.1B-Q8_0.gguf \
  -mv models/miocodec.gguf \
  --tts-wavlm-model models/wavlm_base_plus_2l_f32.gguf \
  --reference-file-json '[{"key":"jp_female","path":"models/jp_female.emb.gguf"}]'
```

#### Server Options

| Flag | Description | Default |
|------|-------------|---------|
| `--host STR` | Bind host | `127.0.0.1` |
| `--port N` | Bind port | `18089` |
| `-np, --parallel N` | Synthesis worker slots | `1` |
| `--parallel-reference-generation N` | Reference generation slots | `--parallel` |
| `--mio-backend-devices LIST` | Comma-separated GPU names | |
| `--llm-shared-context on\|off` | Share llama context across slots | `on` |
| `--reference-file-json JSON` | Preload speaker embedding GGUFs | |

#### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/mio/tts` | Text-to-speech synthesis |
| `POST` | `/mio/generate_reference` | Create speaker embedding from audio |
| `POST` | `/mio/add_reference` | Upload a precomputed embedding GGUF |
| `GET`  | `/mio/references` | List cached reference keys |
| `DELETE` | `/mio/references/:key` | Remove a cached reference |
| `GET`  | `/health` | Server health / status |
| `GET`  | `/` | Built-in web UI |

### WASM (Browser)

```bash
cd examples/wasm && ./build.sh
python3 -m http.server 8787 --bind 127.0.0.1
# Open http://127.0.0.1:8787/
```

## Architecture

```
Text ──► [LLM (llama.cpp)] ──► audio codes
                                    │
Reference Audio ──► [WavLM] ──► SSL features ──► [MioCodec Encoder] ──► speaker embedding
                                    │
audio codes + speaker embedding ──► [MioCodec Decoder] ──► mel spectrogram ──► [iSTFT] ──► WAV
```

### Source Files

| File | Component | Description |
|------|-----------|-------------|
| `src/mio-tts-lib.{h,cpp}` | Core C API | Context management, synthesis, reference embedding |
| `src/miocodec-decoder.{h,cpp}` | MioCodec | Decoder + global embedding encoder |
| `src/wavlm-extractor.{h,cpp}` | WavLM | SSL feature extraction from audio |
| `src/tts-mio-cli.cpp` | CLI | Command-line interface |
| `src/tts-mio-server.cpp` | Server | HTTP server with worker pool |
| `src/mio-tts-mobile-shared.hpp` | Mobile | Shared engine logic for iOS / Android |

## Model Conversion (Advanced)

Only needed when converting your own source checkpoints to GGUF.

```bash
uv venv && uv pip install -r requirements.txt
```

| Source | Command | Output |
|--------|---------|--------|
| HF checkpoint | `uv run python llama.cpp/convert_hf_to_gguf.py MODEL_DIR --outfile out.gguf --outtype q8_0` | LLM GGUF |
| MioCodec weights | `uv run python scripts/convert_miocodec_to_gguf.py MODEL_DIR --samples-per-token 1764 -o out.gguf` | Vocoder GGUF |
| WavLM `.pth` | `uv run python scripts/convert_wavlm_base_plus_to_gguf.py --wavlm-weights model.pth --num-transformer-layers 2 -o out.gguf` | WavLM GGUF |
| Speaker `.pt`/`.npz` | `uv run python scripts/convert_preset_embedding_to_gguf.py input.pt -o out.emb.gguf` | Embedding GGUF |

See `scripts/README.md` for details.

## Notes

- Windows support is currently unverified.
- Default external API base URL in samples: `http://localhost:8080/v1` (Android emulator: `http://10.0.2.2:8080/v1`)

## Acknowledgments

- [MioTTS-Inference](https://github.com/Aratako/MioTTS-Inference) — original inference implementation
- [MioTTS](https://huggingface.co/Aratako/MioTTS-0.1B) — original TTS model
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — LLM inference engine
- [GGML](https://github.com/ggml-org/ggml) — tensor library
- [MioCodec](https://huggingface.co/Aratako/MioCodec-25Hz-24kHz) — audio codec
- [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm) — speech representation model

## License

[MIT License](LICENSE)
