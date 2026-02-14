# MioTTS-Cpp Android Demo

Android sample app for local `mio-tts-cpp` inference with optional external LLM API.

## Features

- Load local models (LLM/MioCodec/WavLM)
- Default speaker embeddings auto-register on model load:
  - `en_female.emb.gguf`
  - `en_male.emb.gguf`
  - `jp_female.emb.gguf`
  - `jp_male.emb.gguf`
- Add speakers from:
  - microphone recording (saved as WAV)
  - audio file picker
  - embedding `.gguf` file picker
- Remove selected speaker from app cache (and saved local `.emb.gguf` when present)
- Speaker dropdown (`en_female`, `en_male`, `jp_female`, `jp_male`, `added_speaker_1`, ...)
- Synthesis parameters:
  - `n_predict` (default 200)
  - `ctx-size`
  - `top-k`
  - `top-p`
  - `temp`
- External API toggle (`llama-server` compatible)
  - Base URL default: `http://10.0.2.2:8080/v1` (Android emulator -> host)
  - API key fixed: `dummy`
- Generation state + elapsed time display

## Prerequisites

- Android Studio (recent stable)
- Android SDK + NDK (`29.0.14206865`)
- Android SDK CMake (recommended: `3.29.3`) + Ninja
- JDK 17

## Open Project

Open this folder in Android Studio:

- `examples/android/MioTTSCppDemoAndroid`

## Model Files

Models are bundled from repository `models/*.gguf` at build time.

Required files:

- `models/MioTTS-0.1B-Q8_0.gguf`
- `models/miocodec.gguf`
- `models/wavlm_base_plus_2l_f32.gguf`
- `models/en_female.emb.gguf`
- `models/en_male.emb.gguf`
- `models/jp_female.emb.gguf`
- `models/jp_male.emb.gguf`

## External API Mode

Turn on `外部API` and keep Base URL as:

- `http://10.0.2.2:8080/v1`

The app calls `POST /chat/completions` and parses Mio codes from:

- `codes_values`
- `codes`
- `audio_codes`
- or `<|s_...|>` tokens in text fields.

## Notes

- On x86_64 emulator, Mio backend is forced to CPU in JNI for stability.
