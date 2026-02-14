# MioTTS SwiftUI Demo (Local Inference)

A SwiftUI sample app that runs local inference with `llama.cpp + mio-tts-lib`.

## Features

- Startup auto-load of bundled models
- Default speakers (bundled):
  - `en_female`
  - `en_male`
  - `jp_female`
  - `jp_male`
- Add speakers by:
  - microphone recording (auto-save + auto-register)
  - audio file import (auto-save + auto-register)
- Remove selected speaker key from app cache (and saved local embedding file if present)
- Auto-generated speaker keys: `added_speaker_1`, `added_speaker_2`, ...
- Synthesis from dropdown-selected speaker
- Optional external LLM API mode:
  - Toggle `外部API` to ON in app settings
  - Configure `BaseURL` (default: `http://localhost:8080`)
  - Default `api-key`: `dummy`
  - App requests audio codes from `/mio/tts` (`codes_only=true`) and synthesizes audio locally

## Files

- `MioTTSCppDemoApp.swift`
- `ContentView.swift`
- `MioTTSClient.swift`
- `MioTTSCppDemo-Bridging-Header.h`
- `Native/MioTTSLocalBridge.h`
- `Native/MioTTSLocalBridge.mm`
- `../MioTTSCppDemo.xcodeproj`

## Build

See `../README.md`.
