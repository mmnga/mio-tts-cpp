# SwiftUI Samples

`MioTTSCppDemo.xcodeproj` is a local inference sample (`llama.cpp + mio-tts-lib`) for iOS.

## Build (same flow as `llama.cpp/examples/llama.swiftui`)

1. Build required XCFrameworks:

```bash
./examples/swiftui/build.sh
```

This script also prepares app-bundled model resources:

- Copies LLM / MioCodec / WavLM GGUF from local files
- Copies default embeddings:
  - `en_female.emb.gguf`
  - `en_male.emb.gguf`
  - `jp_female.emb.gguf`
  - `jp_male.emb.gguf`

Default sources:

- LLM: `./models/MioTTS-0.1B-Q8_0.gguf`
- MioCodec: `./models/miocodec.gguf`
- WavLM: `./models/wavlm_base_plus_2l_f32.gguf`
- Embeddings: `./models/{en_female,en_male,jp_female,jp_male}.emb.gguf`

Override source paths if needed:

- `LLM_MODEL_SOURCE`
- `MIOCODEC_MODEL_SOURCE`
- `WAVLM_MODEL_SOURCE`
- `EMBED_MODEL_SOURCE_DIR`

If your `llama.cpp` checkout is not `./llama.cpp`:

```bash
LLAMA_CPP_SOURCE_DIR=/absolute/path/to/llama.cpp ./examples/swiftui/build.sh
```

Optional build knobs:

- `IOS_SIM_ARCHS` (default: `arm64;x86_64`)
- `IOS_DEVICE_ARCHS` (default: `arm64`)
- `IOS_DEPLOYMENT_TARGET` (default: `16.0`)

2. Open:

- `examples/swiftui/MioTTSCppDemo.xcodeproj`

3. Build target `MioTTSCppDemo`.

## Common error

If Xcode shows:

- `There is no XCFramework found at .../build/ios/apple/llama.xcframework`
- `There is no XCFramework found at .../build/ios/apple/mio_tts.xcframework`

then XCFrameworks are not built yet. Run:

```bash
./examples/swiftui/build.sh
```

If `build.sh` fails on iOS device build with:

- `Signing for "..." requires a development team`

use the latest script in this repository. It disables code signing for generated
native libraries during XCFramework creation.
