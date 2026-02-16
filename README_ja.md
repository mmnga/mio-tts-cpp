# MioTTS-Cpp

[llama.cpp](https://github.com/ggml-org/llama.cpp) と [GGML](https://github.com/ggml-org/ggml) テンソルライブラリを使った [MioTTS](https://huggingface.co/Aratako/MioTTS-0.1B) の C++ 推論実装です。

LLM トークン生成、WavLM 話者埋め込み抽出、MioCodec ボコーダデコード、iSTFT 音声合成までの全 TTS パイプラインを C++17 で実行します。推論時に Python / PyTorch は不要です。

For English instructions, see `README.md`.

## 特徴

- C++17 + GGML バックエンドによるフル TTS パイプライン
- 参照音声からのボイスクローニング (WavLM 特徴抽出 + MioCodec グローバル埋め込み)
- 並列ワーカー、リファレンスキャッシュ、ストリーミング対応の HTTP サーバー
- バッチ処理・スクリプト向け CLI
- モバイル対応: iOS (SwiftUI) / Android (JNI) サンプルアプリ
- ブラウザ対応: WASM サンプル (サーバー不要)
- サンプリングデコード (temperature, top-k, top-p, repetition penalty)
- GGUF モデルフォーマット
- 外部 LLM API 対応 (OpenAI 互換エンドポイント)

## 前提環境

- C++17 コンパイラ (Clang or GCC)
- CMake 3.14+
- `git`

任意:

- `uv` + Python 3.10+ (モデル変換時のみ)
- `wget` or `curl` (モデルダウンロードスクリプト用)

## クイックスタート (macOS / Linux)

```bash
# サブモジュールごとクローン
git clone --recurse-submodules https://github.com/mmnga/mio-tts-cpp.git
cd mio-tts-cpp

# モデル取得
./models_download.sh

# ビルド
./build.sh

# 音声合成
./build/llama-tts-mio \
  -m models/MioTTS-0.1B-Q8_0.gguf \
  -mv models/miocodec.gguf \
  -emb models/jp_female.emb.gguf \
  -p "こんにちは、今日はいい天気ですね。" \
  -o out.wav
```

`models_download.sh` で取得されるモデル:

| ファイル | 説明 |
|----------|------|
| `models/MioTTS-0.1B-Q8_0.gguf` | MioTTS LLM (Q8_0) |
| `models/miocodec.gguf` | MioCodec Vocoder |
| `models/wavlm_base_plus_2l_f32.gguf` | WavLM Base+ (2層, F32) |
| `models/*.emb.gguf` | プリセット話者埋め込み |

## ビルド

```bash
git clone --recurse-submodules https://github.com/mmnga/mio-tts-cpp.git
cd mio-tts-cpp
./build.sh
```

すでにクローン済みで `llama.cpp/` が空の場合:

```bash
git submodule update --init
```

生成物:

| バイナリ | 説明 |
|----------|------|
| `build/llama-tts-mio` | CLI |
| `build/mio-tts-server` | HTTP サーバー |

### ビルドオプション

`build.sh` は以下の環境変数に対応しています:

```bash
GGML_CUDA=ON ./build.sh          # CUDA 有効化
BUILD_TYPE=Debug ./build.sh       # デバッグビルド
JOBS=8 ./build.sh                 # 並列ジョブ数
```

## 使い方

### CLI

```bash
# 基本的な音声合成 (ローカル LLM)
./build/llama-tts-mio \
  -m models/MioTTS-0.1B-Q8_0.gguf \
  -mv models/miocodec.gguf \
  -emb models/jp_female.emb.gguf \
  -p "こんにちは、今日はいい天気ですね。" \
  -o out.wav

# 参照音声からのボイスクローニング
./build/llama-tts-mio \
  -m models/MioTTS-0.1B-Q8_0.gguf \
  -mv models/miocodec.gguf \
  --tts-wavlm-model models/wavlm_base_plus_2l_f32.gguf \
  --tts-reference-audio reference.wav \
  -p "この文はクローンした声で読み上げます。" \
  -o cloned.wav

# 外部 LLM API 使用 (ボコーダのみローカル)
./build/llama-tts-mio \
  -mv models/miocodec.gguf \
  -emb models/en_female.emb.gguf \
  --llm-api-url http://localhost:8080/v1 \
  -p "Hello from an external LLM." \
  -o out.wav
```

#### CLI オプション

| フラグ | 説明 | デフォルト |
|--------|------|-----------|
| `-m, --model FNAME` | MioTTS LLM GGUF | (`--llm-api-url` 未指定時は必須) |
| `-mv, --model-vocoder FNAME` | MioCodec GGUF | (必須) |
| `-p, --prompt TEXT` | 入力テキスト | |
| `--prompt-file FNAME` | 入力テキストファイル | |
| `-o, --output FNAME` | 出力 WAV パス | `output.wav` |
| `-n, --n-predict N` | 最大生成トークン数 | `400` |
| `--temp F` | Temperature | `0.8` |
| `--top-p F` | Top-p | `1.0` |
| `--top-k N` | Top-k | `50` |
| `--repeat-penalty F` | 繰り返しペナルティ | `1.0` |
| `--seed N` | サンプラーシード | `0` |
| `--threads N` | スレッド数 (0 = 自動) | `2` |
| `-ngl N` | LLM の GPU レイヤ数 | `-1` (全層) |
| `-fa [on\|off\|auto]` | Flash attention | `auto` |
| `--tts-wavlm-model FNAME` | WavLM GGUF (参照音声使用時) | |
| `--tts-reference-audio FNAME` | 参照音声ファイル | |
| `-emb, --tts-mio-default-embedding-in FNAME` | デフォルト話者埋め込み GGUF | |
| `--llm-api-url URL` | 外部 LLM API エンドポイント | |

### iOS (SwiftUI)

```bash
./examples/swiftui/build.sh
open examples/swiftui/MioTTSCppDemo.xcodeproj
```

詳細は `examples/swiftui/README.md` を参照してください。

### Android

Android Studio で `examples/android/MioTTSCppDemoAndroid` を開いてください。

詳細は `examples/android/README.md` を参照してください。

### サーバー

```bash
./build/mio-tts-server \
  -m models/MioTTS-0.1B-Q8_0.gguf \
  -mv models/miocodec.gguf \
  --tts-wavlm-model models/wavlm_base_plus_2l_f32.gguf \
  --reference-file-json '[{"key":"jp_female","path":"models/jp_female.emb.gguf"}]'
```

#### サーバーオプション

| フラグ | 説明 | デフォルト |
|--------|------|-----------|
| `--host STR` | バインドホスト | `127.0.0.1` |
| `--port N` | バインドポート | `18089` |
| `-np, --parallel N` | 合成ワーカースロット数 | `1` |
| `--parallel-reference-generation N` | リファレンス生成スロット数 | `--parallel` |
| `--mio-backend-devices LIST` | GPU 名のカンマ区切りリスト | |
| `--llm-shared-context on\|off` | llama コンテキストをスロット間で共有 | `on` |
| `--reference-file-json JSON` | 話者埋め込み GGUF のプリロード | |

#### API エンドポイント

| メソッド | パス | 説明 |
|----------|------|------|
| `POST` | `/mio/tts` | テキスト音声合成 |
| `POST` | `/mio/generate_reference` | 音声からの話者埋め込み生成 |
| `POST` | `/mio/add_reference` | 事前計算済み埋め込み GGUF のアップロード |
| `GET`  | `/mio/references` | キャッシュ済みリファレンスキー一覧 |
| `DELETE` | `/mio/references/:key` | キャッシュ済みリファレンスの削除 |
| `GET`  | `/health` | サーバーステータス |
| `GET`  | `/` | 組み込み Web UI |

### WASM (ブラウザ)

```bash
cd examples/wasm && ./build.sh
python3 -m http.server 8787 --bind 127.0.0.1
# http://127.0.0.1:8787/ を開く
```

## アーキテクチャ

```
テキスト ──► [LLM (llama.cpp)] ──► オーディオコード
                                        │
参照音声 ──► [WavLM] ──► SSL 特徴量 ──► [MioCodec Encoder] ──► 話者埋め込み
                                        │
オーディオコード + 話者埋め込み ──► [MioCodec Decoder] ──► メルスペクトログラム ──► [iSTFT] ──► WAV
```

### ソースファイル

| ファイル | コンポーネント | 説明 |
|----------|----------------|------|
| `src/mio-tts-lib.{h,cpp}` | コア C API | コンテキスト管理、合成、リファレンス埋め込み |
| `src/miocodec-decoder.{h,cpp}` | MioCodec | デコーダ + グローバル埋め込みエンコーダ |
| `src/wavlm-extractor.{h,cpp}` | WavLM | 音声からの SSL 特徴抽出 |
| `src/tts-mio-cli.cpp` | CLI | コマンドラインインターフェース |
| `src/tts-mio-server.cpp` | Server | ワーカープール付き HTTP サーバー |
| `src/mio-tts-mobile-shared.hpp` | Mobile | iOS / Android 共有エンジンロジック |

## モデル変換 (上級)

独自のチェックポイントから GGUF に変換する場合のみ必要です。

```bash
uv venv && uv pip install -r requirements.txt
```

| ソース | コマンド | 出力 |
|--------|---------|------|
| HF チェックポイント | `uv run python llama.cpp/convert_hf_to_gguf.py MODEL_DIR --outfile out.gguf --outtype q8_0` | LLM GGUF |
| MioCodec 重み | `uv run python scripts/convert_miocodec_to_gguf.py MODEL_DIR --samples-per-token 1764 -o out.gguf` | ボコーダ GGUF |
| WavLM `.pth` | `uv run python scripts/convert_wavlm_base_plus_to_gguf.py --wavlm-weights model.pth --num-transformer-layers 2 -o out.gguf` | WavLM GGUF |
| 話者 `.pt`/`.npz` | `uv run python scripts/convert_preset_embedding_to_gguf.py input.pt -o out.emb.gguf` | 埋め込み GGUF |

詳細は `scripts/README.md` を参照してください。

## 補足

- Windows 環境での動作は未確認です。
- サンプルの外部 API 既定 URL: `http://localhost:8080/v1` (Android エミュレータ: `http://10.0.2.2:8080/v1`)

## 謝辞

- [MioTTS-Inference](https://github.com/Aratako/MioTTS-Inference) — オリジナル推論実装
- [MioTTS](https://huggingface.co/Aratako/MioTTS-0.1B) — オリジナル TTS モデル
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — LLM 推論エンジン
- [GGML](https://github.com/ggml-org/ggml) — テンソルライブラリ
- [MioCodec](https://huggingface.co/Aratako/MioCodec-25Hz-24kHz) — オーディオコーデック
- [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm) — 音声表現モデル

## ライセンス

[MIT License](LICENSE)
