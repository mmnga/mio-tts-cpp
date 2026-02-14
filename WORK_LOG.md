# WORK LOG

## 1. 目的

このリポジトリの目的は、`mio-tts-lib` を中核にして、以下を同じモデル資産で動かせる状態にすることです。

- ライブラリ: `mio-tts-lib`（C API）
- CLI: `llama-tts-mio`
- Server: `mio-tts-server`
- iOS demo: `examples/swiftui/MioTTSCppDemo`
- Android demo: `examples/android/MioTTSCppDemoAndroid`
- WASM demo: `examples/wasm`

目標は「サンプルが動く」だけでなく、モバイル運用を意識した実装（メモリ・安定性・使い勝手）に寄せることです。  
将来的には `llama.cpp` と同様に Raspberry Pi / M5Stack 系でも載せられる構成を意識します。

---

## 2. 実装思想

### 2.1 単一ソースで複数ランタイム

- 推論コアは `src/mio-tts-lib.*` と `src/miocodec-decoder.*`, `src/wavlm-extractor.*` に集約。
- iOS/Android/WASM は「UI層 + ブリッジ」で実現し、推論ロジックの重複を避ける。
- モバイル共通運用ロジックは `src/mio-tts-mobile-shared.hpp` に寄せる。

### 2.2 モデル資産の共通化

- 基本モデル命名を `models/` 基準に統一。
- 現行の主モデル:
  - LLM: `MioTTS-0.1B-Q8_0.gguf`
  - MioCodec: `miocodec.gguf`（MioCodec-25Hz-44.1kHz-v2 系）
  - WavLM: `wavlm_base_plus_2l_f32.gguf`
  - Preset embeddings: `en_female / en_male / jp_female / jp_male`
- ダウンロードは `models_download.sh` で実施。既存ファイルがあればスキップ。

### 2.3 モバイル安定性優先

- Mobile では `flash-attn` を無効寄りで運用。
- iOS 実機/Simulator と Android で、MioCodec/WavLM 経路は CPU backend を基本運用。
- iOS Metal では `UPSCALE` 未対応で abort する経路があるため、CPU固定が実運用の安全策。

### 2.4 メモリ重視

- `mio-tts-mobile-shared.hpp` に LLM runtime unload/load 制御を実装。
- 参照作成時は LLM を落としてピークを下げる。
- 参照作成後に必要なら synthesis 用コンテキストを復元。
- アロケータに対する圧力解放 (`malloc_zone_pressure_relief` / `malloc_trim`) を挿入。
- デコーダ/抽出器の GPU graph cache は小さく（内部 limit=1）運用。

---

## 3. 現在の構成と主要エントリ

- Build script: `build.sh`
  - デフォルトターゲット: `mio-tts-lib`, `llama-tts-mio`, `mio-tts-server`, `llama-server`
  - `LLAMA_CPP_SOURCE_DIR` 未指定時は `./llama.cpp` を使用
- Cleanup: `clean.sh`
- Server run helper: `run-tts-server.sh`
- Model download: `models_download.sh`

---

## 4. コンポーネント別の現状

## 4.1 `mio-tts-lib` (core C API)

- 主要API:
  - 初期化/解放: `mio_tts_init_from_file`, `mio_tts_free`
  - 参照作成: `mio_tts_reference_to_embedding`
  - 音声生成: `mio_tts_synthesize`
  - ワークスペース見積/予約: `mio_tts_estimate_*`, `mio_tts_reserve_*`
  - embedding 保存/読み込み: `mio_tts_embedding_save_gguf`, `mio_tts_embedding_load_gguf`

現状の留意点:

- APIとして「部分アンロード（decoderだけ、wavlmだけ等）」は公開していない。
- メモリ制御は基本的に context ライフサイクルで行う。

## 4.2 CLI (`llama-tts-mio`)

- デフォルト:
  - `n_threads=2`
  - `ctx=700`
  - `n_predict=400`
- 外部API対応:
  - `--llm-api-url`, `--llm-api-key`, `--llm-api-model`, `--llm-api-headers`, `--llm-api-mode`

用途:

- 単発生成、バッチ、モデル検証、サーバー不要のデバッグ用途。

## 4.3 Server (`mio-tts-server`)

- HTTP API で音声生成。
- 外部LLM API対応あり（OpenAI互換/汎用モード）。
- 追加話者保存先:
  - `--reference-added-output-dir` を実装済み。
- 並列設定:
  - `--parallel`
  - `--parallel-reference-generation`
  - `--llm-shared-context` 周辺

現状の留意点:

- Apple Metal backend で MioCodec 経路に `UPSCALE` 未対応問題があり、GPU実行で abort する構成がある。
- Server運用時は backend 設定に注意（必要なら CPU backend へ）。

## 4.4 iOS demo (`MioTTSCppDemo`)

実装の主眼:

- モデル起動時ロード、話者ドロップダウン、録音/音声ファイルからの話者追加、再生。
- 追加話者 `added_speaker_N` を `.emb.gguf` として永続化。
- 追加話者削除時は `.emb.gguf` も削除（presetは削除不可）。
- 外部APIトグルあり（BaseURL入力、api-keyは `dummy` 運用）。
- 生成中表示・生成時間表示あり。

留意点:

- iOS は Mio 経路 CPU backend 運用。
- 参照作成時にメモリピークを下げるため LLM/runtime を制御。

## 4.5 Android demo (`MioTTSCppDemoAndroid`)

実装の主眼:

- iOS相当機能を Android UI に実装。
- 録音/音声ファイル/embファイルで話者追加、ドロップダウン選択、削除、再生。
- 外部APIトグル:
  - エミュレータ既定 URL: `http://10.0.2.2:8080/v1`
  - api-key: `dummy`

留意点:

- JNI で Mio 経路 CPU backend 運用。
- 録音品質は端末/エミュレータ設定に影響される（Host Audio 設定等）。

## 4.6 WASM demo (`examples/wasm`)

現状機能:

- 単一WASMモジュールでローカル推論（server不要）。
- 外部APIモードあり。
- 録音/音声ファイルから参照作成あり。
- 追加参照は `localStorage` 永続化。
- backend選択: Auto / CPU / WebGPU（未対応時フォールバック）。

ビルド/配布:

- `examples/wasm/build.sh` で `../../models` から `public/model`, `public/embeddings` に同期。
- `index.json` 自動生成。
- `WASM_WEBGPU=auto|on|off`, `WASM_ASSERTIONS` をサポート。

---

## 5. モデル/変換運用メモ

- 変換は `uv` 前提:
  - `scripts/convert_miocodec_to_gguf.py`
  - `scripts/convert_wavlm_base_plus_to_gguf.py`
  - `scripts/convert_preset_embedding_to_gguf.py`
- LLM変換は `llama.cpp/convert_hf_to_gguf.py` を使用。
- 44.1kHz 系の MioCodec を `miocodec.gguf` 名で運用する方針。

---

## 6. メモリリーク/メモリ増加の観点（重要）

### 6.1 これまで問題化した事象

- iOS/Android で `参照作成 -> 音声生成` を繰り返すと RSS が増加し続ける報告。
- iOS 実機で OOM kill が発生した時期あり。
- Server/実機で Metal の `UPSCALE` 未対応による abort を確認。
- WASM で条件によって Abort/無音化が発生（2GB制限/非finiteなどを疑うケースあり）。

### 6.2 入れてある対策

- 参照作成時:
  - LLM runtime を先に unload
  - 必要時に synthesis context を一時解放して WavLM付き context を作成
  - 処理後に synthesis context を復元
- mobile共有ロジックで allocator pressure relief 呼び出し
- モバイル bridge 側デフォルトを `llm_unload_after_generation=false` にし、
  生成ごとのロード/アンロード揺れを減らす方向に調整

### 6.3 修正済み: モデル重みの ggml_context リーク（致命的）

**根本原因**: `miocodec_decoder` と `wavlm_extractor` のデストラクタおよび `load()` 関数で、
`gguf_init_from_file()` が作成する `ggml_context`（モデル重み全体を保持）に対して
`ggml_free()` が呼ばれていなかった。`gguf_free()` は GGUF メタデータのみ解放し、
`ggml_context` は解放しない。

- コンテキスト破棄のたびに MioCodec 重み（数十〜数百MB）がリーク
- 参照作成時は WavLM 重み（〜100MB）も追加でリーク
- 「参照作成→音声生成」1サイクルあたり数百MBがリークし、繰り返しで OOM kill に直結

**修正箇所**:
- `src/miocodec-decoder.cpp`: デストラクタと `load()` で `ggml_free(ctx_weights_)` を追加
- `src/wavlm-extractor.cpp`: デストラクタと `load()` で `ggml_free(ctx_weights_)` を追加
- `src/mio-tts-mobile-shared.hpp`: 全エラーパスと主要操作後に `release_memory_pressure()` を追加
- `src/miocodec-decoder.h`, `src/wavlm-extractor.h`: 未使用の workspace ベクトル（旧実装の残骸）を削除

### 6.4 修正済み: コンテキスト破棄/再作成によるヒープ断片化（深刻）

**根本原因**: iOS/Android ブリッジで `mio_has_wavlm = false` で初期化していたため、
参照作成のたびに以下のサイクルが発生していた：

1. decoder-only コンテキスト破棄（〜200MB free）
2. LLM モデル破棄（〜500MB free）
3. decoder+wavlm コンテキスト作成（〜300MB malloc + ディスク読み込み）
4. gallocr 計算バッファ確保/解放（〜50-100MB）
5. decoder+wavlm コンテキスト破棄（〜300MB free）
6. decoder-only コンテキスト再作成（〜200MB malloc + ディスク読み込み）
7. LLM モデル再読み込み（〜500MB malloc + ディスク読み込み）

1サイクルあたり〜1.5GB 以上の alloc/free チャーンが発生。
モバイルの malloc 実装は大きなブロックの合体が不完全なため、
ヒープ断片化により OS にページが返却されず RSS が単調増加していた。

**修正**: 初期化時に WavLM を含めてコンテキストを作成（`mio_has_wavlm = true`）。
`create_reference_from_audio` は `mio_has_wavlm` が true の場合、既存コンテキストを
直接使用しコンテキストの破棄/再作成をスキップする。

- ベースライン RSS は〜100MB 増加（WavLM 重み常駐）
- alloc/free チャーンは LLM のみ（〜500MB、mmap 経由なのでページキャッシュで軽減）+ gallocr（〜70MB 一時的）
- 従来比で alloc/free チャーンが 1/3 以下に削減

**修正箇所**:
- `examples/swiftui/.../MioTTSLocalBridge.mm`: `mio_tts_init_from_file` に wavlm パスを渡し `mio_has_wavlm = true` に変更
- `examples/android/.../mio_tts_android_jni.cpp`: 同上

### 6.5 まだ残る可能性がある論点

- LLM の load/free サイクル（参照作成時）による残存チャーン。llama.cpp は mmap を使うため影響は軽微。
- gallocr バッファの反復確保/解放。各〜50-100MB で比較的小さいが、完全解消にはキャッシュ化が必要。
- allocator が OS に即時返却しないことによる RSS 高止まり（リークと見分けづらい）。
- アプリ層（音声ファイル履歴・player/recorder lifecycle）による間接的メモリ保持。

---

## 7. 現在の未解決/要観測課題

1. モバイルでの繰り返し操作時メモリ増加の収束確認（ggml_context リーク修正済み + コンテキスト再利用修正済み）
2. Server の Mio backend を GPUにした場合の Metal `UPSCALE` 問題回避策  
3. デフォルト値の整合（CLI/Server/iOS/Android/WASM で `n_predict` など差異あり）  
4. `mio-tts-lib` API レベルでの「明示的ワークスペース解放」導入要否  
5. Raspberry Pi / M5Stack 向けの最小構成プロファイル（量子化・軽量モデル・スレッド戦略）設計

---

## 8. 今後の推奨タスク順

1. メモリ再現テストを固定シナリオ化（iOS/Android）  
2. `mio-tts-lib` に optional な `shrink/release` API 追加検討  
3. backend matrix テスト（CPU/Metal/WebGPU/外部API）を最小自動化  
4. デフォルトパラメータの統一方針を決め、READMEとUIに反映  
5. Edgeデバイス向けプリセット（低メモリ構成）を追加

---

## 9. 引き継ぎ時の注意

- 現在ワークツリーは変更中ファイルが多い。マージ前に差分の意図を確認すること。
- `build.sh` / `clean.sh` / `models_download.sh` は運用導線の中心なので、互換性を崩さない。
- モバイルは「機能追加より先にメモリ安定性」を優先する。
- Server の GPU backend はモデル演算対応状況（特に Metal）を必ず確認してから有効化する。

