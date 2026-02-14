# wasm browser demo (single module)

`mio-tts-server` を使わず、ブラウザ内だけで処理するサンプルです。

## 構成

- `index.html`
- `miottscpp.js` (UI + ローダ)
- `miottscpp_core.js` / `miottscpp_core.wasm` (build 生成物)
- `public/model/*.gguf`
- `public/embeddings/*.emb.gguf`

## できること

- ローカルLLM: `text -> codes -> wav`
- 外部API: OpenAI互換エンドポイントから codes を取得して `wav` 生成
- 参照音声作成: 録音または音声ファイルから embedding 作成
- 作成した参照(`added_speaker_N`)を `localStorage` に永続化
- backend切替: `Auto / CPU / WebGPU`（WebGPU未対応環境ではCPUへフォールバック）

## 必要ファイル

- `public/model/miocodec.gguf`
- `public/model/MioTTS-0.1B-Q8_0.gguf` (ローカルLLM使用時)
- `public/model/wavlm_base_plus_2l_f32.gguf` (参照作成時)
- `public/embeddings/*.emb.gguf`

## build

Emscripten (`emcc`) が必要です。

```bash
cd examples/wasm
./build.sh
```

`build.sh` は不足時に `../../models/` から自動コピーします。

- `miocodec.gguf`
- `MioTTS-0.1B-Q8_0.gguf`
- `wavlm_base_plus_2l_f32.gguf`
- `*.emb.gguf`

`public/embeddings/index.json` も自動生成します。

### WebGPUビルド

- 既定: `WASM_WEBGPU=auto`（利用可能なら有効化、不可ならCPU-only）
- 強制有効: `WASM_WEBGPU=on ./build.sh`
- 明示無効: `WASM_WEBGPU=off ./build.sh`

## 実行

```bash
cd examples/wasm
python3 -m http.server 8787
```

Open:

- `http://127.0.0.1:8787/`

## 外部API利用時の注意

- 既定 Base URL: `http://localhost:8080/v1`
- 既定 api-key: `dummy`
- ブラウザから呼ぶため、API側でCORS許可が必要です
