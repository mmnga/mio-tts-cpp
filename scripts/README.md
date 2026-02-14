# scripts

`scripts/` には、MioTTS関連モデルをGGUFへ変換する3つのPythonスクリプトがあります。

## 前提

```bash
cd mio-tts-cpp
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

- 必要パッケージ: `gguf`, `numpy`, `torch`, `safetensors`, `PyYAML`
- `gguf` が通常の import で見つからない場合は `GGUF_PY_PATH` で指定できます。

## 1) convert_preset_embedding_to_gguf.py

用途:
- 話者埋め込み（`.pt` / `.npz`）を `*.emb.gguf` に変換
- `--tts-mio-embedding-in` / `embedding_in` で使う入力を作成

基本:

```bash
.venv/bin/python scripts/convert_preset_embedding_to_gguf.py <embedding.(pt|npz)> -o <output.emb.gguf>
```

例:

```bash
.venv/bin/python scripts/convert_preset_embedding_to_gguf.py \
  .tmp/MioTTS-Inference/presets/jp_male.pt \
  -o models/jp_male.emb.gguf
```

`presets` 内の `.pt` をまとめて変換:

```bash
for f in .tmp/MioTTS-Inference/presets/*.pt; do
  b="$(basename "$f" .pt)"
  .venv/bin/python scripts/convert_preset_embedding_to_gguf.py "$f" -o "models/${b}.emb.gguf"
done
```

## 2) convert_wavlm_base_plus_to_gguf.py

用途:
- `wavlm_base_plus.pth`（torchaudio WavLM Base+）をGGUFへ変換
- 参照音声から話者埋め込みを抽出するためのモデルを生成

基本:

```bash
.venv/bin/python scripts/convert_wavlm_base_plus_to_gguf.py \
  --wavlm-weights <wavlm_base_plus.pth> \
  -o <wavlm_base_plus_2l_f32.gguf>
```

主なオプション:
- `--num-transformer-layers` (default: `2`)
- `--sample-rate` (default: `16000`)

例:

```bash
.venv/bin/python scripts/convert_wavlm_base_plus_to_gguf.py \
  --wavlm-weights /path/to/wavlm_base_plus.pth \
  --num-transformer-layers 2 \
  --sample-rate 16000 \
  -o models/wavlm_base_plus_2l_f32.gguf
```

## 3) convert_miocodec_to_gguf.py

用途:
- MioCodec (`config.yaml` + `model.safetensors`) をGGUFへ変換
- デフォルトは `dynamic-global embedding` モードで出力

### Dynamic（デフォルト）

```bash
.venv/bin/python scripts/convert_miocodec_to_gguf.py \
  <codec_dir> \
  -o <miocodec-dynamic.gguf>
```

または明示指定:

```bash
.venv/bin/python scripts/convert_miocodec_to_gguf.py \
  --codec-config <config.yaml> \
  --codec-weights <model.safetensors> \
  -o <miocodec-dynamic.gguf>
```

### Static preset（旧互換）

```bash
.venv/bin/python scripts/convert_miocodec_to_gguf.py \
  <codec_dir> \
  --static-preset-mode \
  --preset-embedding <preset.(pt|npz)> \
  -o <miocodec-static.gguf>
```

主なオプション:
- `--samples-per-token` (default: `960`)
- 44.1kHz / 25Hz モデルの場合は `--samples-per-token 1764`
- `--vocoder-upsample-rates` (default: `8,8,2,2,2`)

44.1kHz v2 例:

```bash
.venv/bin/python scripts/convert_miocodec_to_gguf.py \
  .tmp/MioCodec-25Hz-44.1kHz-v2 \
  --samples-per-token 1764 \
  -o models/miocodec.gguf
```
