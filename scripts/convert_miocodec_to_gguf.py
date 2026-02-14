#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert MioCodec (safetensors + config) to GGUF for native llama-tts decoding. "
            "Default mode exports dynamic-global speaker conditioning. "
            "Use --static-preset-mode only for legacy fixed-speaker exports."
        )
    )
    parser.add_argument(
        "codec_dir",
        nargs="?",
        default="",
        help=(
            "optional MioCodec directory containing config.yaml and model.safetensors. "
            "default output mode is dynamic-global embedding."
        ),
    )
    parser.add_argument("--codec-config", default="", help="path to MioCodec config.yaml")
    parser.add_argument("--codec-weights", default="", help="path to MioCodec model.safetensors")
    parser.add_argument(
        "--preset-embedding",
        default="",
        help=(
            "global embedding preset (.pt or .npz), e.g. MioTTS-Inference presets. "
            "Required only with --static-preset-mode. "
            "For dynamic mode, convert preset with convert_preset_embedding_to_gguf.py and pass at runtime."
        ),
    )
    parser.add_argument(
        "--dynamic-global-embedding",
        action="store_true",
        help=(
            "export runtime-conditioning tensors (decoder AdaLN + global encoder). "
            "This is the default mode."
        ),
    )
    parser.add_argument(
        "--static-preset-mode",
        action="store_true",
        help="legacy fixed-speaker export mode (requires --preset-embedding)",
    )
    parser.add_argument(
        "--samples-per-token",
        type=int,
        default=960,
        help="target samples per content token (MioCodec-25Hz-24kHz uses 960)",
    )
    parser.add_argument(
        "--vocoder-upsample-rates",
        default="8,8,2,2,2",
        help="comma-separated MioVocoder upsample rates (default: 8,8,2,2,2)",
    )
    parser.add_argument("-o", "--outfile", required=True, help="output GGUF path")
    return parser.parse_args()


def import_gguf_module():
    try:
        import gguf  # type: ignore
        return gguf
    except Exception:
        gguf_py_path = os.environ.get("GGUF_PY_PATH", "")
        if gguf_py_path:
            sys.path.insert(0, gguf_py_path)
            import gguf  # type: ignore
            return gguf
        raise RuntimeError(
            "gguf module not found. Install with `pip install gguf` "
            "or set GGUF_PY_PATH to a directory containing the module."
        )


def resolve_codec_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    codec_config = Path(args.codec_config).expanduser() if args.codec_config else None
    codec_weights = Path(args.codec_weights).expanduser() if args.codec_weights else None

    if args.codec_dir:
        codec_dir = Path(args.codec_dir).expanduser()
        if not codec_dir.is_dir():
            raise ValueError(f"codec dir not found: {codec_dir}")

        if codec_config is None:
            codec_config = codec_dir / "config.yaml"
        if codec_weights is None:
            codec_weights = codec_dir / "model.safetensors"

    if codec_config is None or codec_weights is None:
        raise ValueError("set --codec-config and --codec-weights, or pass CODEC_DIR as positional argument")

    if not codec_config.is_file():
        raise ValueError(f"codec config not found: {codec_config}")
    if not codec_weights.is_file():
        raise ValueError(f"codec weights not found: {codec_weights}")

    return codec_config, codec_weights


def load_global_embedding(path: Path) -> torch.Tensor:
    suffix = path.suffix.lower()
    if suffix == ".pt":
        obj = torch.load(path, map_location="cpu", weights_only=True)
    elif suffix == ".npz":
        arr = np.load(path)
        if "global_embedding" in arr:
            obj = arr["global_embedding"]
        elif "embedding" in arr:
            obj = arr["embedding"]
        elif len(arr.files) > 0:
            obj = arr[arr.files[0]]
        else:
            raise ValueError(f"no arrays found in {path}")
    else:
        raise ValueError(f"unsupported embedding format: {path}")

    if isinstance(obj, dict):
        if "global_embedding" in obj:
            obj = obj["global_embedding"]
        elif "embedding" in obj:
            obj = obj["embedding"]

    if not isinstance(obj, torch.Tensor):
        obj = torch.tensor(obj)

    emb = obj.squeeze().float()
    if emb.dim() != 1:
        raise ValueError(f"global embedding must be 1D after squeeze, got shape {tuple(emb.shape)}")
    return emb


def decode_fsq_indices(indices: np.ndarray, levels: list[int]) -> np.ndarray:
    levels_arr = np.array(levels, dtype=np.int64)
    basis = np.cumprod(np.array([1] + levels[:-1], dtype=np.int64))
    codes = (indices[:, None] // basis[None, :]) % levels_arr[None, :]
    half = (levels_arr // 2).astype(np.float32)
    return (codes.astype(np.float32) - half[None, :]) / half[None, :]


def adaln_params(cond_proj_w: torch.Tensor, cond_proj_b: torch.Tensor, global_embedding: torch.Tensor) -> torch.Tensor:
    x = F.silu(global_embedding.float())
    return cond_proj_w.float().matmul(x) + cond_proj_b.float()


def get_cfg(cfg: dict, path: list[str]):
    cur = cfg
    for p in path:
        cur = cur[p]
    return cur


def to_np_f32(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().float().numpy()


def to_np_i32(v: list[int]) -> np.ndarray:
    return np.array(v, dtype=np.int32)


def parse_int_list(csv: str) -> list[int]:
    out: list[int] = []
    for item in csv.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    if not out:
        raise ValueError("empty integer list")
    return out


def weight_norm_to_weight(weight_g: torch.Tensor, weight_v: torch.Tensor) -> torch.Tensor:
    # torch.nn.utils.weight_norm for Conv1d with dim=0
    v = weight_v.float()
    g = weight_g.float()
    v_norm = torch.linalg.vector_norm(v.reshape(v.shape[0], -1), dim=1, keepdim=True).clamp_min(1e-12)
    v_norm = v_norm.unsqueeze(-1)
    return v * (g / v_norm)


def main() -> int:
    args = parse_args()
    # Dynamic-global is the common/default path.
    args.dynamic_global_embedding = not args.static_preset_mode
    codec_config, codec_weights = resolve_codec_paths(args)

    if not args.dynamic_global_embedding and not args.preset_embedding:
        raise ValueError("--preset-embedding is required with --static-preset-mode")
    if args.dynamic_global_embedding and args.preset_embedding:
        print(
            "warning: --preset-embedding is ignored in dynamic mode. "
            "Use convert_preset_embedding_to_gguf.py and pass embedding at runtime.",
            file=sys.stderr,
        )

    gguf = import_gguf_module()

    cfg = yaml.safe_load(codec_config.read_text(encoding="utf-8"))
    init_args = get_cfg(cfg, ["model", "init_args"])
    model_cfg = init_args["config"]

    use_wave_decoder = bool(model_cfg.get("use_wave_decoder", False))
    src_prenet = "wave_prenet" if use_wave_decoder else "mel_prenet"
    src_decoder = "wave_decoder" if use_wave_decoder else "mel_decoder"

    wave_upsampler_factors: list[int] = []
    wave_upsampler_kernel_sizes: list[int] = []
    if use_wave_decoder:
        wave_upsampler_factors = [int(x) for x in (model_cfg.get("wave_upsampler_factors") or [])]
        if wave_upsampler_factors:
            raw_kernel_sizes = model_cfg.get("wave_upsampler_kernel_sizes")
            if raw_kernel_sizes:
                wave_upsampler_kernel_sizes = [int(x) for x in raw_kernel_sizes]
            else:
                wave_upsampler_kernel_sizes = [int(f) * 2 for f in wave_upsampler_factors]
            if len(wave_upsampler_kernel_sizes) != len(wave_upsampler_factors):
                raise ValueError(
                    "wave_upsampler_kernel_sizes and wave_upsampler_factors must have the same length"
                )

    if src_prenet not in init_args or src_decoder not in init_args:
        raise ValueError(f"missing {src_prenet}/{src_decoder} in config")

    prenet_cfg = init_args[src_prenet]["init_args"]
    decoder_cfg = init_args[src_decoder]["init_args"]
    quantizer_cfg = init_args["local_quantizer"]["init_args"]
    global_encoder_cfg = init_args["global_encoder"]["init_args"]

    state = load_file(str(codec_weights), device="cpu")
    global_embedding = load_global_embedding(Path(args.preset_embedding)) if args.preset_embedding else None

    levels = [int(x) for x in quantizer_cfg["levels"]]
    vocab_size = int(np.prod(levels))
    if vocab_size != 12800:
        raise ValueError(f"unexpected vocab size from levels {levels}: {vocab_size} (expected 12800)")

    # Build static token embedding table from FSQ decode + proj_out.
    indices = np.arange(vocab_size, dtype=np.int64)
    fsq_codes = decode_fsq_indices(indices, levels)
    proj_out_w = state["local_quantizer.proj_out.weight"].float().cpu().numpy()  # [768, 5]
    proj_out_b = state["local_quantizer.proj_out.bias"].float().cpu().numpy()  # [768]
    token_embd = fsq_codes @ proj_out_w.T + proj_out_b[None, :]  # [12800, 768]

    n_decoder_layers = int(decoder_cfg["n_layers"])
    decoder_dim = int(decoder_cfg["dim"])
    decoder_adanorm_dim = int(decoder_cfg["adanorm_condition_dim"])

    folded_state: dict[str, torch.Tensor] = dict(state)
    decoder_attn_norm_w: list[torch.Tensor] = []
    decoder_attn_norm_b: list[torch.Tensor] = []
    decoder_ffn_norm_w: list[torch.Tensor] = []
    decoder_ffn_norm_b: list[torch.Tensor] = []
    decoder_final_norm_w: torch.Tensor | None = None
    decoder_final_norm_b: torch.Tensor | None = None

    if not args.dynamic_global_embedding:
        assert global_embedding is not None
        # Fold AdaLN conditioning into fixed affine norms and gated projections.
        for i in range(n_decoder_layers):
            a_w = state[f"{src_decoder}.layers.{i}.attention_norm.condition_proj.1.weight"]
            a_b = state[f"{src_decoder}.layers.{i}.attention_norm.condition_proj.1.bias"]
            f_w = state[f"{src_decoder}.layers.{i}.ffn_norm.condition_proj.1.weight"]
            f_b = state[f"{src_decoder}.layers.{i}.ffn_norm.condition_proj.1.bias"]

            a_params = adaln_params(a_w, a_b, global_embedding)
            f_params = adaln_params(f_w, f_b, global_embedding)

            a_shift, a_scale, a_gate = torch.chunk(a_params, 3, dim=0)
            f_shift, f_scale, f_gate = torch.chunk(f_params, 3, dim=0)

            decoder_attn_norm_w.append((1.0 + a_scale).float())
            decoder_attn_norm_b.append(a_shift.float())
            decoder_ffn_norm_w.append((1.0 + f_scale).float())
            decoder_ffn_norm_b.append(f_shift.float())

            # Gate folding: gate * wo(x) and gate * w2(x)
            wo_name = f"{src_decoder}.layers.{i}.attention.wo.weight"
            w2_name = f"{src_decoder}.layers.{i}.feed_forward.w2.weight"
            folded_state[wo_name] = state[wo_name].float() * a_gate.float().unsqueeze(1)
            folded_state[w2_name] = state[w2_name].float() * f_gate.float().unsqueeze(1)

        n_w = state[f"{src_decoder}.norm.condition_proj.1.weight"]
        n_b = state[f"{src_decoder}.norm.condition_proj.1.bias"]
        n_params = adaln_params(n_w, n_b, global_embedding)
        n_shift, n_scale = torch.chunk(n_params, 2, dim=0)
        decoder_final_norm_w = (1.0 + n_scale).float()
        decoder_final_norm_b = n_shift.float()

    prenet_layers = int(prenet_cfg["n_layers"])
    prenet_dim = int(prenet_cfg["dim"])
    prenet_heads = int(prenet_cfg["n_heads"])
    prenet_ff = int(state[f"{src_prenet}.layers.0.feed_forward.w1.weight"].shape[0])
    prenet_window = int(prenet_cfg["window_size"])

    decoder_heads = int(decoder_cfg["n_heads"])
    decoder_ff = int(state[f"{src_decoder}.layers.0.feed_forward.w1.weight"].shape[0])
    decoder_window = int(decoder_cfg["window_size"])

    n_fft = int(model_cfg["n_fft"])
    hop_length = int(model_cfg["hop_length"])
    sample_rate = int(model_cfg["sample_rate"])
    n_mels = int(model_cfg.get("n_mels", 0))
    rope_theta = float(decoder_cfg.get("rope_theta", 10000.0))
    norm_eps = float(decoder_cfg.get("norm_eps", 1e-5))
    group_norm_eps = 1e-6

    resnet_blocks = int(model_cfg.get("wave_resnet_num_blocks", 0)) if use_wave_decoder else 0
    resnet_groups = int(model_cfg.get("wave_resnet_num_groups", 1)) if use_wave_decoder else 1
    model_type = 0 if use_wave_decoder else 1
    output_bins = (n_fft + 2) if use_wave_decoder else n_mels
    has_wave_upsampler = bool(use_wave_decoder and wave_upsampler_factors and any(
        k.startswith("wave_upsampler.") for k in state.keys()
    ))

    out = Path(args.outfile)
    out.parent.mkdir(parents=True, exist_ok=True)

    writer = gguf.GGUFWriter(
        path=str(out),
        arch="miocodec-dec",
        endianess=gguf.GGUFEndian.LITTLE,
        use_temp_file=False,
    )
    writer.add_type(gguf.GGUFType.MODEL)
    writer.add_name("MioCodec decoder (llama-tts native)")
    writer.add_vocab_size(vocab_size)
    writer.add_embedding_length(prenet_dim)
    if hasattr(writer, "add_embedding_length_out"):
        writer.add_embedding_length_out(output_bins)
    else:
        # Backward compatibility for older gguf package versions.
        writer.add_uint32("embedding_length_out", output_bins)
    writer.add_context_length(4096)

    writer.add_uint32("miocodec.model_type", model_type)  # 0=wave, 1=mel
    writer.add_uint32("miocodec.dynamic_global", 1 if args.dynamic_global_embedding else 0)
    writer.add_uint32("miocodec.sample_rate", sample_rate)
    writer.add_uint32("miocodec.n_fft", n_fft)
    writer.add_uint32("miocodec.hop_length", hop_length)
    writer.add_uint32("miocodec.n_mels", n_mels)
    writer.add_uint32("miocodec.samples_per_token", int(args.samples_per_token))
    writer.add_uint32("miocodec.prenet_layers", prenet_layers)
    writer.add_uint32("miocodec.prenet_dim", prenet_dim)
    writer.add_uint32("miocodec.prenet_heads", prenet_heads)
    writer.add_uint32("miocodec.prenet_ff", prenet_ff)
    writer.add_uint32("miocodec.prenet_window", prenet_window)
    writer.add_uint32("miocodec.decoder_layers", n_decoder_layers)
    writer.add_uint32("miocodec.decoder_dim", decoder_dim)
    writer.add_uint32("miocodec.decoder_heads", decoder_heads)
    writer.add_uint32("miocodec.decoder_ff", decoder_ff)
    writer.add_uint32("miocodec.decoder_window", decoder_window)
    writer.add_uint32("miocodec.decoder_adanorm_dim", decoder_adanorm_dim)
    writer.add_uint32("miocodec.resnet_blocks", resnet_blocks)
    writer.add_uint32("miocodec.resnet_groups", resnet_groups)
    writer.add_uint32("miocodec.wave_upsampler_layers", len(wave_upsampler_factors) if has_wave_upsampler else 0)
    writer.add_float32("miocodec.rope_theta", rope_theta)
    writer.add_float32("miocodec.norm_eps", norm_eps)
    writer.add_float32("miocodec.group_norm_eps", group_norm_eps)

    writer.add_uint32("miocodec.global_encoder.input_channels", int(global_encoder_cfg["input_channels"]))
    writer.add_uint32("miocodec.global_encoder.output_channels", int(global_encoder_cfg["output_channels"]))
    writer.add_uint32("miocodec.global_encoder.dim", int(global_encoder_cfg["dim"]))
    writer.add_uint32("miocodec.global_encoder.intermediate_dim", int(global_encoder_cfg["intermediate_dim"]))
    writer.add_uint32("miocodec.global_encoder.num_layers", int(global_encoder_cfg["num_layers"]))
    if has_wave_upsampler:
        writer.add_tensor("miocodec.wave_upsampler.factors", to_np_i32(wave_upsampler_factors))
        writer.add_tensor("miocodec.wave_upsampler.kernel_sizes", to_np_i32(wave_upsampler_kernel_sizes))

    writer.add_tensor("token_embd", token_embd.astype(np.float32, copy=False))

    def add_from_state(dst: str, src: str) -> None:
        if src not in folded_state:
            raise KeyError(f"missing tensor in state: {src}")
        writer.add_tensor(dst, to_np_f32(folded_state[src]))

    for i in range(prenet_layers):
        add_from_state(f"wave_prenet.blk.{i}.attn_norm.weight", f"{src_prenet}.layers.{i}.attention_norm.weight")
        add_from_state(f"wave_prenet.blk.{i}.attn_norm.bias", f"{src_prenet}.layers.{i}.attention_norm.bias")
        add_from_state(f"wave_prenet.blk.{i}.attn_q.weight", f"{src_prenet}.layers.{i}.attention.wq.weight")
        add_from_state(f"wave_prenet.blk.{i}.attn_k.weight", f"{src_prenet}.layers.{i}.attention.wk.weight")
        add_from_state(f"wave_prenet.blk.{i}.attn_v.weight", f"{src_prenet}.layers.{i}.attention.wv.weight")
        add_from_state(f"wave_prenet.blk.{i}.attn_output.weight", f"{src_prenet}.layers.{i}.attention.wo.weight")
        add_from_state(f"wave_prenet.blk.{i}.ffn_norm.weight", f"{src_prenet}.layers.{i}.ffn_norm.weight")
        add_from_state(f"wave_prenet.blk.{i}.ffn_norm.bias", f"{src_prenet}.layers.{i}.ffn_norm.bias")
        add_from_state(f"wave_prenet.blk.{i}.ffn_gate.weight", f"{src_prenet}.layers.{i}.feed_forward.w1.weight")
        add_from_state(f"wave_prenet.blk.{i}.ffn_down.weight", f"{src_prenet}.layers.{i}.feed_forward.w2.weight")
        add_from_state(f"wave_prenet.blk.{i}.ffn_up.weight", f"{src_prenet}.layers.{i}.feed_forward.w3.weight")

    add_from_state("wave_prenet.norm.weight", f"{src_prenet}.norm.weight")
    add_from_state("wave_prenet.norm.bias", f"{src_prenet}.norm.bias")
    add_from_state("wave_prenet.output.weight", f"{src_prenet}.output_proj.weight")
    add_from_state("wave_prenet.output.bias", f"{src_prenet}.output_proj.bias")

    upsample_key = "wave_conv_upsample" if use_wave_decoder else "mel_conv_upsample"
    add_from_state("wave_upsample.weight", f"{upsample_key}.weight")
    add_from_state("wave_upsample.bias", f"{upsample_key}.bias")

    if use_wave_decoder:
        for i in range(resnet_blocks):
            add_from_state(f"wave_prior.{i}.norm1.weight", f"wave_prior_net.blocks.{i}.norm1.weight")
            add_from_state(f"wave_prior.{i}.norm1.bias", f"wave_prior_net.blocks.{i}.norm1.bias")
            add_from_state(f"wave_prior.{i}.conv1.weight", f"wave_prior_net.blocks.{i}.conv1.weight")
            add_from_state(f"wave_prior.{i}.conv1.bias", f"wave_prior_net.blocks.{i}.conv1.bias")
            add_from_state(f"wave_prior.{i}.norm2.weight", f"wave_prior_net.blocks.{i}.norm2.weight")
            add_from_state(f"wave_prior.{i}.norm2.bias", f"wave_prior_net.blocks.{i}.norm2.bias")
            add_from_state(f"wave_prior.{i}.conv2.weight", f"wave_prior_net.blocks.{i}.conv2.weight")
            add_from_state(f"wave_prior.{i}.conv2.bias", f"wave_prior_net.blocks.{i}.conv2.bias")

        if has_wave_upsampler:
            for i in range(len(wave_upsampler_factors)):
                up_prefix = f"wave_upsampler.upsample_layers.{i}"
                up_w = weight_norm_to_weight(
                    state[f"{up_prefix}.parametrizations.weight.original0"],
                    state[f"{up_prefix}.parametrizations.weight.original1"],
                )
                writer.add_tensor(f"wave_upsampler.up.{i}.weight", to_np_f32(up_w))
                add_from_state(f"wave_upsampler.up.{i}.bias", f"{up_prefix}.bias")

                add_from_state(f"wave_upsampler.snake.{i}.alpha", f"wave_upsampler.snake_activations.{i}.alpha")
                add_from_state(f"wave_upsampler.snake.{i}.beta", f"wave_upsampler.snake_activations.{i}.beta")

                add_from_state(
                    f"wave_upsampler.resblk.{i}.norm1.weight",
                    f"wave_upsampler.resnet_blocks.{i}.norm1.weight",
                )
                add_from_state(
                    f"wave_upsampler.resblk.{i}.norm1.bias",
                    f"wave_upsampler.resnet_blocks.{i}.norm1.bias",
                )
                add_from_state(
                    f"wave_upsampler.resblk.{i}.conv1.weight",
                    f"wave_upsampler.resnet_blocks.{i}.conv1.weight",
                )
                add_from_state(
                    f"wave_upsampler.resblk.{i}.conv1.bias",
                    f"wave_upsampler.resnet_blocks.{i}.conv1.bias",
                )
                add_from_state(
                    f"wave_upsampler.resblk.{i}.norm2.weight",
                    f"wave_upsampler.resnet_blocks.{i}.norm2.weight",
                )
                add_from_state(
                    f"wave_upsampler.resblk.{i}.norm2.bias",
                    f"wave_upsampler.resnet_blocks.{i}.norm2.bias",
                )
                add_from_state(
                    f"wave_upsampler.resblk.{i}.conv2.weight",
                    f"wave_upsampler.resnet_blocks.{i}.conv2.weight",
                )
                add_from_state(
                    f"wave_upsampler.resblk.{i}.conv2.bias",
                    f"wave_upsampler.resnet_blocks.{i}.conv2.bias",
                )

            add_from_state("wave_upsampler.out_proj.weight", "wave_upsampler.out_proj.weight")
            add_from_state("wave_upsampler.out_proj.bias", "wave_upsampler.out_proj.bias")
            add_from_state("wave_upsampler.out_snake.alpha", "wave_upsampler.out_snake.alpha")
            add_from_state("wave_upsampler.out_snake.beta", "wave_upsampler.out_snake.beta")

    for i in range(n_decoder_layers):
        if args.dynamic_global_embedding:
            add_from_state(
                f"wave_decoder.blk.{i}.attn_cond.weight",
                f"{src_decoder}.layers.{i}.attention_norm.condition_proj.1.weight",
            )
            add_from_state(
                f"wave_decoder.blk.{i}.attn_cond.bias",
                f"{src_decoder}.layers.{i}.attention_norm.condition_proj.1.bias",
            )
            add_from_state(
                f"wave_decoder.blk.{i}.ffn_cond.weight",
                f"{src_decoder}.layers.{i}.ffn_norm.condition_proj.1.weight",
            )
            add_from_state(
                f"wave_decoder.blk.{i}.ffn_cond.bias",
                f"{src_decoder}.layers.{i}.ffn_norm.condition_proj.1.bias",
            )
        else:
            writer.add_tensor(f"wave_decoder.blk.{i}.attn_norm.weight", to_np_f32(decoder_attn_norm_w[i]))
            writer.add_tensor(f"wave_decoder.blk.{i}.attn_norm.bias", to_np_f32(decoder_attn_norm_b[i]))
            writer.add_tensor(f"wave_decoder.blk.{i}.ffn_norm.weight", to_np_f32(decoder_ffn_norm_w[i]))
            writer.add_tensor(f"wave_decoder.blk.{i}.ffn_norm.bias", to_np_f32(decoder_ffn_norm_b[i]))

        add_from_state(f"wave_decoder.blk.{i}.attn_q.weight", f"{src_decoder}.layers.{i}.attention.wq.weight")
        add_from_state(f"wave_decoder.blk.{i}.attn_k.weight", f"{src_decoder}.layers.{i}.attention.wk.weight")
        add_from_state(f"wave_decoder.blk.{i}.attn_v.weight", f"{src_decoder}.layers.{i}.attention.wv.weight")
        add_from_state(f"wave_decoder.blk.{i}.attn_output.weight", f"{src_decoder}.layers.{i}.attention.wo.weight")
        add_from_state(f"wave_decoder.blk.{i}.ffn_gate.weight", f"{src_decoder}.layers.{i}.feed_forward.w1.weight")
        add_from_state(f"wave_decoder.blk.{i}.ffn_down.weight", f"{src_decoder}.layers.{i}.feed_forward.w2.weight")
        add_from_state(f"wave_decoder.blk.{i}.ffn_up.weight", f"{src_decoder}.layers.{i}.feed_forward.w3.weight")

    if args.dynamic_global_embedding:
        add_from_state("wave_decoder.norm_cond.weight", f"{src_decoder}.norm.condition_proj.1.weight")
        add_from_state("wave_decoder.norm_cond.bias", f"{src_decoder}.norm.condition_proj.1.bias")
    else:
        assert decoder_final_norm_w is not None and decoder_final_norm_b is not None
        writer.add_tensor("wave_decoder.norm.weight", to_np_f32(decoder_final_norm_w))
        writer.add_tensor("wave_decoder.norm.bias", to_np_f32(decoder_final_norm_b))

    if use_wave_decoder:
        for i in range(resnet_blocks):
            add_from_state(f"wave_post.{i}.norm1.weight", f"wave_post_net.blocks.{i}.norm1.weight")
            add_from_state(f"wave_post.{i}.norm1.bias", f"wave_post_net.blocks.{i}.norm1.bias")
            add_from_state(f"wave_post.{i}.conv1.weight", f"wave_post_net.blocks.{i}.conv1.weight")
            add_from_state(f"wave_post.{i}.conv1.bias", f"wave_post_net.blocks.{i}.conv1.bias")
            add_from_state(f"wave_post.{i}.norm2.weight", f"wave_post_net.blocks.{i}.norm2.weight")
            add_from_state(f"wave_post.{i}.norm2.bias", f"wave_post_net.blocks.{i}.norm2.bias")
            add_from_state(f"wave_post.{i}.conv2.weight", f"wave_post_net.blocks.{i}.conv2.weight")
            add_from_state(f"wave_post.{i}.conv2.bias", f"wave_post_net.blocks.{i}.conv2.bias")

        add_from_state("istft_head.out.weight", "istft_head.out.weight")
        add_from_state("istft_head.out.bias", "istft_head.out.bias")
    else:
        add_from_state("istft_head.out.weight", f"{src_decoder}.output_proj.weight")
        add_from_state("istft_head.out.bias", f"{src_decoder}.output_proj.bias")

        post_layers = sorted(
            {
                int(m.group(1))
                for key in state.keys()
                if (m := re.match(r"^mel_postnet\.convolutions\.(\d+)\.0\.weight$", key)) is not None
            }
        )
        writer.add_uint32("miocodec.mel_postnet_layers", len(post_layers))
        kernel_size = 0
        for i in post_layers:
            conv_w = f"mel_postnet.convolutions.{i}.0.weight"
            conv_b = f"mel_postnet.convolutions.{i}.0.bias"
            norm_w = f"mel_postnet.convolutions.{i}.1.norm.weight"
            norm_b = f"mel_postnet.convolutions.{i}.1.norm.bias"
            if kernel_size == 0:
                kernel_size = int(state[conv_w].shape[-1])
            add_from_state(f"mel_postnet.{i}.conv.weight", conv_w)
            add_from_state(f"mel_postnet.{i}.conv.bias", conv_b)
            add_from_state(f"mel_postnet.{i}.norm.weight", norm_w)
            add_from_state(f"mel_postnet.{i}.norm.bias", norm_b)
        writer.add_uint32("miocodec.mel_postnet_kernel_size", kernel_size)

    # Global encoder (needed for runtime reference-audio conditioning).
    add_from_state("global_encoder.backbone.embed.weight", "global_encoder.backbone.embed.weight")
    add_from_state("global_encoder.backbone.embed.bias", "global_encoder.backbone.embed.bias")
    add_from_state("global_encoder.backbone.norm.weight", "global_encoder.backbone.norm.weight")
    add_from_state("global_encoder.backbone.norm.bias", "global_encoder.backbone.norm.bias")
    add_from_state("global_encoder.backbone.final_norm.weight", "global_encoder.backbone.final_layer_norm.weight")
    add_from_state("global_encoder.backbone.final_norm.bias", "global_encoder.backbone.final_layer_norm.bias")

    n_g_layers = int(global_encoder_cfg["num_layers"])
    for i in range(n_g_layers):
        add_from_state(
            f"global_encoder.backbone.blk.{i}.dwconv.weight",
            f"global_encoder.backbone.convnext.{i}.dwconv.weight",
        )
        add_from_state(
            f"global_encoder.backbone.blk.{i}.dwconv.bias",
            f"global_encoder.backbone.convnext.{i}.dwconv.bias",
        )
        add_from_state(f"global_encoder.backbone.blk.{i}.norm.weight", f"global_encoder.backbone.convnext.{i}.norm.weight")
        add_from_state(f"global_encoder.backbone.blk.{i}.norm.bias", f"global_encoder.backbone.convnext.{i}.norm.bias")
        add_from_state(f"global_encoder.backbone.blk.{i}.pw1.weight", f"global_encoder.backbone.convnext.{i}.pwconv1.weight")
        add_from_state(f"global_encoder.backbone.blk.{i}.pw1.bias", f"global_encoder.backbone.convnext.{i}.pwconv1.bias")
        add_from_state(f"global_encoder.backbone.blk.{i}.pw2.weight", f"global_encoder.backbone.convnext.{i}.pwconv2.weight")
        add_from_state(f"global_encoder.backbone.blk.{i}.pw2.bias", f"global_encoder.backbone.convnext.{i}.pwconv2.bias")
        add_from_state(f"global_encoder.backbone.blk.{i}.gamma", f"global_encoder.backbone.convnext.{i}.gamma")

    add_from_state("global_encoder.pool.attn0.weight", "global_encoder.pooling.attn.0.weight")
    add_from_state("global_encoder.pool.attn0.bias", "global_encoder.pooling.attn.0.bias")
    add_from_state("global_encoder.pool.attn2.weight", "global_encoder.pooling.attn.2.weight")
    add_from_state("global_encoder.pool.attn2.bias", "global_encoder.pooling.attn.2.bias")
    add_from_state("global_encoder.pool.proj.weight", "global_encoder.pooling.proj.weight")
    add_from_state("global_encoder.pool.proj.bias", "global_encoder.pooling.proj.bias")
    add_from_state("global_encoder.pool.norm.weight", "global_encoder.pooling.norm.weight")
    add_from_state("global_encoder.pool.norm.bias", "global_encoder.pooling.norm.bias")

    # Optional bundled vocoder for mel-mode models.
    has_vocoder = any(k.startswith("vocoder.model.") for k in state.keys())
    writer.add_uint32("miocodec.has_vocoder", 1 if has_vocoder else 0)

    if has_vocoder:
        upsample_rates = parse_int_list(args.vocoder_upsample_rates)
        num_ups = len(upsample_rates)

        resblock_ids = sorted(
            {
                int(m.group(1))
                for key in state.keys()
                if (m := re.match(r"^vocoder\.model\.resblocks\.(\d+)\.convs1\.0\.weight_v$", key)) is not None
            }
        )
        if not resblock_ids:
            raise ValueError("vocoder tensors found but no resblocks detected")
        num_resblocks = max(resblock_ids) + 1
        if num_resblocks % num_ups != 0:
            raise ValueError(f"num_resblocks ({num_resblocks}) not divisible by num_ups ({num_ups})")
        num_kernels = num_resblocks // num_ups

        writer.add_uint32("miovocoder.sample_rate", sample_rate)
        writer.add_uint32("miovocoder.n_mels", n_mels)
        writer.add_uint32("miovocoder.num_upsamples", num_ups)
        writer.add_uint32("miovocoder.num_kernels", num_kernels)
        writer.add_tensor("miovocoder.upsample_rates", to_np_i32(upsample_rates))

        def add_weight_norm_conv(dst_prefix: str, src_prefix: str, has_bias: bool) -> None:
            w = weight_norm_to_weight(state[f"{src_prefix}.weight_g"], state[f"{src_prefix}.weight_v"])
            writer.add_tensor(f"{dst_prefix}.weight", to_np_f32(w))
            if has_bias:
                writer.add_tensor(f"{dst_prefix}.bias", to_np_f32(state[f"{src_prefix}.bias"]))

        add_weight_norm_conv("vocoder.conv_pre", "vocoder.model.conv_pre", True)
        add_weight_norm_conv("vocoder.conv_post", "vocoder.model.conv_post", False)

        for i in range(num_ups):
            add_weight_norm_conv(f"vocoder.ups.{i}.after", f"vocoder.model.ups.{i}.convolution_after", True)
            add_weight_norm_conv(f"vocoder.ups.{i}.noise", f"vocoder.model.ups.{i}.convolution_noise", True)

        for rid in range(num_resblocks):
            for c in range(3):
                add_weight_norm_conv(
                    f"vocoder.resblocks.{rid}.convs1.{c}",
                    f"vocoder.model.resblocks.{rid}.convs1.{c}",
                    True,
                )
                add_weight_norm_conv(
                    f"vocoder.resblocks.{rid}.convs2.{c}",
                    f"vocoder.model.resblocks.{rid}.convs2.{c}",
                    True,
                )
            for a in range(6):
                writer.add_tensor(
                    f"vocoder.resblocks.{rid}.acts.{a}.alpha",
                    to_np_f32(state[f"vocoder.model.resblocks.{rid}.activations.{a}.act.alpha"]),
                )
                writer.add_tensor(
                    f"vocoder.resblocks.{rid}.acts.{a}.beta",
                    to_np_f32(state[f"vocoder.model.resblocks.{rid}.activations.{a}.act.beta"]),
                )
                writer.add_tensor(
                    f"vocoder.resblocks.{rid}.acts.{a}.up_filter",
                    to_np_f32(state[f"vocoder.model.resblocks.{rid}.activations.{a}.upsample.filter"]),
                )
                writer.add_tensor(
                    f"vocoder.resblocks.{rid}.acts.{a}.down_filter",
                    to_np_f32(state[f"vocoder.model.resblocks.{rid}.activations.{a}.downsample.lowpass.filter"]),
                )

        writer.add_tensor("vocoder.activation_post.alpha", to_np_f32(state["vocoder.model.activation_post.act.alpha"]))
        writer.add_tensor("vocoder.activation_post.beta", to_np_f32(state["vocoder.model.activation_post.act.beta"]))
        writer.add_tensor(
            "vocoder.activation_post.up_filter",
            to_np_f32(state["vocoder.model.activation_post.upsample.filter"]),
        )
        writer.add_tensor(
            "vocoder.activation_post.down_filter",
            to_np_f32(state["vocoder.model.activation_post.downsample.lowpass.filter"]),
        )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    summary = {
        "outfile": str(out.resolve()),
        "model_type": "wave" if use_wave_decoder else "mel",
        "dynamic_global_embedding": bool(args.dynamic_global_embedding),
        "has_wave_upsampler": bool(has_wave_upsampler),
        "wave_upsampler_factors": wave_upsampler_factors if has_wave_upsampler else [],
        "has_vocoder": bool(has_vocoder),
        "vocab_size": vocab_size,
        "prenet_layers": prenet_layers,
        "decoder_layers": n_decoder_layers,
        "resnet_blocks": resnet_blocks,
        "global_encoder_layers": n_g_layers,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "sample_rate": sample_rate,
        "n_mels": n_mels,
    }
    print(json.dumps(summary, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
