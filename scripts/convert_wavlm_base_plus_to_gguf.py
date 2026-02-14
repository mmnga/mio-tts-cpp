#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert torchaudio WavLM Base+ checkpoint (wavlm_base_plus.pth) "
            "to GGUF for native MioTTS reference-audio speaker embedding extraction."
        )
    )
    parser.add_argument(
        "--wavlm-weights",
        required=True,
        help="path to wavlm_base_plus.pth",
    )
    parser.add_argument(
        "--num-transformer-layers",
        type=int,
        default=2,
        help=(
            "number of transformer layers to export from the encoder "
            "(default: 2; enough for MioCodec global_ssl_layers=[1,2])"
        ),
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="target sample rate expected by WavLM frontend (default: 16000)",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        required=True,
        help="output GGUF path",
    )
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


def to_np_f32(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().float().numpy()


def get_state_dict(path: Path) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        sd = obj["model"]
    else:
        sd = obj
    if not isinstance(sd, dict):
        raise TypeError(f"unexpected checkpoint structure at {path}")
    return sd


def fuse_pos_conv_weight(weight_v: torch.Tensor, weight_g: torch.Tensor) -> torch.Tensor:
    # torchaudio applies weight_norm(..., dim=2) to positional conv weight.
    # For dim=2, normalization is across dimensions except dim=2.
    norm = torch.sqrt(torch.sum(weight_v.float() ** 2, dim=(0, 1), keepdim=True) + 1e-12)
    return weight_v.float() / norm * weight_g.float()


def main() -> int:
    args = parse_args()
    gguf = import_gguf_module()

    sd = get_state_dict(Path(args.wavlm_weights))

    n_layers_avail = 0
    while f"encoder.transformer.layers.{n_layers_avail}.attention.attention.in_proj_weight" in sd:
        n_layers_avail += 1
    if n_layers_avail == 0:
        raise RuntimeError("unable to find transformer layer weights in WavLM checkpoint")

    n_layers = int(args.num_transformer_layers)
    if n_layers < 1 or n_layers > n_layers_avail:
        raise ValueError(
            f"--num-transformer-layers must be in [1, {n_layers_avail}], got {n_layers}"
        )

    out = Path(args.outfile)
    out.parent.mkdir(parents=True, exist_ok=True)

    writer = gguf.GGUFWriter(
        path=str(out),
        arch="wavlm-ssl",
        endianess=gguf.GGUFEndian.LITTLE,
        use_temp_file=False,
    )
    writer.add_type(gguf.GGUFType.MODEL)
    writer.add_name(f"WavLM Base+ (first {n_layers} layers) for MioTTS reference conditioning")

    writer.add_uint32("wavlm.sample_rate", int(args.sample_rate))
    writer.add_uint32("wavlm.n_layers", int(n_layers))
    writer.add_uint32("wavlm.n_heads", 12)
    writer.add_uint32("wavlm.head_dim", 64)
    writer.add_uint32("wavlm.embed_dim", 768)
    writer.add_uint32("wavlm.num_buckets", 320)
    writer.add_uint32("wavlm.max_distance", 800)
    writer.add_float32("wavlm.layer_norm_eps", 1e-5)

    # Frontend conv stack (kernel, stride from torchaudio WavLM Base+ config)
    conv_ks = [10, 3, 3, 3, 3, 2, 2]
    conv_st = [5, 2, 2, 2, 2, 2, 2]
    for i, (k, s) in enumerate(zip(conv_ks, conv_st)):
        writer.add_uint32(f"wavlm.feat.conv{i}.kernel", int(k))
        writer.add_uint32(f"wavlm.feat.conv{i}.stride", int(s))

    def add_tensor(dst: str, src: str) -> None:
        if src not in sd:
            raise KeyError(f"missing tensor in checkpoint: {src}")
        writer.add_tensor(dst, to_np_f32(sd[src]))

    add_tensor("wavlm.feat.conv0.norm.weight", "feature_extractor.conv_layers.0.layer_norm.weight")
    add_tensor("wavlm.feat.conv0.norm.bias", "feature_extractor.conv_layers.0.layer_norm.bias")
    add_tensor("wavlm.feat.conv0.weight", "feature_extractor.conv_layers.0.conv.weight")
    for i in range(1, 7):
        add_tensor(f"wavlm.feat.conv{i}.weight", f"feature_extractor.conv_layers.{i}.conv.weight")

    add_tensor("wavlm.proj.norm.weight", "encoder.feature_projection.layer_norm.weight")
    add_tensor("wavlm.proj.norm.bias", "encoder.feature_projection.layer_norm.bias")
    add_tensor("wavlm.proj.weight", "encoder.feature_projection.projection.weight")
    add_tensor("wavlm.proj.bias", "encoder.feature_projection.projection.bias")

    add_tensor("wavlm.transformer.norm.weight", "encoder.transformer.layer_norm.weight")
    add_tensor("wavlm.transformer.norm.bias", "encoder.transformer.layer_norm.bias")

    # Positional convolution (weight_norm fused into plain conv weight).
    pos_w = fuse_pos_conv_weight(
        sd["encoder.transformer.pos_conv_embed.conv.weight_v"],
        sd["encoder.transformer.pos_conv_embed.conv.weight_g"],
    )
    writer.add_tensor("wavlm.pos_conv.weight", to_np_f32(pos_w))
    add_tensor("wavlm.pos_conv.bias", "encoder.transformer.pos_conv_embed.conv.bias")

    for i in range(n_layers):
        p = f"encoder.transformer.layers.{i}"
        add_tensor(f"wavlm.layer.{i}.attn.in_proj.weight", f"{p}.attention.attention.in_proj_weight")
        add_tensor(f"wavlm.layer.{i}.attn.in_proj.bias", f"{p}.attention.attention.in_proj_bias")
        add_tensor(f"wavlm.layer.{i}.attn.out_proj.weight", f"{p}.attention.attention.out_proj.weight")
        add_tensor(f"wavlm.layer.{i}.attn.out_proj.bias", f"{p}.attention.attention.out_proj.bias")
        add_tensor(f"wavlm.layer.{i}.attn.gru.weight", f"{p}.attention.gru_rel_pos_linear.weight")
        add_tensor(f"wavlm.layer.{i}.attn.gru.bias", f"{p}.attention.gru_rel_pos_linear.bias")
        add_tensor(f"wavlm.layer.{i}.attn.gru_const", f"{p}.attention.gru_rel_pos_const")
        add_tensor(f"wavlm.layer.{i}.norm1.weight", f"{p}.layer_norm.weight")
        add_tensor(f"wavlm.layer.{i}.norm1.bias", f"{p}.layer_norm.bias")
        add_tensor(f"wavlm.layer.{i}.ffn.w1.weight", f"{p}.feed_forward.intermediate_dense.weight")
        add_tensor(f"wavlm.layer.{i}.ffn.w1.bias", f"{p}.feed_forward.intermediate_dense.bias")
        add_tensor(f"wavlm.layer.{i}.ffn.w2.weight", f"{p}.feed_forward.output_dense.weight")
        add_tensor(f"wavlm.layer.{i}.ffn.w2.bias", f"{p}.feed_forward.output_dense.bias")
        add_tensor(f"wavlm.layer.{i}.norm2.weight", f"{p}.final_layer_norm.weight")
        add_tensor(f"wavlm.layer.{i}.norm2.bias", f"{p}.final_layer_norm.bias")

    if "encoder.transformer.layers.0.attention.rel_attn_embed.weight" in sd:
        add_tensor("wavlm.layer.0.attn.rel_embed.weight", "encoder.transformer.layers.0.attention.rel_attn_embed.weight")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    summary = {
        "outfile": str(out.resolve()),
        "n_layers": n_layers,
        "sample_rate": int(args.sample_rate),
    }
    print(json.dumps(summary, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
