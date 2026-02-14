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
            "Convert preset speaker embedding (.pt or .npz) to GGUF "
            "for --tts-mio-embedding-in / --embedding_in."
        )
    )
    parser.add_argument("embedding", help="input embedding path (.pt or .npz)")
    parser.add_argument("-o", "--outfile", required=True, help="output embedding GGUF path")
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


def load_embedding(path: Path) -> torch.Tensor:
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
        elif len(obj) > 0:
            obj = next(iter(obj.values()))

    if not isinstance(obj, torch.Tensor):
        obj = torch.tensor(obj)

    emb = obj.squeeze().float()
    if emb.dim() != 1:
        raise ValueError(f"embedding must be 1D after squeeze, got shape {tuple(emb.shape)}")
    if emb.numel() == 0:
        raise ValueError("embedding is empty")
    if not torch.isfinite(emb).all():
        raise ValueError("embedding contains non-finite values")
    return emb


def main() -> int:
    args = parse_args()

    inp = Path(args.embedding).expanduser()
    out = Path(args.outfile).expanduser()
    if not inp.is_file():
        raise ValueError(f"embedding file not found: {inp}")
    out.parent.mkdir(parents=True, exist_ok=True)

    gguf = import_gguf_module()

    emb = load_embedding(inp).cpu().numpy().astype(np.float32, copy=False)

    writer = gguf.GGUFWriter(
        path=str(out),
        arch="mio-embedding",
        endianess=gguf.GGUFEndian.LITTLE,
        use_temp_file=False,
    )
    writer.add_type(gguf.GGUFType.MODEL)
    writer.add_name("Mio global embedding")
    writer.add_uint32("mio.embedding.dim", int(emb.shape[0]))
    writer.add_tensor("mio.global_embedding", emb)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(
        json.dumps(
            {
                "input": str(inp.resolve()),
                "outfile": str(out.resolve()),
                "dim": int(emb.shape[0]),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
