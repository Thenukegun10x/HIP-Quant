"""Stream a Safetensors checkpoint into the HQ2-native model container.

The writer receives each packed projection immediately, so conversion does not
need a 24 GB BF16 checkpoint plus a second complete quantized copy in memory.
For Gemma 4 12B, ``gemma-mlp`` is the quality-first default: the original
checkpoint remains the source for embeddings, norms, attention, and multimodal
components until their own HQ2 inference paths are selected deliberately.

Example (from the repository root)::

    python tools/quantize_hq2_model.py \
      --input "Own Quant/Gemma4-12B-it/model.safetensors" \
      --output "Own Quant/Gemma4-12B-HQ2-MLP-L8.hq2" \
      --profile gemma-mlp --backend rocm --iterations 8
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import re
import sys
import time
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hq2


def _select(name: str, shape: tuple[int, ...], profile: str) -> bool:
    if len(shape) != 2 or shape[-1] % hq2.BLOCK_SIZE:
        return False
    is_text_layer = name.startswith("model.language_model.layers.")
    if profile == "gemma-mlp":
        return is_text_layer and ".mlp." in name and name.endswith(".weight")
    if profile == "gemma-text-projections":
        return (
            is_text_layer
            and name.endswith(".weight")
            and (".mlp." in name or ".self_attn." in name)
        )
    if profile == "all-2d":
        return name.endswith(".weight")
    raise ValueError(f"Unknown profile {profile!r}")


def _selected_keys(checkpoint: Path, profile: str) -> list[str]:
    from safetensors import safe_open

    with safe_open(checkpoint, framework="pt", device="cpu") as source:
        selected = [
            name
            for name in source.keys()
            if _select(name, tuple(source.get_slice(name).get_shape()), profile)
        ]
    # Safetensors exposes keys lexicographically (0, 1, 10, ...).  Numeric
    # layer order is better for sequential model loading and read-ahead.
    def order(name: str) -> tuple[int, str]:
        match = re.search(r"\.layers\.(\d+)\.", name)
        return (int(match.group(1)) if match else -1, name)

    return sorted(selected, key=order)


def _human_bytes(value: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def convert(args: argparse.Namespace) -> Path:
    import torch
    from safetensors import safe_open

    checkpoint = Path(args.input)
    output = Path(args.output)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if output.suffix.lower() != ".hq2":
        raise ValueError("Output must use the .hq2 extension")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output}; pass --overwrite to replace it")
    if args.backend == "rocm" and not hq2.backend_status()["rocm"].available:
        raise RuntimeError("--backend rocm requested but the native HIP quantizer is unavailable")

    selected = _selected_keys(checkpoint, args.profile)
    if args.max_tensors is not None:
        selected = selected[: args.max_tensors]
    if not selected:
        raise ValueError(f"No tensors matched profile {args.profile!r}")
    importance_vectors: dict[str, np.ndarray] | None = None
    if args.imatrix is not None:
        imatrix_path = Path(args.imatrix)
        if not imatrix_path.is_file():
            raise FileNotFoundError(f"Importance matrix file not found: {imatrix_path}")
        with np.load(imatrix_path, allow_pickle=False) as stored:
            missing = [name for name in selected if name not in stored.files]
            if missing:
                raise KeyError(f"Importance matrix is missing {len(missing)} selected tensors, e.g. {missing[:2]}")
            importance_vectors = {
                name: np.ascontiguousarray(stored[name], dtype=np.float32) for name in selected
            }

    total_values = 0
    total_packed = 0
    with safe_open(checkpoint, framework="pt", device="cpu") as source:
        for name in selected:
            shape = tuple(source.get_slice(name).get_shape())
            total_values += int(shape[0]) * int(shape[1])
            total_packed += hq2.BLOCK_BYTES * (int(shape[0]) * int(shape[1]) // hq2.BLOCK_SIZE)
    print(
        f"Selected {len(selected)} tensors / {total_values:,} weights -> "
        f"{_human_bytes(total_packed)} HQ2 payload ({hq2.BITS_PER_WEIGHT:.2f} bpw).",
        flush=True,
    )
    if args.dry_run:
        for name in selected:
            print(name)
        return output

    metadata = {
        "architecture": args.architecture,
        "quantization": "HQ2",
        "bits_per_weight": hq2.BITS_PER_WEIGHT,
        "profile": args.profile,
        "iterations": args.iterations,
        "source_checkpoint": checkpoint.name,
        "imatrix": None if args.imatrix is None else Path(args.imatrix).name,
        "hybrid_base_checkpoint_required": True,
        "non_hq2_weights": "Load from the source/base checkpoint; only selected projection weights are in this archive.",
    }
    start = time.perf_counter()
    # Direct write avoids an intermediate dequantized or generic-compressed
    # model file.  A failed conversion leaves the incomplete output visibly
    # invalid rather than a file that appears loadable.
    with hq2.HQ2ModelWriter(output, metadata=metadata) as writer:
        with safe_open(checkpoint, framework="pt", device="cpu") as source:
            for index, name in enumerate(selected, start=1):
                bf16 = source.get_tensor(name)
                if bf16.dtype != torch.bfloat16:
                    raise TypeError(f"{name} must be BF16, got {bf16.dtype}")
                # Torch cannot directly expose CPU BF16 as NumPy.  Float32 is
                # the native HIP quantizer input, and only this one matrix is
                # staged at a time.
                values = bf16.float().numpy()
                importance = None
                if importance_vectors is not None:
                    vector = importance_vectors[name]
                    if vector.shape != (values.shape[1],):
                        raise ValueError(
                            f"Importance for {name} has shape {vector.shape}; expected ({values.shape[1]},)"
                        )
                    maximum = float(vector.max())
                    if maximum > 0.0:
                        vector = vector / maximum
                    # Native HQ2 applies the weight per scalar during Lloyd
                    # updates.  Broadcast once per streamed layer; no other
                    # model tensor is retained.
                    importance = np.ascontiguousarray(np.broadcast_to(vector, values.shape))
                packed = hq2.quantize(
                    values, importance=importance, backend=args.backend, iterations=args.iterations
                )
                writer.add(name, packed)
                elapsed = time.perf_counter() - start
                print(
                    f"[{index:3d}/{len(selected):3d}] {name}  "
                    f"{_human_bytes(packed.nbytes)}  elapsed {elapsed:.1f}s",
                    flush=True,
                )
                del bf16, values, importance, packed
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    print(f"Wrote {output} in {time.perf_counter() - start:.1f}s", flush=True)
    return output


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="BF16 Safetensors checkpoint")
    parser.add_argument("--output", required=True, help="Persistent .hq2 model-weight archive")
    parser.add_argument("--architecture", default="gemma4", help="Recorded model architecture metadata")
    parser.add_argument(
        "--profile",
        default="gemma-mlp",
        choices=("gemma-mlp", "gemma-text-projections", "all-2d"),
        help="Which direct-linear weights to quantize",
    )
    parser.add_argument("--backend", default="rocm", choices=("rocm", "cpu"))
    parser.add_argument("--iterations", type=int, default=8, choices=range(1, 17))
    parser.add_argument(
        "--imatrix",
        help="Optional .npz activation importance vectors (one [in_features] vector per selected tensor)",
    )
    parser.add_argument("--max-tensors", type=int, help="Convert only the first N matching tensors (smoke test)")
    parser.add_argument("--dry-run", action="store_true", help="Print selected tensors and projected size")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output .hq2 file")
    return parser.parse_args(argv)


if __name__ == "__main__":
    convert(parse_args())
