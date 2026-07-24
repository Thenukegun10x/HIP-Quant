"""Benchmark direct packed HQ8_G128 W8A16 linear layers on Gemma MLP shapes.

This is a kernel benchmark, not whole-model throughput. The control is
torch.nn.functional.linear using the same decoded HQ8_G128 weights, so it
measures execution speed without conflating source-model quantization quality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hq2
from hq2.safetensors_numpy import SafeTensorNumpyFile


DEFAULT_SOURCE = ROOT / "Own Quant" / "Gemma4-12B-it" / "model.safetensors"
DEFAULT_TENSORS = (
    "model.language_model.layers.0.mlp.gate_proj.weight",
    "model.language_model.layers.0.mlp.down_proj.weight",
)


def _event_time(torch, action, *, warmup: int, iterations: int) -> dict[str, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            action()
        torch.cuda.synchronize()
        samples: list[float] = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            action()
            stop.record()
            stop.synchronize()
            samples.append(float(start.elapsed_time(stop)))
    return {
        "median_ms": float(np.median(samples)),
        "mean_ms": float(np.mean(samples)),
        "min_ms": float(np.min(samples)),
        "max_ms": float(np.max(samples)),
    }


def _error_summary(actual, expected) -> dict[str, float]:
    torch = __import__("torch")
    delta = (actual.float() - expected.float()).abs()
    return {
        "max_abs_error": float(delta.max().item()),
        "mean_abs_error": float(delta.mean().item()),
        "all_finite": bool(torch.isfinite(actual).all().item()),
    }


def _dtype(torch, name: str):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--tensor", action="append", default=[], help="exact weight name; repeat as needed")
    parser.add_argument("--batches", nargs="+", type=int, default=(1, 8, 32, 128))
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0 or any(batch <= 0 for batch in args.batches):
        raise ValueError("warmup must be non-negative; iterations and batches must be positive")

    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError("This benchmark requires ROCm Torch with one visible GPU")
    if not hq2.rocm_hq8_g128_fused_available():
        raise RuntimeError("HQ8_G128 fused extension is unavailable; rebuild with setup_torch.py")
    dtype = _dtype(torch, args.dtype)
    names = tuple(args.tensor) if args.tensor else DEFAULT_TENSORS
    source = SafeTensorNumpyFile(args.source)
    results: list[dict[str, Any]] = []
    generator = torch.Generator(device="cuda").manual_seed(8128)

    for name in names:
        host_weight = source.array(name)
        if host_weight.ndim != 2 or host_weight.shape[1] % 128:
            raise ValueError(f"{name!r} must be a 128-aligned rank-2 matrix, got {host_weight.shape}")
        packed_host = hq2.quantize(host_weight, backend="cpu", format="hq8_g128")
        packed = packed_host.to("cuda")
        layer = hq2.HQ8Linear(packed).eval()
        decoded = packed.dequantize(dtype=dtype).contiguous()
        out_features, in_features = host_weight.shape
        layer_result: dict[str, Any] = {
            "name": name,
            "shape": [int(out_features), int(in_features)],
            "packed_bytes": packed_host.nbytes,
            "packed_bpw": packed_host.bits_per_weight,
            "batches": {},
        }
        for batch in args.batches:
            input = torch.randn((batch, in_features), generator=generator, device="cuda", dtype=dtype)
            with torch.inference_mode():
                direct = layer(input)
                reference = F.linear(input, decoded)
            correctness = _error_summary(direct, reference)
            packed_time = _event_time(
                torch, lambda: layer(input), warmup=args.warmup, iterations=args.iterations
            )
            decoded_time = _event_time(
                torch, lambda: F.linear(input, decoded), warmup=args.warmup, iterations=args.iterations
            )
            flops = 2.0 * batch * out_features * in_features
            packed_time["effective_tflop_s"] = flops / (packed_time["median_ms"] * 1.0e9)
            packed_time["packed_weight_read_gib_s"] = (
                packed_host.nbytes * batch * 1000.0 / packed_time["median_ms"] / (1024**3)
            )
            layer_result["batches"][str(batch)] = {
                "correctness": correctness,
                "packed_hq8_g128_w8a16": packed_time,
                "decoded_bf16_f_linear": decoded_time,
                "packed_vs_decoded_speedup": decoded_time["median_ms"] / packed_time["median_ms"],
            }
            del input, direct, reference
        results.append(layer_result)
        del layer, decoded, packed, packed_host, host_weight
        torch.cuda.empty_cache()

    result = {
        "benchmark": "hq8_g128_w8a16_gemma_mlp_v1",
        "device": torch.cuda.get_device_name(0),
        "dtype": args.dtype,
        "source": str(args.source),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "scope": (
            "Direct packed row-major HQ8_G128 W8A16 linear versus decoded-HQ8 "
            "BF16 F.linear; this is not whole-model or W8A8 throughput."
        ),
        "layers": results,
        "finished_unix_seconds": time.time(),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

