"""Screen the portable HQ8_G128 codec against Q8_0-like 32-value groups.

The tool is CPU-only: it uses SafeTensorNumpyFile.rows() to sample BF16
matrices without importing Torch or initialising ROCm. It is a source-space
codec screen, not a model-quality or inference-speed benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hq2
from hq2.safetensors_numpy import SafeTensorNumpyFile


_DEFAULT_TENSORS = (
    "model.language_model.layers.0.mlp.gate_proj.weight",
    "model.language_model.layers.0.mlp.up_proj.weight",
    "model.language_model.layers.0.mlp.down_proj.weight",
)


def _q8_reference_decode(values: np.ndarray, group_size: int = 32) -> np.ndarray:
    """FP16-scale symmetric-int8 reference math for a specified group size."""

    values = np.ascontiguousarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[-1] % group_size:
        raise ValueError(f"Expected a rank-2 tensor with {group_size}-aligned width, got {values.shape}")
    groups = values.reshape(-1, group_size)
    scales = (np.max(np.abs(groups), axis=1) / 127.0).astype(np.float16).astype(np.float32)
    normalized = np.zeros_like(groups)
    nonzero = scales > 0.0
    normalized[nonzero] = groups[nonzero] / scales[nonzero, None]
    codes = np.clip(np.rint(normalized), -127, 127).astype(np.int8)
    return (scales[:, None] * codes.astype(np.float32)).reshape(values.shape)


def _metrics(source: np.ndarray, decoded: np.ndarray) -> dict[str, float]:
    delta = source.astype(np.float32, copy=False) - decoded.astype(np.float32, copy=False)
    squared = np.square(delta, dtype=np.float32)
    source_squared = np.square(source.astype(np.float32, copy=False), dtype=np.float32)
    mse = float(np.mean(squared, dtype=np.float64))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "relative_mse": mse / max(float(np.mean(source_squared, dtype=np.float64)), np.finfo(np.float64).tiny),
        "max_abs_error": float(np.max(np.abs(delta))) if delta.size else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="BF16/F16 SafeTensors source file")
    parser.add_argument("--tensor", action="append", default=[], help="exact matrix name; repeat as needed")
    parser.add_argument("--rows", type=int, default=128, help="leading rows sampled from each matrix")
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    args = parser.parse_args()
    if args.rows <= 0:
        raise ValueError("--rows must be positive")
    names = tuple(args.tensor) if args.tensor else _DEFAULT_TENSORS

    source = SafeTensorNumpyFile(args.source)
    results: list[dict[str, object]] = []
    q32_sse = 0.0
    g128_sse = 0.0
    total_values = 0
    for name in names:
        descriptor = source.descriptor(name)
        shape = tuple(int(value) for value in descriptor["shape"])
        if len(shape) != 2 or shape[1] % 128:
            raise ValueError(f"{name!r} is not a 128-aligned matrix: {shape}")
        values = source.rows(name, 0, min(args.rows, shape[0]))
        q32 = _q8_reference_decode(values, 32)
        g128 = hq2.quantize(values, backend="cpu", format="hq8_g128").dequantize()
        q32_metrics = _metrics(values, q32)
        g128_metrics = _metrics(values, g128)
        value_count = int(values.size)
        q32_sse += q32_metrics["mse"] * value_count
        g128_sse += g128_metrics["mse"] * value_count
        total_values += value_count
        results.append({
            "name": name,
            "source_shape": list(shape),
            "sample_shape": list(values.shape),
            "q8_0_like_g32": q32_metrics,
            "hq8_g128": g128_metrics,
            "g128_to_g32_mse_ratio": g128_metrics["mse"] / max(q32_metrics["mse"], np.finfo(np.float64).tiny),
        })

    aggregate_q32 = q32_sse / total_values
    aggregate_g128 = g128_sse / total_values
    result = {
        "scope": "source-space codec screen; not teacher-forced model quality or kernel speed",
        "source": str(args.source),
        "sample_rows_per_tensor": args.rows,
        "formats": {
            "q8_0_like_g32": {"group_size": 32, "bpw": 8.5},
            "hq8_g128": {"group_size": 128, "bpw": 8.125},
        },
        "tensors": results,
        "aggregate": {
            "values": total_values,
            "q8_0_like_g32_mse": aggregate_q32,
            "hq8_g128_mse": aggregate_g128,
            "g128_to_g32_mse_ratio": aggregate_g128 / max(aggregate_q32, np.finfo(np.float64).tiny),
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

