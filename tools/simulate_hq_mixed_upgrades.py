"""Simulate targeted HQ2 upgrades selected by a mixed-archive error report.

The full error scan identifies the tensors with the most activation-weighted
output perturbation per stored byte.  This tool then re-quantizes only those
HQ2 tensors as HQ3 (with the same activation calibration) and Q4_0, reporting
the measured diagonal-Hessian error reduction per additional byte.  It is the
bridge between a screening metric and a small, controlled end-to-end ablation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from safetensors import safe_open


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for import_path in (REPOSITORY_ROOT.parent, REPOSITORY_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import hq2


HQ2_BPW = 2.25
HQ3_BPW = 3.5
Q4_0_BPW = 4.5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--error-report", type=Path, required=True)
    parser.add_argument("--imatrix", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--top", type=int, default=8, help="number of HQ2 tensors selected by error per byte")
    parser.add_argument("--hq3-iters", type=int, default=8, choices=range(1, 17))
    parser.add_argument("--rows-per-chunk", type=int, default=2048)
    parser.add_argument("--max-values-per-chunk", type=int, default=8_000_000)
    return parser.parse_args(argv)


def _normalise(vector: np.ndarray) -> np.ndarray:
    vector = np.ascontiguousarray(vector, dtype=np.float32)
    maximum = float(vector.max())
    return vector if maximum <= 0.0 else np.ascontiguousarray(vector / maximum)


def _q4_0_roundtrip(values: np.ndarray) -> np.ndarray:
    """CPU implementation matching the project's native Q4_0 block layout."""

    values = np.ascontiguousarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] % 32:
        raise ValueError(f"Q4_0 expects [rows, columns%32==0], got {values.shape}")
    blocks = values.reshape(-1, 32)
    maximum = blocks[np.arange(blocks.shape[0]), np.abs(blocks).argmax(axis=1)]
    scale = maximum / -8.0
    quantized = np.full(blocks.shape, 8, dtype=np.int16)
    nonzero = scale != 0.0
    quantized[nonzero] = np.trunc(blocks[nonzero] / scale[nonzero, None] + 8.5).astype(np.int16)
    quantized = np.clip(quantized, 0, 15)
    stored_scale = scale.astype(np.float16).astype(np.float32)
    return (stored_scale[:, None] * (quantized.astype(np.float32) - 8.0)).reshape(values.shape)


def _metric_add(total: dict[str, float], source: np.ndarray, restored: np.ndarray, importance: np.ndarray) -> None:
    difference = np.asarray(source, dtype=np.float32) - np.asarray(restored, dtype=np.float32)
    total["weighted_sse"] += float(np.sum(np.square(difference, dtype=np.float32) * importance[None, :], dtype=np.float64))
    total["source_sse"] += float(np.sum(np.square(source, dtype=np.float32) * importance[None, :], dtype=np.float64))


def _measure_q4(source_slice: Any, shape: tuple[int, int], importance: np.ndarray, rows_per_chunk: int) -> dict[str, float]:
    total = {"weighted_sse": 0.0, "source_sse": 0.0}
    rows, _ = shape
    for start in range(0, rows, rows_per_chunk):
        stop = min(rows, start + rows_per_chunk)
        source = np.ascontiguousarray(source_slice[start:stop].float().numpy(), dtype=np.float32)
        restored = _q4_0_roundtrip(source)
        _metric_add(total, source, restored, importance)
        del source, restored
    return total


def _measure_hq3(
    source_slice: Any,
    shape: tuple[int, int],
    importance: np.ndarray,
    *,
    iterations: int,
    rows_per_chunk: int,
    max_values_per_chunk: int,
) -> dict[str, float]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("HQ3 upgrade simulation requires a visible Torch CUDA/ROCm device")
    rows, columns = shape
    if rows_per_chunk <= 0 or max_values_per_chunk <= 0:
        raise ValueError("chunk limits must be positive")
    rows_per_chunk = min(rows_per_chunk, max(1, max_values_per_chunk // columns))
    device_importance = torch.from_numpy(_normalise(importance)).to(device="cuda", dtype=torch.float32)
    total = {"weighted_sse": 0.0, "source_sse": 0.0}
    try:
        for start in range(0, rows, rows_per_chunk):
            stop = min(rows, start + rows_per_chunk)
            source = source_slice[start:stop].to(device="cuda", non_blocking=False)
            weight = device_importance.expand(stop - start, columns).contiguous()
            packed = hq2.quantize(source, importance=weight, backend="torch", format="hq3", iterations=iterations)
            restored = packed.dequantize().cpu().numpy()
            source_cpu = source.float().cpu().numpy()
            _metric_add(total, source_cpu, restored, importance)
            del source, weight, packed, restored, source_cpu
            torch.cuda.empty_cache()
    finally:
        del device_importance
        torch.cuda.empty_cache()
    return total


def _row(name: str, current: dict[str, Any], candidate: str, metric: dict[str, float], bpw: float) -> dict[str, Any]:
    values = int(current["values"])
    current_sse = float(current["activation_weighted_sse"])
    candidate_sse = float(metric["weighted_sse"])
    current_bytes = int(current["payload_bytes"])
    candidate_bytes = int(values * bpw / 8)
    extra_bytes = candidate_bytes - current_bytes
    reduction = current_sse - candidate_sse
    return {
        "name": name,
        "source_format": current["format"],
        "candidate_format": candidate,
        "values": values,
        "current_payload_bytes": current_bytes,
        "candidate_payload_bytes": candidate_bytes,
        "additional_payload_bytes": extra_bytes,
        "current_weighted_sse": current_sse,
        "candidate_weighted_sse": candidate_sse,
        "current_relative_sse": float(current["activation_relative_sse"]),
        "candidate_relative_sse": candidate_sse / max(float(metric["source_sse"]), np.finfo(np.float64).tiny),
        "weighted_sse_reduction": reduction,
        "reduction_per_additional_byte": None if extra_bytes <= 0 else reduction / extra_bytes,
        "candidate_bpw": bpw,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.top <= 0:
        raise ValueError("--top must be positive")
    report = json.loads(args.error_report.read_text(encoding="utf-8"))
    selected = [row for row in report["ranked_by_error_per_payload_byte"] if row["format"] == "HQ2"][:args.top]
    if not selected:
        raise ValueError("Error report contains no calibrated HQ2 tensors")
    with np.load(args.imatrix, allow_pickle=False) as archive:
        importance = {name: np.ascontiguousarray(archive[name], dtype=np.float32) for name in archive.files}
    results: list[dict[str, Any]] = []
    with safe_open(args.source, framework="pt", device="cpu") as source:
        for index, current in enumerate(selected, start=1):
            name = str(current["name"])
            shape = tuple(int(value) for value in current["shape"])
            vector = importance.get(name)
            if vector is None:
                raise KeyError(f"No calibration vector for selected tensor {name}")
            source_slice = source.get_slice(name)
            q4 = _measure_q4(source_slice, shape, vector, args.rows_per_chunk)
            hq3 = _measure_hq3(
                source_slice,
                shape,
                vector,
                iterations=args.hq3_iters,
                rows_per_chunk=args.rows_per_chunk,
                max_values_per_chunk=args.max_values_per_chunk,
            )
            results.append(_row(name, current, "Q4_0", q4, Q4_0_BPW))
            results.append(_row(name, current, "HQ3", hq3, HQ3_BPW))
            print(f"[{index}/{len(selected)}] {name}", flush=True)
    results.sort(key=lambda row: float(row["reduction_per_additional_byte"] or float("-inf")), reverse=True)
    output = {
        "selection": "top calibrated HQ2 tensors by current activation-weighted error per payload byte",
        "source": str(args.source),
        "error_report": str(args.error_report),
        "imatrix": str(args.imatrix),
        "hq3_iterations": args.hq3_iters,
        "rows_per_chunk": args.rows_per_chunk,
        "max_values_per_chunk": args.max_values_per_chunk,
        "candidates": results,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for row in results:
        print(
            f"{row['candidate_format']:4s} {row['name']}: "
            f"relative {row['current_relative_sse']:.4f} -> {row['candidate_relative_sse']:.4f}; "
            f"+{row['additional_payload_bytes'] / 2**20:.2f} MiB; "
            f"reduction/byte={row['reduction_per_additional_byte']:.6e}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
