"""Compare one HQ archive tensor against BF16/F16 SafeTensors without Torch.

This is deliberately a read-only diagnostic.  It avoids importing PyTorch, so
on Windows it does not initialise ROCm just to measure source-to-decoded MSE.
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


def _metrics(source: np.ndarray, restored: np.ndarray) -> dict[str, float]:
    if source.shape != restored.shape:
        raise ValueError(f"shape mismatch: source {source.shape}, decoded {restored.shape}")
    delta = source.astype(np.float32, copy=False) - restored.astype(np.float32, copy=False)
    source_squared = float(np.mean(np.square(source.astype(np.float32, copy=False), dtype=np.float64)))
    mse = float(np.mean(np.square(delta, dtype=np.float64)))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "relative_mse": mse / source_squared if source_squared else 0.0,
        "max_abs_error": float(np.max(np.abs(delta))) if delta.size else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="BF16/F16 SafeTensors file containing --tensor")
    parser.add_argument("--baseline", type=Path, required=True, help="baseline HQ archive")
    parser.add_argument("--candidate", type=Path, required=True, help="candidate HQ archive")
    parser.add_argument("--tensor", required=True, help="exact tensor name")
    parser.add_argument(
        "--candidate-surrogate",
        type=Path,
        help="optional FP16/BF16 SafeTensors materialisation of the candidate tensor",
    )
    parser.add_argument("--output", type=Path, help="optional JSON result path")
    args = parser.parse_args()

    source = SafeTensorNumpyFile(args.source).array(args.tensor)
    baseline_model = hq2.load_model(args.baseline)
    candidate_model = hq2.load_model(args.candidate)
    baseline_entry = baseline_model.descriptor(args.tensor)
    candidate_entry = candidate_model.descriptor(args.tensor)
    baseline = hq2.decode_archive_weight(baseline_model, args.tensor)
    candidate = hq2.decode_archive_weight(candidate_model, args.tensor)
    result = {
        "tensor": args.tensor,
        "source": str(args.source),
        "source_dtype": str(source.dtype),
        "shape": list(source.shape),
        "baseline": {"format": baseline_entry.format.name, **_metrics(source, baseline)},
        "candidate": {"format": candidate_entry.format.name, **_metrics(source, candidate)},
    }
    result["mse_improvement_fraction"] = 1.0 - result["candidate"]["mse"] / result["baseline"]["mse"]
    if args.candidate_surrogate is not None:
        surrogate = SafeTensorNumpyFile(args.candidate_surrogate).array(args.tensor)
        expected_fp16 = candidate.astype(np.float16).astype(np.float32)
        if surrogate.shape != expected_fp16.shape:
            raise ValueError(
                f"surrogate shape mismatch: expected {expected_fp16.shape}, got {surrogate.shape}"
            )
        difference = surrogate.astype(np.float32, copy=False) - expected_fp16
        result["candidate_surrogate"] = {
            "path": str(args.candidate_surrogate),
            "dtype": str(surrogate.dtype),
            "source_error": _metrics(source, surrogate),
            "matches_archive_fp16_exactly": bool(np.array_equal(surrogate, expected_fp16, equal_nan=True)),
            "archive_fp16_max_abs_difference": float(np.max(np.abs(difference))) if difference.size else 0.0,
        }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
