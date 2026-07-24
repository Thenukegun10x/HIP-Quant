"""Kernel-only HIP quantizer benchmark for HQ2 and the compact I-Quants.

The timed region contains only quantizer kernel launches.  It excludes
allocation, host-to-device upload, device-to-host download, and Python
overhead.  Run it in a new process if setting HIP_VISIBLE_DEVICES or
HIP_QUANT_DEVICE so HIP sees the requested device before initialization.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT.parent))

from hip_quant import GGML_TYPE, get_hip_quant  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--cols", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument(
        "--expected-arch",
        default="gfx1201",
        help="Fail before timing unless the selected device arch matches (empty disables the check).",
    )
    parser.add_argument("--out", type=Path, help="Optional JSON results file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rows <= 0 or args.cols <= 0 or args.cols % 256:
        raise SystemExit("--rows must be positive and --cols must be a positive multiple of 256")
    if args.warmup < 0 or args.iterations < 1 or args.trials < 1:
        raise SystemExit("--warmup >= 0, --iterations >= 1, and --trials >= 1 are required")

    quantizer = get_hip_quant()
    device = quantizer.device_name
    arch = quantizer.gcn_arch
    if args.expected_arch and args.expected_arch not in arch:
        raise SystemExit(
            f"Refusing to benchmark {device} ({arch}); expected {args.expected_arch}. "
            "Start a new process with HIP_VISIBLE_DEVICES / HIP_QUANT_DEVICE set to the discrete GPU."
        )

    source = np.random.default_rng(20260716).standard_normal(
        (args.rows, args.cols), dtype=np.float32
    )
    configurations = (
        ("HQ2 (4 Lloyd)", GGML_TYPE["HQ2"], 4),
        ("HQ2 (8 Lloyd)", GGML_TYPE["HQ2"], 8),
        ("IQ2_XXS", GGML_TYPE["IQ2_XXS"], 4),
        ("IQ3_XXS", GGML_TYPE["IQ3_XXS"], 4),
    )

    elements = args.rows * args.cols
    results = []
    print(f"Device: {device} | {arch} | HIP device {quantizer.selected_device}")
    print(
        f"Shape: {args.rows} x {args.cols} | {args.trials} trials x "
        f"{args.iterations} timed launches ({args.warmup} warmup launches each)"
    )
    print(f"{'Configuration':<18} {'median ms':>10} {'min ms':>10} {'max ms':>10} {'Gelem/s':>10}")
    for name, type_num, hq2_iterations in configurations:
        timings_ms = [
            quantizer.benchmark_quantize_kernel(
                source,
                type_num,
                hq2_iterations=hq2_iterations,
                warmup_iterations=args.warmup,
                timed_iterations=args.iterations,
            )
            for _ in range(args.trials)
        ]
        median_ms = float(np.median(timings_ms))
        result = {
            "configuration": name,
            "ggml_type": int(type_num),
            "hq2_iterations": hq2_iterations if type_num == GGML_TYPE["HQ2"] else None,
            "timings_ms": timings_ms,
            "median_ms": median_ms,
            "min_ms": float(min(timings_ms)),
            "max_ms": float(max(timings_ms)),
            "gelements_per_second": elements / (median_ms * 1_000_000.0),
        }
        results.append(result)
        print(
            f"{name:<18} {result['median_ms']:>10.4f} {result['min_ms']:>10.4f} "
            f"{result['max_ms']:>10.4f} {result['gelements_per_second']:>10.3f}"
        )

    payload = {
        "method": (
            "HIP events around direct quantizer kernel launches only; source upload, allocation, "
            "output transfer, and importance matrix are excluded."
        ),
        "device": device,
        "arch": arch,
        "selected_device": quantizer.selected_device,
        "hip_runtime_version": quantizer.hip_runtime_version,
        "shape": [args.rows, args.cols],
        "warmup_launches": args.warmup,
        "timed_launches_per_trial": args.iterations,
        "trials": args.trials,
        "results": results,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
