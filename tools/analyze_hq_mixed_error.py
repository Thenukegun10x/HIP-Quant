"""Rank actual HQ mixed-archive weight error using activation calibration.

This is a screening tool, not an end-to-end benchmark.  It reports the exact
source-to-packed error of the archive already on disk and uses captured input
activation energy to rank which layers should receive an upcast ablation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for import_path in (REPOSITORY_ROOT.parent, REPOSITORY_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from hq2.sensitivity import analyze_mixed_archive_error, summarize_tensor_errors


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="source BF16 Safetensors checkpoint")
    parser.add_argument("--archive", type=Path, required=True, help="mixed .hq archive")
    parser.add_argument("--imatrix", type=Path, help="optional calibration vectors [in_features]")
    parser.add_argument("--json-out", type=Path, required=True, help="write complete JSON report")
    parser.add_argument("--top", type=int, default=24, help="rows to print from each calibrated ranking")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.top <= 0:
        raise ValueError("--top must be positive")
    results = analyze_mixed_archive_error(args.source, args.archive, imatrix_path=args.imatrix)
    report = summarize_tensor_errors(results)
    report["source"] = str(args.source)
    report["archive"] = str(args.archive)
    report["imatrix"] = None if args.imatrix is None else str(args.imatrix)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Scored {report['scored_tensors']} rank-2 packed tensors; {report['calibrated_tensors']} use activation calibration.")
    for title, rows in (
        ("activation weighted output error", report["ranked_by_activation_error"]),
        ("error per packed byte", report["ranked_by_error_per_payload_byte"]),
    ):
        print(f"\nTop {min(args.top, len(rows))} by {title}:")
        for row in rows[:args.top]:
            print(
                f"  {row['format']:4s} {row['name']}: "
                f"weighted_sse={row['activation_weighted_sse']:.6e}, "
                f"relative={row['activation_relative_sse']:.6e}, "
                f"payload={row['payload_bpw']:.2f} BPW",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
