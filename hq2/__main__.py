"""Command-line inspector for direct-loadable HQ archives."""

from __future__ import annotations

import argparse
import sys

from .analyzer import analyze_model, render_analysis


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hq2", description="Inspect an HQ-family quantized model archive")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("analyze", aliases=["inspect"], help="inspect an .hq/.hq1/.hq2/.hq3 archive")
    inspect.add_argument("archive", help="HQ-family model archive path")
    inspect.add_argument("--json", action="store_true", help="emit the complete machine-readable analysis document")
    inspect.add_argument("--tensors", action="store_true", help="include every tensor in the human-readable report")
    inspect.add_argument("--deep", action="store_true", help="sample HQ2 centroids/selectors for physical integrity statistics")
    inspect.add_argument(
        "--deep-bytes",
        type=int,
        default=2 * 1024 * 1024,
        help="maximum HQ2 bytes to inspect per tensor with --deep (default: 2097152)",
    )
    inspect.add_argument("--sha256", action="store_true", help="include a SHA-256 checksum for each packed tensor payload")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    analysis = analyze_model(
        args.archive,
        deep=args.deep,
        checksums=args.sha256,
        deep_sample_bytes=args.deep_bytes,
    )
    if args.json:
        print(analysis.to_json())
    else:
        print(render_analysis(analysis, include_tensors=args.tensors))
    return 0


def analyzer_main() -> int:
    """Entry point for ``hq2-analyze ARCHIVE`` without a redundant subcommand."""
    return main(["analyze", *sys.argv[1:]])


if __name__ == "__main__":  # pragma: no cover - exercised through Python's module runner
    raise SystemExit(main())
