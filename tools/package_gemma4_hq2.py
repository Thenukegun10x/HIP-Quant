"""Build a standalone, download-ready Gemma 4 HQ2 package.

The package contains one ``model.hq2`` archive: selected MLP projections stay
in native 2.25-bpw HQ2, while every other checkpoint tensor is copied exactly
as a typed HQ ``RAW`` payload.  Config/tokenizer assets are copied alongside
it, so recipients need only this directory and ``hip-quant``—not the original
Safetensors checkpoint.

Example::

    python tools/package_gemma4_hq2.py \
      --base "Own Quant/Gemma4-12B-it" \
      --hq2 "Own Quant/Gemma4-12B-HQ2-MLP-L8-IMAT.hq2" \
      --output "Own Quant/Gemma4-12B-HQ2-Standalone"
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hq2
from hq2.archive import HQ2_FORMAT


def _human_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024.0 or unit == "TiB":
            return f"{amount:.2f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1024.0
    raise AssertionError("unreachable")


def _clear_output(path: Path, *, overwrite: bool) -> None:
    if not path.exists():
        path.mkdir(parents=True)
        return
    if any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory {path} is not empty; pass --overwrite to replace it")
        for child in path.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    path.mkdir(parents=True, exist_ok=True)


def _copy_runtime_assets(base: Path, output: Path) -> list[str]:
    copied: list[str] = []
    for source in base.iterdir():
        if not source.is_file() or source.name == "model.safetensors":
            continue
        destination = output / source.name
        shutil.copy2(source, destination)
        copied.append(source.name)
    if "config.json" not in copied or "tokenizer.json" not in copied:
        raise FileNotFoundError("Base checkpoint directory must contain config.json and tokenizer.json")
    return sorted(copied)


def build_package(args: argparse.Namespace) -> Path:
    import torch
    from safetensors import safe_open

    base = Path(args.base)
    hq2_path = Path(args.hq2)
    output = Path(args.output)
    checkpoint = base / "model.safetensors"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Expected base checkpoint at {checkpoint}")
    if not hq2_path.is_file():
        raise FileNotFoundError(f"HQ2 archive not found: {hq2_path}")
    source_hq2 = hq2.load_model(hq2_path)
    hq2_names = tuple(name for name in source_hq2.tensor_names if source_hq2.descriptor(name).format == HQ2_FORMAT)
    if not hq2_names or len(hq2_names) != len(source_hq2.tensor_names):
        raise ValueError("--hq2 must be a canonical HQ2-only hybrid archive")

    with safe_open(checkpoint, framework="pt", device="cpu") as source:
        source_names = tuple(source.keys())
    missing = sorted(set(hq2_names) - set(source_names))
    if missing:
        raise KeyError(f"HQ2 archive has tensors absent from the base checkpoint: {missing[:3]}")
    raw_names = tuple(name for name in source_names if name not in set(hq2_names))
    if args.dry_run:
        print(f"HQ2 tensors: {len(hq2_names)}; lossless RAW tensors: {len(raw_names)}")
        print(f"Output: {output}")
        return output

    _clear_output(output, overwrite=args.overwrite)
    temporary_archive = output / "model.partial.hq2"
    final_archive = output / "model.hq2"
    metadata = {
        "architecture": "gemma4",
        "quantization": "HQ2",
        "bits_per_weight": hq2.BITS_PER_WEIGHT,
        "standalone_package": True,
        "base_checkpoint_required": False,
        "hq2_tensor_count": len(hq2_names),
        "raw_tensor_count": len(raw_names),
        "hq2_source_archive": hq2_path.name,
        "source_checkpoint": checkpoint.name,
        "profile": source_hq2.metadata.get("profile"),
        "iterations": source_hq2.metadata.get("iterations"),
        "imatrix": source_hq2.metadata.get("imatrix"),
    }
    start = time.perf_counter()
    raw_bytes = 0
    with hq2.HQModelWriter(temporary_archive, metadata=metadata) as writer:
        for index, name in enumerate(hq2_names, start=1):
            packed = source_hq2.tensor(name)
            try:
                writer.add(name, packed)
            finally:
                mapping = getattr(packed.packed, "_mmap", None)
                if mapping is not None:
                    mapping.close()
            print(f"[HQ2 {index:3d}/{len(hq2_names):3d}] {name}", flush=True)
        with safe_open(checkpoint, framework="pt", device="cpu") as source:
            for index, name in enumerate(raw_names, start=1):
                value = source.get_tensor(name).contiguous()
                raw = value.view(torch.uint8).numpy()
                writer.add_raw(
                    name,
                    raw,
                    shape=tuple(value.shape),
                    format=hq2.raw_format_for_torch(value.dtype),
                )
                raw_bytes += raw.nbytes
                if index % 32 == 0 or index == len(raw_names):
                    print(
                        f"[RAW {index:3d}/{len(raw_names):3d}] {name}  "
                        f"copied {_human_bytes(raw_bytes)}",
                        flush=True,
                    )
                del value, raw
                gc.collect()
    temporary_archive.replace(final_archive)
    assets = _copy_runtime_assets(base, output)
    manifest = {
        "format": "HQ2_STANDALONE_PACKAGE",
        "version": 1,
        "archive": final_archive.name,
        "architecture": "gemma4",
        "runtime": "hip-quant>=0.11.0",
        "hq2_tensor_count": len(hq2_names),
        "raw_tensor_count": len(raw_names),
        "runtime_assets": assets,
    }
    (output / "hq2-package.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# Standalone Gemma 4 HQ2 package\n\n"
        "Load this directory without the original BF16 checkpoint:\n\n"
        "```python\nimport hq2\nmodel = hq2.load_gemma4_hq2_package('path/to/this-directory')\n```\n",
        encoding="utf-8",
    )
    print(
        f"Standalone package written to {output} in {time.perf_counter() - start:.1f}s "
        f"({final_archive.stat().st_size:,} archive bytes).",
        flush=True,
    )
    return output


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Base Gemma directory containing model.safetensors and tokenizer/config")
    parser.add_argument("--hq2", required=True, help="Calibrated HQ2-only hybrid .hq2 archive")
    parser.add_argument("--output", required=True, help="New standalone package directory")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and report tensor counts without writing")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory")
    return parser.parse_args(argv)


if __name__ == "__main__":
    build_package(parse_args())
