"""Materialize a mixed HQ archive as a sharded Transformers checkpoint.

This is a quality-evaluation bridge, not an inference implementation.  Every
tensor is decoded from the already-built ``.hq`` archive, then saved as BF16
Safetensors shards which Transformers can load using the normal Gemma path.
It therefore tests the exact stored HQ2/HQ3/Q4_0/Q8_0/RAW weights without
claiming the packed dispatch has been benchmarked.

The conversion is streaming: one archive tensor and one output shard are live
at a time.  A full 12B model never needs to fit in host RAM.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import sys
from typing import Iterable

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import hq2
from hq2.sensitivity import decode_archive_weight


_RUNTIME_ASSETS = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)
_TORCH_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
_SURROGATE_DTYPES = (*_TORCH_DTYPES, "mixed-fp16-fp32")


def _human_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024.0 or unit == "TiB":
            return f"{int(amount)} B" if unit == "B" else f"{amount:.2f} {unit}"
        amount /= 1024.0
    raise AssertionError("unreachable")


def _positive_gib(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="source Gemma Safetensors file used to create the archive")
    parser.add_argument(
        "--assets-from",
        type=Path,
        help=(
            "directory containing the Transformers config/tokenizer assets; defaults to "
            "the parent directory of --source (useful when smoke-testing a single Safetensors file)"
        ),
    )
    parser.add_argument("--archive", type=Path, required=True, help="mixed HQ .hq archive to decode")
    parser.add_argument("--output", type=Path, required=True, help="new directory for the sharded surrogate")
    parser.add_argument(
        "--dtype",
        choices=_SURROGATE_DTYPES,
        default="bfloat16",
        help=(
            "surrogate tensor dtype; mixed-fp16-fp32 writes decoded quantized tensors "
            "as FP16 and stored RAW tensors as FP32 for a same-backend GGUF quality control"
        ),
    )
    parser.add_argument("--max-shard-gib", type=_positive_gib, default=2.1, help="maximum output Safetensors shard size")
    parser.add_argument("--overwrite", action="store_true", help="replace prior surrogate shards in --output")
    return parser.parse_args(argv)


def _copy_assets(source_dir: Path, output_dir: Path) -> list[str]:
    copied: list[str] = []
    for name in _RUNTIME_ASSETS:
        source = source_dir / name
        if source.is_file():
            shutil.copy2(source, output_dir / name)
            copied.append(name)
    required = {"config.json", "tokenizer.json", "tokenizer_config.json"}
    missing = sorted(required - set(copied))
    if missing:
        raise FileNotFoundError(f"Source directory lacks required Transformers assets: {missing}")
    return copied


def _remove_previous_output(output_dir: Path, *, overwrite: bool) -> None:
    previous = [*output_dir.glob("model-*.safetensors")]
    index = output_dir / "model.safetensors.index.json"
    manifest = output_dir / "hq-mixed-surrogate.json"
    if (previous or index.exists() or manifest.exists()) and not overwrite:
        raise FileExistsError(f"Output {output_dir} already contains a surrogate; pass --overwrite to replace it")
    for path in previous:
        path.unlink()
    index.unlink(missing_ok=True)
    manifest.unlink(missing_ok=True)


def _close_mapping(value: object) -> None:
    mapping = getattr(value, "_mmap", None)
    if mapping is not None:
        mapping.close()


def _decode_tensor(archive: hq2.HQModel, name: str) -> torch.Tensor:
    """Decode one archive tensor and release every mmap before returning."""

    descriptor = archive.descriptor(name)
    if descriptor.format.name == "RAW":
        raw = archive.raw_tensor(name)
        try:
            return raw.to_torch()
        finally:
            raw.close()
    packed_tensor = None
    if descriptor.format.name in {"HQ2", "HQ3"}:
        packed_tensor = archive.tensor(name)
        value = packed_tensor.dequantize()
    else:
        value = decode_archive_weight(archive, name)
    try:
        return torch.from_numpy(np.ascontiguousarray(value))
    finally:
        del value
        if packed_tensor is not None:
            _close_mapping(packed_tensor.packed)


def _validate_coverage(source_path: Path, archive: hq2.HQModel) -> tuple[str, ...]:
    with safe_open(source_path, framework="pt", device="cpu") as source:
        source_shapes = {name: tuple(int(size) for size in source.get_slice(name).get_shape()) for name in source.keys()}
    archive_names = set(archive.tensor_names)
    source_names = set(source_shapes)
    missing = sorted(source_names - archive_names)
    unexpected = sorted(archive_names - source_names)
    if missing or unexpected:
        raise ValueError(
            "Source/archive tensor coverage differs: "
            f"missing from archive={missing[:3]}, unexpected in archive={unexpected[:3]}"
        )
    mismatches = [
        name for name in archive.tensor_names
        if tuple(archive.descriptor(name).shape) != source_shapes[name]
    ]
    if mismatches:
        name = mismatches[0]
        raise ValueError(
            f"Shape mismatch for {name!r}: archive {archive.descriptor(name).shape}, "
            f"source {source_shapes[name]}"
        )
    return tuple(archive.tensor_names)


def materialize(args: argparse.Namespace) -> Path:
    source_path = args.source.resolve()
    archive_path = args.archive.resolve()
    output_dir = args.output.resolve()
    assets_dir = (args.assets_from or source_path.parent).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    if not assets_dir.is_dir():
        raise NotADirectoryError(assets_dir)
    if output_dir == source_path.parent:
        raise ValueError("Refusing to write a surrogate into the source checkpoint directory")

    archive = hq2.load_model(archive_path)
    names = _validate_coverage(source_path, archive)
    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_previous_output(output_dir, overwrite=args.overwrite)
    assets = _copy_assets(assets_dir, output_dir)

    target_dtype = _TORCH_DTYPES.get(args.dtype)
    max_shard_bytes = int(args.max_shard_gib * 1024**3)
    weight_map: dict[str, str] = {}
    shard: dict[str, torch.Tensor] = {}
    shard_bytes = 0
    output_bytes = 0
    shard_number = 0
    formats: dict[str, dict[str, int]] = {}
    output_dtypes: dict[str, int] = {}

    def flush_shard() -> None:
        nonlocal shard, shard_bytes, shard_number
        if not shard:
            return
        filename = f"model-{shard_number:05d}.safetensors"
        save_file(shard, output_dir / filename, metadata={"format": "pt", "hq_surrogate": "true"})
        for name in shard:
            weight_map[name] = filename
        print(f"[shard {shard_number:02d}] {_human_bytes(shard_bytes)}", flush=True)
        shard_number += 1
        shard = {}
        shard_bytes = 0
        gc.collect()

    for index, name in enumerate(names, start=1):
        descriptor = archive.descriptor(name)
        tensor = _decode_tensor(archive, name)
        if not torch.isfinite(tensor.float()).all():
            raise FloatingPointError(f"Decoded non-finite values for {name!r}")
        tensor_dtype = (
            torch.float32
            if args.dtype == "mixed-fp16-fp32" and descriptor.format.name == "RAW"
            else target_dtype or torch.float16
        )
        tensor = tensor.to(dtype=tensor_dtype).contiguous()
        tensor_bytes = tensor.numel() * tensor.element_size()
        if shard and shard_bytes + tensor_bytes > max_shard_bytes:
            flush_shard()
        shard[name] = tensor
        shard_bytes += tensor_bytes
        output_bytes += tensor_bytes
        format_bucket = formats.setdefault(descriptor.format.name, {"tensors": 0, "values": 0})
        format_bucket["tensors"] += 1
        format_bucket["values"] += tensor.numel()
        dtype_name = str(tensor.dtype).removeprefix("torch.")
        output_dtypes[dtype_name] = output_dtypes.get(dtype_name, 0) + tensor.numel()
        print(f"[{index:03d}/{len(names):03d}] {descriptor.format.name:5s} {name}", flush=True)
        del tensor
        gc.collect()
    flush_shard()

    index = {"metadata": {"total_size": output_bytes, "format": "pt"}, "weight_map": weight_map}
    (output_dir / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "kind": "hq_mixed_quality_surrogate",
        "archive": str(archive_path),
        "archive_file_bytes": archive_path.stat().st_size,
        "archive_metadata": dict(archive.metadata),
        "source": str(source_path),
        "runtime_assets_source": str(assets_dir),
        "source_tensor_count": len(names),
        "surrogate_dtype": args.dtype,
        "surrogate_tensor_bytes": output_bytes,
        "shards": shard_number,
        "max_shard_bytes": max_shard_bytes,
        "runtime_assets": assets,
        "decoded_formats": formats,
        "surrogate_dtypes_by_values": output_dtypes,
        "quality_scope": (
            "Exact packed-weight decode evaluated through a BF16 Transformers surrogate; "
            "not a packed-kernel throughput measurement. RAW FP32 archive tensors are "
            "rounded to the selected surrogate dtype."
        ),
    }
    (output_dir / "hq-mixed-surrogate.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[done] {len(names)} tensors in {shard_number} shards; "
        f"{_human_bytes(output_bytes)} decoded {args.dtype} weights",
        flush=True,
    )
    return output_dir


if __name__ == "__main__":
    materialize(parse_args())
