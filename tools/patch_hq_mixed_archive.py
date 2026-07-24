"""Create a byte-preserving mixed-policy ablation from an existing HQ archive.

The normal mixed quantizer rebuilds every HQ2/HQ3 payload.  That is correct for
a fresh release archive, but it is not an isolated research ablation because
GPU Lloyd refinement can choose slightly different codebooks on a later run.
This tool copies every *unchanged* baseline payload directly into a new archive
and quantizes only exact-name tier overrides from the source checkpoint.

It therefore makes a quality difference attributable to the named policy
change rather than a whole-archive re-quantization.  The output remains a
normal `.hq` archive; the only addition is provenance metadata describing its
byte-preserving parent.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import numpy as np
from safetensors import safe_open


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for import_path in (Path(__file__).resolve().parent, REPOSITORY_ROOT.parent, REPOSITORY_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import hq2
from hq2.mixed_policy import MixedPolicyPlan
from hq2.raw import raw_format
from quantize_hq_mixed_model import (
    _add_hq2_payload,
    _add_hq3_payload,
    _load_imatrix,
    _load_tier_overrides,
    _native_quantize,
    _native_quantizer,
    _normalised_importance,
    _source_plan,
)


def _human_bytes(value: int) -> str:
    current = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if current < 1024.0 or unit == "TiB":
            return f"{current:.2f} {unit}"
        current /= 1024.0
    raise AssertionError("unreachable")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="source Gemma BF16 Safetensors checkpoint")
    parser.add_argument("--baseline", type=Path, required=True, help="baseline .hq archive to copy")
    parser.add_argument("--tier-overrides", type=Path, required=True, help="JSON mapping of exact tensor name to replacement tier")
    parser.add_argument("--output", type=Path, required=True, help="patched output .hq archive")
    parser.add_argument("--imatrix", type=Path, help="activation weighting archive for an overridden HQ2/HQ3 tensor")
    parser.add_argument("--hq2-iters", type=int, default=8, choices=range(1, 17))
    parser.add_argument("--hq2-rows-per-chunk", type=int, default=2048)
    parser.add_argument("--hq2-max-values-per-chunk", type=int, default=2_000_000)
    parser.add_argument("--hq3-iters", type=int, default=4, choices=range(1, 17))
    parser.add_argument("--hq3-rows-per-chunk", type=int, default=512)
    parser.add_argument("--policy-name", default="gemma4-hq2-mixed-patched", help="auditable policy name stored in metadata")
    parser.add_argument("--target-payload-bpw", type=float, default=2.8, help="informational BPW target stored in metadata")
    parser.add_argument("--tmp-dir", type=Path, help="temporary payload directory; defaults beside --output")
    parser.add_argument("--plan-json", type=Path, help="write the exact target policy plan")
    parser.add_argument("--report-json", type=Path, help="write byte-preservation and storage evidence")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output archive")
    return parser.parse_args(argv)


def _verify_baseline(model: hq2.HQModel, baseline_plan: MixedPolicyPlan) -> None:
    baseline_names = set(model.tensor_names)
    planned_names = {entry.name for entry in baseline_plan.tensors}
    if baseline_names != planned_names:
        missing = sorted(planned_names - baseline_names)
        extra = sorted(baseline_names - planned_names)
        raise ValueError(f"Baseline tensor set differs from default policy: missing={missing[:3]}, extra={extra[:3]}")
    mismatches: list[str] = []
    for entry in baseline_plan.tensors:
        descriptor = model.descriptor(entry.name)
        if descriptor.shape != entry.shape or descriptor.format != entry.format:
            mismatches.append(entry.name)
    if mismatches:
        raise ValueError(
            "Baseline does not match the default Gemma HQ mixed policy for "
            f"{len(mismatches)} tensors, e.g. {mismatches[:3]}; patching a different base needs an explicit policy tool"
        )


def _payload_equal(
    first_path: Path,
    first_offset: int,
    second_path: Path,
    second_offset: int,
    nbytes: int,
    *,
    chunk_bytes: int = 8 * 1024 * 1024,
) -> bool:
    """Compare two payload spans without mapping a full model at once."""

    with first_path.open("rb", buffering=chunk_bytes) as first, second_path.open("rb", buffering=chunk_bytes) as second:
        first.seek(first_offset)
        second.seek(second_offset)
        remaining = nbytes
        while remaining:
            count = min(remaining, chunk_bytes)
            if first.read(count) != second.read(count):
                return False
            remaining -= count
    return True


def _verify_copied_payloads(baseline: hq2.HQModel, output: hq2.HQModel, copied_names: list[str]) -> list[str]:
    mismatches: list[str] = []
    for name in copied_names:
        before = baseline.descriptor(name)
        after = output.descriptor(name)
        if before.shape != after.shape or before.format != after.format or before.nbytes != after.nbytes:
            mismatches.append(name)
            continue
        if not _payload_equal(baseline.path, before.offset, output.path, after.offset, before.nbytes):
            mismatches.append(name)
    return mismatches


def _parent_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=8 * 1024 * 1024) as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patch(args: argparse.Namespace) -> Path:
    overrides = _load_tier_overrides(args.tier_overrides)
    if not overrides:
        raise ValueError("--tier-overrides must contain at least one actual override")
    if args.output.resolve() == args.baseline.resolve():
        raise ValueError("--output must differ from --baseline")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}; pass --overwrite to replace it")

    baseline_plan = _source_plan(
        args.input,
        tier_overrides={},
        policy_name="gemma4-hq2-mixed-2p8",
        target_payload_bpw=2.8,
    )
    target_plan = _source_plan(
        args.input,
        tier_overrides=overrides,
        policy_name=args.policy_name,
        target_payload_bpw=args.target_payload_bpw,
    )
    baseline = hq2.load_model(args.baseline)
    _verify_baseline(baseline, baseline_plan)

    changed_names = [entry.name for entry in target_plan.tensors if baseline_plan.tensor(entry.name).tier != entry.tier]
    ignored = sorted(name for name in overrides if name not in changed_names)
    if ignored:
        raise ValueError(f"Overrides that do not change the baseline tier are not meaningful: {ignored}")
    imatrix = _load_imatrix(args.imatrix, target_plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.plan_json is not None:
        args.plan_json.parent.mkdir(parents=True, exist_ok=True)
        args.plan_json.write_text(json.dumps(target_plan.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    parent_sha256 = _parent_sha256(args.baseline)
    metadata: dict[str, Any] = dict(baseline.metadata)
    metadata.update(
        {
            "policy": target_plan.name,
            "target_payload_bpw": target_plan.target_payload_bpw,
            "planned_payload_bpw": target_plan.payload_bits_per_weight,
            "tier_overrides": overrides,
            "patch_mode": "byte-preserving baseline payload copy; only named tier overrides are requantized",
            "parent_archive": args.baseline.name,
            "parent_archive_sha256": parent_sha256,
            "patched_tensors": changed_names,
        }
    )
    tmp_parent = args.tmp_dir if args.tmp_dir is not None else args.output.parent
    tmp_parent.mkdir(parents=True, exist_ok=True)
    quantizer = None
    copied_names: list[str] = []
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="hq2-patch-", dir=tmp_parent) as temp_name:
        temp_dir = Path(temp_name)
        with hq2.HQModelWriter(args.output, metadata=metadata) as writer:
            with safe_open(args.input, framework="pt", device="cpu") as source:
                for index, entry in enumerate(target_plan.tensors, start=1):
                    baseline_entry = baseline_plan.tensor(entry.name)
                    if entry.tier == baseline_entry.tier:
                        original = baseline.descriptor(entry.name)
                        writer.add_raw(
                            entry.name,
                            baseline.payload(entry.name),
                            shape=original.shape,
                            format=original.format,
                            iterations=original.iterations,
                            importance_weighted=original.importance_weighted,
                        )
                        copied_names.append(entry.name)
                        action = "copy"
                    elif entry.tier == "hq2":
                        _add_hq2_payload(
                            writer,
                            source=source,
                            entry=entry,
                            temp_dir=temp_dir,
                            rows_per_chunk=args.hq2_rows_per_chunk,
                            max_values_per_chunk=args.hq2_max_values_per_chunk,
                            iterations=args.hq2_iters,
                            importance_vector=None if imatrix is None else _normalised_importance(imatrix[entry.name]),
                        )
                        action = "requantize-hq2"
                    elif entry.tier == "hq3":
                        _add_hq3_payload(
                            writer,
                            source=source,
                            entry=entry,
                            temp_dir=temp_dir,
                            rows_per_chunk=args.hq3_rows_per_chunk,
                            iterations=args.hq3_iters,
                            importance_vector=None if imatrix is None or entry.name not in imatrix else _normalised_importance(imatrix[entry.name]),
                        )
                        action = "requantize-hq3"
                    else:
                        tensor = source.get_tensor(entry.name)
                        values = None
                        payload = None
                        try:
                            if entry.tier == "f32":
                                payload = np.ascontiguousarray(tensor.float().numpy()).view(np.uint8)
                                writer.add_raw(entry.name, payload, shape=entry.shape, format=raw_format("float32"))
                            else:
                                if quantizer is None:
                                    quantizer = _native_quantizer()
                                values = np.ascontiguousarray(tensor.float().numpy(), dtype=np.float32)
                                payload = _native_quantize(values, tier=entry.tier, quantizer=quantizer)
                                writer.add_raw(entry.name, payload, shape=entry.shape, format=entry.format)
                            action = f"requantize-{entry.tier}"
                        finally:
                            del tensor
                            del values, payload
                    gc.collect()
                    try:
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except ImportError:
                        pass
                    print(f"[{index:03d}/{len(target_plan.tensors):03d}] {action:16s} {entry.name}", flush=True)

    output = hq2.load_model(args.output)
    actual_bpw = hq2.analyze_model(args.output).storage["payload_bits_per_weight"]
    if abs(float(actual_bpw) - target_plan.payload_bits_per_weight) > 1e-12:
        raise RuntimeError(f"Archive payload BPW {actual_bpw} differs from plan {target_plan.payload_bits_per_weight}")
    mismatches = _verify_copied_payloads(baseline, output, copied_names)
    if mismatches:
        raise RuntimeError(f"Byte-preserving patch verification failed for {len(mismatches)} copied tensors, e.g. {mismatches[:3]}")
    report = {
        "baseline": str(args.baseline),
        "baseline_sha256": parent_sha256,
        "output": str(args.output),
        "tier_overrides": overrides,
        "requantized_tensors": changed_names,
        "copied_tensors": len(copied_names),
        "copied_payload_verification": "passed",
        "payload_bpw": float(actual_bpw),
        "payload_bytes": int(output.path.stat().st_size) - int(hq2.analyze_model(args.output).storage["container_overhead_bytes"]),
        "elapsed_seconds": time.perf_counter() - started,
    }
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"Patched {len(changed_names)} tensors; copied {len(copied_names)} byte-identically. "
        f"Wrote {args.output} ({float(actual_bpw):.6f} payload BPW) in {report['elapsed_seconds']:.1f}s.",
        flush=True,
    )
    return args.output


def main(argv: list[str] | None = None) -> int:
    patch(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
