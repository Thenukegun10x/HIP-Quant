"""Stream a Gemma 4 checkpoint into the HQ2-Mixed-2.8 archive policy.

The produced archive contains every source tensor: the fast text-MLP path is
HQ2, V and the tied embedding are HQ3, Q/K/O are Q4_0, uncommon 2-D tensors
are Q8_0, and remaining small/stability tensors are preserved as FP32.  The
converter reads one source tensor (or one HQ3 row chunk) at a time; it never
loads the 24-GB BF16 checkpoint as a whole.

The archive is a storage artifact with exact tier metadata and analyzer
support.  Direct packed inference remains implemented for HQ2/HQ3 only; Q4_0
and Q8_0 need their own packed PyTorch/Vulkan dispatch before this policy can
make an end-to-end throughput claim.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from safetensors import safe_open


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for import_path in (REPOSITORY_ROOT.parent, REPOSITORY_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import hq2
from hq2.mixed_policy import plan_gemma4_hq2_2p8
from hq2.raw import raw_format


def _human_bytes(value: int) -> str:
    current = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if current < 1024.0 or unit == "TiB":
            return f"{current:.2f} {unit}"
        current /= 1024.0
    raise AssertionError("unreachable")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Gemma 4 BF16 Safetensors checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="output .hq/.hq2 archive")
    parser.add_argument("--imatrix", type=Path, help="optional [in_features] activation weights for HQ2 MLP tensors")
    parser.add_argument("--hq2-iters", type=int, default=8, choices=range(1, 17))
    parser.add_argument(
        "--hq2-backend",
        choices=("native", "torch", "native-full"),
        default="native",
        help=(
            "HQ2 conversion backend. 'native' is the bounded native-HIP default; "
            "'torch' is a portability diagnostic; 'native-full' is unsafe for full imatrix-weighted matrices."
        ),
    )
    parser.add_argument("--hq2-rows-per-chunk", type=int, default=2048)
    parser.add_argument(
        "--hq2-max-values-per-chunk",
        type=int,
        default=2_000_000,
        help="upper bound for one Torch HQ2 chunk; caps row chunks for wide matrices",
    )
    parser.add_argument("--hq3-iters", type=int, default=4, choices=range(1, 17))
    parser.add_argument("--hq3-rows-per-chunk", type=int, default=512)
    parser.add_argument(
        "--tier-overrides",
        type=Path,
        help="JSON object mapping exact source tensor names to hq2/hq3/q4_0/q8_0/f32 tiers",
    )
    parser.add_argument(
        "--policy-name",
        default="gemma4-hq2-mixed-2p8",
        help="auditable policy name written into the archive metadata",
    )
    parser.add_argument(
        "--target-payload-bpw",
        type=float,
        default=2.8,
        help="informational BPW target written into the plan and archive metadata",
    )
    parser.add_argument("--tmp-dir", type=Path, help="temporary HQ3 payload directory; defaults beside the output")
    parser.add_argument("--plan-json", type=Path, help="write the exact pre-conversion plan JSON")
    parser.add_argument("--dry-run", action="store_true", help="validate and print the policy without reading values")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output archive")
    return parser.parse_args(argv)


def _load_tier_overrides(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or not all(isinstance(name, str) and isinstance(tier, str) for name, tier in loaded.items()):
        raise ValueError("--tier-overrides must contain a JSON object of tensor-name to tier strings")
    return dict(loaded)


def _source_plan(
    checkpoint: Path,
    *,
    tier_overrides: dict[str, str],
    policy_name: str,
    target_payload_bpw: float,
):
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    with safe_open(checkpoint, framework="pt", device="cpu") as source:
        entries = [(name, tuple(source.get_slice(name).get_shape())) for name in source.keys()]
    return plan_gemma4_hq2_2p8(
        entries,
        tier_overrides=tier_overrides,
        policy_name=policy_name,
        target_payload_bpw=target_payload_bpw,
    )


def _load_imatrix(path: Path | None, plan) -> dict[str, np.ndarray] | None:
    hq2_names = [item.name for item in plan.tensors if item.tier == "hq2"]
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in hq2_names if name not in archive.files]
        if missing:
            raise KeyError(f"Importance archive misses {len(missing)} HQ2 tensors, e.g. {missing[:2]}")
        # HQ2 vectors are required. HQ3 vectors are optional: the token
        # embedding has no Linear input activation, while attention V does.
        candidate_names = [item.name for item in plan.tensors if item.tier in {"hq2", "hq3"}]
        vectors = {
            name: np.ascontiguousarray(archive[name], dtype=np.float32)
            for name in candidate_names
            if name in archive.files
        }
    for entry in (plan.tensor(name) for name in vectors):
        expected = (entry.shape[-1],)
        if vectors[entry.name].shape != expected:
            raise ValueError(f"Importance for {entry.name!r} has {vectors[entry.name].shape}; expected {expected}")
    return vectors


def _normalised_importance(vector: np.ndarray) -> np.ndarray:
    maximum = float(vector.max())
    normalised = vector if maximum <= 0.0 else vector / maximum
    return np.ascontiguousarray(normalised, dtype=np.float32)


def _native_quantize(values: np.ndarray, *, tier: str, quantizer):
    """Use the native DLL only for non-HQ packed baseline codecs.

    The HQ2 native imatrix route is deliberately not used by default: a
    full-layer probe can access-violate on Windows/ROCm.  Torch HQ2 below
    keeps the same portable bytes while bounding live allocations.
    """

    if tier not in {"q4_0", "q8_0"}:
        raise ValueError(f"Native mixed-policy conversion does not support {tier!r}")
    from hip_quant import GGML_TYPE

    type_name = {"q4_0": "Q4_0", "q8_0": "Q8_0"}[tier]
    return quantizer.quantize_numpy(values, GGML_TYPE[type_name])


def _native_hq2_quantize(values: np.ndarray, *, importance: np.ndarray | None, iterations: int, quantizer):
    """Explicitly requested legacy native HQ2 route; use only for diagnosis."""

    from hip_quant import GGML_TYPE

    return quantizer.quantize_numpy(
        values,
        GGML_TYPE["HQ2"],
        imatrix=importance,
        hq2_iterations=iterations,
    )


def _native_quantizer():
    """Lazily load the native DLL only when a native codec is selected."""

    from hip_quant import get_hip_quant

    return get_hip_quant()


def _write_hq2_stream(
    *,
    source,
    entry,
    destination: Path,
    rows_per_chunk: int,
    max_values_per_chunk: int,
    iterations: int,
    importance_vector: np.ndarray | None,
) -> None:
    """Quantize one HQ2 matrix in safe bounded Torch/ROCm chunks.

    The imatrix is a per-input-channel vector.  It is normalised once, then
    expanded only for the current row chunk so no full [out, in] importance
    matrix ever exists in host or GPU memory.
    """

    if rows_per_chunk <= 0:
        raise ValueError("--hq2-rows-per-chunk must be positive")
    if max_values_per_chunk <= 0:
        raise ValueError("--hq2-max-values-per-chunk must be positive")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Torch HQ2 stream conversion requires a visible Torch GPU")
    rows, width = entry.shape
    rows_per_chunk = min(rows_per_chunk, max(1, max_values_per_chunk // width))
    source_slice = source.get_slice(entry.name)
    expected = entry.payload_bytes
    written = 0
    device_vector = None
    if importance_vector is not None:
        device_vector = torch.from_numpy(importance_vector).to(device="cuda", dtype=torch.float32)
    try:
        with destination.open("wb") as file:
            for start in range(0, rows, rows_per_chunk):
                stop = min(rows, start + rows_per_chunk)
                block = source_slice[start:stop].to(device="cuda", non_blocking=False)
                importance = None
                if device_vector is not None:
                    importance = device_vector.expand(stop - start, width).contiguous()
                packed = hq2.quantize(
                    block,
                    importance=importance,
                    backend="torch",
                    format="hq2",
                    iterations=iterations,
                )
                payload = packed.numpy()
                file.write(payload.tobytes(order="C"))
                written += payload.nbytes
                del block, importance, packed, payload
                torch.cuda.empty_cache()
                gc.collect()
                print(f"[hq2] {entry.name} rows {stop:,}/{rows:,}", flush=True)
    finally:
        if device_vector is not None:
            del device_vector
        torch.cuda.empty_cache()
    if written != expected or destination.stat().st_size != expected:
        raise RuntimeError(f"HQ2 payload for {entry.name} is {written} bytes; expected {expected}")


def _write_hq2_native_stream(
    *,
    source,
    entry,
    destination: Path,
    rows_per_chunk: int,
    max_values_per_chunk: int,
    iterations: int,
    importance_vector: np.ndarray | None,
    quantizer,
) -> None:
    """Use the validated native HQ2 kernel without ever sending a full layer.

    Native HIP matches the CPU reference for heavily skewed imatrix inputs.
    A full 59M-weight matrix can nevertheless access-violate on Windows, so
    this path caps each native call to a bounded row chunk.
    """

    if rows_per_chunk <= 0:
        raise ValueError("--hq2-rows-per-chunk must be positive")
    if max_values_per_chunk <= 0:
        raise ValueError("--hq2-max-values-per-chunk must be positive")
    rows, width = entry.shape
    rows_per_chunk = min(rows_per_chunk, max(1, max_values_per_chunk // width))
    source_slice = source.get_slice(entry.name)
    expected = entry.payload_bytes
    written = 0
    from hip_quant import GGML_TYPE

    with destination.open("wb") as file:
        for start in range(0, rows, rows_per_chunk):
            stop = min(rows, start + rows_per_chunk)
            values = np.ascontiguousarray(source_slice[start:stop].float().numpy(), dtype=np.float32)
            importance = None
            if importance_vector is not None:
                importance = np.ascontiguousarray(np.broadcast_to(importance_vector, values.shape))
            payload = quantizer.quantize_numpy(
                values,
                GGML_TYPE["HQ2"],
                imatrix=importance,
                hq2_iterations=iterations,
            )
            file.write(payload.tobytes(order="C"))
            written += payload.nbytes
            del values, importance, payload
            gc.collect()
            print(f"[hq2-native] {entry.name} rows {stop:,}/{rows:,}", flush=True)
    if written != expected or destination.stat().st_size != expected:
        raise RuntimeError(f"Native HQ2 payload for {entry.name} is {written} bytes; expected {expected}")


def _write_hq3_stream(
    *,
    source,
    entry,
    destination: Path,
    rows_per_chunk: int,
    iterations: int,
    importance_vector: np.ndarray | None,
) -> None:
    """Quantize one large HQ3 tensor in bounded GPU chunks into a temp payload."""

    if rows_per_chunk <= 0:
        raise ValueError("--hq3-rows-per-chunk must be positive")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("HQ3 stream conversion requires a visible Torch GPU")
    rows, width = entry.shape
    source_slice = source.get_slice(entry.name)
    expected = entry.payload_bytes
    written = 0
    device_vector = None
    if importance_vector is not None:
        device_vector = torch.from_numpy(importance_vector).to(device="cuda", dtype=torch.float32)
    try:
        with destination.open("wb") as file:
            for start in range(0, rows, rows_per_chunk):
                stop = min(rows, start + rows_per_chunk)
                block = source_slice[start:stop].to(device="cuda", non_blocking=False)
                importance = None
                if device_vector is not None:
                    importance = device_vector.expand(stop - start, width).contiguous()
                packed = hq2.quantize(
                    block,
                    importance=importance,
                    backend="torch",
                    format="hq3",
                    iterations=iterations,
                )
                payload = packed.numpy()
                file.write(payload.tobytes(order="C"))
                written += payload.nbytes
                del block, importance, packed, payload
                torch.cuda.empty_cache()
                gc.collect()
                print(f"[hq3] {entry.name} rows {stop:,}/{rows:,}", flush=True)
    finally:
        if device_vector is not None:
            del device_vector
        torch.cuda.empty_cache()
    if written != expected or destination.stat().st_size != expected:
        raise RuntimeError(f"HQ3 payload for {entry.name} is {written} bytes; expected {expected}")


def _add_hq3_payload(
    writer,
    *,
    source,
    entry,
    temp_dir: Path,
    rows_per_chunk: int,
    iterations: int,
    importance_vector: np.ndarray | None,
) -> None:
    temporary = temp_dir / (entry.name.replace(".", "_") + ".hq3.tmp")
    _write_hq3_stream(
        source=source,
        entry=entry,
        destination=temporary,
        rows_per_chunk=rows_per_chunk,
        iterations=iterations,
        importance_vector=importance_vector,
    )
    blocks = entry.values // entry.format.block_size
    payload = np.memmap(temporary, mode="r", dtype=np.uint8, shape=(blocks, entry.format.block_bytes), order="C")
    try:
        writer.add_raw(
            entry.name,
            payload,
            shape=entry.shape,
            format=entry.format,
            iterations=iterations,
            importance_weighted=importance_vector is not None,
        )
    finally:
        mapping = getattr(payload, "_mmap", None)
        del payload
        if mapping is not None:
            mapping.close()
        temporary.unlink(missing_ok=True)


def _add_hq2_payload(
    writer,
    *,
    source,
    entry,
    temp_dir: Path,
    rows_per_chunk: int,
    max_values_per_chunk: int,
    iterations: int,
    importance_vector: np.ndarray | None,
) -> None:
    temporary = temp_dir / (entry.name.replace(".", "_") + ".hq2.tmp")
    _write_hq2_stream(
        source=source,
        entry=entry,
        destination=temporary,
        rows_per_chunk=rows_per_chunk,
        max_values_per_chunk=max_values_per_chunk,
        iterations=iterations,
        importance_vector=importance_vector,
    )
    blocks = entry.values // entry.format.block_size
    payload = np.memmap(temporary, mode="r", dtype=np.uint8, shape=(blocks, entry.format.block_bytes), order="C")
    try:
        writer.add_raw(
            entry.name,
            payload,
            shape=entry.shape,
            format=entry.format,
            iterations=iterations,
            importance_weighted=importance_vector is not None,
        )
    finally:
        mapping = getattr(payload, "_mmap", None)
        del payload
        if mapping is not None:
            mapping.close()
        temporary.unlink(missing_ok=True)


def _add_hq2_native_payload(
    writer,
    *,
    source,
    entry,
    temp_dir: Path,
    rows_per_chunk: int,
    max_values_per_chunk: int,
    iterations: int,
    importance_vector: np.ndarray | None,
    quantizer,
) -> None:
    temporary = temp_dir / (entry.name.replace(".", "_") + ".hq2.native.tmp")
    _write_hq2_native_stream(
        source=source,
        entry=entry,
        destination=temporary,
        rows_per_chunk=rows_per_chunk,
        max_values_per_chunk=max_values_per_chunk,
        iterations=iterations,
        importance_vector=importance_vector,
        quantizer=quantizer,
    )
    blocks = entry.values // entry.format.block_size
    payload = np.memmap(temporary, mode="r", dtype=np.uint8, shape=(blocks, entry.format.block_bytes), order="C")
    try:
        writer.add_raw(
            entry.name,
            payload,
            shape=entry.shape,
            format=entry.format,
            iterations=iterations,
            importance_weighted=importance_vector is not None,
        )
    finally:
        mapping = getattr(payload, "_mmap", None)
        del payload
        if mapping is not None:
            mapping.close()
        temporary.unlink(missing_ok=True)


def convert(args: argparse.Namespace) -> Path:
    tier_overrides = _load_tier_overrides(args.tier_overrides)
    plan = _source_plan(
        args.input,
        tier_overrides=tier_overrides,
        policy_name=args.policy_name,
        target_payload_bpw=args.target_payload_bpw,
    )
    print(
        f"{plan.name}: {_human_bytes(plan.payload_bytes)} / {plan.logical_value_count:,} values "
        f"= {plan.payload_bits_per_weight:.6f} payload BPW",
        flush=True,
    )
    for tier, item in plan.tier_summary.items():
        print(
            f"  {tier:5s} {int(item['tensors']):3d} tensors, {int(item['values']):,} values, "
            f"{_human_bytes(int(item['payload_bytes']))}",
            flush=True,
        )
    if args.plan_json is not None:
        args.plan_json.parent.mkdir(parents=True, exist_ok=True)
        args.plan_json.write_text(json.dumps(plan.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.dry_run:
        return args.output
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}; pass --overwrite to replace it")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    imatrix = _load_imatrix(args.imatrix, plan)
    quantizer = None
    metadata = {
        "architecture": "gemma4",
        "quantization": "HQ2-Mixed-2.8",
        "policy": plan.name,
        "target_payload_bpw": plan.target_payload_bpw,
        "planned_payload_bpw": plan.payload_bits_per_weight,
        "tier_overrides": tier_overrides,
        "source_checkpoint": args.input.name,
        "hq2_iterations": args.hq2_iters,
        "hq2_backend": args.hq2_backend,
        "hq2_rows_per_chunk": args.hq2_rows_per_chunk if args.hq2_backend != "native-full" else None,
        "hq2_max_values_per_chunk": args.hq2_max_values_per_chunk if args.hq2_backend != "native-full" else None,
        "hq3_iterations": args.hq3_iters,
        "hq2_imatrix": None if args.imatrix is None else args.imatrix.name,
        "hq3_imatrix_tensors": 0 if imatrix is None else sum(
            entry.tier == "hq3" and entry.name in imatrix for entry in plan.tensors
        ),
        "tier_summary": plan.tier_summary,
        "inference_status": "HQ2/HQ3 packed paths are implemented; Q4_0/Q8_0 packed dispatch remains pending.",
    }
    tmp_parent = args.tmp_dir if args.tmp_dir is not None else args.output.parent
    tmp_parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="hq2-mixed-", dir=tmp_parent) as tmp_name:
        temp_dir = Path(tmp_name)
        with hq2.HQModelWriter(args.output, metadata=metadata) as writer:
            with safe_open(args.input, framework="pt", device="cpu") as source:
                for index, entry in enumerate(plan.tensors, start=1):
                    tensor = None
                    if entry.tier == "hq2" and args.hq2_backend == "torch":
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
                    elif entry.tier == "hq2" and args.hq2_backend == "native":
                        if quantizer is None:
                            quantizer = _native_quantizer()
                        _add_hq2_native_payload(
                            writer,
                            source=source,
                            entry=entry,
                            temp_dir=temp_dir,
                            rows_per_chunk=args.hq2_rows_per_chunk,
                            max_values_per_chunk=args.hq2_max_values_per_chunk,
                            iterations=args.hq2_iters,
                            importance_vector=None if imatrix is None else _normalised_importance(imatrix[entry.name]),
                            quantizer=quantizer,
                        )
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
                    else:
                        tensor = source.get_tensor(entry.name)
                        if entry.tier == "f32":
                            payload = np.ascontiguousarray(tensor.float().numpy()).view(np.uint8)
                            writer.add_raw(entry.name, payload, shape=entry.shape, format=raw_format("float32"))
                        else:
                            values = np.ascontiguousarray(tensor.float().numpy(), dtype=np.float32)
                            importance = None
                            if quantizer is None:
                                quantizer = _native_quantizer()
                            if entry.tier == "hq2":
                                # Explicit legacy mode: materialises the full importance
                                # matrix, which is why the bounded Torch route is default.
                                if imatrix is not None:
                                    vector = _normalised_importance(imatrix[entry.name])
                                    importance = np.ascontiguousarray(np.broadcast_to(vector, values.shape))
                                payload = _native_hq2_quantize(
                                    values,
                                    importance=importance,
                                    iterations=args.hq2_iters,
                                    quantizer=quantizer,
                                )
                                writer.add_raw(
                                    entry.name,
                                    payload,
                                    shape=entry.shape,
                                    format=entry.format,
                                    iterations=args.hq2_iters,
                                    importance_weighted=importance is not None,
                                )
                            else:
                                payload = _native_quantize(values, tier=entry.tier, quantizer=quantizer)
                                writer.add_raw(entry.name, payload, shape=entry.shape, format=entry.format)
                            del values, importance, payload
                    if tensor is not None:
                        del tensor
                    gc.collect()
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except ImportError:
                        pass
                    elapsed = time.perf_counter() - start
                    print(f"[{index:03d}/{len(plan.tensors):03d}] {entry.tier:5s} {entry.name} ({elapsed:.1f}s)", flush=True)
    analysis = hq2.analyze_model(args.output)
    actual = analysis.storage["payload_bits_per_weight"]
    if abs(float(actual) - plan.payload_bits_per_weight) > 1e-12:
        raise RuntimeError(f"Archive payload BPW {actual} differs from plan {plan.payload_bits_per_weight}")
    print(f"Wrote {args.output} in {time.perf_counter() - start:.1f}s ({float(actual):.6f} payload BPW)", flush=True)
    return args.output


def main(argv: list[str] | None = None) -> int:
    convert(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
