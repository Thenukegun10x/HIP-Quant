"""Calibration-weighted error analysis for mixed HQ-family archives.

Weight MSE alone is a poor mixed-precision policy signal: an error in a quiet
input channel is much less consequential than the same error in a channel
that receives high-energy activations.  For a linear layer ``y = Wx`` and a
calibration diagonal ``h_j = sum_t x[t, j]^2``, this module measures the
diagonal-Hessian approximation::

    sum_{i,j} (W[i,j] - Wq[i,j])^2 * h[j]

This is exactly the output squared-error contribution under the diagonal
activation approximation.  It is a reliable *screening* metric for choosing
which tensors deserve an expensive teacher-forced ablation; it does not
replace that paired end-to-end validation because layer errors can interact.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .archive import HQ8_G128_FORMAT, HQModel, HQFormatDescriptor, load_model
from .hq8 import decode_hq8_g128_numpy
from .mixed_policy import Q4_0_FORMAT, Q8_0_FORMAT


def _shape2(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) != 2:
        raise ValueError(f"Mixed linear decoder requires a rank-2 tensor, got {shape}")
    return int(shape[0]), int(shape[1])


def decode_q4_0_numpy(packed: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Decode the canonical GGML Q4_0 layout used in an HQ mixed archive."""

    rows, columns = _shape2(shape)
    if columns % 32:
        raise ValueError(f"Q4_0 needs a 32-aligned final dimension, got {shape}")
    blocks = np.ascontiguousarray(packed, dtype=np.uint8).reshape(-1, Q4_0_FORMAT.block_bytes)
    expected = rows * (columns // Q4_0_FORMAT.block_size)
    if blocks.shape[0] != expected:
        raise ValueError(f"Q4_0 has {blocks.shape[0]} blocks; shape {shape} requires {expected}")
    scale = blocks[:, :2].copy().view("<f2").reshape(-1, 1).astype(np.float32)
    codes = blocks[:, 2:]
    # GGML stores values 0..15 in low nibbles and values 16..31 in high
    # nibbles of the same 16-byte field.
    signed = np.concatenate((codes & 0x0F, codes >> 4), axis=1).astype(np.float32) - 8.0
    return (scale * signed).reshape(shape)


def decode_q8_0_numpy(packed: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Decode the canonical GGML Q8_0 layout used in an HQ mixed archive."""

    rows, columns = _shape2(shape)
    if columns % 32:
        raise ValueError(f"Q8_0 needs a 32-aligned final dimension, got {shape}")
    blocks = np.ascontiguousarray(packed, dtype=np.uint8).reshape(-1, Q8_0_FORMAT.block_bytes)
    expected = rows * (columns // Q8_0_FORMAT.block_size)
    if blocks.shape[0] != expected:
        raise ValueError(f"Q8_0 has {blocks.shape[0]} blocks; shape {shape} requires {expected}")
    scale = blocks[:, :2].copy().view("<f2").reshape(-1, 1).astype(np.float32)
    values = blocks[:, 2:].view(np.int8).astype(np.float32)
    return (scale * values).reshape(shape)


def decode_archive_weight(archive: HQModel, name: str) -> np.ndarray:
    """Decode one implemented mixed-archive tensor to a CPU float32 array."""

    descriptor = archive.descriptor(name)
    format = descriptor.format
    if format.name in {"HQ2", "HQ3"}:
        return np.ascontiguousarray(archive.tensor(name).dequantize(), dtype=np.float32)
    payload = archive.payload(name)
    try:
        if format == Q4_0_FORMAT:
            return decode_q4_0_numpy(payload, descriptor.shape)
        if format == Q8_0_FORMAT:
            return decode_q8_0_numpy(payload, descriptor.shape)
        if format == HQ8_G128_FORMAT:
            return decode_hq8_g128_numpy(payload, descriptor.shape)
    finally:
        mapping = getattr(payload, "_mmap", None)
        del payload
        if mapping is not None:
            mapping.close()
    raise NotImplementedError(f"No mixed-archive decoder for {format.name} v{format.version}")


@dataclass(frozen=True)
class TensorError:
    """One tensor's source-to-packed error and calibration status."""

    name: str
    format: str
    shape: tuple[int, ...]
    values: int
    payload_bytes: int
    payload_bpw: float
    mse: float
    relative_mse: float
    activation_weighted_sse: float | None
    activation_relative_sse: float | None
    error_per_payload_byte: float | None
    calibration: str

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["shape"] = list(self.shape)
        return result


def tensor_error(
    source: np.ndarray,
    restored: np.ndarray,
    *,
    name: str,
    format: HQFormatDescriptor,
    payload_bytes: int,
    importance: np.ndarray | None,
) -> TensorError:
    """Measure raw and activation-weighted error for one rank-2 weight."""

    source = np.ascontiguousarray(source, dtype=np.float32)
    restored = np.ascontiguousarray(restored, dtype=np.float32)
    if source.shape != restored.shape or source.ndim != 2:
        raise ValueError(f"Source/restored must be matching rank-2 arrays, got {source.shape} and {restored.shape}")
    difference = source - restored
    squared_error = np.square(difference, dtype=np.float32)
    source_squared = np.square(source, dtype=np.float32)
    values = int(source.size)
    mse = float(np.mean(squared_error, dtype=np.float64))
    relative_mse = mse / max(float(np.mean(source_squared, dtype=np.float64)), np.finfo(np.float32).tiny)
    weighted_sse = None
    weighted_relative = None
    density = None
    calibration = "unavailable"
    if importance is not None:
        vector = np.ascontiguousarray(importance, dtype=np.float32)
        if vector.shape != (source.shape[1],):
            raise ValueError(f"Importance for {name!r} has shape {vector.shape}; expected {(source.shape[1],)}")
        if not np.all(np.isfinite(vector)) or np.any(vector < 0.0):
            raise ValueError(f"Importance for {name!r} must be finite and non-negative")
        weighted_sse = float(np.sum(squared_error * vector[None, :], dtype=np.float64))
        source_sse = float(np.sum(source_squared * vector[None, :], dtype=np.float64))
        weighted_relative = weighted_sse / max(source_sse, np.finfo(np.float64).tiny)
        density = weighted_sse / payload_bytes
        calibration = "activation-diagonal"
    return TensorError(
        name=name,
        format=format.name,
        shape=tuple(int(value) for value in source.shape),
        values=values,
        payload_bytes=int(payload_bytes),
        payload_bpw=float(format.bits_per_weight),
        mse=mse,
        relative_mse=relative_mse,
        activation_weighted_sse=weighted_sse,
        activation_relative_sse=weighted_relative,
        error_per_payload_byte=density,
        calibration=calibration,
    )


def analyze_mixed_archive_error(
    source_path: str | Path,
    archive_path: str | Path,
    *,
    imatrix_path: str | Path | None = None,
) -> list[TensorError]:
    """Scan a source checkpoint and mixed archive one tensor at a time.

    Only rank-2 tensors with a known packed codec are scored.  RAW/F32 and
    non-matrix tensors are intentionally omitted: their quantization error is
    zero or they are outside linear output-error analysis.
    """

    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - package extra guard
        raise RuntimeError("Mixed error analysis requires safetensors") from exc

    source_path = Path(source_path)
    archive = load_model(archive_path)
    importance: Mapping[str, np.ndarray] = {}
    if imatrix_path is not None:
        with np.load(Path(imatrix_path), allow_pickle=False) as stored:
            importance = {name: np.ascontiguousarray(stored[name], dtype=np.float32) for name in stored.files}

    results: list[TensorError] = []
    with safe_open(source_path, framework="pt", device="cpu") as source:
        source_names = set(source.keys())
        missing = [name for name in archive.tensor_names if name not in source_names]
        if missing:
            raise KeyError(f"Archive tensors are absent from source, e.g. {missing[:3]}")
        for name in archive.tensor_names:
            descriptor = archive.descriptor(name)
            if len(descriptor.shape) != 2 or descriptor.format.name == "RAW":
                continue
            restored = decode_archive_weight(archive, name)
            source_weight = source.get_tensor(name).float().numpy()
            try:
                results.append(
                    tensor_error(
                        source_weight,
                        restored,
                        name=name,
                        format=descriptor.format,
                        payload_bytes=descriptor.nbytes,
                        importance=importance.get(name),
                    )
                )
            finally:
                del restored, source_weight
    return results


def summarize_tensor_errors(results: list[TensorError]) -> dict[str, Any]:
    """Produce stable, JSON-ready rankings for a sensitivity report."""

    calibrated = [result for result in results if result.activation_weighted_sse is not None]
    by_format: dict[str, dict[str, float | int]] = {}
    for result in results:
        bucket = by_format.setdefault(result.format, {"tensors": 0, "values": 0, "payload_bytes": 0, "mse_sum": 0.0})
        bucket["tensors"] = int(bucket["tensors"]) + 1
        bucket["values"] = int(bucket["values"]) + result.values
        bucket["payload_bytes"] = int(bucket["payload_bytes"]) + result.payload_bytes
        bucket["mse_sum"] = float(bucket["mse_sum"]) + result.mse * result.values
    for bucket in by_format.values():
        bucket["mean_mse"] = float(bucket.pop("mse_sum")) / int(bucket["values"])
    return {
        "scored_tensors": len(results),
        "calibrated_tensors": len(calibrated),
        "by_format": dict(sorted(by_format.items())),
        "ranked_by_activation_error": [item.as_dict() for item in sorted(
            calibrated, key=lambda item: float(item.activation_weighted_sse), reverse=True
        )],
        "ranked_by_error_per_payload_byte": [item.as_dict() for item in sorted(
            calibrated, key=lambda item: float(item.error_per_payload_byte), reverse=True
        )],
        "uncalibrated": [item.as_dict() for item in results if item.activation_weighted_sse is None],
    }


__all__ = [
    "TensorError",
    "analyze_mixed_archive_error",
    "decode_archive_weight",
    "decode_q4_0_numpy",
    "decode_q8_0_numpy",
    "summarize_tensor_errors",
    "tensor_error",
]
