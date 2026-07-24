"""Portable HQ3 learned-codebook format.

HQ3 keeps HQ2's simple row-major 256-value block contract but upgrades every
block from four to eight learned FP16 centroids.  Its 256 three-bit selectors
occupy 96 bytes, so the physical block is 112 bytes / 3.5 bpw:

``[8 * fp16 centroids][256 * u3 selectors, LSB-first packed 8 values/3 bytes]``.

The bytes are backend-neutral.  The Torch implementation is deliberately
ordinary tensor math so the format is usable on CPU, CUDA, and ROCm before a
specialised device quantizer is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .backends import BackendUnavailable, _torch, _torch_name


HQ3_BLOCK_SIZE = 256
HQ3_CENTROID_COUNT = 8
HQ3_CENTROID_BYTES = HQ3_CENTROID_COUNT * 2
HQ3_INDEX_BYTES = HQ3_BLOCK_SIZE * 3 // 8
HQ3_BLOCK_BYTES = HQ3_CENTROID_BYTES + HQ3_INDEX_BYTES
HQ3_BITS_PER_WEIGHT = HQ3_BLOCK_BYTES * 8 / HQ3_BLOCK_SIZE
HQ3_FORMAT_NAME = "HQ3"
HQ3_FORMAT_VERSION = 1


def validate_hq3_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    shape = tuple(int(value) for value in shape)
    if not shape or any(value <= 0 for value in shape):
        raise ValueError(f"HQ3 requires a non-empty positive shape, got {shape}")
    if shape[-1] % HQ3_BLOCK_SIZE:
        raise ValueError(
            f"HQ3 requires the final dimension to be divisible by {HQ3_BLOCK_SIZE}, got {shape[-1]}"
        )
    return shape


def hq3_block_count(shape: tuple[int, ...]) -> int:
    shape = validate_hq3_shape(shape)
    return int(np.prod(shape, dtype=np.int64) // HQ3_BLOCK_SIZE)


def hq3_packed_nbytes(shape: tuple[int, ...]) -> int:
    return hq3_block_count(shape) * HQ3_BLOCK_BYTES


def _pack_codes_numpy(codes: np.ndarray) -> np.ndarray:
    """Pack eight u3 selectors into three little-endian bytes."""
    groups = np.asarray(codes, dtype=np.uint32).reshape(-1, HQ3_BLOCK_SIZE // 8, 8)
    shifts = (3 * np.arange(8, dtype=np.uint32)).reshape(1, 1, 8)
    words = np.bitwise_or.reduce(groups << shifts, axis=2)
    return np.stack((words & 0xFF, (words >> 8) & 0xFF, (words >> 16) & 0xFF), axis=2).astype(np.uint8).reshape(-1, HQ3_INDEX_BYTES)


def _unpack_codes_numpy(selector_bytes: np.ndarray) -> np.ndarray:
    groups = np.asarray(selector_bytes, dtype=np.uint32).reshape(-1, HQ3_BLOCK_SIZE // 8, 3)
    words = groups[:, :, 0] | (groups[:, :, 1] << 8) | (groups[:, :, 2] << 16)
    shifts = (3 * np.arange(8, dtype=np.uint32)).reshape(1, 1, 8)
    return ((words[:, :, None] >> shifts) & 7).reshape(-1, HQ3_BLOCK_SIZE)


def decode_hq3_numpy(packed: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Exact portable HQ3 decode to float32."""
    shape = validate_hq3_shape(shape)
    blocks = np.ascontiguousarray(packed, dtype=np.uint8).reshape(-1, HQ3_BLOCK_BYTES)
    expected = hq3_block_count(shape)
    if blocks.shape[0] != expected:
        raise ValueError(f"HQ3 has {blocks.shape[0]} blocks; shape {shape} requires {expected}")
    levels = blocks[:, :HQ3_CENTROID_BYTES].copy().view("<f2").reshape(-1, HQ3_CENTROID_COUNT).astype(np.float32)
    codes = _unpack_codes_numpy(blocks[:, HQ3_CENTROID_BYTES:])
    values = np.take_along_axis(levels[:, None, :], codes[:, :, None], axis=2)[..., 0]
    return values.reshape(shape)


@dataclass(frozen=True)
class HQ3Tensor:
    """Format-tagged HQ3 bytes, stored as ``uint8[blocks, 112]``."""

    packed: Any
    shape: tuple[int, ...]
    backend: str
    iterations: int
    importance_weighted: bool

    def __post_init__(self) -> None:
        shape = validate_hq3_shape(self.shape)
        object.__setattr__(self, "shape", shape)
        expected = hq3_block_count(shape)
        packed_shape = tuple(int(value) for value in self.packed.shape)
        if packed_shape != (expected, HQ3_BLOCK_BYTES):
            raise ValueError(f"HQ3 packed shape must be ({expected}, {HQ3_BLOCK_BYTES}), got {packed_shape}")

    @property
    def blocks(self) -> int:
        return hq3_block_count(self.shape)

    @property
    def nbytes(self) -> int:
        return self.blocks * HQ3_BLOCK_BYTES

    @property
    def bits_per_weight(self) -> float:
        return HQ3_BITS_PER_WEIGHT

    def dequantize(self, *, dtype: Any | None = None):
        from .api import dequantize

        return dequantize(self, dtype=dtype)

    def numpy(self) -> np.ndarray:
        if isinstance(self.packed, np.ndarray):
            return np.ascontiguousarray(self.packed)
        try:
            return self.packed.detach().cpu().numpy().copy()
        except AttributeError as exc:  # pragma: no cover - defensive API guard
            raise TypeError(f"Unsupported packed storage type {type(self.packed)!r}") from exc

    def to(self, device: Any, *, non_blocking: bool = False) -> "HQ3Tensor":
        torch = _torch()
        if isinstance(self.packed, torch.Tensor):
            moved = self.packed.to(device=device, non_blocking=non_blocking)
        else:
            host_bytes = self.numpy()
            if not host_bytes.flags.writeable:
                host_bytes = host_bytes.copy()
            moved = torch.from_numpy(host_bytes).to(device=device, non_blocking=non_blocking)
        return HQ3Tensor(
            packed=moved,
            shape=self.shape,
            backend=_torch_name(torch, moved),
            iterations=self.iterations,
            importance_weighted=self.importance_weighted,
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        if path.suffix.lower() in {".hq", ".hq1", ".hq2", ".hq3"}:
            from .archive import save_model

            save_model(path, {"__hq3_tensor__": self}, metadata={"kind": "tensor"})
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            packed=self.numpy(),
            shape=np.asarray(self.shape, dtype=np.int64),
            format=np.asarray(HQ3_FORMAT_NAME),
            version=np.asarray(HQ3_FORMAT_VERSION, dtype=np.int64),
            iterations=np.asarray(self.iterations, dtype=np.int64),
            importance_weighted=np.asarray(self.importance_weighted, dtype=np.bool_),
        )
        return path


def _numpy_values(values: Any) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim < 1:
        raise ValueError(f"HQ3 expects at least one dimension, got {values.shape}")
    validate_hq3_shape(tuple(values.shape))
    return np.ascontiguousarray(values, dtype=np.float32)


def _initial_levels_numpy(x: np.ndarray) -> np.ndarray:
    """Evenly spread signed seeds; Lloyd refinement learns non-uniform levels."""
    amax = np.max(np.abs(x), axis=1)
    seeds = np.linspace(-1.0, 1.0, HQ3_CENTROID_COUNT, dtype=np.float64)
    return amax[:, None] * seeds[None, :]


def quantize_hq3_cpu(values: Any, importance: Any | None, iterations: int) -> HQ3Tensor:
    values = _numpy_values(values)
    weights = None
    if importance is not None:
        weights = np.asarray(importance, dtype=np.float32)
        if weights.shape != values.shape:
            raise ValueError(f"importance shape {weights.shape} != values shape {values.shape}")
        weights = np.ascontiguousarray(weights)
    x = values.reshape(-1, HQ3_BLOCK_SIZE).astype(np.float64)
    w = np.ones_like(x) if weights is None else weights.reshape(-1, HQ3_BLOCK_SIZE).astype(np.float64).clip(0.0, None)
    levels = _initial_levels_numpy(x)
    for _ in range(iterations):
        codes = np.argmin((x[:, :, None] - levels[:, None, :]) ** 2, axis=2)
        for centroid in range(HQ3_CENTROID_COUNT):
            selected = codes == centroid
            counts = np.sum(w * selected, axis=1)
            totals = np.sum(w * x * selected, axis=1)
            update = counts > 0.0
            levels[update, centroid] = totals[update] / counts[update]
    stored_levels = levels.astype(np.float16)
    codes = np.argmin((x[:, :, None] - stored_levels.astype(np.float32)[:, None, :]) ** 2, axis=2)
    centroid_bytes = stored_levels.astype("<f2", copy=False).view(np.uint8).reshape(-1, HQ3_CENTROID_BYTES).copy()
    packed = np.concatenate((centroid_bytes, _pack_codes_numpy(codes)), axis=1)
    return HQ3Tensor(
        packed=packed,
        shape=tuple(values.shape),
        backend="cpu",
        iterations=iterations,
        importance_weighted=weights is not None,
    )


def quantize_hq3_torch(values: Any, importance: Any | None, iterations: int) -> HQ3Tensor:
    torch = _torch()
    if not isinstance(values, torch.Tensor):
        raise TypeError("The Torch HQ3 backend requires a torch.Tensor input")
    if values.ndim < 1:
        raise ValueError(f"HQ3 expects at least one dimension, got {tuple(values.shape)}")
    shape = validate_hq3_shape(tuple(values.shape))
    if not values.is_floating_point():
        raise TypeError(f"HQ3 requires a floating-point tensor, got {values.dtype}")
    x = values.contiguous().reshape(-1, HQ3_BLOCK_SIZE).float()
    if importance is None:
        weights = torch.ones_like(x)
    else:
        if not isinstance(importance, torch.Tensor):
            raise TypeError("Torch HQ3 importance must be a torch.Tensor on the same device")
        if tuple(importance.shape) != shape or importance.device != values.device:
            raise ValueError("Torch HQ3 importance must have the same shape and device as values")
        weights = importance.contiguous().reshape(-1, HQ3_BLOCK_SIZE).float().clamp_min_(0.0)
    seeds = torch.linspace(-1.0, 1.0, HQ3_CENTROID_COUNT, dtype=torch.float32, device=x.device)
    levels = x.abs().amax(dim=1, keepdim=True) * seeds.unsqueeze(0)
    for _ in range(iterations):
        codes = (x.unsqueeze(-1) - levels.unsqueeze(1)).square().argmin(dim=-1)
        totals = torch.zeros_like(levels).scatter_add_(1, codes, weights * x)
        counts = torch.zeros_like(levels).scatter_add_(1, codes, weights)
        # A max-normalised calibration vector can sum to less than one for a
        # populated centroid. Clamp only zero denominators; 1.0 collapses
        # learned HQ3 levels and erases the extra precision tier.
        levels = torch.where(counts > 0, totals / counts.clamp_min_(torch.finfo(counts.dtype).tiny), levels)
    levels = levels.to(torch.float16).to(torch.float32)
    codes = (x.unsqueeze(-1) - levels.unsqueeze(1)).square().argmin(dim=-1)
    centroid_bytes = levels.to(torch.float16).contiguous().view(torch.uint8).reshape(-1, HQ3_CENTROID_BYTES)
    groups = codes.reshape(-1, HQ3_BLOCK_SIZE // 8, 8).to(torch.int32)
    shifts = (3 * torch.arange(8, device=x.device, dtype=torch.int32)).reshape(1, 1, 8)
    words = torch.bitwise_or.reduce(groups << shifts, dim=2) if hasattr(torch.bitwise_or, "reduce") else None
    if words is None:  # Torch has no reduction helper on older releases.
        words = (groups << shifts).sum(dim=2)
    packed_codes = torch.stack((words & 0xFF, (words >> 8) & 0xFF, (words >> 16) & 0xFF), dim=2).to(torch.uint8).reshape(-1, HQ3_INDEX_BYTES)
    return HQ3Tensor(
        packed=torch.cat((centroid_bytes, packed_codes), dim=1),
        shape=shape,
        backend=_torch_name(torch, values),
        iterations=iterations,
        importance_weighted=importance is not None,
    )


def dequantize_hq3_cpu(packed: HQ3Tensor) -> np.ndarray:
    return decode_hq3_numpy(packed.packed, packed.shape)


def dequantize_hq3_torch(packed: HQ3Tensor, dtype: Any | None = None):
    torch = _torch()
    if not isinstance(packed.packed, torch.Tensor):
        raise TypeError("dequantize_hq3_torch requires HQ3 bytes stored in a torch.Tensor")
    raw = packed.packed
    if raw.dtype != torch.uint8:
        raise TypeError(f"HQ3 bytes must have dtype torch.uint8, got {raw.dtype}")
    levels = raw[:, :HQ3_CENTROID_BYTES].contiguous().view(torch.float16).reshape(-1, HQ3_CENTROID_COUNT)
    groups = raw[:, HQ3_CENTROID_BYTES:].reshape(-1, HQ3_BLOCK_SIZE // 8, 3).to(torch.int32)
    words = groups[:, :, 0] | (groups[:, :, 1] << 8) | (groups[:, :, 2] << 16)
    shifts = (3 * torch.arange(8, device=raw.device, dtype=torch.int32)).reshape(1, 1, 8)
    codes = ((words.unsqueeze(-1) >> shifts) & 7).reshape(-1, HQ3_BLOCK_SIZE).long()
    output = torch.gather(levels.float(), 1, codes).reshape(packed.shape)
    return output.to(dtype=dtype) if dtype is not None else output


def quantize_hq3_rocm(*args, **kwargs) -> HQ3Tensor:
    raise BackendUnavailable(
        "The native HIP HQ3 quantizer is not built yet; use backend='torch' for on-device portable HQ3 quantization"
    )


def load_hq3(path: str | Path) -> HQ3Tensor:
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        format_name = str(archive["format"].item())
        version = int(archive["version"].item())
        if format_name != HQ3_FORMAT_NAME or version != HQ3_FORMAT_VERSION:
            raise ValueError(f"Unsupported HQ3 file {path}: format={format_name!r}, version={version}")
        return HQ3Tensor(
            packed=np.ascontiguousarray(archive["packed"], dtype=np.uint8),
            shape=tuple(int(value) for value in archive["shape"]),
            backend="cpu",
            iterations=int(archive["iterations"].item()),
            importance_weighted=bool(archive["importance_weighted"].item()),
        )


__all__ = [
    "HQ3Tensor",
    "HQ3_BITS_PER_WEIGHT",
    "HQ3_BLOCK_BYTES",
    "HQ3_BLOCK_SIZE",
    "HQ3_CENTROID_BYTES",
    "HQ3_CENTROID_COUNT",
    "HQ3_FORMAT_NAME",
    "HQ3_FORMAT_VERSION",
    "decode_hq3_numpy",
    "dequantize_hq3_cpu",
    "dequantize_hq3_torch",
    "hq3_block_count",
    "hq3_packed_nbytes",
    "load_hq3",
    "quantize_hq3_cpu",
    "quantize_hq3_rocm",
    "quantize_hq3_torch",
    "validate_hq3_shape",
]
