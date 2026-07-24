"""Portable reference codec for the HQ8 grouped-int8 experiment.

HQ8_G128 stores 128 signed int8 values and one FP16 scale per group:

    [fp16 scale][128 * int8 values]

Its 130-byte group is 8.125 BPW. This is intentionally a row-major reference
layout for codec and quality screening. A future tile-packed HIP format must
use a distinct descriptor so archives stay self-describing and load safely.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .backends import BackendUnavailable, _torch, _torch_name


HQ8_G128_BLOCK_SIZE = 128
HQ8_G128_SCALE_BYTES = 2
HQ8_G128_VALUE_BYTES = HQ8_G128_BLOCK_SIZE
HQ8_G128_BLOCK_BYTES = HQ8_G128_SCALE_BYTES + HQ8_G128_VALUE_BYTES
HQ8_G128_BITS_PER_WEIGHT = HQ8_G128_BLOCK_BYTES * 8 / HQ8_G128_BLOCK_SIZE
HQ8_G128_FORMAT_NAME = "HQ8_G128"
HQ8_G128_FORMAT_VERSION = 1


def validate_hq8_g128_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    shape = tuple(int(value) for value in shape)
    if not shape or any(value <= 0 for value in shape):
        raise ValueError(f"HQ8_G128 requires a non-empty positive shape, got {shape}")
    if shape[-1] % HQ8_G128_BLOCK_SIZE:
        raise ValueError(
            "HQ8_G128 requires the final dimension to be divisible by "
            f"{HQ8_G128_BLOCK_SIZE}, got {shape[-1]}"
        )
    return shape


def hq8_g128_block_count(shape: tuple[int, ...]) -> int:
    shape = validate_hq8_g128_shape(shape)
    return int(np.prod(shape, dtype=np.int64) // HQ8_G128_BLOCK_SIZE)


def hq8_g128_packed_nbytes(shape: tuple[int, ...]) -> int:
    return hq8_g128_block_count(shape) * HQ8_G128_BLOCK_BYTES


def _numpy_values(values: Any) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim < 1:
        raise ValueError(f"HQ8_G128 expects at least one dimension, got {values.shape}")
    shape = validate_hq8_g128_shape(tuple(values.shape))
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError(f"HQ8_G128 requires floating-point input, got {values.dtype}")
    output = np.ascontiguousarray(values, dtype=np.float32)
    if not np.all(np.isfinite(output)):
        raise ValueError("HQ8_G128 requires finite input values")
    return output.reshape(shape)


def decode_hq8_g128_numpy(packed: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Decode row-major HQ8_G128 bytes exactly to float32."""

    shape = validate_hq8_g128_shape(shape)
    blocks = np.ascontiguousarray(packed, dtype=np.uint8).reshape(-1, HQ8_G128_BLOCK_BYTES)
    expected = hq8_g128_block_count(shape)
    if blocks.shape[0] != expected:
        raise ValueError(f"HQ8_G128 has {blocks.shape[0]} blocks; shape {shape} requires {expected}")
    scales = blocks[:, :HQ8_G128_SCALE_BYTES].copy().view("<f2").reshape(-1, 1).astype(np.float32)
    values = blocks[:, HQ8_G128_SCALE_BYTES:].view(np.int8).astype(np.float32)
    return (scales * values).reshape(shape)


@dataclass(frozen=True)
class HQ8Tensor:
    """A row-major HQ8_G128 tensor stored as uint8[groups, 130]."""

    packed: Any
    shape: tuple[int, ...]
    backend: str
    iterations: int = 0
    importance_weighted: bool = False

    def __post_init__(self) -> None:
        shape = validate_hq8_g128_shape(self.shape)
        object.__setattr__(self, "shape", shape)
        expected = hq8_g128_block_count(shape)
        packed_shape = tuple(int(value) for value in self.packed.shape)
        if packed_shape != (expected, HQ8_G128_BLOCK_BYTES):
            raise ValueError(
                "HQ8_G128 packed shape must be "
                f"({expected}, {HQ8_G128_BLOCK_BYTES}), got {packed_shape}"
            )
        if self.iterations != 0:
            raise ValueError("HQ8_G128 uses direct max-abs quantization; iterations must be zero")
        if self.importance_weighted:
            raise ValueError("HQ8_G128 does not support importance weighting")

    @property
    def blocks(self) -> int:
        return hq8_g128_block_count(self.shape)

    @property
    def nbytes(self) -> int:
        return self.blocks * HQ8_G128_BLOCK_BYTES

    @property
    def bits_per_weight(self) -> float:
        return HQ8_G128_BITS_PER_WEIGHT

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

    def to(self, device: Any, *, non_blocking: bool = False) -> "HQ8Tensor":
        torch = _torch()
        if isinstance(self.packed, torch.Tensor):
            moved = self.packed.to(device=device, non_blocking=non_blocking)
        else:
            host_bytes = self.numpy()
            if not host_bytes.flags.writeable:
                host_bytes = host_bytes.copy()
            moved = torch.from_numpy(host_bytes).to(device=device, non_blocking=non_blocking)
        return HQ8Tensor(
            packed=moved,
            shape=self.shape,
            backend=_torch_name(torch, moved),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        if path.suffix.lower() in {".hq", ".hq1", ".hq2", ".hq3", ".hq8"}:
            from .archive import save_model

            save_model(path, {"__hq8_g128_tensor__": self}, metadata={"kind": "tensor"})
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            packed=self.numpy(),
            shape=np.asarray(self.shape, dtype=np.int64),
            format=np.asarray(HQ8_G128_FORMAT_NAME),
            version=np.asarray(HQ8_G128_FORMAT_VERSION, dtype=np.int64),
            iterations=np.asarray(0, dtype=np.int64),
            importance_weighted=np.asarray(False, dtype=np.bool_),
        )
        return path


def quantize_hq8_g128_cpu(values: Any, importance: Any | None = None) -> HQ8Tensor:
    """Quantize a finite CPU array using FP16 group scales and signed int8 codes."""

    if importance is not None:
        raise ValueError("HQ8_G128 does not support importance weighting")
    values = _numpy_values(values)
    groups = values.reshape(-1, HQ8_G128_BLOCK_SIZE)
    scales = (np.max(np.abs(groups), axis=1) / 127.0).astype(np.float16)
    stored_scales = scales.astype(np.float32)
    normalized = np.zeros_like(groups, dtype=np.float32)
    nonzero = stored_scales > 0.0
    normalized[nonzero] = groups[nonzero] / stored_scales[nonzero, None]
    codes = np.clip(np.rint(normalized), -127, 127).astype(np.int8)
    scale_bytes = scales.astype("<f2", copy=False).view(np.uint8).reshape(-1, HQ8_G128_SCALE_BYTES).copy()
    packed = np.concatenate((scale_bytes, codes.view(np.uint8)), axis=1)
    return HQ8Tensor(packed=packed, shape=tuple(values.shape), backend="cpu")


def quantize_hq8_g128_torch(values: Any, importance: Any | None = None) -> HQ8Tensor:
    """Quantize a Torch tensor on its existing CPU, CUDA, or ROCm device."""

    if importance is not None:
        raise ValueError("HQ8_G128 does not support importance weighting")
    torch = _torch()
    if not isinstance(values, torch.Tensor):
        raise TypeError("The Torch HQ8_G128 backend requires a torch.Tensor input")
    if values.ndim < 1:
        raise ValueError(f"HQ8_G128 expects at least one dimension, got {tuple(values.shape)}")
    shape = validate_hq8_g128_shape(tuple(values.shape))
    if not values.is_floating_point():
        raise TypeError(f"HQ8_G128 requires floating-point input, got {values.dtype}")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("HQ8_G128 requires finite input values")
    groups = values.contiguous().reshape(-1, HQ8_G128_BLOCK_SIZE).float()
    scales = (groups.abs().amax(dim=1) / 127.0).to(torch.float16)
    stored_scales = scales.float()
    safe_scales = stored_scales.clamp_min(torch.finfo(torch.float32).tiny)
    codes = torch.clamp(torch.round(groups / safe_scales[:, None]), -127, 127).to(torch.int8)
    codes = torch.where(stored_scales[:, None] > 0.0, codes, torch.zeros_like(codes))
    scale_bytes = scales.contiguous().view(torch.uint8).reshape(-1, HQ8_G128_SCALE_BYTES)
    packed = torch.cat((scale_bytes, codes.contiguous().view(torch.uint8)), dim=1)
    return HQ8Tensor(packed=packed, shape=shape, backend=_torch_name(torch, values))


def dequantize_hq8_g128_cpu(packed: HQ8Tensor) -> np.ndarray:
    return decode_hq8_g128_numpy(packed.packed, packed.shape)


def dequantize_hq8_g128_torch(packed: HQ8Tensor, dtype: Any | None = None):
    torch = _torch()
    if not isinstance(packed.packed, torch.Tensor):
        raise TypeError("dequantize_hq8_g128_torch requires HQ8_G128 bytes in a torch.Tensor")
    raw = packed.packed
    if raw.dtype != torch.uint8:
        raise TypeError(f"HQ8_G128 bytes must have dtype torch.uint8, got {raw.dtype}")
    scales = raw[:, :HQ8_G128_SCALE_BYTES].contiguous().view(torch.float16).reshape(-1).float()
    values = raw[:, HQ8_G128_SCALE_BYTES:].contiguous().view(torch.int8).float()
    output = (scales[:, None] * values).reshape(packed.shape)
    return output.to(dtype=dtype) if dtype is not None else output


def load_hq8_g128(path: str | Path) -> HQ8Tensor:
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        format_name = str(archive["format"].item())
        version = int(archive["version"].item())
        if format_name != HQ8_G128_FORMAT_NAME or version != HQ8_G128_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported HQ8_G128 file {path}: format={format_name!r}, version={version}"
            )
        return HQ8Tensor(
            packed=np.ascontiguousarray(archive["packed"], dtype=np.uint8),
            shape=tuple(int(value) for value in archive["shape"]),
            backend="cpu",
            iterations=int(archive["iterations"].item()),
            importance_weighted=bool(archive["importance_weighted"].item()),
        )


def quantize_hq8_g128_rocm(*args, **kwargs) -> HQ8Tensor:
    raise BackendUnavailable(
        "The native HIP HQ8_G128 quantizer is not built yet; use backend='torch' "
        "for device-resident reference quantization"
    )


__all__ = [
    "HQ8Tensor",
    "HQ8_G128_BITS_PER_WEIGHT",
    "HQ8_G128_BLOCK_BYTES",
    "HQ8_G128_BLOCK_SIZE",
    "HQ8_G128_FORMAT_NAME",
    "HQ8_G128_FORMAT_VERSION",
    "decode_hq8_g128_numpy",
    "dequantize_hq8_g128_cpu",
    "dequantize_hq8_g128_torch",
    "hq8_g128_block_count",
    "hq8_g128_packed_nbytes",
    "load_hq8_g128",
    "quantize_hq8_g128_cpu",
    "quantize_hq8_g128_rocm",
    "quantize_hq8_g128_torch",
    "validate_hq8_g128_shape",
]
