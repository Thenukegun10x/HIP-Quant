"""Portable HQ2 block format and packed-tensor container.

HQ2 has a deliberately small, backend-independent physical format: a block
contains four FP16 centroids followed by 256 two-bit centroid indices.  The
bytes here are valid on CPU, CUDA, ROCm, and future Vulkan implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


BLOCK_SIZE = 256
CENTROID_COUNT = 4
CENTROID_BYTES = 8
INDEX_BYTES = 64
BLOCK_BYTES = CENTROID_BYTES + INDEX_BYTES
BITS_PER_WEIGHT = BLOCK_BYTES * 8 / BLOCK_SIZE
FORMAT_NAME = "HQ2"
FORMAT_VERSION = 1


def validate_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    shape = tuple(int(value) for value in shape)
    if not shape or any(value <= 0 for value in shape):
        raise ValueError(f"HQ2 requires a non-empty positive shape, got {shape}")
    if shape[-1] % BLOCK_SIZE:
        raise ValueError(
            f"HQ2 requires the final dimension to be divisible by {BLOCK_SIZE}, got {shape[-1]}"
        )
    return shape


def block_count(shape: tuple[int, ...]) -> int:
    shape = validate_shape(shape)
    return int(np.prod(shape, dtype=np.int64) // BLOCK_SIZE)


def packed_nbytes(shape: tuple[int, ...]) -> int:
    return block_count(shape) * BLOCK_BYTES


def decode_numpy(packed: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Exact vectorized HQ2 decode to float32, independent of GPU backend."""
    shape = validate_shape(shape)
    blocks = np.ascontiguousarray(packed, dtype=np.uint8).reshape(-1, BLOCK_BYTES)
    expected = block_count(shape)
    if blocks.shape[0] != expected:
        raise ValueError(f"HQ2 has {blocks.shape[0]} blocks; shape {shape} requires {expected}")
    levels = blocks[:, :CENTROID_BYTES].copy().view("<f2").reshape(-1, CENTROID_COUNT).astype(np.float32)
    shifts = np.array((0, 2, 4, 6), dtype=np.uint8)
    codes = ((blocks[:, CENTROID_BYTES:, None] >> shifts) & 3).reshape(-1, BLOCK_SIZE)
    values = np.take_along_axis(levels[:, None, :], codes[:, :, None], axis=2)[..., 0]
    return values.reshape(shape)


@dataclass(frozen=True)
class HQ2Tensor:
    """A format-tagged HQ2 tensor whose packed bytes may live on CPU or GPU.

    ``packed`` is ``numpy.uint8[blocks, 72]`` for CPU/ROCm quantization or a
    ``torch.uint8[blocks, 72]`` tensor for the generic Torch backend.  The
    bytes are identical in layout; only their residency differs.
    """

    packed: Any
    shape: tuple[int, ...]
    backend: str
    iterations: int
    importance_weighted: bool

    def __post_init__(self) -> None:
        shape = validate_shape(self.shape)
        object.__setattr__(self, "shape", shape)
        expected = block_count(shape)
        packed_shape = tuple(int(value) for value in self.packed.shape)
        if packed_shape != (expected, BLOCK_BYTES):
            raise ValueError(
                f"HQ2 packed shape must be ({expected}, {BLOCK_BYTES}), got {packed_shape}"
            )

    @property
    def blocks(self) -> int:
        return block_count(self.shape)

    @property
    def nbytes(self) -> int:
        return self.blocks * BLOCK_BYTES

    @property
    def bits_per_weight(self) -> float:
        return BITS_PER_WEIGHT

    def dequantize(self, *, dtype: Any | None = None):
        """Decode to a float tensor/array on the packed data's current device."""
        from .api import dequantize

        return dequantize(self, dtype=dtype)

    def numpy(self) -> np.ndarray:
        """Return packed bytes on CPU without changing the HQ2 representation."""
        if isinstance(self.packed, np.ndarray):
            return np.ascontiguousarray(self.packed)
        try:
            return self.packed.detach().cpu().numpy().copy()
        except AttributeError as exc:  # pragma: no cover - defensive API guard
            raise TypeError(f"Unsupported packed storage type {type(self.packed)!r}") from exc

    def to(self, device: Any, *, non_blocking: bool = False) -> "HQ2Tensor":
        """Move packed bytes to a Torch device without requantizing.

        This is the bridge between a portable ``.hq2.npz`` file/NumPy
        quantizer and a CUDA or ROCm inference path.  The 72-byte block data
        is copied verbatim; only where the bytes reside changes.
        """
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("HQ2Tensor.to() requires `pip install hip-quant[torch]`") from exc

        if isinstance(self.packed, torch.Tensor):
            moved = self.packed.to(device=device, non_blocking=non_blocking)
        else:
            host_bytes = self.numpy()
            # HQ2Model uses a read-only memmap.  Torch warns (correctly) that
            # writing through a tensor backed by such an array is undefined;
            # make the short-lived host staging copy explicit before a device
            # transfer rather than exposing the mapping as writable.
            if not host_bytes.flags.writeable:
                host_bytes = host_bytes.copy()
            moved = torch.from_numpy(host_bytes).to(device=device, non_blocking=non_blocking)
        if moved.device.type == "cuda":
            backend = "rocm-torch" if torch.version.hip else "cuda"
        else:
            backend = "torch-cpu"
        return HQ2Tensor(
            packed=moved,
            shape=self.shape,
            backend=backend,
            iterations=self.iterations,
            importance_weighted=self.importance_weighted,
        )

    def save(self, path: str | Path) -> Path:
        """Write a portable ``.npz`` tensor or native direct-loadable ``.hq2`` file."""
        path = Path(path)
        if path.suffix.lower() == ".hq2":
            from .archive import save_model

            save_model(path, {"__hq2_tensor__": self}, metadata={"kind": "tensor"})
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            packed=self.numpy(),
            shape=np.asarray(self.shape, dtype=np.int64),
            format=np.asarray(FORMAT_NAME),
            version=np.asarray(FORMAT_VERSION, dtype=np.int64),
            iterations=np.asarray(self.iterations, dtype=np.int64),
            importance_weighted=np.asarray(self.importance_weighted, dtype=np.bool_),
        )
        return path


def load(path: str | Path) -> HQ2Tensor:
    """Load a portable CPU-resident HQ2 tensor saved by :meth:`HQ2Tensor.save`."""
    path = Path(path)
    if path.suffix.lower() == ".hq2":
        from .archive import load_model

        model = load_model(path)
        if len(model.tensor_names) != 1:
            raise ValueError(f"{path} contains {len(model.tensor_names)} tensors; use hq2.load_model()")
        return model.tensor(model.tensor_names[0])
    with np.load(path, allow_pickle=False) as archive:
        format_name = str(archive["format"].item())
        version = int(archive["version"].item())
        if format_name != FORMAT_NAME or version != FORMAT_VERSION:
            raise ValueError(f"Unsupported HQ2 file {path}: format={format_name!r}, version={version}")
        return HQ2Tensor(
            packed=np.ascontiguousarray(archive["packed"], dtype=np.uint8),
            shape=tuple(int(value) for value in archive["shape"]),
            backend="cpu",
            iterations=int(archive["iterations"].item()),
            importance_weighted=bool(archive["importance_weighted"].item()),
        )
