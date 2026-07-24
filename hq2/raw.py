"""Lossless native tensor storage for standalone HQ model packages.

HQ2 is the compressed projection format.  A complete model also needs a
small number of tensors that should remain exact—embeddings, norms, and model
specific components.  ``RAW`` descriptors preserve those tensors' original
little-endian Torch storage inside the same mmap-friendly HQ archive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


_RAW_LAYOUT = "tensor_row_major_c_order"
_RAW_VERSION = 1
_RAW_DTYPES: dict[str, tuple[int, str]] = {
    "bool": (1, "raw-bool-le"),
    "uint8": (1, "raw-uint8-le"),
    "int8": (1, "raw-int8-le"),
    "int16": (2, "raw-int16-le"),
    "int32": (4, "raw-int32-le"),
    "int64": (8, "raw-int64-le"),
    "float16": (2, "raw-float16-le"),
    "bfloat16": (2, "raw-bfloat16-le"),
    "float32": (4, "raw-float32-le"),
    "float64": (8, "raw-float64-le"),
}


def _dtype_info(dtype_name: str) -> tuple[int, str]:
    try:
        return _RAW_DTYPES[dtype_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported HQ RAW dtype {dtype_name!r}") from exc


def raw_format(dtype_name: str):
    """Create the stable archive descriptor for a lossless Torch dtype."""
    from .archive import HQFormatDescriptor

    item_bytes, packing = _dtype_info(dtype_name)
    return HQFormatDescriptor(
        name="RAW",
        version=_RAW_VERSION,
        layout=_RAW_LAYOUT,
        block_size=1,
        block_bytes=item_bytes,
        bits_per_weight=float(item_bytes * 8),
        packing=packing,
        parameters={"dtype": dtype_name, "endianness": "little"},
    )


def torch_dtype_name(dtype: Any) -> str:
    """Map an available Torch dtype to its archive spelling."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - only relevant without Torch
        raise RuntimeError("RAW Torch dtype conversion requires Torch") from exc
    mapping = {
        torch.bool: "bool",
        torch.uint8: "uint8",
        torch.int8: "int8",
        torch.int16: "int16",
        torch.int32: "int32",
        torch.int64: "int64",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
        torch.float64: "float64",
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise ValueError(f"Torch dtype {dtype!r} cannot be stored as an HQ RAW tensor") from exc


def raw_format_for_torch(dtype: Any):
    """Create a lossless RAW descriptor for a Torch tensor dtype."""
    return raw_format(torch_dtype_name(dtype))


def is_raw_descriptor(descriptor: Any) -> bool:
    """Whether a descriptor is the exact stable RAW storage contract."""
    try:
        dtype_name = str(descriptor.parameters["dtype"])
        item_bytes, packing = _dtype_info(dtype_name)
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    return (
        descriptor.name == "RAW"
        and descriptor.version == _RAW_VERSION
        and descriptor.layout == _RAW_LAYOUT
        and descriptor.block_size == 1
        and descriptor.block_bytes == item_bytes
        and descriptor.bits_per_weight == float(item_bytes * 8)
        and descriptor.packing == packing
        and descriptor.parameters.get("endianness") == "little"
    )


@dataclass(frozen=True)
class HQRawTensor:
    """One lossless native tensor memory-mapped from an HQ package."""

    packed: np.ndarray
    shape: tuple[int, ...]
    dtype_name: str

    def __post_init__(self) -> None:
        item_bytes, _ = _dtype_info(self.dtype_name)
        values = 1
        for dimension in self.shape:
            values *= int(dimension)
        if values <= 0 or self.packed.dtype != np.uint8 or self.packed.ndim != 1:
            raise ValueError("HQRawTensor requires a non-empty one-dimensional uint8 payload")
        if int(self.packed.size) != values * item_bytes:
            raise ValueError(
                f"HQ RAW payload has {self.packed.size} bytes; expected {values * item_bytes} for {self.shape}"
            )

    @property
    def nbytes(self) -> int:
        return int(self.packed.size)

    def to_torch(self, *, device: Any | None = None, non_blocking: bool = False):
        """Reconstruct the original Torch tensor from one packed payload copy."""
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - only relevant without Torch
            raise RuntimeError("HQ RAW tensor loading requires Torch") from exc
        dtype_name = torch_dtype_name_to_torch(self.dtype_name, torch)
        # ``packed`` is a read-only file map.  Make a controlled writable CPU
        # staging copy before giving it to Torch, then reshape its byte storage
        # as the original dtype without a numeric conversion.
        storage = torch.from_numpy(np.array(self.packed, dtype=np.uint8, copy=True))
        value = storage.view(dtype_name).reshape(self.shape)
        return value if device is None else value.to(device=device, non_blocking=non_blocking)

    def close(self) -> None:
        """Release the underlying file map after a one-shot streaming load."""
        mapping = getattr(self.packed, "_mmap", None)
        if mapping is not None:
            mapping.close()


def torch_dtype_name_to_torch(dtype_name: str, torch: Any):
    mapping = {
        "bool": torch.bool,
        "uint8": torch.uint8,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    try:
        return mapping[dtype_name]
    except KeyError as exc:  # pragma: no cover - _dtype_info catches this first
        raise ValueError(f"Unsupported HQ RAW dtype {dtype_name!r}") from exc


__all__ = [
    "HQRawTensor",
    "is_raw_descriptor",
    "raw_format",
    "raw_format_for_torch",
    "torch_dtype_name",
]
