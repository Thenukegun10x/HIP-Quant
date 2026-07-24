"""Stable, backend-neutral public API for portable HQ-family quantization."""

from __future__ import annotations

from typing import Any

import numpy as np

from .backends import BackendUnavailable, backend_status, dequantize_cpu, dequantize_torch, quantize_cpu, quantize_rocm, quantize_torch
from .format import FORMAT_NAME as HQ2_FORMAT_NAME, HQ2Tensor, load as load_hq2
from .hq3 import (
    HQ3_FORMAT_NAME,
    HQ3Tensor,
    dequantize_hq3_cpu,
    dequantize_hq3_torch,
    load_hq3,
    quantize_hq3_cpu,
    quantize_hq3_rocm,
    quantize_hq3_torch,
)
from .hq8 import (
    HQ8_G128_FORMAT_NAME,
    HQ8Tensor,
    dequantize_hq8_g128_cpu,
    dequantize_hq8_g128_torch,
    load_hq8_g128,
    quantize_hq8_g128_cpu,
    quantize_hq8_g128_rocm,
    quantize_hq8_g128_torch,
)


def quantize(
    values: Any,
    *,
    importance: Any | None = None,
    backend: str = "auto",
    iterations: int = 8,
    format: str = "hq2",
) -> HQ2Tensor | HQ3Tensor | HQ8Tensor:
    """Quantize a floating tensor/array into portable HQ-family bytes.

    ``backend='torch'`` works on a Torch tensor's existing CPU/CUDA/ROCm
    device. ``backend='rocm'`` calls the optimized native HIP quantizer for a
    NumPy input. ``backend='cpu'`` is an exact portable reference.  ``auto``
    selects Torch for Torch tensors and CPU for NumPy arrays, avoiding hidden
    GPU initialization or host/device copies.
    """
    if not 1 <= int(iterations) <= 16:
        raise ValueError("iterations must be in [1, 16]")
    iterations = int(iterations)
    format_name = str(format).upper()
    if format_name not in {HQ2_FORMAT_NAME, HQ3_FORMAT_NAME, HQ8_G128_FORMAT_NAME}:
        raise ValueError("format must be 'hq2', 'hq3', or 'hq8_g128'")
    name = backend.lower().replace("_", "-")
    is_torch = _is_torch_tensor(values)
    if name == "auto":
        name = "torch" if is_torch else "cpu"
    if name == "cpu":
        if is_torch:
            values = values.detach().cpu().numpy()
            importance = None if importance is None else importance.detach().cpu().numpy()
        if format_name == HQ2_FORMAT_NAME:
            return quantize_cpu(values, importance, iterations)
        if format_name == HQ3_FORMAT_NAME:
            return quantize_hq3_cpu(values, importance, iterations)
        return quantize_hq8_g128_cpu(values, importance)
    if name == "rocm":
        if is_torch:
            raise TypeError("The native ROCm backend accepts host NumPy input; use backend='torch' for device tensors")
        if format_name == HQ2_FORMAT_NAME:
            return quantize_rocm(values, importance, iterations)
        if format_name == HQ3_FORMAT_NAME:
            return quantize_hq3_rocm(values, importance, iterations)
        return quantize_hq8_g128_rocm(values, importance)
    if name in {"torch", "cuda", "rocm-torch"}:
        if format_name == HQ2_FORMAT_NAME:
            result = quantize_torch(values, importance, iterations)
        elif format_name == HQ3_FORMAT_NAME:
            result = quantize_hq3_torch(values, importance, iterations)
        else:
            result = quantize_hq8_g128_torch(values, importance)
        if name == "cuda" and result.backend != "cuda":
            raise BackendUnavailable(f"backend='cuda' requested, but tensor is on {result.backend}")
        if name == "rocm-torch" and result.backend != "rocm-torch":
            raise BackendUnavailable(f"backend='rocm-torch' requested, but tensor is on {result.backend}")
        return result
    if name == "vulkan":
        raise BackendUnavailable(backend_status()["vulkan"].detail)
    choices = "auto, cpu, rocm, torch, cuda, rocm-torch, vulkan"
    raise ValueError(f"Unknown HQ-family backend {backend!r}; choose one of {choices}")


def dequantize(packed: HQ2Tensor | HQ3Tensor | HQ8Tensor, *, dtype: Any | None = None):
    """Decode packed HQ-family bytes to float32 on their current device."""
    if isinstance(packed, HQ2Tensor):
        if isinstance(packed.packed, np.ndarray):
            output = dequantize_cpu(packed)
            return output.astype(dtype, copy=False) if dtype is not None else output
        return dequantize_torch(packed, dtype=dtype)
    if isinstance(packed, HQ3Tensor):
        if isinstance(packed.packed, np.ndarray):
            output = dequantize_hq3_cpu(packed)
            return output.astype(dtype, copy=False) if dtype is not None else output
        return dequantize_hq3_torch(packed, dtype=dtype)
    if isinstance(packed, HQ8Tensor):
        if isinstance(packed.packed, np.ndarray):
            output = dequantize_hq8_g128_cpu(packed)
            return output.astype(dtype, copy=False) if dtype is not None else output
        return dequantize_hq8_g128_torch(packed, dtype=dtype)
    raise TypeError("dequantize expects an HQ tensor returned by quantize() or load()")


def load(path: str):
    """Load a one-tensor portable HQ2/HQ3 file or archive."""
    from pathlib import Path

    path = Path(path)
    if path.suffix.lower() in {".hq", ".hq1", ".hq2", ".hq3"}:
        from .archive import load_model

        model = load_model(path)
        if len(model.tensor_names) != 1:
            raise ValueError(f"{path} contains {len(model.tensor_names)} tensors; use hq2.load_model()")
        return model.tensor(model.tensor_names[0])
    with np.load(path, allow_pickle=False) as archive:
        format_name = str(archive["format"].item())
    if format_name == HQ2_FORMAT_NAME:
        return load_hq2(path)
    if format_name == HQ3_FORMAT_NAME:
        return load_hq3(path)
    if format_name == HQ8_G128_FORMAT_NAME:
        return load_hq8_g128(path)
    raise ValueError(f"Unsupported HQ file {path}: format={format_name!r}")


def _is_torch_tensor(value: Any) -> bool:
    # NumPy arrays are the overwhelmingly common portable-analysis input. Do
    # not import Torch merely to reject one: on Windows ROCm that can touch the
    # runtime/device probe even though the caller requested backend='cpu'.
    if isinstance(value, np.ndarray):
        return False
    module_name = type(value).__module__.split(".", 1)[0]
    if module_name != "torch":
        return False
    try:
        import torch
    except ImportError:
        return False
    return isinstance(value, torch.Tensor)


__all__ = ["BackendUnavailable", "backend_status", "dequantize", "load", "quantize"]
