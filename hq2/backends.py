"""Backend implementations for the public HQ2 API.

The Torch backend is intentionally written in ordinary Torch operations.  It
therefore runs on both NVIDIA CUDA and AMD ROCm without compiling a custom
extension, and provides a correctness-first portability baseline for future
fused CUDA/HIP/Vulkan kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .format import BLOCK_BYTES, BLOCK_SIZE, CENTROID_BYTES, CENTROID_COUNT, HQ2Tensor, decode_numpy, validate_shape


class BackendUnavailable(RuntimeError):
    """Raised when a requested accelerator backend is not installed or usable."""


@dataclass(frozen=True)
class BackendStatus:
    name: str
    available: bool
    detail: str


def _numpy_array(values: Any) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim < 1:
        raise ValueError(f"HQ2 expects at least one dimension, got {values.shape}")
    validate_shape(tuple(values.shape))
    return np.ascontiguousarray(values, dtype=np.float32)


def quantize_cpu(values: Any, importance: Any | None, iterations: int) -> HQ2Tensor:
    """Portable HQ2 reference implementation with no ROCm dependency."""
    values = _numpy_array(values)
    weights = None
    if importance is not None:
        weights = np.asarray(importance, dtype=np.float32)
        if weights.shape != values.shape:
            raise ValueError(f"importance shape {weights.shape} != values shape {values.shape}")
        weights = np.ascontiguousarray(weights)
    x = values.reshape(-1, BLOCK_SIZE).astype(np.float64)
    w = np.ones_like(x) if weights is None else weights.reshape(-1, BLOCK_SIZE).astype(np.float64).clip(0.0, None)
    amax = np.max(np.abs(x), axis=1)
    levels = np.stack((-amax, -amax / 3.0, amax / 3.0, amax), axis=1)
    for _ in range(iterations):
        codes = np.argmin((x[:, :, None] - levels[:, None, :]) ** 2, axis=2)
        for centroid in range(CENTROID_COUNT):
            selected = codes == centroid
            counts = np.sum(w * selected, axis=1)
            totals = np.sum(w * x * selected, axis=1)
            update = counts > 0.0
            levels[update, centroid] = totals[update] / counts[update]
    stored_levels = levels.astype(np.float16)
    codes = np.argmin((x[:, :, None] - stored_levels.astype(np.float32)[:, None, :]) ** 2, axis=2)
    # HQ2 is a little-endian wire format.  Be explicit rather than inheriting
    # the host's endianness so files made on a rare big-endian NumPy host are
    # still readable by CUDA/ROCm/Vulkan implementations.
    centroid_bytes = stored_levels.astype("<f2", copy=False).view(np.uint8).reshape(-1, CENTROID_BYTES).copy()
    code_groups = codes.reshape(-1, BLOCK_SIZE // 4, 4).astype(np.uint8)
    packed_codes = (
        code_groups[:, :, 0]
        | (code_groups[:, :, 1] << 2)
        | (code_groups[:, :, 2] << 4)
        | (code_groups[:, :, 3] << 6)
    )
    packed = np.concatenate((centroid_bytes, packed_codes), axis=1)
    return HQ2Tensor(
        packed=packed,
        shape=tuple(values.shape),
        backend="cpu",
        iterations=iterations,
        importance_weighted=weights is not None,
    )


def quantize_rocm(values: Any, importance: Any | None, iterations: int) -> HQ2Tensor:
    """Use the existing optimized HIP kernel for a host NumPy array."""
    GGML_TYPE, get_hip_quant = _hip_quant_api()

    values = _numpy_array(values)
    weights = None
    if importance is not None:
        weights = np.asarray(importance, dtype=np.float32)
        if weights.shape != values.shape:
            raise ValueError(f"importance shape {weights.shape} != values shape {values.shape}")
        weights = np.ascontiguousarray(weights)
    try:
        quantizer = get_hip_quant()
        packed = quantizer.quantize_numpy(
            values.reshape(-1, values.shape[-1]),
            GGML_TYPE["HQ2"],
            imatrix=None if weights is None else weights.reshape(-1, values.shape[-1]),
            hq2_iterations=iterations,
        )
    except Exception as exc:
        raise BackendUnavailable(f"ROCm HQ2 backend is unavailable: {exc}") from exc
    return HQ2Tensor(
        packed=packed.reshape(-1, BLOCK_BYTES),
        shape=tuple(values.shape),
        backend="rocm",
        iterations=iterations,
        importance_weighted=weights is not None,
    )


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise BackendUnavailable("The Torch backend requires `pip install hip-quant[torch]`") from exc
    return torch


def _torch_name(torch, values) -> str:
    if values.device.type != "cuda":
        return "torch-cpu"
    return "rocm-torch" if torch.version.hip else "cuda"


def quantize_torch(values: Any, importance: Any | None, iterations: int) -> HQ2Tensor:
    """Quantize a Torch tensor entirely on its current CPU/CUDA/ROCm device."""
    torch = _torch()
    if not isinstance(values, torch.Tensor):
        raise TypeError("The Torch backend requires a torch.Tensor input")
    if values.ndim < 1:
        raise ValueError(f"HQ2 expects at least one dimension, got {tuple(values.shape)}")
    shape = validate_shape(tuple(values.shape))
    if not values.is_floating_point():
        raise TypeError(f"HQ2 requires a floating-point tensor, got {values.dtype}")
    x = values.contiguous().reshape(-1, BLOCK_SIZE).float()
    if importance is None:
        weights = torch.ones_like(x)
    else:
        if not isinstance(importance, torch.Tensor):
            raise TypeError("Torch importance must be a torch.Tensor on the same device")
        if tuple(importance.shape) != shape or importance.device != values.device:
            raise ValueError("Torch importance must have the same shape and device as values")
        weights = importance.contiguous().reshape(-1, BLOCK_SIZE).float().clamp_min_(0.0)

    amax = x.abs().amax(dim=1)
    levels = torch.stack((-amax, -amax / 3.0, amax / 3.0, amax), dim=1)
    for _ in range(iterations):
        codes = (x.unsqueeze(-1) - levels.unsqueeze(1)).square().argmin(dim=-1)
        totals = torch.zeros_like(levels).scatter_add_(1, codes, weights * x)
        counts = torch.zeros_like(levels).scatter_add_(1, codes, weights)
        # Importance weights are routinely normalised to [0, 1]. A populated
        # centroid can therefore have a total weight below one; clamping to
        # 1.0 shrinks it toward zero. Only guard the truly empty case handled
        # by the surrounding ``where``.
        levels = torch.where(counts > 0, totals / counts.clamp_min_(torch.finfo(counts.dtype).tiny), levels)

    # Persisted HQ2 centroids are FP16; assign against exactly those values.
    levels = levels.to(torch.float16).to(torch.float32)
    codes = (x.unsqueeze(-1) - levels.unsqueeze(1)).square().argmin(dim=-1)
    centroid_bytes = levels.to(torch.float16).contiguous().view(torch.uint8).reshape(-1, CENTROID_BYTES)
    code_groups = codes.reshape(-1, BLOCK_SIZE // 4, 4).to(torch.uint8)
    packed_codes = code_groups[:, :, 0] | (code_groups[:, :, 1] << 2) | (code_groups[:, :, 2] << 4) | (code_groups[:, :, 3] << 6)
    packed = torch.cat((centroid_bytes, packed_codes), dim=1)
    return HQ2Tensor(
        packed=packed,
        shape=shape,
        backend=_torch_name(torch, values),
        iterations=iterations,
        importance_weighted=importance is not None,
    )


def dequantize_torch(packed: HQ2Tensor, dtype: Any | None = None):
    torch = _torch()
    if not isinstance(packed.packed, torch.Tensor):
        raise TypeError("dequantize_torch requires HQ2 bytes stored in a torch.Tensor")
    raw = packed.packed
    if raw.dtype != torch.uint8:
        raise TypeError(f"HQ2 bytes must have dtype torch.uint8, got {raw.dtype}")
    levels = raw[:, :CENTROID_BYTES].contiguous().view(torch.float16).reshape(-1, CENTROID_COUNT)
    shifts = torch.tensor((0, 2, 4, 6), device=raw.device, dtype=torch.uint8)
    codes = ((raw[:, CENTROID_BYTES:].unsqueeze(-1) >> shifts) & 3).reshape(-1, BLOCK_SIZE).long()
    output = torch.gather(levels.float(), 1, codes).reshape(packed.shape)
    return output.to(dtype=dtype) if dtype is not None else output


def backend_status() -> dict[str, BackendStatus]:
    """Report actual local availability without silently falling back."""
    statuses = {"cpu": BackendStatus("cpu", True, "portable NumPy reference backend")}
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
        detail = "Torch CUDA device available" if cuda and not torch.version.hip else (
            "Torch ROCm device available" if cuda else "Torch installed; no CUDA/ROCm device visible"
        )
        statuses["torch"] = BackendStatus("torch", True, detail)
        statuses["cuda"] = BackendStatus("cuda", cuda and not bool(torch.version.hip), detail)
        statuses["rocm-torch"] = BackendStatus("rocm-torch", cuda and bool(torch.version.hip), detail)
    except ImportError:
        statuses["torch"] = BackendStatus("torch", False, "install Torch for CUDA/ROCm tensor support")
        statuses["cuda"] = BackendStatus("cuda", False, "install CUDA Torch")
        statuses["rocm-torch"] = BackendStatus("rocm-torch", False, "install ROCm Torch")
    try:
        _, get_hip_quant = _hip_quant_api()
        statuses["rocm"] = BackendStatus("rocm", True, get_hip_quant().device_name)
    except Exception as exc:
        statuses["rocm"] = BackendStatus("rocm", False, f"native HIP backend unavailable: {exc}")
    statuses["vulkan"] = BackendStatus(
        "vulkan", False,
        "Vulkan compute ABI is reserved but the SPIR-V/WGSL backend is not implemented yet",
    )
    return statuses


def dequantize_cpu(packed: HQ2Tensor) -> np.ndarray:
    return decode_numpy(packed.packed, packed.shape)


def _hip_quant_api():
    """Import the native package both installed and directly from this checkout."""
    try:
        from hip_quant import GGML_TYPE, get_hip_quant
    except ModuleNotFoundError:
        # The repository root *is* the hip_quant package, so an editable-style
        # `python` launched from that directory sees it as __init__.  Wheels
        # always take the normal import path above.
        from __init__ import GGML_TYPE, get_hip_quant
    return GGML_TYPE, get_hip_quant
