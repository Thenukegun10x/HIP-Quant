"""
hip_quant/torch_api.py
======================

Phase 2 & 3 Python API for GPU-resident FP8 operations.

This module provides:
  Phase 2 — element-wise FP8 quant / dequant wrappers over ``hip_quant._C``.
  Phase 3 — autograd-safe FP8 linear layers for LLM training:
    ``Fp8LinearFunction``       — base autograd.Function (unscaled)
    ``Fp8Linear``               — drop-in nn.Linear replacement
    ``Fp8ScaledLinearFunction`` — autograd.Function with per-tensor amax scaling
    ``Fp8ScaledLinear``         — Fp8Linear + delayed-scaling via Fp8TensorMeta
    ``convert_to_fp8()``        — convert any nn.Module's Linear layers in-place
  Phase 4 — direct HIP GEMM kernel bindings + ``Fp8TensorMeta`` scaffold.

The NumPy/ctypes API in ``hip_quant.__init__`` is *not* touched.
"""

from __future__ import annotations

import math
import os
import warnings
from typing import Dict, Optional, Sequence, Set, Tuple, Union

# ---------------------------------------------------------------------------
# Lazy imports — file remains importable without torch or the _C extension
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

_C = None
_WMMA_GUARD_CACHE: Dict[int, Tuple[str, str]] = {}
_STOCHASTIC_E5M2_COUNTER = 0
_MXFP4_EMULATION_WARNING_EMITTED = False

MXFP4_BLOCK_SIZE = 32
MXFP4_PACKED_VALUE_BYTES_PER_BLOCK = 16
MXFP4_SCALE_BYTES_PER_BLOCK = 1


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes", "on")


def _warn_mxfp4_emulation() -> None:
    """Emit the one-time, user-visible warning required for the MXFP4 bridge."""
    global _MXFP4_EMULATION_WARNING_EMITTED
    if _MXFP4_EMULATION_WARNING_EMITTED or _env_flag("HIP_QUANT_SUPPRESS_EMULATED_MXFP4_WARNING"):
        return
    warnings.warn(
        "MXFP4 is running through hip_quant's emulation path: packed E2M1 + UE8M0 "
        "weights are expanded to FP8 E4M3 and executed through hipBLASLt. This GPU "
        "does not use native MXFP4 instructions; expect slower speeds and higher "
        "transient VRAM use than a native MXFP4 implementation.",
        RuntimeWarning,
        stacklevel=3,
    )
    _MXFP4_EMULATION_WARNING_EMITTED = True


def _scale_to_float(scale: Union["torch.Tensor", float]) -> float:
    """Single sync point for legacy scalar-scale FP8 GEMM launchers."""
    value = float(scale.item()) if torch.is_tensor(scale) else float(scale)
    if not math.isfinite(value) or value <= 0.0:
        raise FloatingPointError(f"FP8 scale must be finite and positive, got {value!r}")
    return value


def _validate_fp8_scale(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise FloatingPointError(f"{name} must be finite and positive, got {value!r}")
    return value


def _fp8_scale_tensor(
    scale: Union["torch.Tensor", float],
    name: str,
    device: "torch.device",
) -> "torch.Tensor":
    """Return a device scalar without synchronizing a CUDA scale to the host."""
    if torch.is_tensor(scale):
        if scale.numel() != 1:
            raise ValueError(f"{name} must contain exactly one value")
        result = scale.detach().to(device=device, dtype=torch.float32).reshape(())
        valid = torch.isfinite(result) & (result > 0.0)
        if result.is_cuda and hasattr(torch, "_assert_async"):
            torch._assert_async(valid, f"{name} must be finite and positive")
        elif not bool(valid.item()):
            raise FloatingPointError(f"{name} must be finite and positive")
        return result
    value = _validate_fp8_scale(scale, name)
    return torch.full((), value, device=device, dtype=torch.float32)


def _stochastic_e5m2_enabled() -> bool:
    return _env_flag("HIP_QUANT_STOCHASTIC_E5M2")


def _next_stochastic_e5m2_seed() -> int:
    global _STOCHASTIC_E5M2_COUNTER
    env_seed = os.environ.get("HIP_QUANT_STOCHASTIC_E5M2_SEED")
    if env_seed is not None:
        base = int(env_seed, 0)
    elif _TORCH_AVAILABLE:
        base = int(torch.initial_seed())
    else:
        base = 0x9E3779B97F4A7C15
    seed = (base + _STOCHASTIC_E5M2_COUNTER * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    _STOCHASTIC_E5M2_COUNTER += 1
    return seed


def _pad_2d_cols(x: "torch.Tensor", multiple: int = 16) -> Tuple["torch.Tensor", int]:
    pad = (-x.size(1)) % multiple
    if pad:
        x = F.pad(x, (0, pad))
    return x, pad


def _pad_2d_rows(x: "torch.Tensor", multiple: int = 16) -> Tuple["torch.Tensor", int]:
    pad = (-x.size(0)) % multiple
    if pad:
        x = F.pad(x, (0, 0, 0, pad))
    return x, pad


def _fp8_linear_backend() -> str:
    backend = os.environ.get("HIP_QUANT_FP8_LINEAR_BACKEND", "hipblaslt").strip().lower()
    if os.environ.get("HIP_QUANT_USE_CUSTOM_WMMA", "").lower() in ("1", "true", "yes", "on"):
        return "custom"
    if backend in ("custom", "wmma", "hip", "hip_wmma"):
        return "custom"
    return "hipblaslt"


def _as_float8_e4m3(x: "torch.Tensor") -> "torch.Tensor":
    """Reinterpret hip_quant uint8 E4M3 storage as torch.float8_e4m3fn."""
    if x.dtype == torch.float8_e4m3fn:
        return x
    if x.dtype == torch.uint8:
        return x.view(torch.float8_e4m3fn)
    raise TypeError(f"expected uint8 or float8_e4m3fn FP8 storage, got {x.dtype}")


def _hipblaslt_fp8_linear_forward_prequant(
    input_fp8:        "torch.Tensor",
    weight_fp8:       "torch.Tensor",
    bias:             Optional["torch.Tensor"],
    input_inv_scale:  Union["torch.Tensor", float],
    weight_inv_scale: Union["torch.Tensor", float],
    out_dtype:        "torch.dtype",
) -> Optional["torch.Tensor"]:
    """hipBLASLt path for already-quantized E4M3 operands.

    ``input_inv_scale`` / ``weight_inv_scale`` are dequant scales applied by
    ``_scaled_mm`` (i.e. reciprocal of the quant multipliers).
    """
    if _fp8_linear_backend() == "custom" or _env_flag("HIP_QUANT_DISABLE_HIPBLASLT"):
        return None
    if not _TORCH_AVAILABLE or not input_fp8.is_cuda:
        return None
    if not hasattr(torch, "_scaled_mm") or not hasattr(torch, "float8_e4m3fn"):
        return None
    if input_fp8.dim() != 2 or weight_fp8.dim() != 2:
        return None

    scale_a = _fp8_scale_tensor(input_inv_scale, "input_inv_scale", input_fp8.device)
    scale_b = _fp8_scale_tensor(weight_inv_scale, "weight_inv_scale", input_fp8.device)

    a = _as_float8_e4m3(input_fp8.contiguous())
    b = _as_float8_e4m3(weight_fp8.contiguous())
    out_features = b.size(0)

    a, row_pad = _pad_2d_rows(a)
    a, k_pad = _pad_2d_cols(a)
    if k_pad:
        b = F.pad(b, (0, k_pad))
    b, n_pad = _pad_2d_rows(b)
    # Row-major A @ column-major B: keep the transpose non-contiguous.
    b_t = b.contiguous().t()

    try:
        out = torch._scaled_mm(a, b_t, scale_a, scale_b, out_dtype=out_dtype)
    except RuntimeError:
        if _env_flag("HIP_QUANT_HIPBLASLT_STRICT"):
            raise
        return None
    if row_pad:
        out = out[:-row_pad, :]
    if n_pad:
        out = out[:, :out_features]
    if bias is not None:
        out = out + bias.unsqueeze(0)
    return out


def _hipblaslt_fp8_linear_forward(
    input:        "torch.Tensor",
    weight:       "torch.Tensor",
    bias:         Optional["torch.Tensor"],
    input_scale:  Union["torch.Tensor", float] = 1.0,
    weight_scale: Union["torch.Tensor", float] = 1.0,
) -> Optional["torch.Tensor"]:
    """Fast ROCm FP8 GEMM path through PyTorch's hipBLASLt-backed _scaled_mm."""
    if _fp8_linear_backend() == "custom":
        return None
    if os.environ.get("HIP_QUANT_DISABLE_HIPBLASLT", "").lower() in ("1", "true", "yes", "on"):
        return None
    if not _TORCH_AVAILABLE or not input.is_cuda:
        return None
    if not hasattr(torch, "_scaled_mm") or not hasattr(torch, "float8_e4m3fn"):
        return None
    if input.dim() != 2 or weight.dim() != 2:
        return None

    input_scale_t = _fp8_scale_tensor(input_scale, "input_scale", input.device)
    weight_scale_t = _fp8_scale_tensor(weight_scale, "weight_scale", input.device)

    a = input.contiguous()
    b = weight.contiguous()
    scale_is_one = (
        (not torch.is_tensor(input_scale) and float(input_scale) == 1.0)
        and (not torch.is_tensor(weight_scale) and float(weight_scale) == 1.0)
    )
    if not scale_is_one:
        # Prefer one quant pass into uint8 storage that can be reused by
        # autograd save-for-backward callers via the prequant helper.
        a_fp8 = quantize_e4m3((a * input_scale_t).contiguous())
        b_fp8 = quantize_e4m3((b * weight_scale_t).contiguous())
        return _hipblaslt_fp8_linear_forward_prequant(
            a_fp8,
            b_fp8,
            bias,
            input_scale_t.reciprocal(),
            weight_scale_t.reciprocal(),
            input.dtype,
        )

    out_features = b.size(0)
    a, row_pad = _pad_2d_rows(a)
    a, k_pad = _pad_2d_cols(a)
    if k_pad:
        b = F.pad(b, (0, k_pad))
    b, n_pad = _pad_2d_rows(b)

    scale_a = input_scale_t.reciprocal()
    scale_b = weight_scale_t.reciprocal()
    a_fp8 = a.to(torch.float8_e4m3fn)
    b_fp8_t = b.to(torch.float8_e4m3fn).contiguous().t()

    try:
        out = torch._scaled_mm(a_fp8, b_fp8_t, scale_a, scale_b, out_dtype=input.dtype)
    except RuntimeError:
        if os.environ.get("HIP_QUANT_HIPBLASLT_STRICT", "").lower() in ("1", "true", "yes", "on"):
            raise
        return None
    if row_pad:
        out = out[:-row_pad, :]
    if n_pad:
        out = out[:, :out_features]
    if bias is not None:
        out = out + bias.unsqueeze(0)
    return out


def _hipblaslt_fp8_backward(
    grad_output: "torch.Tensor",
    weight:      "torch.Tensor",
    input_f32:   "torch.Tensor",
    weight_scale: Union["torch.Tensor", float] = 1.0,
    input_scale:  Union["torch.Tensor", float] = 1.0,
) -> Optional[Tuple["torch.Tensor", "torch.Tensor"]]:
    """Try backward via torch._scaled_mm with float8_e5m2.

    _scaled_mm convention:
      A row-major [M, K], B column-major [K, N]  ->  A @ B = [M, N].

    Returns (grad_input, grad_weight) or None on failure.
    """
    if _fp8_linear_backend() == "custom" or _env_flag("HIP_QUANT_DISABLE_HIPBLASLT"):
        return None
    if not _TORCH_AVAILABLE or not grad_output.is_cuda:
        return None
    if not hasattr(torch, "_scaled_mm") or not hasattr(torch, "float8_e5m2"):
        return None

    weight_scale_t = _fp8_scale_tensor(weight_scale, "weight_scale", grad_output.device)
    input_scale_t = _fp8_scale_tensor(input_scale, "input_scale", grad_output.device)

    go = grad_output.contiguous()
    w  = weight.contiguous()
    x  = input_f32.contiguous()

    m = go.size(0)
    n = go.size(1)
    k = w.size(1)

    try:
        # grad_input = go @ w  ->  (M,N) @ (N,K) = (M,K)
        # A = go [M,N] row-major, B = w [N,K] column-major
        go_gi, m_pad = _pad_2d_rows(go)
        go_gi, n_pad = _pad_2d_cols(go_gi)
        # Quantize the scaled operand, then use the reciprocal dequant scale.
        # Callers pass quantization multipliers (FP8_MAX / amax), not inverse
        # scales. Applying the multiplier after GEMM amplifies gradients.
        w_gi = (
            w if not torch.is_tensor(weight_scale) and float(weight_scale) == 1.0
            else (w * weight_scale_t).contiguous()
        )
        if n_pad:
            w_gi = F.pad(w_gi, (0, 0, 0, n_pad))
        w_gi, k_pad = _pad_2d_cols(w_gi)
        a_gi = go_gi.to(torch.float8_e5m2)                                  # [M,N] row-major
        b_gi = w_gi.to(torch.float8_e5m2).t().contiguous().t()              # [N,K] column-major
        s_go = torch.ones((), device=go.device, dtype=torch.float32)
        s_w  = weight_scale_t.reciprocal()
        grad_input = torch._scaled_mm(a_gi, b_gi, s_go, s_w, out_dtype=go.dtype)
        if m_pad:
            grad_input = grad_input[:-m_pad, :]
        if k_pad:
            grad_input = grad_input[:, :k]

        # grad_weight = go.T @ input  ->  (N,M) @ (M,K) = (N,K)
        # A = go.T [N,M] row-major, B = input [M,K] column-major
        go_t = go.t().contiguous()
        go_t, n_row_pad = _pad_2d_rows(go_t)
        go_t, m_col_pad = _pad_2d_cols(go_t)
        x_gw = (
            x if not torch.is_tensor(input_scale) and float(input_scale) == 1.0
            else (x * input_scale_t).contiguous()
        )
        if m_col_pad:
            x_gw = F.pad(x_gw, (0, 0, 0, m_col_pad))
        x_gw, k_col_pad = _pad_2d_cols(x_gw)
        a_gw = go_t.to(torch.float8_e5m2)                                  # [N,M] row-major
        b_gw = x_gw.to(torch.float8_e5m2).t().contiguous().t()             # [M,K] column-major
        s_go_t = torch.ones((), device=go.device, dtype=torch.float32)
        s_x    = input_scale_t.reciprocal()
        grad_weight = torch._scaled_mm(a_gw, b_gw, s_go_t, s_x, out_dtype=weight.dtype)
        if n_row_pad:
            grad_weight = grad_weight[:-n_row_pad, :]
        if k_col_pad:
            grad_weight = grad_weight[:, :k]

        return (grad_input, grad_weight)
    except RuntimeError:
        if _env_flag("HIP_QUANT_HIPBLASLT_STRICT"):
            raise
        return None


def _parse_rocm_version(value: Optional[str]) -> Tuple[int, int]:
    if not value:
        return (0, 0)
    parts = []
    for part in str(value).replace("-", ".").split("."):
        if not part.isdigit():
            break
        parts.append(int(part))
        if len(parts) == 2:
            break
    while len(parts) < 2:
        parts.append(0)
    return (parts[0], parts[1])


def _require_gfx12_fp8_wmma(tensor: "torch.Tensor") -> None:
    if os.environ.get("HIP_QUANT_DISABLE_WMMA", "").lower() in ("1", "true", "yes", "on"):
        raise RuntimeError("hip_quant FP8/BF8 WMMA kernels are disabled by HIP_QUANT_DISABLE_WMMA.")
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed. Install torch with ROCm support first.")
    if not torch.cuda.is_available():
        raise RuntimeError("hip_quant FP8/BF8 WMMA kernels require a ROCm/HIP GPU.")

    device = tensor.device.index if tensor.device.index is not None else torch.cuda.current_device()
    cached = _WMMA_GUARD_CACHE.get(device)
    if cached is None:
        props = torch.cuda.get_device_properties(device)
        arch = getattr(props, "gcnArchName", "") or "unknown"
        rocm_version = getattr(torch.version, "hip", None)
        cached = (arch, str(rocm_version or "unknown"))
        _WMMA_GUARD_CACHE[device] = cached
    arch, rocm_version = cached

    if not arch.startswith("gfx12"):
        raise RuntimeError(
            f"hip_quant FP8/BF8 WMMA linear kernels use gfx12/RDNA4 w32 intrinsics; current device arch is {arch}. "
            "CDNA may support FP8/BF16 through MFMA/rocBLASLt paths, but not this RDNA4-specific kernel."
        )
    if _parse_rocm_version(rocm_version) < (7, 2):
        raise RuntimeError(
            f"hip_quant FP8/BF8 WMMA linear kernels require ROCm 7.2+; current torch.version.hip is {rocm_version}."
        )


def _load_extension() -> object:
    """Load ``hip_quant._C`` on first use."""
    global _C
    if _C is not None:
        return _C
    if not _TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is not installed. Install torch with ROCm support first."
        )
    try:
        from hip_quant import _C as _ext  # type: ignore[attr-defined]
        _C = _ext
    except ImportError:
        try:
            import sys, os
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import _C as _ext
            _C = _ext
        except ImportError as exc:
            raise ImportError(
                "hip_quant._C extension not found. "
                "Build it first with:\n"
                "  python setup_torch.py build_ext --inplace"
            ) from exc
    return _C


# ===========================================================================
# Phase 2: element-wise FP8 quantize / dequantize
# ===========================================================================

def quantize_e4m3(x: "torch.Tensor") -> "torch.Tensor":
    """Quantize a float32/float16/bfloat16 GPU tensor to FP8 E4M3 uint8."""
    return _load_extension().quantize_e4m3(x.contiguous())


def quantize_e5m2(x: "torch.Tensor") -> "torch.Tensor":
    """Quantize a float32/float16/bfloat16 GPU tensor to FP8 E5M2 uint8."""
    return _load_extension().quantize_e5m2(x.contiguous())


def refresh_e4m3_shadow(
    x: "torch.Tensor",
    shadow: "torch.Tensor",
    quant_scale: "torch.Tensor",
) -> "torch.Tensor":
    """Refresh an existing E4M3 shadow buffer without an intermediate tensor."""
    return _load_extension().refresh_e4m3_shadow(
        x.contiguous(), shadow, quant_scale.contiguous()
    )


def quantize_e4m3_transpose(x: "torch.Tensor") -> "torch.Tensor":
    """Quantize a rank-2 tensor to E4M3 and transpose it in one GPU pass."""
    return _load_extension().quantize_e4m3_transpose(x.contiguous())


def quantize_e5m2_transpose(x: "torch.Tensor") -> "torch.Tensor":
    """Quantize a rank-2 tensor to E5M2 and transpose it in one GPU pass."""
    return _load_extension().quantize_e5m2_transpose(x.contiguous())


# ===========================================================================
# HIP Graph capture (PyTorch ROCm CUDAGraph frontend)
# ===========================================================================

def _graph_output_clone(value):
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_graph_output_clone(item) for item in value)
    if isinstance(value, list):
        return [_graph_output_clone(item) for item in value]
    raise TypeError("HIP graph callable must return a Tensor, tuple, or list of Tensors")


class Fp8GraphRunner:
    """Replay a fixed-shape FP8 inference callable through a HIP graph.

    PyTorch exposes HIP graph capture through its ``torch.cuda.CUDAGraph`` API
    on ROCm.  Inputs must keep the shape, dtype, device and contiguous layout
    used for capture.  ``replay`` returns independent output tensors by
    default; pass ``clone_output=False`` only when the result is consumed
    before the next replay.
    """

    def __init__(
        self,
        fn,
        *example_inputs: "torch.Tensor",
        warmup_iters: int = 3,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch with ROCm support is required for HIP graph capture")
        if not callable(fn):
            raise TypeError("fn must be callable")
        if not example_inputs:
            raise ValueError("at least one example input is required for HIP graph capture")
        if warmup_iters < 0:
            raise ValueError("warmup_iters must be non-negative")

        self._validate_inputs(example_inputs, reference=None)
        self._fn = fn
        self._device = example_inputs[0].device
        self._static_inputs = tuple(x.detach().clone() for x in example_inputs)

        # Keep allocations and lazy kernel setup outside the capture pool.
        current = torch.cuda.current_stream(self._device)
        warmup = torch.cuda.Stream(device=self._device)
        warmup.wait_stream(current)
        with torch.cuda.stream(warmup), torch.no_grad():
            for _ in range(warmup_iters):
                fn(*self._static_inputs)
        current.wait_stream(warmup)
        torch.cuda.synchronize(self._device)

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph), torch.no_grad():
            self._static_output = fn(*self._static_inputs)
        _graph_output_clone(self._static_output)  # validate result structure once

    @staticmethod
    def _validate_inputs(inputs, reference) -> None:
        if reference is not None and len(inputs) != len(reference):
            raise ValueError(f"expected {len(reference)} inputs, got {len(inputs)}")
        for index, value in enumerate(inputs):
            if not torch.is_tensor(value):
                raise TypeError(f"input {index} must be a torch.Tensor")
            if not value.is_cuda:
                raise ValueError(f"input {index} must be a HIP/CUDA tensor")
            if not value.is_contiguous():
                raise ValueError(f"input {index} must be contiguous for HIP graph replay")
            if value.requires_grad:
                raise ValueError("HIP graph capture is inference-only; inputs must not require gradients")
            if reference is not None:
                expected = reference[index]
                if (value.shape != expected.shape or value.dtype != expected.dtype
                        or value.device != expected.device):
                    raise ValueError(
                        f"input {index} metadata changed; HIP graph replay requires "
                        "the captured shape, dtype, and device"
                    )

    @property
    def static_output(self):
        """The graph-owned output storage, overwritten by each ``replay``."""
        return self._static_output

    def replay(self, *inputs: "torch.Tensor", clone_output: bool = True):
        """Copy ``inputs`` into captured storage, launch the HIP graph, and return output."""
        self._validate_inputs(inputs, self._static_inputs)
        with torch.no_grad():
            for static, value in zip(self._static_inputs, inputs):
                static.copy_(value, non_blocking=True)
            self._graph.replay()
        return _graph_output_clone(self._static_output) if clone_output else self._static_output


def capture_hip_graph(
    fn,
    *example_inputs: "torch.Tensor",
    warmup_iters: int = 3,
) -> Fp8GraphRunner:
    """Capture ``fn(*example_inputs)`` into a replayable HIP graph on ROCm.

    Use this for stable-shape inference only.  The callable must not allocate
    conditionally or depend on CPU-visible tensor values while it is captured.
    """
    return Fp8GraphRunner(fn, *example_inputs, warmup_iters=warmup_iters)


def quantize_e5m2_stochastic(x: "torch.Tensor", seed: Optional[int] = None) -> "torch.Tensor":
    """Quantize to FP8 E5M2 using stochastic rounding on-device.

    Passing ``seed`` makes the result exactly reproducible for the same input.
    If omitted, a process-local counter is mixed with ``torch.initial_seed()``.
    """
    if seed is None:
        seed = _next_stochastic_e5m2_seed()
    return _load_extension().quantize_e5m2_stochastic(
        x.contiguous(), int(seed) & 0xFFFFFFFFFFFFFFFF
    )


def quantize_e4m3_blockwise(
    x: "torch.Tensor",
    block_size: int = 32,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Quantize to FP8 E4M3 with one FP32 dequant scale per last-dim block.

    Returns ``(fp8_bytes, scales)``. For an input shape ``[..., K]``, scales has
    shape ``[..., ceil(K / block_size)]`` and represents ``real ~= fp8 * scale``.
    """
    return _load_extension().quantize_e4m3_blockwise(x.contiguous(), int(block_size))


def quantize_e5m2_blockwise(
    x: "torch.Tensor",
    block_size: int = 32,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Quantize to FP8 E5M2 with one FP32 dequant scale per last-dim block."""
    return _load_extension().quantize_e5m2_blockwise(x.contiguous(), int(block_size))


def quantize_e5m2_blockwise_stochastic(
    x: "torch.Tensor",
    block_size: int = 32,
    seed: Optional[int] = None,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Block-wise E5M2 quantization with stochastic rounding inside each scale block."""
    if seed is None:
        seed = _next_stochastic_e5m2_seed()
    return _load_extension().quantize_e5m2_blockwise_stochastic(
        x.contiguous(), int(block_size), int(seed) & 0xFFFFFFFFFFFFFFFF
    )


def refresh_fp8_blockwise_shadow(
    weight: "torch.Tensor",
    weight_fp8: Optional["torch.Tensor"] = None,
    weight_scales: Optional["torch.Tensor"] = None,
    block_size: int = 32,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Quantize a master weight into block-wise E4M3 shadow storage.

    If ``weight_fp8`` and/or ``weight_scales`` are provided, they are updated
    in-place and returned. Otherwise new tensors are allocated.
    """
    q, scales = quantize_e4m3_blockwise(weight, block_size)
    if weight_fp8 is not None:
        weight_fp8.copy_(q)
        q = weight_fp8
    if weight_scales is not None:
        weight_scales.copy_(scales)
        scales = weight_scales
    return q, scales


def dequantize_e4m3(x: "torch.Tensor") -> "torch.Tensor":
    """Dequantize an FP8 E4M3 uint8 tensor to float32 on-device."""
    return _load_extension().dequantize_e4m3(x.contiguous())


def dequantize_e5m2(x: "torch.Tensor") -> "torch.Tensor":
    """Dequantize an FP8 E5M2 uint8 tensor to float32 on-device."""
    return _load_extension().dequantize_e5m2(x.contiguous())


def dequantize_e4m3_blockwise(
    x: "torch.Tensor",
    scales: "torch.Tensor",
    block_size: int = 32,
) -> "torch.Tensor":
    """Dequantize block-wise FP8 E4M3 bytes using FP32 dequant scales."""
    return _load_extension().dequantize_e4m3_blockwise(
        x.contiguous(), scales.contiguous(), int(block_size)
    )


def dequantize_e5m2_blockwise(
    x: "torch.Tensor",
    scales: "torch.Tensor",
    block_size: int = 32,
) -> "torch.Tensor":
    """Dequantize block-wise FP8 E5M2 bytes using FP32 dequant scales."""
    return _load_extension().dequantize_e5m2_blockwise(
        x.contiguous(), scales.contiguous(), int(block_size)
    )


def adafactor_row_col_mean_square(
    grad: "torch.Tensor",
    eps: float = 0.0,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Compute Adafactor 2-D row/column mean-square stats on-device."""
    return _load_extension().adafactor_row_col_mean_square(grad.contiguous(), float(eps))


GGML_Q_TO_FP8_SUPPORTED = {
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
}


def dequantize_q_to_fp8(
    packed: "torch.Tensor",
    type_num: int,
    n_per_row: int,
    e5m2: bool = False,
) -> "torch.Tensor":
    """Dequantize GGML Q-type packed bytes directly to FP8 on GPU (zero copies).

    The input ``packed`` must be a contiguous uint8 CUDA tensor containing
    raw GGML Q-block bytes already resident on the GPU.  The output is a
    ``[nrows, n_per_row]`` uint8 CUDA tensor of raw FP8 bytes (E4M3 by
    default, E5M2 if ``e5m2=True``).

    Supported types: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K..Q6_K.
    """
    return _load_extension().dequantize_q_to_fp8(
        packed.contiguous(), int(type_num), int(n_per_row), bool(e5m2)
    )


def dequantize_q_to_e4m3(
    packed: "torch.Tensor",
    type_num: int,
    n_per_row: int,
) -> "torch.Tensor":
    """Shortcut: Q-type -> FP8 E4M3 on GPU (zero copies)."""
    return dequantize_q_to_fp8(packed, type_num, n_per_row, e5m2=False)


def dequantize_q_to_e5m2(
    packed: "torch.Tensor",
    type_num: int,
    n_per_row: int,
) -> "torch.Tensor":
    """Shortcut: Q-type -> FP8 E5M2 on GPU (zero copies)."""
    return dequantize_q_to_fp8(packed, type_num, n_per_row, e5m2=True)


def dequantize_mxfp4_to_fp8(
    packed_values: "torch.Tensor",
    block_scales: "torch.Tensor",
    n_per_row: int,
) -> "torch.Tensor":
    """Expand OCP MXFP4 E2M1 + UE8M0 weights to E4M3 bytes on the GPU.

    ``packed_values`` contains two E2M1 values per byte, with the even value
    in the low nibble. ``block_scales`` contains one raw UE8M0 byte per 32
    values. Both tensors must be contiguous ``torch.uint8`` GPU tensors.

    This is deliberately an emulation bridge: it enables MXFP4 checkpoints to
    execute through hipBLASLt's supported FP8 kernels, but does not provide
    native FP4 compute or native-MXFP4 throughput.
    """
    _warn_mxfp4_emulation()
    return _load_extension().dequantize_mxfp4_to_fp8(
        packed_values.contiguous(), block_scales.contiguous(), int(n_per_row)
    )


def _mxfp4_input_scale(input_2d: "torch.Tensor") -> "torch.Tensor":
    """Create a device-resident E4M3 quantization multiplier for activations."""
    amax = input_2d.detach().abs().amax().to(dtype=torch.float32)
    valid = torch.isfinite(amax) & (amax > 0)
    return torch.where(valid, torch.full_like(amax, 448.0) / amax, torch.ones_like(amax))


def mxfp4_linear_forward(
    input: "torch.Tensor",
    packed_weight: "torch.Tensor",
    block_scales: "torch.Tensor",
    bias: Optional["torch.Tensor"] = None,
    n_per_row: Optional[int] = None,
    input_scale: Optional[Union["torch.Tensor", float]] = None,
) -> "torch.Tensor":
    """Inference-only MXFP4 linear through the standard hipBLASLt FP8 path.

    MXFP4 remains the persistent weight storage. Every invocation expands the
    current weight matrix to a temporary E4M3 buffer before calling
    ``torch._scaled_mm`` (hipBLASLt). Set ``n_per_row`` for flat packed input;
    for a ``[out_features, in_features / 2]`` packed tensor it is inferred.

    If the installed hipBLASLt FP8 path declines the GEMM, the function falls
    back to a normal BF16/FP16/FP32 ``F.linear`` operation. It never selects
    hip_quant's experimental custom WMMA kernel.
    """
    if input.ndim < 1:
        raise ValueError("mxfp4_linear_forward: input must have at least one dimension")
    if n_per_row is None:
        if packed_weight.ndim != 2:
            raise ValueError(
                "mxfp4_linear_forward: n_per_row is required when packed_weight is not 2-D"
            )
        n_per_row = int(packed_weight.shape[1]) * 2
    n_per_row = int(n_per_row)
    if n_per_row <= 0 or n_per_row % MXFP4_BLOCK_SIZE:
        raise ValueError("mxfp4_linear_forward: n_per_row must be a positive multiple of 32")
    if input.shape[-1] != n_per_row:
        raise ValueError(
            f"mxfp4_linear_forward: input last dimension ({input.shape[-1]}) "
            f"does not match n_per_row ({n_per_row})"
        )

    _warn_mxfp4_emulation()
    original_shape = input.shape
    input_2d = input.reshape(-1, n_per_row).contiguous()
    weight_fp8 = dequantize_mxfp4_to_fp8(packed_weight, block_scales, n_per_row)

    if input_scale is None:
        input_scale_t = _mxfp4_input_scale(input_2d)
    else:
        input_scale_t = _fp8_scale_tensor(input_scale, "input_scale", input_2d.device)
    input_fp8 = quantize_e4m3((input_2d * input_scale_t).contiguous())

    result = _hipblaslt_fp8_linear_forward_prequant(
        input_fp8,
        weight_fp8,
        bias,
        input_scale_t.reciprocal(),
        1.0,
        input_2d.dtype,
    )
    if result is None:
        # Correctness fallback for installations whose hipBLASLt does not
        # expose FP8 _scaled_mm. This remains deliberately outside the custom
        # WMMA path requested by the caller.
        expanded_weight = dequantize_e4m3(weight_fp8).to(dtype=input_2d.dtype)
        result = F.linear(input_2d, expanded_weight, bias)
    return result.reshape(*original_shape[:-1], result.shape[-1])


def native_mxfp4_contract() -> Dict[str, Union[str, int]]:
    """Return the fixed native CDNA4 MXFP4 hipBLASLt ABI contract.

    This query is useful in CI and does not launch a GEMM, so it can be used
    on Radeon hardware or a development machine without a GPU of its own.
    """
    return dict(_load_extension().native_mxfp4_contract())


def native_mxfp4_capability() -> Dict[str, Union[str, bool]]:
    """Report whether the current device can execute native MXFP4.

    ``available`` is true only for CDNA4 ``gfx950`` with a usable hipBLASLt
    runtime. It is intentionally false on RDNA and earlier CDNA devices; use
    :func:`mxfp4_linear_forward` there for the explicitly warned FP8 bridge.
    """
    return dict(_load_extension().native_mxfp4_capability())


def _native_mxfp4_output_dtype_code(output_dtype: "torch.dtype") -> int:
    if output_dtype == torch.float32:
        return 0
    if output_dtype == torch.float16:
        return 1
    if output_dtype == torch.bfloat16:
        return 2
    raise TypeError(
        "native_mxfp4_linear_forward: output_dtype must be torch.float32, "
        "torch.float16, or torch.bfloat16"
    )


def _validate_native_mxfp4_operands(
    a_values: "torch.Tensor",
    a_scales: "torch.Tensor",
    b_values: "torch.Tensor",
    b_scales: "torch.Tensor",
) -> Tuple[int, int, int]:
    """Validate the hardware-independent portion of the native MXFP4 ABI."""
    operands = {
        "a_values": a_values,
        "a_scales": a_scales,
        "b_values": b_values,
        "b_scales": b_scales,
    }
    for name, tensor in operands.items():
        if tensor.ndim != 2:
            raise ValueError(f"native_mxfp4_linear_forward: {name} must be rank 2")
        if tensor.dtype != torch.uint8:
            raise TypeError(
                f"native_mxfp4_linear_forward: {name} must be torch.uint8 raw storage"
            )

    m = int(a_values.shape[0])
    k = int(a_values.shape[1]) * 2
    n = int(b_values.shape[0])
    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError("native_mxfp4_linear_forward: M, N, and K must be positive")
    if m % 16 or n % 16 or k % 128:
        raise ValueError(
            "native_mxfp4_linear_forward: native hipBLASLt MXFP4 requires "
            "M and N multiples of 16 and K a multiple of 128"
        )
    if int(b_values.shape[1]) * 2 != k:
        raise ValueError(
            "native_mxfp4_linear_forward: a_values and b_values must have the same logical K"
        )
    if tuple(a_scales.shape) != (m, k // MXFP4_BLOCK_SIZE):
        raise ValueError(
            "native_mxfp4_linear_forward: a_scales must have shape [M, K / 32]"
        )
    if tuple(b_scales.shape) != (n, k // MXFP4_BLOCK_SIZE):
        raise ValueError(
            "native_mxfp4_linear_forward: b_scales must have shape [N, K / 32]"
        )
    return m, n, k


def native_mxfp4_linear_forward(
    a_values: "torch.Tensor",
    a_scales: "torch.Tensor",
    b_values: "torch.Tensor",
    b_scales: "torch.Tensor",
    output_dtype: "torch.dtype" = None,
) -> "torch.Tensor":
    """Run a true native MXFP4 E2M1 GEMM on CDNA4 ``gfx950`` only.

    All four storage tensors are raw ``uint8`` GPU tensors. ``a_values`` and
    ``b_values`` have logical shapes ``[M, K]`` and ``[N, K]`` packed as
    ``[M, K / 2]`` and ``[N, K / 2]`` (low nibble is the even E2M1 value).
    Their matching UE8M0 scale tensors have shapes ``[M, K / 32]`` and
    ``[N, K / 32]``. The result is ``A @ B.T`` with shape ``[M, N]``.

    This invokes hipBLASLt with ``HIP_R_4F_E2M1`` and Vec32 UE8M0 scale
    attributes; it does not decode, upcast, or route through the FP8 bridge.
    The native library imposes ``M % 16 == N % 16 == 0``, ``K % 128 == 0``,
    batch size one, and no fused bias/activation epilogue. It deliberately
    raises on every non-``gfx950`` architecture rather than emulating.
    """
    _validate_native_mxfp4_operands(a_values, a_scales, b_values, b_scales)
    if output_dtype is None:
        output_dtype = torch.bfloat16
    output_dtype_code = _native_mxfp4_output_dtype_code(output_dtype)

    # c10's device-guard headers are not currently MSVC-clean in the Windows
    # ROCm wheel. Establish the corresponding PyTorch device guard here so the
    # C++ binding can safely use the current stream without mutating device
    # state itself. CPU tensors still reach mocked extensions in CPU contract
    # tests; the real binding then gives the GPU-specific error.
    if a_values.is_cuda:
        with torch.cuda.device(a_values.device):
            return _load_extension().native_mxfp4_linear_forward(
                a_values.contiguous(), a_scales.contiguous(),
                b_values.contiguous(), b_scales.contiguous(), output_dtype_code,
            )
    return _load_extension().native_mxfp4_linear_forward(
        a_values.contiguous(), a_scales.contiguous(),
        b_values.contiguous(), b_scales.contiguous(), output_dtype_code,
    )


# ===========================================================================
# Phase 3: Autograd-safe FP8 linear
# ===========================================================================

def _sim_fp8_e4m3(x: "torch.Tensor") -> "torch.Tensor":
    """Quantize-then-dequantize in E4M3 — applies FP8 quantization noise."""
    return dequantize_e4m3(quantize_e4m3(x.contiguous()))


def _sim_fp8_e5m2(x: "torch.Tensor") -> "torch.Tensor":
    """Quantize-then-dequantize in E5M2 — applies FP8 quantization noise."""
    return dequantize_e5m2(quantize_e5m2(x.contiguous()))


def _prepare_e5m2_backward_grad_output(
    grad_output: "torch.Tensor",
) -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:
    if not _stochastic_e5m2_enabled():
        return grad_output, None
    grad_output_fp8 = quantize_e5m2_stochastic(grad_output)
    grad_output_e5m2 = dequantize_e5m2(grad_output_fp8).to(grad_output.dtype).contiguous()
    return grad_output_e5m2, grad_output_fp8


def _quantize_e5m2_for_backward(x: "torch.Tensor") -> "torch.Tensor":
    """Apply the configured E5M2 rounding mode without dequantizing again."""
    return quantize_e5m2_stochastic(x) if _stochastic_e5m2_enabled() else quantize_e5m2(x)


def _raw_fp8_linear_backward(
    grad_output: "torch.Tensor",
    weight: "torch.Tensor",
    input_f32: "torch.Tensor",
    weight_scale: Union["torch.Tensor", float] = 1.0,
    input_scale: Union["torch.Tensor", float] = 1.0,
    grad_output_fp8: Optional["torch.Tensor"] = None,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Run the custom backward with each E5M2 operand materialized once.

    The older WMMA path converted the weight/input values inside every output
    tile.  Keeping raw E5M2 buffers gives both gradient GEMMs the same
    pre-quantized grad-output and turns those inner-loop conversions into
    coalesced, one-pass conversion kernels.
    """
    if grad_output_fp8 is None:
        grad_output_fp8 = _quantize_e5m2_for_backward(grad_output.contiguous())

    ext = _load_extension()
    if hasattr(ext, "fp8_linear_backward_input_raw_fp8") and hasattr(
        ext, "fp8_linear_backward_weight_raw_fp8"
    ):
        weight_scale_t = _fp8_scale_tensor(weight_scale, "weight_scale", weight.device)
        input_scale_t = _fp8_scale_tensor(input_scale, "input_scale", input_f32.device)
        weight_fp8 = _quantize_e5m2_for_backward((weight.contiguous() * weight_scale_t).contiguous())
        input_fp8 = _quantize_e5m2_for_backward((input_f32.contiguous() * input_scale_t).contiguous())
        grad_input = fp8_linear_backward_input_raw_fp8(
            grad_output_fp8, weight_fp8, grad_output,
            _scale_to_float(weight_scale_t.reciprocal()),
        )
        grad_weight = fp8_linear_backward_weight_raw_fp8(
            grad_output_fp8, input_fp8, grad_output,
            _scale_to_float(input_scale_t.reciprocal()),
        )
        return grad_input, grad_weight

    # Compatibility with an already-installed extension that predates the raw
    # two-operand ABI. It still reuses grad-output bytes, just not the other
    # repeated casts, so source-only upgrades remain safe.
    grad_input = fp8_linear_backward_input_fp8_grad(
        grad_output_fp8, grad_output, weight, _scale_to_float(weight_scale)
    )
    grad_weight = fp8_linear_backward_weight_fp8_grad(
        grad_output_fp8, grad_output, input_f32, _scale_to_float(input_scale)
    )
    return grad_input, grad_weight


def _cpu_fp8_linear_forward(
    input:        "torch.Tensor",
    weight:       "torch.Tensor",
    bias:         Optional["torch.Tensor"],
    input_scale:  float = 1.0,
    weight_scale: float = 1.0,
) -> "torch.Tensor":
    input_sim = _sim_fp8_e4m3((input * input_scale).contiguous()) * (1.0 / input_scale)
    weight_sim = _sim_fp8_e4m3((weight * weight_scale).contiguous()) * (1.0 / weight_scale)
    out = input_sim.float() @ weight_sim.float().t()
    if bias is not None:
        out = out + bias.float().unsqueeze(0)
    return out.to(input.dtype)


def _cpu_fp8_linear_backward(
    grad_output:  "torch.Tensor",
    weight:       "torch.Tensor",
    input_f32:    "torch.Tensor",
    weight_scale: float = 1.0,
    input_scale:  float = 1.0,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    grad_sim = _sim_fp8_e5m2(grad_output.contiguous()).float()
    weight_sim = _sim_fp8_e5m2((weight * weight_scale).contiguous()).float() * (1.0 / weight_scale)
    input_sim = _sim_fp8_e5m2((input_f32 * input_scale).contiguous()).float() * (1.0 / input_scale)
    grad_input = grad_sim @ weight_sim
    grad_weight = grad_sim.t() @ input_sim
    return grad_input.to(grad_output.dtype), grad_weight.to(weight.dtype)


def _pair(value: Union[int, Tuple[int, int]], name: str) -> Tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, tuple) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError(f"{name} must be an int or a 2-tuple")


def _conv2d_output_hw(
    input_h: int,
    input_w: int,
    kernel_h: int,
    kernel_w: int,
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
) -> Tuple[int, int]:
    out_h = (input_h + 2 * padding[0] - dilation[0] * (kernel_h - 1) - 1) // stride[0] + 1
    out_w = (input_w + 2 * padding[1] - dilation[1] * (kernel_w - 1) - 1) // stride[1] + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError("fp8_conv2d output spatial size must be positive")
    return out_h, out_w


# ---------------------------------------------------------------------------
# Fp8LinearFunction — unscaled
# ---------------------------------------------------------------------------

class Fp8LinearFunction(torch.autograd.Function):
    """Autograd-safe fake-FP8 linear operator.

    Forward  : E4M3 noise on input and weight, then float32 matmul.
    Backward : E5M2 noise on grad_output.

    Activation compression (Feature 1)
    -----------------------------------
    Instead of saving ``input_f32`` (float32, 4 bytes/element) in the autograd
    graph, we save ``input_fp8`` (uint8, 1 byte/element) and dequantize on
    demand in backward.  This cuts the activation portion of the autograd graph
    VRAM by 4×.  For a 512-token, d_model=512 batch of 8 that is:
      8 × 512 × 512 × 4 bytes = 8 MB  →  2 MB  per linear layer.

    Backward accuracy
    -----------------
    ``grad_weight`` uses the FP8-simulated activation (consistent with what
    flowed through forward).  ``grad_input`` uses the full-precision weight
    (master-weight convention for mixed-precision training).
    """

    @staticmethod
    def forward(
        ctx,
        input:  "torch.Tensor",
        weight: "torch.Tensor",
        bias:   Optional["torch.Tensor"],
    ) -> "torch.Tensor":

        input_c = input.contiguous()
        weight_c = weight.contiguous()

        # Save compressed activation for backward; forward GEMM quantizes A/B
        # in-register and uses gfx12 FP8 WMMA.
        input_fp8 = quantize_e4m3(input_c)   # uint8 — 4× smaller

        # Save compressed activation + full-precision weight
        ctx.has_bias = bias is not None
        if bias is not None:
            ctx.save_for_backward(input_fp8, weight, bias)
        else:
            ctx.save_for_backward(input_fp8, weight)

        if not input_c.is_cuda:
            return _cpu_fp8_linear_forward(input_c, weight_c, bias)

        out = _hipblaslt_fp8_linear_forward(input_c, weight_c, bias)
        if out is not None:
            return out

        return fp8_linear_forward_fp8_input(input_fp8, weight_c, input_c, 1.0, 1.0, bias)

    @staticmethod
    def backward(ctx, grad_output: "torch.Tensor"):
        if ctx.has_bias:
            input_fp8, weight, bias = ctx.saved_tensors
        else:
            input_fp8, weight = ctx.saved_tensors
            bias = None

        # Decompress activation on demand
        input_f32 = dequantize_e4m3(input_fp8)

        grad_output_c = grad_output.contiguous()

        if not grad_output_c.is_cuda:
            grad_input, grad_weight = _cpu_fp8_linear_backward(
                grad_output_c, weight, input_f32, weight_scale=1.0, input_scale=1.0
            )
            grad_bias = grad_output.sum(0) if bias is not None else None
            return grad_input, grad_weight, grad_bias

        grad_output_for_backward, grad_output_fp8 = _prepare_e5m2_backward_grad_output(grad_output_c)

        # Try hipBLASLt e5m2 backward path first
        hipblaslt_result = _hipblaslt_fp8_backward(
            grad_output_for_backward, weight, input_f32, weight_scale=1.0, input_scale=1.0
        )
        if hipblaslt_result is not None:
            grad_input, grad_weight = hipblaslt_result
            grad_bias = grad_output.sum(0) if bias is not None else None
            return grad_input, grad_weight, grad_bias

        grad_input, grad_weight = _raw_fp8_linear_backward(
            grad_output_c, weight, input_f32, 1.0, 1.0, grad_output_fp8
        )
        if grad_weight.dtype != weight.dtype:
            grad_weight = grad_weight.to(weight.dtype)
        grad_bias   = grad_output.sum(0) if bias is not None else None

        return grad_input, grad_weight, grad_bias


# ---------------------------------------------------------------------------
# Fp8ScaledLinearFunction — E4M3 with per-tensor amax scaling
# ---------------------------------------------------------------------------

class Fp8ScaledLinearFunction(torch.autograd.Function):
    """Autograd-safe scaled FP8 linear (activation compression + amax scaling).

    Two improvements over ``Fp8LinearFunction``:
    1. Per-tensor amax scaling: scales input/weight to fill ±448 before
       quantizing, then divides out after, reducing quantization noise.
    2. Activation compression: saves uint8 FP8 bytes in ctx (not float32),
       storing ``input_scale`` as a ctx attribute to allow correct
       scaled dequantization in backward.

    input_scale and weight_scale may be device scalars; neither is differentiated.
    """

    @staticmethod
    def forward(
        ctx,
        input:        "torch.Tensor",
        weight:       "torch.Tensor",
        bias:         Optional["torch.Tensor"],
        input_scale:  float,
        weight_scale: float,
        input_fp8:    Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":

        # Snapshot mutable delayed-scale metadata for this autograd context
        # without copying it to the host.
        if torch.is_tensor(input_scale):
            input_scale = input_scale.detach().clone()
        if torch.is_tensor(weight_scale):
            weight_scale = weight_scale.detach().clone()

        input_c = input.contiguous()
        weight_c = weight.contiguous()

        # A module using delayed amax may already have produced this activation
        # in the fused scan+quantize kernel. Direct callers retain the original
        # quantization behavior through the optional argument.
        ctx._num_inputs = len(ctx.needs_input_grad)
        if input_fp8 is None:
            input_fp8 = quantize_e4m3((input_c * input_scale).contiguous())
        else:
            input_fp8 = input_fp8.contiguous()
        weight_fp8 = quantize_e4m3((weight_c * weight_scale).contiguous())

        ctx.has_bias    = bias is not None
        ctx.input_scale = input_scale          # needed to dequantise in backward
        ctx.weight_scale = weight_scale
        if bias is not None:
            ctx.save_for_backward(input_fp8, weight, bias)
        else:
            ctx.save_for_backward(input_fp8, weight)

        if not input_c.is_cuda:
            return _cpu_fp8_linear_forward(input_c, weight_c, bias, input_scale, weight_scale)

        out = _hipblaslt_fp8_linear_forward_prequant(
            input_fp8,
            weight_fp8,
            bias,
            1.0 / input_scale if torch.is_tensor(input_scale) else (1.0 / _scale_to_float(input_scale)),
            1.0 / weight_scale if torch.is_tensor(weight_scale) else (1.0 / _scale_to_float(weight_scale)),
            input_c.dtype,
        )
        if out is not None:
            return out

        return fp8_linear_forward_fp8_input(
            input_fp8, weight_c, input_c,
            _scale_to_float(input_scale), _scale_to_float(weight_scale), bias
        )

    @staticmethod
    def backward(ctx, grad_output: "torch.Tensor"):
        if ctx.has_bias:
            input_fp8, weight, bias = ctx.saved_tensors
        else:
            input_fp8, weight = ctx.saved_tensors
            bias = None

        # Decompress activation using the saved scale
        input_f32 = dequantize_e4m3(input_fp8) * (1.0 / ctx.input_scale)

        grad_output_c = grad_output.contiguous()

        if not grad_output_c.is_cuda:
            grad_input, grad_weight = _cpu_fp8_linear_backward(
                grad_output_c, weight, input_f32,
                weight_scale=ctx.weight_scale, input_scale=1.0
            )
            grad_bias = grad_output.sum(0) if bias is not None else None
            return (grad_input, grad_weight, grad_bias, None, None, None)[:ctx._num_inputs]

        grad_output_for_backward, grad_output_fp8 = _prepare_e5m2_backward_grad_output(grad_output_c)

        # Try hipBLASLt e5m2 backward path first
        hipblaslt_result = _hipblaslt_fp8_backward(
            grad_output_for_backward, weight, input_f32,
            weight_scale=ctx.weight_scale, input_scale=1.0
        )
        if hipblaslt_result is not None:
            grad_input, grad_weight = hipblaslt_result
            grad_bias = grad_output.sum(0) if bias is not None else None
            return (grad_input, grad_weight, grad_bias, None, None, None)[:ctx._num_inputs]

        grad_input, grad_weight = _raw_fp8_linear_backward(
            grad_output_c, weight, input_f32,
            ctx.weight_scale, 1.0, grad_output_fp8,
        )
        if grad_weight.dtype != weight.dtype:
            grad_weight = grad_weight.to(weight.dtype)
        grad_bias   = grad_output.sum(0) if bias is not None else None

        return (grad_input, grad_weight, grad_bias, None, None, None)[:ctx._num_inputs]


# ---------------------------------------------------------------------------
# Fp8Linear — unscaled nn.Module
# ---------------------------------------------------------------------------

class Fp8Linear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` using fake-FP8 forward/backward.

    Master weights stored as float32.  E4M3 forward noise, E5M2 backward noise.
    No per-tensor scaling — use ``Fp8ScaledLinear`` for LLM training where
    activation magnitudes vary widely across layers.

    Args:
        in_features:  input size.
        out_features: output size.
        bias:         learnable bias (default True).
        device:       device for parameters.
        dtype:        dtype for parameters (default float32).

    Shape:
        Input  : ``(*, in_features)``
        Output : ``(*, out_features)``
    """

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        bias:         bool = True,
        device:       Optional[Union[str, "torch.device"]] = None,
        dtype:        Optional["torch.dtype"] = None,
    ) -> None:
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.in_features  = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, **factory)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, **factory))
        else:
            self.register_parameter("bias", None)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Match nn.Linear's default init exactly
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features).contiguous()
        out  = Fp8LinearFunction.apply(x_2d, self.weight, self.bias)
        return out.reshape(*orig_shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )

    @classmethod
    def from_linear(cls, linear: "nn.Linear") -> "Fp8Linear":
        """Create an ``Fp8Linear`` by copying weights from an ``nn.Linear``.

        Example::

            fp8_layer = Fp8Linear.from_linear(model.lm_head)
        """
        layer = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        with torch.no_grad():
            layer.weight.copy_(linear.weight)
            if linear.bias is not None:
                layer.bias.copy_(linear.bias)
        return layer

    def to_linear(self) -> "nn.Linear":
        """Convert back to a standard ``nn.Linear`` (copies weights)."""
        linear = nn.Linear(
            self.in_features, self.out_features,
            bias=self.bias is not None,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        with torch.no_grad():
            linear.weight.copy_(self.weight)
            if self.bias is not None:
                linear.bias.copy_(self.bias)
        return linear


# ---------------------------------------------------------------------------
# Fp8ScaledLinear — per-tensor amax scaling (recommended for LLM training)
# ---------------------------------------------------------------------------

class Fp8ScaledLinear(nn.Module):
    """``Fp8Linear`` with per-tensor delayed amax scaling.

    At each forward call the input and weight amaxes are measured and stored
    in rolling ``Fp8TensorMeta`` histories.  The derived scale factors are
    used to fill the ±448 E4M3 range before quantizing, then divided out
    after dequantizing.  This keeps quantization noise low even when
    activation magnitudes are much smaller than 448 (common in early training
    and in the first few layers of an LLM).

    The amax measurement runs inside ``torch.no_grad()`` and does not affect
    the autograd graph. Scale values remain device-resident on the hipBLASLt
    path, avoiding per-layer host synchronization.

    Args:
        in_features:  input size.
        out_features: output size.
        bias:         learnable bias (default True).
        history_len:  rolling window length for amax history (default 16).
        device:       device for parameters.
        dtype:        dtype for parameters.
    """

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        bias:         bool = True,
        history_len:  int  = 16,
        device:       Optional[Union[str, "torch.device"]] = None,
        dtype:        Optional["torch.dtype"] = None,
    ) -> None:
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.in_features  = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, **factory)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, **factory))
        else:
            self.register_parameter("bias", None)

        meta_device = str(self.weight.device)
        self.input_meta  = Fp8TensorMeta(history_len=history_len, device=meta_device)
        self.weight_meta = Fp8TensorMeta(history_len=history_len, device=meta_device)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features).contiguous()

        fused_input_fp8 = None
        input_scale_for_forward = None
        with torch.no_grad():
            if self.input_meta.scale.device != x_2d.device:
                self.input_meta.to(x_2d.device)
            if self.weight_meta.scale.device != self.weight.device:
                self.weight_meta.to(self.weight.device)
            ext = _load_extension()
            if x_2d.is_cuda and hasattr(ext, "quantize_e4m3_delayed_amax"):
                fused_input_fp8, input_scale_for_forward = self.input_meta.quantize_e4m3_delayed(x_2d)
            else:
                self.input_meta.update(x_2d)
            self.weight_meta.update(self.weight)

        if fused_input_fp8 is None:
            out = Fp8ScaledLinearFunction.apply(
                x_2d, self.weight, self.bias,
                self.input_meta.scale, self.weight_meta.scale,
            )
        else:
            out = Fp8ScaledLinearFunction.apply(
                x_2d, self.weight, self.bias,
                input_scale_for_forward, self.weight_meta.scale, fused_input_fp8,
            )
        return out.reshape(*orig_shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"history_len={self.input_meta._history_len}"
        )

    @classmethod
    def from_linear(
        cls,
        linear:      "nn.Linear",
        history_len: int = 16,
    ) -> "Fp8ScaledLinear":
        """Create an ``Fp8ScaledLinear`` from an existing ``nn.Linear``."""
        layer = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            history_len=history_len,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        with torch.no_grad():
            layer.weight.copy_(linear.weight)
            if linear.bias is not None:
                layer.bias.copy_(linear.bias)
        return layer

    def to_linear(self) -> "nn.Linear":
        """Convert back to a standard ``nn.Linear``."""
        linear = nn.Linear(
            self.in_features, self.out_features,
            bias=self.bias is not None,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        with torch.no_grad():
            linear.weight.copy_(self.weight)
            if self.bias is not None:
                linear.bias.copy_(self.bias)
        return linear


# ---------------------------------------------------------------------------
# FP8 Conv2d — im2col lowering into the hipBLASLt-backed FP8 linear path
# ---------------------------------------------------------------------------

def fp8_conv2d(
    input:        "torch.Tensor",
    weight:       "torch.Tensor",
    bias:         Optional["torch.Tensor"] = None,
    stride:       Union[int, Tuple[int, int]] = 1,
    padding:      Union[int, Tuple[int, int]] = 0,
    dilation:     Union[int, Tuple[int, int]] = 1,
    groups:       int = 1,
    input_scale:  Optional[float] = None,
    weight_scale: Optional[float] = None,
) -> "torch.Tensor":
    """FP8 E4M3 conv2d using unfold + the existing FP8 linear backend.

    The lowered matrix multiply is routed through ``Fp8ScaledLinearFunction``,
    so hipBLASLt is used by default when PyTorch exposes ``torch._scaled_mm``;
    the custom gfx12 WMMA kernel remains the fallback path.
    """
    if input.dim() != 4:
        raise ValueError("fp8_conv2d: input must be NCHW with shape [N, C, H, W]")
    if weight.dim() != 4:
        raise ValueError("fp8_conv2d: weight must have shape [out_channels, in_channels/groups, kH, kW]")
    if not input.is_cuda or not weight.is_cuda:
        raise RuntimeError("fp8_conv2d: input and weight must be CUDA/HIP tensors")
    if input.device != weight.device:
        raise RuntimeError("fp8_conv2d: input and weight must be on the same device")
    if bias is not None and (not bias.is_cuda or bias.device != input.device):
        raise RuntimeError("fp8_conv2d: bias must be on the same CUDA/HIP device as input")

    stride = _pair(stride, "stride")
    padding = _pair(padding, "padding")
    dilation = _pair(dilation, "dilation")
    groups = int(groups)
    if groups <= 0:
        raise ValueError("fp8_conv2d: groups must be positive")

    batch, in_channels, input_h, input_w = input.shape
    out_channels, weight_in_channels, kernel_h, kernel_w = weight.shape
    if in_channels % groups != 0 or out_channels % groups != 0:
        raise ValueError("fp8_conv2d: in_channels and out_channels must be divisible by groups")
    if weight_in_channels != in_channels // groups:
        raise ValueError("fp8_conv2d: weight channel dimension does not match input channels/groups")
    if bias is not None and (bias.dim() != 1 or bias.size(0) != out_channels):
        raise ValueError("fp8_conv2d: bias must have shape [out_channels]")

    out_h, out_w = _conv2d_output_hw(
        int(input_h), int(input_w), int(kernel_h), int(kernel_w), stride, padding, dilation
    )

    with torch.no_grad():
        if input_scale is None:
            input_scale = _scale_to_float(448.0 / input.detach().abs().max().clamp(min=1e-12))
        if weight_scale is None:
            weight_scale = _scale_to_float(448.0 / weight.detach().abs().max().clamp(min=1e-12))

    columns = F.unfold(input.contiguous(), (kernel_h, kernel_w), dilation, padding, stride)
    locations = columns.size(-1)
    input_2d = columns.transpose(1, 2).reshape(batch * locations, -1).contiguous()
    weight_2d = weight.contiguous().reshape(out_channels, -1)

    if groups == 1:
        k_pad = (-input_2d.size(1)) % 16
        if k_pad:
            input_2d = F.pad(input_2d, (0, k_pad))
            weight_2d = F.pad(weight_2d, (0, k_pad))
        out_pad = (-out_channels) % 16
        matmul_bias = bias
        if out_pad:
            weight_2d = F.pad(weight_2d, (0, 0, 0, out_pad))
            matmul_bias = None if bias is None else F.pad(bias, (0, out_pad))
        out_2d = Fp8ScaledLinearFunction.apply(
            input_2d, weight_2d, matmul_bias, float(input_scale), float(weight_scale)
        )
        if out_pad:
            out_2d = out_2d[:, :out_channels]
    else:
        in_per_group = weight_in_channels * kernel_h * kernel_w
        out_per_group = out_channels // groups
        chunks = []
        for group_idx in range(groups):
            in_start = group_idx * in_per_group
            out_start = group_idx * out_per_group
            group_bias = None if bias is None else bias.narrow(0, out_start, out_per_group).contiguous()
            group_input = input_2d.narrow(1, in_start, in_per_group).contiguous()
            group_weight = weight_2d.narrow(0, out_start, out_per_group).contiguous()
            k_pad = (-in_per_group) % 16
            if k_pad:
                group_input = F.pad(group_input, (0, k_pad))
                group_weight = F.pad(group_weight, (0, k_pad))
            out_pad = (-out_per_group) % 16
            if out_pad:
                group_weight = F.pad(group_weight, (0, 0, 0, out_pad))
                group_bias = None if group_bias is None else F.pad(group_bias, (0, out_pad))
            chunks.append(Fp8ScaledLinearFunction.apply(
                group_input,
                group_weight,
                group_bias,
                float(input_scale),
                float(weight_scale),
            )[:, :out_per_group])
        out_2d = torch.cat(chunks, dim=1)

    return out_2d.reshape(batch, locations, out_channels).transpose(1, 2).reshape(
        batch, out_channels, out_h, out_w
    ).contiguous()


class Fp8Conv2d(nn.Module):
    """Drop-in ``nn.Conv2d``-style module backed by ``fp8_conv2d``.

    Only zero-padding mode is implemented. Forward uses E4M3 FP8 with dynamic
    per-tensor amax scaling and autograd support inherited from the FP8 linear
    lowering path.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  Union[int, Tuple[int, int]],
        stride:       Union[int, Tuple[int, int]] = 1,
        padding:      Union[int, Tuple[int, int]] = 0,
        dilation:     Union[int, Tuple[int, int]] = 1,
        groups:       int = 1,
        bias:         bool = True,
        padding_mode: str = "zeros",
        device:       Optional[Union[str, "torch.device"]] = None,
        dtype:        Optional["torch.dtype"] = None,
    ) -> None:
        super().__init__()
        if padding_mode != "zeros":
            raise ValueError("Fp8Conv2d only supports padding_mode='zeros'")
        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError("in_channels and out_channels must be divisible by groups")

        factory = {"device": device, "dtype": dtype}
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size, "kernel_size")
        self.stride = _pair(stride, "stride")
        self.padding = _pair(padding, "padding")
        self.dilation = _pair(dilation, "dilation")
        self.groups = int(groups)
        self.padding_mode = padding_mode

        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels // self.groups, *self.kernel_size, **factory
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, **factory))
        else:
            self.register_parameter("bias", None)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return fp8_conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding}, dilation={self.dilation}, "
            f"groups={self.groups}, bias={self.bias is not None}"
        )

    @classmethod
    def from_conv2d(cls, conv: "nn.Conv2d") -> "Fp8Conv2d":
        """Create an ``Fp8Conv2d`` by copying weights from ``nn.Conv2d``."""
        layer = cls(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )
        with torch.no_grad():
            layer.weight.copy_(conv.weight)
            if conv.bias is not None:
                layer.bias.copy_(conv.bias)
        return layer

    def to_conv2d(self) -> "nn.Conv2d":
        """Convert back to a standard ``nn.Conv2d``."""
        conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            bias=self.bias is not None,
            padding_mode=self.padding_mode,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        with torch.no_grad():
            conv.weight.copy_(self.weight)
            if self.bias is not None:
                conv.bias.copy_(self.bias)
        return conv


def fp8_conv1d(
    input:        "torch.Tensor",
    weight:       "torch.Tensor",
    bias:         Optional["torch.Tensor"] = None,
    stride:       int = 1,
    padding:      int = 0,
    dilation:     int = 1,
    groups:       int = 1,
    input_scale:  Optional[float] = None,
    weight_scale: Optional[float] = None,
) -> "torch.Tensor":
    """FP8 E4M3 conv1d using the same hipBLASLt-backed lowering as conv2d."""
    if input.dim() != 3:
        raise ValueError("fp8_conv1d: input must have shape [N, C, L]")
    if weight.dim() != 3:
        raise ValueError("fp8_conv1d: weight must have shape [out_channels, in_channels/groups, kL]")
    out = fp8_conv2d(
        input.unsqueeze(2),
        weight.unsqueeze(2),
        bias,
        stride=(1, int(stride)),
        padding=(0, int(padding)),
        dilation=(1, int(dilation)),
        groups=groups,
        input_scale=input_scale,
        weight_scale=weight_scale,
    )
    return out.squeeze(2)


class Fp8Conv1d(nn.Module):
    """Drop-in ``nn.Conv1d``-style module backed by ``fp8_conv1d``."""

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int,
        stride:       int = 1,
        padding:      int = 0,
        dilation:     int = 1,
        groups:       int = 1,
        bias:         bool = True,
        padding_mode: str = "zeros",
        device:       Optional[Union[str, "torch.device"]] = None,
        dtype:        Optional["torch.dtype"] = None,
    ) -> None:
        super().__init__()
        if padding_mode != "zeros":
            raise ValueError("Fp8Conv1d only supports padding_mode='zeros'")
        if in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError("in_channels and out_channels must be divisible by groups")

        factory = {"device": device, "dtype": dtype}
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation = int(dilation)
        self.groups = int(groups)
        self.padding_mode = padding_mode

        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels // self.groups, self.kernel_size, **factory
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, **factory))
        else:
            self.register_parameter("bias", None)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return fp8_conv1d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding}, dilation={self.dilation}, "
            f"groups={self.groups}, bias={self.bias is not None}"
        )

    @classmethod
    def from_conv1d(cls, conv: "nn.Conv1d") -> "Fp8Conv1d":
        """Create an ``Fp8Conv1d`` by copying weights from ``nn.Conv1d``."""
        layer = cls(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size[0],
            conv.stride[0],
            conv.padding[0],
            conv.dilation[0],
            conv.groups,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )
        with torch.no_grad():
            layer.weight.copy_(conv.weight)
            if conv.bias is not None:
                layer.bias.copy_(conv.bias)
        return layer

    def to_conv1d(self) -> "nn.Conv1d":
        """Convert back to a standard ``nn.Conv1d``."""
        conv = nn.Conv1d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            bias=self.bias is not None,
            padding_mode=self.padding_mode,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        with torch.no_grad():
            conv.weight.copy_(self.weight)
            if self.bias is not None:
                conv.bias.copy_(self.bias)
        return conv


# ---------------------------------------------------------------------------
# convert_to_fp8 — one-call model converter
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Fp8ShadowLinear — FP8 weight storage + floating-point master (Feature 2)
# ---------------------------------------------------------------------------

class Fp8ShadowLinearFunction(torch.autograd.Function):
    """FP8 linear where the weight lives as uint8 but gradients flow to the master dtype.

    Forward:
      - Input:  apply E4M3 noise (scaled), save as uint8 (activation compression)
      - Weight: dequantise from the uint8 shadow buffer
    Backward:
      - grad_input  : uses the master weight (accurate direction signal)
      - grad_weight : straight-through to master weight using compressed activation
      - input_scale, weight_inv_scale, bias: not differentiable → None gradient

    The trick: we accept both ``weight_master`` (Parameter, tracked by
    autograd) AND ``weight_fp8`` (uint8 buffer, not tracked).  The forward
    uses weight_fp8 for cheap dequant; the backward sends gradient to
    weight_master via the straight-through estimator.
    """

    @staticmethod
    def forward(
        ctx,
        input:           "torch.Tensor",           # [M, K] float32/float16/bfloat16
        weight_master:   "torch.Tensor",           # [N, K] master Parameter
        weight_fp8:      "torch.Tensor",           # [N, K] uint8,   Buffer
        weight_inv_scale: float,                   # 1 / weight_scale
        input_scale:     float,                    # 448 / amax(input)
        bias:            Optional["torch.Tensor"], # [N] or None
        input_fp8:       Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":

        # Metadata tensors are updated in-place on later forwards, so retain
        # device-side snapshots for this backward pass.
        if torch.is_tensor(weight_inv_scale):
            weight_inv_scale = weight_inv_scale.detach().clone()
        if torch.is_tensor(input_scale):
            input_scale = input_scale.detach().clone()

        input_c = input.contiguous()

        # A fused delayed-amax caller has already emitted input_fp8. Preserve
        # the standalone Function contract by quantizing when it is omitted.
        ctx._num_inputs = len(ctx.needs_input_grad)
        if input_fp8 is None:
            input_fp8 = quantize_e4m3((input_c * input_scale).contiguous())
        else:
            input_fp8 = input_fp8.contiguous()
        weight_fp8_c = weight_fp8.contiguous()

        ctx.has_bias       = bias is not None
        ctx.input_scale    = input_scale
        ctx.weight_scale   = 1.0 / weight_inv_scale
        if bias is not None:
            ctx.save_for_backward(input_fp8, weight_master, bias)
        else:
            ctx.save_for_backward(input_fp8, weight_master)

        if not input_c.is_cuda:
            input_sim = dequantize_e4m3(input_fp8).float() * (1.0 / input_scale)
            weight_sim = dequantize_e4m3(weight_fp8_c).float() * weight_inv_scale
            out = input_sim @ weight_sim.t()
            if bias is not None:
                out = out + bias.float().unsqueeze(0)
            return out.to(input_c.dtype)

        out = _hipblaslt_fp8_linear_forward_prequant(
            input_fp8,
            weight_fp8_c,
            bias,
            1.0 / input_scale if torch.is_tensor(input_scale) else (1.0 / _scale_to_float(input_scale)),
            weight_inv_scale if torch.is_tensor(weight_inv_scale) else _scale_to_float(weight_inv_scale),
            input_c.dtype,
        )
        if out is not None:
            return out

        return fp8_linear_forward_fp8_input_weight(
            input_fp8, weight_fp8_c, input_c,
            _scale_to_float(weight_inv_scale), _scale_to_float(input_scale), bias
        )

    @staticmethod
    def backward(ctx, grad_output: "torch.Tensor"):
        if ctx.has_bias:
            input_fp8, weight_master, bias = ctx.saved_tensors
        else:
            input_fp8, weight_master = ctx.saved_tensors
            bias = None

        # Decompress activation
        input_f32 = dequantize_e4m3(input_fp8) * (1.0 / ctx.input_scale)

        grad_output_c = grad_output.contiguous()

        if not grad_output_c.is_cuda:
            grad_input, grad_weight_master = _cpu_fp8_linear_backward(
                grad_output_c, weight_master, input_f32,
                weight_scale=ctx.weight_scale, input_scale=1.0
            )
            grad_bias = grad_output.sum(0) if bias is not None else None
            return (grad_input, grad_weight_master, None, None, None, grad_bias, None)[:ctx._num_inputs]

        grad_output_for_backward, grad_output_fp8 = _prepare_e5m2_backward_grad_output(grad_output_c)

        # Try hipBLASLt e5m2 backward path first
        hipblaslt_result = _hipblaslt_fp8_backward(
            grad_output_for_backward, weight_master, input_f32,
            weight_scale=ctx.weight_scale, input_scale=1.0
        )
        if hipblaslt_result is not None:
            grad_input, grad_weight_master = hipblaslt_result
            grad_bias = grad_output.sum(0) if bias is not None else None
            return (grad_input, grad_weight_master, None, None, None, grad_bias, None)[:ctx._num_inputs]

        grad_input, grad_weight_master = _raw_fp8_linear_backward(
            grad_output_c, weight_master, input_f32,
            ctx.weight_scale, 1.0, grad_output_fp8,
        )
        if grad_weight_master.dtype != weight_master.dtype:
            grad_weight_master = grad_weight_master.to(weight_master.dtype)
        grad_bias          = grad_output.sum(0) if bias is not None else None

        # Returns align with forward args; the optional final activation is a
        # non-differentiable raw storage tensor.
        return (grad_input, grad_weight_master, None, None, None, grad_bias, None)[:ctx._num_inputs]


class Fp8ShadowLinear(nn.Module):
    """Linear layer with FP8 weight storage and float32/float16/bfloat16 master weights.

    VRAM layout per layer (N×K weight matrix):
      ``weight_master``  fp32/fp16/bf16 [N, K] — seen by optimizer
      ``weight_fp8``     uint8    [N, K]  — 1 byte/param, used in forward
      ``bias``           master dtype [N] — negligible

    Net saving vs ``nn.Linear``: weight VRAM is kept at 1 byte/param during
    the forward pass.  The master weight is still kept for optimizer updates,
    but can now be fp16/bf16 to cut persistent parameter and gradient VRAM.
    Combined with ``Adafactor``, optimizer state drops dramatically:
    no first moment + factored second moment ≈ (N+K)/NK << 1 of weight size.

    Per-tensor amax scaling (same as ``Fp8ScaledLinear``) keeps quantization
    noise low across all layers.

    Args:
        in_features:  input size.
        out_features: output size.
        bias:         learnable bias (default True).
        history_len:  amax rolling window length.
        device:       device for parameters.
        dtype:        dtype for master weight (float32, float16, or bfloat16).

    Compatibility:
        ``layer.weight`` is a property that returns ``weight_master``, so
        weight-tying (``lm_head.weight = embed.weight``) works as expected.
    """

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        bias:         bool = True,
        history_len:  int  = 16,
        device:       Optional[Union[str, "torch.device"]] = None,
        dtype:        Optional["torch.dtype"] = None,
    ) -> None:
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.in_features  = in_features
        self.out_features = out_features

        # Master weight — the optimizer's target
        self.weight_master = nn.Parameter(
            torch.empty(out_features, in_features, **factory)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, **factory))
        else:
            self.register_parameter("bias", None)

        # uint8 shadow — 1 byte/param, recomputed from master each forward
        self.register_buffer(
            "weight_fp8",
            torch.zeros(out_features, in_features, dtype=torch.uint8,
                        **{"device": device} if device else {}),
        )

        meta_device = str(self.weight_master.device)
        self.input_meta  = Fp8TensorMeta(history_len=history_len, device=meta_device)
        self.weight_meta = Fp8TensorMeta(history_len=history_len, device=meta_device)
        self._shadow_version = -1

        self._reset_parameters()

    # ------------------------------------------------------------------
    @property
    def weight(self) -> "nn.Parameter":
        """Alias for weight_master — allows weight-tying to work normally."""
        return self.weight_master

    # ------------------------------------------------------------------
    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight_master, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_master)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def _sync_shadow(self) -> None:
        """Re-quantise a changed master weight; reuse it for repeated forwards."""
        current_version = int(self.weight_master._version)
        if (
            current_version == self._shadow_version
            and self.weight_meta.scale.device == self.weight_master.device
        ):
            return
        if self.weight_meta.scale.device != self.weight_master.device:
            self.weight_meta.to(self.weight_master.device)
        self.weight_meta.update(self.weight_master)
        ext = _load_extension()
        if self.weight_master.is_cuda and hasattr(ext, "refresh_e4m3_shadow"):
            refresh_e4m3_shadow(self.weight_master, self.weight_fp8, self.weight_meta.scale)
        else:
            self.weight_fp8.copy_(
                quantize_e4m3((self.weight_master * self.weight_meta.scale).contiguous())
            )
        self._shadow_version = current_version

    # ------------------------------------------------------------------
    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features).contiguous()

        fused_input_fp8 = None
        input_scale_for_forward = None
        with torch.no_grad():
            self._sync_shadow()
            if self.input_meta.scale.device != x_2d.device:
                self.input_meta.to(x_2d.device)
            ext = _load_extension()
            if x_2d.is_cuda and hasattr(ext, "quantize_e4m3_delayed_amax"):
                fused_input_fp8, input_scale_for_forward = self.input_meta.quantize_e4m3_delayed(x_2d)
            else:
                self.input_meta.update(x_2d)

        if fused_input_fp8 is None:
            out = Fp8ShadowLinearFunction.apply(
                x_2d, self.weight_master, self.weight_fp8,
                self.weight_meta.inv_scale, self.input_meta.scale, self.bias,
            )
        else:
            out = Fp8ShadowLinearFunction.apply(
                x_2d, self.weight_master, self.weight_fp8,
                self.weight_meta.inv_scale, input_scale_for_forward, self.bias,
                fused_input_fp8,
            )
        return out.reshape(*orig_shape[:-1], self.out_features)

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"history_len={self.input_meta._history_len}"
        )

    @classmethod
    def from_linear(
        cls,
        linear:      "nn.Linear",
        history_len: int = 16,
    ) -> "Fp8ShadowLinear":
        """Create an ``Fp8ShadowLinear`` from an existing ``nn.Linear``."""
        layer = cls(
            linear.in_features, linear.out_features,
            bias=linear.bias is not None,
            history_len=history_len,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        with torch.no_grad():
            layer.weight_master.copy_(linear.weight)
            if linear.bias is not None:
                layer.bias.copy_(linear.bias)
        return layer

    def to_linear(self) -> "nn.Linear":
        """Convert back to ``nn.Linear`` (copies master weights)."""
        linear = nn.Linear(
            self.in_features, self.out_features,
            bias=self.bias is not None,
            device=self.weight_master.device,
            dtype=self.weight_master.dtype,
        )
        with torch.no_grad():
            linear.weight.copy_(self.weight_master)
            if self.bias is not None:
                linear.bias.copy_(self.bias)
        return linear


# ---------------------------------------------------------------------------

def convert_to_fp8(
    model:       "nn.Module",
    shadow:      bool                = False,
    scaled:      bool                = True,
    history_len: int                 = 16,
    skip_names:  Optional[Set[str]] = None,
) -> "nn.Module":
    """Replace all ``nn.Linear`` layers in *model* with FP8 equivalents.

    Three modes (in order of increasing VRAM savings):

    ``shadow=False, scaled=False``  →  ``Fp8Linear``
        FP8 noise only, weights still float32 in memory.

    ``shadow=False, scaled=True`` (default)
        →  ``Fp8ScaledLinear``
        FP8 noise + per-tensor amax scaling.  Activation VRAM 4× lower
        (uint8 saved in autograd graph).

    ``shadow=True``                 →  ``Fp8ShadowLinear``
        All of the above PLUS weights stored as uint8 at rest.
        Forward pass sees 1 byte/param instead of 4 bytes/param.
        The float32 master weight is kept for the optimizer.
        Pair with ``Adafactor`` to also cut optimizer state VRAM.

    Args:
        model:       any ``nn.Module`` — mutated in-place.
        shadow:      if True, use ``Fp8ShadowLinear`` (FP8 weight storage).
        scaled:      if True and shadow=False, use ``Fp8ScaledLinear``.
                     Ignored when shadow=True (shadow always uses scaling).
        history_len: amax rolling window for scale tracking.
        skip_names:  set of fully-qualified submodule names to leave unchanged.
                     Example: ``{"lm_head"}`` for weight-tied output projection.

    Returns:
        The same *model* object (mutated in-place) for chaining.

    Example::

        model = MyGPT(vocab=32000, d_model=512, n_layers=6)

        # Maximum VRAM savings: FP8 weights + FP8 activations + Adafactor
        convert_to_fp8(model, shadow=True, skip_names={"lm_head"})
        model.cuda()
        opt = Adafactor(model.parameters(), relative_step=True)
    """
    if skip_names is None:
        skip_names = set()
    _replace_linear(model, "", shadow=shadow, scaled=scaled,
                    history_len=history_len, skip_names=skip_names)
    return model


def _replace_linear(
    module:      "nn.Module",
    prefix:      str,
    shadow:      bool,
    scaled:      bool,
    history_len: int,
    skip_names:  Set[str],
) -> None:
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}".lstrip(".")
        if full_name in skip_names:
            continue
        if isinstance(child, nn.Linear):
            if shadow:
                rep = Fp8ShadowLinear.from_linear(child, history_len=history_len)
            elif scaled:
                rep = Fp8ScaledLinear.from_linear(child, history_len=history_len)
            else:
                rep = Fp8Linear.from_linear(child)
            setattr(module, name, rep)
        else:
            _replace_linear(child, full_name, shadow=shadow, scaled=scaled,
                            history_len=history_len, skip_names=skip_names)


# ===========================================================================
# Phase 4 preview: real FP8 GEMM kernel bindings
# ===========================================================================

def fp8_linear_forward(
    input:  "torch.Tensor",
    weight: "torch.Tensor",
    bias:   Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    """FP8 linear forward via the custom HIP tiled GEMM kernel (Phase 4).

    Correctness-first stub — replace with rocBLASLt for production throughput.
    """
    _require_gfx12_fp8_wmma(input)
    return _load_extension().fp8_linear_forward(
        input.contiguous(), weight.contiguous(), bias
    )


def fp8_linear_forward_scaled(
    input:        "torch.Tensor",
    weight:       "torch.Tensor",
    bias:         Optional["torch.Tensor"] = None,
    input_scale:  float = 1.0,
    weight_scale: float = 1.0,
) -> "torch.Tensor":
    """Scaled FP8 linear forward via gfx12 E4M3 WMMA."""
    _require_gfx12_fp8_wmma(input)
    return _load_extension().fp8_linear_forward_scaled(
        input.contiguous(), weight.contiguous(), bias, float(input_scale), float(weight_scale)
    )


def fp8_linear_forward_fp8_weight(
    input:            "torch.Tensor",
    weight_fp8:       "torch.Tensor",
    weight_inv_scale: float,
    input_scale:      float,
    bias:             Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    """Scaled FP8 linear forward using a pre-quantized E4M3 weight buffer."""
    _require_gfx12_fp8_wmma(input)
    return _load_extension().fp8_linear_forward_fp8_weight(
        input.contiguous(), weight_fp8.contiguous(),
        float(weight_inv_scale), float(input_scale), bias
    )


def fp8_linear_forward_fp8_input(
    input_fp8:           "torch.Tensor",
    weight:              "torch.Tensor",
    output_dtype_source: "torch.Tensor",
    input_scale:         float,
    weight_scale:        float,
    bias:                Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    """Scaled FP8 linear forward using pre-quantized E4M3 input."""
    _require_gfx12_fp8_wmma(output_dtype_source)
    return _load_extension().fp8_linear_forward_fp8_input(
        input_fp8.contiguous(), weight.contiguous(), output_dtype_source,
        float(input_scale), float(weight_scale), bias
    )


def fp8_linear_forward_fp8_input_weight(
    input_fp8:           "torch.Tensor",
    weight_fp8:          "torch.Tensor",
    output_dtype_source: "torch.Tensor",
    weight_inv_scale:    float,
    input_scale:         float,
    bias:                Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    """Scaled FP8 linear forward using pre-quantized E4M3 input and weight."""
    _require_gfx12_fp8_wmma(output_dtype_source)
    return _load_extension().fp8_linear_forward_fp8_input_weight(
        input_fp8.contiguous(), weight_fp8.contiguous(), output_dtype_source,
        float(weight_inv_scale), float(input_scale), bias
    )


def fp8_grouped_linear_forward_fp8_input(
    shared_input_fp8:      "torch.Tensor",
    weights_fp8:           Sequence["torch.Tensor"],
    output_dtype_source:   "torch.Tensor",
    input_inv_scale:       Union["torch.Tensor", float] = 1.0,
    weight_inv_scales:     Optional[Sequence[Union["torch.Tensor", float]]] = None,
    biases:                 Optional[Sequence[Optional["torch.Tensor"]]] = None,
) -> Tuple["torch.Tensor", ...]:
    """Run multiple FP8 projections from one shared E4M3 activation buffer.

    ``shared_input_fp8`` is raw ``uint8`` E4M3 storage with shape ``[M, K]``.
    Each grouped weight is likewise raw E4M3 ``[N_i, K]`` storage. This API
    deliberately does not re-quantize the input per projection, which is
    useful for Q/K/V and gated-MLP projection groups.

    ``*_inv_scale`` values are FP32 dequant scales: ``real ~= fp8 * scale``.
    The result is a tuple in the same order as ``weights_fp8``.
    """
    if shared_input_fp8.dtype != torch.uint8 or shared_input_fp8.ndim != 2:
        raise TypeError("shared_input_fp8 must be a rank-2 torch.uint8 E4M3 tensor")
    weights = tuple(weights_fp8)
    if not weights:
        raise ValueError("weights_fp8 must contain at least one projection")
    if weight_inv_scales is None:
        inv_scales: Tuple[Union["torch.Tensor", float], ...] = (1.0,) * len(weights)
    else:
        inv_scales = tuple(weight_inv_scales)
        if len(inv_scales) != len(weights):
            raise ValueError("weight_inv_scales must have one entry per weight")
    if biases is None:
        group_biases: Tuple[Optional["torch.Tensor"], ...] = (None,) * len(weights)
    else:
        group_biases = tuple(biases)
        if len(group_biases) != len(weights):
            raise ValueError("biases must have one entry per weight")

    input_inv_scale_t = _fp8_scale_tensor(
        input_inv_scale, "input_inv_scale", shared_input_fp8.device
    )
    outputs = []
    fallback_input_f32 = None
    for index, (weight_fp8, weight_inv_scale, bias) in enumerate(
        zip(weights, inv_scales, group_biases)
    ):
        if weight_fp8.dtype != torch.uint8 or weight_fp8.ndim != 2:
            raise TypeError(f"weights_fp8[{index}] must be a rank-2 torch.uint8 E4M3 tensor")
        if weight_fp8.device != shared_input_fp8.device or weight_fp8.shape[1] != shared_input_fp8.shape[1]:
            raise ValueError(f"weights_fp8[{index}] must share the input device and K dimension")
        weight_inv_scale_t = _fp8_scale_tensor(
            weight_inv_scale, f"weight_inv_scales[{index}]", shared_input_fp8.device
        )
        out = _hipblaslt_fp8_linear_forward_prequant(
            shared_input_fp8, weight_fp8, bias,
            input_inv_scale_t, weight_inv_scale_t, output_dtype_source.dtype,
        )
        if out is None and shared_input_fp8.is_cuda:
            # The custom kernel takes a quantization multiplier for input but
            # an inverse multiplier for the FP8 shadow weight.
            try:
                out = fp8_linear_forward_fp8_input_weight(
                    shared_input_fp8, weight_fp8, output_dtype_source,
                    _scale_to_float(weight_inv_scale_t),
                    _scale_to_float(input_inv_scale_t.reciprocal()), bias,
                )
            except RuntimeError:
                # A portable decode+matmul fallback keeps the grouped API
                # usable on non-gfx12 systems where hipBLASLt declines FP8.
                out = None
        if out is None:
            if fallback_input_f32 is None:
                fallback_input_f32 = dequantize_e4m3(shared_input_fp8) * input_inv_scale_t
            weight_f32 = dequantize_e4m3(weight_fp8) * weight_inv_scale_t
            out = fallback_input_f32.matmul(weight_f32.t())
            if bias is not None:
                out = out + bias
            out = out.to(output_dtype_source.dtype)
        outputs.append(out)
    return tuple(outputs)


# Short name for callers that already know the input is raw FP8.
fp8_grouped_linear = fp8_grouped_linear_forward_fp8_input


def pack_fp8_weight_for_wmma(weight_fp8: "torch.Tensor") -> "torch.Tensor":
    """Pack E4M3 weights into the lane-native ``[NT, KT, 2, 16, 8]`` WMMA layout."""
    return _load_extension().pack_fp8_weight_for_wmma(weight_fp8.contiguous())


def fp8_linear_forward_fp8_input_weight_packed(
    input_fp8:           "torch.Tensor",
    weight_packed:       "torch.Tensor",
    output_dtype_source: "torch.Tensor",
    output_features:     int,
    weight_inv_scale:    float,
    input_scale:         float,
    bias:                Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    """Scaled FP8 linear using pre-quantized input and packed custom-WMMA weight."""
    _require_gfx12_fp8_wmma(output_dtype_source)
    return _load_extension().fp8_linear_forward_fp8_input_weight_packed(
        input_fp8.contiguous(), weight_packed.contiguous(), output_dtype_source,
        int(output_features), float(weight_inv_scale), float(input_scale), bias
    )


def fp8_linear_forward_v2_input_weight(
    input_fp8:           "torch.Tensor",
    weight_fp8:          "torch.Tensor",
    output_dtype_source: "torch.Tensor",
    weight_inv_scale:    float,
    input_scale:         float,
    bias:                Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    """V2 cooperative LDS-staged FP8 linear using pre-quantized E4M3 input and weight."""
    _require_gfx12_fp8_wmma(output_dtype_source)
    return _load_extension().fp8_linear_forward_v2_input_weight(
        input_fp8.contiguous(), weight_fp8.contiguous(), output_dtype_source,
        float(weight_inv_scale), float(input_scale), bias
    )


def fp8_linear_forward_blockwise_quantized(
    input_fp8:           "torch.Tensor",
    input_scales:        "torch.Tensor",
    weight_fp8:          "torch.Tensor",
    weight_scales:       "torch.Tensor",
    output_dtype_source: "torch.Tensor",
    block_size:          int = 32,
    bias:                Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    """Fused block-scaled FP8 linear from pre-quantized E4M3 tensors."""
    return _load_extension().fp8_linear_forward_blockwise(
        input_fp8.contiguous(), input_scales.contiguous(),
        weight_fp8.contiguous(), weight_scales.contiguous(),
        output_dtype_source, int(block_size), bias
    )


def fp8_linear_forward_blockwise(
    input:      "torch.Tensor",
    weight:     "torch.Tensor",
    bias:       Optional["torch.Tensor"] = None,
    block_size: int = 32,
) -> "torch.Tensor":
    """Quantize input/weight block-wise then run fused block-scaled FP8 linear."""
    input_2d = input.contiguous()
    weight_2d = weight.contiguous()
    input_fp8, input_scales = quantize_e4m3_blockwise(input_2d, block_size)
    weight_fp8, weight_scales = quantize_e4m3_blockwise(weight_2d, block_size)
    return fp8_linear_forward_blockwise_quantized(
        input_fp8, input_scales, weight_fp8, weight_scales,
        input_2d, block_size, bias
    )


def fp8_linear_backward_input(
    grad_output: "torch.Tensor",
    weight:      "torch.Tensor",
) -> "torch.Tensor":
    """grad_input = E5M2(grad_output) @ weight  (HIP kernel, Phase 4)."""
    _require_gfx12_fp8_wmma(grad_output)
    return _load_extension().fp8_linear_backward_input(
        grad_output.contiguous(), weight.contiguous()
    )


def fp8_linear_backward_input_scaled(
    grad_output:  "torch.Tensor",
    weight:       "torch.Tensor",
    weight_scale: float,
) -> "torch.Tensor":
    """Scaled grad_input via gfx12 BF8/E5M2 WMMA."""
    _require_gfx12_fp8_wmma(grad_output)
    return _load_extension().fp8_linear_backward_input_scaled(
        grad_output.contiguous(), weight.contiguous(), float(weight_scale)
    )


def fp8_linear_backward_weight(
    grad_output: "torch.Tensor",
    input:       "torch.Tensor",
) -> "torch.Tensor":
    """grad_weight = E5M2(grad_output).T @ input  (HIP kernel, Phase 4)."""
    _require_gfx12_fp8_wmma(grad_output)
    return _load_extension().fp8_linear_backward_weight(
        grad_output.contiguous(), input.contiguous()
    )


def fp8_linear_backward_weight_scaled(
    grad_output: "torch.Tensor",
    input:       "torch.Tensor",
    input_scale: float,
) -> "torch.Tensor":
    """Scaled grad_weight via gfx12 BF8/E5M2 WMMA."""
    _require_gfx12_fp8_wmma(grad_output)
    return _load_extension().fp8_linear_backward_weight_scaled(
        grad_output.contiguous(), input.contiguous(), float(input_scale)
    )


def fp8_linear_backward_input_fp8_grad(
    grad_output_fp8:          "torch.Tensor",
    grad_output_dtype_source: "torch.Tensor",
    weight:                   "torch.Tensor",
    weight_scale:             float = 1.0,
) -> "torch.Tensor":
    """grad_input using pre-quantized E5M2 grad_output."""
    _require_gfx12_fp8_wmma(grad_output_dtype_source)
    return _load_extension().fp8_linear_backward_input_fp8_grad(
        grad_output_fp8.contiguous(), grad_output_dtype_source,
        weight.contiguous(), float(weight_scale)
    )


def fp8_linear_backward_weight_fp8_grad(
    grad_output_fp8:          "torch.Tensor",
    grad_output_dtype_source: "torch.Tensor",
    input:                    "torch.Tensor",
    input_scale:              float = 1.0,
) -> "torch.Tensor":
    """grad_weight using pre-quantized E5M2 grad_output."""
    _require_gfx12_fp8_wmma(grad_output_dtype_source)
    return _load_extension().fp8_linear_backward_weight_fp8_grad(
        grad_output_fp8.contiguous(), grad_output_dtype_source,
        input.contiguous(), float(input_scale)
    )


def fp8_linear_backward_input_raw_fp8(
    grad_output_fp8:      "torch.Tensor",
    weight_fp8:           "torch.Tensor",
    output_dtype_source:  "torch.Tensor",
    weight_inv_scale:     float = 1.0,
) -> "torch.Tensor":
    """grad-input from already-quantized E5M2 grad-output and weight bytes."""
    _require_gfx12_fp8_wmma(output_dtype_source)
    return _load_extension().fp8_linear_backward_input_raw_fp8(
        grad_output_fp8.contiguous(), weight_fp8.contiguous(), output_dtype_source,
        float(weight_inv_scale),
    )


def fp8_linear_backward_weight_raw_fp8(
    grad_output_fp8:      "torch.Tensor",
    input_fp8:            "torch.Tensor",
    output_dtype_source:  "torch.Tensor",
    input_inv_scale:      float = 1.0,
) -> "torch.Tensor":
    """grad-weight from already-quantized E5M2 grad-output and input bytes."""
    _require_gfx12_fp8_wmma(output_dtype_source)
    return _load_extension().fp8_linear_backward_weight_raw_fp8(
        grad_output_fp8.contiguous(), input_fp8.contiguous(), output_dtype_source,
        float(input_inv_scale),
    )


# ===========================================================================
# Phase 4 preview: Fp8TensorMeta — delayed-scaling amax tracker
# ===========================================================================

class Fp8TensorMeta:
    """Per-tensor FP8 scale management with a delayed-scaling strategy.

    Maintains a rolling ``amax_history`` ring buffer.  Scale is derived from
    the *maximum* observed amax across the window so that a single outlier
    batch does not cause the scale to spike.

    Attributes:
        scale:         float32 [1] — multiply tensor by this before quantizing.
        inv_scale:     float32 [1] — multiply dequantized values by this to
                       recover the original magnitude.
        amax_history:  float32 [history_len] — ring buffer of observed amaxes.
    """

    _FP8_E4M3_MAX: float = 448.0

    def __init__(
        self,
        history_len: int = 16,
        device:      Optional[str] = None,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available.")
        if history_len <= 0:
            raise ValueError("history_len must be positive")
        # Default to CPU so constructing layers for unit tests does not
        # initialize ROCm/CUDA (Windows teardown can hang after GPU init).
        # Modules move meta onto the parameter/activation device on first use.
        dev = device or "cpu"
        self.scale        = torch.ones(1,  dtype=torch.float32, device=dev)
        self.inv_scale    = torch.ones(1,  dtype=torch.float32, device=dev)
        self.amax_history = torch.zeros(history_len,
                                        dtype=torch.float32, device=dev)
        self.found_nonfinite = torch.zeros(1, dtype=torch.bool, device=dev)
        # Persistent scratch buffers keep the fused GPU path allocation-free
        # after construction. They are intentionally not checkpoint state.
        self._amax_scratch = torch.zeros(1, dtype=torch.float32, device=dev)
        self._sample_nonfinite = torch.zeros(1, dtype=torch.int32, device=dev)
        self._history_len = history_len
        self._ptr         = 0

    def update(self, tensor: "torch.Tensor") -> None:
        """Record a finite amax and refresh scale without poisoning metadata."""
        detached = tensor.detach()
        finite = torch.isfinite(detached)
        all_finite = finite.all()
        self.found_nonfinite.logical_or_(~all_finite.reshape_as(self.found_nonfinite))

        # Keep the previous history entry when this sample contains NaN/Inf.
        # The original tensor remains non-finite so a training-step guard can
        # skip the update, while future finite batches retain valid scales.
        finite_amax = torch.where(finite, detached.abs(), 0.0).max()
        slot = self._ptr % self._history_len
        previous = self.amax_history[slot].clone()
        self.amax_history[slot].copy_(torch.where(all_finite, finite_amax, previous))
        self._ptr += 1
        observed_max = self.amax_history.max().clamp(min=1e-12)
        new_scale = (self._FP8_E4M3_MAX / observed_max).float()
        self.scale.copy_(new_scale)
        self.inv_scale.copy_((1.0 / new_scale).float())

    def clear_nonfinite(self) -> None:
        """Clear the sticky non-finite observation flag after a skipped step."""
        self.found_nonfinite.zero_()

    def quantize_e4m3(self, x: "torch.Tensor") -> "torch.Tensor":
        """Scale then quantize to FP8 E4M3."""
        return quantize_e4m3((x * self.scale).contiguous())

    def quantize_e4m3_delayed(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Quantize with the current scale and asynchronously collect next amax.

        Returns raw E4M3 bytes and the immutable scale snapshot consumed for
        those bytes. The GPU implementation fuses the activation scan with
        quantization, commits amax history on-device, and only affects scale
        for the following call. CPU and older-extension fallbacks retain the
        same delayed ordering with the existing reference operations.
        """
        if self.scale.device != x.device:
            self.to(x.device)
        slot = self._ptr % self._history_len
        ext = _load_extension()
        if x.is_cuda and hasattr(ext, "quantize_e4m3_delayed_amax"):
            fp8, scale_snapshot = ext.quantize_e4m3_delayed_amax(
                x.contiguous(), self.scale, self.inv_scale, self.amax_history,
                self._amax_scratch, self._sample_nonfinite, self.found_nonfinite,
                int(slot),
            )
            self._ptr += 1
        else:
            scale_snapshot = self.scale.detach().clone()
            fp8 = quantize_e4m3((x * scale_snapshot).contiguous())
            self.update(x)
        return fp8, scale_snapshot

    def dequantize_e4m3(self, x: "torch.Tensor") -> "torch.Tensor":
        """Dequantize FP8 E4M3 then apply inverse scale."""
        return dequantize_e4m3(x) * self.inv_scale

    def to(self, device: Union[str, "torch.device"]) -> "Fp8TensorMeta":
        """Move internal tensors to *device* (returns self for chaining)."""
        self.scale        = self.scale.to(device)
        self.inv_scale    = self.inv_scale.to(device)
        self.amax_history = self.amax_history.to(device)
        self.found_nonfinite = self.found_nonfinite.to(device)
        self._amax_scratch = self._amax_scratch.to(device)
        self._sample_nonfinite = self._sample_nonfinite.to(device)
        return self

    def state_dict(self) -> Dict[str, "torch.Tensor"]:
        """Serialisable state for checkpointing."""
        return {
            "scale":        self.scale,
            "inv_scale":    self.inv_scale,
            "amax_history": self.amax_history,
            "found_nonfinite": self.found_nonfinite,
            "ptr":          torch.tensor(self._ptr),
        }

    def load_state_dict(self, state: Dict[str, "torch.Tensor"]) -> None:
        """Restore state from ``state_dict()``."""
        self.scale        = state["scale"]
        self.inv_scale    = state["inv_scale"]
        self.amax_history = state["amax_history"]
        self.found_nonfinite = state.get(
            "found_nonfinite",
            torch.zeros(1, dtype=torch.bool, device=self.scale.device),
        )
        self._amax_scratch = torch.zeros(1, dtype=torch.float32, device=self.scale.device)
        self._sample_nonfinite = torch.zeros(1, dtype=torch.int32, device=self.scale.device)
        self._ptr         = int(state["ptr"].item())
        self._history_len = len(self.amax_history)


# ===========================================================================
# Feature 3: Adafactor optimiser
# ===========================================================================

class Adafactor(torch.optim.Optimizer):
    """Adafactor: adaptive learning rates with sublinear memory cost.

    Reference: Shazeer & Stern (2018) https://arxiv.org/abs/1802.04821

    VRAM advantage over AdamW
    --------------------------
    For a weight matrix W ∈ R^{N×K}:

    AdamW stores:
      first moment  m  ∈ R^{N×K}   (4 bytes/param)
      second moment v  ∈ R^{N×K}   (4 bytes/param)
      → 2 × model_params floats of optimizer state

    Adafactor stores:
      row factor  R  ∈ R^N          (4 bytes × N)
      col factor  C  ∈ R^K          (4 bytes × K)
      no first moment
      → (N+K)/(N×K) of AdamW's v state  ≈ 0.05% for 4096×4096

    For a 500M-parameter model this typically means:
      AdamW optimizer state: ~4 GB
      Adafactor optimizer state: ~4 MB

    Recommended usage
    -----------------
    Use ``relative_step=True`` (default) to let the optimiser derive its own
    learning rate from the weight magnitude — no ``lr`` argument needed::

        opt = Adafactor(model.parameters(), relative_step=True,
                        weight_decay=0.1)

    For fine-tuning where you want a fixed lr::

        opt = Adafactor(model.parameters(), lr=1e-4, relative_step=False,
                        scale_parameter=False)

    Args:
        params:           iterable of parameters or param groups.
        lr:               explicit learning rate.  Must be None when
                          relative_step=True.
        beta2_decay:      exponent d for β₂ₜ = 1 − t^d.  Default -0.8.
        eps:              (eps1, eps2).  eps1 stabilises the second-moment
                          estimate near zero; eps2 sets the minimum scale for
                          relative-step lr.  Defaults (1e-30, 1e-3).
        clip_threshold:   RMS clip threshold for normalised updates. Default 1.0.
        relative_step:    derive lr from weight magnitude (default True).
        scale_parameter:  scale lr by rms(W) (requires relative_step=True).
        warmup_init:      start with a very small relative step (default False).
        weight_decay:     decoupled L2 penalty.  Applied after the update.
    """

    def __init__(
        self,
        params,
        lr:              Optional[float] = None,
        beta2_decay:     float           = -0.8,
        eps:             Tuple[float, float] = (1e-30, 1e-3),
        clip_threshold:  float           = 1.0,
        relative_step:   bool            = True,
        scale_parameter: bool            = True,
        warmup_init:     bool            = False,
        weight_decay:    float           = 0.0,
    ) -> None:
        if lr is not None and relative_step:
            raise ValueError(
                "Provide either an explicit lr= or relative_step=True, not both."
            )
        if not relative_step and lr is None:
            raise ValueError(
                "Must provide lr= when relative_step=False."
            )
        defaults = dict(
            lr              = lr,
            beta2_decay     = beta2_decay,
            eps             = eps,
            clip_threshold  = clip_threshold,
            relative_step   = relative_step,
            scale_parameter = scale_parameter,
            warmup_init     = warmup_init,
            weight_decay    = weight_decay,
        )
        super().__init__(params, defaults)
        self.last_step_skipped = False

    # ------------------------------------------------------------------
    @staticmethod
    def _rms(t: "torch.Tensor") -> float:
        """Root-mean-square of a tensor (scalar result)."""
        return (t.norm(2) / (t.numel() ** 0.5)).item()

    def _get_lr(self, group: dict, state: dict) -> float:
        if group["relative_step"]:
            # Relative step: α_t = max(ε₂, rms(W)) × min(ρ̂, 1/√t)
            min_step = 1e-6 if group["warmup_init"] else 1e-2
            rel      = min(min_step, 1.0 / math.sqrt(state["step"]))
            scale    = max(group["eps"][1], state["rms"]) if group["scale_parameter"] else 1.0
            return scale * rel
        return group["lr"]

    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Preflight every gradient before mutating any parameter or optimizer
        # state. This gives GradScaler-like all-or-nothing behavior when one
        # layer produces NaN/Inf instead of partially corrupting the model.
        finite_by_device = {}
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Adafactor does not support sparse gradients.")
                device = p.grad.device
                is_finite = torch.isfinite(p.grad).all()
                finite_by_device[device] = (
                    is_finite if device not in finite_by_device
                    else finite_by_device[device] & is_finite
                )
        if any(not bool(is_finite.item()) for is_finite in finite_by_device.values()):
            self.last_step_skipped = True
            return loss
        self.last_step_skipped = False

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                # Work in float32 regardless of parameter dtype
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                p_f32    = p.float() if p.dtype != torch.float32 else p
                factored = grad.dim() >= 2     # factorise 2-D+ params
                state    = self.state[p]

                # ---- Initialise state on first step ----------------------
                if len(state) == 0:
                    state["step"] = 0
                    if factored:
                        # Row factor: mean over last dim (cols)
                        state["exp_avg_sq_row"] = torch.zeros(
                            grad.shape[:-1], dtype=torch.float32, device=p.device
                        )
                        # Col factor: mean over second-to-last dim (rows)
                        state["exp_avg_sq_col"] = torch.zeros(
                            grad.shape[:-2] + grad.shape[-1:],
                            dtype=torch.float32, device=p.device,
                        )
                    else:
                        # 1-D params (bias, embedding): store full V
                        state["exp_avg_sq"] = torch.zeros_like(p_f32)
                    state["rms"] = 0.0

                state["step"] += 1
                state["rms"]   = self._rms(p_f32)
                lr             = self._get_lr(group, state)

                # β₂ₜ = 1 − t^d   (→ 1 as t grows, gives slower-decaying EMA)
                beta2t = 1.0 - math.pow(state["step"], group["beta2_decay"])
                eps1   = group["eps"][0]

                # ---- Second-moment update --------------------------------
                sq_grad = grad.pow(2).add_(eps1)

                if factored:
                    R = state["exp_avg_sq_row"]
                    C = state["exp_avg_sq_col"]

                    # R_t = β₂ₜ R_{t-1} + (1-β₂ₜ) mean_j(g² + ε₁)
                    R.mul_(beta2t).add_(sq_grad.mean(dim=-1),  alpha=1.0 - beta2t)
                    # C_t = β₂ₜ C_{t-1} + (1-β₂ₜ) mean_i(g² + ε₁)
                    C.mul_(beta2t).add_(sq_grad.mean(dim=-2), alpha=1.0 - beta2t)

                    # Reconstruct V̂^{-1/2}:
                    # V̂[i,j] = R[i]*C[j]/mean(R)
                    # u[i,j]  = g[i,j] * sqrt(mean(R)) / (sqrt(R[i]) * sqrt(C[j]))
                    r_factor = (R / R.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
                    c_factor = C.rsqrt().unsqueeze(-2)
                    update   = torch.mul(r_factor, torch.mul(c_factor, grad))
                else:
                    V = state["exp_avg_sq"]
                    V.mul_(beta2t).add_(sq_grad, alpha=1.0 - beta2t)
                    update = V.rsqrt().mul_(grad)

                # ---- RMS clip -------------------------------------------
                update_rms = self._rms(update)
                update.div_(max(1.0, update_rms / group["clip_threshold"]))

                # ---- Weight update --------------------------------------
                p_f32.add_(update, alpha=-lr)

                # ---- Decoupled weight decay -----------------------------
                if group["weight_decay"] != 0.0:
                    p_f32.add_(p_f32, alpha=-group["weight_decay"] * lr)

                # Cast back if parameter is not float32
                if p.dtype != torch.float32:
                    p.copy_(p_f32)

        return loss


def wave_attn(
    q: "torch.Tensor",
    k: "torch.Tensor",
    v: "torch.Tensor",
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> "torch.Tensor":
    """
    WaveAttention: Native FP8 WMMA flash attention forward kernel for AMD GFX12 (RDNA4).
    Fast-exp, parallel K/V tile loads, adaptive Q/K tile selection.
    Bypasses AOTriton entirely — uses v_wmma_f32_16x16x16_fp8_fp8 directly.
    Expects q, k, v as FP8 uint8 tensors (or float32/float16/bfloat16 tensors auto-quantized).
    """
    _require_gfx12_fp8_wmma(q)

    if scale is None:
        scale = 1.0 / (q.size(-1) ** 0.5)

    q_fp8 = q if q.dtype == torch.uint8 else quantize_e4m3(q)
    k_fp8 = k if k.dtype == torch.uint8 else quantize_e4m3(k)
    v_fp8 = v if v.dtype == torch.uint8 else quantize_e4m3(v)

    return _load_extension().wave_attn_forward(
        q_fp8.contiguous(),
        k_fp8.contiguous(),
        v_fp8.contiguous(),
        float(scale),
        1.0, 1.0, 1.0,
        bool(is_causal),
    )


# ═══════════════════════════════════════════════════════════════════════════
# High-Level QoL API — production-ready wrappers
# ═══════════════════════════════════════════════════════════════════════════

def quantize(
    x: "torch.Tensor",
    dtype: str = "e4m3",
    block_size: Optional[int] = None,
) -> "torch.Tensor":
    """Quantize a float / bfloat16 tensor to FP8.

    ``dtype`` is one of ``"e4m3"``, ``"e5m2"``.
    ``block_size`` enables blockwise quantization (default: per-tensor).
    """
    _fn = {"e4m3": quantize_e4m3, "e5m2": quantize_e5m2}
    if block_size:
        _fn = {"e4m3": quantize_e4m3_blockwise, "e5m2": quantize_e5m2_blockwise}
    f = _fn.get(dtype)
    if f is None:
        raise ValueError(f"Unknown quantize dtype: {dtype}")
    return f(x, block_size=block_size) if block_size else f(x)


def dequantize(x: "torch.Tensor") -> "torch.Tensor":
    """Dequantize FP8 uint8 tensor back to float32 (auto-detects E4M3/E5M2).

    For MXFP4, use ``mxfp4_to_fp8`` first, then this function.
    """
    if x.dtype != torch.uint8:
        raise ValueError(f"Expected uint8 tensor, got {x.dtype}")
    return dequantize_e4m3(x)


attention = wave_attn  # alias for cleaner API


def mxfp4_to_fp8(
    packed_values: "torch.Tensor",
    block_scales: "torch.Tensor",
    n_per_row: Optional[int] = None,
) -> "torch.Tensor":
    """Convert MXFP4 packed weights to FP8 E4M3.

    Raises RuntimeWarning once per session (emulation path).
    """
    if n_per_row is None:
        if packed_values.ndim == 2:
            n_per_row = int(packed_values.shape[1]) * 2
        else:
            raise ValueError("n_per_row required for non-2D packed input")
    return dequantize_mxfp4_to_fp8(packed_values, block_scales, n_per_row)


# ═══════════════════════════════════════════════════════════════════════════
# QuantizedLinear — drop-in nn.Linear replacement
# ═══════════════════════════════════════════════════════════════════════════

class QuantizedLinear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` with automatic FP8 / MXFP4 quantization.

    Weight is stored in its native format and quantized lazily on first forward.

    Parameters
    ----------
    in_features, out_features : int
    bias : bool
        Include a bias term (stored in float32).
    weight_format : str
        ``"fp8_e4m3"`` — quantize float weight to E4M3 (default).
        ``"mxfp4"`` — pre-packed MXFP4 weight.
    pre_quantized : bool
        If True, the weight passed to ``__init__`` is already FP8 uint8.

    Examples
    --------
    >>> layer = QuantizedLinear(4096, 14336)
    >>> y = layer(x)  # weight auto-quantized, input auto-quantized

    >>> # Pre-quantized weight
    >>> layer = QuantizedLinear(4096, 14336, pre_quantized=True)
    >>> layer.weight_fp8 = my_fp8_weight
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_format: str = "fp8_e4m3",
        pre_quantized: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_format = weight_format
        self._fp8_weight = None
        self._fp8_bias = None

        if weight_format == "fp8_e4m3":
            self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        elif weight_format == "mxfp4":
            self.register_buffer("packed_weight", torch.empty(1), persistent=True)
            self.register_buffer("block_scales", torch.empty(1), persistent=True)
        else:
            raise ValueError(f"Unknown weight_format: {weight_format}")

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        self._pre_quantized = pre_quantized

    @classmethod
    def from_mxfp4(
        cls,
        packed_weight: "torch.Tensor",
        block_scales: "torch.Tensor",
        bias: Optional["torch.Tensor"] = None,
    ) -> "QuantizedLinear":
        """Create from pre-packed MXFP4 weight and scales."""
        n_per_row = int(packed_weight.shape[1]) * 2
        out_features = packed_weight.shape[0]
        layer = cls(n_per_row, out_features, bias=bias is not None, weight_format="mxfp4")
        layer.packed_weight = packed_weight.contiguous().to(layer.packed_weight.device)
        layer.block_scales = block_scales.contiguous().to(layer.block_scales.device)
        if bias is not None:
            layer.bias.data = bias.contiguous().to(layer.bias.device)
        return layer

    def _quantize_weight(self, device=None):
        if self._fp8_weight is not None:
            return

        if self.weight_format == "fp8_e4m3":
            w = getattr(self, "weight", None)
            if w is not None:
                if device is not None and w.device != device:
                    w = w.to(device)
                self._fp8_weight = quantize_e4m3(w.data.contiguous().to(torch.float32))
                try:
                    del self.weight
                except AttributeError:
                    pass
        elif self.weight_format == "mxfp4":
            self._fp8_weight = dequantize_mxfp4_to_fp8(
                self.packed_weight, self.block_scales, self.in_features
            )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        self._quantize_weight(device=x.device)

        # Move bias to input device if needed
        bias = self.bias
        if bias is not None and bias.device != x.device:
            bias = bias.to(x.device)

        *batch, in_dim = x.shape
        x_2d = x.reshape(-1, in_dim)

        if x.dtype == torch.uint8:
            x_fp8 = x_2d.contiguous()
            scale_inv = None
        else:
            x_flat = x_2d.float()
            amax = x_flat.abs().max().clamp(min=1e-6)
            scale = 1.0 / amax
            x_fp8 = quantize_e4m3((x_flat * scale).contiguous())
            scale_inv = scale.reciprocal()

        scale_val = float(scale_inv.item()) if scale_inv is not None else 1.0
        result = _hipblaslt_fp8_linear_forward_prequant(x_fp8, self._fp8_weight, bias, scale_val, 1.0, x.dtype)
        if result is None:
            fallback_x = dequantize_e4m3(x_fp8) * scale_val
            fallback_w = dequantize_e4m3(self._fp8_weight)
            result = F.linear(fallback_x, fallback_w, bias)
        return result.reshape(*batch, self.out_features)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, format={self.weight_format}"


def convert_to_quantized(
    module: "nn.Module",
    weight_format: str = "fp8_e4m3",
    exclude: Optional[Set[type]] = None,
) -> "nn.Module":
    """Recursively replace ``nn.Linear`` layers with ``QuantizedLinear``.

    Weights are preserved — the float tensors are transferred to the new layer.

    .. code-block:: python

        model = convert_to_quantized(model)  # model is now FP8-ready
        model = model.cuda()
        y = model(x)  # forward uses FP8 via hipBLASLt
    """
    if exclude is None:
        exclude = set()

    for name, child in module.named_children():
        if type(child) in exclude:
            continue
        if isinstance(child, nn.Linear) and not isinstance(child, QuantizedLinear):
            new = QuantizedLinear(
                child.in_features, child.out_features,
                bias=child.bias is not None,
                weight_format=weight_format,
            )
            new.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                new.bias.data.copy_(child.bias.data)
            setattr(module, name, new)
        else:
            convert_to_quantized(child, weight_format=weight_format, exclude=exclude)
    return module






