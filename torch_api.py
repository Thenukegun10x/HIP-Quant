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
from typing import Dict, Optional, Set, Tuple, Union

# ---------------------------------------------------------------------------
# Lazy imports — file remains importable without torch or the _C extension
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

_C = None
_WMMA_GUARD_CACHE: Dict[int, Tuple[str, str]] = {}


def _scale_to_float(scale: "torch.Tensor") -> float:
    """Single sync point for legacy scalar-scale FP8 GEMM launchers."""
    return float(scale.item())


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
    if os.environ.get("HIP_QUANT_ENABLE_GFX12_WMMA", "").lower() not in ("1", "true", "yes", "on"):
        raise RuntimeError(
            "hip_quant FP8/BF8 WMMA kernels are disabled by default because unstable kernels can hang or reset the GPU. "
            "Set HIP_QUANT_ENABLE_GFX12_WMMA=1 only for controlled testing on ROCm 7.2+ gfx12 systems."
        )
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
            f"hip_quant FP8/BF8 WMMA linear kernels require ROCm 7.2+; current torch.version.hip is {rocm_version}. "
            "ROCm 7.1 and older have a gfx12 FP8 WMMA bug that can hang or zero GPU memory."
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


def dequantize_e4m3(x: "torch.Tensor") -> "torch.Tensor":
    """Dequantize an FP8 E4M3 uint8 tensor to float32 on-device."""
    return _load_extension().dequantize_e4m3(x.contiguous())


def dequantize_e5m2(x: "torch.Tensor") -> "torch.Tensor":
    """Dequantize an FP8 E5M2 uint8 tensor to float32 on-device."""
    return _load_extension().dequantize_e5m2(x.contiguous())


# ===========================================================================
# Phase 3: Autograd-safe FP8 linear
# ===========================================================================

def _sim_fp8_e4m3(x: "torch.Tensor") -> "torch.Tensor":
    """Quantize-then-dequantize in E4M3 — applies FP8 quantization noise."""
    return dequantize_e4m3(quantize_e4m3(x.contiguous()))


def _sim_fp8_e5m2(x: "torch.Tensor") -> "torch.Tensor":
    """Quantize-then-dequantize in E5M2 — applies FP8 quantization noise."""
    return dequantize_e5m2(quantize_e5m2(x.contiguous()))


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
        grad_output_fp8 = quantize_e5m2(grad_output_c)
        grad_input = fp8_linear_backward_input_fp8_grad(
            grad_output_fp8, grad_output_c, weight, 1.0
        )
        grad_weight = fp8_linear_backward_weight_fp8_grad(
            grad_output_fp8, grad_output_c, input_f32, 1.0
        )
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

    input_scale and weight_scale are plain Python floats — not tracked.
    """

    @staticmethod
    def forward(
        ctx,
        input:        "torch.Tensor",
        weight:       "torch.Tensor",
        bias:         Optional["torch.Tensor"],
        input_scale:  float,
        weight_scale: float,
    ) -> "torch.Tensor":

        input_c = input.contiguous()
        weight_c = weight.contiguous()

        # Save compressed scaled activation for backward. Forward itself uses
        # scaled in-register E4M3 quantization and gfx12 FP8 WMMA.
        input_fp8 = quantize_e4m3((input_c * input_scale).contiguous())

        ctx.has_bias    = bias is not None
        ctx.input_scale = input_scale          # needed to dequantise in backward
        ctx.weight_scale = weight_scale
        if bias is not None:
            ctx.save_for_backward(input_fp8, weight, bias)
        else:
            ctx.save_for_backward(input_fp8, weight)

        return fp8_linear_forward_fp8_input(
            input_fp8, weight_c, input_c, input_scale, weight_scale, bias
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
        grad_output_fp8 = quantize_e5m2(grad_output_c)
        grad_input = fp8_linear_backward_input_fp8_grad(
            grad_output_fp8, grad_output_c, weight, ctx.weight_scale
        )
        grad_weight = fp8_linear_backward_weight_fp8_grad(
            grad_output_fp8, grad_output_c, input_f32, 1.0
        )
        grad_bias   = grad_output.sum(0) if bias is not None else None

        return grad_input, grad_weight, grad_bias, None, None


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
    the autograd graph.  Scale values are passed as plain Python floats to
    ``Fp8ScaledLinearFunction`` so they do not create extra graph nodes.

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

        dev_str = str(device) if device is not None else None
        self.input_meta  = Fp8TensorMeta(history_len=history_len, device=dev_str)
        self.weight_meta = Fp8TensorMeta(history_len=history_len, device=dev_str)

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

        with torch.no_grad():
            self.input_meta.update(x_2d)
            self.weight_meta.update(self.weight)

        out = Fp8ScaledLinearFunction.apply(
            x_2d, self.weight, self.bias,
            _scale_to_float(self.input_meta.scale),
            _scale_to_float(self.weight_meta.scale),
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
    ) -> "torch.Tensor":

        input_c = input.contiguous()

        # Compress activation for backward. Forward consumes this scale and the
        # pre-quantized E4M3 weight shadow inside a gfx12 FP8 WMMA kernel.
        input_fp8 = quantize_e4m3((input_c * input_scale).contiguous())

        ctx.has_bias       = bias is not None
        ctx.input_scale    = input_scale
        ctx.weight_scale   = 1.0 / weight_inv_scale
        if bias is not None:
            ctx.save_for_backward(input_fp8, weight_master, bias)
        else:
            ctx.save_for_backward(input_fp8, weight_master)

        return fp8_linear_forward_fp8_input_weight(
            input_fp8, weight_fp8, input_c, weight_inv_scale, input_scale, bias
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
        grad_output_fp8 = quantize_e5m2(grad_output_c)
        grad_input = fp8_linear_backward_input_fp8_grad(
            grad_output_fp8, grad_output_c, weight_master, ctx.weight_scale
        )
        grad_weight_master = fp8_linear_backward_weight_fp8_grad(
            grad_output_fp8, grad_output_c, input_f32, 1.0
        )
        grad_bias          = grad_output.sum(0) if bias is not None else None

        # Returns align with forward args:
        # input, weight_master, weight_fp8, weight_inv_scale, input_scale, bias
        return grad_input, grad_weight_master, None, None, None, grad_bias


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

        dev_str = str(device) if device is not None else None
        self.input_meta  = Fp8TensorMeta(history_len=history_len, device=dev_str)
        self.weight_meta = Fp8TensorMeta(history_len=history_len, device=dev_str)

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
        """Re-quantise weight_master → weight_fp8.  Always inside no_grad."""
        self.weight_meta.update(self.weight_master)
        self.weight_fp8.copy_(
            quantize_e4m3((self.weight_master * self.weight_meta.scale).contiguous())
        )

    # ------------------------------------------------------------------
    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features).contiguous()

        with torch.no_grad():
            self._sync_shadow()
            self.input_meta.update(x_2d)

        out = Fp8ShadowLinearFunction.apply(
            x_2d,
            self.weight_master,
            self.weight_fp8,
            _scale_to_float(self.weight_meta.inv_scale),
            _scale_to_float(self.input_meta.scale),
            self.bias,
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
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scale        = torch.ones(1,  dtype=torch.float32, device=dev)
        self.inv_scale    = torch.ones(1,  dtype=torch.float32, device=dev)
        self.amax_history = torch.zeros(history_len,
                                        dtype=torch.float32, device=dev)
        self._history_len = history_len
        self._ptr         = 0

    def update(self, tensor: "torch.Tensor") -> None:
        """Record the absolute maximum of *tensor* and refresh scale."""
        amax = tensor.abs().max().detach()
        self.amax_history[self._ptr % self._history_len] = amax
        self._ptr += 1
        observed_max   = self.amax_history.max().clamp(min=1e-12)
        self.scale     = (self._FP8_E4M3_MAX / observed_max).float()
        self.inv_scale = (1.0 / self.scale).float()

    def quantize_e4m3(self, x: "torch.Tensor") -> "torch.Tensor":
        """Scale then quantize to FP8 E4M3."""
        return quantize_e4m3((x * self.scale).contiguous())

    def dequantize_e4m3(self, x: "torch.Tensor") -> "torch.Tensor":
        """Dequantize FP8 E4M3 then apply inverse scale."""
        return dequantize_e4m3(x) * self.inv_scale

    def to(self, device: Union[str, "torch.device"]) -> "Fp8TensorMeta":
        """Move internal tensors to *device* (returns self for chaining)."""
        self.scale        = self.scale.to(device)
        self.inv_scale    = self.inv_scale.to(device)
        self.amax_history = self.amax_history.to(device)
        return self

    def state_dict(self) -> Dict[str, "torch.Tensor"]:
        """Serialisable state for checkpointing."""
        return {
            "scale":        self.scale,
            "inv_scale":    self.inv_scale,
            "amax_history": self.amax_history,
            "ptr":          torch.tensor(self._ptr),
        }

    def load_state_dict(self, state: Dict[str, "torch.Tensor"]) -> None:
        """Restore state from ``state_dict()``."""
        self.scale        = state["scale"]
        self.inv_scale    = state["inv_scale"]
        self.amax_history = state["amax_history"]
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

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                # Work in float32 regardless of parameter dtype
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adafactor does not support sparse gradients."
                    )

                p_f32    = p.data.float() if p.data.dtype != torch.float32 else p.data
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
                if p.data.dtype != torch.float32:
                    p.data.copy_(p_f32)

        return loss
