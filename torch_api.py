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
    """Quantize a float32 GPU tensor to FP8 E4M3 (returned as uint8)."""
    return _load_extension().quantize_e4m3(x.contiguous())


def quantize_e5m2(x: "torch.Tensor") -> "torch.Tensor":
    """Quantize a float32 GPU tensor to FP8 E5M2 (returned as uint8)."""
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

    Backward accuracy fix
    ---------------------
    We save ``input_f32`` (the FP8-simulated activation) rather than the raw
    input and use it to compute ``grad_weight``.  This matches real FP8
    hardware where the weight-gradient accumulator sees the quantized
    activation, not the full-precision one.  The full-precision weight is
    kept for ``grad_input`` (standard mixed-precision master-weight convention).
    """

    @staticmethod
    def forward(
        ctx,
        input:  "torch.Tensor",
        weight: "torch.Tensor",
        bias:   Optional["torch.Tensor"],
    ) -> "torch.Tensor":

        input_f32  = _sim_fp8_e4m3(input)
        weight_f32 = _sim_fp8_e4m3(weight)

        ctx.has_bias = bias is not None
        if bias is not None:
            ctx.save_for_backward(input_f32, weight, bias)
        else:
            ctx.save_for_backward(input_f32, weight)

        out = input_f32.matmul(weight_f32.t())
        if bias is not None:
            out = out + bias
        return out

    @staticmethod
    def backward(ctx, grad_output: "torch.Tensor"):
        if ctx.has_bias:
            input_f32, weight, bias = ctx.saved_tensors
        else:
            input_f32, weight = ctx.saved_tensors
            bias = None

        grad_f32    = _sim_fp8_e5m2(grad_output.contiguous())
        grad_input  = grad_f32.matmul(weight)
        grad_weight = grad_f32.t().matmul(input_f32)
        grad_bias   = grad_output.sum(0) if bias is not None else None

        return grad_input, grad_weight, grad_bias


# ---------------------------------------------------------------------------
# Fp8ScaledLinearFunction — E4M3 with per-tensor amax scaling
# ---------------------------------------------------------------------------

class Fp8ScaledLinearFunction(torch.autograd.Function):
    """Autograd-safe FP8 linear with per-tensor amax scaling.

    Before quantizing, multiplies by a scale so the tensor fills the ±448
    FP8 E4M3 range more completely, then divides after dequantizing.
    Net math is identical to unscaled but with lower quantization noise
    (noise is proportional to 1/scale, and scale = 448 / amax).

    The input_scale and weight_scale arguments are plain Python floats —
    they are not differentiable and do not appear in the autograd graph.
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

        input_f32  = _sim_fp8_e4m3(input  * input_scale)  * (1.0 / input_scale)
        weight_f32 = _sim_fp8_e4m3(weight * weight_scale) * (1.0 / weight_scale)

        ctx.has_bias = bias is not None
        if bias is not None:
            ctx.save_for_backward(input_f32, weight, bias)
        else:
            ctx.save_for_backward(input_f32, weight)

        out = input_f32.matmul(weight_f32.t())
        if bias is not None:
            out = out + bias
        return out

    @staticmethod
    def backward(ctx, grad_output: "torch.Tensor"):
        if ctx.has_bias:
            input_f32, weight, bias = ctx.saved_tensors
        else:
            input_f32, weight = ctx.saved_tensors
            bias = None

        grad_f32    = _sim_fp8_e5m2(grad_output.contiguous())
        grad_input  = grad_f32.matmul(weight)
        grad_weight = grad_f32.t().matmul(input_f32)
        grad_bias   = grad_output.sum(0) if bias is not None else None

        # None for input_scale and weight_scale (not differentiable)
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
            float(self.input_meta.scale.item()),
            float(self.weight_meta.scale.item()),
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

def convert_to_fp8(
    model:       "nn.Module",
    scaled:      bool                = True,
    history_len: int                 = 16,
    skip_names:  Optional[Set[str]] = None,
) -> "nn.Module":
    """Replace all ``nn.Linear`` layers in *model* with FP8 equivalents.

    Walks the module tree recursively and replaces each ``nn.Linear`` with
    either ``Fp8ScaledLinear`` (recommended, default) or ``Fp8Linear``.
    Weights are copied; the replacement happens in-place.

    Args:
        model:       any ``nn.Module`` — mutated in-place.
        scaled:      if True (default) use ``Fp8ScaledLinear`` with per-tensor
                     amax scaling.  if False use plain ``Fp8Linear``.
        history_len: amax history window (only relevant when scaled=True).
        skip_names:  set of fully-qualified submodule names to leave unchanged.
                     Useful for weight-tied layers, e.g. ``{"lm_head"}``.

    Returns:
        The same *model* object (mutated in-place) for convenient chaining.

    Example::

        model = MyGPT(vocab=32000, d_model=512, n_layers=6)
        convert_to_fp8(model, scaled=True, skip_names={"lm_head"})
        model.cuda()
        # Every nn.Linear in the model is now Fp8ScaledLinear except lm_head
    """
    if skip_names is None:
        skip_names = set()
    _replace_linear(model, "", scaled=scaled,
                    history_len=history_len, skip_names=skip_names)
    return model


def _replace_linear(
    module:      "nn.Module",
    prefix:      str,
    scaled:      bool,
    history_len: int,
    skip_names:  Set[str],
) -> None:
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}".lstrip(".")
        if full_name in skip_names:
            continue
        if isinstance(child, nn.Linear):
            if scaled:
                rep = Fp8ScaledLinear.from_linear(child,
                                                  history_len=history_len)
            else:
                rep = Fp8Linear.from_linear(child)
            setattr(module, name, rep)
        else:
            _replace_linear(child, full_name, scaled=scaled,
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
    return _load_extension().fp8_linear_forward(
        input.contiguous(), weight.contiguous(), bias
    )


def fp8_linear_backward_input(
    grad_output: "torch.Tensor",
    weight:      "torch.Tensor",
) -> "torch.Tensor":
    """grad_input = E5M2(grad_output) @ weight  (HIP kernel, Phase 4)."""
    return _load_extension().fp8_linear_backward_input(
        grad_output.contiguous(), weight.contiguous()
    )


def fp8_linear_backward_weight(
    grad_output: "torch.Tensor",
    input:       "torch.Tensor",
) -> "torch.Tensor":
    """grad_weight = E5M2(grad_output).T @ input  (HIP kernel, Phase 4)."""
    return _load_extension().fp8_linear_backward_weight(
        grad_output.contiguous(), input.contiguous()
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
