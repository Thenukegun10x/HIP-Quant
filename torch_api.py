"""
hip_quant/torch_api.py
======================

Phase 2 & 3 Python API for GPU-resident FP8 operations.

This module provides:
  - Thin Python wrappers over the compiled ``hip_quant._C`` extension
    (Phase 2: element-wise quantize / dequantize).
  - ``Fp8LinearFunction`` — autograd-safe fake-FP8 linear layer (Phase 3).
  - ``Fp8Linear`` — drop-in ``nn.Module`` replacement (Phase 3).
  - ``Fp8TensorMeta`` — scale / amax tracking scaffold (Phase 4 preview).

The NumPy/ctypes API in ``hip_quant.__init__`` is *not* touched; this module
is purely additive and optional.

Import guard
------------
The ``_C`` extension is only available after building with ``setup_torch.py``.
This file is importable even without the extension (it raises a clear error
at call time rather than at import time) so that the rest of the package
can be imported in environments without PyTorch.
"""

from __future__ import annotations

from typing import Optional

# Lazy import so the file can be imported without torch installed.
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# Lazy import of the C extension — not required at import time.
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
    """Quantize a float32 GPU tensor to FP8 E4M3 (returned as uint8).

    Args:
        x: float32, contiguous, CUDA/HIP device tensor.

    Returns:
        uint8 tensor of same shape on the same device.
    """
    ext = _load_extension()
    return ext.quantize_e4m3(x.contiguous())


def quantize_e5m2(x: "torch.Tensor") -> "torch.Tensor":
    """Quantize a float32 GPU tensor to FP8 E5M2 (returned as uint8).

    Args:
        x: float32, contiguous, CUDA/HIP device tensor.

    Returns:
        uint8 tensor of same shape on the same device.
    """
    ext = _load_extension()
    return ext.quantize_e5m2(x.contiguous())


def dequantize_e4m3(x: "torch.Tensor") -> "torch.Tensor":
    """Dequantize an FP8 E4M3 uint8 tensor to float32 on-device.

    Args:
        x: uint8, contiguous, CUDA/HIP device tensor.

    Returns:
        float32 tensor of same shape on the same device.
    """
    ext = _load_extension()
    return ext.dequantize_e4m3(x.contiguous())


def dequantize_e5m2(x: "torch.Tensor") -> "torch.Tensor":
    """Dequantize an FP8 E5M2 uint8 tensor to float32 on-device.

    Args:
        x: uint8, contiguous, CUDA/HIP device tensor.

    Returns:
        float32 tensor of same shape on the same device.
    """
    ext = _load_extension()
    return ext.dequantize_e5m2(x.contiguous())


# ===========================================================================
# Phase 3: Autograd-safe fake FP8 linear
# ===========================================================================

class Fp8LinearFunction(torch.autograd.Function):
    """Fake-FP8 linear operation with full autograd support.

    Forward pass:
        Quantize activations and weights to E4M3, dequantize back to float32,
        then run standard matmul + optional bias.

    Backward pass:
        Quantize grad_output to E5M2, dequantize back, then compute grad_input
        and grad_weight using standard matmul.

    This keeps all tensors on GPU and honours autograd without requiring a
    custom FP8 GEMM kernel.  Performance is not optimised (matmul still runs
    in float32); use Phase 4 kernels for throughput.
    """

    @staticmethod
    def forward(ctx, input: "torch.Tensor",
                weight: "torch.Tensor",
                bias: Optional["torch.Tensor"] = None) -> "torch.Tensor":
        # Bug 5 fix: save_for_backward only accepts Tensors.
        # Store bias existence as a plain Python bool on ctx, then
        # conditionally include the bias tensor in saved_tensors.
        ctx.has_bias = bias is not None
        if bias is not None:
            ctx.save_for_backward(input, weight, bias)
        else:
            ctx.save_for_backward(input, weight)

        # Quantize → dequantize to simulate FP8 precision loss on device
        input_fp8  = quantize_e4m3(input)
        weight_fp8 = quantize_e4m3(weight)
        input_f32  = dequantize_e4m3(input_fp8)
        weight_f32 = dequantize_e4m3(weight_fp8)

        output = input_f32.matmul(weight_f32.t())
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output: "torch.Tensor"):
        if ctx.has_bias:
            input, weight, bias = ctx.saved_tensors
        else:
            input, weight = ctx.saved_tensors
            bias = None

        # Quantize gradient to E5M2 (wider dynamic range for gradients)
        grad_fp8 = quantize_e5m2(grad_output)
        grad_f32 = dequantize_e5m2(grad_fp8)

        grad_input  = grad_f32.matmul(weight)
        grad_weight = grad_f32.t().matmul(input)
        grad_bias   = grad_output.sum(0) if bias is not None else None

        return grad_input, grad_weight, grad_bias


class Fp8Linear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` using fake-FP8 forward/backward.

    Weights are stored as float32 master weights.  The FP8 quantization is
    applied at every forward call (simulating training with FP8 hardware).

    Args:
        in_features:  size of each input sample.
        out_features: size of each output sample.
        bias:         if True, add a learnable bias.

    Shape:
        Input  : ``(*, in_features)``
        Output : ``(*, out_features)``

    Example::

        layer = Fp8Linear(512, 256).cuda()
        x = torch.randn(32, 512, device="cuda")
        y = layer(x)          # FP8-quantized forward
        y.sum().backward()    # FP8-quantized backward
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True) -> None:
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=0.01)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # Flatten batch dims, apply FP8 linear, unflatten.
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)
        out = Fp8LinearFunction.apply(x_2d, self.weight, self.bias)
        return out.reshape(*orig_shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}")


# ===========================================================================
# Phase 4 preview: real FP8 GEMM bindings
# ===========================================================================

def fp8_linear_forward(
    input: "torch.Tensor",
    weight: "torch.Tensor",
    bias: Optional["torch.Tensor"] = None,
) -> "torch.Tensor":
    """GPU FP8 linear forward using the custom HIP GEMM kernel.

    Unlike ``Fp8LinearFunction``, this calls into the C++ kernel directly
    without an intermediate Python-level matmul.

    Args:
        input:  float32 [M, K] contiguous CUDA tensor.
        weight: float32 [N, K] contiguous CUDA tensor.
        bias:   float32 [N] contiguous CUDA tensor (optional).

    Returns:
        float32 [M, N] tensor on the same device.
    """
    ext = _load_extension()
    return ext.fp8_linear_forward(input.contiguous(), weight.contiguous(), bias)


def fp8_linear_backward_input(
    grad_output: "torch.Tensor",
    weight: "torch.Tensor",
) -> "torch.Tensor":
    """Compute grad_input = quant(grad_output) @ weight using the HIP kernel.

    Args:
        grad_output: float32 [M, N].
        weight:      float32 [N, K].

    Returns:
        float32 [M, K].
    """
    ext = _load_extension()
    return ext.fp8_linear_backward_input(
        grad_output.contiguous(), weight.contiguous()
    )


def fp8_linear_backward_weight(
    grad_output: "torch.Tensor",
    input: "torch.Tensor",
) -> "torch.Tensor":
    """Compute grad_weight = quant(grad_output).T @ input using the HIP kernel.

    Args:
        grad_output: float32 [M, N].
        input:       float32 [M, K].

    Returns:
        float32 [N, K].
    """
    ext = _load_extension()
    return ext.fp8_linear_backward_weight(
        grad_output.contiguous(), input.contiguous()
    )


# ===========================================================================
# Phase 4 preview: scale / amax tracking scaffold
# ===========================================================================

class Fp8TensorMeta:
    """Metadata for FP8 scale management.

    Tracks per-tensor scale, inverse-scale, and an amax history buffer
    compatible with a delayed-scaling strategy (similar to Transformer Engine).

    Attributes:
        scale:         float32 scalar tensor — multiply by this to go F32→FP8.
        inv_scale:     float32 scalar tensor — multiply FP8 value by this to
                       recover approximate float32.
        amax_history:  float32 tensor of shape ``(history_len,)`` — rolling
                       window of observed absolute maxima.
    """

    def __init__(self, history_len: int = 16,
                 device: Optional[str] = None) -> None:
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
        # Use the maximum observed amax across history for stability
        observed_max = self.amax_history.max().clamp(min=1e-12)
        # FP8 E4M3 max finite value is 448
        self.scale     = (448.0 / observed_max).float()
        self.inv_scale = (1.0 / self.scale).float()

    def quantize_e4m3(self, x: "torch.Tensor") -> "torch.Tensor":
        """Scale *x* then quantize to FP8 E4M3."""
        return quantize_e4m3(x * self.scale)

    def dequantize_e4m3(self, x: "torch.Tensor") -> "torch.Tensor":
        """Dequantize FP8 E4M3 then apply inverse scale."""
        return dequantize_e4m3(x) * self.inv_scale
