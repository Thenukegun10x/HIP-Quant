"""PyTorch inference modules for direct packed HQ2/HQ3 execution.

The ROCm fast path calls ``hip_quant._C.hq2_linear_forward``.  It consumes
the native [out_features * (in_features / 256), 72] HQ2 bytes directly and
does not allocate a dequantized weight matrix.  Other Torch devices retain a
correct decode-plus-linear fallback, which keeps checkpoints portable.
"""

from __future__ import annotations

from typing import Any

from .format import BLOCK_BYTES, BLOCK_SIZE, HQ2Tensor
from .hq3 import HQ3_BLOCK_BYTES, HQ3_BLOCK_SIZE, HQ3Tensor
from .hq8 import HQ8_G128_BLOCK_BYTES, HQ8_G128_BLOCK_SIZE, HQ8Tensor

try:  # Keep ``import hq2`` lightweight for CPU-only installations.
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - exercised on Torch-free installs
    torch = None
    F = None


def _require_torch():
    if torch is None:
        raise RuntimeError("HQ2 PyTorch inference requires `pip install hip-quant[torch]`")
    return torch


def _rocm_extension():
    """Load the packaged HIP extension, including direct source-tree use."""
    try:
        from hip_quant.torch_api import _load_extension
    except ModuleNotFoundError:
        # Source-tree use: the repository root itself is the hip_quant package.
        from torch_api import _load_extension
    try:
        return _load_extension()
    except (ImportError, OSError) as packaged_error:
        # ``python`` launched from the repository root can import ``hq2`` but
        # does not have the parent directory on sys.path, so torch_api cannot
        # resolve ``hip_quant._C``.  The in-place extension is nevertheless
        # importable as the root-level ``_C`` module.  Keep the normal wheel
        # path first and use this only as a source-tree compatibility bridge.
        try:
            import _C
        except (ImportError, OSError):
            raise packaged_error
        return _C


def rocm_fused_available() -> bool:
    """Whether the current process can use direct packed-HQ2 ROCm inference."""
    if torch is None or not torch.cuda.is_available() or not torch.version.hip:
        return False
    try:
        return hasattr(_rocm_extension(), "hq2_linear_forward")
    except (ImportError, OSError, RuntimeError):
        return False


def rocm_hq3_fused_available() -> bool:
    """Whether the current process can use direct packed-HQ3 ROCm inference."""
    if torch is None or not torch.cuda.is_available() or not torch.version.hip:
        return False
    try:
        return hasattr(_rocm_extension(), "hq3_linear_forward")
    except (ImportError, OSError, RuntimeError):
        return False


def rocm_hq8_g128_fused_available() -> bool:
    """Whether direct packed HQ8_G128 W8A16 ROCm inference is available."""
    if torch is None or not torch.cuda.is_available() or not torch.version.hip:
        return False
    try:
        return hasattr(_rocm_extension(), "hq8_g128_linear_forward")
    except (ImportError, OSError, RuntimeError):
        return False


def _as_packed_tensor(value: HQ2Tensor | Any, out_features: int, in_features: int):
    runtime = _require_torch()
    if isinstance(value, HQ2Tensor):
        if value.shape != (out_features, in_features):
            raise ValueError(
                f"HQ2 tensor shape {value.shape} does not match ({out_features}, {in_features})"
            )
        packed = value.packed
    else:
        packed = value
    if not isinstance(packed, runtime.Tensor):
        raise TypeError("packed_weight must be an HQ2Tensor or torch.uint8 tensor")
    expected_blocks = out_features * (in_features // BLOCK_SIZE)
    if packed.dtype != runtime.uint8 or tuple(packed.shape) != (expected_blocks, BLOCK_BYTES):
        raise ValueError(
            "packed_weight must have HQ2 shape "
            f"({expected_blocks}, {BLOCK_BYTES}) and dtype torch.uint8, got {tuple(packed.shape)} {packed.dtype}"
        )
    return packed


def hq2_linear(
    input: "torch.Tensor",
    packed_weight: HQ2Tensor | "torch.Tensor",
    *,
    out_features: int,
    in_features: int,
    bias: "torch.Tensor | None" = None,
    force_reference: bool = False,
) -> "torch.Tensor":
    """Run an HQ2 linear layer from packed bytes.

    On ROCm inference this uses the fused HIP kernel.  The kernel accumulates
    in FP32 and emits the input dtype.  The fallback is intentionally exact in
    layout but materializes a decoded matrix, so it is for portability and
    training/debugging rather than the intended high-performance path.
    """
    runtime = _require_torch()
    if not isinstance(input, runtime.Tensor) or input.ndim < 1:
        raise TypeError("input must be a floating Torch tensor with at least one dimension")
    if not input.is_floating_point():
        raise TypeError(f"HQ2Linear input must be floating point, got {input.dtype}")
    if int(in_features) <= 0 or int(in_features) % BLOCK_SIZE:
        raise ValueError("in_features must be a positive multiple of 256")
    if int(out_features) <= 0 or input.shape[-1] != int(in_features):
        raise ValueError("input final dimension must equal positive out/in feature configuration")
    packed = _as_packed_tensor(packed_weight, int(out_features), int(in_features))
    if packed.device != input.device:
        raise ValueError("input and packed HQ2 weight must be on the same device; move the module with .to(device)")
    if bias is not None:
        if not isinstance(bias, runtime.Tensor) or bias.device != input.device:
            raise ValueError("bias must be a Torch tensor on the same device as input")
        if bias.ndim != 1 or bias.shape[0] != int(out_features):
            raise ValueError("bias must have shape [out_features]")
        if bias.dtype != input.dtype:
            raise ValueError("bias and input must have the same dtype")

    original_shape = tuple(input.shape[:-1])
    flat_input = input.contiguous().reshape(-1, int(in_features))
    requires_backward = runtime.is_grad_enabled() and (
        flat_input.requires_grad or (bias is not None and bias.requires_grad)
    )
    use_fused = (
        not force_reference
        and not requires_backward
        and flat_input.is_cuda
        and bool(runtime.version.hip)
        and rocm_fused_available()
    )
    if use_fused:
        output = _rocm_extension().hq2_linear_forward(
            flat_input,
            packed.contiguous(),
            int(out_features),
            int(in_features),
            bias.contiguous() if bias is not None else None,
        )
    else:
        storage = HQ2Tensor(
            packed=packed,
            shape=(int(out_features), int(in_features)),
            backend="torch-reference",
            iterations=0,
            importance_weighted=False,
        )
        output = F.linear(flat_input, storage.dequantize(dtype=input.dtype), bias)
    return output.reshape(*original_shape, int(out_features))


if torch is not None:

    class HQ2Linear(torch.nn.Module):
        """Inference-first ``nn.Linear`` replacement backed by packed HQ2 bytes."""

        def __init__(self, weight: HQ2Tensor, bias: "torch.Tensor | None" = None) -> None:
            super().__init__()
            if not isinstance(weight, HQ2Tensor) or len(weight.shape) != 2:
                raise TypeError("HQ2Linear weight must be a rank-2 HQ2Tensor")
            self.out_features, self.in_features = weight.shape
            packed = weight.packed
            if not isinstance(packed, torch.Tensor):
                packed = torch.from_numpy(weight.numpy().copy())
            self.register_buffer("packed_weight", packed.contiguous())
            if bias is not None:
                if bias.ndim != 1 or bias.shape[0] != self.out_features:
                    raise ValueError("HQ2Linear bias must have shape [out_features]")
                self.register_buffer("bias", bias.detach().contiguous())
            else:
                self.register_buffer("bias", None)

        @classmethod
        def from_linear(
            cls,
            linear: "torch.nn.Linear",
            *,
            importance: "torch.Tensor | None" = None,
            iterations: int = 8,
        ) -> "HQ2Linear":
            """Quantize a normal ``nn.Linear`` once and retain only HQ2 bytes."""
            from .api import quantize

            backend = "torch" if linear.weight.is_cuda else "cpu"
            packed = quantize(linear.weight.detach(), importance=importance, backend=backend, iterations=iterations)
            return cls(packed, linear.bias)

        @classmethod
        def from_archive(cls, weight: HQ2Tensor, bias: "torch.Tensor | None" = None) -> "HQ2Linear":
            """Construct a layer from ``hq2.load_model(...).tensor(name)`` bytes."""
            return cls(weight, bias)

        def forward(self, input: "torch.Tensor") -> "torch.Tensor":
            return hq2_linear(
                input,
                self.packed_weight,
                out_features=self.out_features,
                in_features=self.in_features,
                bias=self.bias,
            )

        def extra_repr(self) -> str:
            return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, format=HQ2"

else:

    class HQ2Linear:  # pragma: no cover - importable Torch-free API sentinel
        def __init__(self, *args, **kwargs) -> None:
            _require_torch()




def _as_hq8_g128_packed_tensor(value: HQ8Tensor | Any, out_features: int, in_features: int):
    runtime = _require_torch()
    if isinstance(value, HQ8Tensor):
        if value.shape != (out_features, in_features):
            raise ValueError(
                f"HQ8_G128 tensor shape {value.shape} does not match ({out_features}, {in_features})"
            )
        packed = value.packed
    else:
        packed = value
    if not isinstance(packed, runtime.Tensor):
        raise TypeError("packed_weight must be an HQ8Tensor or torch.uint8 tensor")
    expected_groups = out_features * (in_features // HQ8_G128_BLOCK_SIZE)
    if packed.dtype != runtime.uint8 or tuple(packed.shape) != (expected_groups, HQ8_G128_BLOCK_BYTES):
        raise ValueError(
            "packed_weight must have HQ8_G128 shape "
            f"({expected_groups}, {HQ8_G128_BLOCK_BYTES}) and dtype torch.uint8, "
            f"got {tuple(packed.shape)} {packed.dtype}"
        )
    return packed


def hq8_g128_linear(
    input: "torch.Tensor",
    packed_weight: HQ8Tensor | "torch.Tensor",
    *,
    out_features: int,
    in_features: int,
    bias: "torch.Tensor | None" = None,
    force_reference: bool = False,
) -> "torch.Tensor":
    """Run an HQ8_G128 W8A16 linear layer from packed bytes."""

    runtime = _require_torch()
    if not isinstance(input, runtime.Tensor) or input.ndim < 1:
        raise TypeError("input must be a floating Torch tensor with at least one dimension")
    if not input.is_floating_point():
        raise TypeError(f"HQ8_G128 input must be floating point, got {input.dtype}")
    if int(in_features) <= 0 or int(in_features) % HQ8_G128_BLOCK_SIZE:
        raise ValueError("in_features must be a positive multiple of 128")
    if int(out_features) <= 0 or input.shape[-1] != int(in_features):
        raise ValueError("input final dimension must equal positive out/in feature configuration")
    packed = _as_hq8_g128_packed_tensor(packed_weight, int(out_features), int(in_features))
    if packed.device != input.device:
        raise ValueError("input and packed HQ8_G128 weight must be on the same device; move the module with .to(device)")
    if bias is not None:
        if not isinstance(bias, runtime.Tensor) or bias.device != input.device:
            raise ValueError("bias must be a Torch tensor on the same device as input")
        if bias.ndim != 1 or bias.shape[0] != int(out_features):
            raise ValueError("bias must have shape [out_features]")
        if bias.dtype != input.dtype:
            raise ValueError("bias and input must have the same dtype")

    original_shape = tuple(input.shape[:-1])
    flat_input = input.contiguous().reshape(-1, int(in_features))
    requires_backward = runtime.is_grad_enabled() and (
        flat_input.requires_grad or (bias is not None and bias.requires_grad)
    )
    use_fused = (
        not force_reference
        and not requires_backward
        and flat_input.is_cuda
        and bool(runtime.version.hip)
        and rocm_hq8_g128_fused_available()
    )
    if use_fused:
        output = _rocm_extension().hq8_g128_linear_forward(
            flat_input,
            packed.contiguous(),
            int(out_features),
            int(in_features),
            bias.contiguous() if bias is not None else None,
        )
    else:
        storage = HQ8Tensor(
            packed=packed,
            shape=(int(out_features), int(in_features)),
            backend="torch-reference",
        )
        output = F.linear(flat_input, storage.dequantize(dtype=input.dtype), bias)
    return output.reshape(*original_shape, int(out_features))


if torch is not None:

    class HQ8Linear(torch.nn.Module):
        """Inference-first nn.Linear replacement backed by HQ8_G128 bytes."""

        def __init__(self, weight: HQ8Tensor, bias: "torch.Tensor | None" = None) -> None:
            super().__init__()
            if not isinstance(weight, HQ8Tensor) or len(weight.shape) != 2:
                raise TypeError("HQ8Linear weight must be a rank-2 HQ8Tensor")
            self.out_features, self.in_features = weight.shape
            packed = weight.packed
            if not isinstance(packed, torch.Tensor):
                packed = torch.from_numpy(weight.numpy().copy())
            self.register_buffer("packed_weight", packed.contiguous())
            if bias is not None:
                if bias.ndim != 1 or bias.shape[0] != self.out_features:
                    raise ValueError("HQ8Linear bias must have shape [out_features]")
                self.register_buffer("bias", bias.detach().contiguous())
            else:
                self.register_buffer("bias", None)

        @classmethod
        def from_linear(cls, linear: "torch.nn.Linear") -> "HQ8Linear":
            from .api import quantize

            backend = "torch" if linear.weight.is_cuda else "cpu"
            packed = quantize(linear.weight.detach(), backend=backend, format="hq8_g128")
            return cls(packed, linear.bias)

        @classmethod
        def from_archive(cls, weight: HQ8Tensor, bias: "torch.Tensor | None" = None) -> "HQ8Linear":
            return cls(weight, bias)

        def forward(self, input: "torch.Tensor") -> "torch.Tensor":
            return hq8_g128_linear(
                input,
                self.packed_weight,
                out_features=self.out_features,
                in_features=self.in_features,
                bias=self.bias,
            )

        def extra_repr(self) -> str:
            return (
                f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, format=HQ8_G128"
            )

else:

    class HQ8Linear:  # pragma: no cover - importable Torch-free API sentinel
        def __init__(self, *args, **kwargs) -> None:
            _require_torch()

def _as_hq3_packed_tensor(value: HQ3Tensor | Any, out_features: int, in_features: int):
    runtime = _require_torch()
    if isinstance(value, HQ3Tensor):
        if value.shape != (out_features, in_features):
            raise ValueError(
                f"HQ3 tensor shape {value.shape} does not match ({out_features}, {in_features})"
            )
        packed = value.packed
    else:
        packed = value
    if not isinstance(packed, runtime.Tensor):
        raise TypeError("packed_weight must be an HQ3Tensor or torch.uint8 tensor")
    expected_blocks = out_features * (in_features // HQ3_BLOCK_SIZE)
    if packed.dtype != runtime.uint8 or tuple(packed.shape) != (expected_blocks, HQ3_BLOCK_BYTES):
        raise ValueError(
            "packed_weight must have HQ3 shape "
            f"({expected_blocks}, {HQ3_BLOCK_BYTES}) and dtype torch.uint8, got {tuple(packed.shape)} {packed.dtype}"
        )
    return packed


def hq3_linear(
    input: "torch.Tensor",
    packed_weight: HQ3Tensor | "torch.Tensor",
    *,
    out_features: int,
    in_features: int,
    bias: "torch.Tensor | None" = None,
    force_reference: bool = False,
) -> "torch.Tensor":
    """Run an HQ3 linear layer from packed bytes.

    ROCm uses the direct packed kernel once the extension is rebuilt. Other
    Torch devices decode only for the portability/reference fallback.
    """
    runtime = _require_torch()
    if not isinstance(input, runtime.Tensor) or input.ndim < 1:
        raise TypeError("input must be a floating Torch tensor with at least one dimension")
    if not input.is_floating_point():
        raise TypeError(f"HQ3Linear input must be floating point, got {input.dtype}")
    if int(in_features) <= 0 or int(in_features) % HQ3_BLOCK_SIZE:
        raise ValueError("in_features must be a positive multiple of 256")
    if int(out_features) <= 0 or input.shape[-1] != int(in_features):
        raise ValueError("input final dimension must equal positive out/in feature configuration")
    packed = _as_hq3_packed_tensor(packed_weight, int(out_features), int(in_features))
    if packed.device != input.device:
        raise ValueError("input and packed HQ3 weight must be on the same device; move the module with .to(device)")
    if bias is not None:
        if not isinstance(bias, runtime.Tensor) or bias.device != input.device:
            raise ValueError("bias must be a Torch tensor on the same device as input")
        if bias.ndim != 1 or bias.shape[0] != int(out_features):
            raise ValueError("bias must have shape [out_features]")
        if bias.dtype != input.dtype:
            raise ValueError("bias and input must have the same dtype")

    original_shape = tuple(input.shape[:-1])
    flat_input = input.contiguous().reshape(-1, int(in_features))
    requires_backward = runtime.is_grad_enabled() and (
        flat_input.requires_grad or (bias is not None and bias.requires_grad)
    )
    use_fused = (
        not force_reference
        and not requires_backward
        and flat_input.is_cuda
        and bool(runtime.version.hip)
        and rocm_hq3_fused_available()
    )
    if use_fused:
        output = _rocm_extension().hq3_linear_forward(
            flat_input,
            packed.contiguous(),
            int(out_features),
            int(in_features),
            bias.contiguous() if bias is not None else None,
        )
    else:
        storage = HQ3Tensor(
            packed=packed,
            shape=(int(out_features), int(in_features)),
            backend="torch-reference",
            iterations=0,
            importance_weighted=False,
        )
        output = F.linear(flat_input, storage.dequantize(dtype=input.dtype), bias)
    return output.reshape(*original_shape, int(out_features))


if torch is not None:

    class HQ3Linear(torch.nn.Module):
        """Inference-first ``nn.Linear`` replacement backed by packed HQ3 bytes."""

        def __init__(self, weight: HQ3Tensor, bias: "torch.Tensor | None" = None) -> None:
            super().__init__()
            if not isinstance(weight, HQ3Tensor) or len(weight.shape) != 2:
                raise TypeError("HQ3Linear weight must be a rank-2 HQ3Tensor")
            self.out_features, self.in_features = weight.shape
            packed = weight.packed
            if not isinstance(packed, torch.Tensor):
                packed = torch.from_numpy(weight.numpy().copy())
            self.register_buffer("packed_weight", packed.contiguous())
            if bias is not None:
                if bias.ndim != 1 or bias.shape[0] != self.out_features:
                    raise ValueError("HQ3Linear bias must have shape [out_features]")
                self.register_buffer("bias", bias.detach().contiguous())
            else:
                self.register_buffer("bias", None)

        @classmethod
        def from_linear(
            cls,
            linear: "torch.nn.Linear",
            *,
            importance: "torch.Tensor | None" = None,
            iterations: int = 8,
        ) -> "HQ3Linear":
            from .api import quantize

            backend = "torch" if linear.weight.is_cuda else "cpu"
            packed = quantize(
                linear.weight.detach(), importance=importance, backend=backend,
                iterations=iterations, format="hq3",
            )
            return cls(packed, linear.bias)

        @classmethod
        def from_archive(cls, weight: HQ3Tensor, bias: "torch.Tensor | None" = None) -> "HQ3Linear":
            return cls(weight, bias)

        def forward(self, input: "torch.Tensor") -> "torch.Tensor":
            return hq3_linear(
                input,
                self.packed_weight,
                out_features=self.out_features,
                in_features=self.in_features,
                bias=self.bias,
            )

        def extra_repr(self) -> str:
            return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, format=HQ3"

else:

    class HQ3Linear:  # pragma: no cover - importable Torch-free API sentinel
        def __init__(self, *args, **kwargs) -> None:
            _require_torch()


__all__ = [
    "HQ2Linear", "HQ3Linear", "HQ8Linear", "hq2_linear", "hq3_linear", "hq8_g128_linear",
    "rocm_fused_available", "rocm_hq3_fused_available", "rocm_hq8_g128_fused_available",
]
