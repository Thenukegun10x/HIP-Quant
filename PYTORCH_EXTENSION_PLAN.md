# PyTorch FP8 Training Extension Plan

This document describes what still needs to be implemented to make `hip_quant` usable inside PyTorch training loops without CPU/NumPy round-trips.

## Current State

- The existing `hip_quant` API is an offline quantization/conversion library.
- It accepts NumPy arrays in CPU RAM through `ctypes`.
- It launches HIP kernels through `hip_quantize.dll`.
- It returns NumPy `uint8` buffers.
- This path is not suitable for PyTorch training because it bypasses autograd and forces GPU tensors through CPU memory.

## Target Architecture

Add a separate optional PyTorch extension layer that accepts GPU-resident `torch::Tensor` objects directly.

The NumPy API should remain stable. The PyTorch integration should live in separate files/modules so offline quantization and training integration do not become tangled.

## Files To Add

- `torch_ext/pytorch_bindings.cpp`
- `torch_ext/fp8_quant_kernels.hip` or `torch_ext/fp8_quant_kernels.cu`
- `torch_ext/fp8_linear_kernels.hip` or `torch_ext/fp8_linear_kernels.cu`
- `hip_quant/torch_api.py` or root-level `torch_api.py`, depending on final package layout
- `setup.py` or `setup_torch.py` for extension builds
- tests under `tests/torch/` once tests are allowed in the repo

## Build System

Use PyTorch's extension tooling:

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="hip_quant_torch",
    ext_modules=[
        CUDAExtension(
            "hip_quant._C",
            [
                "torch_ext/pytorch_bindings.cpp",
                "torch_ext/fp8_quant_kernels.hip",
                "torch_ext/fp8_linear_kernels.hip",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--offload-arch=gfx1201"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

Notes:

- On ROCm PyTorch, `CUDAExtension` is still the PyTorch API name, even though HIP/ROCm is used underneath.
- Local environment currently has `torch 2.9.1+rocm7.2.1` and sees the GPU.
- Existing standalone DLL targets ROCm 7.1, so extension builds may need separate ROCm/PyTorch compatibility handling.

## Phase 1: GPU-Resident FP8 Quantize/Dequantize

Implement these C++ binding functions first:

```cpp
torch::Tensor quantize_e4m3(torch::Tensor input);
torch::Tensor quantize_e5m2(torch::Tensor input);
torch::Tensor dequantize_e4m3(torch::Tensor input);
torch::Tensor dequantize_e5m2(torch::Tensor input);
```

Expected behavior:

- Input to quantize: `torch.float32`, contiguous, HIP/CUDA device tensor.
- Output from quantize: `torch.uint8`, same shape as input, same device.
- Input to dequantize: `torch.uint8`, contiguous, HIP/CUDA device tensor.
- Output from dequantize: `torch.float32`, same shape as input, same device.
- No NumPy.
- No CPU memory transfer.
- No `ctypes`.

Validation checks in bindings:

```cpp
TORCH_CHECK(input.is_cuda(), "input must be a HIP/CUDA tensor");
TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
```

Kernel launch shape:

- Flatten tensor to `numel()` elements.
- Use `256` threads per block.
- Use `(numel + 255) / 256` blocks.
- Each thread converts one element.

Reuse the device helper logic already implemented in `hip_quant_util.h`:

- `fp32_to_fp8_e4m3`
- `fp32_to_fp8_e5m2`
- `fp8_e4m3_to_fp32`
- `fp8_e5m2_to_fp32`

## Phase 2: Python Torch API

Add a small Python wrapper module:

```python
import torch
from hip_quant import _C


def quantize_e4m3(x: torch.Tensor) -> torch.Tensor:
    return _C.quantize_e4m3(x.contiguous())


def quantize_e5m2(x: torch.Tensor) -> torch.Tensor:
    return _C.quantize_e5m2(x.contiguous())


def dequantize_e4m3(x: torch.Tensor) -> torch.Tensor:
    return _C.dequantize_e4m3(x.contiguous())


def dequantize_e5m2(x: torch.Tensor) -> torch.Tensor:
    return _C.dequantize_e5m2(x.contiguous())
```

This phase proves the extension can operate directly on VRAM tensors.

## Phase 3: Autograd-Safe Fake FP8 Linear

Before writing a true FP8 GEMM kernel, implement a correctness-first autograd wrapper that quantizes/dequantizes on GPU, then calls PyTorch matmul:

```python
class Fp8LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)

        input_fp8 = quantize_e4m3(input)
        weight_fp8 = quantize_e4m3(weight)
        input_f32 = dequantize_e4m3(input_fp8)
        weight_f32 = dequantize_e4m3(weight_fp8)

        output = input_f32.matmul(weight_f32.t())
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors

        grad_fp8 = quantize_e5m2(grad_output)
        grad_f32 = dequantize_e5m2(grad_fp8)

        grad_input = grad_f32.matmul(weight)
        grad_weight = grad_f32.t().matmul(input)
        grad_bias = grad_output.sum(0) if bias is not None else None
        return grad_input, grad_weight, grad_bias
```

Purpose:

- Keeps autograd explicit.
- Keeps tensors on GPU.
- Lets training experiments begin before custom FP8 GEMM is complete.
- Uses E4M3 for forward activations/weights and E5M2 for backward gradients.

Limitations:

- This is fake FP8 training, not high-performance FP8 GEMM.
- Matmul still happens through PyTorch in float32/bfloat16 after dequantization.
- Performance will not match real FP8 hardware kernels.

## Phase 4: Real FP8 Linear Kernels

Add extension bindings:

```cpp
torch::Tensor fp8_linear_forward(torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> bias);
torch::Tensor fp8_linear_backward_input(torch::Tensor grad_output, torch::Tensor weight);
torch::Tensor fp8_linear_backward_weight(torch::Tensor grad_output, torch::Tensor input);
```

Forward convention:

- `input`: quantize to E4M3.
- `weight`: quantize to E4M3.
- Compute `input @ weight.T`.

Backward convention:

- `grad_output`: quantize to E5M2.
- `weight`: keep E4M3 or original master weight depending on experiment.
- Compute `grad_input = grad_output @ weight`.
- Compute `grad_weight = grad_output.T @ input`.

Implementation options:

- Start with simple tiled HIP GEMM for correctness.
- Later replace with rocBLASLt or composable kernel FP8 path if available for the target ROCm/PyTorch stack.
- Keep scaling policy explicit. Real FP8 training usually needs per-tensor or per-block scale factors.

## Scaling/Amax Tracking

FP8 training usually needs scale management. Add this after basic kernels work.

Needed concepts:

- Track `amax` per tensor or per channel.
- Maintain scale and inverse scale tensors.
- Quantize with `scaled = input * inv_scale`.
- Dequantize with `output = fp8_to_f32(byte) * scale`.
- Consider delayed scaling similar to Transformer Engine.

Potential API:

```python
class Fp8TensorMeta:
    scale: torch.Tensor
    inv_scale: torch.Tensor
    amax_history: torch.Tensor
```

## Tests To Add

Minimum tests:

- `quantize_e4m3` returns `torch.uint8` on the same GPU.
- `quantize_e5m2` returns `torch.uint8` on the same GPU.
- `dequantize_e4m3` and `dequantize_e5m2` return `torch.float32` on the same GPU.
- All extension functions reject CPU tensors.
- All extension functions reject unsupported dtypes.
- E4M3 bytes match the existing reference behavior for edge values.
- E5M2 bytes match the existing reference behavior for edge values.
- `Fp8LinearFunction` passes `torch.autograd.gradcheck` in a small fake-FP8 mode where possible.
- A tiny model trains for a few steps without CPU transfers or autograd breaks.

## Important Non-Goals For First Pass

- Do not replace the existing NumPy/DLL API.
- Do not promise stock GGML compatibility for project-local FP8 type IDs.
- Do not optimize GEMM before the tensor extension and autograd path are correct.
- Do not add CPU fallback unless there is a concrete test need.
