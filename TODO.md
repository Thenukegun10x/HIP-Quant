# TODO

## PyTorch Training Extension

`hip_quant` is currently an inference/conversion quantization library. It accepts NumPy arrays in CPU memory through `ctypes`, launches HIP kernels through `hip_quantize.dll`, and returns NumPy byte buffers. That path is useful for offline quantization, but it is not suitable for PyTorch FP8 training because it bypasses autograd and would require GPU tensors to round-trip through CPU RAM.

See `PYTORCH_EXTENSION_PLAN.md` for the concrete implementation checklist and phased design.

### Goals

- Build a native PyTorch C++/HIP extension using `torch.utils.cpp_extension` / `torch::Tensor` APIs.
- Accept and return GPU-resident `torch::Tensor` objects directly, without NumPy or host copies.
- Expose FP8 E4M3 and E5M2 tensor quantization/dequantization functions for training workloads.
- Wrap extension calls in `torch.autograd.Function` so forward and backward behavior is explicit and autograd-safe.
- Support mixed FP8 training conventions:
  - Forward path: activations and weights in `E4M3`.
  - Backward path: gradients in `E5M2`, weights remaining in `E4M3`.
- Add PyTorch tests that verify device placement, dtype/shape contracts, gradient flow, and numerical behavior.

### Proposed Work Items

- Create a separate `torch_ext/` or `hip_quant_torch/` module instead of overloading the existing NumPy API.
- Refactor reusable FP8 encode/decode device helpers so both the DLL and PyTorch extension can include them.
- Add C++ wrappers with `#include <torch/extension.h>` for GPU tensor input/output validation.
- Add HIP kernels that operate on PyTorch tensor data pointers and write tensor outputs allocated by PyTorch.
- Add Python `torch.autograd.Function` wrappers for FP8 quantized training paths.
- Add packaging hooks for ROCm/PyTorch extension builds, likely separate from the current pure-Python wheel path.
- Document the distinction between offline NumPy quantization and online PyTorch training extension usage.
