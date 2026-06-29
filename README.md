<div align="center">
  <h1>🚀 hip-quant</h1>
  <p><b>Blazing Fast On-Device Tensor Quantization for AMD GPUs</b></p>
  <p>
    <img alt="ROCm 7.1" src="https://img.shields.io/badge/ROCm-7.1-ED1C24?logo=amd"/>
    <img alt="gfx1201" src="https://img.shields.io/badge/arch-gfx1201-blue"/>
    <img alt="Python 3.8+" src="https://img.shields.io/badge/python-3.8+-3776AB?logo=python&logoColor=white"/>
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.9%2BROCm-EE4C2C?logo=pytorch"/>
  </p>
</div>

`hip-quant` is a standalone Python library and highly optimized HIP C++ backend that quantizes `float32` tensors directly on AMD GPUs — no CPU round-trips.

It ships **two independent APIs** that can be used together or separately:

| API | Purpose | Requires |
|---|---|---|
| **NumPy / ctypes** (offline) | Offline GGUF-format quantization via `hip_quantize.dll` | ROCm 7.1, numpy |
| **PyTorch extension** (training) | GPU-resident FP8 training ops with full autograd | PyTorch 2.x + ROCm, built `_C` extension |

---

## ⚡ Supported Quantization Formats

### 🔢 Standard & K-Quants (offline API)
- **Legacy:** `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`
- **K-Quants:** `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`

### 🧠 I-Quants (Importance Matrix)
Non-linear quants that preserve quality at extreme low bits:
- `IQ1_S`, `IQ2_XXS`, `IQ2_XS`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL`, `IQ4_XS`

### ⚖️ Ternary Quants
For models trained to be ternary (BitNet, TriLM):
- `TQ1_0` (1.69 bpw), `TQ2_0` (2.06 bpw)

### 🧪 FP8 Formats (both APIs)
| Format | Layout | Use case |
|---|---|---|
| `F8_E4M3` | 1s·4e·3m, bias=7, max=±448, NaN only | Forward activations & weights |
| `F8_E5M2` | 1s·5e·2m, bias=15, max=±57344, ±Inf+NaN | Backward gradients |

Both use OCP standard semantics with round-to-nearest-even. Math is validated against `ml_dtypes` reference — 90/90 test cases pass.

---

## 🛠️ Build

### Offline DLL (NumPy API)
Requires `hipcc` at `C:\Program Files\AMD\ROCm\7.1\bin\hipcc.exe`:
```powershell
.\build.ps1
```

### PyTorch Extension (`_C`)
Requires PyTorch with ROCm support (`torch 2.x+rocm`):
```powershell
python setup_torch.py build_ext --inplace
```

---

## 📦 Installation

```powershell
# Core offline library only
python -m build
pip install dist/hip_quant-0.4.1-py3-none-any.whl

# With PyTorch optional dependency declared
pip install "hip-quant[torch]"
```

---

## 🐍 Usage

### Offline NumPy API

```python
import numpy as np
from hip_quant import quantize

weights = np.random.randn(4096, 4096).astype(np.float32)

# Quantize directly to Q4_K on the GPU — byte-exact match to llama.cpp
q4k_bytes = quantize(weights, type_num=12)  # 12 = Q4_K
```

#### FP8 (offline)
```python
from hip_quant import GGML_TYPE, get_hip_quant

hq = get_hip_quant()
x    = np.random.randn(4096, 4096).astype(np.float32)
grad = (np.random.randn(4096, 4096) * 128).astype(np.float32)

x_e4m3    = hq.quantize_numpy(x,    GGML_TYPE["F8_E4M3"])  # forward
grad_e5m2 = hq.quantize_numpy(grad, GGML_TYPE["F8_E5M2"])  # backward
```

#### CLI
```powershell
hip-quant --help
python -m hip_quant --help
```

---

### PyTorch Training API

> **Requires:** `python setup_torch.py build_ext --inplace` first.

#### Element-wise FP8 quant / dequant (Phase 1 & 2)

```python
import torch
from hip_quant.torch_api import quantize_e4m3, dequantize_e4m3
from hip_quant.torch_api import quantize_e5m2, dequantize_e5m2

x = torch.randn(1024, 1024, device="cuda")  # stays on GPU the whole time

x_fp8  = quantize_e4m3(x)          # torch.uint8, same shape, same device
x_back = dequantize_e4m3(x_fp8)    # torch.float32, no CPU transfer
```

#### Fake-FP8 Linear (autograd-safe, Phase 3)

`Fp8LinearFunction` uses **E4M3** for forward activations/weights and **E5M2** for backward gradients. It also implements **Activation Compression**, saving `uint8` tensors in the autograd graph to cut activation VRAM by 4×.

```python
from hip_quant.torch_api import convert_to_fp8, Adafactor

# Drop-in replacement for all nn.Linear layers in a model
model = MySmallLM(...)

# shadow=True: replaces nn.Linear with Fp8ShadowLinear
# Weights are stored as uint8 in memory, forward pass decompresses on the fly
# Cuts weight VRAM by 4×
convert_to_fp8(model, shadow=True, skip_names={"lm_head"})
model.cuda()

# Adafactor optimizer: adaptive learning rates with sublinear memory cost
# Cuts optimizer state VRAM by ~1000× compared to AdamW
opt = Adafactor(model.parameters(), relative_step=True)
```

**Combined VRAM savings for a 500M-param LLM:**
Before: ~7.6 GB (Weights 2GB, Acts 1.6GB, AdamW 4GB)
After: ~0.9 GB (Weights 0.5GB, Acts 0.4GB, Adafactor 4MB)

#### Direct autograd.Function

```python
from hip_quant.torch_api import Fp8LinearFunction

out = Fp8LinearFunction.apply(input, weight, bias)  # bias optional
```

#### Fused FP8 Linear (gfx12 WMMA kernels)

```python
from hip_quant import (
    fp8_linear_forward,
    fp8_linear_backward_input,
    fp8_linear_backward_weight,
)

# [M,K] @ [N,K].T = [M,N]
# forward: E4M3 x E4M3 WMMA, backward: E5M2/BF8 x E5M2/BF8 WMMA
out        = fp8_linear_forward(input, weight, bias=None)
grad_in    = fp8_linear_backward_input(grad_output, weight)
grad_wt    = fp8_linear_backward_weight(grad_output, input)
```

These functions are also used by `Fp8LinearFunction` / `Fp8Linear` after the
extension is built.

#### Scale / amax tracking (Phase 4 scaffold)

```python
from hip_quant.torch_api import Fp8TensorMeta

meta = Fp8TensorMeta(history_len=16, device="cuda")
meta.update(x)                   # records amax, updates scale/inv_scale

x_fp8  = meta.quantize_e4m3(x)  # scaled, then quantized
x_back = meta.dequantize_e4m3(x_fp8)  # dequantized, then rescaled
```

---

## 🔒 Memory Safety

All PyTorch extension functions are guarded against:
- Non-CUDA tensors (`TORCH_CHECK(is_cuda)`)
- Non-contiguous layout (`TORCH_CHECK(is_contiguous)`)
- Wrong dtype (`TORCH_CHECK(scalar_type == kFloat32 / kUInt8)`)
- Dimension mismatch for GEMM
- **`int64 → int` narrowing** — explicit `checked_int()` with `TORCH_CHECK`
- **Hardware grid limit** — `gridDim.y ≤ 65535` validated before launch
- **Cross-device pointers** — `input.device() == weight.device()` checked
- **Empty tensors** — `numel == 0` early-return before `dim3(0)` (UB in HIP)

---

## 🧪 Running Tests

#### Math tests (no GPU required)
```powershell
python tests/torch/test_math_fp8.py
# 90/90 pass — validated against ml_dtypes reference
```

#### PyTorch GPU tests
```powershell
# Build extension first
python setup_torch.py build_ext --inplace

pytest tests/torch/test_fp8.py -v
```

#### Full Pipeline Tests (CPU Mock)
```powershell
python tests/test_pipeline.py -v
# Tests full integration of Adafactor, Shadow Linear, and activation compression
# Pure-Python mock, 100% test coverage for the API
```

---

## 🗂️ Project Structure

```
hip_quant/
├── __init__.py              # NumPy / ctypes offline API
├── __main__.py              # CLI entry point
├── torch_api.py             # PyTorch FP8 training API (Phases 1–4)
├── setup_torch.py           # PyTorch C++ extension build script
├── build.ps1                # DLL build script (hipcc)
├── hip_quantize.cpp         # Offline quantization kernels (DLL source)
├── hip_quant_util.h         # Shared FP8 / FP16 device helpers
├── hip_quant_types.h        # GGML block type definitions
├── kernels/                 # Per-format offline HIP kernels (.cu)
├── torch_ext/               # PyTorch extension source
│   ├── pytorch_bindings.cpp # C++ bindings (TORCH_CHECK, pybind11)
│   ├── fp8_quant_kernels.hip# Element-wise quant/dequant kernels
│   └── fp8_linear_kernels.hip# Tiled FP8 GEMM kernels
└── tests/torch/             # GPU test suite (pytest)
```

---

## 📋 Architecture Notes

- **gfx1201 target** — all kernels compiled with `--offload-arch=gfx1201`
- **No CPU transfers** — the PyTorch extension operates exclusively on device pointers obtained from `tensor.data_ptr()` and uses `at::cuda::getCurrentCUDAStream()`
- **Phase 4 GEMM** is a correctness-first tiled stub. Replace the inner loop with a `rocBLASLt` FP8 GEMM path for production throughput once validated on your ROCm stack
- **Offline API unchanged** — the NumPy/ctypes path is untouched; both APIs coexist cleanly
