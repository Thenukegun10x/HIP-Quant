<div align="center">
  <h1>🚀 hip-quant</h1>
  <p><b>Blazing Fast On-Device Tensor Quantization for AMD GPUs</b></p>
  <p>
    <img alt="ROCm 7.2.1" src="https://img.shields.io/badge/ROCm-7.2.1-ED1C24?logo=amd"/>
    <img alt="RDNA4" src="https://img.shields.io/badge/RDNA4-gfx1200%20%7C%20gfx1201-blue"/>
    <img alt="RDNA3" src="https://img.shields.io/badge/RDNA3-gfx1100%20%7C%20gfx1101%20%7C%20gfx1102%20%7C%20gfx1103-0096FF"/>
    <img alt="CDNA" src="https://img.shields.io/badge/CDNA-gfx90a%20%7C%20gfx942-purple"/>
    <img alt="BF16 FP16" src="https://img.shields.io/badge/PyTorch-BF16%20%7C%20FP16-green"/>
    <img alt="Python 3.8+" src="https://img.shields.io/badge/python-3.8+-3776AB?logo=python&logoColor=white"/>
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.9%2BROCm-EE4C2C?logo=pytorch"/>
  </p>
</div>

`hip-quant` is a standalone Python library and highly optimized HIP C++ backend that quantizes tensors directly on AMD GPUs with no CPU round-trips. The offline GGUF path consumes `float32`; the PyTorch FP8 training extension accepts `float32`, `float16`, and `bfloat16` tensors.

It ships **two independent APIs** that can be used together or separately:

| API | Purpose | Requires |
|---|---|---|
| **NumPy / ctypes** (offline) | Offline GGUF-format quantization via packaged DLL | ROCm runtime, numpy |
| **PyTorch extension** (training) | GPU-resident FP8 training ops with full autograd | PyTorch 2.x + ROCm, built `_C` extension |

## Hardware Status

Runtime validation is currently on RDNA4. The PyTorch FP8 WMMA kernels target `gfx1200` and `gfx1201`; `gfx1200` is treated as the cut-down `gfx1201` die with the same relevant FP8 WMMA capabilities.

CDNA support is included for the offline NumPy/DLL quantization path and compatibility tooling. The default DLL build now emits one all-target DLL for `gfx90a`, `gfx942`, RDNA3 `gfx1100`-`gfx1103`, and RDNA4 `gfx1200`/`gfx1201`. The gfx12 WMMA FP8 GEMM test is intentionally disabled on CDNA; CDNA can support FP8/BF16 through MFMA/rocBLASLt-style paths, but not this RDNA4-specific gfx12 WMMA builtin path.

### ⚠️ gfx12 FP8 WMMA Safety (Windows RDNA4)

The PyTorch FP8 training API (`Fp8Linear`, `Fp8ScaledLinear`, `Fp8ShadowLinear`) uses `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12` for the fused GEMM forward/backward pass. On **ROCm 7.1 with Windows gfx1201**, these WMMA intrinsics can trigger a GPU TDR (driver timeout) that corrupts GPU memory and may restart the PC.

**Root cause:** The ROCm 7.1 HIP runtime has a stability issue with `gfx12` WMMA instructions on RDNA4. The kernel launch succeeds but the GPU can hang asynchronously, causing subsequent `tensor.item()` calls to read back corrupted memory — typically manifesting as a `ZeroDivisionError` at `torch_api.py:495` (`1.0 / weight_inv_scale` with a zeroed GPU value).

**Fix:** Wheels now package a ROCm 7.2.1-built DLL named `hip_quantize_rocm721.dll` next to the legacy `hip_quantize.dll`. On Windows, `HipQuant()` prefers `hip_quantize_rocm721.dll` when present and searches the active Python environment's ROCm/PyTorch DLL directories before the system ROCm 7.1 path. The legacy DLL can still be forced with `HIP_QUANT_DLL_VARIANT=legacy`.

**Default safety policy:** gfx12 FP8/BF8 WMMA kernels are disabled by default because bad driver/compiler combinations can hang or reset the GPU. Enable them only for controlled testing:
```powershell
$env:HIP_QUANT_ENABLE_GFX12_WMMA = "1"
```

Force-disable WMMA regardless of runtime/device:
```powershell
$env:HIP_QUANT_DISABLE_WMMA = "1"
```

Validated local test system:
- GPU: AMD Radeon RX 9070 XT, `gfx1201`, 16 GB VRAM
- CPU: AMD Ryzen 7 7800X3D, 8 cores / 16 threads
- RAM: 32 GB system memory
- OS/toolchain: Windows, Visual Studio 2022 Build Tools, ROCm installed at `C:\Program Files\AMD\ROCm\7.1`
- PyTorch venv: `C:\venvs\medusa_rocm\Scripts\python.exe`
- PyTorch: `2.9.1+rocm7.2.1`, HIP runtime: `7.2.53211-158bd99533`
- FP8 WMMA microtest verified with the packaged ROCm 7.2.1 DLL: 50 bounded launches through `fp8_gemm_test_wmma`

> **Note:** The offline NumPy/DLL quantization path does **not** use WMMA and is unaffected. It works with both ROCm 7.1 and 7.2 runtimes. The packaged ROCm 7.2.1 DLL is preferred on Windows to avoid ROCm 7.1 gfx12 WMMA hazards when optional FP8 GEMM tests are enabled.

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
Default build emits one DLL for CDNA, RDNA3, and RDNA4 targets. By default it uses `C:\Program Files\AMD\ROCm\7.1\bin\hipcc.exe`; pass `-RocmBin` to use a ROCm/PyTorch venv toolchain:
```powershell
.\build.ps1

# Build the packaged ROCm 7.2.1 DLL from a PyTorch ROCm venv
.\build.ps1 -Output hip_quantize_rocm721.dll -RocmBin "C:\venvs\medusa_rocm\Scripts"

# Custom target set
.\build.ps1 -Arch "gfx942,gfx1200,gfx1201"
```

The build script adds `-mno-wavefrontsize64` so gfx12 `w32` WMMA code is compiled as Wave32.

### PyTorch Extension (`_C`)
Requires PyTorch with ROCm support (`torch 2.x+rocm`):
```powershell
& "C:\venvs\medusa_rocm\Scripts\python.exe" setup_torch.py build_ext --inplace
```

To build a PyPI wheel that includes the compiled `_C.pyd` extension, build with
the extension flag from the ROCm/PyTorch environment:
```powershell
$env:HIP_QUANT_BUILD_TORCH_EXT = "1"
& "C:\venvs\medusa_rocm\Scripts\python.exe" -m build --wheel --no-isolation
```

Without `HIP_QUANT_BUILD_TORCH_EXT=1`, `python -m build` creates a Windows wheel
that packages the ctypes DLLs but does not include `_C.pyd`. The PyTorch
extension can still be built locally with `setup_torch.py build_ext --inplace`.

---

## 📦 Installation

```powershell
# Binary wheel with packaged ROCm 7.2.1 ctypes DLL
pip install dist/hip_quant-0.4.8-cp312-cp312-win_amd64.whl

# With PyTorch optional dependency declared
pip install "hip-quant[torch]"
```

On Windows, DLL resolution order is:
- `HIP_QUANT_DLL` or `HIP_QUANT_DLL_PATH`, if set
- `hip_quantize_rocm721.dll`
- `hip_quantize.dll`

Runtime DLL directories include `HIP_QUANT_ROCM_BIN`, `HIP_QUANT_ROCM_HOME`, `ROCM_HOME`, `ROCM_PATH`, `HIP_PATH`, the active venv's `_rocm_sdk_core\bin`, `torch\lib`, `Scripts`, then the system ROCm 7.1 path.

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

`Fp8LinearFunction` uses **E4M3** for forward activations/weights and **E5M2** for backward gradients. It accepts `torch.float32`, `torch.float16`, and `torch.bfloat16` inputs/weights. It also implements **Activation Compression**, saving `uint8` tensors in the autograd graph to cut activation VRAM by 4× versus FP32, and 2× versus FP16/BF16.

BF16/FP16 support applies to:
- `quantize_e4m3()` and `quantize_e5m2()` inputs
- `Fp8LinearFunction` forward/backward
- `Fp8Linear`, `Fp8ScaledLinear`, `Fp8ShadowLinear`, `Fp8Conv1d`, and `Fp8Conv2d` module parameters and gradients
- `Fp8ShadowLinear` master weights, so user-selected BF16/FP16 master weights reduce persistent parameter and gradient VRAM versus FP32

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

#### FP8 Conv1d / Conv2d

`fp8_conv1d`, `fp8_conv2d`, `Fp8Conv1d`, and `Fp8Conv2d` lower convolution to
an unfold/im2col matrix multiply and reuse the same FP8 scaled linear backend.
That means hipBLASLt via PyTorch `torch._scaled_mm` is used first when
available, while the custom gfx12 WMMA path remains the fallback/testing path.

```python
import torch
from hip_quant.torch_api import Fp8Conv1d, Fp8Conv2d, fp8_conv1d, fp8_conv2d

x1 = torch.randn(8, 16, 1024, device="cuda", dtype=torch.bfloat16)
conv1 = Fp8Conv1d(16, 32, kernel_size=3, padding=1,
                  device="cuda", dtype=torch.bfloat16)
y1 = conv1(x1)
y1_func = fp8_conv1d(x1, conv1.weight, conv1.bias, padding=1)

x = torch.randn(8, 3, 224, 224, device="cuda", dtype=torch.bfloat16)
conv = Fp8Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                 device="cuda", dtype=torch.bfloat16)

y = conv(x)

# Functional form mirrors torch.nn.functional.conv2d for numeric parameters.
y2 = fp8_conv2d(x, conv.weight, conv.bias, stride=2, padding=1)
```

Supported convolution options: numeric `stride`, `padding`, `dilation`, and
`groups` with zero padding mode. Inputs and weights must be CUDA/HIP tensors.

**Combined VRAM savings for a 500M-param LLM:**
Before: ~7.6 GB (Weights 2GB, Acts 1.6GB, AdamW 4GB)
After: ~0.9 GB (Weights 0.5GB, Acts 0.4GB, Adafactor 4MB)

#### Direct autograd.Function

```python
from hip_quant.torch_api import Fp8LinearFunction

out = Fp8LinearFunction.apply(input, weight, bias)  # bias optional
```

#### Fused FP8 Linear Fallback (gfx12 WMMA kernels)

The high-level `Fp8Linear`, `Fp8ScaledLinear`, `Fp8ShadowLinear`, `Fp8Conv1d`,
and `Fp8Conv2d` APIs try the hipBLASLt-backed PyTorch `_scaled_mm` route first.
These direct custom WMMA entry points are the fallback/testing path.

These kernels are disabled by default. Enable only after validating your ROCm
runtime and GPU stability:
```powershell
$env:HIP_QUANT_ENABLE_GFX12_WMMA = "1"
```

```python
from hip_quant import (
    fp8_linear_forward,
    fp8_linear_forward_scaled,
    fp8_linear_forward_fp8_weight,
    fp8_linear_backward_input,
    fp8_linear_backward_input_scaled,
    fp8_linear_backward_weight,
    fp8_linear_backward_weight_scaled,
)

# [M,K] @ [N,K].T = [M,N]
# forward: E4M3 x E4M3 WMMA, backward: E5M2/BF8 x E5M2/BF8 WMMA
out        = fp8_linear_forward(input, weight, bias=None)
grad_in    = fp8_linear_backward_input(grad_output, weight)
grad_wt    = fp8_linear_backward_weight(grad_output, input)

# Scaled path used by Fp8ScaledLinear and Fp8ShadowLinear
out_scaled = fp8_linear_forward_scaled(input, weight, bias, input_scale, weight_scale)
grad_in_s  = fp8_linear_backward_input_scaled(grad_output, weight, weight_scale)
grad_wt_s  = fp8_linear_backward_weight_scaled(grad_output, input, input_scale)
```

These functions are also used by `Fp8Linear`, `Fp8ScaledLinear`, and
`Fp8ShadowLinear` after the extension is built.

#### gfx1201 FP8/BF16 Microbenchmark

Measured on the validated local RX 9070 XT `gfx1201` system with PyTorch
`2.9.1+rocm7.2.1` and `HIP_QUANT_ENABLE_GFX12_WMMA=1`:

```powershell
$env:HIP_QUANT_ENABLE_GFX12_WMMA = "1"
& "C:\venvs\medusa_rocm\Scripts\python.exe" tests\torch\bench_fp8.py
```

```text
Elementwise FP8 ops, shape=(4096, 4096), dtype=bf16
quantize_e4m3:   0.243 ms
quantize_e5m2:   0.205 ms
dequantize_e4m3: 0.218 ms
dequantize_e5m2: 0.206 ms
Fp8ShadowLinear, batch=32, in=4096, out=4096, dtype=bf16
forward:          2.553 ms
forward+backward: 5.622 ms
total wall time: 0.86 s
```

The benchmark is available at `tests/torch/bench_fp8.py`. Without
`HIP_QUANT_ENABLE_GFX12_WMMA=1`, it reports only the elementwise FP8 timings and
skips WMMA linear kernels.

The 0.4.8 FP8/BF16 optimization pass is primarily a speed and memory-bandwidth
improvement: it reuses pre-quantized FP8 activations/gradients, skips redundant
output zeroing, fuses bias stores, vectorizes elementwise FP8 kernels, and caches
offline FP8 temporary buffers. Persistent VRAM savings are still mainly provided
by `Fp8ShadowLinear` FP8 weight shadows and activation compression; this release
reduces transient allocations and extra memory passes around those features.

RDNA3 (`gfx11`) and CDNA devices are rejected for this specific builtin path.
CDNA FP8/BF16 GEMM should use an MFMA/rocBLASLt implementation instead.

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
- Wrong dtype (`float32` / `float16` / `bfloat16` for floating inputs, `uint8` for FP8 buffers)
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
& "C:\venvs\medusa_rocm\Scripts\python.exe" setup_torch.py build_ext --inplace

& "C:\venvs\medusa_rocm\Scripts\python.exe" -m pytest tests/torch/test_fp8.py -v
```

#### Full Pipeline Tests (CPU Mock)
```powershell
python tests/test_pipeline.py -v
# Tests full integration of Adafactor, Shadow Linear, and activation compression
# Pure-Python mock, 100% test coverage for the API
```

#### Compatibility Tests (CPU + DLL)
```powershell
& "C:\venvs\medusa_rocm\Scripts\python.exe" -m pytest tests/test_compat.py -v

# Device/compat reports
& "C:\venvs\medusa_rocm\Scripts\python.exe" -m hip_quant --info
& "C:\venvs\medusa_rocm\Scripts\python.exe" -m hip_quant --compat
```

#### Optional gfx12 FP8 WMMA Stress Test

Only run this on a stable ROCm 7.2+ gfx12 system. It can still reset the GPU on
bad driver/runtime combinations.
```powershell
$env:PYTHONPATH = "C:\path\to\src"
$env:HIP_QUANT_ENABLE_GFX12_WMMA = "1"
& "C:\venvs\medusa_rocm\Scripts\python.exe" test_fp8_gemm.py
```

The release DLL was locally checked with 50 bounded `fp8_gemm_test_wmma`
launches on `gfx1201` and HIP runtime `70253211`.

---

## 📤 Release / PyPI Upload

Build the distributables:
```powershell
$env:HIP_QUANT_BUILD_TORCH_EXT = "1"
& "C:\venvs\medusa_rocm\Scripts\python.exe" -m build --no-isolation
```

Check the artifacts:
```powershell
& "C:\venvs\medusa_rocm\Scripts\python.exe" -m twine check `
  "dist\hip_quant-0.4.8-cp312-cp312-win_amd64.whl" `
  "dist\hip_quant-0.4.8.tar.gz"
```

Upload to PyPI:
```powershell
& "C:\venvs\medusa_rocm\Scripts\python.exe" -m twine upload `
  "dist\hip_quant-0.4.8-cp312-cp312-win_amd64.whl" `
  "dist\hip_quant-0.4.8.tar.gz"
```

Do not upload stale universal wheels such as `hip_quant-0.4.8-py3-none-any.whl`.
The Windows wheel is intentionally platform-tagged because it contains DLLs.

Suggested release order:
- Build and run `twine check`
- Upload to TestPyPI or PyPI
- Install the uploaded package in a clean venv and verify `HipQuant().dll_path` resolves to `hip_quantize_rocm721.dll`
- Commit/tag the exact source and DLL used for the PyPI upload

---

## 🗂️ Project Structure

```
hip_quant/
├── __init__.py              # NumPy / ctypes offline API
├── __main__.py              # CLI entry point
├── torch_api.py             # PyTorch FP8 training API (Phases 1–4)
├── device_info.py           # GPU/DLL compatibility probe helpers
├── cdna_compat.py           # CDNA feature table, build configs, CPU refs
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

- **RDNA4 PyTorch target** — FP8 WMMA extension kernels are compiled with `--offload-arch=gfx1200` and `--offload-arch=gfx1201`
- **Default offline DLL target** — `build.ps1` compiles the portable DLL quantization kernels for `gfx90a`, `gfx942`, RDNA3 `gfx1100`-`gfx1103`, and RDNA4 `gfx1200`/`gfx1201`
- **Current validation scope** — runtime-tested locally on `gfx1201` RX 9070 XT; `gfx1200` and CDNA code objects are build-validated and need separate hardware runtime validation
- **BF16/FP16 PyTorch support** — FP8 quantization and linear kernels accept FP32, FP16, and BF16 tensors, accumulating in FP32 registers and storing results in the input/master dtype
- **Device-resident kernels** — FP8 tensor data stays on device through `tensor.data_ptr()`; scalar scale metadata is passed through legacy float launcher arguments. Non-MSVC builds use PyTorch's current stream, while Windows/MSVC ROCm builds currently fall back to the default HIP stream because the PyTorch HIP stream headers do not compile cleanly under MSVC.
- **Phase 4 GEMM** is a correctness-first tiled stub. Replace the inner loop with a `rocBLASLt` FP8 GEMM path for production throughput once validated on your ROCm stack
- **Offline API unchanged** — the NumPy/ctypes path is untouched; both APIs coexist cleanly
