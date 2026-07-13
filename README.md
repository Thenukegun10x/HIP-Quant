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

### gfx12 FP8 WMMA (Windows RDNA4)

The PyTorch FP8 training API (`Fp8Linear`, `Fp8ScaledLinear`, and
`Fp8ShadowLinear`) and the ctypes micro-GEMM use the gfx12 Wave32 FP8/BF8
WMMA intrinsics. They are enabled by default on a `gfx1200`/`gfx1201` device
with ROCm/HIP 7.2 or newer. Both the Python and native entry points reject
other architectures or older runtimes before launch.

Windows wheels prefer the packaged ROCm 7.2.1-built
`hip_quantize_rocm721.dll`; set `HIP_QUANT_DLL_VARIANT=legacy` only when the
legacy DLL is deliberately required. To explicitly suppress the custom WMMA
path on a compatible system:
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
- Full WMMA diagnostic/stress suite verified: native diagnostic, padded and uneven-K GEMMs, 512²/1024² matrices, and repeated launches. Every numeric check compares the GEMM result with the decoded E4M3 operands actually consumed by WMMA.

> **Note:** The offline NumPy/DLL quantization path does not use WMMA. The
> custom PyTorch and micro-GEMM WMMA paths require gfx12 and ROCm/HIP 7.2+.

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

Default FP8 quantization uses OCP standard semantics with round-to-nearest-even. The PyTorch extension also exposes opt-in stochastic E5M2 rounding for backward gradients.

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
Requires PyTorch with ROCm support (`torch 2.x+rocm`) and an **x64** MSVC
toolchain (`Hostx64\x64\link.exe`). Do not build from an x86 Developer shell —
that produces `temp.win32` objects and fails to link a 64-bit `_C.pyd`.

```powershell
# From "x64 Native Tools Command Prompt for VS 2022", or after vcvars64.bat:
& "C:\venvs\medusa_rocm\Scripts\python.exe" setup_torch.py build_ext --inplace
```

Expect `build\temp.win-amd64-cpython-312` and `rc.exe` /
`link.exe` from the Windows SDK x64 and MSVC Hostx64 paths.

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

### Runtime codebooks and GPU execution features

I-Quant lookup data is shipped as five versioned binary files under
`codebooks/`, rather than being compiled into the DLL. The loader finds them
next to the DLL; set `HIP_QUANT_CODEBOOK_DIR` when deliberately using a DLL
outside its package directory. Regenerate and verify the checked-in assets with:

```powershell
& "C:\venvs\medusa_rocm\Scripts\python.exe" tools\export_iq_codebooks.py --check
```

The PyTorch extension also exposes `quantize_e4m3_transpose()` and
`quantize_e5m2_transpose()` for a rank-2 tensor, avoiding a materialised
full-precision transpose. For stable-shape inference, `capture_hip_graph()`
captures a callable through PyTorch's ROCm `CUDAGraph` interface (HIP Graph):

```python
runner = hip_quant.capture_hip_graph(model, example_tokens)
logits = runner.replay(next_tokens)
```

`replay()` returns independent outputs by default; use
`clone_output=False` only when the graph-owned output will be consumed before
the next replay.

### Current release capabilities

- WMMA diagnostics use Wave32-correct launch shapes, cooperative-group tiled
  barriers for LDS staging, and decoded-FP8 GEMM references; the custom path
  is enabled on supported gfx12 ROCm/HIP 7.2+ systems.
- I-Quant quantizers are parallelized to 256 threads, and legacy quantizers
  use Wave32-safe shuffle reductions. I-/T-Quant dequantization to E4M3/E5M2
  and direct FP8-to-quant fused kernels are covered by byte-level tests.
- The PyTorch extension uses PyTorch's current HIP stream, supports seeded
  stochastic E5M2 conversion, non-temporal FP8 memory hints where supported,
  and fused FP8 quantize-plus-transpose kernels.
- I-Quant codebooks are external, versioned package assets verified by CRC;
  the captured HIP Graph runner provides stable-shape replay without Python
  launch overhead.

---

## 📦 Installation

```powershell
# Binary wheel with packaged ROCm 7.2.1 ctypes DLL and PyTorch extension
pip install dist/hip_quant-0.5.6-cp312-cp312-win_amd64.whl

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

#### Q → FP8 dequantization (offline)
`dequantize_to_fp8` expands packed GGML Q blocks **directly** to raw FP8 bytes on
the GPU. Each thread reconstructs a scalar from the Q block and immediately
encodes it as E4M3 or E5M2 in the same kernel, so no float32 buffer is
allocated or transferred. Supported source types: legacy `Q4_0/Q4_1/Q5_0/Q5_1/
Q8_0/Q8_1`, K-quants `Q2_K` through `Q6_K`, I-quants `IQ1_S` through
`IQ4_XS` (including `IQ4_NL`), and ternary `TQ1_0`/`TQ2_0`.

```python
from hip_quant import GGML_TYPE, get_hip_quant

hq = get_hip_quant()
w  = np.random.randn(4096, 4096).astype(np.float32)

# First quantize to a narrow Q type, then expand straight to FP8 bytes
q4k = hq.quantize_numpy(w, GGML_TYPE["Q4_K"])          # 4-bit K-quant
e4m3 = hq.dequantize_to_e4m3(q4k, GGML_TYPE["Q4_K"], 4096)  # uint8 (4096,4096)
e5m2 = hq.dequantize_to_e5m2(q4k, GGML_TYPE["Q4_K"], 4096)  # uint8 (4096,4096)

# Generic form: pick output format at runtime
e4m3 = hq.dequantize_to_fp8(q4k, GGML_TYPE["Q4_K"], 4096, output_format="E4M3")
```

- The output array shape is `(nrows, n_per_row)` with one FP8 byte per logical
  element — same layout as `quantize_numpy(..., GGML_TYPE["F8_E4M3"])`.
- If the source type matches the requested FP8 format (`F8_E4M3`/`F8_E5M2`), the
  path short-circuits to a host byte copy and skips the GPU altogether.

#### CLI
```powershell
hip-quant --help
python -m hip_quant --help
```

---

### PyTorch Training API

> **Requires:** `python setup_torch.py build_ext --inplace` first.

#### Q → FP8 dequantization
`dequantize_q_to_fp8` expands packed GGML Q blocks **directly** to raw FP8 bytes on the GPU. It reads packed PyTorch GPU bytes and writes `torch.uint8` directly into another GPU tensor, eliminating expensive PCI-e transfers and host CPU decoding.

The PyTorch extension currently supports legacy `Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1`
and K-quants `Q2_K` through `Q6_K`; I-quants and ternary quants remain
offline-DLL-only for this particular API.

```python
import torch
import hip_quant.torch_api as hq
from hip_quant import GGML_TYPE

# Packed Q4_K bytes, residing on the GPU
packed_q = torch.load("q4k_tensor.pt").cuda()

# Direct GPU-to-GPU expansion -> torch.uint8
e4m3 = hq.dequantize_q_to_fp8(packed_q, GGML_TYPE["Q4_K"], n_per_row=4096, e5m2=False)

# Shortcuts
e4m3 = hq.dequantize_q_to_e4m3(packed_q, GGML_TYPE["Q4_K"], 4096)
e5m2 = hq.dequantize_q_to_e5m2(packed_q, GGML_TYPE["Q4_K"], 4096)
```

The output shape is `[nrows, n_per_row]`.

---

#### Element-wise FP8 quant / dequant (Phase 1 & 2)

```python
import torch
from hip_quant.torch_api import quantize_e4m3, dequantize_e4m3
from hip_quant.torch_api import quantize_e5m2, quantize_e5m2_stochastic, dequantize_e5m2

x = torch.randn(1024, 1024, device="cuda")  # stays on GPU the whole time

x_fp8  = quantize_e4m3(x)          # torch.uint8, same shape, same device
x_back = dequantize_e4m3(x_fp8)    # torch.float32, no CPU transfer

g_fp8 = quantize_e5m2_stochastic(x, seed=1234)  # reproducible stochastic E5M2
```

#### Stochastic E5M2 Backward Gradients

E5M2 has the range needed for backward gradients, but only two mantissa bits.
For tiny gradients, deterministic round-to-nearest-even can repeatedly flush or
bias values. `quantize_e5m2_stochastic()` rounds between adjacent E5M2 bins with
probability proportional to the input value's distance between those bins, using
a stateless per-element hash of `(seed, element_index)`.

Use it directly:
```python
from hip_quant.torch_api import quantize_e5m2_stochastic, dequantize_e5m2

grad_fp8 = quantize_e5m2_stochastic(grad, seed=42)
grad_sim = dequantize_e5m2(grad_fp8)
```

Enable stochastic E5M2 for FP8 linear backward `grad_output` quantization:
```powershell
$env:HIP_QUANT_STOCHASTIC_E5M2 = "1"

# Optional deterministic base seed for reproducible experiments
$env:HIP_QUANT_STOCHASTIC_E5M2_SEED = "1234"
```

This path is opt-in. It stochastic-quantizes `grad_output` once, dequantizes
those exact FP8 choices back to the training dtype, then reuses the existing
hipBLASLt/custom backward matrix kernels.

#### Block-wise FP8 Scaling

The PyTorch extension also exposes block-wise FP8 quantization. Values are stored
as raw FP8 bytes plus one FP32 dequant scale per block along the last dimension:

```text
real_value ~= fp8_value * fp32_block_scale
```

For an input shape `[..., K]`, the scale tensor has shape
`[..., ceil(K / block_size)]`.

```python
from hip_quant.torch_api import (
    quantize_e4m3_blockwise,
    quantize_e5m2_blockwise,
    quantize_e5m2_blockwise_stochastic,
    dequantize_e4m3_blockwise,
    refresh_fp8_blockwise_shadow,
)

x = torch.randn(8, 4096, device="cuda", dtype=torch.bfloat16)
grad = torch.randn_like(x)
weight = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)

# Forward activations/weights: E4M3 + FP32 per-block scales
x_fp8, x_scales = quantize_e4m3_blockwise(x, block_size=32)
x_back = dequantize_e4m3_blockwise(x_fp8, x_scales, block_size=32)

# Backward gradients: E5M2 + stochastic rounding + FP32 per-block scales
g_fp8, g_scales = quantize_e5m2_blockwise_stochastic(grad, block_size=32, seed=1234)

# Master weight -> block-wise FP8 shadow buffers
weight_fp8, weight_scales = refresh_fp8_blockwise_shadow(weight, block_size=32)
```

Block-wise scaling is useful when a tensor has uneven dynamic range across its
last dimension. It usually reduces FP8 quantization error compared with one
global scale for the entire tensor. Existing per-tensor FP8 APIs remain unchanged.

#### Block-scaled Linear and Adafactor Kernel Helpers

Two lower-level training helpers are available for experiments and future fused
training paths:

```python
from hip_quant.torch_api import (
    adafactor_row_col_mean_square,
    fp8_linear_forward_blockwise,
    fp8_linear_forward_blockwise_quantized,
)

# GPU-side Adafactor 2-D statistics
row_ms, col_ms = adafactor_row_col_mean_square(grad_2d, eps=1e-30)

# Convenience path: quantize input/weight block-wise, then run block-scaled FP8 linear
out = fp8_linear_forward_blockwise(input, weight, bias=bias, block_size=32)

# Pre-quantized path: consumes FP8 bytes + FP32 scale tensors directly
out = fp8_linear_forward_blockwise_quantized(
    input_fp8, input_scales,
    weight_fp8, weight_scales,
    output_dtype_source=input,
    block_size=32,
    bias=bias,
)
```

The current block-scaled linear kernel is correctness-first and intentionally
does not use gfx12 WMMA yet. It validates the `FP8 bytes + per-block scales`
layout and math before replacing the inner loop with a tiled/WMMA or rocBLASLt
implementation. `Adafactor` is a full Python optimizer step with factored
second-moment state; GPU-side row/column mean-square helpers exist, while a
fully fused Adafactor update kernel remains a future optimization.

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

# One nonfinite gradient skips the entire optimizer step (no state poison).
loss.backward()
opt.step()
if opt.last_step_skipped:
    # reduce loss scale / skip weight update for this step
    pass
```

Training-path numerical guards:
- hipBLASLt backward applies quant scales as *pre-quant multipliers* and uses
  the reciprocal as the GEMM dequant scale (avoids exploding `grad_input`).
- `Fp8TensorMeta.update` ignores NaN/Inf amax samples and keeps the last valid
  scale (`found_nonfinite` sticky flag).
- `Adafactor.step` preflights all gradients and sets `last_step_skipped=True`
  instead of mutating state when any gradient is nonfinite.
- Weight-gradient kernels can retain FP32 accumulation for master weights.
- `Fp8ShadowLinear` caches the FP8 weight shadow between optimizer updates;
  hipBLASLt reuses pre-quantized E4M3 bytes instead of re-casting masters.

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

#### Fused FP8 Linear (gfx12 WMMA kernels)

The high-level `Fp8Linear`, `Fp8ScaledLinear`, `Fp8ShadowLinear`, `Fp8Conv1d`,
and `Fp8Conv2d` APIs try the hipBLASLt-backed PyTorch `_scaled_mm` route first.
These direct custom WMMA entry points are the fallback/testing path on a
compatible gfx12 ROCm/HIP 7.2+ device. Set `HIP_QUANT_DISABLE_WMMA=1` to
explicitly disable them.

```python
from hip_quant import (
    fp8_linear_forward,
    fp8_linear_forward_scaled,
    fp8_linear_forward_fp8_weight,
    fp8_linear_forward_blockwise,
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

# Correctness-first block-scaled FP8 path, no WMMA requirement
out_block = fp8_linear_forward_blockwise(input, weight, bias, block_size=32)
```

These functions are also used by `Fp8Linear`, `Fp8ScaledLinear`, and
`Fp8ShadowLinear` after the extension is built.

#### gfx1201 FP8/BF16 Microbenchmark

Measured on the validated local RX 9070 XT `gfx1201` system with PyTorch
`2.9.1+rocm7.2.1`:

```powershell
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

The benchmark is available at `tests/torch/bench_fp8.py`. It measures custom
WMMA automatically on a compatible device and reports the runtime guard reason
otherwise.

The 0.6.0 FP8/BF16 optimization pass is primarily a speed and memory-bandwidth
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
- **Positive finite scales** — invalid FP8 scales raise before launch / GEMM
- **Nonfinite training step** — delayed scales and Adafactor refuse to poison state

---

## 🧪 Running Tests

#### Math tests (no GPU required)
```powershell
python tests/torch/test_math_fp8.py
# 90/90 pass — validated against ml_dtypes reference
```

#### Full pipeline tests (CPU mock, no GPU required)
```powershell
# Preferred: pure unittest (no GPU init)
& "C:\venvs\medusa_rocm\Scripts\python.exe" -c "import unittest, tests.test_pipeline as t; unittest.main(module=t, exit=True)"

# Or pytest
& "C:\venvs\medusa_rocm\Scripts\python.exe" -m pytest tests/test_pipeline.py -q
```

`Fp8TensorMeta` and scaled/shadow modules keep delayed-scale metadata on the
parameter device (CPU by default). That avoids accidental ROCm init during the
mocked CPU suite. If a GPU-enabled run still stalls on process exit under
Windows, force CPU visibility:

```powershell
$env:CUDA_VISIBLE_DEVICES = ""
$env:HIP_VISIBLE_DEVICES = ""
```

#### PyTorch GPU tests
```powershell
# Build extension first (x64 VS toolchain)
& "C:\venvs\medusa_rocm\Scripts\python.exe" setup_torch.py build_ext --inplace
& "C:\venvs\medusa_rocm\Scripts\python.exe" -m pytest tests/torch/test_fp8.py -v
```

#### Compatibility Tests (CPU + DLL)
```powershell
& "C:\venvs\medusa_rocm\Scripts\python.exe" -m pytest tests/test_compat.py -v

# Device/compat reports
& "C:\venvs\medusa_rocm\Scripts\python.exe" -m hip_quant --info
& "C:\venvs\medusa_rocm\Scripts\python.exe" -m hip_quant --compat
```

#### Optional gfx12 FP8 WMMA Stress Test

Requires a ROCm/HIP 7.2+ gfx12 system.
```powershell
& "C:\venvs\medusa_rocm\Scripts\python.exe" -m pytest tests\test_wmma_diag.py -v

# Include 512²/1024² and repeated-large coverage
$env:HIP_QUANT_WMMA_STRESS = "1"
& "C:\venvs\medusa_rocm\Scripts\python.exe" -m pytest tests\test_wmma_diag.py -v
```

The release DLL was checked on `gfx1201` with HIP runtime `70253211`.

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
  "dist\hip_quant-0.5.6-cp312-cp312-win_amd64.whl" `
  "dist\hip_quant-0.5.6.tar.gz"
```

Upload to PyPI:
```powershell
& "C:\venvs\medusa_rocm\Scripts\python.exe" -m twine upload `
  "dist\hip_quant-0.5.6-cp312-cp312-win_amd64.whl" `
  "dist\hip_quant-0.5.6.tar.gz"
```

Do not upload stale universal wheels such as `hip_quant-0.5.6-py3-none-any.whl`.
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
- **Device-resident kernels** — FP8 tensor data stays on device through `tensor.data_ptr()`. hipBLASLt training paths keep delayed scales device-resident where possible; legacy custom WMMA launchers still take scalar float scales. Block-wise FP8 metadata stays in device FP32 scale tensors. All extension launches use PyTorch's current HIP stream, including the Windows/MSVC build.
- **Phase 4 GEMM** includes gfx12 WMMA per-tensor-scale paths, packed-weight WMMA variants, and a correctness-first block-scaled FP8 linear path. Large training shapes prefer hipBLASLt via `torch._scaled_mm`; custom WMMA remains the fallback/small-shape path.
- **Adafactor** provides a complete optimizer step in Python with nonfinite step skipping. GPU-side row/column mean-square reductions for 2-D gradients exist; a fully fused Adafactor update kernel is a future optimization target.
- **Offline API unchanged** — the NumPy/ctypes path is untouched; both APIs coexist cleanly
