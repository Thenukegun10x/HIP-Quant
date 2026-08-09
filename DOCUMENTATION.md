# hip-quant Documentation

## Overview

`hip-quant` is a standalone repository and python module for HIP/ROCm-based tensor quantization. It takes standard `float32` tensors and quantizes them directly on-device using highly optimized HIP C++ kernels, targeting AMD GPUs (specifically the `gfx1201` architecture / RDNA4) with ROCm 7.1.

It implements a wide variety of GGML-compatible quantization formats plus newer extensions:

### Supported Quantization Types
- **Legacy/Standard**: Q4_0, Q4_1, Q5_0, Q5_1, Q6_K, Q8_0, Q8_1, F8_E4M3
- **HQ Family (high quality)**: HQ2/AQ2, HQ3/AQ3, HQ8, IQ1_S, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS
- **I-Quants (byte-exact vs llama.cpp)**: IQ2_S, IQ1_M
- **Binary (byte-exact vs llama.cpp)**: Q1_0 (1.125 bpw, group 128), Q2_0 (2.25 bpw, group 64)
- **Other**: Q8_K (intermediate 8.25 bpw, block 256), BF16 (2 B/element cast)
- **FP8**: E4M3 and E5M2 (via MXFP4 bridge)

### Key Features Added Since Last Documentation Update
1. **WaveAttention FP8 Backward** (`torch_ext/wave_attn_backward.hip`, `wave_attn_diag.hip`) — native GFX12 WMMA backward for `_WaveAttentionFn` in `torch_api.py`. Computes dQ/dK/dV on-device without SDPA recomputation, with separate preprocess pass.
2. **HQ Family Module** (`hq2/`) — HQ2/AQ2 and HQ3/AQ3 quantization formats with full Python API: `hq2.api`, `hq2.hq3.py` (HQ3 variant), `hq2/archive.py`. Includes block structures in `hip_quant_types.h` for `hqb16_4x8`/`hqb16_4x4`/`hqb16_2x4` tile layouts.
3. **MXFP4 Support** — MX-format FP4 quantization (`mxfp4_to_fp8_kernels.hip`, `fp8_dequantize_linear_kernel_v2.hip`). Used in Q-to-FP8 conversion pipeline via `_to_mx_f8` and `_from_mx_f8`.
4. **Q-to-FP8 Bridge** — Dequantization bridge connecting quantized weights to FP8 for mixed-precision inference (see `q_to_fp8.py`, `fp8_dequantize_linear_kernel_v2.hip`).
5. **IQ2_S / IQ1_M Kernels** — on-device quantizers for ggml types 22/29 (`kernels/quant_iq2_s.cu`, `kernels/quant_iq1_m.cu`; 256-element blocks, 82/56 bytes, `codebooks/iq2_s.bin`). Verified byte-exact against `llama-quantize.exe` via `e2e_iq2s_iq1m.py` and `tests/test_iq2s_iq1m_byte_exact.py` across seeds, shapes, and magnitudes.
6. **Q1_0 / Q2_0 Kernels** — binary/2-bit quantizers for ggml types 41/42 (`kernels/quant_q1_0.cu`, `kernels/quant_q2_0.cu`; 128/64-element groups, 18 bytes, `d` = mean abs / max abs + sign bit / 2-bit codes). Byte-exact vs `llama-quantize.exe`.
7. **AQ2 Remap** — experimental AQ2 family moved from ggml IDs 39-41 to 31-33, freeing 41/42 for llama's Q1_0/Q2_0 and 39/40 for MXFP4/NVFP4.
8. **Q8_K / BF16** — Q8_K quantizer (`kernels/quant_q8_K.cu`, 292 B blocks, `d = -max/127` + bsums; byte-exact vs `quantize_row_q8_K_ref` replica) and BF16 cast kernel (`kernels/quant_bf16.cu`, RNE matching `ggml_fp32_to_bf16`, byte-exact vs `llama-quantize.exe`). Both registered as ggml types 15/30 with dequant-to-FP8 paths.
9. **Dequant for Q1_0 / Q2_0 / IQ2_S / IQ1_M** — `dequant_q1_0_to_fp8_kernel` / `dequant_q2_0_to_fp8_kernel` (sign-bit / `(q-1)*d` decode) plus `dequant_iq2_s_to_fp8_kernel` / `dequant_iq1_m_to_fp8_kernel` (codebook grid + sign/delta decode) added to `kernels/dequant_to_fp8.cu`, wired into the FP8 bridge for types 41/42/22/29. IQ2_S/IQ1_M decodes verified element-wise against `dequantize_row_iq2_s`/`dequantize_row_iq1_m` (0/1024 beyond E4M3 tolerance) and pinned with golden fixtures in the test suite.

## Project Structure

### Core Modules (`hip_quant/`)
```
__init__.py              # Python wrapper using ctypes → hip_quantize.dll, exports quantize_* functions
build.ps1                # PowerShell build script invoking hipcc to compile C++ source into DLL
hip_quant_types.h        # Shared header defining quant block structs (hqb8_4x4/hq2_4x8 etc.) and enum types
hip_quantize.cpp         # Native implementation of all quantization formats + FP8 conversion bridge
```

### Kernels (`kernels/`) — 30 HIP kernel files for on-device operations:
- `fp8_linear_kernels_v2.hip` / `fp8_linear_warmup_kernel_v2.hip` — FP8 linear layer kernels (v2 with warmup)
- `hq2_fp4_to_q6k_bf16_linear.hip` — HQ2 to Q6_K conversion for BF16 linear layers
- `mxfp4_to_fp8_kernels.hip` — MXFP4 quantization and FP8 dequantization kernels
- `fp8_dequantize_kernel_v1.hip` / `fp8_dequantize_linear_kernel_v1.hip` — v1 FP8 dequantization (legacy)
- `fp8_dequantize_linear_kernel_v2.hip` — v2 FP8 linear layer dequantization kernel
- `mxfp4_to_fp32_kernels.hip` / `mx_f8_to_float_kernel.hip` — MXFP4 to float conversion kernels

### PyTorch Extension (`torch_ext/`) — 5 HIP files for training-time operations:
- `wave_attn_backward.hip` — Native GFX12 FP8 WMMA backward kernel for WaveAttention, computing dQ/dK/dV on-device. Follows FlashAttention-2 with separate preprocess pass. Requires exact wave-per-sub-tile alignment (`THREADS/32 == K_TILE/16`).
- `wave_attn_diag.hip` — Diagonal preprocessing kernel supporting both FP8 E4M3 and BF16 formats, used by WaveAttention backward for computing D = sum_d(dO_id * O_id).
- `fp8_linear_forward_kernel_v2.hip` / `fp8_linear_warmup_kernel_v2.hip` — Forward pass kernels with warmup support.
- `mx_f8_to_float_kernel.hip` — MXFP4 to float conversion kernel for torch extension integration.

### Python Modules (`hq2/`)
```
api.py          # HQ2/AQ2 public API: quantize, dequantize, compute metrics (entropic_error, max_abs_error)
archive.py      # Archive management utilities for HQ format weight storage
hq3.py          # HQ3 variant implementation with modified block structures
```

### Testing (`tests/`) — 18 test files covering offline and GPU pipelines:
- `test_pipeline.py` / `test_torch_pipeline.py` — End-to-end pipeline tests (CPU-mocked for unit testing)
- `test_wave_attn_backward_math.py` — WaveAttention backward correctness with ragged lengths, non-unit scales, small dO magnitudes. ~0.999 cosine vs FP32 SDPA reference.
- `test_mxfp4_q_to_fp8_accuracy.py` / `test_mxfp4_fallback_mode_correctness.py` — MXFP4 to FP8 conversion accuracy tests with fallback mode validation.
- `test_hip_graph.py` / `hipgraph_utils.py` — HIP graph capture utilities for kernel performance optimization.

## Build System

### DLL (Offline/NumPy) Build
```powershell
.\build.ps1
```
Requires: `hipcc.exe` at `C:\Program Files\AMD\ROCm\7.1\bin\hipcc.exe`. Compiles C++ source into `hip_quantize.dll`.

### PyTorch Extension (Training/GPU) Build
```powershell
$env:HIP_QUANT_ENABLE_GFX12_WMMA='1'
"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
"C:\venvs\medusa_rocm\Scripts\python.exe" setup_torch.py build_ext --inplace
```
Requires: x64 VS toolchain, ROCm 7.1 SDK, CUDA_VISIBLE_DEVICES and HIP_VISIBLE_DEVICES set appropriately for testing.

### Packaging & PyPI Publishing
Uses `pyproject.toml` + `setuptools`. Bundle Python wrapper with compiled DLL:
```bash
python -m build       # Build .whl and .tar.gz
twine upload dist/*   # Upload to PyPI (requires credentials)
```

## Architecture Support
- **Primary Target**: AMD RDNA4 (`gfx1201`) — uses FP8 E4M3 WMMA intrinsics for WaveAttention backward.
- **Fallback**: SDPA-based gradients when `HIP_QUANT_WAVE_ATTN_SDPA_BACKWARD=1` is set (debugging only).

## Testing Notes

### CPU Pipeline Suite (`tests/test_pipeline.py`)
Mocked `_C`, no GPU required:
```powershell
& 'C:\venvs\medusa_rocm\Scripts\python.exe' -c "import unittest, tests.test_pipeline as t; unittest.main(module=t, exit=True)"
# or
& 'C:\venvs\medusa_rocm\Scripts\python.exe' -m pytest tests\test_pipeline.py -q
```

### GPU / WaveAttention Tests
Require real extension + x64 VS toolchain. Enable WMMA:
```powershell
$env:HIP_QUANT_ENABLE_GFX12_WMMA='1'
```

### Venv & Build Environment
- Python venv: `C:\venvs\medusa_rocm\Scripts\python.exe` (Python 3.12)
- ROCm SDK: version 7.1.x
- VS Toolchain: x64 Developer Command Prompt (`vcvars64.bat`)

## Line Counts (as of last update)
| File | Lines |
|------|-------|
| `__init__.py` | ~1,136 |
| `hip_quantize.cpp` | ~2,800+ |
| `torch_api.py` | ~3,573 |
