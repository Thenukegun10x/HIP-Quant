# hip-quant Documentation

## Overview

`hip-quant` is a high-performance ROCm/HIP library and Python module for AMD GPUs (specifically targeting RDNA4 `gfx1200`/`gfx1201`, RDNA3, and CDNA architectures) with ROCm 7.1+.

Initially built as an on-device quantization library, **hip-quant 2.0** expands into a complete accelerated LLM inference and kernel suite. It provides:
1. **Native AOT Quantized GEMV Engine**: Memory-bandwidth bound single-pass GEMV kernels for GGUF formats (`Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2_K`..`Q6_K`, `IQ` family).
2. **Gated DeltaNet State Space Model (SSM) Suite**: High-throughput fused recurrent kernels, causal 1D convolution, and gated RMSNorm for linear attention and hybrid models (e.g. Qwen 3.5 / Qwen 3.8).
3. **WaveAttention Suite**: Native GFX12 FP8 WMMA flash attention across prefill, decode, long-context chunking, and full autograd backward without SDPA recomputation.
4. **Streaming GGUF Parser & Tensor Loader**: Zero-dependency pure-Python streaming loader mapping quantized GGUF tensors directly to GPU VRAM.
5. **GPU-SMI v1.2 Monitoring**: Bundled lightweight CLI for real-time VRAM allocation, compute load, junction temperatures, power, and per-process memory tracking.
6. **On-Device Quantization**: Direct on-GPU quantization for standard FP32/FP16 tensors with zero host round-trips.

---

## License

`hip-quant` is open-source software licensed under the **Apache License, Version 2.0**.
See the [LICENSE](file:///C:/Users/armor/Desktop/hip_quant/LICENSE) file for the full license text.

---

## Supported Quantization Types
- **Legacy/Standard**: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q1_0, Q2_0, F8_E4M3
- **K-Quants**: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K
- **HQ Family (high quality)**: HQ2/AQ2, HQ3/AQ3, HQ8, IQ1_S, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS
- **I-Quants (byte-exact vs llama.cpp)**: IQ2_S, IQ1_M
- **Ternary**: TQ1_0, TQ2_0
- **Other**: BF16 (2 B/element cast)
- **FP8**: E4M3 and E5M2 (via MXFP4 bridge), MXFP8 microscaling

#### imatrix contract (I-Quants)
llama.cpp imatrix is one float per input column shared across rows
(`quantize_iq2_xs` reuses the same pointer every row). hip_quant kernels
index `imatrix + row*n_per_row`, so `quantize_numpy` requires a full
`arr.shape` matrix — tile llama `.dat` column vectors across rows first.
`quantize_from_fp8` does not validate imatrix shape yet (pass a full matrix).
Weighting math matches llama; the byte-exact suite currently covers the
all-ones (identity) case only. Full rollout: `IMATRIX_PLAN.md`.

---

## What's New in 2.0.0
1. **Native AOT Quantized GEMV Engine** (`torch_ext/gemv_q_kernels.hip`) — Ahead-of-time single-pass dequantization + dot product kernels for 15+ quantization formats (`Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL`, `IQ4_XS`). Features warp-cooperative 32-lane shuffle reductions and split-K chunking for high memory-bandwidth efficiency.
2. **Gated DeltaNet SSM Engine** (`torch_ext/ssm_kernels.hip`) — Hardware-optimized decode and state-update kernels for hybrid SSM/Transformer models like Qwen 3.5 / 3.8. Includes fused SSM Conv1D, recurrent state stepping, and plain + gated RMSNorm with stride-aware weight broadcasting (`w_stride = 0` for shared `[D]`, `w_stride = D` for per-head `[H, D]`).
3. **WaveAttention Suite** (`torch_ext/wave_attn_*.hip`) — Native GFX12 FP8 WMMA flash attention calling `v_wmma_f32_16x16x16_fp8_fp8` directly. Encompasses prefill (`wave_attn_prefill.hip`), decode with INT4 KV cache (`kv_iu4_kernels.hip`, `wave_attn_decode.hip`), long-context chunking (`wave_attn_long.hip`), and full autograd backward (`wave_attn_backward.hip`, `wave_attn_diag.hip`).
4. **Streaming Zero-Dependency GGUF Loader** (`gguf.py`, `gguf_loader.py`) — Pure-Python GGUF header and tensor parser streaming multi-gigabyte models directly to GPU memory without host duplication.
5. **GPU-SMI v1.2 Integration** (`smi.py`, `tools/gpu-smi.exe`) — Integrated low-overhead GPU monitor tracking committed/shared VRAM, core/junction thermals, fan speeds, board power, and per-process allocations. Accessible directly via `hip-quant smi` or `python -m hip_quant.smi`.
6. **Apache 2.0 License** — Formally relicensed under Apache 2.0.

## Project Structure

### Core Modules (`hip_quant/`)
```
__init__.py              # Python wrapper using ctypes → hip_quantize.dll, exports quantize_* functions & CLI
torch_api.py             # PyTorch high-level API for GEMVs, WaveAttention, and SSM kernels
gguf.py                  # Zero-dependency streaming GGUF format parser
gguf_loader.py           # High-speed GGUF weight loader directly to VRAM
smi.py                   # Embedded GPU-SMI CLI and Python monitor
build.ps1                # PowerShell build script invoking hipcc to compile C++ source into DLL
hip_quant_types.h        # Shared header defining quant block structs (hqb8_4x4/hq2_4x8 etc.) and enum types
hip_quantize.cpp         # Native implementation of all quantization formats + FP8 conversion bridge
```

### PyTorch Extension (`torch_ext/`) — HIP files for training + inference:
- `pytorch_bindings.cpp` — Bindings with `TORCH_CHECK` validation (CUDA, contiguity, dtype, numel contracts) on every entry point.
- `gemv_q_kernels.hip` — Native AOT GEMV for quantized decode (Q4_0/Q8_0 and IQ/Q_K family) used by `hip_inference`.
- `ssm_kernels.hip` — Gated DeltaNet decode, SSM Conv1D, plain + gated RMSNorm for Qwen3.5/3.8 inference. The gated kernel supports shared `[D]` and per-head `[H,D]` weights via `w_stride` (dispatched + validated in the binding); passing any other size raises.
- `wave_attn_prefill.hip` / `wave_attn_decode.hip` — High-performance WaveAttention FP8 WMMA forward paths.
- `wave_attn_long.hip` — Long-context chunked WaveAttention execution.
- `wave_attn_backward.hip` — Native GFX12 FP8 WMMA backward kernel for WaveAttention, computing dQ/dK/dV on-device. Follows FlashAttention-2 with separate preprocess pass. Requires exact wave-per-sub-tile alignment (`THREADS/32 == K_TILE/16`).
- `wave_attn_diag.hip` — Diagonal preprocessing kernel supporting both FP8 E4M3 and BF16 formats, used by WaveAttention backward for computing D = sum_d(dO_id * O_id).
- `kv_iu4_kernels.hip` — Asymmetric INT4 KV-cache quantization routines.

### Utilities (`tools/`)
- `gpu-smi.exe` — Standalone native Windows GPU monitoring executable bundled with wheel.

### Kernels (`kernels/`) — 30 HIP kernel files for on-device operations:
- `fp8_linear_kernels_v2.hip` / `fp8_linear_warmup_kernel_v2.hip` — FP8 linear layer kernels (v2 with warmup)
- `hq2_fp4_to_q6k_bf16_linear.hip` — HQ2 to Q6_K conversion for BF16 linear layers
- `mxfp4_to_fp8_kernels.hip` — MXFP4 quantization and FP8 dequantization kernels
- `fp8_dequantize_kernel_v1.hip` / `fp8_dequantize_linear_kernel_v1.hip` — v1 FP8 dequantization (legacy)
- `fp8_dequantize_linear_kernel_v2.hip` — v2 FP8 linear layer dequantization kernel
- `mxfp4_to_fp32_kernels.hip` / `mx_f8_to_float_kernel.hip` — MXFP4 to float conversion kernels

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

## Line Counts (2.0.0)
| File | Lines |
|------|:-----:|
| `torch_api.py` | 3,762 |
| `torch_ext/pytorch_bindings.cpp` | 3,298 |
| `hip_quantize.cpp` | 1,887 |
| `torch_ext/gemv_q_kernels.hip` | 1,624 |
| `__init__.py` | 1,209 |
| `torch_ext/wave_attn_backward.hip` | 797 |
| `torch_ext/ssm_kernels.hip` | 405 |

## Related: hip_inference

Nested PyTorch inference engine (`hip_inference/`, own repo + README/AGENTS).
Qwen3 dense + Qwen3.5/3.8 hybrid (SSM) runners on the native kernels.
`python -m hip_inference.debug_decode` is the permanent per-step latency +
NaN/inf health profiler — use it before theorizing about tok/s. Current
decode profile (Qwen3.8-27B, Sep 2026): FFN ~64%, SSM ~30%, attention ~10%,
LM head ~9ms; all quantized GEMVs run ~65-77 GB/s effective vs ~640 GB/s
HBM roof (see `hip_inference/README.md`).
