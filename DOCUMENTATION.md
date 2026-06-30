# hip-quant — Complete Documentation

## 1. Overview

`hip-quant` is a Python library and highly-optimized HIP C++ backend for quantizing tensors directly on AMD GPUs without CPU round-trips. It provides **two independent APIs**:

| API | Purpose | Runtime Dependency |
|---|---|---|
| **NumPy/ctypes** (offline) | GGUF-format quantization via a packaged DLL | ROCm HIP runtime, numpy |
| **PyTorch Extension** (training) | GPU-resident FP8 training ops with full autograd | PyTorch 2.x + ROCm, built `_C.pyd` |

---

## 2. Supported Quantization Formats

### 2.1 Type Enumeration

Defined in `hip_quant/__init__.py` as dict `GGML_TYPE`:

| Type Name | ID | Block Size | Block Bytes | Category |
|---|---|---|---|---|
| `Q4_0` | 2 | 32 | 18 | Legacy 4-bit symmetric |
| `Q4_1` | 3 | 32 | 20 | Legacy 4-bit asymmetric |
| `Q5_0` | 6 | 32 | 22 | Legacy 5-bit symmetric |
| `Q5_1` | 7 | 32 | 24 | Legacy 5-bit asymmetric |
| `Q8_0` | 8 | 32 | 34 | Legacy 8-bit symmetric |
| `Q8_1` | 9 | 32 | 36 | Legacy 8-bit asymmetric |
| `Q2_K` | 10 | 256 | 84 | K-Quant 2-bit |
| `Q3_K` | 11 | 256 | 110 | K-Quant 3-bit |
| `Q4_K` | 12 | 256 | 144 | K-Quant 4-bit |
| `Q5_K` | 13 | 256 | 176 | K-Quant 5-bit |
| `Q6_K` | 14 | 256 | 210 | K-Quant 6-bit |
| `IQ2_XXS` | 16 | 256 | 66 | I-Quant 2-bit (extreme) |
| `IQ2_XS` | 17 | 256 | 74 | I-Quant 2-bit |
| `IQ3_XXS` | 18 | 256 | 98 | I-Quant 3-bit (extreme) |
| `IQ1_S` | 19 | 256 | 50 | I-Quant 1-bit |
| `IQ4_NL` | 20 | 32 | 18 | I-Quant 4-bit (non-linear) |
| `IQ3_S` | 21 | 256 | 110 | I-Quant 3-bit |
| `IQ4_XS` | 23 | 256 | 136 | I-Quant 4-bit (extreme) |
| `TQ1_0` | 34 | 256 | 54 | Ternary quant 1.69 bpw |
| `TQ2_0` | 35 | 256 | 66 | Ternary quant 2.06 bpw |
| `F8_E4M3` | 36 | 32 | 32 | FP8 E4M3 (forward activations) |
| `F8_E5M2` | 37 | 32 | 32 | FP8 E5M2 (backward gradients) |

### 2.2 FP8 Encoding Details

Both formats use OCP (Open Compute Project) standard semantics with round-to-nearest-even:

**E4M3** (`hip_quant_util.h:79-137`):
- Layout: 1 sign | 4 exponent | 3 mantissa, bias=7
- Max finite: ±448, NaN only (S.1111.111 = 0x7F/0xFF), no infinities
- Used for: forward activations and weights

**E5M2** (`hip_quant_util.h:173-231`):
- Layout: 1 sign | 5 exponent | 2 mantissa, bias=15
- Max finite: ±57344, supports infinities and NaN
- Used for: backward gradients

---

## 3. Project Structure

```
hip_quant/
├── __init__.py                 # NumPy/ctypes offline API — the main module (561 lines)
│   ├── GGML_TYPE              # Dict mapping type names to numeric IDs
│   ├── GGML_TYPE_BLOCK_SIZE   # Block size per type
│   ├── GGML_TYPE_BLOCK_BYTES  # Byte size per block per type
│   ├── HipQuant class         # Main ctypes wrapper
│   │   ├── quantize_numpy()   # F32 → quantized bytes
│   │   ├── quantize_from_fp8()# FP8 input → requantize (E4M3 or E5M2 source)
│   │   ├── fp8_gemm_test_wmma()# Micro FP8 GEMM via gfx12 WMMA intrinsic
│   │   ├── type_size/blck_size/row_size
│   │   └── device_name/device_count/gcn_arch/hip_runtime_version
│   ├── get_hip_quant()        # Singleton loader
│   ├── quantize()             # Shorthand
│   ├── probe_device()         # GPU/DLL probe (delegates to device_info.py)
│   ├── report_device()        # Formatted GPU report
│   └── cpu_reference_quantize()# CPU reference for testing
│
├── __main__.py                 # CLI entry point (hip-quant command)
│   ├── --info                 # Print device/DLL information
│   ├── --compat               # Print CDNA/RDNA compatibility report
│   ├── --list-types           # List supported types
│   ├── --type / -t            # Quantization type (default: Q4_K)
│   ├── --fp8-source           # Input is uint8 FP8, not float32
│   ├── --imatrix              # Importance matrix (.npy)
│   └── --dll                  # Path to custom DLL
│
├── torch_api.py                # PyTorch FP8 training API (1177 lines)
│   ├── Phase 2: quantize_e4m3, quantize_e5m2, dequantize_e4m3, dequantize_e5m2
│   ├── Phase 3: Autograd-safe FP8 linear layers
│   │   ├── Fp8LinearFunction          # Base autograd.Function (unscaled)
│   │   ├── Fp8ScaledLinearFunction    # Per-tensor amax scaling
│   │   ├── Fp8ShadowLinearFunction    # FP8 weight storage + master weight
│   │   ├── Fp8Linear                  # nn.Module (unscaled)
│   │   ├── Fp8ScaledLinear            # nn.Module with delayed scaling
│   │   ├── Fp8ShadowLinear            # nn.Module with FP8 weight storage
│   │   └── convert_to_fp8()           # One-call model converter
│   ├── Phase 4: Direct HIP GEMM kernel bindings
│   │   ├── fp8_linear_forward()
│   │   ├── fp8_linear_forward_scaled()
│   │   ├── fp8_linear_forward_fp8_weight()
│   │   ├── fp8_linear_backward_input()
│   │   ├── fp8_linear_backward_input_scaled()
│   │   ├── fp8_linear_backward_weight()
│   │   └── fp8_linear_backward_weight_scaled()
│   ├── Fp8TensorMeta                  # Delayed-scaling amax tracker
│   └── Adafactor                      # Memory-efficient optimizer
│
├── device_info.py              # GPU/DLL compatibility probe (328 lines)
│   ├── DeviceProperties dataclass
│   │   ├── name, gcn_arch, major, minor, cu_count
│   │   ├── total_memory, free_memory, shared_mem, warp_size
│   │   ├── has_wmma, hip_runtime_version, dll_loaded
│   │   └── Properties: arch_family, arch_note, memory_gb, is_cdna, is_rdna4
│   ├── probe_device(dll_path) # Probe GPU via shared library
│   ├── report(dev)            # Formatted report string
│   └── ARCH_FAMILIES          # Architecture classification table
│
├── cdna_compat.py              # CDNA compatibility and CPU reference (334 lines)
│   ├── ARCH_FEATURES          # Feature table: wmma, mfma, fp8, dp4a, wave32, wave64
│   ├── arch_supports_feature()# Query feature support
│   ├── get_build_archs()      # Recommended --offload-arch list
│   ├── build_config_for_arch()# Build config dict
│   ├── cpu_reference_quantize()# Bit-exact CPU emulation (Q4_0, Q8_0)
│   └── suggest_emulation()    # CDNA testing guidance
│
├── diagnose_fp8_crash.py       # 8-stage diagnostic tool (638 lines)
│   ├── Stage 1: System information
│   ├── Stage 2: PyTorch basic CUDA ops
│   ├── Stage 3: DLL basic HIP operations (no WMMA)
│   ├── Stage 4: PyTorch extension quant/dequant (no WMMA)
│   ├── Stage 5: FP8 WMMA GEMM via DLL
│   ├── Stage 6: FP8 linear forward via PyTorch extension
│   ├── Stage 7: Progressive stress test (TDR detection)
│   └── Stage 8: GPU stability aftermath check
│
├── setup_torch.py               # PyTorch C++ extension build script (109 lines)
│   ├── _short_path()          # Windows 8.3 short-path helper
│   ├── _configure_windows_toolchain()# Auto-detects VS2022 + ROCm 7.1
│   └── CUDAExtension config   # Compiles pytorch_bindings.cpp + .hip files
│
├── setup.py                    # setuptools setup script (117 lines)
│   ├── _torch_extension_config()# Optional _C build via HIP_QUANT_BUILD_TORCH_EXT
│   └── bdist_wheel            # Forces platform-tagged wheel (root_is_pure=False)
│
├── build.ps1                    # Windows DLL build script (hipcc) (127 lines)
│   ├── Parameters: -Output, -CDNA, -All, -Arch, -RocmBin
│   ├── Default: all archs (gfx90a, gfx942, gfx1100-1103, gfx1200-1201)
│   └── Uses Get-ShortPath for space-safe paths
│
├── build.sh                     # Linux/macOS DLL build script (65 lines)
│   ├── Same arch options via ALL, CDNA, ARCH env vars
│   └── Uses -fPIC instead of Windows-specific flags
│
├── hip_quantize.cpp             # Main DLL source: kernel dispatch + HIP init (781 lines)
│   ├── Dynamic GPU selection (picks device with most CUs)
│   ├── I-Quant lookup table upload (IQ1S, IQ2XXS, IQ2XS, IQ3XXS, IQ3S)
│   ├── Thread-local GPU buffer caching
│   ├── quantize_tensor()       # F32 → any GGML type
│   ├── quantize_tensor_fp8_input() # FP8 E4M3 → requantize
│   ├── quantize_tensor_fp8_e5m2_input() # FP8 E5M2 → requantize
│   ├── fp8_gemm_test_wmma()    # Micro FP8 GEMM via gfx12 WMMA
│   ├── get_device_count, get_device_name, get_arch_name
│   ├── get_device_prop, get_device_memory, device_has_wmma
│   ├── get_hip_runtime_version
│   └── quantize_reset()        # Free cached GPU buffers
│
├── hip_quant_types.h            # GGML block struct definitions (145 lines)
│   ├── block_q4_0 through block_f8_e5m2 (20 block types)
│   ├── QK_K = 256 for K-Quants and I-Quants
│   └── ggml_half = uint16_t (FP16 storage)
│
├── hip_quant_util.h             # Device-side conversion functions (261 lines)
│   ├── nearest_int()           # Round-to-nearest-even
│   ├── fp32_to_fp16 / fp16_to_fp32
│   ├── fp32_to_bf16 / bf16_to_fp32
│   ├── fp32_to_fp8_e4m3 / fp8_e4m3_to_fp32
│   └── fp32_to_fp8_e5m2 / fp8_e5m2_to_fp32
│
├── hip_iquant_util.h            # I-Quant device utilities (nearest neighbour tables)
│
├── kernels/                     # 24 per-format HIP kernel files
│   ├── quant_q4_0.cu – quant_tq2_0.cu  (standard + K-quants)
│   ├── quant_iq1_s.cu – quant_iq4_xs.cu (I-Quants)
│   ├── quant_f8_e4m3.cu, quant_f8_e5m2.cu (FP8 formats)
│   ├── fp8_expand.cu           # FP8 → float32 device expansion
│   └── fp8_gemm_test.cu        # WMMA micro GEMM (__builtin_amdgcn_wmma)
│
├── torch_ext/                   # PyTorch extension source
│   ├── pytorch_bindings.cpp     # Pybind11 bindings + safety checks (642 lines)
│   │   ├── quantize_e4m3, quantize_e5m2
│   │   ├── dequantize_e4m3, dequantize_e5m2
│   │   ├── fp8_linear_forward (4 variants)
│   │   ├── fp8_linear_backward_input (2 variants)
│   │   ├── fp8_linear_backward_weight (2 variants)
│   │   ├── checked_int()       # int64 → int narrowing guard
│   │   ├── check_grid_dim()    # HW grid dimension limit guard
│   │   └── check_gfx12_fp8_wmma_runtime()  # WMMA safety gates
│   ├── fp8_quant_kernels.hip   # Element-wise quant/dequant device kernels (147 lines)
│   │   ├── quant_e4m3_kernel, quant_e5m2_kernel
│   │   ├── dequant_e4m3_kernel, dequant_e5m2_kernel
│   │   └── C-linkage launch wrappers
│   └── fp8_linear_kernels.hip  # Tiled FP8 WMMA GEMM kernels (392 lines)
│       ├── fp8_gemm_e4m3_kernel         # C = A @ B.T (on-the-fly E4M3 quant)
│       ├── fp8_gemm_e4m3_weight_fp8_kernel# C = A @ B_fp8.T (pre-quantized weight)
│       ├── fp8_gemm_backward_input_kernel# grad_input (E5M2/BF8 WMMA)
│       ├── fp8_gemm_backward_weight_kernel# grad_weight (E5M2/BF8 WMMA)
│       └── C-linkage launch wrappers
│
├── tests/                       # Test suites
│   ├── test_pipeline.py         # Full pipeline test (CPU mock, no GPU) (1044 lines)
│   │   ├── 12 test suites covering all API components
│   │   └── Pure-Python FP8 math mock for CPU execution
│   ├── test_compat.py           # CDNA compatibility checker tests (155 lines)
│   └── torch/test_fp8.py        # GPU-requiring pytest tests (439 lines)
│       ├── Requires PyTorch + ROCm + built _C extension
│       └── Covers quant/dequant, Fp8Linear*, Fp8TensorMeta
│
├── build/                       # Build artifacts directory
├── dist/                        # Distribution packages (wheels + tar.gz)
├── pyproject.toml               # Package metadata + build config
├── MANIFEST.in                  # Source distribution includes
└── PYTORCH_EXTENSION_PLAN.md    # Original implementation plan
```

---

## 4. Build System

### 4.1 Offline DLL (`hip_quantize.dll` / `libhip_quantize.so`)

**Windows (PowerShell):**
```powershell
.\build.ps1                                           # All archs (default)
.\build.ps1 -Output hip_quantize_rocm721.dll -RocmBin "C:\venvs\medusa_rocm\Scripts"  # ROCm 7.2.1
.\build.ps1 -Arch "gfx942,gfx1200,gfx1201"            # Custom arch set
.\build.ps1 -CDNA                                      # CDNA + RDNA4
```

**Linux/macOS:**
```bash
./build.sh                                              # All archs (default)
ALL=true ./build.sh                                     # All archs (explicit)
CDNA=true ./build.sh                                    # CDNA + RDNA4
ARCH=gfx942,gfx1201 ./build.sh                          # Custom arch set
```

Build flags used by both scripts:
- `-O3` — maximum optimization
- `-mno-wavefrontsize64` — force Wave32 for gfx12 WMMA compatibility (critical — without this, RDNA4 cards hang)
- `-ffp-contract=off` — prevent FMA contractions that break bit-exact GGML output
- `-DHIP_QUANT_HAS_CDNA=1` — set when any gfx9 target is included

### 4.2 PyTorch Extension (`_C.pyd`)

**Build via setup_torch.py:**
```powershell
& "C:\venvs\medusa_rocm\Scripts\python.exe" setup_torch.py build_ext --inplace
```

**Build into wheel:**
```powershell
$env:HIP_QUANT_BUILD_TORCH_EXT = "1"
& "C:\venvs\medusa_rocm\Scripts\python.exe" -m build --wheel --no-isolation
```

Extension build specifics:
- Compiles `torch_ext/pytorch_bindings.cpp`, `torch_ext/fp8_quant_kernels.hip`, `torch_ext/fp8_linear_kernels.hip`
- Uses `CUDAExtension` (the PyTorch name, even on ROCm/HIP)
- Targets `--offload-arch=gfx1200` and `--offload-arch=gfx1201`
- On Windows, uses `_short_path()` to wrap all include paths, preventing MSVC build failures when the project is in a path with spaces

### 4.3 `src/hip_quant/` Alternative Package Layout

There is a parallel package layout under `src/hip_quant/` with:
- Its own `__init__.py` (simplified, no PyTorch exports)
- Its own `build.ps1` (simpler, uses `$Arch` from env instead of complex params)
- Its own `build.py` (Python wrapper that invokes the PS1 script)
- Its own `hip_quantize.cpp`, `kernels/`, headers, etc.

This layout is used by the `build.py` entry point:
```powershell
python -m hip_quant.build --arch all --rocm-bin "C:\Program Files\AMD\ROCm\7.1\bin"
```

---

## 5. Architecture Support

### 5.1 Feature Table (`cdna_compat.py`)

| Arch Prefix | Family | WMMA | MFMA | FP8 | DP4a | Wave32 | Wave64 |
|---|---|---|---|---|---|---|---|
| gfx90a | CDNA2 | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ |
| gfx940/941/942 | CDNA3 | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ |
| gfx1100-1103 | RDNA3 | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ |
| gfx1150 | RDNA3.5 | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ |
| gfx1200/1201 | RDNA4 | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ |

### 5.2 WMMA Safety Policy

The gfx12 FP8/BF8 WMMA kernels (`__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12` and `__builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12`) are **disabled by default** because ROCm 7.1 can hang or corrupt GPU memory when using them.

**Enable:**
```powershell
$env:HIP_QUANT_ENABLE_GFX12_WMMA = "1"
```

**Force-disable:**
```powershell
$env:HIP_QUANT_DISABLE_WMMA = "1"
```

Runtime guards (checked in both C++ and Python):
1. Arch must be gfx12 (RDNA4)
2. HIP runtime must be ≥ 7.2
3. `HIP_QUANT_ENABLE_GFX12_WMMA` must be explicitly set to `1`

---

## 6. HIP Kernel Details

### 6.1 Offline Quantization Kernels (DLL)

Located in `kernels/` and `#include`-d into `hip_quantize.cpp`.

**Architecture:** One kernel per quant type. Each block uses a warp (32 threads) for the standard types, or 256 threads for K-Quants/I-Quants. The dispatch function `dispatch_quantize_kernel()` selects the kernel via a `switch` statement.

**Quantization flow:**
1. `quantize_tensor()` / `quantize_tensor_fp8_input()` called from Python via ctypes
2. HIP device selected (auto-picks GPU with most CUs)
3. I-Quant lookup tables uploaded to GPU constants (IQ tables for grid search)
4. Input data uploaded to GPU via `hipMemcpy` (host → device)
5. Kernel dispatched: `hipLaunchKernelGGL(kernel, gridDim, blockDim, 0, 0, ...)`
6. Result copied back via `hipMemcpy` (device → host)
7. Thread-local GPU buffers are cached for reuse, freed via `quantize_reset()`

### 6.2 WMMA Micro-GEMM Kernel (`fp8_gemm_test.cu`)

```
One wave (32 threads) per 16×16 output tile
Thread layout:
  lane_wrapped = tid % 16   → column in C/B, row in A
  lane_group   = tid / 16   → high/low half of K

Each thread holds 8 float32 accumulator values (v8f).
K dimension is tiled in chunks of 16.
```

Data packing:
- A: `[16×16 FP8]` packed into 2× int32 per thread (8 bytes)
- B: `[16×16 FP8]` packed into 2× int32 per thread (8 bytes)
- WMMA intrinsic: `__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a_vec, b_vec, acc)`

### 6.3 PyTorch Extension Kernels (`fp8_quant_kernels.hip`, `fp8_linear_kernels.hip`)

**Element-wise quant/dequant:**
- 256 threads per block, `(numel + 255) / 256` blocks
- Each thread processes one element
- Supports float32, float16, and bfloat16 input via `load_float_input()`

**FP8 Linear GEMM (forward):**
- Same WMMA tiling as the micro-benchmark
- On-the-fly E4M3 quantization in registers: value loaded → multiplied by scale → `fp32_to_fp8_e4m3()` → packed
- Three variants:
  - `fp8_gemm_e4m3_kernel` — both A and B quantized from float32 master
  - `fp8_gemm_e4m3_weight_fp8_kernel` — B already in FP8 E4M3 (pre-quantized weight shadow)
  - Scaled variant: `out_scale = 1.0 / (input_scale * weight_scale)`

**FP8 Linear GEMM (backward):**
- Same WMMA tiling but uses BF8 variant: `__builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12`
- Gradients quantized to E5M2 (`fp32_to_fp8_e5m2()`) before WMMA
- `fp8_gemm_backward_input_kernel`: `grad_input[m,k] = sum_n E5M2(grad_output[m,n]) * E5M2(weight[n,k])`
- `fp8_gemm_backward_weight_kernel`: `grad_weight[n,k] = sum_m E5M2(grad_output[m,n]) * E5M2(input[m,k])`

---

## 7. PyTorch Training API Architecture

### 7.1 Activation Compression

```
Forward:  float32 → E4M3 quantization → saved in ctx (uint8, 4× smaller)
Backward: saved uint8 → E4M3 dequantization → used for grad_weight
          (grad_input uses full-precision master weight for accurate gradients)
```

### 7.2 FP8 Training Conventions

| Direction | Activation | Weight | Grad Output | Grad Input | Grad Weight |
|---|---|---|---|---|---|
| Forward | E4M3 | E4M3 | — | — | — |
| Backward | — | E4M3 | E5M2/BF8 | E5M2/BF8 | E5M2/BF8 |

### 7.3 VRAM Savings Layers

| Technique | What Gets Saved | Savings vs FP32 |
|---|---|---|
| Activation compression | Autograd graph activations | 4× |
| Fp8ShadowLinear weight storage | Weight memory (1 byte/param) | 4× |
| Adafactor optimizer | Second-moment state | ~1000× for 2D params |
| BF16/FP16 master weights | Parameter/gradient memory | 2× |

**Combined for a 500M-param LLM:**
```
Before: ~7.6 GB (Weights 2GB, Activations 1.6GB, AdamW 4GB)
After:  ~0.9 GB (Weights 0.5GB, Activations 0.4GB, Adafactor 4MB)
```

### 7.4 Fp8TensorMeta — Delayed Scaling

Tracks a rolling `amax_history` ring buffer. Scale is derived from the maximum observed amax across the window:
```
amax = tensor.abs().max()
amax_history[ptr % history_len] = amax
scale = 448.0 / max(amax_history)      # for E4M3
inv_scale = 1.0 / scale
```

### 7.5 Adafactor Optimizer

Memory-efficient alternative to AdamW:
```
AdamW:  stores m ∈ R^N×K, v ∈ R^N×K           → 2 floats/param
Adafactor: stores R ∈ R^N, C ∈ R^K, no m       → (N+K)/(N·K) floats/param
```

The factored second-moment estimate:
```
V̂[i,j] = R[i] · C[j] / mean(R)
R_t = β₂ₜ R_{t-1} + (1-β₂ₜ) · meanⱼ(g² + ε₁)
C_t = β₂ₜ C_{t-1} + (1-β₂ₜ) · meanᵢ(g² + ε₁)
```

---

## 8. Safety Guards

All layers of the codebase include runtime safety checks:

**`pytorch_bindings.cpp`:**
- `TORCH_CHECK(input.is_cuda())` — rejects CPU tensors
- `TORCH_CHECK(input.is_contiguous())` — rejects non-contiguous layouts
- `float_dtype_code()` — validates float32/float16/bfloat16 only, rejects float64
- `checked_int()` — traps int64→int narrowing before it silently overflows
- `check_grid_dim()` — validates gridDim ≤ 65535 per HIP hardware limit
- `check_gfx12_fp8_wmma_runtime()` — multi-stage WMMA safety gate (arch, runtime version, env vars)

**`hip_quantize.cpp`:**
- `ensure_initialized()` — auto-selects GPU with most CUs
- Buffer size tracking — reallocates GPU buffers only when needed
- Every `hipMalloc`, `hipMemcpy`, `hipMemset`, `hipGetLastError`, `hipDeviceSynchronize` checked for errors
- `quantize_reset()` — thread-safe buffer cleanup

**`fp8_quant_kernels.hip`:**
- `safe_blocks()` — asserts numel > 0, verifies block count fits in uint32

**Python layer:**
- DLL resolution searches multiple locations; fails gracefully with clear error
- WMMA operations gated behind `HIP_QUANT_ENABLE_GFX12_WMMA` env var
- `quantize_numpy()` validates block size alignment, dtype, contiguity

---

## 9. Windows-Specific Fixes Applied

Three critical Windows/RDNA4 fixes have been applied to this repository:

### 9.1 RDNA4 GPU Hang — `-mno-wavefrontsize64`

Added to both `build.ps1` and `setup_torch.py` nvcc args. Without this flag, LLVM/Clang defaults to Wave64 for AMDGPU code generation, but RDNA4 (gfx12) hardware only supports Wave32. Running Wave64 code on RDNA4 physically hangs the GPU.

### 9.2 MSVC Include Order — `pytorch_bindings.cpp`

Moved `#include <hip/hip_runtime.h>` to the very top of the file, before `#include <torch/all.h>`. MSVC's `cl.exe` fails with `HIP_vector_base` errors when PyTorch headers pull in `amd_hip_vector_types.h` before the core HIP runtime structures are defined.

### 9.3 Windows Path With Spaces — `_short_path()` Wrapping

Added `_short_path()` wrapping around all PyTorch include paths in `setup_torch.py`. When a user clones the project into a directory path containing spaces (e.g., `C:\AI Pipeline\`), MSVC receives unquoted include paths and fails with `Cannot open include file: 'torch/all.h'`. The 8.3 short-path conversion (e.g., `C:\AIPIPE~1\`) bypasses this.

---

## 10. Testing

### 10.1 CPU Pipeline Tests (no GPU required)
```powershell
python tests/test_pipeline.py      # 12 test suites, all mocked
python tests/test_compat.py -v     # Compatibility + CPU reference tests
```

### 10.2 GPU Tests (requires built _C extension)
```powershell
pytest tests/torch/test_fp8.py -v  # Real GPU, full pipeline
```

### 10.3 FP8 Crash Diagnostic
```powershell
python diagnose_fp8_crash.py       # 8-stage progressive diagnosis
```

### 10.4 WMMA Stress Test (gfx12 only)
```powershell
$env:HIP_QUANT_ENABLE_GFX12_WMMA = "1"
python test_fp8_gemm.py
```

---

## 11. Common Issues

### DLL not found
```
Set HIP_QUANT_ROCM_BIN or HIP_QUANT_ROCM_HOME to your ROCm bin directory.
Resolution order: env vars → venv _rocm_sdk_core\bin → venv torch\lib → system ROCm 7.1
```

### GPU hang with WMMA kernels
```
Root cause: ROCm 7.1 + gfx12 WMMA bug
Fix: Set HIP_QUANT_ENABLE_GFX12_WMMA only on ROCm 7.2+ systems
Check: python diagnose_fp8_crash.py (Stage 5 isolates WMMA specifically)
```

### MSVC compilation errors
```
"Cannot open include file: 'torch/all.h'"
Fix: Clone to a path without spaces, or the _short_path() fix handles it.
```

### "CUDA error: device-side assert triggered"
```
Usually from the WMMA backward kernels — E5M2/BF8 path not fully validated.
Set HIP_QUANT_DISABLE_WMMA=1 and use only the quantize/dequantize functions.
```
