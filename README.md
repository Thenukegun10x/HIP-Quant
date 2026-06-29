# HIP-Quant

GPU-accelerated quantization kernels for GGML quantization types, written in HIP for AMD ROCm GPUs.

Drop-in replacement for `ggml_quantize_chunk` with byte-exact output and a simple Python interface.

## Requirements

- AMD ROCm 7.1+ SDK (`hipcc`, `hipruntime`)
- Python 3.10+
- NumPy

PyTorch is not required. If your weights are in a PyTorch tensor, convert them to a CPU `float32` NumPy array before calling HIP-Quant.

## Build

```powershell
python -m hip_quant.build
```

Produces `hip_quantize.dll`.

You can override the ROCm path or GPU arch:

```powershell
python -m hip_quant.build --rocm-bin "C:\Program Files\AMD\ROCm\7.1\bin" --arch gfx1201
```

## Usage

```python
import numpy as np
from hip_quant import HipQuant

# Load or create float32 weights as a NumPy array.
# Shape is (nrows, n_per_row), and n_per_row must be a multiple of the quant block size.
weights = np.random.randn(4096, 11008).astype(np.float32)

hq = HipQuant()
print(hq.device_name)

# Type names and numeric IDs both work.
qweight = hq.quantize_numpy(weights, "Q4_K")

# qweight is a flat np.uint8 array containing GGML-packed quantized bytes.
print(qweight.dtype, qweight.shape)

# Row-shaped output is often easier to write into GGUF tensor buffers.
qrows = hq.quantize_rows(weights, "Q4_K")
print(qrows.shape)  # (nrows, packed_row_bytes)

# Write raw packed bytes directly.
hq.quantize_to_file(weights, "Q4_K", "weight.Q4_K.bin")
```

### Imatrix Types

`IQ2_XXS`, `IQ2_XS`, and `IQ1_S` require an importance matrix for GGML-compatible quantization:

```python
weights = np.random.randn(256, 4096).astype(np.float32)
imatrix = np.ones_like(weights, dtype=np.float32)

hq = HipQuant()
qweight = hq.quantize_numpy(weights, "IQ2_XS", imatrix=imatrix)
```

If you intentionally want to run without an imatrix, pass `require_imatrix=False`.

### CLI

```powershell
python -m hip_quant --type Q4_K weights.npy weights.Q4_K.bin
```

With imatrix:

```powershell
python -m hip_quant --type IQ2_XS --imatrix imatrix.npy weights.npy weights.IQ2_XS.bin
```

Installed console scripts are also provided:

```powershell
hip-quant --list-types
hip-quant --info
hip-quant -t Q5_K weights.npy weights.Q5_K.bin
hip-quant-build --arch gfx1201
```

### Optional PyTorch Input

```python
import torch
from hip_quant import HipQuant

tensor = torch.randn(4096, 11008, dtype=torch.float32, device="cuda")
weights = tensor.detach().cpu().numpy().astype("float32", copy=False)

hq = HipQuant()
qweight = hq.quantize_numpy(weights, "Q4_K")
```

### Supported types

| Type | ID | Block size | Block bytes | Status |
|------|----|-----------|-------------|--------|
| Q4_0 | 2  | 32 | 18 | ✅ Byte-exact |
| Q4_1 | 3  | 32 | 20 | ✅ Byte-exact |
| Q5_0 | 6  | 32 | 22 | ✅ Byte-exact |
| Q5_1 | 7  | 32 | 24 | ✅ Byte-exact |
| Q8_0 | 8  | 32 | 34 | ✅ Byte-exact |
| Q8_1 | 9  | 32 | 36 | ✅ HIP-only |
| Q2_K | 10 | 256 | 84 | ✅ Byte-exact |
| Q3_K | 11 | 256 | 110 | ✅ Byte-exact |
| Q4_K | 12 | 256 | 144 | ✅ Byte-exact |
| Q5_K | 13 | 256 | 176 | ✅ Byte-exact |
| Q6_K | 14 | 256 | 210 | ✅ Byte-exact |
| IQ2_XXS | 16 | 256 | 66 | ✅ Byte-exact |
| IQ2_XS | 17 | 256 | 74 | ✅ Byte-exact |
| IQ3_XXS | 18 | 256 | 98 | ✅ Byte-exact |
| IQ1_S | 19 | 256 | 50 | ✅ Byte-exact |
| IQ4_NL | 20 | 32 | 18 | ✅ Byte-exact |
| IQ3_S | 21 | 256 | 110 | ✅ Byte-exact |
| IQ4_XS | 23 | 256 | 136 | ✅ Byte-exact |

## How it works

`HipQuant` loads `hip_quantize.dll` with `ctypes`. The DLL copies a contiguous `float32` NumPy array to the selected HIP device, runs the requested quantization kernel, and returns the packed GGML bytes as `np.uint8`.

The kernels mirror `ggml_quantize_chunk` behavior. The build uses `-ffp-contract=off` to disable FMA contraction where needed, which keeps floating-point evaluation byte-exact with the CPU reference.

## Project structure

```
src/hip_quant/
├── __init__.py              # Python bindings (HipQuant class)
├── __main__.py              # CLI entry point
├── build.py                 # python -m hip_quant.build helper
├── build.ps1                # Compilation script (hipcc)
├── hip_quant_types.h        # Block struct definitions
├── hip_quant_util.h         # FP16 conversion helpers
├── hip_iquant_util.h        # IQ codebook tables (GPU-side)
├── hip_quant_iq1s_data.h    # IQ1_S grid/map/neighbour data
├── hip_quant_iq2xxs_data.h  # IQ2_XXS grid/map/neighbour data
├── hip_quant_iq2xs_data.h   # IQ2_XS grid/map/neighbour data
├── hip_quant_iq3xxs_data.h  # IQ3_XXS grid/map data
├── hip_quant_iq3s_data.h    # IQ3_S grid/map/neighbour data
├── hip_quantize.cpp         # Host harness (DLL entry points, dispatch)
└── kernels/
    ├── quant_q4_0.cu        # Q4_0 kernel
    ├── quant_q4_1.cu        # Q4_1 kernel
    ├── quant_q5_0.cu        # Q5_0 kernel
    ├── quant_q5_1.cu        # Q5_1 kernel
    ├── quant_q8_0.cu        # Q8_0 kernel
    ├── quant_q8_1.cu        # Q8_1 kernel (HIP-only)
    ├── quant_q2_K.cu        # Q2_K kernel
    ├── quant_q3_K.cu        # Q3_K kernel
    ├── quant_q4_K.cu        # Q4_K kernel
    ├── quant_q5_K.cu        # Q5_K kernel
    ├── quant_q6_K.cu        # Q6_K kernel
    ├── quant_iq1_s.cu       # IQ1_S kernel
    ├── quant_iq2_xxs.cu     # IQ2_XXS kernel
    ├── quant_iq2_xs.cu      # IQ2_XS kernel
    ├── quant_iq3_xxs.cu     # IQ3_XXS kernel
    ├── quant_iq3_s.cu       # IQ3_S kernel
    ├── quant_iq4_nl.cu      # IQ4_NL kernel
    └── quant_iq4_xs.cu      # IQ4_XS kernel
tests/
└── test_all.py              # Unified diff test for all supported types
```

## Reference

- [Medusa: LLM Inference Acceleration via Multi-Token Prediction](https://arxiv.org/abs/2401.10774)
- [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) — CPU reference implementations
