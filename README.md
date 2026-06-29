# HIP-Quant

GPU-accelerated quantization kernels for every GGML quantization type — written in HIP for AMD ROCm GPUs.

Drop-in replacement for `ggml_quantize_chunk` with identical byte-exact output (13 types at 0.00% diff) and a simple Python interface.

## Requirements

- AMD ROCm 7.1+ SDK (`hipcc`, `hipruntime`)
- Python 3.10+
- PyTorch with ROCm support (`pip install torch --index-url https://download.pytorch.org/whl/rocm6.2`)

## Build

```powershell
cd src\hip_quant
.\build.ps1
```

Produces `hip_quantize.dll`.

## Usage

```python
import torch
from hip_quant import HipQuant, GGML_TYPE

# Load your model weights (float32)
weights = torch.randn(4096, 11008, dtype=torch.float32)

# Quantize to Q4_K
hq = HipQuant()
qweight = hq.quantize(weights, GGML_TYPE["Q4_K"])

# qweight is a torch.Tensor of uint8 (the packed quantized representation)
print(qweight.shape)  # e.g. torch.Size([4096, 11008])
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
| IQ3_XXS | 18 | 256 | 98 | ⚠️ In progress |
| IQ4_NL | 20 | 32 | 18 | ✅ Byte-exact |
| IQ4_XS | 23 | 256 | 136 | ✅ Byte-exact |

## How it works

`HipQuant` loads `hip_quantize.dll` which contains HIP kernels that run on the GPU. Each kernel implements the exact same algorithm as the corresponding `ggml_quantize_chunk` reference — using `-ffp-contract=off` to disable FMA contraction and ensure bit-exact float evaluation.

## Project structure

```
src/hip_quant/
├── __init__.py              # Python bindings (HipQuant class)
├── build.ps1                # Compilation script (hipcc)
├── hip_quant_types.h        # Block struct definitions
├── hip_quant_util.h         # FP16 conversion helpers
├── hip_iquant_util.h        # IQ codebook tables (GPU-side)
├── hip_quant_iq3xxs_data.h  # IQ3_XXS grid/map data
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
    ├── quant_iq3_xxs.cu     # IQ3_XXS kernel
    ├── quant_iq4_nl.cu      # IQ4_NL kernel
    └── quant_iq4_xs.cu      # IQ4_XS kernel
tests/
├── test_all.py              # Unified diff test for all types
├── test_q4_k.py             # Q4_K specific test
└── test_q5_k.py             # Q5_K specific test
```

## Reference

- [Medusa: LLM Inference Acceleration via Multi-Token Prediction](https://arxiv.org/abs/2401.10774)
- [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) — CPU reference implementations
