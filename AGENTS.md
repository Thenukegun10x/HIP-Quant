# hip-quant

## Overview
`hip-quant` is a standalone repository and python module for HIP/ROCm-based tensor quantization. It is specifically built for AMD GPUs (targeting ROCm 7.1 and the `gfx1201` architecture) and is designed to take standard `float32` tensors and quantize them directly on-device using highly optimized HIP C++ kernels.

It implements a wide variety of GGML-compatible quantization formats, making it extremely useful for large language model inference acceleration on AMD hardware. 

Supported quantization types include:
- Legacy/Standard: `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`
- K-Quants: `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`
- I-Quants: `IQ1_S`, `IQ2_XXS`, `IQ2_XS`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL`, `IQ4_XS`

## Project Structure
- `__init__.py`: The Python wrapper. It uses `ctypes` to map the `hip_quantize.dll` native functions directly to python.
- `build.ps1`: PowerShell build script that invokes `hipcc` to compile the native code.
- `hip_quantize.cpp` & `hip_quant_types.h`: C++ source and header files defining the quantization kernels and block structures.
- `kernels/`: Sub-directory containing individual HIP kernels.
- `hip_quantize.dll`: The compiled Windows DLL used at runtime.

## Build Instructions
To compile the C++ source into a DLL, use the PowerShell script:
```powershell
.\build.ps1
```
*Note: This script strictly requires `hipcc` located at `C:\Program Files\AMD\ROCm\7.1\bin\hipcc.exe`.*

## Packaging & Publishing to PyPI
We use `pyproject.toml` and `setuptools` to bundle the Python wrapper together with the compiled `hip_quantize.dll`. 
1. Build the package (`.whl` and `.tar.gz`): `python -m build`
2. Upload to PyPI: `twine upload dist/*`

## Agent Conventions
- **Performance First**: Keep C++ kernels optimized for HIP and `gfx1201`. Memory throughput is key.
- **Python-Native Interop**: When modifying Python code, ensure `ctypes` signatures perfectly match the types exposed by `hip_quantize.cpp` to prevent segfaults.
- **DLL Resolution**: The path to `hip_quantize.dll` is dynamically resolved in `__init__.py` to support `pip install` workflows. Do not hardcode absolute paths in the python wrapper.
- **Packaging**: Any new header files or kernels must be included in `MANIFEST.in` and the `package-data` section of `pyproject.toml`.
