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

## WaveAttention backward (`torch_ext/wave_attn_backward.hip`)
Native GFX12 FP8 WMMA backward for `wave_attn`, computing `dQ`/`dK`/`dV` on-device
without SDPA recomputation. Wired into `_WaveAttentionFn` in `torch_api.py`; set
`HIP_QUANT_WAVE_ATTN_SDPA_BACKWARD=1` to fall back to SDPA for debugging.

Follows FlashAttention-2, including its separate preprocess pass for
`D_i = sum_d dO_id * O_id`. `D` is a full-row reduction, so it **cannot** be
accumulated inside the K-tile loop — the preprocess kernel materialises it up
front. This is why `wave_attn_backward` takes `O` as an argument.

Invariants that are easy to break silently:
- The dK/dV block must have exactly one wave per 16-key sub-tile
  (`THREADS/32 == K_TILE/16`); fewer waves leaves the uncovered keys at zero.
  A `static_assert` guards this — do not derive its thread count from `Q_TILE`.
- Per-lane validity predicates (`q_valid`, `k_valid`, forward `valid_wave`) must
  never gate control flow. They are divergent *within* a wave whenever the
  sequence length is not a multiple of 16, and both kernels rely on
  `__syncthreads()` and on `__shfl_sync(..., lg*8 + i)` broadcasts that read
  lanes 0..15. Use them as load/store predicates only.
- LDS tile pitches are padded by `WAVE_ATTN_LDS_PAD`. Unpadded pitches
  (`Dim`, `K_TILE`, `16`) alias onto one or two LDS banks and cost a 16-way
  conflict on every WMMA operand fetch.

Accuracy is ~0.999 cosine vs an FP32 SDPA reference — the FP8 E4M3 floor —
and is invariant to sequence length, tile alignment, quantization scale, and
gradient magnitude. `test_backward_correctness.py` covers the aligned unit-scale
case; `test_backward_math.py` covers ragged lengths, non-unit `q/k/v` scales,
and small-magnitude `dO`.

## Gated RMSNorm shared weight (`torch_ext/ssm_kernels.hip`, Sep 2026)
Qwen3.5 `ssm_norm` is a SHARED `[head_dim]` vector, but the gated kernel used
to index `w[h*head_dim+i]` (per-head layout) — heads >= 1 read out of bounds.
Prefill was immune (torch fallback broadcasts correctly); decode collapsed to
garbage/NaN. Fixed on both sides: the kernel takes `w_stride` (0 = shared,
dispatched by `w.numel()` in the binding with hard `TORCH_CHECK`s) and
`hip_inference/core/norm.py` tiles once (cached) as defense-in-depth.
Never pass a short `w` to a kernel that indexes per-head without a stride.

## imatrix convention (quantize path)
llama.cpp imatrix is ONE float per input COLUMN shared across rows
(`quantize_iq2_xs` reuses the same pointer every row; size `ne[0]`).
hip_quant kernels index `imatrix + row*n_per_row` (full-matrix layout).
`quantize_numpy` therefore requires `imatrix.shape == arr.shape` — a raw
llama `.dat` vector must be tiled first. `quantize_from_fp8` historically
lacked that shape check (host/device OOB read); see `IMATRIX_PLAN.md` for
the full fix sequence (shape checks → per-column accept → tests).

## hip_inference (`hip_inference/`)
Nested PyTorch inference engine (own repo + `AGENTS.md`). Notable: `qwen35.py`
pre-casts all dense F32/BF16 weights to fp16 once at load (per-token `.to()`
cost ~12ms/layer before); `debug_decode.py` (`python -m
hip_inference.debug_decode`) is the permanent per-step latency + NaN/inf
health profiler — use it before theorizing about tok/s.

## Agent Conventions
- **Performance First**: Keep C++ kernels optimized for HIP and `gfx1201`. Memory throughput is key.
- **Python-Native Interop**: When modifying Python code, ensure `ctypes` signatures perfectly match the types exposed by `hip_quantize.cpp` to prevent segfaults.
- **DLL Resolution**: The path to `hip_quantize.dll` is dynamically resolved in `__init__.py` to support `pip install` workflows. Do not hardcode absolute paths in the python wrapper.
- **Packaging**: Any new header files or kernels must be included in `MANIFEST.in` and the `package-data` section of `pyproject.toml`.

## Testing Notes
- **CPU pipeline suite** (`tests/test_pipeline.py`): mocked `_C`, no GPU required.
  ```powershell
  & 'C:\venvs\medusa_rocm\Scripts\python.exe' -c "import unittest, tests.test_pipeline as t; unittest.main(module=t, exit=True)"
  # or
  & 'C:\venvs\medusa_rocm\Scripts\python.exe' -m pytest tests\test_pipeline.py -q
  ```
- **Why pytest looked hung**: `Fp8TensorMeta` used to default to CUDA when available even for CPU layers. That initialized ROCm during “CPU” tests; on Windows the process often stalls in GPU teardown after tests already passed (no failure output). Meta now defaults to CPU / parameter device.
- **If a GPU run still stalls on exit**: force CPU visibility for pipeline tests:
  ```powershell
  $env:CUDA_VISIBLE_DEVICES=''; $env:HIP_VISIBLE_DEVICES=''
  ```
- **GPU / WMMA tests**: require the real extension and an x64 VS toolchain build. Enable only when intentional:
  ```powershell
  $env:HIP_QUANT_ENABLE_GFX12_WMMA='1'
  ```
- **Venv**: `C:\venvs\medusa_rocm\Scripts\python.exe`
- **x64 extension build** (not x86 Developer PowerShell):
  ```bat
  "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
  "C:\venvs\medusa_rocm\Scripts\python.exe" setup_torch.py build_ext --inplace
  ```
  Expect `temp.win-amd64-cpython-312` and `Hostx64\x64\link.exe`.

## Release & toolchain matrix
Two PyTorch ABIs ship from one source tree. `setup_torch.py:58` auto-selects the
ROCm toolchain from `torch.version.hip`: `7.14.*` (TheRock) → `C:/TheRock/build`,
otherwise `C:/Program Files/AMD/ROCm/<major.minor>`. **Never build the 2.9.x
artifact against TheRock (or vice versa)** — the extension links the HIP ABI and
will fail to load with an ABI mismatch. The two venvs are independent; the build
only mutates `PATH`/env for its own process, so neither breaks the other.

| | torch 2.9.1 build (main) | torch 2.15 build (post215) |
|---|---|---|
| Venv | `C:\venvs\medusa_rocm\Scripts\python.exe` | `C:\Users\armor\Desktop\AI pipeline\.venv\Scripts\python.exe` |
| torch | `2.9.1+rocm7.2.1` | `2.15.0a0+gitf07882e` |
| HIP | `7.2.53211-158bd99533` | `7.14.60850` |
| ROCm toolchain | `C:\Program Files\AMD\ROCm\7.2` (venv also has pip `rocm-sdk 7.2.1`) | `C:\TheRock\build` (TheRock 7.14.0; tarball `C:\TheRock\therock-dist-windows-gfx120X-all-7.14.0.tar.gz`) |
| Version | `X.Y.Z` | `X.Y.Z.post215` |
| Git tag | `vX.Y.Z` | `vX.Y.Z.post215` |
| Local backup | `_C.cp312-win_amd64.torch29.pyd` | `_C.cp312-win_amd64.torch215.pyd` |

Only `_C.cp312-win_amd64.pyd` is tracked, and it is what `_load_extension()` imports
(`torch_api.py:431`). The `.torch29.pyd` / `.torch215.pyd` files are gitignored
local backups swapped in/out by hand. **Each release tag is a complete,
self-consistent snapshot** — a torch 2.15 user checks out the `.post215` tag
(e.g. `v2.1.0` = 2.9.1, `v2.1.0.post215` = 2.15).

`HIP_QUANT_ARCH` overrides the default 8-arch list
(`gfx90a,gfx942,gfx1100,gfx1101,gfx1102,gfx1103,gfx1200,gfx1201`); set it (e.g.
`gfx1201`) for a fast single-arch dev build. Release builds use the default
(multi-arch fatbin).

**Device enumeration differs between the two runtimes.** The torch 2.9.1 venv
sees only the RX 9070 XT (device 0). Under the torch 2.15 / TheRock runtime the
**integrated GPU is device 0 and the 9070 XT is device 1**, so the first kernel
launch on device 0 fails with `hipErrorInvalidImage` ("device kernel image is
invalid") — there is no code image for the iGPU. Select the dGPU explicitly:
```powershell
$env:HIP_VISIBLE_DEVICES='1'   # TheRock/torch 2.15 runtime
```

`setup_torch.py` honours `HIP_QUANT_ROCM_HOME`/`ROCM_HOME`/`ROCM_PATH`/`HIP_PATH`
first, then falls back to the torch-version heuristic. A globally-exported
`HIP_PATH=C:\Program Files\AMD\ROCm\7.2` therefore wins over TheRock — set
`HIP_QUANT_ROCM_HOME=C:\TheRock\build` to force the 7.14 toolchain.

### Release steps
1. Bump `__version__` (`__init__.py`) and `pyproject.toml` `version`; update the
   `## What's New` README section.
2. In the 2.9.1 venv (default multi-arch), build, copy the result over
   `_C.cp312-win_amd64.pyd`, commit, tag `vX.Y.Z`, push.
3. Bump to `X.Y.Z.post215`, build in the 2.15 venv (auto-uses TheRock), copy over
   the tracked `_C`, commit, tag `vX.Y.Z.post215`, push.
