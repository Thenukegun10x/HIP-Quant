r"""Microbenchmarks for hip_quant FP8 torch paths.

Run with:
    C:\venvs\medusa_rocm\Scripts\python.exe tests\torch\bench_fp8.py

Set HIP_QUANT_ENABLE_GFX12_WMMA=1 to include WMMA linear benchmarks.
"""

from __future__ import annotations

import os
import time

import torch

from hip_quant.torch_api import (
    Fp8ShadowLinear,
    dequantize_e4m3,
    dequantize_e5m2,
    quantize_e4m3,
    quantize_e5m2,
)


def _time_cuda(fn, warmup: int = 20, iters: int = 100) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA/HIP device is required")

    device = "cuda"
    torch.manual_seed(1234)

    x = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
    x_fp8_e4m3 = quantize_e4m3(x)
    x_fp8_e5m2 = quantize_e5m2(x)

    print("Elementwise FP8 ops, shape=(4096, 4096), dtype=bf16")
    print(f"quantize_e4m3:   {_time_cuda(lambda: quantize_e4m3(x)):.3f} ms")
    print(f"quantize_e5m2:   {_time_cuda(lambda: quantize_e5m2(x)):.3f} ms")
    print(f"dequantize_e4m3: {_time_cuda(lambda: dequantize_e4m3(x_fp8_e4m3)):.3f} ms")
    print(f"dequantize_e5m2: {_time_cuda(lambda: dequantize_e5m2(x_fp8_e5m2)):.3f} ms")

    if os.environ.get("HIP_QUANT_ENABLE_GFX12_WMMA", "").lower() not in {"1", "true", "yes", "on"}:
        print("Skipping WMMA linear benchmark; set HIP_QUANT_ENABLE_GFX12_WMMA=1 to enable.")
        return

    layer = Fp8ShadowLinear(4096, 4096, device=device, dtype=torch.bfloat16)
    inp = torch.randn(32, 4096, device=device, dtype=torch.bfloat16, requires_grad=True)

    def forward_only() -> None:
        layer(inp)

    def forward_backward() -> None:
        out = layer(inp)
        loss = out.float().square().mean()
        loss.backward()
        layer.zero_grad(set_to_none=True)
        inp.grad = None

    print("Fp8ShadowLinear, batch=32, in=4096, out=4096, dtype=bf16")
    print(f"forward:          {_time_cuda(forward_only, warmup=5, iters=20):.3f} ms")
    print(f"forward+backward: {_time_cuda(forward_backward, warmup=3, iters=10):.3f} ms")


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    print(f"total wall time: {time.perf_counter() - start:.2f} s")
