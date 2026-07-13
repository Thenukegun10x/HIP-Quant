r"""Microbenchmarks for hip_quant FP8 torch paths.

Run with:
    C:\venvs\medusa_rocm\Scripts\python.exe tests\torch\bench_fp8.py

The custom WMMA measurements run automatically on a compatible ROCm 7.2+
gfx12 device.  Set HIP_QUANT_DISABLE_WMMA=1 to suppress them explicitly.
"""

from __future__ import annotations

import os
import sys
import time

import torch

from hip_quant.torch_api import (
    Fp8ShadowLinear,
    dequantize_e4m3,
    dequantize_e5m2,
    fp8_linear_forward_fp8_input_weight,
    fp8_linear_forward_fp8_input_weight_packed,
    pack_fp8_weight_for_wmma,
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


class _temporary_env:
    def __init__(self, **values: str) -> None:
        self.values = values
        self.previous: dict[str, str | None] = {}

    def __enter__(self) -> None:
        for key, value in self.values.items():
            self.previous[key] = os.environ.get(key)
            os.environ[key] = value

    def __exit__(self, exc_type, exc, tb) -> None:
        for key, value in self.previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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

    print("Fp8ShadowLinear (default hipBLASLt backend), batch=32, in=4096, out=4096, dtype=bf16")
    print(f"forward:          {_time_cuda(forward_only, warmup=5, iters=20):.3f} ms")
    print(f"forward+backward: {_time_cuda(forward_backward, warmup=3, iters=10):.3f} ms")

    try:
        with _temporary_env(HIP_QUANT_FP8_LINEAR_BACKEND="custom"):
            print("Fp8ShadowLinear custom WMMA backend, batch=32, in=4096, out=4096, dtype=bf16")
            print(f"forward:          {_time_cuda(forward_only, warmup=5, iters=20):.3f} ms")
            print(f"forward+backward: {_time_cuda(forward_backward, warmup=3, iters=10):.3f} ms")
    except RuntimeError as exc:
        print(f"Custom WMMA unavailable: {exc}")

    if hasattr(torch, "_scaled_mm") and hasattr(torch, "float8_e4m3fn"):
        weight = layer.weight_master.detach().contiguous()
        inp_detached = inp.detach().contiguous()
        scale_a = torch.ones((), device=device, dtype=torch.float32)
        scale_b = torch.ones((), device=device, dtype=torch.float32)
        inp_f8 = inp_detached.to(torch.float8_e4m3fn)
        weight_f8_t = weight.to(torch.float8_e4m3fn).contiguous().t()

        def hipblaslt_scaled_mm_precast() -> None:
            torch._scaled_mm(inp_f8, weight_f8_t, scale_a, scale_b, out_dtype=torch.bfloat16)

        def custom_wmma_precast() -> None:
            fp8_linear_forward_fp8_input_weight(
                quantize_e4m3(inp_detached),
                quantize_e4m3(weight),
                inp_detached,
                weight_inv_scale=1.0,
                input_scale=1.0,
                bias=None,
            )

        def hipblaslt_scaled_mm_with_cast() -> None:
            torch._scaled_mm(
                inp_detached.to(torch.float8_e4m3fn),
                weight.to(torch.float8_e4m3fn).contiguous().t(),
                scale_a,
                scale_b,
                out_dtype=torch.bfloat16,
            )

        print("PyTorch/hipBLASLt raw _scaled_mm, batch=32, in=4096, out=4096, dtype=bf16")
        try:
            print(f"precast FP8:      {_time_cuda(hipblaslt_scaled_mm_precast, warmup=5, iters=20):.3f} ms")
            print(f"cast+matmul:      {_time_cuda(hipblaslt_scaled_mm_with_cast, warmup=3, iters=10):.3f} ms")
            try:
                input_fp8 = quantize_e4m3(inp_detached)
                weight_fp8 = quantize_e4m3(weight)
                weight_packed = pack_fp8_weight_for_wmma(weight_fp8)

                def custom_wmma_prequantized() -> None:
                    fp8_linear_forward_fp8_input_weight(
                        input_fp8,
                        weight_fp8,
                        inp_detached,
                        weight_inv_scale=1.0,
                        input_scale=1.0,
                        bias=None,
                    )

                def custom_wmma_prepacked() -> None:
                    fp8_linear_forward_fp8_input_weight_packed(
                        input_fp8,
                        weight_packed,
                        inp_detached,
                        output_features=weight.size(0),
                        weight_inv_scale=1.0,
                        input_scale=1.0,
                        bias=None,
                    )

                def custom_pack_weight() -> None:
                    pack_fp8_weight_for_wmma(weight_fp8)

                print(f"custom WMMA precast: {_time_cuda(custom_wmma_prequantized, warmup=5, iters=20):.3f} ms")
                print(f"custom WMMA prepacked: {_time_cuda(custom_wmma_prepacked, warmup=5, iters=20):.3f} ms")
                print(f"custom pack weight: {_time_cuda(custom_pack_weight, warmup=3, iters=10):.3f} ms")
                print(f"custom quant+WMMA:  {_time_cuda(custom_wmma_precast, warmup=3, iters=10):.3f} ms")
            except RuntimeError as exc:
                print(f"custom WMMA: {exc}")
        except RuntimeError as exc:
            print(f"hipBLASLt raw: {exc}")
    else:
        print("PyTorch/hipBLASLt raw _scaled_mm: not available")


if __name__ == "__main__":
    start = time.perf_counter()
    status = 0
    try:
        main()
        print(f"total wall time: {time.perf_counter() - start:.2f} s")
    except BaseException:
        status = 1
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        if os.environ.get("HIP_QUANT_BENCH_NO_FORCE_EXIT", "").lower() not in {"1", "true", "yes", "on"}:
            os._exit(status)
