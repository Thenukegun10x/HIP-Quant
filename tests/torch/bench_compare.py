r"""hipBLASLt vs custom WMMA FP8 kernel comparison benchmark.

Run with:
    $env:HIP_QUANT_ENABLE_GFX12_WMMA = "1"
    & "C:\venvs\medusa_rocm\Scripts\python.exe" tests\torch\bench_compare.py

Flags:
    HIP_QUANT_ENABLE_GFX12_WMMA=1  — enables custom WMMA to be tested
    HIP_QUANT_BENCH_NO_FORCE_EXIT=1 — keep process alive after finish
"""

from __future__ import annotations

import os
import sys
import time

import torch

_REPO_PARENT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
sys.path.insert(0, os.path.abspath(_REPO_PARENT))

from hip_quant.torch_api import (
    quantize_e4m3,
    quantize_e5m2,
    dequantize_e4m3,
    fp8_linear_forward_fp8_input,
    fp8_linear_forward_fp8_input_weight,
    fp8_linear_backward_input_fp8_grad,
    fp8_linear_backward_weight_fp8_grad,
)


SHAPES = [
    (1, 4096, 4096),
    (4, 4096, 4096),
    (16, 4096, 4096),
    (32, 4096, 4096),
    (64, 4096, 4096),
    (128, 4096, 4096),
    (32, 2048, 2048),
    (32, 8192, 8192),
    (32, 4096, 11008),
    (32, 11008, 4096),
    (1, 14336, 4096),
    (1, 4096, 14336),
]

DTYPES = [torch.float32, torch.float16, torch.bfloat16]

SCALES = [
    (1.0, 1.0),
    (0.5, 0.5),
    (0.125, 0.125),
]


def _time_cuda(fn, warmup=10, iters=50) -> float:
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


# ── Forward helpers ────────────────────────────────────────────────────────

def _scaled_mm_call(inp, w, si, sw, fp8_dtype):
    a = inp.contiguous()
    b = w.contiguous()
    if si != 1.0:
        a = (a * si).contiguous()
    if sw != 1.0:
        b = (b * sw).contiguous()
    sa = torch.full((), 1.0 / float(si), device=inp.device, dtype=torch.float32)
    sb = torch.full((), 1.0 / float(sw), device=inp.device, dtype=torch.float32)
    a_fp8 = a.to(fp8_dtype)
    b_fp8_t = b.to(fp8_dtype).contiguous().t()
    return torch._scaled_mm(a_fp8, b_fp8_t, sa, sb, out_dtype=inp.dtype)


def bench_fwd_raw(inp, w, si, sw, fp8_dtype):
    try:
        _scaled_mm_call(inp, w, si, sw, fp8_dtype)
    except RuntimeError:
        return None
    return _time_cuda(
        lambda: _scaled_mm_call(inp, w, si, sw, fp8_dtype),
        warmup=10, iters=50,
    )


def bench_fwd_our(inp, w, si, sw):
    inp_fp8 = quantize_e4m3(inp)
    return _time_cuda(
        lambda: fp8_linear_forward_fp8_input(inp_fp8, w, inp, si, sw, None),
        warmup=10, iters=50,
    )


def bench_fwd_wmma(inp, w, si, sw):
    inp_scaled = (inp * si).contiguous()
    inp_fp8 = quantize_e4m3(inp_scaled)
    w_scaled = (w * sw).contiguous()
    w_fp8 = quantize_e4m3(w_scaled)
    return _time_cuda(
        lambda: fp8_linear_forward_fp8_input_weight(
            inp_fp8, w_fp8, inp, 1.0 / sw, si, None,
        ),
        warmup=10, iters=50,
    )


# ── Backward helpers ───────────────────────────────────────────────────────

def _scaled_mm_bwd_call(go, w, x, weight_scale, input_scale):
    a_gi = go.contiguous().to(torch.float8_e5m2)  # [M,N] row-major
    b_gi = w.contiguous().to(torch.float8_e5m2).t().contiguous().t()  # [N,K] column-major
    s_go = torch.ones((), device=go.device, dtype=torch.float32)
    s_w  = torch.full((), float(weight_scale), device=go.device, dtype=torch.float32)
    gi = torch._scaled_mm(a_gi, b_gi, s_go, s_w, out_dtype=go.dtype)

    a_gw = go.t().contiguous().to(torch.float8_e5m2)  # [N,M] row-major
    b_gw = x.contiguous().to(torch.float8_e5m2).t().contiguous().t()  # [M,K] column-major
    s_go_t = torch.ones((), device=go.device, dtype=torch.float32)
    s_x    = torch.full((), float(input_scale), device=go.device, dtype=torch.float32)
    gw = torch._scaled_mm(a_gw, b_gw, s_go_t, s_x, out_dtype=go.dtype)

    gi = None
    gw = None
    return gi, gw


def bench_bwd_raw(go, w, x, weight_scale, input_scale):
    try:
        _scaled_mm_bwd_call(go, w, x, weight_scale, input_scale)
    except RuntimeError:
        return None
    return _time_cuda(
        lambda: _scaled_mm_bwd_call(go, w, x, weight_scale, input_scale),
        warmup=10, iters=50,
    )


def bench_bwd_wmma(go, w, x, weight_scale, input_scale):
    go_c = go.contiguous()
    go_fp8 = quantize_e5m2(go_c)
    try:
        fp8_linear_backward_input_fp8_grad(go_fp8, go_c, w, weight_scale)
        fp8_linear_backward_weight_fp8_grad(go_fp8, go_c, x, input_scale)
    except RuntimeError:
        return None
    return _time_cuda(
        lambda: (
            fp8_linear_backward_input_fp8_grad(go_fp8, go_c, w, weight_scale),
            fp8_linear_backward_weight_fp8_grad(go_fp8, go_c, x, input_scale),
            None,
        ),
        warmup=10, iters=50,
    )


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA/HIP device is required")

    device = "cuda"
    props = torch.cuda.get_device_properties(0)
    gcn = getattr(props, "gcnArchName", props.name)
    print(f"Device: {props.name}  ({gcn})")
    print(f"PyTorch: {torch.__version__}  HIP: {torch.version.hip}")
    print()

    has_scaled_mm = (
        hasattr(torch, "_scaled_mm")
        and hasattr(torch, "float8_e4m3fn")
        and hasattr(torch, "float8_e5m2")
    )

    wmma_available = (
        os.environ.get("HIP_QUANT_ENABLE_GFX12_WMMA", "").lower()
        in {"1", "true", "yes", "on"}
    )

    torch.manual_seed(1234)

    hdr = (f"{'M':>6} {'K':>6} {'N':>6} {'dtype':>8} {'s_in':>5} {'s_wt':>5}  "
           f"{'fwd_e4m3':>13} {'fwd_e5m2':>13} {'fwd_our':>13} {'fwd_wmma':>13}  "
           f"{'bwd_raw':>13} {'bwd_wmma':>13}  "
           f"{'note'}")
    sep = "-" * (len(hdr) + 8)
    print(hdr)
    print(sep)

    for m, k, n in SHAPES:
        for dtype in DTYPES:
            inp = torch.randn(m, k, device=device, dtype=dtype)
            w   = torch.randn(n, k, device=device, dtype=dtype)

            for si, sw in SCALES:
                # Forward
                t_fe4 = bench_fwd_raw(inp, w, si, sw, torch.float8_e4m3fn) if has_scaled_mm else None
                t_fe5 = bench_fwd_raw(inp, w, si, sw, torch.float8_e5m2) if has_scaled_mm else None
                t_four = bench_fwd_our(inp, w, si, sw)
                t_fwmma = bench_fwd_wmma(inp, w, si, sw) if wmma_available else None

                # Backward: use a fresh grad_output and decompressed activation
                go = torch.randn(m, n, device=device, dtype=dtype)
                x_deq = dequantize_e4m3(quantize_e4m3(inp))
                if si != 1.0:
                    x_deq = x_deq * (1.0 / si)

                t_br = bench_bwd_raw(go, w, x_deq, sw, 1.0) if has_scaled_mm else None
                t_bw = bench_bwd_wmma(go, w, x_deq, sw, 1.0) if wmma_available else None

                # Format
                s_fe4   = f"{t_fe4:.3f}" if t_fe4 is not None else "N/A"
                s_fe5   = f"{t_fe5:.3f}" if t_fe5 is not None else "N/A"
                s_four  = f"{t_four:.3f}" if t_four is not None else "N/A"
                s_fwmma = f"{t_fwmma:.3f}" if t_fwmma is not None else "N/A"
                s_br    = f"{t_br:.3f}" if t_br is not None else "N/A"
                s_bw    = f"{t_bw:.3f}" if t_bw is not None else "N/A"

                note = ""
                if t_fe4 is not None and t_fwmma is not None:
                    r = t_fwmma / t_fe4
                    if r > 2.0:
                        note = f"fwd_wmma {r:.1f}x slower"
                    elif r < 0.5:
                        note = f"fwd_wmma {1/r:.1f}x faster!"
                    else:
                        note = f"fwd_wmma {r:.2f}x"
                if t_br is not None and t_bw is not None:
                    r = t_bw / t_br
                    note += f"  bwd_wmma {r:.2f}x"

                print(f"{m:>6} {k:>6} {n:>6} {str(dtype):>8} "
                      f"{si:>5.2f} {sw:>5.2f}  "
                      f"{s_fe4:>13} {s_fe5:>13} {s_four:>13} {s_fwmma:>13}  "
                      f"{s_br:>13} {s_bw:>13}  "
                      f"{note}")
                sys.stdout.flush()


if __name__ == "__main__":
    start = time.perf_counter()
    status = 0
    try:
        main()
        print(f"\ntotal wall time: {time.perf_counter() - start:.2f} s")
    except BaseException:
        status = 1
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        if os.environ.get("HIP_QUANT_BENCH_NO_FORCE_EXIT", "").lower() not in {"1", "true", "yes", "on"}:
            os._exit(status)
