"""Bench MXFP8 true UE8M0 vs FP32-scale blockwise — gfx12 w32, 32-thread warp kernel.

Run after: python setup_torch.py build_ext --inplace
   python bench_mxfp8.py --e4m3 --e5m2 --csv
"""

import argparse
import time

import torch

try:
    import hip_quant.torch_api as hq
    from hip_quant import _C as _ext
except Exception as e:
    raise SystemExit(f"hip_quant._C not built: {e}")

def _has_mx():
    return hasattr(_ext, "quantize_mxfp8_e4m3")

def bench_once(x, fns, warm=5, iters=20):
    for _ in range(warm):
        for fn in fns:
            fn(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        for fn in fns:
            fn(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = (time.perf_counter()-t0)/iters
    # bytes: read x (4B or 2B) + write q(1B)+scales(1/32 B) + dequant write 4B
    return dt

def gb_per_s(x, dt):
    n = x.numel()
    # quant: read 4B + write 1B + 1/32 B ~1.031
    # dequant: read 1B+0.031 + write 4B
    # single quant+dequant roundtrip ~ 10.06B per elem
    bytes_per = 10.0625 if x.dtype==torch.float32 else 7.06  # f16/bf16 read 2B
    return (n * bytes_per) / (dt * 1e9)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e4m3", action="store_true", help="bench E4M3")
    ap.add_argument("--e5m2", action="store_true", help="bench E5M2")
    ap.add_argument("--shapes", default="4096x4096,8192x8192,16384x4096", help="MxK shapes")
    ap.add_argument("--dtype", default="float32", choices=["float32","float16","bfloat16"])
    ap.add_argument("--csv", action="store_true")
    args = ap.parse_args()
    if not _has_mx():
        raise SystemExit("rebuild with mxfp8_kernels.hip: python setup_torch.py build_ext --inplace")
    if not args.e4m3 and not args.e5m2:
        args.e4m3=True
    dtype = {"float32":torch.float32,"float16":torch.float16,"bfloat16":torch.bfloat16}[args.dtype]
    shapes=[]
    for s in args.shapes.split(","):
        a,b=map(int,s.lower().split("x"))
        shapes.append((a,b))
    header="shape,dtype,format,ms,GB/s,err_max"
    if args.csv:
        print(header)
    else:
        print(header)
        print("-"*len(header))
    for M,K in shapes:
        x = torch.randn(M,K, device="cuda", dtype=dtype)
        for fmt, q_fn, dq_fn in [
            ("E4M3", hq.quantize_mxfp8_e4m3, hq.dequantize_mxfp8_e4m3),
            ("E5M2", hq.quantize_mxfp8_e5m2, hq.dequantize_mxfp8_e5m2),
        ]:
            if (fmt=="E4M3" and not args.e4m3) or (fmt=="E5M2" and not args.e5m2):
                continue
            def fns(_x=x, _q=q_fn, _dq=dq_fn):
                q,s=_q(_x); return _dq(q,s)
            dt = bench_once(x, [lambda _x=x, _q=q_fn, _dq=dq_fn: _dq(*_q(_x))])
            q,s=q_fn(x); y=dq_fn(q,s)
            err=(y.float()-x.float()).abs().max().item()
            gb = gb_per_s(x, dt)
            row=f"{M}x{K},{args.dtype},{fmt},{dt*1e3:.3f},{gb:.1f},{err:.4f}"
            print(row)
            # also show FP32-scale baseline for comparison
            if fmt=="E4M3":
                def base(_x=x):
                    qq,ss=hq.quantize_e4m3_blockwise(_x,32); return hq.dequantize_e4m3_blockwise(qq,ss,32)
                dt2=bench_once(x, [lambda: base(x)])
                gb2=gb_per_s(x, dt2)
                print(f"  baseline FP32-scale E4M3 block32 {dt2*1e3:.3f}ms {gb2:.1f}GB/s (MX {gb/gb2:.2f}x)")

if __name__=="__main__":
    main()
