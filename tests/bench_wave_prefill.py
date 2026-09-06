"""Warm prefill timings with explicit local imports and quantization accuracy.

Run from any cwd: python tests/bench_wave_prefill.py --lengths 256 1024 2048
Times include warm-up separately and compare the public inference wrapper too.
No model is loaded. GPU events and synchronized wall time are both reported.
"""
from pathlib import Path
import argparse
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT.parent), str(ROOT / "hip_inference")]

import torch
import torch.nn.functional as F
from hip_quant import torch_api as T
from hip_inference.core.attention import attention_forward


def measure(fn, repeats):
    start = time.perf_counter()
    for _ in range(5):
        out = fn()
    torch.cuda.synchronize()
    cold_ms = (time.perf_counter() - start) * 1000
    begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    start = time.perf_counter()
    begin.record()
    for _ in range(repeats):
        out = fn()
    end.record()
    torch.cuda.synchronize()
    return out, begin.elapsed_time(end) / repeats, (time.perf_counter()-start)*1000/repeats, cold_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[256, 1024, 2048])
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--heads", type=int, default=24)
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--prefix", type=int, default=0)
    args = ap.parse_args()
    ext = T._load_extension()
    print(f"extension={ext.__file__}", flush=True)
    print(f"GPU={torch.cuda.get_device_properties(0)} dequant_q_to_fp16={hasattr(ext, 'dequant_q_to_fp16')}", flush=True)
    torch.manual_seed(123)
    for length in args.lengths:
        q = torch.randn(1,args.heads,length,args.dim,device="cuda",dtype=torch.float16)
        k = torch.randn(1,args.kv_heads,length+args.prefix,args.dim,device="cuda",dtype=torch.float16)
        v = torch.randn_like(k)
        rep = args.heads // args.kv_heads
        kr, vr = k.repeat_interleave(rep,1), v.repeat_interleave(rep,1)
        q8, k8, v8 = [T.quantize_e4m3(t) for t in (q, kr, vr)]
        kg8, vg8 = [T.quantize_e4m3(t) for t in (k, v)]
        qb, kb, vb = [t.to(torch.bfloat16) for t in (q,k,v)]
        q4, k4 = [T.quantize_int4_packed(t, 0.5) for t in (q, kr)]
        vp, vs, vz = T.kv_quant_v_i4(v[0].contiguous(),4)
        v_deq = T.kv_dequant_v_i4(vp,vs,vz).unsqueeze(0)
        scale = args.dim ** -0.5
        mask = None
        if args.prefix:
            mask = torch.arange(k.shape[2],device='cuda')[None,:] <= (args.prefix+torch.arange(length,device='cuda')[:,None])
        def sdpa(qi, ki, vi):
            return F.scaled_dot_product_attention(qi,ki,vi,attn_mask=mask,is_causal=mask is None,enable_gqa=qi.shape[1]!=ki.shape[1])
        ref = sdpa(q.float(),k.float(),v.float())
        iu4_ref = sdpa(q.float(),k.float(),v_deq.float())
        cases = {
            "sdpa_fp16": lambda: sdpa(q,k,v),
            "sdpa_bf16": lambda: sdpa(qb,kb,vb),
            "fp8_kernel": lambda: ext.wave_attn_prefill_forward(q8,k8,v8,scale,1.,1.,1.,True,False)[0],
            "int4_qk_kernel": lambda: ext.wave_attn_prefill_forward(q4,k4,v8,scale,.5,.5,1.,True,True)[0],
            "wrapper_fp16": lambda: attention_forward(q,k,v,args.dim),
            "wrapper_iu4": lambda: attention_forward(q,k,vp.unsqueeze(0),args.dim,v_scales=vs.unsqueeze(0),v_zp=vz.unsqueeze(0)),
        }
        if hasattr(ext, 'wave_attn_prefill_gqa_forward'):
            cases['gqa_fp8_kernel'] = lambda: ext.wave_attn_prefill_gqa_forward(q8,kg8,vg8,scale)[0]
            cases['gqa_iu4_kernel'] = lambda: ext.wave_attn_prefill_gqa_forward(q8,kg8,vp.unsqueeze(0),scale,v_scales=vs.unsqueeze(0),v_zp=vz.unsqueeze(0))[0]
        print(f"\nshape Q={tuple(q.shape)} K={tuple(k.shape)}",flush=True)
        for name, fn in cases.items():
            out, gpu, wall, cold = measure(fn,args.repeats)
            target = iu4_ref if 'iu4' in name else ref
            cos = F.cosine_similarity(out.float().flatten(),target.flatten(),dim=0).item()
            err = (out.float()-target).abs().max().item()
            print(f"{name:20s} gpu_ms={gpu:.4f} wall_ms={wall:.4f} warmup_ms={cold:.2f} cosine={cos:.7f} maxerr={err:.5f}",flush=True)


if __name__ == "__main__":
    main()
    # Avoid Windows ROCm DLL teardown stalls after the diagnostic completes.
    sys.stdout.flush()
    os._exit(0)
