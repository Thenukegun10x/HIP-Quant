# tests/bench_attn_compare.py
import torch
import torch.nn.functional as F
import math, time, sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch_api as hip_quant

def bench(name, fn, warmup=5, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters * 1000.0
    return elapsed

def flops(B, H, S, D, ms):
    ops = 4.0 * B * H * S * S * D
    tflops = ops / ms * 1e-9
    return tflops

configs = [
    (1, 1, 16, 64), (1, 1, 32, 64), (1, 1, 64, 64),
    (1, 1, 128, 64), (1, 1, 256, 64), (1, 1, 512, 64),
    (2, 4, 32, 64), (2, 4, 64, 64), (2, 4, 128, 64),
    (2, 4, 256, 64), (2, 4, 512, 64),
    (4, 8, 32, 64), (4, 8, 64, 64), (4, 8, 128, 64),
    (4, 8, 256, 64), (4, 8, 512, 64),
    (4, 8, 128, 128), (4, 8, 256, 128), (4, 8, 512, 128),
]

print(f"{'Config':<20} {'FlashAttn2(FP16)':<18} {'WaveAttn(FP8)':<18} {'SDPA(FP32)':<15}")
print(f"{'':20} {'ms':>6} {'TFLOP/s':>8}   {'ms':>6} {'TFLOP/s':>8}   {'ms':>6} {'TFLOP/s':>6}")
print("-" * 95)

for B, H, S, D in configs:
    torch.manual_seed(42)
    scale = 1.0 / math.sqrt(D)

    q16 = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    k16 = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    v16 = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    flash_ms = bench('flash', lambda: F.scaled_dot_product_attention(q16, k16, v16, scale=scale))

    q32 = q16.float(); k32 = k16.float(); v32 = v16.float()
    q_fp8 = hip_quant.quantize_e4m3(q32).contiguous()
    k_fp8 = hip_quant.quantize_e4m3(k32).contiguous()
    v_fp8 = hip_quant.quantize_e4m3(v32).contiguous()
    ext = hip_quant._load_extension()

    wave_ms = bench('wave', lambda: ext.wave_attn_forward(q_fp8, k_fp8, v_fp8, float(scale), 1.0, 1.0, 1.0, False))
    sdpa_ms = bench('sdpa', lambda: F.scaled_dot_product_attention(q32, k32, v32, scale=scale))

    print(f"B={B},H={H},S={S:3d},D={D:<6} {flash_ms:6.3f} {flops(B,H,S,D,flash_ms):8.2f}   "
          f"{wave_ms:6.3f} {flops(B,H,S,D,wave_ms):8.2f}   "
          f"{sdpa_ms:6.3f} {flops(B,H,S,D,sdpa_ms):6.2f}")
    print(f"  WaveAttn speedup: {flash_ms/wave_ms:.2f}x vs FlashAttn2, {sdpa_ms/wave_ms:.2f}x vs SDPA FP32")
    print()
