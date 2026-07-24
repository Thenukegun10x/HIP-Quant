import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
import torch_api as hip_quant

print('=== ROCmAttention-v8 Correctness & Speed ===')
configs = [
    (1, 1, 16, 16, 64, False),
    (2, 4, 32, 32, 64, False),
    (2, 4, 64, 64, 64, False),
    (4, 8, 128, 128, 64, False),
    (4, 8, 256, 256, 64, False),
    (4, 8, 512, 512, 64, False),
    (2, 4, 32, 32, 64, True),
    (2, 4, 64, 64, 64, True),
    (4, 8, 128, 128, 64, True),
    (2, 4, 128, 128, 128, False),
]

all_ok = True
for (B, H, Seq_Q, Seq_K, Dim, causal) in configs:
    torch.manual_seed(42)
    q = torch.randn(B, H, Seq_Q, Dim, device='cuda', dtype=torch.float32)
    k = torch.randn(B, H, Seq_K, Dim, device='cuda', dtype=torch.float32)
    v = torch.randn(B, H, Seq_K, Dim, device='cuda', dtype=torch.float32)
    scale = 1.0 / math.sqrt(Dim)
    ref = F.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=causal)
    out = hip_quant.wave_attn(q, k, v, is_causal=causal, scale=scale)
    cos = F.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
    l1 = (out - ref).abs().mean().item()
    ok = cos > 0.90
    if not ok: all_ok = False
    causal_s = "(Causal)" if causal else ""
    print(f"B={B},H={H},S={Seq_Q:3d},D={Dim:3d} {causal_s:9s}  cos={cos:.4f}  L1={l1:.4f}  {'OK' if ok else 'FAIL'}")

if all_ok:
    print("\n=== All correctness checks passed ===")


# ── Kernel stability benchmark (multiple seeds, edge cases) ──
print("\n=== Stability: 20 random seeds, S=256, D=64 ===")
cos_vals = []
for seed in range(20):
    torch.manual_seed(seed)
    B, H, S, D = 4, 8, 256, 64
    scale = 1.0 / math.sqrt(D)
    q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float32)
    k = torch.randn(B, H, S, D, device='cuda', dtype=torch.float32)
    v = torch.randn(B, H, S, D, device='cuda', dtype=torch.float32)
    ref = F.scaled_dot_product_attention(q, k, v, scale=scale)
    out = hip_quant.rocm_attn_v8(q, k, v, scale=scale)
    cos = F.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
    has_nan = torch.isnan(out).any().item()
    has_inf = torch.isinf(out).any().item()
    cos_vals.append(cos)
    status = "OK" if (not has_nan and not has_inf and cos > 0.99) else "FAIL"
    if status != "OK":
        print(f"  seed={seed:2d}: cos={cos:.4f}  NaN={has_nan}  Inf={has_inf}  {status}")

import statistics
print(f"  min_cos={min(cos_vals):.4f}  mean={statistics.mean(cos_vals):.4f}  stdev={statistics.stdev(cos_vals):.5f}  {'STABLE' if min(cos_vals) > 0.99 else 'UNSTABLE'}")

# ── Edge cases ──
print("\n=== Edge case: extreme input scale ===")
torch.manual_seed(42)
q = torch.randn(4, 8, 128, 64, device='cuda') * 100.0
k = torch.randn(4, 8, 128, 64, device='cuda') * 100.0
v = torch.randn(4, 8, 128, 64, device='cuda')
scale = 1.0 / math.sqrt(64)
ref = F.scaled_dot_product_attention(q, k, v, scale=scale)
out = hip_quant.rocm_attn_v8(q, k, v, scale=scale)
cos = F.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
print(f"  large-scale input: cos={cos:.4f}  {'OK' if cos > 0.90 else 'FAIL'}")

print("\n=== Edge case: zero rows ===")
torch.manual_seed(42)
q = torch.randn(2, 1, 64, 64, device='cuda')
k = torch.randn(2, 1, 64, 64, device='cuda')
v = torch.randn(2, 1, 64, 64, device='cuda')
q[0, 0, 0, :] = 0.0  # zero first query row
ref = F.scaled_dot_product_attention(q, k, v, scale=scale)
out = hip_quant.rocm_attn_v8(q, k, v, scale=scale)
has_nan = torch.isnan(out).any().item()
cos = F.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
print(f"  zero query row: cos={cos:.4f}  NaN={has_nan}  {'OK' if cos > 0.90 and not has_nan else 'FAIL'}")

# Speed comparison
print("\n=== Speed: v7 vs v8 vs SDPA ===")
import time

speed_configs = [
    (1, 1, 16, 64),
    (1, 1, 32, 64),
    (1, 1, 64, 64),
    (1, 1, 128, 64),
    (1, 1, 256, 64),
    (1, 1, 512, 64),
    (2, 4, 128, 64),
    (2, 4, 256, 64),
    (2, 4, 512, 64),
    (4, 8, 128, 64),
    (4, 8, 256, 64),
    (4, 8, 512, 64),
    (4, 8, 128, 128),
    (4, 8, 256, 128),
    (4, 8, 512, 128),
]

ext = hip_quant._load_extension()
warmup, iters = 5, 50

for B, H, S, D in speed_configs:
    torch.manual_seed(42)
    scale = 1.0 / math.sqrt(D)
    q32 = torch.randn(B, H, S, D, device='cuda')
    k32 = torch.randn(B, H, S, D, device='cuda')
    v32 = torch.randn(B, H, S, D, device='cuda')
    q_fp8 = hip_quant.quantize_e4m3(q32).contiguous()
    k_fp8 = hip_quant.quantize_e4m3(k32).contiguous()
    v_fp8 = hip_quant.quantize_e4m3(v32).contiguous()

    # v7
    def run_v7():
        return ext.rocm_attn_v7_forward(q_fp8, k_fp8, v_fp8, float(scale), 1.0, 1.0, 1.0, False)
    # v8
    def run_v8():
        return ext.rocm_attn_v8_forward(q_fp8, k_fp8, v_fp8, float(scale), 1.0, 1.0, 1.0, False)
    # SDPA
    def run_sdpa():
        return F.scaled_dot_product_attention(q32, k32, v32, scale=scale)

    for _ in range(warmup):
        run_v7(); run_v8(); run_sdpa()
    torch.cuda.synchronize()

    # v7
    start = time.perf_counter()
    for _ in range(iters): run_v7()
    torch.cuda.synchronize()
    v7_ms = (time.perf_counter() - start) / iters * 1000

    # v8
    start = time.perf_counter()
    for _ in range(iters): run_v8()
    torch.cuda.synchronize()
    v8_ms = (time.perf_counter() - start) / iters * 1000

    # SDPA
    start = time.perf_counter()
    for _ in range(iters): run_sdpa()
    torch.cuda.synchronize()
    sdpa_ms = (time.perf_counter() - start) / iters * 1000

    flops = 4.0 * B * H * S * S * D
    print(f"B={B},H={H},S={S:3d},D={D:3d}  v7={v7_ms:6.3f}ms  v8={v8_ms:6.3f}ms  SDPA={sdpa_ms:6.3f}ms  v8/v7={v7_ms/v8_ms:.2f}x")
