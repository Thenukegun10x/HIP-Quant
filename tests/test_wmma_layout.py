# tests/test_wmma_layout.py
import torch
import torch.nn.functional as F
import math
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch_api as hip_quant

B, H, Seq_Q, Seq_K, Dim = 1, 1, 16, 16, 64
device = "cuda"

torch.manual_seed(42)
q = torch.randn(B, H, Seq_Q, Dim, device=device, dtype=torch.float32)
k = torch.randn(B, H, Seq_K, Dim, device=device, dtype=torch.float32)
v = torch.randn(B, H, Seq_K, Dim, device=device, dtype=torch.float32)
softmax_scale = 1.0 / math.sqrt(Dim)

q_fp8 = hip_quant.quantize_e4m3(q)
k_fp8 = hip_quant.quantize_e4m3(k)
v_fp8 = hip_quant.quantize_e4m3(v)

# Dequantized FP8 reference
q_deq = q_fp8.view(torch.float8_e4m3fn).to(torch.float32)
k_deq = k_fp8.view(torch.float8_e4m3fn).to(torch.float32)
v_deq = v_fp8.view(torch.float8_e4m3fn).to(torch.float32)

ref_fp8_out = F.scaled_dot_product_attention(q_deq, k_deq, v_deq, scale=softmax_scale, is_causal=False)

custom_out = hip_quant._load_extension().rocm_attn_v0_forward(
    q_fp8.contiguous(),
    k_fp8.contiguous(),
    v_fp8.contiguous(),
    float(softmax_scale),
    1.0, 1.0, 1.0,
    False
)

S_ref = torch.matmul(q_deq, k_deq.transpose(-1, -2)) * softmax_scale
print("S_ref score [0,0,:4,:4]:\n", S_ref[0, 0, :4, :4])

print("ref_fp8_out [0,0,:4,:4]:\n", ref_fp8_out[0, 0, :4, :4])
print("custom_out   [0,0,:4,:4]:\n", custom_out[0, 0, :4, :4])

cos_sim = F.cosine_similarity(custom_out.flatten(), ref_fp8_out.flatten(), dim=0).item()
print(f"Cosine Similarity (custom vs ref_fp8_out): {cos_sim:.4f}")
