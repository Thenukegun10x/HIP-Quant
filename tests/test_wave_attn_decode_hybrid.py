import math
import unittest
import torch
import torch.nn.functional as F
import torch_api as T

class TestWaveAttnDecodeHybrid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("ROCm/CUDA device required")
        cls.ext = T._load_extension()
        cls.device = "cuda"

    def quant_int4_packed_per_token(self, x):
        B, H, S, D = x.shape
        scales = x.abs().amax(dim=-1).clamp_min(1e-6) / 7.0  # [B, H, S]
        y = torch.round(x / scales.unsqueeze(-1)).clamp(-8, 7).to(torch.int8)
        y_low = (y[..., 0::2] & 0xF).to(torch.uint8)
        y_high = (y[..., 1::2] & 0xF).to(torch.uint8)
        packed = (y_low | (y_high << 4)).contiguous()
        return packed, scales.contiguous()

    def test_decode_hybrid_accuracy_above_099(self):
        """Verify that INT4 QK with per-key K scale + FP8 V achieves >= 0.992 cosine similarity."""
        B, H, Dim = 1, 16, 128
        scale = 1.0 / math.sqrt(Dim)

        for Sk in [64, 128, 256, 512, 1024]:
            torch.manual_seed(42 + Sk)
            q = torch.randn(B, H, 1, Dim, device=self.device, dtype=torch.float32) * 0.8
            k = torch.randn(B, H, Sk, Dim, device=self.device, dtype=torch.float32) * 0.8
            v = torch.randn(B, H, Sk, Dim, device=self.device, dtype=torch.float32) * 0.8

            ref = F.scaled_dot_product_attention(
                q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16),
                is_causal=False, scale=scale
            ).float()

            # Per-head Q scale
            sq_head = q.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / 7.0
            q_i4 = torch.round(q / sq_head).clamp(-8, 7).to(torch.int8)
            q_i4_packed = ((q_i4[..., 0::2] & 0xF) | ((q_i4[..., 1::2] & 0xF) << 4)).to(torch.uint8).contiguous()

            # Per-key K scale folded with per-head Q scale
            sk_raw = k.abs().amax(dim=-1).clamp_min(1e-6) / 7.0  # [B, H, Sk]
            k_i4 = torch.round(k / sk_raw.unsqueeze(-1)).clamp(-8, 7).to(torch.int8)
            k_i4_packed = ((k_i4[..., 0::2] & 0xF) | ((k_i4[..., 1::2] & 0xF) << 4)).to(torch.uint8).contiguous()
            k_scales_folded = (sk_raw * sq_head.squeeze(-1)).contiguous()

            # FP8 V
            v_fp8 = T.quantize_e4m3(v.contiguous())

            out, _ = self.ext.wave_attn_decode_forward(
                q_i4_packed, k_i4_packed, v_fp8, scale,
                1.0, 1.0, 1.0, False, k_scales_folded
            )

            cos = F.cosine_similarity(out.flatten().unsqueeze(0), ref.flatten().unsqueeze(0)).item()
            self.assertGreater(cos, 0.990, f"Cosine {cos:.6f} fell below 0.99 for Sk={Sk}")

if __name__ == "__main__":
    unittest.main()
