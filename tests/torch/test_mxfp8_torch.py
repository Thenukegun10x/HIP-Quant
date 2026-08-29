"""GPU MXFP8 OCP UE8M0 quant/dequant — numerical correctness vs Python reference.

Requires: torch ROCm + hip_quant._C rebuilt with torch_ext/mxfp8_kernels.hip
Skips gracefully when no HIP/CUDA device is visible (CI without GPU).
"""

import math
import struct
import unittest

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

try:
    import hip_quant.torch_api as hq
    from hip_quant import _C as _ext
    HAS_EXT = True
except Exception:
    HAS_EXT = False
    hq = None
    _ext = None


def _fp16_to_fp32_bits(u):
    sign = (u >> 15) & 1
    exp = (u >> 10) & 0x1F
    mant = u & 0x3FF
    if exp == 0 and mant == 0:
        return struct.unpack("f", struct.pack("I", sign << 31))[0]
    # not needed for this test — only used via torch half tensors
    return 0.0


# Python reference — must match hip_quant_util.h exactly (RNE, saturate)
def _fp32_to_fp8_e4m3_bits(u):
    sign = u >> 31
    abs_u = u & 0x7FFFFFFF
    if abs_u == 0:
        return sign << 7
    if abs_u > 0x7F800000:
        return (sign << 7) | 0x7F
    if abs_u == 0x7F800000:
        return (sign << 7) | 0x7E
    f32_exp = (abs_u >> 23) & 0xFF
    f32_mant = abs_u & 0x7FFFFF
    if f32_exp == 0:
        return sign << 7
    exp = f32_exp - 127 + 7
    if exp <= 0:
        shift = 1 - exp
        if shift > 4:
            return sign << 7
        full = 0x800000 | f32_mant
        total = 20 + shift
        result = full >> total
        remainder = full & ((1 << total) - 1)
        midpoint = 1 << (total - 1)
        if remainder > midpoint or (remainder == midpoint and (result & 1)):
            result += 1
        if result >= 8:
            return (sign << 7) | (1 << 3)
        return (sign << 7) | (result & 0x7)
    fp8_mant = (f32_mant >> 20) & 0x7
    rnd = f32_mant & 0xFFFFF
    if rnd > 0x80000 or (rnd == 0x80000 and (fp8_mant & 1)):
        fp8_mant += 1
        if fp8_mant >= 8:
            fp8_mant = 0
            exp += 1
    if exp >= 16 or (exp == 15 and fp8_mant == 7):
        return (sign << 7) | 0x7E
    return (sign << 7) | (exp << 3) | fp8_mant


def fp32_to_fp8_e4m3(f):
    return _fp32_to_fp8_e4m3_bits(struct.unpack("<I", struct.pack("<f", float(f)))[0])


def _fp32_to_fp8_e5m2_bits(u):
    sign = u >> 31
    abs_u = u & 0x7FFFFFFF
    if abs_u == 0:
        return sign << 7
    if abs_u > 0x7F800000:
        return (sign << 7) | 0x7F
    if abs_u == 0x7F800000:
        return (sign << 7) | 0x7C
    f32_exp = (abs_u >> 23) & 0xFF
    f32_mant = abs_u & 0x7FFFFF
    if f32_exp == 0:
        return sign << 7
    exp = f32_exp - 127 + 15
    if exp <= 0:
        shift = 1 - exp
        if shift > 3:
            return sign << 7
        full = 0x800000 | f32_mant
        total = 21 + shift
        result = full >> total
        remainder = full & ((1 << total) - 1)
        midpoint = 1 << (total - 1)
        if remainder > midpoint or (remainder == midpoint and (result & 1)):
            result += 1
        if result >= 4:
            return (sign << 7) | (1 << 2)
        return (sign << 7) | (result & 0x3)
    fp8_mant = (f32_mant >> 21) & 0x3
    rnd = f32_mant & 0x1FFFFF
    if rnd > 0x100000 or (rnd == 0x100000 and (fp8_mant & 1)):
        fp8_mant += 1
        if fp8_mant >= 4:
            fp8_mant = 0
            exp += 1
    if exp >= 31:
        return (sign << 7) | 0x7B
    return (sign << 7) | (exp << 2) | fp8_mant


def fp32_to_fp8_e5m2(f):
    return _fp32_to_fp8_e5m2_bits(struct.unpack("<I", struct.pack("<f", float(f)))[0])


def ue8m0_to_scale(b):
    if b == 0xFF:
        return float("nan")
    return math.ldexp(1.0, b - 127)


def amax_to_ue8m0(amax, max_elem=448.0):
    if not math.isfinite(amax) or amax <= 0:
        return 0
    ratio = amax / max_elem
    u = struct.unpack("<I", struct.pack("<f", ratio))[0]
    exp_bits = (u >> 23) & 0xFF
    if exp_bits == 0:
        ce = math.ceil(math.log2(ratio) - 1e-7)
        return max(0, min(254, ce + 127))
    exp = int(exp_bits) - 127
    mant = u & 0x007FFFFF
    ce = exp if mant == 0 else exp + 1
    return max(0, min(254, ce + 127))


def reference_quant_mxfp8(x_1d, max_elem, to_fp8):
    """Reference per-32 microscaling for a flat list."""
    out_q, out_ue = [], []
    for i in range(0, len(x_1d), 32):
        block = x_1d[i:i+32]
        amax = max((abs(v) for v in block if math.isfinite(v)), default=0.0)
        ue = amax_to_ue8m0(amax, max_elem)
        scale = ue8m0_to_scale(ue)
        inv = 1.0 / scale if scale else 0.0
        for v in block:
            q = to_fp8(v * inv if math.isfinite(v) else v)
            out_q.append(q)
            out_ue.append(ue)
        # keep per-block ue aligned: repeat 32 times above, but scales storage is 1 per 32
    # scales per row = ceil(K/32) — one ue per 32
    num_blocks = (len(x_1d) + 31) // 32
    scales = [amax_to_ue8m0(max((abs(v) for v in x_1d[i*32:(i+1)*32] if math.isfinite(v)), default=0.0), max_elem)
              for i in range(num_blocks)]
    # rebuild q correctly (above out_q already correct per-block)
    # recompute to avoid double-count
    qs = []
    for bi in range(num_blocks):
        amax = max((abs(v) for v in x_1d[bi*32:(bi+1)*32] if math.isfinite(v)), default=0.0)
        ue = amax_to_ue8m0(amax, max_elem)
        sc = ue8m0_to_scale(ue)
        inv = 1.0/sc if sc else 0
        for v in x_1d[bi*32:(bi+1)*32]:
            qs.append(to_fp8(v*inv if math.isfinite(v) else v))
    return qs, scales


requires_gpu = unittest.skipUnless(
    TORCH_AVAILABLE and HAS_EXT and torch.cuda.is_available() and hasattr(_ext, "quantize_mxfp8_e4m3"),
    "MXFP8 kernels require HIP/CUDA GPU and rebuilt hip_quant._C (mxfp8_kernels.hip)"
)


@requires_gpu
class TestMxFp8TorchNumerical(unittest.TestCase):
    def _check_e4m3(self, x_cpu, rtol=0, atol=0):
        x = x_cpu.cuda()
        q, s = hq.quantize_mxfp8_e4m3(x)
        y = hq.dequantize_mxfp8_e4m3(q, s)
        # python reference
        flat = x_cpu.flatten().tolist()
        ref_q, ref_s = reference_quant_mxfp8(flat, 448.0, fp32_to_fp8_e4m3)
        self.assertEqual(q.cpu().flatten().tolist(), ref_q)
        self.assertEqual(s.cpu().flatten().tolist(), ref_s)
        # dequant reference
        ref_y = []
        blocks = (len(flat) + 31)//32
        for bi in range(blocks):
            sc = ue8m0_to_scale(ref_s[bi])
            for j in range(32):
                idx = bi*32+j
                if idx >= len(flat): break
                # fp8 -> fp32 helper
                b = ref_q[idx]
                # e4m3_to_fp32
                sign=(b>>7)&1; exp=(b>>3)&0xF; mant=b&0x7
                if exp==15 and mant==7:
                    v=float("nan")
                elif exp==0 and mant==0:
                    v=0.0
                elif exp==0:
                    v=mant*0.001953125 * (-1 if sign else 1)
                else:
                    v=struct.unpack("<f", struct.pack("<I", (sign<<31)|((exp+120)<<23)|(mant<<20)))[0]
                ref_y.append(v*sc if not math.isnan(v) and not math.isnan(sc) else float("nan"))
        torch.testing.assert_close(y.cpu().flatten()[:len(ref_y)], torch.tensor(ref_y, dtype=torch.float32), rtol=rtol, atol=atol, equal_nan=True)

    def test_zero_block(self):
        x = torch.zeros(1, 32, dtype=torch.float32)
        q, s = hq.quantize_mxfp8_e4m3(x.cuda())
        self.assertEqual(int(s[0,0].item()), 0)
        self.assertTrue(torch.all(q==0))
        y = hq.dequantize_mxfp8_e4m3(q, s)
        torch.testing.assert_close(y, x.cuda(), rtol=0, atol=0)

    def test_one_and_224_and_448_boundaries(self):
        for val, exp_ue, exp_q in [(1.0, 119, 0x78), (224.0, 126, 0x7E), (448.0, 127, 0x7E)]:
            x = torch.full((1,32), val, dtype=torch.float32).cuda()
            q, s = hq.quantize_mxfp8_e4m3(x)
            self.assertEqual(int(s[0,0].item()), exp_ue, f"ue for {val}")
            self.assertEqual(int(q[0,0].item()), exp_q, f"q[0] for {val}")

    def test_five_hundred_increases_scale(self):
        x = torch.full((1,32), 500.0, dtype=torch.float32).cuda()
        q, s = hq.quantize_mxfp8_e4m3(x)
        self.assertEqual(int(s[0,0].item()), 128)
        self.assertEqual(int(q[0,0].item()), 0x78)  # 256*2
        y = hq.dequantize_mxfp8_e4m3(q, s)
        self.assertAlmostEqual(float(y[0,0].item()), 512.0, delta=1e-3)

    def test_tail_not_multiple_of_32(self):
        for cols in (33, 31, 63):
            x = torch.randn(2, cols, dtype=torch.float32).cuda()
            for fn_q, fn_dq in [(hq.quantize_mxfp8_e4m3, hq.dequantize_mxfp8_e4m3),
                                (hq.quantize_mxfp8_e5m2, hq.dequantize_mxfp8_e5m2)]:
                q, s = fn_q(x)
                self.assertEqual(q.shape, x.shape)
                self.assertEqual(s.shape, (2, (cols+31)//32))
                self.assertEqual(s.dtype, torch.uint8)
                y = fn_dq(q, s)
                self.assertEqual(y.shape, x.shape)

    def test_random_roundtrip_e4m3_matches_reference(self):
        torch.manual_seed(0)
        x = (torch.randn(4, 64, dtype=torch.float32) * 4).cpu()
        self._check_e4m3(x)

    def test_random_roundtrip_e5m2_matches_reference(self):
        torch.manual_seed(1)
        x = (torch.randn(2, 96, dtype=torch.float32) * 200).cpu()
        flat = x.flatten().tolist()
        ref_q, ref_s = reference_quant_mxfp8(flat, 57344.0, fp32_to_fp8_e5m2)
        q, s = hq.quantize_mxfp8_e5m2(x.cuda())
        self.assertEqual(q.cpu().flatten().tolist(), ref_q)
        self.assertEqual(s.cpu().flatten().tolist(), ref_s)
        y = hq.dequantize_mxfp8_e5m2(q, s)
        # verify y via reference dequant
        blocks=len(ref_s)
        ref_y=[]
        for bi in range(blocks):
            sc=ue8m0_to_scale(ref_s[bi])
            for j in range(32):
                idx=bi*32+j
                if idx>=len(flat): break
                b=ref_q[idx]; sign=(b>>7)&1; exp=(b>>2)&0x1F; mant=b&0x3
                if exp==31 and mant==0:
                    v=float("inf") * (-1 if sign else 1)
                elif exp==31:
                    v=float("nan")
                elif exp==0 and mant==0:
                    v=0.0
                elif exp==0:
                    v=mant*0.0000152587890625 * (-1 if sign else 1)
                else:
                    v=struct.unpack("<f", struct.pack("<I", (sign<<31)|((exp+112)<<23)|(mant<<21)))[0]
                ref_y.append(v*sc if not math.isnan(v) else float("nan"))
        torch.testing.assert_close(y.cpu().flatten()[:len(ref_y)], torch.tensor(ref_y, dtype=torch.float32), rtol=0, atol=0, equal_nan=True)

    def test_dtypes_f16_bf16(self):
        for dtype in (torch.float16, torch.bfloat16):
            x = torch.tensor([[1.0, -1.0, 0.0, 2.0]], dtype=dtype, device="cuda")
            for fn in (hq.quantize_mxfp8_e4m3, hq.quantize_mxfp8_e5m2):
                q, s = fn(x)
                self.assertEqual(q.dtype, torch.uint8)
                self.assertEqual(s.dtype, torch.uint8)
                y = (hq.dequantize_mxfp8_e4m3 if fn==hq.quantize_mxfp8_e4m3 else hq.dequantize_mxfp8_e5m2)(q, s)
                self.assertEqual(y.dtype, torch.float32)

    def test_nan_inf_preserved_e4m3(self):
        x = torch.tensor([[float("nan"), float("inf"), -float("inf"), 1.0] + [0.0]*28], dtype=torch.float32, device="cuda")
        q, s = hq.quantize_mxfp8_e4m3(x)
        # NaN byte is 0x7F, Inf saturates to 0x7E for E4M3
        self.assertEqual(int(q[0,0].item()) & 0x7F, 0x7F)
        self.assertEqual(int(q[0,1].item()), 0x7E)
        self.assertEqual(int(q[0,2].item()), 0xFE)

    def test_vs_fp32_scale_error_within_expected(self):
        torch.manual_seed(2)
        x = torch.randn(8, 64, dtype=torch.float32, device="cuda")
        q_mx, s_mx = hq.quantize_mxfp8_e4m3(x)
        y_mx = hq.dequantize_mxfp8_e4m3(q_mx, s_mx)
        q_fp32, s_fp32 = hq.quantize_e4m3_blockwise(x, block_size=32)
        y_fp32 = hq.dequantize_e4m3_blockwise(q_fp32, s_fp32, block_size=32)
        err_mx = (y_mx - x).abs().max().item()
        err_fp32 = (y_fp32 - x).abs().max().item()
        # MX pow2 scale is ~1.4x worse than FP32 scale (checked in plan), allow 2x
        self.assertLess(err_mx, err_fp32 * 2.0 + 1e-6)

    def test_large_row_counts(self):
        for nrows in (33, 130):
            x = torch.randn(nrows, 64, dtype=torch.float32, device="cuda")
            q, s = hq.quantize_mxfp8_e4m3(x)
            self.assertEqual(q.shape, x.shape)
            self.assertEqual(s.shape, (nrows, 2))
            y = hq.dequantize_mxfp8_e4m3(q, s)
            self.assertEqual(y.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
