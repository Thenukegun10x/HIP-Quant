"""CPU-only MXFP8 numerical correctness — OCP UE8M0 reference.

No GPU required. Validates the Python reference that the HIP kernels
implement bit-exact (RNE, saturate, ceil(log2)) before GPU tests run.
"""

import math
import struct
import unittest


# ---------------------------------------------------------------------------
# FP8 helpers — copy of hip_quant_util.h fp32_bits_to_fp8_{e4m3,e5m2}
# ---------------------------------------------------------------------------
def _fp32_to_fp8_e4m3_bits(u: int) -> int:
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


def fp32_to_fp8_e4m3(f: float) -> int:
    return _fp32_to_fp8_e4m3_bits(struct.unpack("<I", struct.pack("<f", f))[0])


def fp8_e4m3_to_fp32(b: int) -> float:
    sign = (b >> 7) & 1
    exp = (b >> 3) & 0xF
    mant = b & 0x7
    if exp == 15 and mant == 7:
        return struct.unpack("<f", struct.pack("<I", (sign << 31) | 0x7FC00000))[0]
    if exp == 0 and mant == 0:
        return struct.unpack("<f", struct.pack("<I", sign << 31))[0]
    if exp == 0:
        return (-1 if sign else 1) * mant * 0.001953125
    i = (sign << 31) | ((exp + 120) << 23) | (mant << 20)
    return struct.unpack("<f", struct.pack("<I", i))[0]


def _fp32_to_fp8_e5m2_bits(u: int) -> int:
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


def fp32_to_fp8_e5m2(f: float) -> int:
    return _fp32_to_fp8_e5m2_bits(struct.unpack("<I", struct.pack("<f", f))[0])


def fp8_e5m2_to_fp32(b: int) -> float:
    sign = (b >> 7) & 1
    exp = (b >> 2) & 0x1F
    mant = b & 0x3
    if exp == 31:
        if mant == 0:
            return struct.unpack("<f", struct.pack("<I", (sign << 31) | 0x7F800000))[0]
        return struct.unpack("<f", struct.pack("<I", (sign << 31) | 0x7FC00000))[0]
    if exp == 0 and mant == 0:
        return struct.unpack("<f", struct.pack("<I", sign << 31))[0]
    if exp == 0:
        return (-1 if sign else 1) * mant * 0.0000152587890625
    i = (sign << 31) | ((exp + 112) << 23) | (mant << 21)
    return struct.unpack("<f", struct.pack("<I", i))[0]


# ---------------------------------------------------------------------------
# UE8M0 helpers — mirrors mxfp8_kernels.hip
# ---------------------------------------------------------------------------
def ue8m0_to_scale(b: int) -> float:
    if b == 0xFF:
        return float("nan")
    return math.ldexp(1.0, b - 127)


def amax_to_ue8m0(amax: float, max_elem: float = 448.0) -> int:
    if not math.isfinite(amax) or amax <= 0:
        return 0
    ratio = amax / max_elem
    # ceil(log2) via integer exponent — must match device __float_as_uint path
    u = struct.unpack("<I", struct.pack("<f", ratio))[0]
    exp_bits = (u >> 23) & 0xFF
    if exp_bits == 0:
        # subnormal fallback
        ce = math.ceil(math.log2(ratio) - 1e-7)
        ue = ce + 127
        return max(0, min(254, ue))
    exp = int(exp_bits) - 127
    mant = u & 0x007FFFFF
    ce = exp if mant == 0 else exp + 1
    return max(0, min(254, ce + 127))


def quant_mxfp8_block(block, max_elem=448.0, to_fp8=fp32_to_fp8_e4m3, to_fp32=fp8_e4m3_to_fp32):
    amax = max((abs(x) for x in block if math.isfinite(x)), default=0.0)
    ue = amax_to_ue8m0(amax, max_elem)
    scale = ue8m0_to_scale(ue)
    inv = 1.0 / scale if scale != 0 else 0.0
    qs = [to_fp8(x * inv if math.isfinite(x) else x) for x in block]
    deq = [to_fp32(q) * scale for q in qs]
    return qs, ue, scale, deq


class TestMxFp8Numerical(unittest.TestCase):
    def test_ue_boundaries_e4m3(self):
        cases = [
            (0.5, 118), (1.0, 119), (2.0, 120), (4.0, 121), (8.0, 122),
            (16.0, 123), (32.0, 124), (64.0, 125), (112.0, 125),
            (224.0, 126), (448.0, 127), (896.0, 128),
        ]
        for amax, exp_ue in cases:
            with self.subTest(amax=amax):
                self.assertEqual(amax_to_ue8m0(amax), exp_ue)
        # ceiling behaviour
        self.assertEqual(amax_to_ue8m0(448.0), 127)
        self.assertEqual(amax_to_ue8m0(448.0 * 1.0001), 128)
        self.assertEqual(amax_to_ue8m0(224.0001), 127)

    def test_zero_block(self):
        qs, ue, sc, deq = quant_mxfp8_block([0.0] * 32)
        self.assertEqual(ue, 0)
        self.assertTrue(all(q == 0 for q in qs))
        self.assertTrue(all(v == 0.0 for v in deq))

    def test_small_block(self):
        # 1e-6 -> ratio ~2.2e-9 -> ceil -28 -> ue 99 -> scale 2^-28
        qs, ue, sc, deq = quant_mxfp8_block([1e-6] * 32)
        self.assertEqual(ue, 99)
        self.assertAlmostEqual(sc, math.ldexp(1.0, 99 - 127), places=12)

    def test_one_maps_through_scale(self):
        qs, ue, sc, deq = quant_mxfp8_block([1.0] * 32)
        self.assertEqual(ue, 119)
        self.assertAlmostEqual(sc, 0.00390625)
        # 1.0/0.00390625=256 -> E4M3 0x78 -> 256*scale =1.0
        self.assertEqual(qs[0], 0x78)
        self.assertAlmostEqual(deq[0], 1.0, places=6)

    def test_max_448(self):
        qs, ue, sc, deq = quant_mxfp8_block([448.0] * 32)
        self.assertEqual(ue, 127)
        self.assertEqual(sc, 1.0)
        self.assertEqual(qs[0], 0x7E)
        self.assertEqual(deq[0], 448.0)

    def test_over_max_increases_scale(self):
        qs, ue, sc, deq = quant_mxfp8_block([500.0] * 32)
        self.assertEqual(ue, 128)
        self.assertEqual(sc, 2.0)
        self.assertEqual(qs[0], 0x78)
        self.assertEqual(deq[0], 512.0)

    def test_e5m2_max(self):
        qs, ue, sc, deq = quant_mxfp8_block([57344.0] * 32, max_elem=57344.0,
                                            to_fp8=fp32_to_fp8_e5m2, to_fp32=fp8_e5m2_to_fp32)
        self.assertEqual(ue, 127)
        self.assertEqual(qs[0], 0x7B)
        self.assertEqual(deq[0], 57344.0)

    def test_nan_and_inf_preserved(self):
        import math as m
        blk = [float("nan"), float("inf"), -float("inf"), 1.0] + [0.0] * 28
        qs, _, _, _ = quant_mxfp8_block(blk)
        self.assertEqual(qs[0] & 0x7F, 0x7F)  # NaN
        self.assertEqual(qs[1], 0x7E)  # Inf saturates to max for E4M3
        self.assertEqual(qs[2], 0xFE)

    def test_random_error_bound_e4m3(self):
        # limit *finite* reconstruction error (NaN blocks are not measured)
        import random
        random.seed(0)
        max_err = 0.0
        for _ in range(500):
            blk = [random.uniform(-8, 8) for _ in range(32)]
            _, _, _, deq = quant_mxfp8_block(blk)
            for x, y in zip(blk, deq):
                max_err = max(max_err, abs(x - y))
        # power-of-two scale loses ~40% vs FP32 scale (checked in torch_api plan)
        self.assertLess(max_err, 0.6)

    def test_ue8m0_reserved_nan(self):
        self.assertTrue(math.isnan(ue8m0_to_scale(0xFF)))
        self.assertEqual(amax_to_ue8m0(float("inf")), 0)
        self.assertEqual(amax_to_ue8m0(float("nan")), 0)

    def test_subnormal_ratio_regression(self):
        # tiny amax still yields valid ue, not overflow
        ue = amax_to_ue8m0(1e-38, max_elem=448.0)
        self.assertGreaterEqual(ue, 0)
        self.assertLessEqual(ue, 254)


if __name__ == "__main__":
    unittest.main()
