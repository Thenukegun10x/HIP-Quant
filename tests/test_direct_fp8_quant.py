"""Direct FP8→Quant fused kernel tests.

Verifies that the fused FP8→GGML kernels produce output bit-identical
to the 2-pass (expand→quant) path. Also checks fallback for types
without fused kernels.
"""
import unittest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hip_quant import HipQuant, get_hip_quant

hip = get_hip_quant()

Q4_0 = 2; Q4_1 = 3; Q5_0 = 6; Q5_1 = 7; Q8_0 = 8; Q8_1 = 9
Q2_K = 10; Q4_K = 12

FUSED_TYPES = [Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1]
TYPE_NAMES = {2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
              10: "Q2_K", 12: "Q4_K"}


def _cpu_fp32_to_fp8_e4m3(float_val):
    """Pure-python F32→FP8 E4M3 conversion (u8 write helper)."""
    import struct
    u32, = struct.unpack('I', struct.pack('f', float(float_val)))
    sign = (u32 >> 31) & 1
    abs_u = u32 & 0x7FFFFFFF
    if abs_u == 0:
        return sign << 7
    if abs_u > 0x7F800000:  # NaN
        return (sign << 7) | 0x7F
    if abs_u == 0x7F800000:  # Inf
        return (sign << 7) | 0x7E
    f32_exp = (abs_u >> 23) & 0xFF
    f32_mant = abs_u & 0x7FFFFF
    if f32_exp == 0:  # subnormal
        return sign << 7
    exp = f32_exp - 127 + 7
    if exp <= 0:
        shift = 1 - exp
        if shift > 4:
            return sign << 7
        full = 0x800000 | f32_mant
        total_shift = 20 + shift
        result = full >> total_shift
        remainder = full & ((1 << total_shift) - 1)
        mid = 1 << (total_shift - 1)
        if remainder > mid or (remainder == mid and (result & 1)):
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


def _cpu_fp32_to_fp8_e5m2(float_val):
    """Pure-python F32→FP8 E5M2 conversion (u8 write helper)."""
    import struct
    u32, = struct.unpack('I', struct.pack('f', float(float_val)))
    sign = (u32 >> 31) & 1
    abs_u = u32 & 0x7FFFFFFF
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
        total_shift = 21 + shift
        result = full >> total_shift
        remainder = full & ((1 << total_shift) - 1)
        mid = 1 << (total_shift - 1)
        if remainder > mid or (remainder == mid and (result & 1)):
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


def _f32_to_fp8(f32, fmt="E4M3"):
    """Convert float32 numpy array to FP8 uint8 bytes (CPU side)."""
    flat = f32.ravel()
    if fmt == "E5M2":
        out = np.array([_cpu_fp32_to_fp8_e5m2(v) for v in flat], dtype=np.uint8)
    else:
        out = np.array([_cpu_fp32_to_fp8_e4m3(v) for v in flat], dtype=np.uint8)
    return out.reshape(f32.shape)


def _cpu_fp8_to_f32(value, fmt="E4M3"):
    """Decode one raw E4M3 or E5M2 byte using the device conversion rules."""
    import struct

    value = int(value)
    sign = (value >> 7) & 1
    if fmt == "E5M2":
        exp, mant = (value >> 2) & 0x1F, value & 0x03
        if exp == 31:
            return float("nan") if mant else (float("-inf") if sign else float("inf"))
        if exp == 0:
            result = mant * 0.0000152587890625  # 2^-16
            return -result if sign else result
        i32 = (sign << 31) | ((exp + 112) << 23) | (mant << 21)
    else:
        exp, mant = (value >> 3) & 0x0F, value & 0x07
        if exp == 15 and mant == 7:
            return float("nan")
        if exp == 0:
            result = mant * 0.001953125  # 2^-9
            return -result if sign else result
        i32 = (sign << 31) | ((exp + 120) << 23) | (mant << 20)
    return struct.unpack("f", struct.pack("I", i32))[0]


def _fp8_to_f32(fp8, fmt="E4M3"):
    """CPU reference for the FP8-expansion kernel used by the two-pass path."""
    decoded = np.array(
        [_cpu_fp8_to_f32(value, fmt) for value in fp8.ravel()], dtype=np.float32
    )
    return decoded.reshape(fp8.shape)


def _make_test_data(nrows, n_per_row, fmt="E4M3", seed=42):
    """Generate FP8 bytes and the F32 values produced by expanding those bytes."""
    rng = np.random.RandomState(seed)
    source_f32 = rng.randn(nrows, n_per_row).astype(np.float32)
    # Normalize to reasonable range for sub-1-byte quantization
    source_f32 = source_f32 / np.abs(source_f32).max() * 2.0
    source_f32 = np.ascontiguousarray(source_f32, dtype=np.float32)
    fp8 = _f32_to_fp8(source_f32, fmt)
    return _fp8_to_f32(fp8, fmt), fp8


class TestFusedEqualsTwoPass(unittest.TestCase):
    """Fused FP8→Quant path must be bit-identical to 2-pass (FP8→F32→Quant)."""

    NROWS = 2
    N_PER_ROW = 256

    @classmethod
    def setUpClass(cls):
        cls.f32_e4, cls.fp8_e4 = _make_test_data(cls.NROWS, cls.N_PER_ROW, "E4M3", 123)
        cls.f32_e5, cls.fp8_e5 = _make_test_data(cls.NROWS, cls.N_PER_ROW, "E5M2", 123)

    def _check(self, type_num, src_fmt, fp8, f32, imatrix=None):
        name = TYPE_NAMES[type_num]
        two_pass = hip.quantize_numpy(f32, type_num, imatrix=imatrix)
        fused = hip.quantize_from_fp8(fp8, type_num, imatrix=imatrix,
                                       source_format=src_fmt)
        self.assertTrue(
            np.array_equal(two_pass, fused),
            f"{name} {src_fmt}: mismatch at byte {np.argmax(two_pass != fused)} "
            f"/ {len(two_pass)}"
        )

    def test_q4_0_e4m3(self): self._check(Q4_0, "E4M3", self.fp8_e4, self.f32_e4)
    def test_q4_0_e5m2(self): self._check(Q4_0, "E5M2", self.fp8_e5, self.f32_e5)
    def test_q4_1_e4m3(self): self._check(Q4_1, "E4M3", self.fp8_e4, self.f32_e4)
    def test_q4_1_e5m2(self): self._check(Q4_1, "E5M2", self.fp8_e5, self.f32_e5)
    def test_q5_0_e4m3(self): self._check(Q5_0, "E4M3", self.fp8_e4, self.f32_e4)
    def test_q5_0_e5m2(self): self._check(Q5_0, "E5M2", self.fp8_e5, self.f32_e5)
    def test_q5_1_e4m3(self): self._check(Q5_1, "E4M3", self.fp8_e4, self.f32_e4)
    def test_q5_1_e5m2(self): self._check(Q5_1, "E5M2", self.fp8_e5, self.f32_e5)
    def test_q8_0_e4m3(self): self._check(Q8_0, "E4M3", self.fp8_e4, self.f32_e4)
    def test_q8_0_e5m2(self): self._check(Q8_0, "E5M2", self.fp8_e5, self.f32_e5)
    def test_q8_1_e4m3(self): self._check(Q8_1, "E4M3", self.fp8_e4, self.f32_e4)
    def test_q8_1_e5m2(self): self._check(Q8_1, "E5M2", self.fp8_e5, self.f32_e5)


class TestFusedMultiRow(unittest.TestCase):
    def test_multi_row_q4_0(self):
        f32, fp8 = _make_test_data(4, 128, "E4M3", 77)
        ref = hip.quantize_numpy(f32, Q4_0)
        out = hip.quantize_from_fp8(fp8, Q4_0, source_format="E4M3")
        self.assertTrue(np.array_equal(ref, out))

    def test_multi_row_q8_0(self):
        f32, fp8 = _make_test_data(3, 256, "E5M2", 88)
        ref = hip.quantize_numpy(f32, Q8_0)
        out = hip.quantize_from_fp8(fp8, Q8_0, source_format="E5M2")
        self.assertTrue(np.array_equal(ref, out))

    def test_large_single_row(self):
        f32, fp8 = _make_test_data(1, 1024, "E4M3", 99)
        ref = hip.quantize_numpy(f32, Q4_0)
        out = hip.quantize_from_fp8(fp8, Q4_0, source_format="E4M3")
        self.assertTrue(np.array_equal(ref, out))


class TestFusedFallback(unittest.TestCase):
    """Types without fused kernels fall back to 2-pass and still work."""

    def test_q2_k_fallback_e4m3(self):
        f32, fp8 = _make_test_data(1, 256, "E4M3", 111)
        ref = hip.quantize_numpy(f32, Q2_K)
        out = hip.quantize_from_fp8(fp8, Q2_K, source_format="E4M3")
        self.assertTrue(np.array_equal(ref, out))

    def test_q4_k_fallback_e5m2(self):
        f32, fp8 = _make_test_data(1, 256, "E5M2", 222)
        ref = hip.quantize_numpy(f32, Q4_K)
        out = hip.quantize_from_fp8(fp8, Q4_K, source_format="E5M2")
        self.assertTrue(np.array_equal(ref, out))


class TestFusedEdgeCases(unittest.TestCase):
    def test_zero_input(self):
        fp8 = np.zeros(256, dtype=np.uint8)
        for t in FUSED_TYPES:
            with self.subTest(t=TYPE_NAMES[t]):
                out = hip.quantize_from_fp8(fp8, t, source_format="E4M3")
                self.assertEqual(len(out), hip.row_size(t, 256))

    def test_uniform_input(self):
        fp8 = np.full(256, 0x40, dtype=np.uint8)  # E4M3: ~1.0
        for t in FUSED_TYPES:
            with self.subTest(t=TYPE_NAMES[t]):
                out = hip.quantize_from_fp8(fp8, t, source_format="E4M3")
                self.assertGreater(len(out), 0)

    def test_negative_input(self):
        fp8 = np.full(256, 0xC0, dtype=np.uint8)  # E4M3: ~ -1.0
        for t in [Q4_0, Q5_0, Q8_0]:
            with self.subTest(t=TYPE_NAMES[t]):
                out = hip.quantize_from_fp8(fp8, t, source_format="E4M3")
                self.assertGreater(len(out), 0)


class TestFusedWithImatrix(unittest.TestCase):
    def test_q8_0_with_imatrix(self):
        nrows, n_per_row = 1, 256
        f32, fp8 = _make_test_data(nrows, n_per_row, "E4M3", 333)
        imatrix = np.random.RandomState(444).rand(nrows * n_per_row).astype(np.float32) + 0.5
        imatrix = imatrix / imatrix.mean()
        im2d = imatrix.reshape(nrows, n_per_row)
        ref = hip.quantize_numpy(f32, Q8_0, imatrix=im2d)
        out = hip.quantize_from_fp8(fp8, Q8_0, imatrix=im2d, source_format="E4M3")
        self.assertTrue(np.array_equal(ref, out))


if __name__ == '__main__':
    unittest.main()
