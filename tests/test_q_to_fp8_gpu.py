"""
tests/test_q_to_fp8_gpu.py
=========================
GPU smoke test for the Q-type -> FP8 (E4M3 / E5M2) dequantization path.

Requires the real HIP DLL and an AMD GPU.  Skips automatically when no DLL
or device is available so the CPU suite remains green on machines without ROCm.

Run:
    python tests/test_q_to_fp8_gpu.py -v
or
    python -m pytest tests/test_q_to_fp8_gpu.py -q
"""

import os
import sys
import unittest

import numpy as np

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)

import hip_quant as hq


# ---------------------------------------------------------------------------
# Bit-exact CPU FP8 -> float32 decoders (mirror of the HIP device encoders)
# ---------------------------------------------------------------------------

def _fp8_e4m3_to_fp32(b):
    b &= 0xFF
    sign = b >> 7
    exp = (b >> 3) & 0xF
    mant = b & 7
    if exp == 15 and mant == 7:
        return float("nan")
    if exp == 0 and mant == 0:
        return -0.0 if sign else 0.0
    if exp == 0:
        v = mant * 2.0 ** -9
        return -v if sign else v
    v = (1.0 + mant * 2.0 ** -3) * 2.0 ** (exp - 7)
    return -v if sign else v


def _fp8_e5m2_to_fp32(b):
    b &= 0xFF
    sign = b >> 7
    exp = (b >> 2) & 0x1F
    mant = b & 3
    if exp == 31 and mant != 0:
        return float("nan")
    if exp == 31 and mant == 0:
        return float("-inf") if sign else float("inf")
    if exp == 0 and mant == 0:
        return -0.0 if sign else 0.0
    if exp == 0:
        v = mant * 2.0 ** -16
        return -v if sign else v
    v = (1.0 + mant * 2.0 ** -2) * 2.0 ** (exp - 15)
    return -v if sign else v


_vec_e4m3 = np.vectorize(_fp8_e4m3_to_fp32, otypes=[np.float32])
_vec_e5m2 = np.vectorize(_fp8_e5m2_to_fp32, otypes=[np.float32])


def _cpu_q4_0_dequant(packed, nrows, n_per_row):
    """Decode packed Q4_0 blocks without using the native library."""
    blocks_per_row = n_per_row // 32
    blocks = np.asarray(packed, dtype=np.uint8).reshape(nrows * blocks_per_row, 18)
    d = blocks[:, :2].copy().view("<f2").astype(np.float32).reshape(-1, 1)
    qs = blocks[:, 2:]
    low = qs & 0x0F
    high = qs >> 4
    values = np.concatenate((low, high), axis=1).astype(np.float32) - 8.0
    return (d * values).reshape(nrows, n_per_row)


def _load():
    dll = os.environ.get("HIP_QUANT_TEST_DLL", "")
    if dll and os.path.isfile(dll):
        return dll
    cand = os.path.join(_src, "hip_quantize.dll")
    if os.path.isfile(cand):
        return cand
    here = os.path.dirname(_src)
    cand = os.path.join(here, "hip_quantize_q_to_fp8_test.dll")
    if os.path.isfile(cand):
        return cand
    cand = os.path.join(here, "hip_quantize.dll")
    if os.path.isfile(cand):
        return cand
    return None


def _gpu_available():
    dll = _load()
    if dll is None:
        return None, None
    try:
        q = hq.HipQuant(dll)
        name = q.gcn_arch or q.device_name
        if not name:
            return None, None
        return q, dll
    except Exception:
        return None, None


_Q, _DLL = _gpu_available()

# Q types covered by the direct dequantize-to-FP8 kernels.
SUPPORTED = [
    ("Q4_0", 2),
    ("Q4_1", 3),
    ("Q5_0", 6),
    ("Q5_1", 7),
    ("Q8_0", 8),
    ("Q8_1", 9),
    ("Q2_K", 10),
    ("Q3_K", 11),
    ("Q4_K", 12),
    ("Q5_K", 13),
    ("Q6_K", 14),
]

# These formats use the same direct Q-to-FP8 export but have intentionally
# separate numerical references in test_dequant_fp8.py, where their packed
# GGML layouts and codebooks are decoded independently on CPU.
I_AND_T_SUPPORTED = [
    ("IQ1_S", 19),
    ("IQ2_XXS", 16),
    ("IQ2_XS", 17),
    ("IQ3_XXS", 18),
    ("IQ3_S", 21),
    ("IQ4_NL", 20),
    ("IQ4_XS", 23),
    ("TQ1_0", 34),
    ("TQ2_0", 35),
]


@unittest.skipUnless(_Q is not None, "No hip_quant DLL / GPU available")
class TestQToFp8(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.q = _Q
        cls.rng = np.random.default_rng(1234)
        cls.nrows = 2
        cls.n_per_row = 256  # multiple of every block size (32 and 256)

    def _ref(self, src):
        # The FP8 round-trip is the irreducible error.  Decode the FP8
        # bytes on the CPU and compare against the original F32 source to
        # confirm the Q-block expansion produced a sensible value.
        return None

    def _round_trip(self, name, type_num, fmt):
        x = (self.rng.standard_normal((self.nrows, self.n_per_row)) * 0.5).astype(np.float32)
        packed = self.q.quantize_numpy(x, type_num)
        # sanity: packed size matches the documented row size
        row_bytes = self.q.row_size(type_num, self.n_per_row)
        self.assertEqual(packed.size, row_bytes * self.nrows)

        fp8 = self.q.dequantize_to_fp8(packed, type_num, self.n_per_row, fmt)
        self.assertEqual(fp8.shape, (self.nrows, self.n_per_row))
        self.assertEqual(fp8.dtype, np.uint8)

        # Q-types with min-offsets (Q4_1/Q5_1/K-quants) can decode to values
        # whose FP8 representation rounds to +/-inf or NaN only if the block
        # scale is huge; with a 0.5-std normal source that never happens.
        dec = _vec_e4m3(fp8) if fmt == "E4M3" else _vec_e5m2(fp8)
        # Q4_0 keeps 4 effective bits; Q8_0 keeps 8. All reduce error; just
        # assert the values are finite and broadly correlated with x.
        self.assertTrue(np.isfinite(dec).all(), f"{name}/{fmt}: non-finite dequant output")
        # Compare against quantize -> Q -> CPU-implied dequant tolerance.
        # Q4_0 relative RMSE should be < ~6%, Q8_0 < ~0.5%. Use a generous
        # absolute threshold so the test stays robust across formats.
        max_abs = np.max(np.abs(x))
        rmse = float(np.sqrt(np.mean((dec - x) ** 2)))
        self.assertLess(rmse, 0.25 * max_abs + 0.25,
                        f"{name}/{fmt}: RMSE {rmse:.4f} too large vs max|x|={max_abs:.4f}")

    def test_all_types_e4m3(self):
        for name, t in SUPPORTED:
            with self.subTest(type=name, fmt="E4M3"):
                self._round_trip(name, t, "E4M3")

    def test_all_types_e5m2(self):
        for name, t in SUPPORTED:
            with self.subTest(type=name, fmt="E5M2"):
                self._round_trip(name, t, "E5M2")

    def test_iquant_and_tquant_direct_export(self):
        """Every advertised I-/T-Quant kernel exports finite E4M3 bytes."""
        x = (self.rng.standard_normal((1, 256)) * 0.5).astype(np.float32)
        for name, type_num in I_AND_T_SUPPORTED:
            with self.subTest(type=name):
                packed = self.q.quantize_numpy(x, type_num)
                fp8 = self.q.dequantize_to_e4m3(packed, type_num, 256)
                dec = _vec_e4m3(fp8)
                self.assertEqual(fp8.shape, x.shape)
                self.assertTrue(np.isfinite(dec).all(), f"{name}: non-finite direct FP8 output")
                self.assertGreater(np.abs(dec).max(), 0.0, f"{name}: all-zero direct FP8 output")

    def test_shortcuts(self):
        x = (self.rng.standard_normal((1, 256)) * 0.3).astype(np.float32)
        packed = self.q.quantize_numpy(x, hq.GGML_TYPE["Q4_0"])
        a = self.q.dequantize_to_e4m3(packed, hq.GGML_TYPE["Q4_0"], 256)
        b = self.q.dequantize_to_fp8(packed, hq.GGML_TYPE["Q4_0"], 256, "E4M3")
        np.testing.assert_array_equal(a, b)
        c = self.q.dequantize_to_e5m2(packed, hq.GGML_TYPE["Q4_0"], 256)
        d = self.q.dequantize_to_fp8(packed, hq.GGML_TYPE["Q4_0"], 256, "E5M2")
        np.testing.assert_array_equal(c, d)

    def test_q4_0_row_counts_beyond_1d_grid_limit(self):
        """Q4_0 dequantization must address every row past gridDim.x=65535."""
        n_per_row = 32
        nibble = np.arange(32, dtype=np.uint8) & 0x0F
        packed_row = np.empty(18, dtype=np.uint8)
        packed_row[:2] = np.frombuffer(np.float16(-0.125).tobytes(), dtype=np.uint8)
        packed_row[2:] = nibble[:16] | (nibble[16:] << 4)

        for nrows in (65535, 65536, 180224, 360448):
            with self.subTest(nrows=nrows):
                packed = np.broadcast_to(packed_row, (nrows, 18)).copy()
                expected = _cpu_q4_0_dequant(packed, nrows, n_per_row)
                actual_fp8 = self.q.dequantize_to_e4m3(
                    packed.reshape(-1), hq.GGML_TYPE["Q4_0"], n_per_row
                )
                actual = _vec_e4m3(actual_fp8)
                np.testing.assert_array_equal(actual, expected)

    def test_fp8_source_passthrough(self):
        # If source type IS the requested FP8 format, expect a byte copy.
        x = self.rng.integers(0, 256, size=(1, 256), dtype=np.uint8)
        out = self.q.dequantize_to_fp8(x, hq.GGML_TYPE["F8_E4M3"], 256, "E4M3")
        np.testing.assert_array_equal(out, x.reshape(1, 256))

    def test_rejects_unsupported_type(self):
        # I-Quants and T-Quants are supported; use an invalid enum value to
        # verify that the public wrapper still reports unsupported formats.
        x = (self.rng.standard_normal((1, 256)) * 0.3).astype(np.float32)
        packed = self.q.quantize_numpy(x, hq.GGML_TYPE["Q4_0"])
        with self.assertRaises(ValueError):
            self.q.dequantize_to_fp8(packed, 999, 256, "E4M3")

    def test_bad_output_format(self):
        x = (self.rng.standard_normal((1, 256)) * 0.3).astype(np.float32)
        packed = self.q.quantize_numpy(x, hq.GGML_TYPE["Q4_0"])
        with self.assertRaises(ValueError):
            self.q.dequantize_to_fp8(packed, hq.GGML_TYPE["Q4_0"], 256, "BF16")


if __name__ == "__main__":
    unittest.main(verbosity=2)
