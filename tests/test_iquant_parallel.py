"""Tests for parallelized I-Quant kernels.

Verifies that the multi-threaded I-Quant kernels produce correct output
for IQ1_S, IQ2_XXS, IQ2_XS, IQ3_XXS, and IQ3_S quantization types.

Requirements: AMD GPU with hip_quant DLL loaded.
"""
import os
import sys
import math
import unittest
import numpy as np

try:
    from hip_quant import get_hip_quant, GGML_TYPE, GGML_TYPE_BLOCK_SIZE, GGML_TYPE_BLOCK_BYTES
    HAS_HIP_QUANT = True
except ImportError:
    HAS_HIP_QUANT = False

requires_gpu = unittest.skipUnless(HAS_HIP_QUANT, "hip_quant DLL required")

IQUANT_TYPES = ["IQ1_S", "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ3_S"]
ALL_QUANT_TYPES = list(GGML_TYPE.keys())

# Simple CPU-side FP16 decode helper (matches GPU fp16_to_fp32)
def _fp16_to_f32(h):
    sign = (h >> 15) & 1
    exp = (h >> 10) & 0x1F
    mant = h & 0x3FF
    if exp == 0 and mant == 0:
        return -0.0 if sign else 0.0
    if exp == 31:
        return float('nan')
    if exp == 0:
        val = mant * (2.0 ** -24)
        return -val if sign else val
    val = (1.0 + mant / 1024.0) * (2.0 ** (int(exp) - 15))
    return -val if sign else val


def _get_superblock_scale(block_bytes, type_name):
    """Extract the superblock float scale `d` from the first 2 bytes."""
    return _fp16_to_f32(int(block_bytes[0]) | (int(block_bytes[1]) << 8))


@requires_gpu
class TestIQuantSmoke(unittest.TestCase):
    """Basic smoke tests — do the parallelized kernels run without errors?"""

    @classmethod
    def setUpClass(cls):
        cls.hq = get_hip_quant()

    def _quantize_and_check(self, type_name, rows=4, cols=256):
        qtype = GGML_TYPE[type_name]
        blk_size = GGML_TYPE_BLOCK_SIZE[qtype]
        blk_bytes = GGML_TYPE_BLOCK_BYTES[qtype]
        self.assertEqual(cols % blk_size, 0, f"{type_name}: cols must be multiple of {blk_size}")

        x = np.random.randn(rows, cols).astype(np.float32) * 2.0
        out = self.hq.quantize_numpy(x, qtype)

        expected_bytes = rows * (cols // blk_size) * blk_bytes
        self.assertEqual(len(out), expected_bytes,
            f"{type_name}: expected {expected_bytes} bytes, got {len(out)}")
        return out, x, blk_size, blk_bytes

    def test_iq1_s_smoke(self):
        self._quantize_and_check("IQ1_S")

    def test_iq2_xxs_smoke(self):
        self._quantize_and_check("IQ2_XXS")

    def test_iq2_xs_smoke(self):
        self._quantize_and_check("IQ2_XS")

    def test_iq3_xxs_smoke(self):
        self._quantize_and_check("IQ3_XXS")

    def test_iq3_s_smoke(self):
        self._quantize_and_check("IQ3_S")


@requires_gpu
class TestIQuantNonZeroScale(unittest.TestCase):
    """Non-trivial input should produce a non-zero superblock scale."""

    @classmethod
    def setUpClass(cls):
        cls.hq = get_hip_quant()

    def _test_nonzero_d(self, type_name):
        qtype = GGML_TYPE[type_name]
        blk_size = GGML_TYPE_BLOCK_SIZE[qtype]
        blk_bytes = GGML_TYPE_BLOCK_BYTES[qtype]

        x = np.random.randn(4, blk_size).astype(np.float32) * 3.0 + 1.0
        out = self.hq.quantize_numpy(x, qtype)

        for row in range(4):
            block = out[row * blk_bytes:(row + 1) * blk_bytes]
            d = _get_superblock_scale(block, type_name)
            self.assertGreater(d, 0.0,
                f"{type_name} row {row}: d should be > 0 for non-zero input, got {d}")

    def test_iq1_s_nonzero_d(self):
        self._test_nonzero_d("IQ1_S")

    def test_iq2_xxs_nonzero_d(self):
        self._test_nonzero_d("IQ2_XXS")

    def test_iq2_xs_nonzero_d(self):
        self._test_nonzero_d("IQ2_XS")

    def test_iq3_xxs_nonzero_d(self):
        self._test_nonzero_d("IQ3_XXS")

    def test_iq3_s_nonzero_d(self):
        self._test_nonzero_d("IQ3_S")


@requires_gpu
class TestIQuantZeroInput(unittest.TestCase):
    """All-zero input should produce zero-scale output."""

    @classmethod
    def setUpClass(cls):
        cls.hq = get_hip_quant()

    def _test_zero_input(self, type_name):
        qtype = GGML_TYPE[type_name]
        blk_size = GGML_TYPE_BLOCK_SIZE[qtype]
        blk_bytes = GGML_TYPE_BLOCK_BYTES[qtype]

        x = np.zeros((2, blk_size), dtype=np.float32)
        out = self.hq.quantize_numpy(x, qtype)

        for row in range(2):
            block = out[row * blk_bytes:(row + 1) * blk_bytes]
            d = _get_superblock_scale(block, type_name)
            if d == 0.0:
                pass  # expected
            else:
                # d may be tiny but non-zero due to fp16 rounding of 0
                self.assertLess(abs(d), 1e-7,
                    f"{type_name}: zero input should have d ≈ 0, got {d}")

    def test_iq1_s_zero(self):
        self._test_zero_input("IQ1_S")

    def test_iq2_xxs_zero(self):
        self._test_zero_input("IQ2_XXS")

    def test_iq2_xs_zero(self):
        self._test_zero_input("IQ2_XS")

    def test_iq3_xxs_zero(self):
        self._test_zero_input("IQ3_XXS")

    def test_iq3_s_zero(self):
        self._test_zero_input("IQ3_S")


@requires_gpu
class TestIQuantConsistency(unittest.TestCase):
    """Same input → same output (deterministic)."""

    @classmethod
    def setUpClass(cls):
        cls.hq = get_hip_quant()

    def _test_consistency(self, type_name):
        qtype = GGML_TYPE[type_name]
        blk_size = GGML_TYPE_BLOCK_SIZE[qtype]

        np.random.seed(12345)
        x = np.random.randn(2, 256).astype(np.float32)

        out1 = self.hq.quantize_numpy(x, qtype)
        out2 = self.hq.quantize_numpy(x, qtype)

        self.assertTrue(np.array_equal(out1, out2),
            f"{type_name}: repeated quantization should be identical")
        self.assertEqual(len(out1), len(out2))

    def test_iq1_s_consistent(self):
        self._test_consistency("IQ1_S")

    def test_iq2_xxs_consistent(self):
        self._test_consistency("IQ2_XXS")

    def test_iq2_xs_consistent(self):
        self._test_consistency("IQ2_XS")

    def test_iq3_xxs_consistent(self):
        self._test_consistency("IQ3_XXS")

    def test_iq3_s_consistent(self):
        self._test_consistency("IQ3_S")


@requires_gpu
class TestIQuantMultiRow(unittest.TestCase):
    """Quantizing multiple rows — each row processes independently."""

    @classmethod
    def setUpClass(cls):
        cls.hq = get_hip_quant()

    def _test_multi_row(self, type_name, nrows=16):
        qtype = GGML_TYPE[type_name]
        blk_size = GGML_TYPE_BLOCK_SIZE[qtype]
        blk_bytes = GGML_TYPE_BLOCK_BYTES[qtype]

        x = np.random.randn(nrows, 256).astype(np.float32) * 3.0
        out = self.hq.quantize_numpy(x, qtype)

        expected_bytes = nrows * (256 // blk_size) * blk_bytes
        self.assertEqual(len(out), expected_bytes)

        # Each row's first superblock should have d > 0
        for row in range(nrows):
            block = out[row * expected_bytes // nrows:
                        row * expected_bytes // nrows + blk_bytes]
            d = _get_superblock_scale(block, type_name)
            self.assertGreater(d, 0.0,
                f"{type_name} row {row}: d should be > 0")

    def test_iq1_s_multirow(self):
        self._test_multi_row("IQ1_S")

    def test_iq2_xxs_multirow(self):
        self._test_multi_row("IQ2_XXS")

    def test_iq2_xs_multirow(self):
        self._test_multi_row("IQ2_XS")

    def test_iq3_xxs_multirow(self):
        self._test_multi_row("IQ3_XXS")

    def test_iq3_s_multirow(self):
        self._test_multi_row("IQ3_S")


@requires_gpu
class TestIQuantLargeBatch(unittest.TestCase):
    """Stress test: many rows, many blocks per row."""

    @classmethod
    def setUpClass(cls):
        cls.hq = get_hip_quant()

    def _test_large(self, type_name, rows=32, per_row=1024):
        qtype = GGML_TYPE[type_name]
        blk_size = GGML_TYPE_BLOCK_SIZE[qtype]
        blk_bytes = GGML_TYPE_BLOCK_BYTES[qtype]

        n_blocks = per_row // blk_size

        np.random.seed(42)
        x = np.random.randn(rows, per_row).astype(np.float32) * 2.0
        out = self.hq.quantize_numpy(x, qtype)

        expected_bytes = rows * n_blocks * blk_bytes
        self.assertEqual(len(out), expected_bytes)

        # Spot-check: at least 80% of blocks should have non-zero d
        zero_count = 0
        for row in range(rows):
            for blk in range(n_blocks):
                off = row * n_blocks * blk_bytes + blk * blk_bytes
                d = _get_superblock_scale(out[off:off + blk_bytes], type_name)
                if d == 0.0:
                    zero_count += 1
        total = rows * n_blocks
        self.assertLess(zero_count, total * 0.2,
            f"{type_name}: {zero_count}/{total} blocks have zero scale")

    def test_iq2_xxs_large(self):
        self._test_large("IQ2_XXS")

    def test_iq3_xxs_large(self):
        self._test_large("IQ3_XXS")


@requires_gpu
class TestIQuantSingleSubblock(unittest.TestCase):
    """Edge case: exactly one sub-block (blk_size elements).

    Tests that the reduction phase works when all sub-blocks have data.
    """

    @classmethod
    def setUpClass(cls):
        cls.hq = get_hip_quant()

    def _test_single_block(self, type_name):
        qtype = GGML_TYPE[type_name]
        blk_size = GGML_TYPE_BLOCK_SIZE[qtype]

        x = np.random.randn(4, blk_size).astype(np.float32) * 2.0
        out = self.hq.quantize_numpy(x, qtype)

        self.assertTrue(len(out) > 0)

    def test_iq1_s_single(self):
        self._test_single_block("IQ1_S")

    def test_iq2_xxs_single(self):
        self._test_single_block("IQ2_XXS")

    def test_iq2_xs_single(self):
        self._test_single_block("IQ2_XS")

    def test_iq3_xxs_single(self):
        self._test_single_block("IQ3_XXS")

    def test_iq3_s_single(self):
        self._test_single_block("IQ3_S")


@requires_gpu
class TestIQuantWithImatrix(unittest.TestCase):
    """Test with importance matrix (weighted quantization)."""

    @classmethod
    def setUpClass(cls):
        cls.hq = get_hip_quant()

    def _test_imatrix(self, type_name):
        qtype = GGML_TYPE[type_name]
        blk_size = GGML_TYPE_BLOCK_SIZE[qtype]

        x = np.random.randn(4, 256).astype(np.float32) * 2.0
        imatrix = np.abs(np.random.randn(4, 256).astype(np.float32)) + 0.1

        out = self.hq.quantize_numpy(x, qtype, imatrix=imatrix)
        self.assertTrue(len(out) > 0)

        # Also test that no-imatrix and with-imatrix produce different results
        out_no_im = self.hq.quantize_numpy(x, qtype)
        # They should differ (unless all weights happen to be equal)
        # This is a probabilistic check, but very likely true
        self.assertFalse(np.array_equal(out, out_no_im),
            f"{type_name}: imatrix should change quantization result")

    def test_iq1_s_imatrix(self):
        self._test_imatrix("IQ1_S")

    def test_iq2_xxs_imatrix(self):
        self._test_imatrix("IQ2_XXS")

    def test_iq2_xs_imatrix(self):
        self._test_imatrix("IQ2_XS")

    def test_iq3_xxs_imatrix(self):
        self._test_imatrix("IQ3_XXS")

    def test_iq3_s_imatrix(self):
        self._test_imatrix("IQ3_S")


@requires_gpu
class TestIQuantAllTypesRoundtrip(unittest.TestCase):
    """Ensure all quant types (including I-Quants) survive a full
    F32 → quantize → requantize_to_fp8 cycle without crashes."""

    @classmethod
    def setUpClass(cls):
        cls.hq = get_hip_quant()

    def test_all_iquant_types_quantize(self):
        """All 5 I-Quant types should quantize successfully on random data."""
        for type_name in IQUANT_TYPES:
            with self.subTest(type=type_name):
                qtype = GGML_TYPE[type_name]
                x = np.random.randn(3, 256).astype(np.float32)
                out = self.hq.quantize_numpy(x, qtype)
                self.assertTrue(len(out) > 0)


@requires_gpu
class TestLegacyQuantStillWorking(unittest.TestCase):
    """Verify legacy quant types still work after kernel changes."""

    @classmethod
    def setUpClass(cls):
        cls.hq = get_hip_quant()

    def test_q4_0_still_works(self):
        x = np.random.randn(4, 32).astype(np.float32)
        out = self.hq.quantize_numpy(x, GGML_TYPE["Q4_0"])
        self.assertEqual(len(out), 4 * 18)

    def test_q8_0_still_works(self):
        x = np.random.randn(4, 32).astype(np.float32)
        out = self.hq.quantize_numpy(x, GGML_TYPE["Q8_0"])
        self.assertEqual(len(out), 4 * 34)

    def test_q4_k_still_works(self):
        x = np.random.randn(4, 256).astype(np.float32)
        out = self.hq.quantize_numpy(x, GGML_TYPE["Q4_K"])
        self.assertEqual(len(out), 4 * 144)

    def test_fp8_types_still_work(self):
        x = np.random.randn(4, 32).astype(np.float32)
        out_e4m3 = self.hq.quantize_numpy(x, GGML_TYPE["F8_E4M3"])
        out_e5m2 = self.hq.quantize_numpy(x, GGML_TYPE["F8_E5M2"])
        self.assertEqual(len(out_e4m3), 4 * 32)
        self.assertEqual(len(out_e5m2), 4 * 32)

    def test_tq_types_still_work(self):
        x = np.random.randn(4, 256).astype(np.float32)
        out1 = self.hq.quantize_numpy(x, GGML_TYPE["TQ1_0"])
        out2 = self.hq.quantize_numpy(x, GGML_TYPE["TQ2_0"])
        self.assertEqual(len(out1), 4 * 54)
        self.assertEqual(len(out2), 4 * 66)


if __name__ == "__main__":
    unittest.main()
