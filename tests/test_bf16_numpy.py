"""CPU tests for the loader-side vectorized BF16 decoder."""

import unittest

import numpy as np

from hip_quant import bf16_to_fp32


class TestBf16ToFp32(unittest.TestCase):
    def test_decodes_normal_values_and_preserves_shape(self):
        # 1.0, -2.5, +0.0, and +infinity in IEEE bfloat16 bit form.
        raw = np.array([[0x3F80, 0xC020], [0x0000, 0x7F80]], dtype=np.uint16)

        actual = bf16_to_fp32(raw)

        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(actual.shape, raw.shape)
        np.testing.assert_array_equal(actual, np.array([[1.0, -2.5], [0.0, np.inf]], dtype=np.float32))

    def test_bytes_input_and_requested_shape(self):
        raw = np.array([0x3F80, 0xBF80, 0x4000, 0xC000], dtype="<u2")

        actual = bf16_to_fp32(raw.tobytes(), shape=(2, 2))

        np.testing.assert_array_equal(
            actual, np.array([[1.0, -1.0], [2.0, -2.0]], dtype=np.float32)
        )

    def test_preserves_signed_zero_and_nan_bit_class(self):
        raw = np.array([0x8000, 0x7FC1], dtype=np.uint16)

        actual = bf16_to_fp32(raw)

        self.assertTrue(np.signbit(actual[0]))
        self.assertEqual(actual[0], 0.0)
        self.assertTrue(np.isnan(actual[1]))
        self.assertEqual(actual.view(np.uint32)[1], np.uint32(0x7FC10000))

    def test_rejects_non_raw_bf16_input_and_bad_shape(self):
        with self.assertRaisesRegex(TypeError, "uint16"):
            bf16_to_fp32(np.array([1.0], dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "requires 3 values"):
            bf16_to_fp32(np.array([0x3F80, 0x4000], dtype=np.uint16), shape=(3,))
        with self.assertRaisesRegex(ValueError, "multiple of 2"):
            bf16_to_fp32(b"\x80")


if __name__ == "__main__":
    unittest.main()
