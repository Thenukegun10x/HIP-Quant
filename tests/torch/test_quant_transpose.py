"""GPU contract tests for fused FP8 quantize + transpose kernels."""

import unittest

import torch

from hip_quant.torch_api import (
    quantize_e4m3,
    quantize_e4m3_transpose,
    quantize_e5m2,
    quantize_e5m2_transpose,
)


@unittest.skipUnless(torch.cuda.is_available(), "requires a ROCm/HIP GPU")
class TestQuantizeTranspose(unittest.TestCase):
    def _assert_matches_two_pass(self, fused, quantize, dtype):
        for rows, cols in ((1, 1), (7, 33), (33, 7), (37, 65)):
            with self.subTest(dtype=dtype, rows=rows, cols=cols):
                source = torch.randn(rows, cols, device="cuda", dtype=dtype)
                expected = quantize(source).transpose(0, 1).contiguous()
                actual = fused(source)
                self.assertEqual(actual.dtype, torch.uint8)
                self.assertEqual(tuple(actual.shape), (cols, rows))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_e4m3_matches_two_pass(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            self._assert_matches_two_pass(quantize_e4m3_transpose, quantize_e4m3, dtype)

    def test_e5m2_matches_two_pass(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            self._assert_matches_two_pass(quantize_e5m2_transpose, quantize_e5m2, dtype)
