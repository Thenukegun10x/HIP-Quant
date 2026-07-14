"""CPU-only contract tests for the MXFP4 compatibility API."""

import unittest
import warnings
from unittest import mock

import torch

import hip_quant.torch_api as hq


class _FakeExtension:
    def dequantize_mxfp4_to_fp8(self, packed_values, block_scales, n_per_row):
        nrows = packed_values.numel() // (n_per_row // 2)
        assert block_scales.numel() == nrows * (n_per_row // 32)
        return torch.zeros((nrows, n_per_row), dtype=torch.uint8)


class TestMxFp4Compatibility(unittest.TestCase):
    def setUp(self):
        self.previous_warning_state = hq._MXFP4_EMULATION_WARNING_EMITTED
        hq._MXFP4_EMULATION_WARNING_EMITTED = False

    def tearDown(self):
        hq._MXFP4_EMULATION_WARNING_EMITTED = self.previous_warning_state

    def test_decoder_warns_once_and_preserves_logical_shape(self):
        packed = torch.zeros((3, 16), dtype=torch.uint8)
        scales = torch.full((3, 1), 127, dtype=torch.uint8)

        with mock.patch.object(hq, "_load_extension", return_value=_FakeExtension()):
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                first = hq.dequantize_mxfp4_to_fp8(packed, scales, 32)
                second = hq.dequantize_mxfp4_to_fp8(packed, scales, 32)

        self.assertEqual(first.shape, (3, 32))
        self.assertEqual(second.shape, (3, 32))
        self.assertEqual(len(captured), 1)
        self.assertIs(captured[0].category, RuntimeWarning)
        self.assertIn("not use native MXFP4 instructions", str(captured[0].message))
        self.assertIn("slower speeds", str(captured[0].message))

    def test_linear_rejects_a_non_mxfp4_width_before_launch(self):
        activations = torch.zeros((1, 31))
        packed = torch.zeros((1, 16), dtype=torch.uint8)
        scales = torch.zeros((1, 1), dtype=torch.uint8)

        with self.assertRaisesRegex(ValueError, "multiple of 32"):
            hq.mxfp4_linear_forward(activations, packed, scales, n_per_row=31)


if __name__ == "__main__":
    unittest.main()
