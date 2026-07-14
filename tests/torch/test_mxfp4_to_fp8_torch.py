"""GPU regression tests for the MXFP4 E2M1 + UE8M0 compatibility decoder."""

import unittest
import warnings

import torch

import hip_quant.torch_api as hq


@unittest.skipUnless(torch.cuda.is_available(), "MXFP4 decoder test requires a HIP/CUDA GPU")
class TestMxFp4ToFp8Torch(unittest.TestCase):
    def test_all_e2m1_codes_and_multiple_launch_blocks(self):
        # 33 rows forces two 1,024-value kernel launches. The packed layout is
        # low nibble = even logical element, high nibble = odd element.
        codes = torch.arange(32, dtype=torch.uint8) & 0x0F
        packed_row = codes[0::2] | (codes[1::2] << 4)
        packed = packed_row.repeat(33, 1).cuda()
        scales = torch.full((33, 1), 127, dtype=torch.uint8, device=packed.device)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            actual_fp8 = hq.dequantize_mxfp4_to_fp8(packed, scales, n_per_row=32)
        actual = hq.dequantize_e4m3(actual_fp8).cpu()

        one_block = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
             0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0] * 2,
            dtype=torch.float32,
        )
        expected = one_block.repeat(33, 1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_reserved_ue8m0_scale_maps_to_nan(self):
        packed = torch.full((1, 16), 0x22, dtype=torch.uint8, device="cuda")
        scales = torch.full((1, 1), 0xFF, dtype=torch.uint8, device="cuda")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            actual_fp8 = hq.dequantize_mxfp4_to_fp8(packed, scales, n_per_row=32)
        actual = hq.dequantize_e4m3(actual_fp8)
        self.assertTrue(torch.isnan(actual).all().item())


if __name__ == "__main__":
    unittest.main()
