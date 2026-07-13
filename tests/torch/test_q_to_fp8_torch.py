import unittest
import torch
import hip_quant.torch_api as hq
import hip_quant as hq_base
import numpy as np

class TestQToFp8Torch(unittest.TestCase):
    def test_q_to_fp8_e4m3(self):
        # We need some dummy Q4_0 packed bytes
        rng = np.random.default_rng(1234)
        x = (rng.standard_normal((2, 256)) * 0.5).astype(np.float32)
        q = hq_base.get_hip_quant()
        packed_np = q.quantize_numpy(x, hq_base.GGML_TYPE["Q4_0"])
        
        # Move packed bytes to torch GPU tensor
        packed_torch = torch.from_numpy(packed_np).cuda()
        
        # Call new torch_api function
        fp8_torch = hq.dequantize_q_to_fp8(packed_torch, hq_base.GGML_TYPE["Q4_0"], 256, e5m2=False)
        
        # Verify shape and type
        self.assertEqual(fp8_torch.shape, (2, 256))
        self.assertEqual(fp8_torch.dtype, torch.uint8)
        self.assertTrue(fp8_torch.is_cuda)
        
        # Verify it matches the DLL ctypes path
        fp8_np = q.dequantize_to_fp8(packed_np, hq_base.GGML_TYPE["Q4_0"], 256, "E4M3")
        np.testing.assert_array_equal(fp8_torch.cpu().numpy(), fp8_np)

    def test_q_to_fp8_large_row_counts(self):
        """The PyTorch binding must accept rows beyond its old 65535 check."""
        n_per_row = 32
        nibble = np.arange(32, dtype=np.uint8) & 0x0F
        packed_row = np.empty(18, dtype=np.uint8)
        packed_row[:2] = np.frombuffer(np.float16(-0.125).tobytes(), dtype=np.uint8)
        packed_row[2:] = nibble[:16] | (nibble[16:] << 4)
        q = hq_base.get_hip_quant()

        for nrows in (65535, 65536, 180224, 360448):
            with self.subTest(nrows=nrows):
                packed = np.broadcast_to(packed_row, (nrows, 18)).copy()
                packed_torch = torch.from_numpy(packed.reshape(-1)).cuda()
                actual = hq.dequantize_q_to_fp8(
                    packed_torch, hq_base.GGML_TYPE["Q4_0"], n_per_row, e5m2=False
                )
                expected = q.dequantize_to_e4m3(
                    packed.reshape(-1), hq_base.GGML_TYPE["Q4_0"], n_per_row
                )
                self.assertEqual(actual.shape, (nrows, n_per_row))
                np.testing.assert_array_equal(actual.cpu().numpy(), expected)

if __name__ == "__main__":
    unittest.main()
