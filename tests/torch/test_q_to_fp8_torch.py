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

if __name__ == "__main__":
    unittest.main()
