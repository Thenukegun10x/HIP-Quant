"""GPU contract tests for native hipBLASLt MXFP4, including non-CDNA safety."""

import os
import unittest

import torch

import hip_quant.torch_api as hq


@unittest.skipUnless(torch.cuda.is_available(), "native MXFP4 contract test requires a HIP/CUDA GPU")
class TestNativeMxFp4TorchContract(unittest.TestCase):
    @staticmethod
    def _valid_operands(device):
        return (
            torch.full((16, 64), 0x22, dtype=torch.uint8, device=device),
            torch.full((16, 4), 127, dtype=torch.uint8, device=device),
            torch.full((16, 64), 0x22, dtype=torch.uint8, device=device),
            torch.full((16, 4), 127, dtype=torch.uint8, device=device),
        )

    def test_compiled_contract_is_available_on_any_architecture(self):
        contract = hq.native_mxfp4_contract()
        capability = hq.native_mxfp4_capability()

        self.assertEqual(contract["a_type"], "HIP_R_4F_E2M1")
        self.assertEqual(contract["b_type"], "HIP_R_4F_E2M1")
        self.assertEqual(contract["scale_type"], "HIP_R_8F_UE8M0")
        self.assertEqual(contract["required_arch"], "gfx950")
        self.assertIn("device_arch", capability)
        self.assertIn("available", capability)

    def test_non_gfx950_refuses_before_a_native_kernel_launch(self):
        capability = hq.native_mxfp4_capability()
        if capability["device_is_gfx950"]:
            self.skipTest("gfx950 has a separately opted-in native execution test")

        with self.assertRaisesRegex(RuntimeError, "gfx950"):
            hq.native_mxfp4_linear_forward(*self._valid_operands("cuda"))

    @unittest.skipUnless(
        os.environ.get("HIP_QUANT_TEST_NATIVE_MXFP4") == "1",
        "set HIP_QUANT_TEST_NATIVE_MXFP4=1 on a gfx950 runner to execute native MXFP4",
    )
    def test_gfx950_executes_true_e2m1_vec32_ue8m0_gemm(self):
        capability = hq.native_mxfp4_capability()
        if not capability["available"]:
            self.skipTest(str(capability["reason"]))

        # E2M1 code 0x2 is exactly +1.0 and UE8M0 127 is scale 1.  Therefore
        # every output element is the exact K=128 dot product.
        output = hq.native_mxfp4_linear_forward(*self._valid_operands("cuda"))
        torch.testing.assert_close(
            output.float(), torch.full((16, 16), 128.0, device=output.device), rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
