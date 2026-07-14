"""CPU-only ABI tests for the strict native CDNA4 MXFP4 entry point."""

import unittest
from unittest import mock

import torch

import hip_quant.torch_api as hq


class _FakeNativeExtension:
    def __init__(self):
        self.calls = []

    def native_mxfp4_contract(self):
        return {
            "a_type": "HIP_R_4F_E2M1",
            "b_type": "HIP_R_4F_E2M1",
            "scale_type": "HIP_R_8F_UE8M0",
            "scale_mode": "HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0",
            "trans_a": "HIPBLAS_OP_T",
            "trans_b": "HIPBLAS_OP_N",
            "m_multiple": 16,
            "n_multiple": 16,
            "k_multiple": 128,
            "batch_count": 1,
            "epilogue": "none",
            "required_arch": "gfx950",
        }

    def native_mxfp4_capability(self):
        return {
            "compiled": True,
            "library_loaded": False,
            "device_arch": "unavailable",
            "device_is_gfx950": False,
            "available": False,
            "required_arch": "gfx950",
            "reason": "test double",
        }

    def native_mxfp4_linear_forward(
        self, a_values, a_scales, b_values, b_scales, output_dtype
    ):
        self.calls.append((a_values, a_scales, b_values, b_scales, output_dtype))
        return torch.zeros((a_values.shape[0], b_values.shape[0]), dtype=torch.bfloat16)


class TestNativeMxFp4Contract(unittest.TestCase):
    @staticmethod
    def _valid_operands():
        # M=N=16, K=128 meets the documented native hipBLASLt constraints.
        return (
            torch.zeros((16, 64), dtype=torch.uint8),
            torch.full((16, 4), 127, dtype=torch.uint8),
            torch.zeros((16, 64), dtype=torch.uint8),
            torch.full((16, 4), 127, dtype=torch.uint8),
        )

    def test_contract_is_machine_checkable_without_a_cdna_gpu(self):
        fake = _FakeNativeExtension()
        with mock.patch.object(hq, "_load_extension", return_value=fake):
            contract = hq.native_mxfp4_contract()
            capability = hq.native_mxfp4_capability()

        self.assertEqual(contract["a_type"], "HIP_R_4F_E2M1")
        self.assertEqual(contract["scale_type"], "HIP_R_8F_UE8M0")
        self.assertEqual(contract["scale_mode"], "HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0")
        self.assertEqual((contract["m_multiple"], contract["n_multiple"], contract["k_multiple"]), (16, 16, 128))
        self.assertEqual(contract["required_arch"], "gfx950")
        self.assertFalse(capability["available"])

    def test_valid_native_layout_reaches_extension_without_emulation(self):
        fake = _FakeNativeExtension()
        operands = self._valid_operands()

        with mock.patch.object(hq, "_load_extension", return_value=fake):
            output = hq.native_mxfp4_linear_forward(*operands)

        self.assertEqual(output.shape, (16, 16))
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(fake.calls[0][-1], 2)  # bfloat16 ABI code

    def test_shape_gate_rejects_invalid_native_requirements_before_launch(self):
        a_values, a_scales, b_values, b_scales = self._valid_operands()

        with self.assertRaisesRegex(ValueError, "multiples of 16"):
            hq.native_mxfp4_linear_forward(a_values[:15], a_scales[:15], b_values, b_scales)
        with self.assertRaisesRegex(ValueError, r"shape \[N, K / 32\]"):
            hq.native_mxfp4_linear_forward(a_values, a_scales, b_values, b_scales[:, :3])
        with self.assertRaisesRegex(TypeError, "torch.uint8"):
            hq.native_mxfp4_linear_forward(
                a_values.float(), a_scales, b_values, b_scales
            )

    def test_unsupported_output_dtype_does_not_fall_back(self):
        fake = _FakeNativeExtension()
        operands = self._valid_operands()
        with mock.patch.object(hq, "_load_extension", return_value=fake):
            with self.assertRaisesRegex(TypeError, "output_dtype"):
                hq.native_mxfp4_linear_forward(*operands, output_dtype=torch.float64)
        self.assertEqual(fake.calls, [])


if __name__ == "__main__":
    unittest.main()
