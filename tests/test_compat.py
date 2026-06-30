"""
tests/test_compat.py
====================
Tests for the CDNA compatibility checker and CPU reference emulator.
Runs on CPU — no GPU required.
"""

import os
import sys
import numpy as np
import unittest

# Ensure the parent directory is on the path so we can import hip_quant
_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)

from hip_quant.device_info import probe_device, report, DeviceProperties
from hip_quant.cdna_compat import (
    cpu_reference_quantize,
    arch_supports_feature,
    get_build_archs,
    build_config_for_arch,
    suggest_emulation,
)


class TestProbeDevice(unittest.TestCase):

    def test_probe_no_dll(self):
        dev = probe_device(dll_path="nonexistent.dll")
        self.assertIsInstance(dev, DeviceProperties)
        self.assertFalse(dev.dll_loaded)

    def test_report_returns_string(self):
        dev = probe_device(dll_path="nonexistent.dll")
        out = report(dev)
        self.assertIsInstance(out, str)
        self.assertIn("GPU not available", out)


class TestCPUReferenceQuantize(unittest.TestCase):

    def test_q4_0_single_block(self):
        arr = np.random.randn(32).astype(np.float32)
        out = cpu_reference_quantize(arr, "Q4_0")
        # Q4_0 block: 2 byte d + 16 byte packed qs = 18 bytes
        self.assertEqual(out.shape, (18,))
        self.assertEqual(out.dtype, np.uint8)

    def test_q8_0_single_block(self):
        arr = np.random.randn(32).astype(np.float32)
        out = cpu_reference_quantize(arr, "Q8_0")
        # Q8_0 block: 2 byte d + 32 byte qs = 34 bytes
        self.assertEqual(out.shape, (34,))
        self.assertEqual(out.dtype, np.uint8)

    def test_q8_0_known_values(self):
        arr = np.array([[1.0, -1.0] + [0.0] * 30], dtype=np.float32)
        out = cpu_reference_quantize(arr, "Q8_0")
        self.assertEqual(out[2], 127)
        self.assertEqual(out[3], 129)  # int8 -127 encoded as uint8

    def test_q4_0_multi_row(self):
        arr = np.random.randn(2, 64).astype(np.float32)
        out = cpu_reference_quantize(arr, "Q4_0")
        # 2 rows * 2 blocks/row * 18 bytes = 72 bytes
        self.assertEqual(out.shape, (72,))
        self.assertEqual(out.dtype, np.uint8)

    def test_q4_0_known_values(self):
        arr = np.array([[1.0, -1.0] + [0.0] * 30], dtype=np.float32)
        out = cpu_reference_quantize(arr, "Q4_0")
        # d = max/8 = 1.0/8 = 0.125 -> fp16 0x1800 (no sign), or
        # d = max/-8 = -0.125 -> fp16 0xAC00 (signed)
        # Either way, the upper byte should be non-zero
        self.assertGreater(out[1], 0)

    def test_raises_on_unknown_type(self):
        with self.assertRaises(ValueError):
            cpu_reference_quantize(np.zeros(32), "NOT_A_TYPE")


class TestArchFeatures(unittest.TestCase):

    def test_rdna4_wmma(self):
        self.assertTrue(arch_supports_feature("gfx1201", "wmma"))

    def test_cdna3_no_wmma(self):
        self.assertFalse(arch_supports_feature("gfx942", "wmma"))

    def test_cdna3_mfma(self):
        self.assertTrue(arch_supports_feature("gfx942", "mfma"))

    def test_unknown_arch(self):
        self.assertFalse(arch_supports_feature("gfx000", "wmma"))


class TestBuildConfig(unittest.TestCase):

    def test_rdna4_config(self):
        cfg = build_config_for_arch("rdna4")
        self.assertIn("gfx1200", cfg["archs"])
        self.assertIn("gfx1201", cfg["archs"])

    def test_cdna_config(self):
        cfg = build_config_for_arch("cdna")
        self.assertTrue(any(a.startswith("gfx9") for a in cfg["archs"]))

    def test_cdna3_config(self):
        cfg = build_config_for_arch("cdna3")
        self.assertIn("gfx942", cfg["archs"])

    def test_all_config(self):
        cfg = build_config_for_arch("all")
        self.assertIn("gfx1200", cfg["archs"])
        self.assertIn("gfx942", cfg["archs"])
        self.assertIn("gfx1100", cfg["archs"])

    def test_config_has_note(self):
        cfg = build_config_for_arch("rdna4")
        self.assertIn("note", cfg)
        self.assertIsInstance(cfg["note"], str)


class TestGetBuildArchs(unittest.TestCase):

    def test_build_archs_all(self):
        archs = get_build_archs("all")
        self.assertGreaterEqual(len(archs), 8)

    def test_build_archs_cdna(self):
        archs = get_build_archs("cdna")
        for a in archs:
            self.assertTrue(a.startswith("gfx9"))

    def test_build_archs_rdna4(self):
        archs = get_build_archs("rdna4")
        self.assertEqual(archs, ["gfx1200", "gfx1201"])

    def test_build_archs_rdna3(self):
        archs = get_build_archs("rdna3")
        self.assertTrue(all(a.startswith("gfx11") for a in archs))


class TestSuggestEmulation(unittest.TestCase):

    def test_suggest_returns_string(self):
        out = suggest_emulation()
        self.assertIsInstance(out, str)
        self.assertIn("CDNA", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
