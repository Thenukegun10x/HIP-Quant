"""WMMA Stability Diagnostic Tests for hip_quant.

Tests the gfx12 FP8 WMMA intrinsic across multiple dimensions and patterns.
Requires a ROCm 7.2+ gfx12/RDNA4 GPU.

Usage:
    # Quick diag (safe, small tensors)
    python -m pytest tests\test_wmma_diag.py -v

    # With stress (larger tensors, more iterations)
    $env:HIP_QUANT_WMMA_STRESS='1'
    python -m pytest tests\test_wmma_diag.py -v
"""
import os
import sys
import math
import unittest
import numpy as np

try:
    from hip_quant import get_hip_quant, GGML_TYPE
    HAS_HIP_QUANT = True
except ImportError:
    HAS_HIP_QUANT = False

STRESS_MODE = os.environ.get("HIP_QUANT_WMMA_STRESS", "").lower() in (
    "1", "true", "yes", "on"
)


def requires_wmma():
    return unittest.skipUnless(
        HAS_HIP_QUANT,
        "hip_quant DLL required"
    )


def _round_up_32(value):
    return (value + 31) & ~31


def _decode_e4m3(fp8):
    """Decode raw OCP E4M3FN bytes to the FP32 operands consumed by WMMA."""
    values = np.asarray(fp8, dtype=np.uint8)
    sign = np.where(values & 0x80, -1.0, 1.0).astype(np.float32)
    exponent = ((values >> 3) & 0x0F).astype(np.int32)
    mantissa = (values & 0x07).astype(np.float32)
    normal = (1.0 + mantissa / 8.0) * np.exp2(exponent - 7)
    subnormal = mantissa * np.float32(2.0 ** -9)
    decoded = sign * np.where(exponent == 0, subnormal, normal)
    decoded[(exponent == 15) & (mantissa == 7)] = np.nan
    return decoded.astype(np.float32, copy=False)


def _quantize_wmma_operands(hq, a_f32, b_f32):
    """Quantize WMMA operands with legal FP8 rows and return their FP32 meaning.

    GGML's FP8 quantizer operates on blocks of 32 values.  WMMA accepts a
    logical K that is not a multiple of either 16 or 32, provided its leading
    dimensions have enough storage for the final tile.  Padding is therefore
    part of the input layout, not part of the GEMM being checked.
    """
    a_f32 = np.asarray(a_f32, dtype=np.float32)
    b_f32 = np.asarray(b_f32, dtype=np.float32)
    M, K = a_f32.shape
    if b_f32.shape[0] != K:
        raise ValueError("A and B must agree on K")
    N = b_f32.shape[1]
    lda = _round_up_32(K)
    ldb = _round_up_32(N)

    a_padded = np.zeros((M, lda), dtype=np.float32)
    b_padded = np.zeros((K, ldb), dtype=np.float32)
    a_padded[:, :K] = a_f32
    b_padded[:, :N] = b_f32

    a_fp8 = hq.quantize_numpy(a_padded, GGML_TYPE["F8_E4M3"]).reshape(M, lda)
    b_fp8 = hq.quantize_numpy(b_padded, GGML_TYPE["F8_E4M3"]).reshape(K, ldb)
    return a_fp8, b_fp8, _decode_e4m3(a_fp8[:, :K]), _decode_e4m3(b_fp8[:, :N]), lda, ldb


def _assert_wmma_matches_fp8(testcase, hq, a_f32, b_f32):
    """Run WMMA and compare it against the exact decoded FP8 input tensors."""
    M, K = a_f32.shape
    N = b_f32.shape[1]
    a_fp8, b_fp8, a_ref, b_ref, lda, ldb = _quantize_wmma_operands(hq, a_f32, b_f32)
    actual = hq.fp8_gemm_test_wmma(a_fp8, b_fp8, M, N, K, lda=lda, ldb=ldb)
    testcase.assertIsNotNone(actual, "WMMA GEMM returned None")
    expected = a_ref @ b_ref
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=1e-3)
    return actual


def _require_wmma_runtime(cls):
    cls.hq = get_hip_quant()
    arch = cls.hq.gcn_arch
    if not arch.startswith("gfx12"):
        raise unittest.SkipTest(f"Not a gfx12 device: {arch}")
    runtime = cls.hq.hip_runtime_version
    if runtime and runtime < 70200000:
        raise unittest.SkipTest(f"ROCm {runtime} < 7.2 required for WMMA")


@requires_wmma()
class TestWMMADiagnostics(unittest.TestCase):
    """Basic WMMA diagnostic tests (DLL-level)."""

    @classmethod
    def setUpClass(cls):
        _require_wmma_runtime(cls)

    def test_01_diag_kernel(self):
        """Run the built-in WMMA diagnostic kernel (4 tests)."""
        result = self.hq.wmma_diag()
        if result is None:
            self.skipTest("wmma_diag not available in DLL")
        if result["gated"]:
            self.skipTest(f"WMMA gated: {result['gate_reason']}")

        self.assertTrue(result["test1_ok"],
            f"Test1 known-values: output={result['test1_value']} (expected ~{64 * 1000})")
        self.assertTrue(result["test2_ok"],
            f"Test2 repeated: flag={result['test2_repeated_ok']} (expected 0)")
        self.assertTrue(result["test3_ok"],
            f"Test3 LDS-staged: output={result['test3_lds_value']}")
        self.assertEqual(result["test4_multiwave"], 16,
            f"Test4 multi-wave: {result['test4_multiwave']}/16 waves OK")
        self.assertTrue(result["ok"], "All four diag tests should pass")

    def test_02_small_gemm_accuracy(self):
        """Verify WMMA GEMM produces numerically correct output at M=N=K=16."""
        M = N = K = 16
        np.random.seed(42)
        A_f32 = np.random.randn(M, K).astype(np.float32) * 0.5
        B_f32 = np.random.randn(K, N).astype(np.float32) * 0.5
        _assert_wmma_matches_fp8(self, self.hq, A_f32, B_f32)

    def test_03_nonzero_output_at_all_sizes(self):
        """Ensure WMMA produces non-zero output across multiple sizes."""
        sizes = [(16, 16, 16), (32, 32, 32), (64, 64, 64)]
        for M, N, K in sizes:
            with self.subTest(f"{M}x{N}x{K}"):
                A_f32 = np.ones((M, K), dtype=np.float32)
                B_f32 = np.ones((K, N), dtype=np.float32)
                C = _assert_wmma_matches_fp8(self, self.hq, A_f32, B_f32)
                self.assertGreater(np.abs(C).max(), 0.0)

    def test_04_nonsquare_shapes(self):
        """Test non-square matrix shapes."""
        for M, N, K in [(32, 64, 128), (64, 32, 128), (128, 64, 32)]:
            with self.subTest(f"{M}x{N}x{K}"):
                A_f32 = np.random.randn(M, K).astype(np.float32) * 0.3
                B_f32 = np.random.randn(K, N).astype(np.float32) * 0.3
                _assert_wmma_matches_fp8(self, self.hq, A_f32, B_f32)

    def test_05_repeated_stress(self):
        """Run WMMA GEMM 50 times rapidly to detect intermittent hangs."""
        M = N = K = 64
        A_f32 = np.random.randn(M, K).astype(np.float32) * 0.3
        B_f32 = np.random.randn(K, N).astype(np.float32) * 0.3
        n_iters = 200 if STRESS_MODE else 50
        for i in range(n_iters):
            C = _assert_wmma_matches_fp8(self, self.hq, A_f32, B_f32)
            self.assertTrue(np.all(np.isfinite(C)), f"Non-finite WMMA output at iter {i}")
            # Tiny perturbation to prevent caching
            noise = np.random.randn(K, N).astype(np.float32) * 1e-6
            B_f32 = B_f32 + noise


@requires_wmma()
class TestWMMAMediumSizes(unittest.TestCase):
    """Medium-size WMMA GEMM tests."""

    @classmethod
    def setUpClass(cls):
        _require_wmma_runtime(cls)

    def test_128x128(self):
        M = N = K = 128
        A_f32 = np.random.randn(M, K).astype(np.float32) * 0.5
        B_f32 = np.random.randn(K, N).astype(np.float32) * 0.5
        C = _assert_wmma_matches_fp8(self, self.hq, A_f32, B_f32)
        self.assertGreater(np.abs(C).max(), 0.0)

    def test_256x256(self):
        M = N = K = 256
        A_f32 = np.random.randn(M, K).astype(np.float32) * 0.5
        B_f32 = np.random.randn(K, N).astype(np.float32) * 0.5
        C = _assert_wmma_matches_fp8(self, self.hq, A_f32, B_f32)
        self.assertGreater(np.abs(C).max(), 0.0)


@requires_wmma()
class TestWMMALargeStress(unittest.TestCase):
    """Large-size WMMA stress tests — enabled only in STRESS_MODE."""

    @classmethod
    def setUpClass(cls):
        if not STRESS_MODE:
            raise unittest.SkipTest("Set HIP_QUANT_WMMA_STRESS=1 for large stress tests")
        _require_wmma_runtime(cls)

    def test_512x512(self):
        M = N = K = 512
        A_f32 = np.random.randn(M, K).astype(np.float32) * 0.5
        B_f32 = np.random.randn(K, N).astype(np.float32) * 0.5
        C = _assert_wmma_matches_fp8(self, self.hq, A_f32, B_f32)
        self.assertGreater(np.abs(C).max(), 0.0)

    def test_1024x1024(self):
        M = N = K = 1024
        A_f32 = np.random.randn(M, K).astype(np.float32) * 0.5
        B_f32 = np.random.randn(K, N).astype(np.float32) * 0.5
        C = _assert_wmma_matches_fp8(self, self.hq, A_f32, B_f32)
        self.assertGreater(np.abs(C).max(), 0.0)

    def test_repeated_large(self):
        """Run large GEMM 10 times to catch transient issues."""
        M = N = K = 256
        A_f32 = np.random.randn(M, K).astype(np.float32) * 0.3
        for i in range(10):
            B_f32 = np.random.randn(K, N).astype(np.float32) * 0.3
            C = _assert_wmma_matches_fp8(self, self.hq, A_f32, B_f32)
            self.assertGreater(np.abs(C).max(), 0.0,
                f"Iter {i}: WMMA produced zero output")


@requires_wmma()
class TestWMMABoundaryConditions(unittest.TestCase):
    """Boundary condition tests for WMMA kernels."""

    @classmethod
    def setUpClass(cls):
        _require_wmma_runtime(cls)

    def test_edge_values(self):
        """Test WMMA with finite E4M3 boundary values."""
        M = N = K = 16
        A_f32 = np.array([[0.0, 448.0, -448.0, 1e-6, -1e-6,
                           224.0, -224.0, 56.0,
                           42.0, -42.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]],
                         dtype=np.float32)
        A_f32 = np.tile(A_f32, (16, 1))
        B_f32 = np.ones((K, N), dtype=np.float32)

        C = _assert_wmma_matches_fp8(self, self.hq, A_f32, B_f32)
        self.assertTrue(np.all(np.isfinite(C)),
            "WMMA output contains NaN or Inf")

    def test_uneven_K(self):
        """Test with K not a multiple of 16 (should work via padding)."""
        M = N = 32
        for K in [17, 31, 33, 47, 63]:
            with self.subTest(f"K={K}"):
                A_f32 = np.random.randn(M, K).astype(np.float32) * 0.3
                B_f32 = np.random.randn(K, N).astype(np.float32) * 0.3
                C = _assert_wmma_matches_fp8(self, self.hq, A_f32, B_f32)
                self.assertTrue(np.all(np.isfinite(C)))


@requires_wmma()
class TestWMMAGating(unittest.TestCase):
    """Test that WMMA gating works correctly."""

    @classmethod
    def setUpClass(cls):
        _require_wmma_runtime(cls)

    def test_disable_env_var(self):
        """HIP_QUANT_DISABLE_WMMA=1 should disable WMMA."""
        old = os.environ.get("HIP_QUANT_DISABLE_WMMA")
        try:
            os.environ["HIP_QUANT_DISABLE_WMMA"] = "1"
            hq = get_hip_quant()
            # Even if the env var is set after init, the C function checks it
            A = np.zeros((16, 16), dtype=np.uint8)
            B = np.zeros((16, 16), dtype=np.uint8)
            with self.assertRaisesRegex(RuntimeError, "HIP_QUANT_DISABLE_WMMA"):
                hq.fp8_gemm_test_wmma(A, B, 16, 16, 16)
        finally:
            if old is not None:
                os.environ["HIP_QUANT_DISABLE_WMMA"] = old
            else:
                os.environ.pop("HIP_QUANT_DISABLE_WMMA", None)

    def test_gate_returns_reason(self):
        """wmma_diag gate_reason should be informative."""
        hq = get_hip_quant()
        result = hq.wmma_diag()
        if result is None:
            self.skipTest("wmma_diag not in DLL")
        if result["gated"]:
            self.assertIn(result["gate_reason"],
                ["not gfx12 device", "ROCm < 7.2", "WMMA disabled by env var", "None"])
        else:
            self.assertEqual(result["gate_reason"], "None")


if __name__ == "__main__":
    unittest.main()
