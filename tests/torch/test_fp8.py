"""
tests/torch/test_fp8.py
=======================

Minimal test suite for the hip_quant PyTorch FP8 extension.

Run with:
    pytest tests/torch/test_fp8.py -v

Requires:
    - PyTorch with ROCm support (torch 2.9+)
    - hip_quant._C extension built via: python setup_torch.py build_ext --inplace
    - At least one GPU visible to torch.cuda
"""

import math
import pytest
import torch

# ---------------------------------------------------------------------------
# Skip the whole module gracefully if torch or the extension is not available
# ---------------------------------------------------------------------------
torch_available = True
try:
    import torch
except ImportError:
    torch_available = False

extension_available = False
if torch_available:
    try:
        from hip_quant import _C  # type: ignore[attr-defined]
        extension_available = True
    except ImportError:
        pass

pytestmark = pytest.mark.skipif(
    not torch_available or not extension_available,
    reason="Requires PyTorch with ROCm and hip_quant._C extension built",
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("No CUDA/HIP GPU available")
    return torch.device("cuda")


@pytest.fixture
def float32_tensor(device):
    """Random float32 tensor with values spanning the FP8 range."""
    torch.manual_seed(42)
    return torch.randn(128, 64, device=device, dtype=torch.float32)


@pytest.fixture
def fp8_e4m3_tensor(float32_tensor):
    from hip_quant.torch_api import quantize_e4m3
    return quantize_e4m3(float32_tensor)


@pytest.fixture
def fp8_e5m2_tensor(float32_tensor):
    from hip_quant.torch_api import quantize_e5m2
    return quantize_e5m2(float32_tensor)


def _expanded_scales(scales, cols, block_size):
    return scales.repeat_interleave(block_size, dim=-1)[..., :cols]


# ===========================================================================
# Phase 1 — quantize_e4m3
# ===========================================================================

class TestQuantizeE4M3:
    def test_output_dtype(self, float32_tensor):
        from hip_quant.torch_api import quantize_e4m3
        out = quantize_e4m3(float32_tensor)
        assert out.dtype == torch.uint8, f"Expected uint8, got {out.dtype}"

    def test_output_device(self, float32_tensor, device):
        from hip_quant.torch_api import quantize_e4m3
        out = quantize_e4m3(float32_tensor)
        assert out.device.type == device.type

    def test_output_shape(self, float32_tensor):
        from hip_quant.torch_api import quantize_e4m3
        out = quantize_e4m3(float32_tensor)
        assert out.shape == float32_tensor.shape

    def test_rejects_cpu_tensor(self):
        from hip_quant.torch_api import quantize_e4m3
        cpu_t = torch.randn(8, device="cpu")
        with pytest.raises((RuntimeError, Exception)):
            quantize_e4m3(cpu_t)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_accepts_half_and_bfloat16(self, device, dtype):
        from hip_quant.torch_api import quantize_e4m3
        t = torch.tensor([1.0, -1.0, 0.0], device=device, dtype=dtype)
        out = quantize_e4m3(t)
        assert out.dtype == torch.uint8
        assert out.tolist()[:2] == [0x38, 0xB8]

    def test_zero_quantizes_to_zero(self, device):
        from hip_quant.torch_api import quantize_e4m3
        z = torch.zeros(1, device=device, dtype=torch.float32)
        out = quantize_e4m3(z)
        # Both +0 and -0 map to byte 0x00 or 0x80; either is a zero encoding
        assert out.item() in (0x00, 0x80), f"Zero mapped to unexpected byte {out.item():#04x}"

    def test_positive_one_encoding(self, device):
        from hip_quant.torch_api import quantize_e4m3
        # 1.0 in E4M3 (bias=7): exp=7, mant=0  → 0b0_0111_000 = 0x38
        t = torch.tensor([1.0], device=device, dtype=torch.float32)
        out = quantize_e4m3(t)
        assert out.item() == 0x38, (
            f"1.0 should encode to 0x38 in E4M3, got {out.item():#04x}"
        )

    def test_negative_one_encoding(self, device):
        from hip_quant.torch_api import quantize_e4m3
        # -1.0 in E4M3: 0b1_0111_000 = 0xB8
        t = torch.tensor([-1.0], device=device, dtype=torch.float32)
        out = quantize_e4m3(t)
        assert out.item() == 0xB8, (
            f"-1.0 should encode to 0xB8 in E4M3, got {out.item():#04x}"
        )

    def test_max_saturates(self, device):
        from hip_quant.torch_api import quantize_e4m3
        # Values beyond 448 should saturate to 0x7E (max finite positive)
        t = torch.tensor([1e9], device=device, dtype=torch.float32)
        out = quantize_e4m3(t)
        assert out.item() == 0x7E, (
            f"Large positive should saturate to 0x7E, got {out.item():#04x}"
        )


# ===========================================================================
# Phase 1 — quantize_e5m2
# ===========================================================================

class TestQuantizeE5M2:
    def test_output_dtype(self, float32_tensor):
        from hip_quant.torch_api import quantize_e5m2
        out = quantize_e5m2(float32_tensor)
        assert out.dtype == torch.uint8

    def test_output_device(self, float32_tensor, device):
        from hip_quant.torch_api import quantize_e5m2
        out = quantize_e5m2(float32_tensor)
        assert out.device.type == device.type

    def test_output_shape(self, float32_tensor):
        from hip_quant.torch_api import quantize_e5m2
        out = quantize_e5m2(float32_tensor)
        assert out.shape == float32_tensor.shape

    def test_rejects_cpu_tensor(self):
        from hip_quant.torch_api import quantize_e5m2
        cpu_t = torch.randn(8, device="cpu")
        with pytest.raises((RuntimeError, Exception)):
            quantize_e5m2(cpu_t)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_accepts_half_and_bfloat16(self, device, dtype):
        from hip_quant.torch_api import quantize_e5m2
        t = torch.tensor([1.0, -1.0, 0.0], device=device, dtype=dtype)
        out = quantize_e5m2(t)
        assert out.dtype == torch.uint8
        assert out.tolist()[:2] == [0x3C, 0xBC]

    def test_positive_one_encoding(self, device):
        from hip_quant.torch_api import quantize_e5m2
        # 1.0 in E5M2 (bias=15): exp=15, mant=0 → 0b0_01111_00 = 0x3C
        t = torch.tensor([1.0], device=device, dtype=torch.float32)
        out = quantize_e5m2(t)
        assert out.item() == 0x3C, (
            f"1.0 should encode to 0x3C in E5M2, got {out.item():#04x}"
        )

    def test_stochastic_output_contract(self, device):
        if not hasattr(_C, "quantize_e5m2_stochastic"):
            pytest.skip("extension must be rebuilt with quantize_e5m2_stochastic")
        from hip_quant.torch_api import quantize_e5m2_stochastic
        t = torch.randn(128, 64, device=device, dtype=torch.float32)
        out = quantize_e5m2_stochastic(t, seed=123)
        assert out.dtype == torch.uint8
        assert out.device.type == device.type
        assert out.shape == t.shape

    def test_stochastic_rejects_cpu_tensor(self):
        if not hasattr(_C, "quantize_e5m2_stochastic"):
            pytest.skip("extension must be rebuilt with quantize_e5m2_stochastic")
        from hip_quant.torch_api import quantize_e5m2_stochastic
        with pytest.raises((RuntimeError, Exception)):
            quantize_e5m2_stochastic(torch.randn(8, device="cpu"), seed=1)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_stochastic_accepts_half_and_bfloat16(self, device, dtype):
        if not hasattr(_C, "quantize_e5m2_stochastic"):
            pytest.skip("extension must be rebuilt with quantize_e5m2_stochastic")
        from hip_quant.torch_api import quantize_e5m2_stochastic
        t = torch.tensor([1.0, -1.0, 0.0], device=device, dtype=dtype)
        out = quantize_e5m2_stochastic(t, seed=123)
        assert out.dtype == torch.uint8
        assert out.tolist()[:2] == [0x3C, 0xBC]

    def test_stochastic_special_values(self, device):
        if not hasattr(_C, "quantize_e5m2_stochastic"):
            pytest.skip("extension must be rebuilt with quantize_e5m2_stochastic")
        from hip_quant.torch_api import quantize_e5m2_stochastic
        t = torch.tensor([float("inf"), -float("inf"), float("nan"), 1e9], device=device)
        out = quantize_e5m2_stochastic(t, seed=123)
        assert out[0].item() == 0x7C
        assert out[1].item() == 0xFC
        assert out[2].item() & 0x7C == 0x7C
        assert out[2].item() & 0x03 != 0
        assert out[3].item() == 0x7C

    def test_stochastic_reproducible_with_seed(self, device):
        if not hasattr(_C, "quantize_e5m2_stochastic"):
            pytest.skip("extension must be rebuilt with quantize_e5m2_stochastic")
        from hip_quant.torch_api import quantize_e5m2_stochastic
        t = torch.full((8192,), 1.125, device=device, dtype=torch.float32)
        a = quantize_e5m2_stochastic(t, seed=111)
        b = quantize_e5m2_stochastic(t, seed=111)
        c = quantize_e5m2_stochastic(t, seed=112)
        assert torch.equal(a, b)
        assert not torch.equal(a, c)

    def test_stochastic_exact_values_remain_exact(self, device):
        if not hasattr(_C, "quantize_e5m2_stochastic"):
            pytest.skip("extension must be rebuilt with quantize_e5m2_stochastic")
        from hip_quant.torch_api import quantize_e5m2_stochastic
        t = torch.full((4096,), 1.0, device=device, dtype=torch.float32)
        a = quantize_e5m2_stochastic(t, seed=1)
        b = quantize_e5m2_stochastic(t, seed=2)
        assert torch.all(a == 0x3C).item()
        assert torch.equal(a, b)

    def test_stochastic_midpoint_is_unbiased(self, device):
        if not hasattr(_C, "quantize_e5m2_stochastic"):
            pytest.skip("extension must be rebuilt with quantize_e5m2_stochastic")
        from hip_quant.torch_api import quantize_e5m2_stochastic, dequantize_e5m2
        t = torch.full((32768,), 1.125, device=device, dtype=torch.float32)
        out = quantize_e5m2_stochastic(t, seed=98765)
        lower = (out == 0x3C).sum().item()
        upper = (out == 0x3D).sum().item()
        assert lower + upper == t.numel()
        up_ratio = upper / t.numel()
        assert 0.45 <= up_ratio <= 0.55
        mean = dequantize_e5m2(out).mean().item()
        assert mean == pytest.approx(1.125, abs=0.02)

    def test_stochastic_fractional_distance_probability(self, device):
        if not hasattr(_C, "quantize_e5m2_stochastic"):
            pytest.skip("extension must be rebuilt with quantize_e5m2_stochastic")
        from hip_quant.torch_api import quantize_e5m2_stochastic
        # 1.0625 is 25% of the way from 1.0 (0x3C) to 1.25 (0x3D).
        t = torch.full((32768,), 1.0625, device=device, dtype=torch.float32)
        out = quantize_e5m2_stochastic(t, seed=2468)
        upper = (out == 0x3D).sum().item()
        up_ratio = upper / t.numel()
        assert 0.20 <= up_ratio <= 0.30

    def test_stochastic_preserves_negative_bins(self, device):
        if not hasattr(_C, "quantize_e5m2_stochastic"):
            pytest.skip("extension must be rebuilt with quantize_e5m2_stochastic")
        from hip_quant.torch_api import quantize_e5m2_stochastic
        t = torch.full((32768,), -1.125, device=device, dtype=torch.float32)
        out = quantize_e5m2_stochastic(t, seed=13579)
        lower = (out == 0xBC).sum().item()
        upper = (out == 0xBD).sum().item()
        assert lower + upper == t.numel()
        assert 0.45 <= upper / t.numel() <= 0.55

    def test_stochastic_tiny_gradients_do_not_all_flush_to_zero(self, device):
        if not hasattr(_C, "quantize_e5m2_stochastic"):
            pytest.skip("extension must be rebuilt with quantize_e5m2_stochastic")
        from hip_quant.torch_api import quantize_e5m2_stochastic
        # Halfway between zero and the smallest E5M2 subnormal, useful for
        # catching early-gradient collapse in deterministic underflow cases.
        t = torch.full((32768,), 2.0 ** -17, device=device, dtype=torch.float32)
        out = quantize_e5m2_stochastic(t, seed=97531)
        zeros = (out == 0x00).sum().item()
        min_subs = (out == 0x01).sum().item()
        assert zeros + min_subs == t.numel()
        assert 0.45 <= min_subs / t.numel() <= 0.55

    def test_backward_helper_uses_stochastic_env_flag(self, device, monkeypatch):
        if not hasattr(_C, "quantize_e5m2_stochastic"):
            pytest.skip("extension must be rebuilt with quantize_e5m2_stochastic")
        from hip_quant import torch_api as hqt
        t = torch.full((1024,), 1.125, device=device, dtype=torch.float32)

        monkeypatch.delenv("HIP_QUANT_STOCHASTIC_E5M2", raising=False)
        prepared, fp8 = hqt._prepare_e5m2_backward_grad_output(t)
        assert prepared is t
        assert fp8 is None

        monkeypatch.setenv("HIP_QUANT_STOCHASTIC_E5M2", "1")
        monkeypatch.setenv("HIP_QUANT_STOCHASTIC_E5M2_SEED", "1234")
        prepared, fp8 = hqt._prepare_e5m2_backward_grad_output(t)
        assert fp8 is not None
        assert fp8.dtype == torch.uint8
        assert torch.equal(prepared, hqt.dequantize_e5m2(fp8).to(t.dtype))


# ===========================================================================
# Phase 1 — dequantize_e4m3
# ===========================================================================

class TestDequantizeE4M3:
    def test_output_dtype(self, fp8_e4m3_tensor):
        from hip_quant.torch_api import dequantize_e4m3
        out = dequantize_e4m3(fp8_e4m3_tensor)
        assert out.dtype == torch.float32

    def test_output_device(self, fp8_e4m3_tensor, device):
        from hip_quant.torch_api import dequantize_e4m3
        out = dequantize_e4m3(fp8_e4m3_tensor)
        assert out.device.type == device.type

    def test_output_shape(self, fp8_e4m3_tensor):
        from hip_quant.torch_api import dequantize_e4m3
        out = dequantize_e4m3(fp8_e4m3_tensor)
        assert out.shape == fp8_e4m3_tensor.shape

    def test_rejects_cpu_tensor(self):
        from hip_quant.torch_api import dequantize_e4m3
        cpu_t = torch.zeros(8, dtype=torch.uint8)
        with pytest.raises((RuntimeError, Exception)):
            dequantize_e4m3(cpu_t)

    def test_rejects_wrong_dtype(self, device):
        from hip_quant.torch_api import dequantize_e4m3
        t = torch.zeros(8, device=device, dtype=torch.float32)
        with pytest.raises((RuntimeError, Exception)):
            dequantize_e4m3(t)

    def test_round_trip_close(self, float32_tensor):
        """Quantize then dequantize should be close (within FP8 precision)."""
        from hip_quant.torch_api import quantize_e4m3, dequantize_e4m3
        fp8  = quantize_e4m3(float32_tensor)
        back = dequantize_e4m3(fp8)
        # FP8 E4M3 has ~3 bits of mantissa; expect ~12.5% max relative error
        diff = (back - float32_tensor).abs()
        ref  = float32_tensor.abs().clamp(min=1e-6)
        rel  = (diff / ref).mean().item()
        assert rel < 0.20, f"Round-trip mean relative error too large: {rel:.4f}"

    def test_one_roundtrip(self, device):
        from hip_quant.torch_api import quantize_e4m3, dequantize_e4m3
        t    = torch.tensor([1.0], device=device, dtype=torch.float32)
        back = dequantize_e4m3(quantize_e4m3(t))
        assert abs(back.item() - 1.0) < 1e-5, f"1.0 round-trip failed: {back.item()}"


# ===========================================================================
# Phase 1 — dequantize_e5m2
# ===========================================================================

class TestDequantizeE5M2:
    def test_output_dtype(self, fp8_e5m2_tensor):
        from hip_quant.torch_api import dequantize_e5m2
        out = dequantize_e5m2(fp8_e5m2_tensor)
        assert out.dtype == torch.float32

    def test_output_device(self, fp8_e5m2_tensor, device):
        from hip_quant.torch_api import dequantize_e5m2
        out = dequantize_e5m2(fp8_e5m2_tensor)
        assert out.device.type == device.type

    def test_round_trip_close(self, float32_tensor):
        """E5M2 has 2-bit mantissa; wider range, less precision than E4M3."""
        from hip_quant.torch_api import quantize_e5m2, dequantize_e5m2
        fp8  = quantize_e5m2(float32_tensor)
        back = dequantize_e5m2(fp8)
        diff = (back - float32_tensor).abs()
        ref  = float32_tensor.abs().clamp(min=1e-6)
        rel  = (diff / ref).mean().item()
        assert rel < 0.35, f"E5M2 round-trip mean relative error too large: {rel:.4f}"


# ===========================================================================
# Phase 1 — block-wise FP8 quantize / dequantize
# ===========================================================================

class TestBlockwiseFp8:
    def test_e4m3_contract(self, device):
        if not hasattr(_C, "quantize_e4m3_blockwise"):
            pytest.skip("extension must be rebuilt with block-wise FP8 ops")
        from hip_quant.torch_api import quantize_e4m3_blockwise

        x = torch.randn(3, 70, device=device, dtype=torch.float32)
        q, scales = quantize_e4m3_blockwise(x, block_size=16)

        assert q.dtype == torch.uint8
        assert q.shape == x.shape
        assert q.device == x.device
        assert scales.dtype == torch.float32
        assert scales.shape == (3, 5)
        assert scales.device == x.device
        assert torch.all(scales > 0)

    def test_e4m3_matches_scaled_elementwise(self, device):
        if not hasattr(_C, "quantize_e4m3_blockwise"):
            pytest.skip("extension must be rebuilt with block-wise FP8 ops")
        from hip_quant.torch_api import (
            dequantize_e4m3,
            dequantize_e4m3_blockwise,
            quantize_e4m3,
            quantize_e4m3_blockwise,
        )

        torch.manual_seed(123)
        x = torch.randn(4, 70, device=device, dtype=torch.float32)
        x[:, 32:] *= 0.015625
        block_size = 16

        q, scales = quantize_e4m3_blockwise(x, block_size)
        y = dequantize_e4m3_blockwise(q, scales, block_size)

        scale_full = _expanded_scales(scales, x.size(-1), block_size)
        expected = dequantize_e4m3(quantize_e4m3((x / scale_full).contiguous())) * scale_full
        torch.testing.assert_close(y, expected, rtol=0, atol=0)

    def test_e5m2_matches_scaled_elementwise(self, device):
        if not hasattr(_C, "quantize_e5m2_blockwise"):
            pytest.skip("extension must be rebuilt with block-wise FP8 ops")
        from hip_quant.torch_api import (
            dequantize_e5m2,
            dequantize_e5m2_blockwise,
            quantize_e5m2,
            quantize_e5m2_blockwise,
        )

        torch.manual_seed(456)
        x = torch.randn(2, 3, 65, device=device, dtype=torch.float32) * 128.0
        block_size = 32

        q, scales = quantize_e5m2_blockwise(x, block_size)
        y = dequantize_e5m2_blockwise(q, scales, block_size)

        assert q.shape == x.shape
        assert scales.shape == (2, 3, 3)

        scale_full = _expanded_scales(scales, x.size(-1), block_size)
        expected = dequantize_e5m2(quantize_e5m2((x / scale_full).contiguous())) * scale_full
        torch.testing.assert_close(y, expected, rtol=0, atol=0)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_accepts_half_and_bfloat16(self, device, dtype):
        if not hasattr(_C, "quantize_e4m3_blockwise"):
            pytest.skip("extension must be rebuilt with block-wise FP8 ops")
        from hip_quant.torch_api import dequantize_e4m3_blockwise, quantize_e4m3_blockwise

        x = torch.tensor([[1.0, -1.0, 0.0, 2.0]], device=device, dtype=dtype)
        q, scales = quantize_e4m3_blockwise(x, block_size=4)
        y = dequantize_e4m3_blockwise(q, scales, block_size=4)

        assert q.dtype == torch.uint8
        assert scales.shape == (1, 1)
        assert y.dtype == torch.float32
        torch.testing.assert_close(y, x.float(), rtol=0, atol=1e-6)

    def test_rejects_bad_block_size(self, device):
        if not hasattr(_C, "quantize_e4m3_blockwise"):
            pytest.skip("extension must be rebuilt with block-wise FP8 ops")
        from hip_quant.torch_api import quantize_e4m3_blockwise

        with pytest.raises((RuntimeError, Exception)):
            quantize_e4m3_blockwise(torch.randn(8, device=device), block_size=0)

    def test_e5m2_stochastic_blockwise_reproducible(self, device):
        if not hasattr(_C, "quantize_e5m2_blockwise_stochastic"):
            pytest.skip("extension must be rebuilt with stochastic block-wise FP8 ops")
        from hip_quant.torch_api import quantize_e5m2_blockwise_stochastic

        x = torch.full((1, 4096), 1.125, device=device, dtype=torch.float32)
        x[0, 0] = 57344.0  # forces scale=1.0 for this block
        a, scales_a = quantize_e5m2_blockwise_stochastic(x, block_size=4096, seed=123)
        b, scales_b = quantize_e5m2_blockwise_stochastic(x, block_size=4096, seed=123)
        c, _ = quantize_e5m2_blockwise_stochastic(x, block_size=4096, seed=124)

        assert torch.equal(a, b)
        assert torch.equal(scales_a, scales_b)
        assert not torch.equal(a[:, 1:], c[:, 1:])
        upper_ratio = (a[:, 1:] == 0x3D).sum().item() / (x.numel() - 1)
        assert 0.45 <= upper_ratio <= 0.55

    def test_refresh_fp8_blockwise_shadow_copies_buffers(self, device):
        if not hasattr(_C, "quantize_e4m3_blockwise"):
            pytest.skip("extension must be rebuilt with block-wise FP8 ops")
        from hip_quant.torch_api import refresh_fp8_blockwise_shadow, quantize_e4m3_blockwise

        weight = torch.randn(5, 33, device=device, dtype=torch.float32)
        expected_q, expected_scales = quantize_e4m3_blockwise(weight, block_size=16)
        weight_fp8 = torch.empty_like(expected_q)
        weight_scales = torch.empty_like(expected_scales)

        q, scales = refresh_fp8_blockwise_shadow(weight, weight_fp8, weight_scales, block_size=16)

        assert q.data_ptr() == weight_fp8.data_ptr()
        assert scales.data_ptr() == weight_scales.data_ptr()
        assert torch.equal(q, expected_q)
        torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)

    def test_adafactor_row_col_mean_square(self, device):
        if not hasattr(_C, "adafactor_row_col_mean_square"):
            pytest.skip("extension must be rebuilt with Adafactor reduction ops")
        from hip_quant.torch_api import adafactor_row_col_mean_square

        torch.manual_seed(789)
        grad = torch.randn(17, 31, device=device, dtype=torch.float32)
        row, col = adafactor_row_col_mean_square(grad, eps=1e-7)

        torch.testing.assert_close(row, grad.pow(2).mean(dim=-1) + 1e-7, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(col, grad.pow(2).mean(dim=-2) + 1e-7, rtol=1e-6, atol=1e-7)

    def test_fp8_linear_forward_blockwise_matches_dequant_reference(self, device):
        if not hasattr(_C, "fp8_linear_forward_blockwise"):
            pytest.skip("extension must be rebuilt with block-wise linear op")
        from hip_quant.torch_api import (
            dequantize_e4m3_blockwise,
            fp8_linear_forward_blockwise_quantized,
            quantize_e4m3_blockwise,
        )

        torch.manual_seed(321)
        x = torch.randn(7, 33, device=device, dtype=torch.float32)
        w = torch.randn(11, 33, device=device, dtype=torch.float32)
        bias = torch.randn(11, device=device, dtype=torch.float32)
        block_size = 16
        x_fp8, x_scales = quantize_e4m3_blockwise(x, block_size)
        w_fp8, w_scales = quantize_e4m3_blockwise(w, block_size)

        out = fp8_linear_forward_blockwise_quantized(
            x_fp8, x_scales, w_fp8, w_scales, x, block_size, bias
        )
        ref = dequantize_e4m3_blockwise(x_fp8, x_scales, block_size).matmul(
            dequantize_e4m3_blockwise(w_fp8, w_scales, block_size).t()
        ) + bias

        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    def test_packed_wmma_weight_matches_row_major_wmma(self, device, monkeypatch):
        if not hasattr(_C, "pack_fp8_weight_for_wmma"):
            pytest.skip("extension must be rebuilt with packed WMMA weight op")
        monkeypatch.setenv("HIP_QUANT_ENABLE_GFX12_WMMA", "1")
        from hip_quant.torch_api import (
            fp8_linear_forward_fp8_input_weight,
            fp8_linear_forward_fp8_input_weight_packed,
            pack_fp8_weight_for_wmma,
            quantize_e4m3,
        )

        torch.manual_seed(654)
        x = torch.randn(17, 33, device=device, dtype=torch.bfloat16)
        w = torch.randn(19, 33, device=device, dtype=torch.bfloat16)
        x_fp8 = quantize_e4m3(x)
        w_fp8 = quantize_e4m3(w)
        w_packed = pack_fp8_weight_for_wmma(w_fp8)

        row_major = fp8_linear_forward_fp8_input_weight(
            x_fp8, w_fp8, x, weight_inv_scale=1.0, input_scale=1.0, bias=None
        )
        packed = fp8_linear_forward_fp8_input_weight_packed(
            x_fp8, w_packed, x, output_features=w.size(0),
            weight_inv_scale=1.0, input_scale=1.0, bias=None
        )
        torch.testing.assert_close(packed, row_major, rtol=0, atol=0)

    def test_v2_lds_staged_wmma_matches_v1(self, device, monkeypatch):
        """V2 cooperative LDS-staged FP8 kernel matches V1 row-major output."""
        if not hasattr(_C, "fp8_linear_forward_v2_input_weight"):
            pytest.skip("extension must be rebuilt with V2 LDS-staged kernel")
        monkeypatch.setenv("HIP_QUANT_ENABLE_GFX12_WMMA", "1")
        from hip_quant.torch_api import (
            fp8_linear_forward_fp8_input_weight,
            fp8_linear_forward_v2_input_weight,
            quantize_e4m3,
        )

        torch.manual_seed(789)
        x = torch.randn(17, 33, device=device, dtype=torch.bfloat16)
        w = torch.randn(19, 33, device=device, dtype=torch.bfloat16)
        x_fp8 = quantize_e4m3(x)
        w_fp8 = quantize_e4m3(w)

        v1 = fp8_linear_forward_fp8_input_weight(
            x_fp8, w_fp8, x, weight_inv_scale=1.0, input_scale=1.0, bias=None
        )
        v2 = fp8_linear_forward_v2_input_weight(
            x_fp8, w_fp8, x, weight_inv_scale=1.0, input_scale=1.0, bias=None
        )
        torch.testing.assert_close(v2, v1, rtol=0, atol=0)


# ===========================================================================
# Phase 3 — Fp8LinearFunction autograd
# ===========================================================================

class TestFp8LinearFunction:
    def test_forward_shape(self, device):
        from hip_quant.torch_api import Fp8LinearFunction
        M, K, N = 8, 16, 32
        inp = torch.randn(M, K, device=device, dtype=torch.float32)
        wt  = torch.randn(N, K, device=device, dtype=torch.float32)
        out = Fp8LinearFunction.apply(inp, wt, None)
        assert out.shape == (M, N)

    def test_forward_with_bias(self, device):
        from hip_quant.torch_api import Fp8LinearFunction
        M, K, N = 4, 8, 12
        inp  = torch.randn(M, K, device=device, dtype=torch.float32)
        wt   = torch.randn(N, K, device=device, dtype=torch.float32)
        bias = torch.randn(N,    device=device, dtype=torch.float32)
        out  = Fp8LinearFunction.apply(inp, wt, bias)
        assert out.shape == (M, N)

    def test_backward_computes(self, device):
        """Backward should run without error and produce non-None gradients."""
        from hip_quant.torch_api import Fp8LinearFunction
        M, K, N = 4, 8, 6
        inp = torch.randn(M, K, device=device, requires_grad=True)
        wt  = torch.randn(N, K, device=device, requires_grad=True)
        out = Fp8LinearFunction.apply(inp, wt, None)
        out.sum().backward()
        assert inp.grad is not None, "grad_input is None"
        assert wt.grad  is not None, "grad_weight is None"
        assert inp.grad.shape == inp.shape
        assert wt.grad.shape  == wt.shape

    def test_no_cpu_transfers(self, device):
        """Forward+backward must not move data to CPU."""
        from hip_quant.torch_api import Fp8LinearFunction
        M, K, N = 8, 16, 8
        inp = torch.randn(M, K, device=device, requires_grad=True)
        wt  = torch.randn(N, K, device=device, requires_grad=True)
        out = Fp8LinearFunction.apply(inp, wt, None)
        out.sum().backward()
        # All tensors must still be on GPU
        assert inp.grad.device.type == device.type
        assert wt.grad.device.type  == device.type

    def test_rejects_float64(self, device):
        from hip_quant.torch_api import Fp8LinearFunction
        M, K, N = 3, 4, 5
        inp = torch.randn(M, K, device=device, dtype=torch.float64,
                          requires_grad=True)
        wt  = torch.randn(N, K, device=device, dtype=torch.float64,
                          requires_grad=True)
        with pytest.raises((RuntimeError, Exception)):
            Fp8LinearFunction.apply(inp, wt, None)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_half_and_bfloat16_forward_backward(self, device, dtype):
        from hip_quant.torch_api import Fp8LinearFunction
        M, K, N = 8, 16, 12
        inp = torch.randn(M, K, device=device, dtype=dtype, requires_grad=True)
        wt = torch.randn(N, K, device=device, dtype=dtype, requires_grad=True)
        bias = torch.randn(N, device=device, dtype=dtype, requires_grad=True)
        out = Fp8LinearFunction.apply(inp, wt, bias)
        assert out.dtype == dtype
        assert out.shape == (M, N)
        out.float().sum().backward()
        assert inp.grad is not None and inp.grad.dtype == dtype
        assert wt.grad is not None and wt.grad.dtype == dtype
        assert bias.grad is not None and bias.grad.dtype == dtype


# ===========================================================================
# Phase 3 — Fp8Linear nn.Module
# ===========================================================================

class TestFp8Linear:
    def test_forward_shape(self, device):
        from hip_quant.torch_api import Fp8Linear
        layer = Fp8Linear(64, 32).to(device)
        x = torch.randn(16, 64, device=device)
        y = layer(x)
        assert y.shape == (16, 32)

    def test_batched_forward(self, device):
        from hip_quant.torch_api import Fp8Linear
        layer = Fp8Linear(16, 8).to(device)
        x = torch.randn(4, 10, 16, device=device)
        y = layer(x)
        assert y.shape == (4, 10, 8)

    def test_backward_updates_params(self, device):
        from hip_quant.torch_api import Fp8Linear
        layer = Fp8Linear(16, 8).to(device)
        opt = torch.optim.SGD(layer.parameters(), lr=0.01)
        x = torch.randn(4, 16, device=device)
        loss = layer(x).sum()
        loss.backward()
        opt.step()
        # Just check it doesn't crash and grads exist
        assert layer.weight.grad is not None

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_half_and_bfloat16_module(self, device, dtype):
        from hip_quant.torch_api import Fp8Linear
        layer = Fp8Linear(16, 8, dtype=dtype).to(device)
        x = torch.randn(4, 16, device=device, dtype=dtype, requires_grad=True)
        y = layer(x)
        assert y.dtype == dtype
        y.float().mean().backward()
        assert x.grad is not None and x.grad.dtype == dtype
        assert layer.weight.grad is not None and layer.weight.grad.dtype == dtype

    def test_tiny_training_loop(self, device):
        """Run a tiny model for a few steps; no CPU transfers allowed."""
        from hip_quant.torch_api import Fp8Linear
        import torch.nn as nn

        model = nn.Sequential(
            Fp8Linear(32, 16),
            nn.ReLU(),
            Fp8Linear(16, 4),
        ).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(5):
            x    = torch.randn(8, 32, device=device)
            loss = model(x).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            # Verify loss is a finite scalar
            assert math.isfinite(loss.item()), f"Loss became non-finite: {loss.item()}"


class TestFp8ShadowLinear:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_half_and_bfloat16_shadow_module(self, device, dtype):
        from hip_quant.torch_api import Fp8ShadowLinear
        layer = Fp8ShadowLinear(16, 8, dtype=dtype).to(device)
        x = torch.randn(4, 16, device=device, dtype=dtype, requires_grad=True)
        y = layer(x)
        assert y.dtype == dtype
        y.float().mean().backward()
        assert x.grad is not None and x.grad.dtype == dtype
        assert layer.weight_master.grad is not None
        assert layer.weight_master.grad.dtype == dtype
        assert layer.weight_fp8.dtype == torch.uint8


# ===========================================================================
# Phase 4 — Fp8TensorMeta scale tracking
# ===========================================================================

class TestFp8TensorMeta:
    def test_initial_scale_is_one(self, device):
        from hip_quant.torch_api import Fp8TensorMeta
        meta = Fp8TensorMeta(device=str(device))
        assert meta.scale.item() == pytest.approx(1.0)
        assert meta.inv_scale.item() == pytest.approx(1.0)

    def test_update_adjusts_scale(self, device):
        from hip_quant.torch_api import Fp8TensorMeta
        meta = Fp8TensorMeta(device=str(device))
        t = torch.full((4,), 2.0, device=device)
        meta.update(t)
        # amax=2.0, scale should be 448/2 = 224
        assert meta.scale.item() == pytest.approx(224.0, rel=1e-4)

    def test_quantize_dequantize_roundtrip(self, device):
        from hip_quant.torch_api import Fp8TensorMeta
        meta = Fp8TensorMeta(device=str(device))
        t = torch.tensor([1.0, -1.0, 0.5], device=device)
        meta.update(t)
        q    = meta.quantize_e4m3(t)
        back = meta.dequantize_e4m3(q)
        assert back.dtype == torch.float32
        assert back.device.type == device.type


# ===========================================================================
# FP8 Conv2d — unfold + hipBLASLt-backed FP8 linear
# ===========================================================================

class TestFp8Conv2d:
    def test_fp8_conv2d_shape_and_backward(self, device):
        if not hasattr(torch, "_scaled_mm") or not hasattr(torch, "float8_e4m3fn"):
            pytest.skip("fp8_conv2d test requires PyTorch _scaled_mm hipBLASLt path")

        from hip_quant.torch_api import fp8_conv2d

        torch.manual_seed(123)
        x = torch.randn(2, 3, 8, 8, device=device, requires_grad=True)
        weight = torch.randn(4, 3, 3, 3, device=device, requires_grad=True)
        bias = torch.randn(4, device=device, requires_grad=True)

        out = fp8_conv2d(x, weight, bias, stride=2, padding=1)
        ref = torch.nn.functional.conv2d(x, weight, bias, stride=2, padding=1)

        assert out.shape == ref.shape
        assert out.dtype == x.dtype
        out.float().mean().backward()
        assert x.grad is not None
        assert weight.grad is not None
        assert bias.grad is not None

    def test_fp8_conv2d_module_from_conv2d(self, device):
        if not hasattr(torch, "_scaled_mm") or not hasattr(torch, "float8_e4m3fn"):
            pytest.skip("Fp8Conv2d test requires PyTorch _scaled_mm hipBLASLt path")

        from hip_quant.torch_api import Fp8Conv2d

        conv = torch.nn.Conv2d(3, 4, 3, padding=1).to(device)
        fp8_conv = Fp8Conv2d.from_conv2d(conv)
        x = torch.randn(2, 3, 8, 8, device=device)

        out = fp8_conv(x)
        assert out.shape == conv(x).shape
        assert fp8_conv.to_conv2d().weight.shape == conv.weight.shape


class TestFp8Conv1d:
    def test_fp8_conv1d_shape_and_backward(self, device):
        if not hasattr(torch, "_scaled_mm") or not hasattr(torch, "float8_e4m3fn"):
            pytest.skip("fp8_conv1d test requires PyTorch _scaled_mm hipBLASLt path")

        from hip_quant.torch_api import fp8_conv1d

        torch.manual_seed(321)
        x = torch.randn(2, 3, 17, device=device, requires_grad=True)
        weight = torch.randn(5, 3, 3, device=device, requires_grad=True)
        bias = torch.randn(5, device=device, requires_grad=True)

        out = fp8_conv1d(x, weight, bias, stride=2, padding=1)
        ref = torch.nn.functional.conv1d(x, weight, bias, stride=2, padding=1)

        assert out.shape == ref.shape
        out.float().mean().backward()
        assert x.grad is not None
        assert weight.grad is not None
        assert bias.grad is not None

    def test_fp8_conv1d_module_from_conv1d(self, device):
        if not hasattr(torch, "_scaled_mm") or not hasattr(torch, "float8_e4m3fn"):
            pytest.skip("Fp8Conv1d test requires PyTorch _scaled_mm hipBLASLt path")

        from hip_quant.torch_api import Fp8Conv1d

        conv = torch.nn.Conv1d(3, 5, 3, padding=1).to(device)
        fp8_conv = Fp8Conv1d.from_conv1d(conv)
        x = torch.randn(2, 3, 17, device=device)

        out = fp8_conv(x)
        assert out.shape == conv(x).shape
        assert fp8_conv.to_conv1d().weight.shape == conv.weight.shape
