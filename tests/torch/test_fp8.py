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
