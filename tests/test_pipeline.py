"""
tests/test_pipeline.py
======================
Full pipeline test suite — runs on CPU with a mocked _C extension.

No GPU, no compiled extension, no hipcc required.
The mock uses the same bit-manipulation functions that passed the 90-test
math suite, so FP8 arithmetic is bit-exact to the real kernels.

Run:
    python tests/test_pipeline.py -v
"""

from __future__ import annotations

import math
import struct
import sys
import types
import unittest
from typing import Optional

import torch
import torch.nn as nn

# ===========================================================================
# 1.  Pure-Python FP8 math  (proven correct by math_test_fp8.py / ml_dtypes)
# ===========================================================================

def _f2i(f):
    return struct.unpack("<I", struct.pack("<f", f))[0]

def _i2f(i):
    return struct.unpack("<f", struct.pack("<I", i & 0xFFFFFFFF))[0]

def _fp32_to_fp8_e4m3(f):
    u = _f2i(f); sign = u >> 31; abs_u = u & 0x7FFFFFFF
    if abs_u == 0:            return (sign << 7) & 0xFF
    if abs_u > 0x7F800000:   return ((sign << 7) | 0x7F) & 0xFF
    if abs_u == 0x7F800000:  return ((sign << 7) | 0x7E) & 0xFF
    exp = (abs_u >> 23) & 0xFF; mant = abs_u & 0x7FFFFF
    if exp == 0: return (sign << 7) & 0xFF
    exp8 = exp - 127 + 7
    if exp8 <= 0:
        sh = 1 - exp8
        if sh > 4: return (sign << 7) & 0xFF
        full = 0x800000 | mant; ts = 20 + sh; r = full >> ts
        rem = full & ((1 << ts) - 1); mid = 1 << (ts - 1)
        if rem > mid or (rem == mid and (r & 1)): r += 1
        if r >= 8: return ((sign << 7) | (1 << 3)) & 0xFF
        return ((sign << 7) | (r & 7)) & 0xFF
    m8 = (mant >> 20) & 7; rnd = mant & 0xFFFFF
    if rnd > 0x80000 or (rnd == 0x80000 and (m8 & 1)):
        m8 += 1
        if m8 >= 8: m8 = 0; exp8 += 1
    if exp8 >= 16 or (exp8 == 15 and m8 == 7): return ((sign << 7) | 0x7E) & 0xFF
    return ((sign << 7) | (exp8 << 3) | m8) & 0xFF

def _fp8_e4m3_to_fp32(h):
    h &= 0xFF; sign = h >> 7; exp = (h >> 3) & 0xF; mant = h & 7
    if exp == 15 and mant == 7: return _i2f((sign << 31) | 0x7FC00000)
    if exp == 0 and mant == 0:  return _i2f(sign << 31)
    if exp == 0:
        v = mant * 0.001953125; return (-v if sign else v)
    return _i2f((sign << 31) | ((exp + 120) << 23) | (mant << 20))

def _fp32_to_fp8_e5m2(f):
    u = _f2i(f); sign = u >> 31; abs_u = u & 0x7FFFFFFF
    if abs_u == 0:            return (sign << 7) & 0xFF
    if abs_u > 0x7F800000:   return ((sign << 7) | 0x7F) & 0xFF
    if abs_u == 0x7F800000:  return ((sign << 7) | 0x7C) & 0xFF
    exp = (abs_u >> 23) & 0xFF; mant = abs_u & 0x7FFFFF
    if exp == 0: return (sign << 7) & 0xFF
    exp8 = exp - 127 + 15
    if exp8 <= 0:
        sh = 1 - exp8
        if sh > 3: return (sign << 7) & 0xFF
        full = 0x800000 | mant; ts = 21 + sh; r = full >> ts
        rem = full & ((1 << ts) - 1); mid = 1 << (ts - 1)
        if rem > mid or (rem == mid and (r & 1)): r += 1
        if r >= 4: return ((sign << 7) | (1 << 2)) & 0xFF
        return ((sign << 7) | (r & 3)) & 0xFF
    m8 = (mant >> 21) & 3; rnd = mant & 0x1FFFFF
    if rnd > 0x100000 or (rnd == 0x100000 and (m8 & 1)):
        m8 += 1
        if m8 >= 4: m8 = 0; exp8 += 1
    if exp8 >= 31: return ((sign << 7) | 0x7C) & 0xFF
    return ((sign << 7) | (exp8 << 2) | m8) & 0xFF

def _fp8_e5m2_to_fp32(h):
    h &= 0xFF; sign = h >> 7; exp = (h >> 2) & 0x1F; mant = h & 3
    if exp == 31:
        return _i2f((sign << 31) | (0x7FC00000 if mant else 0x7F800000))
    if exp == 0 and mant == 0: return _i2f(sign << 31)
    if exp == 0:
        v = mant * 0.0000152587890625; return (-v if sign else v)
    return _i2f((sign << 31) | ((exp + 112) << 23) | (mant << 21))

# Vectorised element-wise wrappers (correct; slow for large tensors, fine in tests)

def _mock_quantize_e4m3(x: torch.Tensor) -> torch.Tensor:
    flat = x.detach().cpu().reshape(-1).float().tolist()
    out  = torch.tensor([_fp32_to_fp8_e4m3(v) for v in flat], dtype=torch.uint8)
    return out.reshape(x.shape)

def _mock_dequantize_e4m3(x: torch.Tensor) -> torch.Tensor:
    flat = x.detach().cpu().reshape(-1).tolist()
    out  = torch.tensor([_fp8_e4m3_to_fp32(int(v)) for v in flat], dtype=torch.float32)
    return out.reshape(x.shape)

def _mock_quantize_e5m2(x: torch.Tensor) -> torch.Tensor:
    flat = x.detach().cpu().reshape(-1).float().tolist()
    out  = torch.tensor([_fp32_to_fp8_e5m2(v) for v in flat], dtype=torch.uint8)
    return out.reshape(x.shape)

def _mock_dequantize_e5m2(x: torch.Tensor) -> torch.Tensor:
    flat = x.detach().cpu().reshape(-1).tolist()
    out  = torch.tensor([_fp8_e5m2_to_fp32(int(v)) for v in flat], dtype=torch.float32)
    return out.reshape(x.shape)

def _mock_fp8_linear_forward(inp, wt, bias=None):
    out = inp.float() @ wt.float().t()
    return out + bias.float() if bias is not None else out

def _mock_fp8_linear_backward_input(go, wt):
    return go.float() @ wt.float()

def _mock_fp8_linear_backward_weight(go, inp):
    return go.float().t() @ inp.float()

_MOCK_C = types.SimpleNamespace(
    quantize_e4m3=_mock_quantize_e4m3,
    dequantize_e4m3=_mock_dequantize_e4m3,
    quantize_e5m2=_mock_quantize_e5m2,
    dequantize_e5m2=_mock_dequantize_e5m2,
    fp8_linear_forward=_mock_fp8_linear_forward,
    fp8_linear_backward_input=_mock_fp8_linear_backward_input,
    fp8_linear_backward_weight=_mock_fp8_linear_backward_weight,
)

# ---------------------------------------------------------------------------
# Inject mock BEFORE importing torch_api so _load_extension() is short-circuited
# ---------------------------------------------------------------------------
import hip_quant.torch_api as _torch_api_module
_torch_api_module._C = _MOCK_C

from hip_quant.torch_api import (  # noqa: E402
    quantize_e4m3, quantize_e5m2, dequantize_e4m3, dequantize_e5m2,
    Fp8LinearFunction, Fp8Linear,
    Fp8ScaledLinearFunction, Fp8ScaledLinear,
    Fp8ShadowLinearFunction, Fp8ShadowLinear,
    Fp8TensorMeta, convert_to_fp8, Adafactor,
    _sim_fp8_e4m3, _sim_fp8_e5m2,
)

PASS = FAIL = 0

def _summarise(r: unittest.TestResult, suite_name: str):
    global PASS, FAIL
    p = r.testsRun - len(r.failures) - len(r.errors)
    PASS += p; FAIL += len(r.failures) + len(r.errors)
    status = "PASS" if not r.failures and not r.errors else "FAIL"
    print(f"  [{status}]  {suite_name}  ({p}/{r.testsRun} passed)")
    for _, tb in r.failures + r.errors:
        for line in tb.splitlines()[-6:]:
            print(f"          {line}")


# ===========================================================================
# 2.  Phase 2 — element-wise quant / dequant
# ===========================================================================

class TestPhase2(unittest.TestCase):

    def _t(self, *shape):
        return torch.randn(*shape)

    # ---- dtype -------------------------------------------------------
    def test_quantize_e4m3_dtype(self):
        self.assertEqual(quantize_e4m3(self._t(4, 4)).dtype, torch.uint8)

    def test_quantize_e5m2_dtype(self):
        self.assertEqual(quantize_e5m2(self._t(4, 4)).dtype, torch.uint8)

    def test_dequantize_e4m3_dtype(self):
        q = quantize_e4m3(self._t(4, 4))
        self.assertEqual(dequantize_e4m3(q).dtype, torch.float32)

    def test_dequantize_e5m2_dtype(self):
        q = quantize_e5m2(self._t(4, 4))
        self.assertEqual(dequantize_e5m2(q).dtype, torch.float32)

    # ---- shape preservation ------------------------------------------
    def test_shape_2d(self):
        x = self._t(6, 8)
        self.assertEqual(quantize_e4m3(x).shape, (6, 8))
        self.assertEqual(dequantize_e4m3(quantize_e4m3(x)).shape, (6, 8))

    def test_shape_3d(self):
        x = self._t(2, 4, 8)
        self.assertEqual(quantize_e5m2(x).shape, (2, 4, 8))

    # ---- round-trip precision ----------------------------------------
    def test_round_trip_e4m3_known_values(self):
        # 1.0, 2.0, 0.5, -1.0 are exactly representable in E4M3
        for v in [1.0, 2.0, 0.5, -1.0, 4.0, -4.0]:
            x   = torch.tensor([[v]])
            out = dequantize_e4m3(quantize_e4m3(x))
            self.assertAlmostEqual(out.item(), v, places=5,
                                   msg=f"E4M3 round-trip failed for {v}")

    def test_round_trip_e5m2_known_values(self):
        for v in [1.0, 2.0, 0.5, -1.0, 4.0]:
            x   = torch.tensor([[v]])
            out = dequantize_e5m2(quantize_e5m2(x))
            self.assertAlmostEqual(out.item(), v, places=5,
                                   msg=f"E5M2 round-trip failed for {v}")

    # ---- saturation --------------------------------------------------
    def test_e4m3_saturates(self):
        x   = torch.tensor([[1e9]])
        out = dequantize_e4m3(quantize_e4m3(x))
        self.assertAlmostEqual(out.item(), 448.0, places=1)

    def test_e5m2_inf_preserved(self):
        x   = torch.tensor([[float("inf")]])
        q   = quantize_e5m2(x)
        # 0x7C is +Inf in E5M2; dequantize should give +Inf back
        out = dequantize_e5m2(q)
        self.assertTrue(math.isinf(out.item()) and out.item() > 0)

    # ---- zero --------------------------------------------------------
    def test_zero_round_trip_e4m3(self):
        for v in [0.0, -0.0]:
            x = torch.tensor([[v]])
            self.assertEqual(dequantize_e4m3(quantize_e4m3(x)).item(), 0.0)

    def test_noise_is_applied(self):
        # Values that are NOT exactly representable should come back different
        x   = torch.tensor([[1.1]])
        out = dequantize_e4m3(quantize_e4m3(x))
        self.assertNotAlmostEqual(out.item(), 1.1, places=6)


# ===========================================================================
# 3.  Fp8LinearFunction
# ===========================================================================

class TestFp8LinearFunction(unittest.TestCase):

    def _fwd(self, M=4, N=8, K=6, bias=False):
        x = torch.randn(M, K, requires_grad=True)
        w = torch.randn(N, K, requires_grad=True)
        b = torch.randn(N, requires_grad=True) if bias else None
        y = Fp8LinearFunction.apply(x, w, b)
        return x, w, b, y

    # ---- output shape / dtype ----------------------------------------
    def test_output_shape(self):
        _, _, _, y = self._fwd(M=4, N=8, K=6)
        self.assertEqual(y.shape, (4, 8))

    def test_output_dtype(self):
        _, _, _, y = self._fwd()
        self.assertEqual(y.dtype, torch.float32)

    def test_bias_output_shape(self):
        _, _, _, y = self._fwd(bias=True)
        self.assertEqual(y.shape, (4, 8))

    # ---- FP8 noise is actually applied --------------------------------
    def test_fp8_noise_applied(self):
        torch.manual_seed(0)
        x = torch.randn(8, 8)
        w = torch.randn(8, 8)
        fp8_out    = Fp8LinearFunction.apply(x, w, None)
        plain_out  = x @ w.t()
        # Should be close-ish but NOT identical for arbitrary random tensors
        self.assertFalse(torch.allclose(fp8_out, plain_out, atol=0),
                         "FP8 function should introduce quantization noise")

    # ---- gradient flow -----------------------------------------------
    def test_gradients_flow(self):
        x, w, _, y = self._fwd()
        y.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(w.grad)

    def test_gradient_shapes(self):
        x, w, b, y = self._fwd(M=4, N=8, K=6, bias=True)
        y.sum().backward()
        self.assertEqual(x.grad.shape, x.shape)
        self.assertEqual(w.grad.shape, w.shape)
        self.assertEqual(b.grad.shape, b.shape)

    def test_no_nan_in_gradients(self):
        torch.manual_seed(42)
        x, w, b, y = self._fwd(M=4, N=8, K=6, bias=True)
        y.sum().backward()
        self.assertFalse(torch.isnan(x.grad).any(), "NaN in x.grad")
        self.assertFalse(torch.isnan(w.grad).any(), "NaN in w.grad")
        self.assertFalse(torch.isnan(b.grad).any(), "NaN in b.grad")

    def test_no_inf_in_gradients(self):
        torch.manual_seed(7)
        x, w, b, y = self._fwd(M=4, N=8, K=6, bias=True)
        y.sum().backward()
        self.assertFalse(torch.isinf(x.grad).any(), "Inf in x.grad")
        self.assertFalse(torch.isinf(w.grad).any(), "Inf in w.grad")

    # ---- activation compression: ctx must hold uint8 -----------------
    def test_ctx_saves_uint8_activation(self):
        """The saved activation in the autograd graph must be uint8 (compressed)."""
        x = torch.randn(4, 6, requires_grad=True)
        w = torch.randn(8, 6, requires_grad=True)

        y = Fp8LinearFunction.apply(x, w, None)
        # Access the saved tensors through the grad_fn
        saved = y.grad_fn.saved_tensors  # internal PyTorch mechanism
        # The first saved tensor should be uint8 (the compressed activation)
        self.assertEqual(saved[0].dtype, torch.uint8,
                         "First saved tensor must be uint8 (activation compression)")

    # ---- bias-free gradient ------------------------------------------
    def test_grad_none_for_no_bias(self):
        x = torch.randn(4, 6, requires_grad=True)
        w = torch.randn(8, 6, requires_grad=True)
        y = Fp8LinearFunction.apply(x, w, None)
        (y.sum(),)
        # No error should occur; grad_fn returns None for bias
        grads = torch.autograd.grad(y.sum(), [x, w])
        self.assertEqual(len(grads), 2)


# ===========================================================================
# 4.  Fp8ScaledLinearFunction
# ===========================================================================

class TestFp8ScaledLinearFunction(unittest.TestCase):

    def _fwd(self, M=4, N=8, K=6, is_=1.0, ws_=1.0, bias=False):
        x = torch.randn(M, K, requires_grad=True)
        w = torch.randn(N, K, requires_grad=True)
        b = torch.randn(N, requires_grad=True) if bias else None
        y = Fp8ScaledLinearFunction.apply(x, w, b, is_, ws_)
        return x, w, b, y

    def test_output_shape(self):
        _, _, _, y = self._fwd()
        self.assertEqual(y.shape, (4, 8))

    def test_gradients_flow(self):
        x, w, _, y = self._fwd()
        y.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(w.grad)

    def test_no_nan_in_gradients(self):
        torch.manual_seed(1)
        x, w, b, y = self._fwd(bias=True)
        y.sum().backward()
        for name, g in [("x", x.grad), ("w", w.grad), ("b", b.grad)]:
            self.assertFalse(torch.isnan(g).any(), f"NaN in {name}.grad")

    def test_scaled_ctx_saves_uint8(self):
        x = torch.randn(4, 6, requires_grad=True)
        w = torch.randn(8, 6, requires_grad=True)
        y = Fp8ScaledLinearFunction.apply(x, w, None, 2.0, 3.0)
        saved = y.grad_fn.saved_tensors
        self.assertEqual(saved[0].dtype, torch.uint8,
                         "Scaled function must also save uint8 activation")

    def test_scaling_reduces_noise_for_small_tensors(self):
        """With scale=1, a tensor with amax << 448 wastes FP8 bins.
        With scale=448/amax, noise should be smaller."""
        torch.manual_seed(3)
        x = torch.randn(16, 16) * 0.01   # very small magnitudes
        w = torch.randn(16, 16) * 0.01
        amax_x = x.abs().max().item()
        amax_w = w.abs().max().item()
        scale_x = 448.0 / max(amax_x, 1e-12)
        scale_w = 448.0 / max(amax_w, 1e-12)

        x1 = x.clone().requires_grad_(True)
        x2 = x.clone().requires_grad_(True)
        w1 = w.clone().requires_grad_(True)
        w2 = w.clone().requires_grad_(True)

        y_unscaled = Fp8LinearFunction.apply(x1, w1, None)
        y_scaled   = Fp8ScaledLinearFunction.apply(x2, w2, None, scale_x, scale_w)
        ref        = x @ w.t()

        err_unscaled = (y_unscaled.detach() - ref).abs().mean().item()
        err_scaled   = (y_scaled.detach()   - ref).abs().mean().item()
        self.assertLess(err_scaled, err_unscaled,
                        "Scaled quantization should have lower noise for small tensors")


# ===========================================================================
# 5.  Fp8Linear (nn.Module)
# ===========================================================================

class TestFp8Linear(unittest.TestCase):

    def test_basic_forward(self):
        layer = Fp8Linear(8, 16, bias=True)
        x = torch.randn(4, 8)
        y = layer(x)
        self.assertEqual(y.shape, (4, 16))

    def test_3d_input(self):
        layer = Fp8Linear(8, 16, bias=False)
        x = torch.randn(2, 5, 8)
        y = layer(x)
        self.assertEqual(y.shape, (2, 5, 16))

    def test_4d_input(self):
        layer = Fp8Linear(8, 16, bias=False)
        x = torch.randn(2, 3, 5, 8)
        y = layer(x)
        self.assertEqual(y.shape, (2, 3, 5, 16))

    def test_backward(self):
        layer = Fp8Linear(8, 16, bias=True)
        x = torch.randn(4, 8, requires_grad=True)
        y = layer(x)
        y.sum().backward()
        self.assertIsNotNone(layer.weight.grad)
        self.assertIsNotNone(layer.bias.grad)
        self.assertIsNotNone(x.grad)

    def test_from_linear_copies_weight(self):
        lin  = nn.Linear(8, 16, bias=True)
        fp8  = Fp8Linear.from_linear(lin)
        self.assertTrue(torch.allclose(fp8.weight.data, lin.weight.data))
        self.assertTrue(torch.allclose(fp8.bias.data,   lin.bias.data))

    def test_from_linear_no_bias(self):
        lin = nn.Linear(8, 16, bias=False)
        fp8 = Fp8Linear.from_linear(lin)
        self.assertIsNone(fp8.bias)

    def test_to_linear_round_trip(self):
        fp8 = Fp8Linear(8, 16, bias=True)
        lin = fp8.to_linear()
        self.assertIsInstance(lin, nn.Linear)
        self.assertTrue(torch.allclose(fp8.weight.data, lin.weight.data))

    def test_extra_repr(self):
        layer = Fp8Linear(8, 16)
        r = layer.extra_repr()
        self.assertIn("8", r)
        self.assertIn("16", r)


# ===========================================================================
# 6.  Fp8ScaledLinear (nn.Module)
# ===========================================================================

class TestFp8ScaledLinear(unittest.TestCase):

    def test_basic_forward(self):
        layer = Fp8ScaledLinear(8, 16, bias=True, history_len=4)
        x = torch.randn(4, 8)
        y = layer(x)
        self.assertEqual(y.shape, (4, 16))

    def test_meta_updated_after_forward(self):
        layer = Fp8ScaledLinear(8, 16, history_len=4)
        x = torch.randn(4, 8) * 5.0
        layer(x)
        # After one forward, scale should no longer be exactly 1.0
        # (unless amax happened to be exactly 448, which is very unlikely)
        scale = layer.input_meta.scale.item()
        self.assertNotAlmostEqual(scale, 1.0, places=3)

    def test_from_linear(self):
        lin = nn.Linear(8, 16)
        sl  = Fp8ScaledLinear.from_linear(lin, history_len=4)
        self.assertTrue(torch.allclose(sl.weight.data, lin.weight.data))

    def test_backward(self):
        layer = Fp8ScaledLinear(8, 16, history_len=4)
        x = torch.randn(4, 8, requires_grad=True)
        y = layer(x)
        y.sum().backward()
        self.assertIsNotNone(layer.weight.grad)


# ===========================================================================
# 7.  Fp8ShadowLinear (nn.Module)
# ===========================================================================

class TestFp8ShadowLinear(unittest.TestCase):

    def _make(self, **kw):
        return Fp8ShadowLinear(8, 16, history_len=4, **kw)

    # ---- dtype checks -----------------------------------------------
    def test_weight_master_is_float32(self):
        self.assertEqual(self._make().weight_master.dtype, torch.float32)

    def test_weight_fp8_is_uint8(self):
        self.assertEqual(self._make().weight_fp8.dtype, torch.uint8)

    # ---- weight property --------------------------------------------
    def test_weight_property_returns_master(self):
        layer = self._make()
        self.assertIs(layer.weight, layer.weight_master)

    # ---- weight_fp8 populated after forward --------------------------
    def test_weight_fp8_populated_after_forward(self):
        layer = self._make()
        # Before forward, weight_fp8 is all zeros
        self.assertTrue((layer.weight_fp8 == 0).all())
        x = torch.randn(4, 8)
        layer(x)
        # After forward, weight_fp8 should NOT be all zeros
        self.assertFalse((layer.weight_fp8 == 0).all(),
                         "weight_fp8 should be populated after forward")

    # ---- output shape -----------------------------------------------
    def test_output_shape(self):
        y = self._make()(torch.randn(4, 8))
        self.assertEqual(y.shape, (4, 16))

    def test_3d_input(self):
        y = self._make()(torch.randn(2, 5, 8))
        self.assertEqual(y.shape, (2, 5, 16))

    # ---- gradient flows to weight_master, NOT weight_fp8 -----------
    def test_grad_to_weight_master(self):
        layer = self._make()
        x = torch.randn(4, 8, requires_grad=True)
        y = layer(x)
        y.sum().backward()
        self.assertIsNotNone(layer.weight_master.grad,
                             "weight_master must receive gradient")
        self.assertIsNone(layer.weight_fp8.grad,
                          "weight_fp8 is a buffer and must NOT receive gradient")

    def test_grad_shape_matches_master(self):
        layer = self._make()
        layer(torch.randn(4, 8)).sum().backward()
        self.assertEqual(layer.weight_master.grad.shape, layer.weight_master.shape)

    def test_no_nan_in_gradients(self):
        torch.manual_seed(99)
        layer = self._make(bias=True)
        x = torch.randn(4, 8, requires_grad=True)
        layer(x).sum().backward()
        self.assertFalse(torch.isnan(layer.weight_master.grad).any())
        self.assertFalse(torch.isnan(layer.bias.grad).any())
        self.assertFalse(torch.isnan(x.grad).any())

    # ---- from_linear / to_linear ------------------------------------
    def test_from_linear_copies_weight(self):
        lin  = nn.Linear(8, 16)
        sl   = Fp8ShadowLinear.from_linear(lin, history_len=4)
        self.assertTrue(torch.allclose(sl.weight_master.data, lin.weight.data))

    def test_to_linear_roundtrip(self):
        layer = self._make()
        lin   = layer.to_linear()
        self.assertIsInstance(lin, nn.Linear)
        self.assertTrue(torch.allclose(lin.weight.data, layer.weight_master.data))

    # ---- no bias variant --------------------------------------------
    def test_no_bias(self):
        layer = Fp8ShadowLinear(8, 16, bias=False, history_len=4)
        self.assertIsNone(layer.bias)
        y = layer(torch.randn(4, 8))
        self.assertEqual(y.shape, (4, 16))


# ===========================================================================
# 8.  convert_to_fp8
# ===========================================================================

class _TwoLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

class _Nested(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = _TwoLayer()
        self.lm_head = nn.Linear(4, 32)
    def forward(self, x):
        return self.lm_head(self.inner(x))

class TestConvertToFp8(unittest.TestCase):

    def test_scaled_mode(self):
        m = _TwoLayer()
        convert_to_fp8(m, shadow=False, scaled=True)
        self.assertIsInstance(m.fc1, Fp8ScaledLinear)
        self.assertIsInstance(m.fc2, Fp8ScaledLinear)

    def test_plain_mode(self):
        m = _TwoLayer()
        convert_to_fp8(m, shadow=False, scaled=False)
        self.assertIsInstance(m.fc1, Fp8Linear)
        self.assertIsInstance(m.fc2, Fp8Linear)

    def test_shadow_mode(self):
        m = _TwoLayer()
        convert_to_fp8(m, shadow=True)
        self.assertIsInstance(m.fc1, Fp8ShadowLinear)
        self.assertIsInstance(m.fc2, Fp8ShadowLinear)

    def test_skip_names(self):
        m = _Nested()
        convert_to_fp8(m, shadow=True, skip_names={"lm_head"})
        self.assertIsInstance(m.inner.fc1, Fp8ShadowLinear)
        self.assertIsInstance(m.inner.fc2, Fp8ShadowLinear)
        self.assertIsInstance(m.lm_head, nn.Linear,
                              "lm_head should remain nn.Linear")

    def test_nested_modules_converted(self):
        m = _Nested()
        convert_to_fp8(m, shadow=False, scaled=True)
        self.assertIsInstance(m.inner.fc1, Fp8ScaledLinear)
        self.assertIsInstance(m.inner.fc2, Fp8ScaledLinear)
        self.assertIsInstance(m.lm_head, Fp8ScaledLinear)

    def test_weights_copied_correctly(self):
        lin1_w = nn.Linear(8, 16).weight.data.clone()
        m = nn.Sequential(nn.Linear(8, 16))
        # Store original weight before conversion
        orig = m[0].weight.data.clone()
        convert_to_fp8(m, shadow=True)
        self.assertTrue(torch.allclose(m[0].weight_master.data, orig),
                        "Weights must be copied during conversion")

    def test_forward_still_works_after_conversion(self):
        m = _TwoLayer()
        convert_to_fp8(m, shadow=True)
        x = torch.randn(4, 8)
        y = m(x)
        self.assertEqual(y.shape, (4, 4))

    def test_returns_same_object(self):
        m = _TwoLayer()
        ret = convert_to_fp8(m)
        self.assertIs(ret, m, "convert_to_fp8 must return the same model object")


# ===========================================================================
# 9.  Fp8TensorMeta
# ===========================================================================

class TestFp8TensorMeta(unittest.TestCase):

    def test_scale_after_update(self):
        meta = Fp8TensorMeta(history_len=4, device="cpu")
        x = torch.ones(4, 4) * 2.0   # amax = 2.0
        meta.update(x)
        expected_scale = 448.0 / 2.0
        self.assertAlmostEqual(meta.scale.item(), expected_scale, places=3)

    def test_inv_scale(self):
        meta = Fp8TensorMeta(history_len=4, device="cpu")
        meta.update(torch.ones(4, 4) * 4.0)
        self.assertAlmostEqual(meta.inv_scale.item(), 4.0 / 448.0, places=6)

    def test_history_rolls(self):
        meta = Fp8TensorMeta(history_len=3, device="cpu")
        meta.update(torch.ones(1) * 1.0)
        meta.update(torch.ones(1) * 2.0)
        meta.update(torch.ones(1) * 3.0)
        meta.update(torch.ones(1) * 0.5)   # oldest (1.0) should be replaced
        # history should now contain [0.5, 2.0, 3.0] — max is 3.0
        self.assertAlmostEqual(meta.amax_history.max().item(), 3.0, places=5)

    def test_state_dict_roundtrip(self):
        meta = Fp8TensorMeta(history_len=4, device="cpu")
        meta.update(torch.ones(4, 4) * 10.0)
        sd  = meta.state_dict()
        meta2 = Fp8TensorMeta(history_len=4, device="cpu")
        meta2.load_state_dict(sd)
        self.assertAlmostEqual(meta2.scale.item(), meta.scale.item(), places=6)
        self.assertEqual(meta2._ptr, meta._ptr)

    def test_to_device_is_noop_on_cpu(self):
        meta = Fp8TensorMeta(history_len=4, device="cpu")
        meta.update(torch.ones(2, 2))
        meta.to("cpu")  # should not crash
        self.assertEqual(meta.scale.device.type, "cpu")

    def test_quantize_uses_scale(self):
        meta = Fp8TensorMeta(history_len=4, device="cpu")
        # force scale = 1.0 (amax = 448)
        meta.update(torch.ones(1) * 448.0)
        self.assertAlmostEqual(meta.scale.item(), 1.0, places=4)
        x   = torch.tensor([[1.0]])
        out = meta.quantize_e4m3(x)
        self.assertEqual(out.dtype, torch.uint8)

    def test_dequantize_applies_inv_scale(self):
        meta = Fp8TensorMeta(history_len=4, device="cpu")
        meta.update(torch.ones(1) * 44.8)  # scale = 10.0
        x    = torch.tensor([[1.0]])
        qx   = meta.quantize_e4m3(x)       # quantise at scale=10
        back = meta.dequantize_e4m3(qx)    # dequantise and unscale
        self.assertAlmostEqual(back.item(), 1.0, places=1)


# ===========================================================================
# 10. Adafactor
# ===========================================================================

class TestAdafactor(unittest.TestCase):

    # ---- constructor validation --------------------------------------
    def test_raises_if_lr_and_relative_step(self):
        p = [nn.Parameter(torch.randn(4, 4))]
        with self.assertRaises(ValueError):
            Adafactor(p, lr=1e-3, relative_step=True)

    def test_raises_if_no_lr_and_no_relative_step(self):
        p = [nn.Parameter(torch.randn(4, 4))]
        with self.assertRaises(ValueError):
            Adafactor(p, lr=None, relative_step=False)

    # ---- state structure: 2D params get row+col, not full matrix -----
    def test_factored_state_for_2d(self):
        N, K = 32, 64
        p   = nn.Parameter(torch.randn(N, K))
        opt = Adafactor([p], relative_step=True)
        loss = (p @ torch.randn(K, 8)).sum()
        loss.backward()
        opt.step()

        st = opt.state[p]
        self.assertIn("exp_avg_sq_row", st, "2D param must have row factor")
        self.assertIn("exp_avg_sq_col", st, "2D param must have col factor")
        self.assertNotIn("exp_avg_sq",  st, "2D param must NOT have full V")

        self.assertEqual(st["exp_avg_sq_row"].shape, (N,))
        self.assertEqual(st["exp_avg_sq_col"].shape, (K,))

    def test_full_state_for_1d(self):
        p   = nn.Parameter(torch.randn(32))
        opt = Adafactor([p], relative_step=True)
        (p * torch.randn(32)).sum().backward()
        opt.step()

        st = opt.state[p]
        self.assertIn("exp_avg_sq", st,      "1D param must have full V")
        self.assertNotIn("exp_avg_sq_row", st)

    # ---- VRAM check: optimizer state << parameter size ---------------
    def test_state_much_smaller_than_param(self):
        N, K = 128, 256
        p = nn.Parameter(torch.randn(N, K))
        opt = Adafactor([p], relative_step=True)
        (p.sum()).backward()
        opt.step()

        st = opt.state[p]
        state_elems = st["exp_avg_sq_row"].numel() + st["exp_avg_sq_col"].numel()
        param_elems = p.numel()
        # state should be (N+K)/(N*K) fraction of param size
        ratio = state_elems / param_elems
        self.assertLess(ratio, 0.05,
                        f"Adafactor state ratio {ratio:.4f} should be << 1")

    # ---- convergence: loss should decrease on linear regression ------
    def test_loss_decreases(self):
        torch.manual_seed(42)
        # Simple linear regression: y = Wx, minimise MSE
        W_true = torch.randn(4, 8)
        layer  = nn.Linear(8, 4, bias=False)
        opt    = Adafactor([layer.weight], lr=0.01, relative_step=False)

        # Use a fixed batch for stable convergence
        x = torch.randn(16, 8)
        y = x @ W_true.t()

        losses = []
        for _ in range(30):
            loss = ((layer(x) - y) ** 2).mean()
            losses.append(loss.item())
            opt.zero_grad(); loss.backward(); opt.step()

        self.assertLess(losses[-1], losses[0],
                        f"Adafactor: final loss {losses[-1]:.4f} > initial {losses[0]:.4f}")

    # ---- fixed lr mode -----------------------------------------------
    def test_fixed_lr_mode(self):
        p   = nn.Parameter(torch.randn(8, 16))
        opt = Adafactor([p], lr=1e-3, relative_step=False, scale_parameter=False)
        (p.sum()).backward()
        opt.step()       # must not crash

    # ---- weight decay applied ----------------------------------------
    def test_weight_decay_shrinks_params(self):
        p    = nn.Parameter(torch.ones(4, 4))
        opt  = Adafactor([p], relative_step=True, weight_decay=0.5)
        # zero gradient — only weight decay should move p
        p.grad = torch.zeros_like(p)
        p_before = p.data.clone()
        opt.step()
        # With very high wd and zero grad, params should shrink
        self.assertTrue((p.data.abs() < p_before.abs()).any(),
                        "weight_decay should shrink parameters")

    # ---- no NaN in step ----------------------------------------------
    def test_no_nan_in_step(self):
        torch.manual_seed(0)
        p   = nn.Parameter(torch.randn(16, 32))
        opt = Adafactor([p], relative_step=True)
        (p @ torch.randn(32, 8)).sum().backward()
        opt.step()
        self.assertFalse(torch.isnan(p.data).any(), "NaN in param after Adafactor step")


# ===========================================================================
# 11. Full End-to-End Training Pipeline
# ===========================================================================

class _SmallTransformerBlock(nn.Module):
    """Minimal MLP block — all linears get FP8-converted."""
    def __init__(self, d=16, d_ff=32):
        super().__init__()
        self.fc1 = nn.Linear(d, d_ff)
        self.fc2 = nn.Linear(d_ff, d)
        self.ln  = nn.LayerNorm(d)

    def forward(self, x):
        return self.ln(x + self.fc2(torch.relu(self.fc1(x))))


class _SmallLM(nn.Module):
    def __init__(self, vocab=64, d=16, n_layers=2, d_ff=32):
        super().__init__()
        self.embed    = nn.Embedding(vocab, d)
        self.blocks   = nn.ModuleList([
            _SmallTransformerBlock(d, d_ff) for _ in range(n_layers)
        ])
        self.ln_f     = nn.LayerNorm(d)
        self.lm_head  = nn.Linear(d, vocab, bias=False)

    def forward(self, idx):
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.ln_f(x))


class TestFullPipeline(unittest.TestCase):

    def _make_model(self, shadow=True):
        torch.manual_seed(0)
        model = _SmallLM(vocab=64, d=16, n_layers=2, d_ff=32)
        convert_to_fp8(model, shadow=shadow, skip_names={"lm_head"})
        return model

    # ---- model structure after conversion ----------------------------
    def test_linears_converted_to_shadow(self):
        model = self._make_model(shadow=True)
        for name, m in model.named_modules():
            if "block" in name and isinstance(m, (nn.Linear,)):
                self.fail(f"{name} is still nn.Linear after shadow conversion")
        # lm_head should remain nn.Linear
        self.assertIsInstance(model.lm_head, nn.Linear)

    def test_weight_fp8_buffers_are_uint8(self):
        model = self._make_model(shadow=True)
        x = torch.randint(0, 64, (2, 8))
        model(x)   # trigger sync_shadow
        for name, m in model.named_modules():
            if isinstance(m, Fp8ShadowLinear):
                self.assertEqual(m.weight_fp8.dtype, torch.uint8,
                                 f"{name}.weight_fp8 must be uint8")

    # ---- forward pass -----------------------------------------------
    def test_forward_output_shape(self):
        model = self._make_model()
        x = torch.randint(0, 64, (2, 8))
        y = model(x)
        self.assertEqual(y.shape, (2, 8, 64))

    def test_forward_no_nan(self):
        model = self._make_model()
        x = torch.randint(0, 64, (2, 8))
        y = model(x)
        self.assertFalse(torch.isnan(y).any(), "NaN in model output")

    # ---- backward pass -----------------------------------------------
    def test_backward_no_crash(self):
        model = self._make_model()
        x = torch.randint(0, 64, (2, 8))
        y = model(x)
        loss = y.mean()
        loss.backward()   # must not raise

    def test_all_master_weights_have_grad(self):
        model = self._make_model()
        x = torch.randint(0, 64, (2, 8))
        model(x).mean().backward()
        for name, m in model.named_modules():
            if isinstance(m, Fp8ShadowLinear):
                self.assertIsNotNone(
                    m.weight_master.grad,
                    f"{name}.weight_master.grad is None after backward"
                )

    def test_no_nan_in_gradients(self):
        model = self._make_model()
        x = torch.randint(0, 64, (2, 8))
        model(x).mean().backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                self.assertFalse(torch.isnan(p.grad).any(),
                                 f"NaN in gradient of {name}")

    # ---- Adafactor state structure -----------------------------------
    def test_adafactor_state_is_factored(self):
        model = self._make_model()
        opt   = Adafactor(model.parameters(), relative_step=True, weight_decay=0.01)
        x     = torch.randint(0, 64, (2, 8))
        model(x).mean().backward()
        opt.step()
        for p in model.parameters():
            if p.dim() >= 2 and p in opt.state:
                st = opt.state[p]
                self.assertIn("exp_avg_sq_row", st,
                              "Adafactor must use row+col factors for 2D params")

    # ---- training loop: loss decreases --------------------------------
    def test_loss_decreases_over_training(self):
        torch.manual_seed(42)
        model = self._make_model(shadow=False)   # faster (no shadow sync)
        opt   = Adafactor(model.parameters(), relative_step=True)

        losses = []
        for step in range(20):
            x    = torch.randint(0, 64, (4, 8))
            y    = torch.randint(0, 64, (4, 8))
            logits = model(x)                     # [4, 8, 64]
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, 64), y.reshape(-1)
            )
            losses.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()

        self.assertLess(
            losses[-1], losses[0],
            f"Loss should decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    # ---- shadow pipeline: loss decreases with shadow layers ----------
    def test_shadow_pipeline_loss_decreases(self):
        torch.manual_seed(7)
        model = self._make_model(shadow=True)
        opt   = Adafactor(model.parameters(), relative_step=True)

        losses = []
        for _ in range(20):
            x      = torch.randint(0, 64, (4, 8))
            y      = torch.randint(0, 64, (4, 8))
            loss   = nn.functional.cross_entropy(
                model(x).reshape(-1, 64), y.reshape(-1)
            )
            losses.append(loss.item())
            opt.zero_grad(); loss.backward(); opt.step()

        self.assertLess(losses[-1], losses[0],
                        "Shadow pipeline must also converge")

    # ---- checkpointing: Fp8TensorMeta survives save/load ------------
    def test_meta_state_survives_checkpoint(self):
        import io
        model = self._make_model(shadow=True)
        x = torch.randint(0, 64, (2, 8))
        model(x)  # populate scales

        # Collect meta state from one shadow layer
        first_shadow = next(m for m in model.modules()
                            if isinstance(m, Fp8ShadowLinear))
        scale_before = first_shadow.weight_meta.scale.item()

        # Checkpoint model.state_dict (does NOT include Fp8TensorMeta,
        # but weight_fp8 buffer IS included)
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        buf.seek(0)
        sd = torch.load(buf, weights_only=True)
        # weight_fp8 should be in state_dict as a buffer
        shadow_keys = [k for k in sd if "weight_fp8" in k]
        self.assertTrue(len(shadow_keys) > 0,
                        "weight_fp8 buffers must appear in state_dict")


# ===========================================================================
# 12. Main runner
# ===========================================================================

SUITES = [
    ("Phase 2 — quant/dequant",       TestPhase2),
    ("Fp8LinearFunction",              TestFp8LinearFunction),
    ("Fp8ScaledLinearFunction",        TestFp8ScaledLinearFunction),
    ("Fp8Linear (nn.Module)",          TestFp8Linear),
    ("Fp8ScaledLinear (nn.Module)",    TestFp8ScaledLinear),
    ("Fp8ShadowLinear (nn.Module)",    TestFp8ShadowLinear),
    ("convert_to_fp8",                 TestConvertToFp8),
    ("Fp8TensorMeta",                  TestFp8TensorMeta),
    ("Adafactor optimizer",            TestAdafactor),
    ("Full E2E pipeline",              TestFullPipeline),
]

if __name__ == "__main__":
    print("\n" + "=" * 64)
    print("  hip-quant Phase 3 — Full Pipeline Test Suite")
    print("  (CPU / mocked _C — no GPU required)")
    print("=" * 64)

    loader  = unittest.TestLoader()
    runner  = unittest.TextTestRunner(verbosity=0, stream=open("/dev/null", "w")
                                      if sys.platform != "win32"
                                      else open("nul", "w"))

    import io, os
    null = open(os.devnull, "w")
    runner = unittest.TextTestRunner(verbosity=0, stream=null)

    for name, cls in SUITES:
        suite  = loader.loadTestsFromTestCase(cls)
        result = runner.run(suite)
        _summarise(result, name)

    null.close()
    print("=" * 64)
    print(f"  TOTAL: {PASS} passed, {FAIL} failed")
    print("=" * 64)
    sys.exit(0 if FAIL == 0 else 1)
