"""CPU-safe contract tests for the GQA prefill ``with_lse`` Python API.

Covers ``hip_quant.torch_api.wave_attn_prefill_with_lse`` /
``wave_attn_prefill`` / ``wave_attn_gqa_decode`` with a mocked ``_C``
extension — no GPU, no compiled extension required. GPU numerics for the
underlying ``wave_attn_prefill_gqa_forward`` native entry live in
``tests/test_wave_prefill_gqa.py`` (ROCm venv, intentional GPU run).

Run from the repo root:
    python -m pytest tests/test_prefill_with_lse.py -q
    python tests/test_prefill_with_lse.py -v
"""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT.parent), str(ROOT)]

CALLS: list[dict] = []


def _mock_quantize_e4m3(x: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(x, dtype=torch.uint8)


def _mock_gqa_forward(q8, k8, v8, softmax_scale,
                      q_scale=1.0, k_scale=1.0, v_scale=1.0,
                      is_causal=True, v_scales=None, v_zp=None):
    CALLS.append({
        "q8": q8, "k8": k8, "v8": v8,
        "softmax_scale": softmax_scale,
        "q_scale": q_scale, "k_scale": k_scale, "v_scale": v_scale,
        "is_causal": is_causal,
        "v_scales": v_scales, "v_zp": v_zp,
    })
    b, h, sq, d = q8.shape
    out = torch.zeros(b, h, sq, d, dtype=torch.float16)
    lse = torch.zeros(b, h, sq, dtype=torch.float32)
    return out, lse


_MOCK_C = types.SimpleNamespace(
    quantize_e4m3=_mock_quantize_e4m3,
    wave_attn_prefill_gqa_forward=_mock_gqa_forward,
)

import hip_quant.torch_api as T  # noqa: E402

# NOTE: module-level code must NOT mutate T._C / T._require_gfx12_fp8_wmma.
# tests/test_pipeline.py injects its own _C mock at import time; a permanent
# patch here would clobber it whenever both files run in one pytest session
# (54 failures). Each test installs this file's mock in setUp and restores
# whatever was there in tearDown.


class TestPrefillWithLse(unittest.TestCase):
    def setUp(self):
        CALLS.clear()
        self._prev_C = T._C
        self._prev_guard = T._require_gfx12_fp8_wmma
        T._C = _MOCK_C
        T._require_gfx12_fp8_wmma = lambda tensor: None  # noqa: E731 — CPU stand-in

    def tearDown(self):
        T._C = self._prev_C
        T._require_gfx12_fp8_wmma = self._prev_guard
        CALLS.clear()

    def _qkv(self, sq=4, sk=8, dim=16, dtype=torch.float32):
        q = torch.randn(1, 2, sq, dim, dtype=dtype)
        k = torch.randn(1, 1, sk, dim, dtype=dtype)
        v = torch.randn(1, 1, sk, dim, dtype=dtype)
        return q, k, v

    def test_with_lse_returns_tuple_shapes_dtypes(self):
        q, k, v = self._qkv()
        out, lse = T.wave_attn_prefill_with_lse(q, k, v)
        self.assertEqual(out.shape, q.shape)
        self.assertEqual(out.dtype, q.dtype)  # cast back to q dtype
        self.assertEqual(lse.shape, (1, 2, 4))
        self.assertEqual(lse.dtype, torch.float32)
        self.assertEqual(len(CALLS), 1)

    def test_defaults_forwarded_to_native(self):
        q, k, v = self._qkv(dim=16)
        T.wave_attn_prefill_with_lse(q, k, v)
        call = CALLS[0]
        self.assertAlmostEqual(call["softmax_scale"], 16 ** -0.5)
        self.assertAlmostEqual(call["q_scale"], 0.5)
        self.assertAlmostEqual(call["k_scale"], 0.5)  # float K branch
        self.assertAlmostEqual(call["v_scale"], 1.0)
        self.assertTrue(call["is_causal"])
        self.assertIsNone(call["v_scales"])
        self.assertIsNone(call["v_zp"])

    def test_prefill_returns_out_only_and_matches(self):
        q, k, v = self._qkv()
        out_only = T.wave_attn_prefill(q, k, v)
        self.assertIsInstance(out_only, torch.Tensor)
        self.assertEqual(out_only.shape, q.shape)
        CALLS.clear()
        out2, _ = T.wave_attn_prefill_with_lse(q, k, v)
        self.assertTrue(torch.equal(out_only, out2))

    def test_gqa_decode_alias(self):
        self.assertIs(T.wave_attn_gqa_decode, T.wave_attn_prefill)

    def test_explicit_scale_and_noncausal(self):
        q, k, v = self._qkv()
        T.wave_attn_prefill(q, k, v, softmax_scale=0.125, is_causal=False)
        call = CALLS[0]
        self.assertAlmostEqual(call["softmax_scale"], 0.125)
        self.assertFalse(call["is_causal"])

    def test_prequantized_k_passthrough(self):
        q, _, v = self._qkv()
        k8 = torch.zeros(1, 1, 8, 16, dtype=torch.uint8)
        T.wave_attn_prefill_with_lse(q, k8, v)
        call = CALLS[0]
        self.assertAlmostEqual(call["k_scale"], 1.0)
        self.assertTrue(torch.equal(call["k8"], k8))

    def test_iu4_metadata_passthrough(self):
        q, k, _ = self._qkv()
        dim = q.shape[-1]
        vp = torch.zeros(1, 1, 8, dim // 2, dtype=torch.uint8)
        vs = torch.ones(1, 1, 8, 4, dtype=torch.float16)
        vz = torch.zeros(1, 1, 8, 4, dtype=torch.uint8)
        out, _ = T.wave_attn_prefill_with_lse(q, k, vp, v_scales=vs, v_zp=vz)
        self.assertEqual(out.shape, q.shape)
        call = CALLS[0]
        self.assertTrue(torch.equal(call["v_scales"], vs))
        self.assertTrue(torch.equal(call["v_zp"], vz))

    def test_scales_without_zp_raises_before_native(self):
        q, k, v = self._qkv()
        vs = torch.ones(1, 1, 8, 4, dtype=torch.float16)
        with self.assertRaises(ValueError):
            T.wave_attn_prefill_with_lse(q, k, v, v_scales=vs)
        self.assertEqual(CALLS, [])

    def test_uint8_q_raises(self):
        q = torch.zeros(1, 2, 4, 16, dtype=torch.uint8)
        _, k, v = self._qkv()
        with self.assertRaises(ValueError):
            T.wave_attn_prefill(q, k, v)
        self.assertEqual(CALLS, [])

    def test_non4d_raises(self):
        with self.assertRaises(ValueError):
            T.wave_attn_prefill(torch.randn(4, 16), torch.randn(4, 16),
                                torch.randn(4, 16))
        self.assertEqual(CALLS, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
