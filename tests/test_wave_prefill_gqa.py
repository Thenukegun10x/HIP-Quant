"""Intentional GPU correctness checks for the inference prefill kernel.

Run directly with the ROCm venv; includes ragged and cached causal queries,
multiple batches, grouped heads, IU4 metadata, and a nondefault GPU stream.
"""
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT.parent), str(ROOT / 'hip_inference')]

import torch
import torch.nn.functional as F
from hip_quant import torch_api as T


class TestPrefillGQA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest('GPU required')
        cls.ext = T._load_extension()
        print(f'extension={cls.ext.__file__}',flush=True)
        cls.ext.wave_attn_prefill_gqa_forward  # A missing native path must fail.

    def check_case(self, dim, sq, sk, batch=1, heads=6, kv_heads=2, iu4=False, causal=True, groups=4):
        torch.manual_seed(dim+sq+sk)
        q = torch.randn(batch,heads,sq,dim,device='cuda',dtype=torch.float16)
        k = torch.randn(batch,kv_heads,sk,dim,device='cuda',dtype=torch.float16)
        v = torch.randn_like(k) * 0.7 + 0.2
        qs, ks, vscl = 0.5, 0.25, 0.75
        q8, k8 = T.quantize_e4m3(q / qs), T.quantize_e4m3(k / ks)
        scales = zp = None
        if iu4:
            packed, scales, zp = T.kv_quant_v_i4(v.view(batch*kv_heads,sk,dim),groups)
            vd = T.kv_dequant_v_i4(packed,scales,zp).view_as(v)
            v8 = T.quantize_e4m3(vd)
            vi = packed.view(batch,kv_heads,sk,dim//2)
            scales, zp = [a.view(batch,kv_heads,sk,groups) for a in (scales,zp)]
        else:
            v8 = T.quantize_e4m3(v)
            vi = v8
        out, lse = self.ext.wave_attn_prefill_gqa_forward(q8,k8,vi,dim**-.5,qs,ks,vscl,causal,scales,zp)
        # Independent FP32 reference using the quantized inputs. This isolates
        # kernel indexing/masking errors from the unavoidable FP8 input error.
        qd, kd, vd = [T.dequantize_e4m3(a) for a in (q8,k8,v8)]
        kd = kd.repeat_interleave(heads//kv_heads,dim=1)
        vd = vd.repeat_interleave(heads//kv_heads,dim=1)
        scores = qd @ kd.transpose(-1,-2) * (dim**-.5 * qs * ks)
        if causal:
            mask = torch.arange(sk,device='cuda')[None,:] > sk-sq+torch.arange(sq,device='cuda')[:,None]
            scores.masked_fill_(mask, -torch.inf)
        expected_lse = scores.logsumexp(-1)
        expected = scores.softmax(-1) @ vd * vscl
        self.assertTrue(torch.isfinite(out).all())
        self.assertGreater(F.cosine_similarity(out.float().flatten(),expected.flatten(),dim=0).item(), .9993)
        self.assertLess(((out.float()-expected).norm()/expected.norm()).item(), .04)
        torch.testing.assert_close(lse,expected_lse,atol=2e-5,rtol=2e-5)
        if iu4:
            dense, _ = self.ext.wave_attn_prefill_gqa_forward(q8,k8,v8,dim**-.5,qs,ks,vscl,causal)
            torch.testing.assert_close(out,dense,atol=2e-4,rtol=2e-3)

    def test_shapes_masks_and_iu4(self):
        for dim in (64,128,256):
            for sq,sk in ((3,3),(17,65),(64,64),(65,129),(129,129),(257,513)):
                for iu4 in (False,True):
                    with self.subTest(dim=dim,sq=sq,sk=sk,iu4=iu4):
                        self.check_case(dim,sq,sk,iu4=iu4)
        self.check_case(128,67,129,batch=2,heads=4,kv_heads=4,causal=False,iu4=True,groups=8)

    def test_nondefault_stream(self):
        with torch.cuda.stream(torch.cuda.Stream()):
            self.check_case(256,67,191,iu4=True)
        torch.cuda.synchronize()

    def test_invalid_inputs(self):
        q=torch.zeros(1,6,64,64,device='cuda',dtype=torch.uint8)
        k=torch.zeros(1,2,64,64,device='cuda',dtype=torch.uint8)
        with self.assertRaises(RuntimeError):
            self.ext.wave_attn_prefill_gqa_forward(q,k[:,:1],k,0.125)
        with self.assertRaises(RuntimeError):
            self.ext.wave_attn_prefill_gqa_forward(q,k.transpose(2,3),k,0.125)
        with self.assertRaises(RuntimeError):
            self.ext.wave_attn_prefill_gqa_forward(q,k,k,0.125,v_scales=torch.ones(1,2,64,4,device='cuda',dtype=torch.float16))


if __name__ == '__main__':
    result = unittest.main(exit=False).result
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if result.wasSuccessful() else 1)
