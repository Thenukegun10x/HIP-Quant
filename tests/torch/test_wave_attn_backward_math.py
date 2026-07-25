"""
tests/torch/test_wave_attn_backward_math.py
===========================================

Math regression tests for the native WaveAttention backward kernel.

These cover the three axes that an aligned, unit-scale, well-conditioned
benchmark cannot see, each of which corresponds to a bug that was live at some
point:

  A. Ragged sequence lengths -- exercises the tail path.  Per-lane validity
     predicates are divergent *within* a wave when the length is not a multiple
     of 16, so using one to gate control flow desynchronises __syncthreads()
     and makes __shfl_sync(..., lg*8 + i) broadcasts read retired lanes.

  B. Short Seq_Q against long Seq_K -- the dK/dV block assigns one wave per
     16-key sub-tile, so its thread count must not be derived from Q_TILE.
     When it was, every key beyond the wave coverage stayed identically zero.

  C. Non-unit q/k/v scales, and small-magnitude dO -- the returned gradients
     must be w.r.t. the true-valued tensors (X_true = x_scale * X_fp8), and
     accuracy must not depend on gradient magnitude (dS is linear in dO, so a
     fixed dS scale sends it through E4M3's subnormal floor).

Run with:
    pytest tests/torch/test_wave_attn_backward_math.py -v

Requires:
    - PyTorch with ROCm support
    - hip_quant._C built via: python setup_torch.py build_ext --inplace
    - A gfx1200/gfx1201 GPU visible to torch.cuda
"""

import os
import sys

import pytest

# The repo directory *is* the `hip_quant` package, so its parent must be on the
# path for an in-place (non pip-installed) checkout.
#   .../<parent>/hip_quant/tests/torch/<this file>  ->  .../<parent>
_REPO_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)))
if _REPO_PARENT not in sys.path:
    sys.path.insert(0, _REPO_PARENT)

torch_available = True
try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch_available = False

# NB: probe via torch_api._load_extension() rather than `from hip_quant import
# _C`.  The latter raises a DLL-load error on Windows ROCm unless the package is
# pip-installed, which silently skips the whole module.
extension_available = False
if torch_available and torch.cuda.is_available():
    try:
        from hip_quant.torch_api import _load_extension
        _load_extension()
        extension_available = True
    except Exception:
        pass

pytestmark = pytest.mark.skipif(
    not extension_available,
    reason="Requires PyTorch with ROCm, a visible GPU, and hip_quant._C built",
)

# Cosine floor. FP8 E4M3 carries ~3 mantissa bits; every configuration here
# should land at ~0.999, so 0.99 flags a real defect rather than quantisation.
COS_TOL = 0.99


def _cos(a, b):
    return F.cosine_similarity(
        a.flatten().unsqueeze(0).double(), b.flatten().unsqueeze(0).double()
    ).item()


def _run(B, H, Sq, Sk, D, causal,
         q_scale=1.0, k_scale=1.0, v_scale=1.0, grad_mag=1.0, seed=0):
    """Returns (dq_cos, dk_cos, dv_cos) against an FP32 SDPA reference."""
    from hip_quant.torch_api import (
        _load_extension, quantize_e4m3, dequantize_e4m3,
    )

    torch.manual_seed(seed)
    ext = _load_extension()
    scale = 1.0 / (D ** 0.5)

    q_fp8 = quantize_e4m3(torch.randn((B, H, Sq, D), device="cuda"))
    k_fp8 = quantize_e4m3(torch.randn((B, H, Sk, D), device="cuda"))
    v_fp8 = quantize_e4m3(torch.randn((B, H, Sk, D), device="cuda"))

    # The kernel's contract is attention over the *true* tensors
    # X_true = x_scale * X_fp8, so the reference differentiates w.r.t. those.
    q_t = (dequantize_e4m3(q_fp8) * q_scale).detach().requires_grad_(True)
    k_t = (dequantize_e4m3(k_fp8) * k_scale).detach().requires_grad_(True)
    v_t = (dequantize_e4m3(v_fp8) * v_scale).detach().requires_grad_(True)

    grad_out = torch.randn((B, H, Sq, D), device="cuda") * grad_mag

    with torch.enable_grad():
        out_ref = F.scaled_dot_product_attention(
            q_t, k_t, v_t, attn_mask=None, dropout_p=0.0,
            is_causal=causal, scale=scale,
        )
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(
        out_ref, (q_t, k_t, v_t), grad_outputs=grad_out
    )

    out_hip, lse = ext.wave_attn_forward(
        q_fp8.contiguous(), k_fp8.contiguous(), v_fp8.contiguous(),
        float(scale), q_scale, k_scale, v_scale, bool(causal),
    )
    dq, dk, dv = ext.wave_attn_backward(
        q_fp8.contiguous(), k_fp8.contiguous(), v_fp8.contiguous(),
        out_hip.contiguous().float(), grad_out.contiguous().float(), lse,
        float(scale), q_scale, k_scale, v_scale, bool(causal),
    )
    return _cos(dq, dq_ref), _cos(dk, dk_ref), _cos(dv, dv_ref)


def _assert_all(cosines):
    dq_c, dk_c, dv_c = cosines
    assert dq_c >= COS_TOL, f"dQ cosine {dq_c:.4f}"
    assert dk_c >= COS_TOL, f"dK cosine {dk_c:.4f}"
    assert dv_c >= COS_TOL, f"dV cosine {dv_c:.4f}"


# --- A. Ragged sequence lengths (tail / barrier handling) -------------------
@pytest.mark.parametrize("Sq,Sk,D,causal", [
    (17,  17,  64,  False),
    (33,  33,  64,  False),
    (100, 100, 128, False),
    (130, 130, 128, False),
    (100, 100, 128, True),
    (255, 255, 128, True),
    (48,  200, 128, False),
    (200, 48,  128, False),
])
def test_ragged_sequence_lengths(Sq, Sk, D, causal):
    _assert_all(_run(1, 1, Sq, Sk, D, causal))


# --- B. Short Seq_Q vs long Seq_K (one wave per key sub-tile) ---------------
@pytest.mark.parametrize("Sq,Sk", [
    (16, 64), (16, 128), (32, 128), (32, 256), (16, 256),
])
def test_short_query_long_key(Sq, Sk):
    _assert_all(_run(1, 1, Sq, Sk, 128, False))


def test_short_query_writes_every_key():
    """dK/dV must be written for every key, not just the first wave's slice."""
    from hip_quant.torch_api import _load_extension, quantize_e4m3

    torch.manual_seed(0)
    ext = _load_extension()
    Sq, Sk, D = 16, 128, 128
    scale = 1.0 / (D ** 0.5)

    q_fp8 = quantize_e4m3(torch.randn((1, 1, Sq, D), device="cuda"))
    k_fp8 = quantize_e4m3(torch.randn((1, 1, Sk, D), device="cuda"))
    v_fp8 = quantize_e4m3(torch.randn((1, 1, Sk, D), device="cuda"))
    grad_out = torch.randn((1, 1, Sq, D), device="cuda")

    out, lse = ext.wave_attn_forward(
        q_fp8.contiguous(), k_fp8.contiguous(), v_fp8.contiguous(),
        float(scale), 1.0, 1.0, 1.0, False,
    )
    _, dk, dv = ext.wave_attn_backward(
        q_fp8.contiguous(), k_fp8.contiguous(), v_fp8.contiguous(),
        out.contiguous().float(), grad_out.contiguous().float(), lse,
        float(scale), 1.0, 1.0, 1.0, False,
    )
    assert int((dk[0, 0].abs().sum(-1) == 0).sum()) == 0, "dK has all-zero key rows"
    assert int((dv[0, 0].abs().sum(-1) == 0).sum()) == 0, "dV has all-zero key rows"


# --- C. Scales and gradient magnitude --------------------------------------
@pytest.mark.parametrize("q_scale,k_scale,v_scale", [
    (2.0, 1.0, 1.0),
    (1.0, 2.0, 1.0),
    (1.0, 1.0, 2.0),
    (0.5, 4.0, 0.25),
    (3.0, 3.0, 3.0),
])
def test_non_unit_scales(q_scale, k_scale, v_scale):
    _assert_all(_run(1, 2, 128, 128, 128, False,
                     q_scale=q_scale, k_scale=k_scale, v_scale=v_scale))


def test_non_unit_scales_ragged_causal():
    _assert_all(_run(1, 2, 100, 100, 128, True,
                     q_scale=0.5, k_scale=4.0, v_scale=0.25))


@pytest.mark.parametrize("grad_mag", [1e-1, 1e-2, 1e-3, 1e-4, 1e-6])
def test_small_gradient_magnitude(grad_mag):
    _assert_all(_run(1, 2, 128, 128, 128, False, grad_mag=grad_mag))
