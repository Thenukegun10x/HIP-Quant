"""
hip_quant.attention — SageAttention-style drop-in helpers for WaveAttention

Provides:
  * is_wave_compatible(q,k,v) — shape/arch check without launching
  * wave_available() — device + ROCm + extension check
  * wave_sdpa(*, is_causal, scale) — drop-in for F.scaled_dot_product_attention
  * patch_sdpa() / unpatch_sdpa() — monkey-patch SDPA globally (like SageAttn)
  * patch_transformers(model) — swap nn.Linear → QuantizedLinear + SDPA → wave

RDNA4 only (gfx1200/1201, ROCm 7.2+). All helpers fall back to SDPA on
unsupported hardware/dtypes/shapes so `patch_sdpa()` is safe to call on any GPU.
"""
from __future__ import annotations

import os
import warnings
from typing import Optional

try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

_PATCHED_SDPA = False
_ORIG_SDPA = None
_ARCH_CACHE = {}


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes", "on")


def wave_available() -> bool:
    """True if WaveAttention can run on this machine (gfx12 + ROCm 7.2 + ext)."""
    if not _TORCH_AVAILABLE or not torch.cuda.is_available():
        return False
    if _env_flag("HIP_QUANT_DISABLE_WMMA") or _env_flag("HIP_QUANT_DISABLE_WAVE"):
        return False
    try:
        from .torch_api import _require_gfx12_fp8_wmma  # reuse guard
        # probe with a dummy tensor on current device
        dummy = torch.empty(1, device="cuda")
        _require_gfx12_fp8_wmma(dummy)
        return True
    except Exception:
        return False


def is_wave_compatible(
    q: "torch.Tensor",
    k: "torch.Tensor",
    v: "torch.Tensor",
    is_causal: bool = False,
    scale: Optional[float] = None,
    attn_mask=None,
    dropout_p: float = 0.0,
) -> bool:
    """Fast eligibility check — no quant, no sync, no arch bypass side-effects.

    Returns False if shapes/dtypes/device/flags require SDPA fallback.
    Checks Dim%16 (WMMA tile), 4D [B,H,S,D], matching B/H/D, causal+attn_mask
    incompatibility, dropout, and gfx12 arch.
    """
    if _env_flag("HIP_QUANT_DISABLE_WAVE") or _env_flag("HIP_QUANT_DISABLE_WMMA"):
        return False
    if not _TORCH_AVAILABLE or not torch.cuda.is_available():
        return False
    # dropout / attn_mask not supported by wave
    if dropout_p != 0.0 or attn_mask is not None:
        return False
    if not (torch.is_tensor(q) and torch.is_tensor(k) and torch.is_tensor(v)):
        return False
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        return False
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        return False
    if q.shape[0] != k.shape[0] or q.shape[1] != k.shape[1]:
        return False
    if q.shape[0] != v.shape[0] or q.shape[1] != v.shape[1]:
        return False
    if q.size(3) != k.size(3) or q.size(3) != v.size(3):
        return False
    D = int(q.size(3))
    if D == 0 or D % 16 != 0:  # WMMA 16x16x16 requires Dim%16
        return False
    if D < 16 or D > 256:  # practical tile coverage (K_TILE 64/128)
        return False
    # S can be any, but warn if tiny — still works via Q_TILE=16 path
    # arch / ROCm guard (cached, no hipGetDeviceProperties every call after first)
    try:
        dev = q.device.index if q.device.index is not None else torch.cuda.current_device()
        if dev in _ARCH_CACHE:
            ok = _ARCH_CACHE[dev]
        else:
            props = torch.cuda.get_device_properties(dev)
            arch = getattr(props, "gcnArchName", "") or ""
            ok = arch.startswith("gfx12")
            # also check ROCm version >=7.2 without calling torch_api (avoid warning spam)
            rocm = getattr(torch.version, "hip", "") or ""
            # parse major.minor loosely
            try:
                parts = str(rocm).replace("-", ".").split(".")
                maj = int(parts[0]) if parts[0].isdigit() else 0
                minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
                if (maj, minor) < (7, 2):
                    ok = False
            except Exception:
                pass
            _ARCH_CACHE[dev] = ok
        if not ok:
            return False
    except Exception:
        return False
    # dtypes: wave quantizes f32/f16/bf16/uint8; others fall back
    allowed = (torch.float32, torch.float16, torch.bfloat16, torch.uint8)
    if q.dtype not in allowed or k.dtype not in allowed or v.dtype not in allowed:
        # also allow float8_e4m3fn if present
        try:
            if q.dtype == torch.float8_e4m3fn:
                pass
            else:
                return False
        except AttributeError:
            return False
    return True


def wave_sdpa(
    query: "torch.Tensor",
    key: "torch.Tensor",
    value: "torch.Tensor",
    attn_mask=None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    *,
    force_wave: bool = False,
) -> "torch.Tensor":
    """Drop-in replacement for F.scaled_dot_product_attention.

    Falls back to SDPA when !is_wave_compatible unless force_wave=True
    (then raises the underlying wave error for debugging).
    Respects grad: no_grad / eval → wave_attn_forward_fast, else wave_attn (autograd).
    """
    if not force_wave and not is_wave_compatible(
        query, key, value, is_causal=is_causal, scale=scale, attn_mask=attn_mask, dropout_p=dropout_p
    ):
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
        )
    # try wave, fallback on any RuntimeError unless forced
    try:
        from .torch_api import wave_attn, wave_attn_forward_fast

        # fast path for inference (no grad) saves LSE/save_for_backward
        if not torch.is_grad_enabled() or not (query.requires_grad or key.requires_grad or value.requires_grad):
            return wave_attn_forward_fast(query, key, value, is_causal=is_causal, scale=scale)
        return wave_attn(query, key, value, is_causal=is_causal, scale=scale)
    except Exception as exc:
        if force_wave:
            raise
        warnings.warn(f"hip_quant wave_sdpa fallback to SDPA: {exc}", RuntimeWarning, stacklevel=2)
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
        )


def patch_sdpa() -> bool:
    """Monkey-patch F.scaled_dot_product_attention → wave_sdpa globally.

    Safe to call on any GPU — non-gfx12 falls back via is_wave_compatible.
    Returns True if patched, False if already patched or HIP_QUANT_DISABLE_WAVE=1.
    Like `sageattention`'s `sageattn` drop-in.
    """
    global _PATCHED_SDPA, _ORIG_SDPA
    if _env_flag("HIP_QUANT_DISABLE_WAVE"):
        warnings.warn("HIP_QUANT_DISABLE_WAVE=1 — patch_sdpa() is a no-op", RuntimeWarning)
        return False
    if _PATCHED_SDPA:
        return False
    if not _TORCH_AVAILABLE:
        raise RuntimeError("torch is not installed")
    _ORIG_SDPA = F.scaled_dot_product_attention

    def _patched(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        return wave_sdpa(
            query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
        )

    # preserve metadata
    _patched.__name__ = "wave_patched_sdpa"
    _patched.__doc__ = wave_sdpa.__doc__
    _patched._hip_quant_orig = _ORIG_SDPA  # type: ignore[attr-defined]
    F.scaled_dot_product_attention = _patched
    # also patch torch.nn.functional alias if separate
    try:
        import torch.nn.functional as F2

        if F2.scaled_dot_product_attention is not _patched:
            F2.scaled_dot_product_attention = _patched
    except Exception:
        pass
    _PATCHED_SDPA = True
    return True


def unpatch_sdpa() -> bool:
    """Restore original F.scaled_dot_product_attention. Returns True if unpatched."""
    global _PATCHED_SDPA, _ORIG_SDPA
    if not _PATCHED_SDPA or _ORIG_SDPA is None:
        return False
    F.scaled_dot_product_attention = _ORIG_SDPA
    try:
        import torch.nn.functional as F2

        F2.scaled_dot_product_attention = _ORIG_SDPA
    except Exception:
        pass
    _PATCHED_SDPA = False
    _ORIG_SDPA = None
    return True


def is_patched() -> bool:
    return _PATCHED_SDPA


def patch_transformers(
    model: Optional["torch.nn.Module"] = None,
    *,
    patch_sdpa_global: bool = True,
    convert_linear: bool = False,
    weight_format: str = "fp8_e4m3",
) -> "torch.nn.Module | None":
    """Best-effort HF transformers drop-in.

    * If patch_sdpa_global: calls patch_sdpa() so any model using SDPA benefits.
    * If model is given and convert_linear: replaces nn.Linear → QuantizedLinear
      via hip_quant.torch_api.convert_to_quantized (for inference).
    Returns model (mutated in-place) or None if only global patch was requested.
    """
    if patch_sdpa_global:
        patch_sdpa()
    if model is not None and convert_linear:
        try:
            from .torch_api import convert_to_quantized

            convert_to_quantized(model, weight_format=weight_format)
        except Exception as exc:
            warnings.warn(f"patch_transformers convert_linear failed: {exc}", RuntimeWarning)
    return model
