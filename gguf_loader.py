"""
hip_quant/gguf_loader.py
========================

Load GGUF weight tensors straight onto the GPU.

* Plain types (F32/F16/BF16/I8/I16/I32/I64/F64) are viewed zero-copy
  from the file mmap and staged with one H2D copy.
* GGML Q-types supported by ``dequantize_q_to_fp8`` (legacy Q4_0..Q8_1
  and Q2_K..Q6_K, see ``torch_api.GGML_Q_TO_FP8_SUPPORTED``) are staged
  as raw blocks and dequantized to FP8 E4M3 on-device. Output is
  ``[nrows, n_per_row]`` uint8 (``n_per_row == ne[0]``), which matches
  ``torch.nn.Linear`` ``(out, in)`` layout, so the result can feed
  ``fp8_linear_*`` directly (optionally via ``as_fp8_e4m3``).
* Anything else (I-quants until their kernels land, Q8_K, TQ, MXFP4,
  ...) raises a clear error naming the tensor.

The mmap stays open for the load and is closed before return; GPU
tensors own their storage afterwards.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import torch

from . import gguf
from .torch_api import GGML_Q_TO_FP8_SUPPORTED, dequantize_q_to_e4m3

_PLAIN_DTYPES = {
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F64": torch.float64,
}


def load_gguf(
    path: str,
    device: str = "cuda",
    to_fp8: bool = True,
    view_fp8: bool = False,
    native_q: bool = False,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = ("nextn",),
) -> Dict[str, torch.Tensor]:
    """Load (a subset of) a GGUF file onto ``device``.

    Returns ``{tensor_name: tensor}``. If ``native_q=True``, supported types
    (Q4_0, Q8_0) are loaded zero-copy as raw packed uint8 tensors with
    ``.ggml_type`` and ``.orig_shape`` attached.
    """
    gf = gguf.load(path)
    out: Dict[str, torch.Tensor] = {}
    with gf.open():
        for t in gf.tensors:
            if include is not None and not any(s in t.name for s in include):
                continue
            if exclude is not None and any(s in t.name for s in exclude):
                continue
            out[t.name] = _load_tensor(gf, t, device, to_fp8, view_fp8, native_q)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return out


def _load_tensor(gf: gguf.GGUFFile, t: gguf.GGUFTensor,
                 device: str, to_fp8: bool, view_fp8: bool,
                 native_q: bool = False) -> torch.Tensor:
    raw = gf.raw_bytes(t)
    if t.type_name in _PLAIN_DTYPES:
        # read-only mmap view: frombuffer warns but is safe here since the
        # CPU tensor is transient (copied H2D on the next line).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cpu = torch.frombuffer(raw, dtype=_PLAIN_DTYPES[t.type_name])
        return cpu.reshape(t.shape).to(device)
    if native_q and t.ggml_type in (2, 8) and len(t.shape) == 2 and "blk." in t.name:  # GGML_TYPE_Q4_0, GGML_TYPE_Q8_0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            staged = torch.frombuffer(raw, dtype=torch.uint8).to(device)
        staged.ggml_type = t.ggml_type  # type: ignore[attr-defined]
        staged.orig_shape = t.shape      # type: ignore[attr-defined]
        return staged
    if t.ggml_type in GGML_Q_TO_FP8_SUPPORTED and to_fp8:
        _, block_size, _ = GGML_Q_TO_FP8_SUPPORTED[t.ggml_type]
        n_per_row = t.shape[-1]
        nrows = t.n_elements // n_per_row
        assert t.n_elements % n_per_row == 0, t.name
        assert n_per_row % block_size == 0, (t.name, t.shape)
        staged = torch.frombuffer(raw, dtype=torch.uint8).to(device)
        fp8 = dequantize_q_to_e4m3(staged, t.ggml_type, n_per_row)
        fp8 = fp8.reshape(nrows, n_per_row)
        return fp8.view(torch.float8_e4m3fn) if view_fp8 else fp8
    raise gguf.GGUFError(
        f"tensor {t.name!r} ({t.type_name}): no GPU path yet "
        f"(need q_to_fp8 kernel for type {t.ggml_type})")
