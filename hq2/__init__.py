"""HQ-family: portable learned-codebook quantization for model weights.

Quick start::

    import hq2
    packed = hq2.quantize(weights, backend="auto", iterations=8)
    restored = packed.dequantize()
"""

from __future__ import annotations

from importlib import import_module

from .api import BackendUnavailable, backend_status, dequantize, load, quantize
from .analyzer import HQAnalysis, analyze_model, render_analysis
from .archive import (
    HQ2_FORMAT,
    HQ3_FORMAT,
    HQ8_G128_FORMAT,
    HQ2Model,
    HQ2ModelWriter,
    HQFormatDescriptor,
    HQModel,
    HQModelWriter,
    HQTensorDescriptor,
    load_model,
    repack_model,
    save_model,
)
from .format import BITS_PER_WEIGHT, BLOCK_BYTES, BLOCK_SIZE, FORMAT_NAME, FORMAT_VERSION, HQ2Tensor
from .hq3 import (
    HQ3_BITS_PER_WEIGHT,
    HQ3_BLOCK_BYTES,
    HQ3_BLOCK_SIZE,
    HQ3_FORMAT_NAME,
    HQ3_FORMAT_VERSION,
    HQ3Tensor,
    decode_hq3_numpy,
)
from .hq8 import (
    HQ8_G128_BITS_PER_WEIGHT,
    HQ8_G128_BLOCK_BYTES,
    HQ8_G128_BLOCK_SIZE,
    HQ8_G128_FORMAT_NAME,
    HQ8_G128_FORMAT_VERSION,
    HQ8Tensor,
    decode_hq8_g128_numpy,
)
from .raw import HQRawTensor, raw_format, raw_format_for_torch
from .mixed_policy import (
    F32_FORMAT,
    Q4_0_FORMAT,
    Q8_0_FORMAT,
    MixedPolicyPlan,
    MixedTensorPlan,
    format_for_tier,
    gemma4_hq2_2p8_tier,
    plan_gemma4_hq2_2p8,
)
from .sensitivity import (
    TensorError,
    analyze_mixed_archive_error,
    decode_archive_weight,
    decode_q4_0_numpy,
    decode_q8_0_numpy,
    summarize_tensor_errors,
    tensor_error,
)

# Archive inspection, conversion, and CPU decoding must not import ROCm Torch
# merely because a caller imports ``hq2``.  Apart from wasted startup time,
# eager Torch loading can trigger fragile Windows ROCm runtime teardown in a
# short-lived analysis process.  Keep the inference/Transformers surface
# source-compatible through ``__getattr__`` while loading it only on demand.
_TORCH_INFERENCE_EXPORTS = frozenset({
    "HQ2Linear",
    "HQ3Linear",
    "HQ8Linear",
    "hq2_linear",
    "hq3_linear",
    "hq8_g128_linear",
    "rocm_fused_available",
    "rocm_hq3_fused_available",
    "rocm_hq8_g128_fused_available",
})
_TRANSFORMERS_EXPORTS = frozenset({"load_gemma4_hq2", "load_gemma4_hq2_package"})


def __getattr__(name: str):
    if name in _TORCH_INFERENCE_EXPORTS:
        module = import_module(".torch_inference", __name__)
    elif name in _TRANSFORMERS_EXPORTS:
        module = import_module(".transformers_loader", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(module, name)
    globals()[name] = value
    return value

__all__ = [
    "BackendUnavailable",
    "BITS_PER_WEIGHT",
    "BLOCK_BYTES",
    "BLOCK_SIZE",
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "HQ2Tensor",
    "HQ3Tensor",
    "HQ8Tensor",
    "HQRawTensor",
    "TensorError",
    "MixedPolicyPlan",
    "MixedTensorPlan",
    "HQAnalysis",
    "HQ2_FORMAT",
    "HQ3_FORMAT",
    "HQ8_G128_FORMAT",
    "Q4_0_FORMAT",
    "Q8_0_FORMAT",
    "F32_FORMAT",
    "HQ2Linear",
    "HQ3Linear",
    "HQ8Linear",
    "HQ3_BITS_PER_WEIGHT",
    "HQ3_BLOCK_BYTES",
    "HQ3_BLOCK_SIZE",
    "HQ3_FORMAT_NAME",
    "HQ3_FORMAT_VERSION",
    "HQ8_G128_BITS_PER_WEIGHT",
    "HQ8_G128_BLOCK_BYTES",
    "HQ8_G128_BLOCK_SIZE",
    "HQ8_G128_FORMAT_NAME",
    "HQ8_G128_FORMAT_VERSION",
    "HQ2Model",
    "HQ2ModelWriter",
    "HQFormatDescriptor",
    "HQModel",
    "HQModelWriter",
    "HQTensorDescriptor",
    "backend_status",
    "analyze_model",
    "analyze_mixed_archive_error",
    "dequantize",
    "decode_hq3_numpy",
    "decode_hq8_g128_numpy",
    "decode_archive_weight",
    "decode_q4_0_numpy",
    "decode_q8_0_numpy",
    "format_for_tier",
    "gemma4_hq2_2p8_tier",
    "hq2_linear",
    "hq3_linear",
    "hq8_g128_linear",
    "load",
    "load_gemma4_hq2",
    "load_gemma4_hq2_package",
    "load_model",
    "repack_model",
    "raw_format",
    "raw_format_for_torch",
    "plan_gemma4_hq2_2p8",
    "render_analysis",
    "quantize",
    "rocm_fused_available",
    "rocm_hq3_fused_available",
    "rocm_hq8_g128_fused_available",
    "save_model",
    "summarize_tensor_errors",
    "tensor_error",
]
