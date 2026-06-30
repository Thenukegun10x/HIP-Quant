"""
CDNA Compatibility Module
=========================
Utilities for supporting AMD CDNA (Instinct) GPUs alongside RDNA4.

Provides:
  - cpu_reference_quantize() — CPU-based reference quantizer for testing
    without a GPU (bit-exact to the HIP kernels, uses same algorithms)
  - build_config_for_arch() — returns optimal build flags for any AMD GPU
  - arch_supports_feature() — feature query table
  - suggest_emulation() — guidance for testing CDNA paths on other hardware
"""

import struct
import math
import numpy as np
from .device_info import DeviceProperties, probe_device
from . import GGML_TYPE, GGML_TYPE_BLOCK_SIZE, GGML_TYPE_BLOCK_BYTES

# =========================================================================
# Architecture feature table
# =========================================================================

ARCH_FEATURES = {
    # Format: arch_prefix -> {feature: supported_bool}
    # "gfx90a": CDNA2 (MI250)
    # "gfx942": CDNA3 (MI300X)
    # "gfx1100": RDNA3
    # "gfx1200": RDNA4
    "gfx90a":  {"wmma": False, "mfma": True,  "fp8": True,  "dp4a": True,  "wave32": True,  "wave64": True},
    "gfx940":  {"wmma": False, "mfma": True,  "fp8": True,  "dp4a": True,  "wave32": True,  "wave64": True},
    "gfx941":  {"wmma": False, "mfma": True,  "fp8": True,  "dp4a": True,  "wave32": True,  "wave64": True},
    "gfx942":  {"wmma": False, "mfma": True,  "fp8": True,  "dp4a": True,  "wave32": True,  "wave64": True},
    "gfx1100": {"wmma": False, "mfma": False, "fp8": False, "dp4a": True,  "wave32": True,  "wave64": False},
    "gfx1101": {"wmma": False, "mfma": False, "fp8": False, "dp4a": True,  "wave32": True,  "wave64": False},
    "gfx1102": {"wmma": False, "mfma": False, "fp8": False, "dp4a": True,  "wave32": True,  "wave64": False},
    "gfx1103": {"wmma": False, "mfma": False, "fp8": False, "dp4a": True,  "wave32": True,  "wave64": False},
    "gfx1150": {"wmma": False, "mfma": False, "fp8": False, "dp4a": True,  "wave32": True,  "wave64": False},
    "gfx1200": {"wmma": True,  "mfma": False, "fp8": True,  "dp4a": True,  "wave32": True,  "wave64": False},
    "gfx1201": {"wmma": True,  "mfma": False, "fp8": True,  "dp4a": True,  "wave32": True,  "wave64": False},
}

CDNA_ARCHS = {
    "gfx90a": "MI250/MI210",
    "gfx940": "MI300A",
    "gfx941": "MI300X",
    "gfx942": "MI300X",
}

RDNA_ARCHS = {
    "gfx1200": "RDNA4",
    "gfx1201": "RDNA4",
}


def arch_supports_feature(arch: str, feature: str) -> bool:
    """Check if a given GCN arch string supports a feature."""
    for prefix, features in sorted(ARCH_FEATURES.items(), key=lambda x: -len(x[0])):
        if arch.startswith(prefix):
            return features.get(feature, False)
    return False


def get_build_archs(target: str = "auto") -> list:
    """Return recommended --offload-arch list for a given target.
    
    Args:
        target: "auto" (detect), "rdna4", "cdna3", "cdna2", "all",
                or a specific arch like "gfx942".
    
    Returns:
        List of architecture strings (e.g. ["gfx1200", "gfx1201"])
    """
    if target == "all":
        return ["gfx90a", "gfx942",
                "gfx1100", "gfx1101", "gfx1102", "gfx1103",
                "gfx1200", "gfx1201"]
    if target == "cdna3":
        return ["gfx942"]
    if target == "cdna2":
        return ["gfx90a"]
    if target == "cdna":
        return ["gfx90a", "gfx942"]
    if target == "rdna3":
        return ["gfx1100", "gfx1101", "gfx1102", "gfx1103"]
    if target == "rdna4":
        return ["gfx1200", "gfx1201"]
    if target == "rdna":
        return ["gfx1100", "gfx1101", "gfx1102", "gfx1103",
                "gfx1200", "gfx1201"]
    if not target or target == "auto":
        dev = probe_device()
        if dev.gcn_arch:
            return [dev.gcn_arch]
        return ["gfx1200", "gfx1201"]
    return [target]


def build_config_for_arch(target: str = "auto") -> dict:
    """Return a dict of build configuration for a given arch target.
    
    Returns:
        dict with keys: archs, extra_flags, defines, note
    """
    archs = get_build_archs(target)
    
    base = {
        "archs": archs,
        "extra_flags": [],
        "defines": [],
        "note": "",
    }
    
    # Check if CDNA archs are included
    has_cdna = any(a.startswith("gfx9") for a in archs)
    has_rdna4 = any(a.startswith("gfx12") for a in archs)
    
    if has_rdna4 and has_cdna:
        # Mixed arch build — need to be careful with intrinsics
        if has_cdna:
            base["note"] = "Mixed CDNA + RDNA4 build. WMMA kernels will only run on gfx12 devices."
            base["extra_flags"].append("-DCDNA_COMPAT=1")
    elif has_cdna:
        base["note"] = "CDNA-only build. WMMA kernels excluded. Use MFMA kernels for GEMM."
        base["defines"].append("-DHIP_QUANT_CDNA=1")
    elif has_rdna4:
        base["note"] = "RDNA4-only build (current default)."
    
    return base


# =========================================================================
# CPU Reference Quantizer (bit-exact emulator)
# =========================================================================
# These functions replicate the GPU kernel logic on CPU for testing
# and verification without requiring a GPU.

def _fp32_to_fp16(f: float) -> int:
    """float32 -> IEEE 754 half (matches hip_quant_util.h fp32_to_fp16)."""
    u = struct.unpack("<I", struct.pack("<f", f))[0]
    sign = (u >> 16) & 0x8000
    exp = ((u >> 23) & 0xFF) - 127 + 15
    mant = u & 0x007FFFFF
    if exp > 30:
        return sign | 0x7C00 | (0x200 if mant else 0)
    if exp <= 0:
        mant = (mant | 0x800000) >> (1 - exp)
        if mant == 0:
            return sign
        while not (mant & 0x3E00000):
            mant <<= 1
        exp = 1
        mant >>= 13
        return sign | (exp << 10) | (mant & 0x3FF)
    rnd = mant & 0x1FFF
    mant >>= 13
    if rnd > 0x1000 or (rnd == 0x1000 and (mant & 1)):
        mant += 1
        if mant & 0x400:
            mant = 0
            exp += 1
    if exp >= 30:
        return sign | 0x7C00
    return sign | (exp << 10) | (mant & 0x3FF)


def _quantize_block_q4_0(values: np.ndarray) -> bytes:
    """Q4_0: symmetric 4-bit, block size 32.
    
    Mirrors kernels/quant_q4_0.cu logic.
    """
    assert len(values) == 32
    max_val = float(values[0])
    for v in values[1:]:
        v = float(v)
        if abs(v) > abs(max_val):
            max_val = v
    d = max_val / -8.0
    id_ = 1.0 / d if d != 0 else 0.0
    d_bits = _fp32_to_fp16(d)
    qs = bytearray()
    for v in values:
        q = int(v * id_ + 8.5)
        q = max(0, min(15, q))
        qs.append(q)
    packed = bytearray()
    packed += struct.pack("<H", d_bits)
    for i in range(16):
        packed.append(qs[i] | (qs[i + 16] << 4))
    return bytes(packed)


def _quantize_block_q8_0(values: np.ndarray) -> bytes:
    """Q8_0: symmetric 8-bit, block size 32."""
    assert len(values) == 32
    amax = max(abs(v) for v in values)
    d = amax / 127.0
    id_ = 1.0 / d if d > 0 else 0.0
    d_bits = _fp32_to_fp16(d)
    packed = bytearray()
    packed += struct.pack("<H", d_bits)
    for v in values:
        scaled = float(v) * id_
        q = math.floor(scaled + 0.5) if scaled >= 0 else math.ceil(scaled - 0.5)
        q = max(-127, min(127, q))
        packed.append(q & 0xFF)
    return bytes(packed)


def cpu_reference_quantize(arr: np.ndarray, type_name: str) -> np.ndarray:
    """CPU reference quantization for a types that have a simple block structure.
    
    Useful for:
      - Testing without a GPU
      - Validating GPU kernel output
      - Emulating CDNA paths on CPU
    
    Args:
        arr: float32 array (1D or 2D)
        type_name: e.g. "Q4_0", "Q8_0" (must be in GGML_TYPE)
    
    Returns:
        uint8 numpy array of quantized bytes
    """
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    nrows, n_per_row = arr.shape
    type_id = GGML_TYPE.get(type_name)
    if type_id is None:
        raise ValueError(f"Unknown type: {type_name}")
    blck = GGML_TYPE_BLOCK_SIZE.get(type_id, 0)
    if blck == 0:
        raise ValueError(f"Unsupported type for CPU reference: {type_name}")
    blck_bytes = GGML_TYPE_BLOCK_BYTES.get(type_id, 0)
    if blck_bytes == 0:
        raise ValueError(f"Unknown block bytes for: {type_name}")
    
    n_blocks = n_per_row // blck
    out = bytearray()
    
    for row in range(nrows):
        for b in range(n_blocks):
            block_vals = arr[row, b * blck:(b + 1) * blck]
            if type_name == "Q4_0":
                out.extend(_quantize_block_q4_0(block_vals))
            elif type_name == "Q8_0":
                out.extend(_quantize_block_q8_0(block_vals))
            else:
                raise NotImplementedError(
                    f"CPU reference not implemented for {type_name}. "
                    f"Supported: Q4_0, Q8_0"
                )
    
    out_arr = np.frombuffer(bytes(out), dtype=np.uint8)
    expected = nrows * n_blocks * blck_bytes
    assert len(out_arr) == expected, f"Expected {expected}, got {len(out_arr)}"
    return out_arr


# =========================================================================
# Emulation mode management
# =========================================================================

_EMULATION_MODE = None  # None = auto, "cpu", "gpu"


def set_emulation_mode(mode: str):
    """Set the emulation mode for CDNA testing.
    
    Args:
        mode: "auto" (detect GPU, use GPU if available), 
              "cpu" (force CPU reference),
              "gpu" (force GPU with fallback warnings)
    """
    global _EMULATION_MODE
    if mode not in ("auto", "cpu", "gpu"):
        raise ValueError(f"Invalid emulation mode: {mode}")
    _EMULATION_MODE = mode


def get_emulation_mode() -> str:
    global _EMULATION_MODE
    if _EMULATION_MODE is None:
        dev = probe_device()
        if dev.device_count > 0 and dev.dll_loaded:
            return "gpu"
        return "cpu"
    return _EMULATION_MODE


def suggest_emulation() -> str:
    """Print guidance on how to test CDNA compatibility without CDNA hardware."""
    dev = probe_device()
    lines = []
    lines.append("=" * 56)
    lines.append("  CDNA Compatibility - Testing Guide")
    lines.append("=" * 56)
    if dev.dll_loaded:
        lines.append(f"  Current GPU      : {dev.gcn_arch} ({dev.arch_note})")
        lines.append(f"  Architecture      : {dev.arch_family}")
    
    lines.append("")
    lines.append("  To test CDNA compatibility without CDNA hardware:")
    lines.append("")
    lines.append("  1. CPU Emulation (recommended for unit tests)")
    lines.append("     from hip_quant.cdna_compat import cpu_reference_quantize")
    lines.append("     q = cpu_reference_quantize(arr, 'Q4_0')")
    lines.append("")
    lines.append("  2. Cross-compile with CDNA archs")
    lines.append("     .\\build.ps1 -CDNA")
    lines.append("     (adds ROCm 7.1 Windows compile-tested gfx90a/gfx942 targets)")
    lines.append("")
    lines.append("  3. Run the existing test suite")
    lines.append("     python -m pytest tests/")
    lines.append("     All quantization kernels are pure compute and portable.")
    lines.append("")
    lines.append("  4. The *only* non-portable kernel is fp8_gemm_test_wmma")
    lines.append("     which uses gfx12-specific __builtin_amdgcn_wmma.")
    lines.append("     For CDNA, you'd need MFMA-based kernels instead.")
    lines.append("")
    lines.append("  5. To verify your code runs on CDNA without a card:")
    lines.append("     - hip_quant.device_info.probe_device() will report capabilities")
    lines.append("     - All quantization types are portable across AMD GPUs")
    lines.append("=" * 56)
    return "\n".join(lines)


__all__ = [
    "arch_supports_feature", "get_build_archs", "build_config_for_arch",
    "cpu_reference_quantize",
    "set_emulation_mode", "get_emulation_mode", "suggest_emulation",
    "ARCH_FEATURES", "CDNA_ARCHS", "RDNA_ARCHS",
]
