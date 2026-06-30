import ctypes
import os
import sys
from dataclasses import dataclass, field

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if os.name == "nt":
    _ROCM_BIN = r"C:\Program Files\AMD\ROCm\7.1\bin"
else:
    _rocm_home = os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH") or "/opt/rocm"
    _ROCM_BIN = os.path.join(_rocm_home, "bin")

_DLL_DIR_HANDLES = []

def _runtime_dll_dirs():
    if os.name != "nt":
        return [_ROCM_BIN]
    dirs = []
    rocm_bin = os.environ.get("HIP_QUANT_ROCM_BIN")
    if rocm_bin:
        dirs.append(rocm_bin)
    for env_name in ("HIP_QUANT_ROCM_HOME", "ROCM_HOME", "ROCM_PATH", "HIP_PATH"):
        rocm_home = os.environ.get(env_name)
        if rocm_home:
            dirs.append(os.path.join(rocm_home, "bin"))
    dirs.extend([
        os.path.join(sys.prefix, "Lib", "site-packages", "_rocm_sdk_core", "bin"),
        os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib"),
        os.path.join(sys.prefix, "Scripts"),
        _ROCM_BIN,
    ])
    return dirs

def _add_runtime_dll_dirs():
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    seen = set()
    for path in _runtime_dll_dirs():
        path = os.path.normpath(path)
        if path in seen or not os.path.isdir(path):
            continue
        seen.add(path)
        _DLL_DIR_HANDLES.append(os.add_dll_directory(path))

# Architecture family classifications
ARCH_FAMILIES = {
    "gfx9":  {"family": "CDNA",  "note": "CDNA 1 (MI100/MI50)"},
    "gfx90a": {"family": "CDNA",  "note": "CDNA 2 (MI200 series)"},
    "gfx940": {"family": "CDNA",  "note": "CDNA 3 (MI300A)"},
    "gfx941": {"family": "CDNA",  "note": "CDNA 3 (MI300X)"},
    "gfx942": {"family": "CDNA",  "note": "CDNA 3 (MI300X)"},
    "gfx1010": {"family": "RDNA",  "note": "RDNA 1 (RX 5000)"},
    "gfx1011": {"family": "RDNA",  "note": "RDNA 1"},
    "gfx1012": {"family": "RDNA",  "note": "RDNA 1"},
    "gfx1030": {"family": "RDNA",  "note": "RDNA 2 (RX 6000)"},
    "gfx1031": {"family": "RDNA",  "note": "RDNA 2"},
    "gfx1032": {"family": "RDNA",  "note": "RDNA 2"},
    "gfx1100": {"family": "RDNA",  "note": "RDNA 3 (RX 7000)"},
    "gfx1101": {"family": "RDNA",  "note": "RDNA 3"},
    "gfx1102": {"family": "RDNA",  "note": "RDNA 3"},
    "gfx1103": {"family": "RDNA",  "note": "RDNA 3 (RX 7600)"},
    "gfx1150": {"family": "RDNA",  "note": "RDNA 3.5 (Strix Point)"},
    "gfx1151": {"family": "RDNA",  "note": "RDNA 3.5"},
    "gfx1200": {"family": "RDNA",  "note": "RDNA 4 (RX 9000)"},
    "gfx1201": {"family": "RDNA",  "note": "RDNA 4"},
}

# Quantized type that work on ANY arch (pure compute, no arch-specific intrinsics)
PORTABLE_TYPES = {
    "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q8_1",
    "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K",
    "IQ1_S", "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ3_S",
    "IQ4_NL", "IQ4_XS",
    "TQ1_0", "TQ2_0", "F8_E4M3", "F8_E5M2",
}

# Features that require specific arch support
WMMA_REQUIRED = {"fp8_gemm_test_wmma"}
FP8_EXPAND_REQUIRED = {"quantize_from_fp8"}  # expand kernels are portable


@dataclass
class DeviceProperties:
    name: str = ""
    gcn_arch: str = ""
    major: int = 0
    minor: int = 0
    cu_count: int = 0
    total_memory: int = 0
    free_memory: int = 0
    shared_mem_per_block: int = 0
    warp_size: int = 0
    max_threads_per_block: int = 0
    has_wmma: bool = False
    hip_runtime_version: int = 0
    device_count: int = 0
    dll_loaded: bool = False
    _dll_base: str = ""
    info: list = field(default_factory=list)

    @property
    def arch_family(self) -> str:
        for prefix, fam in sorted(ARCH_FAMILIES.items(), key=lambda x: -len(x[0])):
            if self.gcn_arch.startswith(prefix):
                return fam["family"]
        return "Unknown"

    @property
    def arch_note(self) -> str:
        for prefix, fam in sorted(ARCH_FAMILIES.items(), key=lambda x: -len(x[0])):
            if self.gcn_arch.startswith(prefix):
                return fam["note"]
        return self.gcn_arch

    @property
    def memory_gb(self) -> float:
        return self.total_memory / (1024**3)

    @property
    def memory_free_gb(self) -> float:
        return self.free_memory / (1024**3)

    @property
    def is_cdna(self) -> bool:
        return self.arch_family == "CDNA"

    @property
    def is_rdna4(self) -> bool:
        return self.gcn_arch.startswith("gfx12")

    @property
    def is_rdna3(self) -> bool:
        return self.gcn_arch.startswith("gfx11") or self.gcn_arch.startswith("gfx115")


def _resolve_dll(dll_path=None):
    if dll_path is not None:
        return dll_path
    env_dll = os.environ.get("HIP_QUANT_DLL") or os.environ.get("HIP_QUANT_DLL_PATH")
    win_names = ["hip_quantize_rocm721.dll", "hip_quantize.dll"]
    if os.environ.get("HIP_QUANT_DLL_VARIANT", "").lower() in ("7.1", "71", "rocm71", "legacy"):
        win_names.reverse()
    names = win_names if os.name == "nt" else ["libhip_quantize.so"]
    candidates = [env_dll] if env_dll else []
    for root in (_PKG_DIR, os.path.join(_PKG_DIR, ".."), os.path.join(_PKG_DIR, "..", "..", "src")):
        for name in names:
            candidates.append(os.path.join(root, name))
    for p in candidates:
        p = os.path.normpath(p)
        if os.path.isfile(p):
            return p
    return None


def probe_device(dll_path=None) -> DeviceProperties:
    """Probe GPU device via the shared library and return a DeviceProperties dataclass.
    
    Safe to call even if no library is found or no GPU is present — 
    returns best-effort data with dll_loaded=False if anything fails.
    """
    dp = DeviceProperties()
    dll_path = _resolve_dll(dll_path)
    if dll_path is None or not os.path.isfile(dll_path):
        dp.info.append("No hip_quantize shared library found")
        return dp

    try:
        _add_runtime_dll_dirs()
        dll = ctypes.CDLL(dll_path)
    except Exception as e:
        dp.info.append(f"Failed to load shared library: {e}")
        return dp

    dp.dll_loaded = True
    dp._dll_base = os.path.basename(dll_path)

    # Device count
    dll.get_device_count.restype = ctypes.c_int
    try:
        dp.device_count = dll.get_device_count()
    except Exception:
        dp.device_count = 0

    if dp.device_count == 0:
        dp.info.append("No HIP devices found")
        return dp

    # Device name + arch
    try:
        dll.get_device_name.restype = ctypes.c_char_p
        dp.name = dll.get_device_name().decode("utf-8", errors="replace")
    except Exception:
        pass

    try:
        dll.get_arch_name.argtypes = [ctypes.c_char_p, ctypes.c_int]
        dll.get_arch_name.restype = ctypes.c_int
        buf = ctypes.create_string_buffer(256)
        dll.get_arch_name(buf, 256)
        dp.gcn_arch = buf.value.decode("utf-8", errors="replace").strip()
    except Exception:
        pass

    # Device properties
    try:
        dll.get_device_prop.argtypes = [
            ctypes.c_char_p, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ]
        dll.get_device_prop.restype = ctypes.c_int
        name_buf = ctypes.create_string_buffer(256)
        major = ctypes.c_int()
        minor = ctypes.c_int()
        cu_count = ctypes.c_int()
        total_mem = ctypes.c_size_t()
        shared_mem = ctypes.c_size_t()
        warp_size = ctypes.c_int()
        max_threads = ctypes.c_int()
        dll.get_device_prop(
            name_buf, 256,
            ctypes.byref(major), ctypes.byref(minor),
            ctypes.byref(cu_count),
            ctypes.byref(total_mem),
            ctypes.byref(shared_mem),
            ctypes.byref(warp_size),
            ctypes.byref(max_threads),
        )
        dp.name = name_buf.value.decode("utf-8", errors="replace")
        dp.major = major.value
        dp.minor = minor.value
        dp.cu_count = cu_count.value
        dp.total_memory = total_mem.value
        dp.shared_mem_per_block = shared_mem.value
        dp.warp_size = warp_size.value
        dp.max_threads_per_block = max_threads.value
    except Exception as e:
        dp.info.append(f"get_device_prop failed: {e}")

    # Memory info
    try:
        dll.get_device_memory.argtypes = [
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        dll.get_device_memory.restype = ctypes.c_int
        free_mem = ctypes.c_size_t()
        total_mem2 = ctypes.c_size_t()
        dll.get_device_memory(ctypes.byref(free_mem), ctypes.byref(total_mem2))
        dp.free_memory = free_mem.value
    except Exception:
        pass

    # WMMA support
    try:
        dll.device_has_wmma.restype = ctypes.c_int
        dll.device_has_wmma.argtypes = []
        dp.has_wmma = bool(dll.device_has_wmma())
    except Exception:
        pass

    try:
        dll.get_hip_runtime_version.restype = ctypes.c_int
        dll.get_hip_runtime_version.argtypes = []
        dp.hip_runtime_version = int(dll.get_hip_runtime_version())
        if dp.hip_runtime_version and dp.hip_runtime_version < 70200000 and dp.has_wmma:
            dp.has_wmma = False
            dp.info.append(
                f"ROCm/HIP runtime {dp.hip_runtime_version} detected; this package's gfx12 FP8/BF8 WMMA kernels are disabled because ROCm 7.1 and older can hang or zero GPU memory."
            )
    except Exception:
        pass

    return dp


def report(dev: DeviceProperties) -> str:
    """Return a formatted report string from DeviceProperties."""
    lines = []
    if not dev.dll_loaded:
        lines.append("hip_quantize shared library not loaded — GPU not available")
        for msg in dev.info:
            lines.append(f"  {msg}")
        return "\n".join(lines)

    lines.append(f"{'='*56}")
    lines.append(f"  HIP Device Report")
    lines.append(f"{'='*56}")
    lines.append(f"  Device count      : {dev.device_count}")
    lines.append(f"  Active device     : {dev.name}")
    lines.append(f"  GCN arch          : {dev.gcn_arch}")
    lines.append(f"  Architecture      : {dev.arch_family} ({dev.arch_note})")
    lines.append(f"  Compute Capability: {dev.major}.{dev.minor}")
    lines.append(f"  Compute Units (CUs): {dev.cu_count}")
    lines.append(f"  Total VRAM        : {dev.memory_gb:.2f} GiB")
    lines.append(f"  Free VRAM         : {dev.memory_free_gb:.2f} GiB")
    lines.append(f"  Shared mem/block  : {dev.shared_mem_per_block:,} bytes")
    lines.append(f"  Warp size         : {dev.warp_size}")
    lines.append(f"  Max threads/block : {dev.max_threads_per_block}")
    if dev.hip_runtime_version:
        lines.append(f"  HIP runtime       : {dev.hip_runtime_version}")
    lines.append(f"  gfx12 FP8 WMMA    : {'YES' if dev.has_wmma else 'NO (requires gfx12 + ROCm 7.2+)'}")
    lines.append(f"  Library           : {dev._dll_base}")
    lines.append(f"{'='*56}")

    # Feature compatibility
    lines.append("")
    lines.append("  Feature Compatibility:")
    if dev.gcn_arch:
        lines.append(f"    All quantization types     : {'PORTABLE' if dev.gcn_arch else 'N/A'}")
        lines.append(f"    FP8 expand (E4M3/E5M2)     : {'PORTABLE' if dev.gcn_arch else 'N/A'}")
        lines.append(f"    FP8/BF8 WMMA gfx12 path   : {'AVAILABLE' if dev.has_wmma else 'NOT AVAILABLE for this device/runtime'}")
        if dev.is_cdna:
            lines.append("    CDNA-specific note          : CDNA can use FP8/BF16 MFMA/rocBLASLt-style paths, not this RDNA4 gfx12 WMMA builtin")

    for msg in dev.info:
        lines.append(f"  [{msg}]")

    return "\n".join(lines)


__all__ = [
    "DeviceProperties", "probe_device", "report",
    "ARCH_FAMILIES", "PORTABLE_TYPES",
]
