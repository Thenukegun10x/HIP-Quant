import ctypes
import numpy as np
import os
import sys

__version__ = "0.5.1"

_TORCH_EXPORTS = {
    "quantize_e4m3",
    "quantize_e5m2",
    "dequantize_e4m3",
    "dequantize_e5m2",
    "fp8_linear_forward",
    "fp8_linear_forward_scaled",
    "fp8_linear_forward_fp8_weight",
    "fp8_linear_forward_fp8_input",
    "fp8_linear_forward_fp8_input_weight",
    "fp8_linear_backward_input",
    "fp8_linear_backward_input_scaled",
    "fp8_linear_backward_weight",
    "fp8_linear_backward_weight_scaled",
    "fp8_linear_backward_input_fp8_grad",
    "fp8_linear_backward_weight_fp8_grad",
    "Fp8LinearFunction",
    "Fp8Linear",
    "Fp8ScaledLinearFunction",
    "Fp8ScaledLinear",
    "Fp8ShadowLinearFunction",
    "Fp8ShadowLinear",
    "fp8_conv1d",
    "Fp8Conv1d",
    "fp8_conv2d",
    "Fp8Conv2d",
    "Fp8TensorMeta",
    "convert_to_fp8",
    "Adafactor",
}

__all__ = [
    "GGML_TYPE",
    "GGML_TYPE_BLOCK_SIZE",
    "GGML_TYPE_BLOCK_BYTES",
    "HipQuant",
    "get_hip_quant",
    "quantize",
    # Compatibility / device info
    "probe_device",
    "report_device",
    "suggest_cdna_emulation",
    "get_build_config",
    "cpu_reference_quantize",
    *_TORCH_EXPORTS,
]

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))

# Type enum values matching GGML_TYPE_* in ggml.h
GGML_TYPE = {
    "Q4_0": 2,
    "Q4_1": 3,
    "Q5_0": 6,
    "Q5_1": 7,
    "Q8_0": 8,
    "Q8_1": 9,
    "Q2_K": 10,
    "Q3_K": 11,
    "Q4_K": 12,
    "Q5_K": 13,
    "Q6_K": 14,
    "IQ2_XXS": 16,
    "IQ2_XS": 17,
    "IQ3_XXS": 18,
    "IQ1_S": 19,
    "IQ4_NL": 20,
    "IQ3_S": 21,
    "IQ4_XS": 23,
    "TQ1_0": 34,
    "TQ2_0": 35,
    "F8_E4M3": 36,
    "F8_E5M2": 37,
}

GGML_TYPE_BLOCK_SIZE = {
    2: 32,
    3: 32,
    6: 32,
    7: 32,
    8: 32,
    9: 32,
    10: 256,
    11: 256,
    12: 256,
    13: 256,
    14: 256,
    16: 256,
    17: 256,
    18: 256,
    19: 256,
    20: 32,
    21: 256,
    23: 256,
    34: 256,
    35: 256,
    36: 32,
    37: 32,
}

GGML_TYPE_BLOCK_BYTES = {
    2: 18,
    3: 20,
    6: 22,
    7: 24,
    8: 34,
    9: 36,
    10: 84,
    11: 110,
    12: 144,
    13: 176,
    14: 210,
    16: 66,
    17: 74,
    18: 98,
    19: 50,
    20: 18,
    21: 110,
    23: 136,
    34: 54,
    35: 66,
    36: 32,
    37: 32,
}

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

def _shared_library_candidates():
    env_dll = os.environ.get("HIP_QUANT_DLL") or os.environ.get("HIP_QUANT_DLL_PATH")
    if env_dll:
        yield env_dll

    win_names = ["hip_quantize_rocm721.dll", "hip_quantize.dll"]
    if os.environ.get("HIP_QUANT_DLL_VARIANT", "").lower() in ("7.1", "71", "rocm71", "legacy"):
        win_names.reverse()
    names = win_names if os.name == "nt" else ["libhip_quantize.so"]
    roots = [
        _PKG_DIR,
        os.path.join(_PKG_DIR, ".."),
        os.path.join(_PKG_DIR, "..", "..", "src"),
    ]
    for root in roots:
        for name in names:
            yield os.path.join(root, name)

class HipQuant:
    def __init__(self, dll_path=None):
        _add_runtime_dll_dirs()
        if dll_path is None:
            candidates = list(_shared_library_candidates())
            for p in candidates:
                p = os.path.normpath(p)
                if os.path.isfile(p):
                    dll_path = p
                    break
            if dll_path is None:
                raise FileNotFoundError(
                    f"hip_quantize shared library not found. Tried: {candidates}"
                )
        self._dll_path = dll_path
        self._dll = ctypes.CDLL(dll_path)
        self._dll.quantize_tensor.restype = ctypes.c_size_t
        self._dll.quantize_tensor.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_float),
        ]
        self._dll.quantize_tensor_fp8_input.restype = ctypes.c_size_t
        self._dll.quantize_tensor_fp8_input.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_float),
        ]
        try:
            self._quantize_tensor_fp8_e5m2_input = self._dll.quantize_tensor_fp8_e5m2_input
            self._quantize_tensor_fp8_e5m2_input.restype = ctypes.c_size_t
            self._quantize_tensor_fp8_e5m2_input.argtypes = [
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.POINTER(ctypes.c_float),
            ]
        except AttributeError:
            self._quantize_tensor_fp8_e5m2_input = None
        self._dll.ggml_type_size_for.restype = ctypes.c_size_t
        self._dll.ggml_type_size_for.argtypes = [ctypes.c_int]
        self._dll.ggml_blck_size_for.restype = ctypes.c_size_t
        self._dll.ggml_blck_size_for.argtypes = [ctypes.c_int]
        self._dll.ggml_row_size_for.restype = ctypes.c_size_t
        self._dll.ggml_row_size_for.argtypes = [ctypes.c_int, ctypes.c_int64]
        self._dll.get_device_name.restype = ctypes.c_char_p
        self._dll.get_device_count.restype = ctypes.c_int
        self._dll.get_device_count.argtypes = []
        try:
            self._get_arch_name = self._dll.get_arch_name
            self._get_arch_name.restype = ctypes.c_int
            self._get_arch_name.argtypes = [ctypes.c_char_p, ctypes.c_int]
        except AttributeError:
            self._get_arch_name = None
        try:
            self._get_hip_runtime_version = self._dll.get_hip_runtime_version
            self._get_hip_runtime_version.restype = ctypes.c_int
            self._get_hip_runtime_version.argtypes = []
        except AttributeError:
            self._get_hip_runtime_version = None
        self._dll.fp8_gemm_test_wmma.restype = ctypes.c_int
        self._dll.fp8_gemm_test_wmma.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._dll.quantize_reset.restype = None
        self._dll.quantize_reset.argtypes = []

    @property
    def dll_path(self):
        return self._dll_path

    @property
    def device_count(self):
        return self._dll.get_device_count()

    @property
    def device_name(self):
        return self._dll.get_device_name().decode("utf-8")

    @property
    def gcn_arch(self):
        if self._get_arch_name is None:
            return ""
        buf = ctypes.create_string_buffer(256)
        self._get_arch_name(buf, 256)
        return buf.value.decode("utf-8", errors="replace").strip()

    @property
    def hip_runtime_version(self):
        if self._get_hip_runtime_version is None:
            return 0
        return int(self._get_hip_runtime_version())

    def type_size(self, type_num):
        return self._dll.ggml_type_size_for(int(type_num))

    def blck_size(self, type_num):
        return self._dll.ggml_blck_size_for(int(type_num))

    def row_size(self, type_num, n_per_row):
        return self._dll.ggml_row_size_for(int(type_num), n_per_row)

    def quantize_reset(self):
        self._dll.quantize_reset()

    def fp8_gemm_test_wmma(self, A_fp8, B_fp8, M, N, K, lda=None, ldb=None, ldc=None):
        """Micro FP8 GEMM via rocWMMA WMMA.

        Takes pre-quantized FP8 E4M3 matrices A (MxK) and B (KxN),
        computes C = A * B using GPU WMMA instructions, returns float32 C.

        All matrices are row-major. M, N must be multiples of 16.

        Args:
            A_fp8: uint8 numpy array (M, lda) of FP8 E4M3 values
            B_fp8: uint8 numpy array (K, ldb) of FP8 E4M3 values
            M, N, K: matrix dimensions
            lda, ldb, ldc: leading dimensions (defaults: K, N, N)

        Returns:
            float32 numpy array (M, N) = A @ B, or None on failure.
        """
        if lda is None: lda = K
        if ldb is None: ldb = N
        if ldc is None: ldc = N
        M = int(M); N = int(N); K = int(K)
        lda = int(lda); ldb = int(ldb); ldc = int(ldc)
        if M <= 0 or N <= 0 or K <= 0:
            raise ValueError("M, N, and K must be positive")
        if M % 16 != 0 or N % 16 != 0:
            raise ValueError("M and N must be multiples of 16 for fp8_gemm_test_wmma")
        if lda < K or ldb < N or ldc < N:
            raise ValueError("lda must be >= K, ldb >= N, and ldc >= N")
        a_arr = np.asarray(A_fp8)
        b_arr = np.asarray(B_fp8)
        if a_arr.ndim != 2 or a_arr.shape[0] < M or a_arr.shape[1] < lda:
            raise ValueError("A_fp8 must have shape at least (M, lda)")
        if b_arr.ndim != 2 or b_arr.shape[0] < K or b_arr.shape[1] < ldb:
            raise ValueError("B_fp8 must have shape at least (K, ldb)")
        if os.environ.get("HIP_QUANT_DISABLE_WMMA", "").lower() in ("1", "true", "yes", "on"):
            raise RuntimeError("FP8 WMMA is disabled by HIP_QUANT_DISABLE_WMMA.")
        if os.environ.get("HIP_QUANT_ENABLE_GFX12_WMMA", "").lower() not in ("1", "true", "yes", "on"):
            raise RuntimeError(
                "FP8 WMMA is disabled by default because unstable kernels can hang or reset the GPU. "
                "Set HIP_QUANT_ENABLE_GFX12_WMMA=1 only for controlled testing on ROCm 7.2+ gfx12 systems."
            )
        arch = self.gcn_arch
        if not arch.startswith("gfx12"):
            raise RuntimeError(
                f"FP8 WMMA test requires the gfx12/RDNA4 w32 intrinsic path; current device arch is {arch or 'unknown'}. "
                "CDNA may support FP8/BF16 through MFMA/rocBLASLt paths, but not this RDNA4-specific kernel."
            )
        runtime_version = self.hip_runtime_version
        if runtime_version and runtime_version < 70200000:
            raise RuntimeError(
                f"FP8 WMMA test is disabled on ROCm/HIP runtime {runtime_version}. "
                "Use the packaged ROCm 7.2.1 DLL/runtime; ROCm 7.1 and older can hang or zero GPU memory."
            )
        A_fp8 = np.ascontiguousarray(a_arr[:M, :lda], dtype=np.uint8)
        B_fp8 = np.ascontiguousarray(b_arr[:K, :ldb], dtype=np.uint8)
        C = np.empty((M, ldc), dtype=np.float32)
        ret = self._dll.fp8_gemm_test_wmma(
            A_fp8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            B_fp8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            M, N, K, lda, ldb, ldc,
        )
        if ret != 0:
            return None
        return C[:, :N]

    def quantize_numpy(self, arr, type_num, imatrix=None):
        """Quantize a float32 numpy array to the given GGML type.

        Args:
            arr: 2-D float32 numpy array (nrows, n_per_row) or 1-D.
            type_num: GGML type number (use GGML_TYPE dict).
            imatrix: Optional importance matrix (same shape as arr).

        Returns:
            np.uint8 array of quantized bytes.
        """
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        nrows, n_per_row = arr.shape
        blck = self.blck_size(type_num)
        if blck <= 0:
            raise ValueError(f"Unsupported type: {type_num}")
        if n_per_row % blck != 0:
            raise ValueError(
                f"n_per_row ({n_per_row}) must be multiple of block size ({blck})"
            )
        out_nbytes = nrows * (self.type_size(type_num) * (n_per_row // blck))
        dst = np.empty(out_nbytes, dtype=np.uint8)
        src_ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        dst_ptr = dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        im_ptr = None
        if imatrix is not None:
            imatrix = np.ascontiguousarray(imatrix, dtype=np.float32)
            if imatrix.shape != arr.shape:
                raise ValueError(f"imatrix shape {imatrix.shape} != arr shape {arr.shape}")
            im_ptr = imatrix.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        result = self._dll.quantize_tensor(
            int(type_num), src_ptr, dst_ptr, nrows, n_per_row, im_ptr
        )
        if result != out_nbytes:
            raise RuntimeError(
                f"Quantize returned {result} bytes, expected {out_nbytes}"
            )
        return dst

    def quantize_numpy_to(self, arr, type_num, dst, imatrix=None):
        """Quantize into a pre-allocated uint8 buffer. Modifies dst in-place."""
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        nrows, n_per_row = arr.shape
        src_ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        dst_ptr = dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        im_ptr = None
        if imatrix is not None:
            imatrix = np.ascontiguousarray(imatrix, dtype=np.float32)
            im_ptr = imatrix.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        result = self._dll.quantize_tensor(
            int(type_num), src_ptr, dst_ptr, nrows, n_per_row, im_ptr
        )
        if result != len(dst):
            raise RuntimeError(
                f"Quantize returned {result} bytes, buffer has {len(dst)}"
            )
        return dst

    def quantize_from_fp8(self, arr_fp8, type_num, imatrix=None, source_format="E4M3"):
        """Quantize from FP8 input to the given GGML type.

        Accepts FP8 E4M3 or E5M2 encoded data (uint8 array, 1 byte per element).
        The data is expanded to float32 on the GPU before quantizing,
        using 4x less host memory and transfer bandwidth than float32.

        Best for low-bit targets (Q4_0 through Q5_K) where quantization
        noise dominates over FP8 input precision. For Q8_0+ and I-Quants,
        prefer quantize_numpy() with float32 input.

        Args:
            arr_fp8: 2-D uint8 numpy array (nrows, n_per_row) of FP8 values,
                     or the output of quantize_numpy(..., GGML_TYPE["F8_E4M3"])
                     / quantize_numpy(..., GGML_TYPE["F8_E5M2"]).
            type_num: GGML type number for the output format.
            imatrix: Optional float32 importance matrix (same logical shape).
            source_format: "E4M3"/GGML_TYPE["F8_E4M3"] or
                           "E5M2"/GGML_TYPE["F8_E5M2"].

        Returns:
            np.uint8 array of quantized bytes.
        """
        arr_fp8 = np.ascontiguousarray(arr_fp8, dtype=np.uint8)
        if arr_fp8.ndim == 1:
            arr_fp8 = arr_fp8.reshape(1, -1)
        nrows, n_per_row = arr_fp8.shape
        blck = self.blck_size(type_num)
        if blck <= 0:
            raise ValueError(f"Unsupported type: {type_num}")
        if n_per_row % blck != 0:
            raise ValueError(
                f"n_per_row ({n_per_row}) must be multiple of block size ({blck})"
            )
        out_nbytes = nrows * (self.type_size(type_num) * (n_per_row // blck))
        dst = np.empty(out_nbytes, dtype=np.uint8)
        src_ptr = arr_fp8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        dst_ptr = dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        im_ptr = None
        if imatrix is not None:
            imatrix = np.ascontiguousarray(imatrix, dtype=np.float32)
            im_ptr = imatrix.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        if isinstance(source_format, str):
            source_format = source_format.upper()
        if source_format in ("E4M3", "F8_E4M3", GGML_TYPE["F8_E4M3"]):
            result = self._dll.quantize_tensor_fp8_input(
                int(type_num), src_ptr, dst_ptr, nrows, n_per_row, im_ptr
            )
        elif source_format in ("E5M2", "F8_E5M2", GGML_TYPE["F8_E5M2"]):
            if self._quantize_tensor_fp8_e5m2_input is None:
                raise RuntimeError("Loaded hip_quantize library does not support E5M2 FP8 input")
            result = self._quantize_tensor_fp8_e5m2_input(
                int(type_num), src_ptr, dst_ptr, nrows, n_per_row, im_ptr
            )
        else:
            raise ValueError(f"Unsupported FP8 source_format: {source_format}")
        if result != out_nbytes:
            raise RuntimeError(
                f"Quantize (FP8 input) returned {result} bytes, expected {out_nbytes}"
            )
        return dst


_default_instance = None

def get_hip_quant(dll_path=None):
    global _default_instance
    if dll_path is not None:
        return HipQuant(dll_path)
    if _default_instance is None:
        _default_instance = HipQuant()
    return _default_instance


def quantize(arr, type_num):
    return get_hip_quant().quantize_numpy(arr, type_num)


# =========================================================================
# Compatibility / Device Info helpers
# =========================================================================

def probe_device(dll_path=None):
    """Probe HIP device and return a DeviceProperties dataclass.

    Safe to call without a GPU — returns graceful fallback info.
    """
    from .device_info import probe_device as _probe
    return _probe(dll_path)


def report_device(dll_path=None):
    """Print a formatted GPU compatibility report."""
    from .device_info import probe_device, report
    dev = probe_device(dll_path)
    return report(dev)


def suggest_cdna_emulation():
    """Print guidance on testing CDNA compatibility without CDNA hardware."""
    from .cdna_compat import suggest_emulation
    return suggest_emulation()


def get_build_config(target="auto"):
    """Get recommended build configuration for a given arch target.

    Args:
        target: "auto", "rdna4", "cdna", "cdna3", "all", or specific arch

    Returns:
        dict with archs, extra_flags, defines, note
    """
    from .cdna_compat import build_config_for_arch
    return build_config_for_arch(target)


def cpu_reference_quantize(arr, type_name):
    """CPU-based reference quantization for testing without a GPU.

    Args:
        arr: float32 numpy array
        type_name: str like "Q4_0", "Q8_0"

    Returns:
        uint8 numpy array
    """
    from .cdna_compat import cpu_reference_quantize as _cpu_ref
    return _cpu_ref(arr, type_name)


def __getattr__(name):
    if name in _TORCH_EXPORTS:
        try:
            from . import torch_api
        except ImportError as exc:
            raise ImportError(
                f"hip_quant.{name} requires the PyTorch extension. "
                "Install ROCm PyTorch and run `python setup_torch.py build_ext --inplace`."
            ) from exc
        value = getattr(torch_api, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
