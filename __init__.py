import ctypes
import numpy as np
import os
import sys

__version__ = "2.0.0.post215"


_TORCH_EXPORTS = {
    # ── High-Level QoL API ─────────────────────────────────
    "quantize",
    "dequantize",
    "attention",
    "mxfp4_to_fp8",
    "QuantizedLinear",
    "convert_to_quantized",
    "check",
    # ── Drop-in attention (SageAttn-style) ─────────────────
    "wave_available",
    "is_wave_compatible",
    "wave_sdpa",
    "patch_sdpa",
    "unpatch_sdpa",
    "is_patched",
    "patch_transformers",
    # ── GPU SMI (easy query while training) ────────────────
    "query",
    "query_one",
    "brief",
    "status",
    "format_gpu",
    "format_table",
    "GpuMonitor",
    # ── Quantization / dequantization (low-level) ──────────
    "quantize_e4m3",
    "quantize_e5m2",
    "quantize_e4m3_transpose",
    "quantize_e5m2_transpose",
    "quantize_e5m2_stochastic",
    "quantize_e4m3_blockwise",
    "quantize_e5m2_blockwise",
    "quantize_e5m2_blockwise_stochastic",
    "refresh_fp8_blockwise_shadow",
    "dequantize_e4m3",
    "dequantize_e5m2",
    "dequantize_e4m3_blockwise",
    "dequantize_e5m2_blockwise",
    # ── Attention ──────────────────────────────────────────
    "wave_attn",
    # ── Linear / GEMM ─────────────────────────────────────
    "fp8_linear_forward",
    "fp8_linear_forward_scaled",
    "fp8_linear_forward_fp8_weight",
    "fp8_linear_forward_fp8_input",
    "fp8_linear_forward_fp8_input_weight",
    "pack_fp8_weight_for_wmma",
    "fp8_linear_forward_fp8_input_weight_packed",
    "fp8_linear_forward_blockwise",
    "fp8_linear_forward_blockwise_quantized",
    "fp8_linear_backward_input",
    "fp8_linear_backward_input_scaled",
    "fp8_linear_backward_weight",
    "fp8_linear_backward_weight_scaled",
    "fp8_linear_backward_input_fp8_grad",
    "fp8_linear_backward_weight_fp8_grad",
    # ── MXFP4 ─────────────────────────────────────────────
    "dequantize_mxfp4_to_fp8",
    "mxfp4_linear_forward",
    "native_mxfp4_contract",
    "native_mxfp4_capability",
    "native_mxfp4_linear_forward",
    "MXFP4_BLOCK_SIZE",
    "MXFP4_PACKED_VALUE_BYTES_PER_BLOCK",
    "MXFP4_SCALE_BYTES_PER_BLOCK",
    # ── GGML / Q to FP8 dequant ───────────────────────────
    "dequantize_q_to_fp8",
    "dequantize_q_to_e4m3",
    "dequantize_q_to_e5m2",
    "GGML_Q_TO_FP8_SUPPORTED",
    # ── nn.Module wrappers ────────────────────────────────
    "Fp8LinearFunction",
    "Fp8Linear",
    "Fp8ScaledLinearFunction",
    "Fp8ScaledLinear",
    "Fp8ShadowLinearFunction",
    "Fp8ShadowLinear",
    "Fp8TensorMeta",
    "Fp8GraphRunner",
    "capture_hip_graph",
    # ── Conv / misc ───────────────────────────────────────
    "fp8_conv1d",
    "Fp8Conv1d",
    "fp8_conv2d",
    "Fp8Conv2d",
    "convert_to_fp8",
    "Adafactor",
    "adafactor_row_col_mean_square",
}

__all__ = [
    "GGML_TYPE",
    "GGML_TYPE_BLOCK_SIZE",
    "GGML_TYPE_BLOCK_BYTES",
    "HipQuant",
    "get_hip_quant",
    "quantize",
    "bf16_to_fp32",
    # Compatibility / device info
    "probe_device",
    "report_device",
    "suggest_cdna_emulation",
    "get_build_config",
    "cpu_reference_quantize",
    "info",
    "demo",
    "check",
    "available",
    "__version__",
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
    "IQ2_S": 22,
    "IQ4_XS": 23,
    "IQ1_M": 29,
    "TQ1_0": 34,
    "TQ2_0": 35,
    "HQ2": 38,
    "AQ2": 31,
    "AQ2_QK": 32,
    "AQ2_VO": 33,
    "Q1_0": 41,
    "Q2_0": 42,
    "Q8_K": 15,
    "BF16": 30,
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
    22: 256,
    23: 256,
    29: 256,
    34: 256,
    35: 256,
    38: 256,
    31: 256,
    32: 256,
    33: 256,
    36: 32,
    37: 32,
    41: 128,
    42: 64,
    15: 256,
    30: 1,
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
    22: 82,
    23: 136,
    29: 56,
    34: 54,
    35: 66,
    38: 72,
    31: 72,
    32: 72,
    33: 72,
    36: 32,
    37: 32,
    41: 18,
    42: 18,
    15: 292,
    30: 2,
}

def _find_default_rocm_bin():
    for env in ("HIP_QUANT_ROCM_BIN", "ROCM_PATH", "HIP_PATH", "ROCM_HOME", "HIP_QUANT_ROCM_HOME"):
        val = os.environ.get(env)
        if val:
            p = val if val.endswith("bin") else os.path.join(val, "bin")
            if os.path.isdir(p):
                return p
    if os.name == "nt":
        candidates = sorted(glob.glob(r"C:\Program Files\AMD\ROCm\*\bin"), reverse=True)
        if candidates:
            return candidates[0]
        return r"C:\Program Files\AMD\ROCm\7.1\bin"
    rocm_home = os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH") or "/opt/rocm"
    return os.path.join(rocm_home, "bin")

_ROCM_BIN = _find_default_rocm_bin()

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
        try:
            self._quantize_tensor_hq2_iters = self._dll.quantize_tensor_hq2_iters
            self._quantize_tensor_hq2_iters.restype = ctypes.c_size_t
            self._quantize_tensor_hq2_iters.argtypes = [
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
            ]
        except AttributeError:
            self._quantize_tensor_hq2_iters = None
        try:
            self._quantize_tensor_aq2_iters = self._dll.quantize_tensor_aq2_iters
            self._quantize_tensor_aq2_iters.restype = ctypes.c_size_t
            self._quantize_tensor_aq2_iters.argtypes = [
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
            ]
        except AttributeError:
            self._quantize_tensor_aq2_iters = None
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
        try:
            self._dequantize_tensor_to_fp8 = self._dll.dequantize_tensor_to_fp8
            self._dequantize_tensor_to_fp8.restype = ctypes.c_size_t
            self._dequantize_tensor_to_fp8.argtypes = [
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int,
            ]
        except AttributeError:
            self._dequantize_tensor_to_fp8 = None
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
            self._get_selected_device = self._dll.get_selected_device
            self._get_selected_device.restype = ctypes.c_int
            self._get_selected_device.argtypes = []
        except AttributeError:
            self._get_selected_device = None
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
        try:
            self._benchmark_quantize_kernel = self._dll.benchmark_quantize_kernel
            self._benchmark_quantize_kernel.restype = ctypes.c_int
            self._benchmark_quantize_kernel.argtypes = [
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_float),
            ]
        except AttributeError:
            self._benchmark_quantize_kernel = None
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
        try:
            self._quantize_stochastic_e5m2 = self._dll.quantize_tensor_fp8_e5m2_stochastic
            self._quantize_stochastic_e5m2.restype = ctypes.c_size_t
            self._quantize_stochastic_e5m2.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_uint64,
            ]
        except AttributeError:
            self._quantize_stochastic_e5m2 = None
        try:
            self._wmma_diag = self._dll.wmma_diag
            self._wmma_diag.restype = ctypes.c_int
            self._wmma_diag.argtypes = [ctypes.POINTER(ctypes.c_int)]
        except AttributeError:
            self._wmma_diag = None

    @property
    def dll_path(self):
        return self._dll_path

    @property
    def device_count(self):
        return self._dll.get_device_count()

    @property
    def selected_device(self):
        """HIP device index selected by the native library, or -1 if unavailable."""
        if self._get_selected_device is None:
            return -1
        return int(self._get_selected_device())

    @property
    def device_name(self):
        raw = self._dll.get_device_name()
        if not raw:
            raise RuntimeError(
                "HIP device initialization failed; check HIP_VISIBLE_DEVICES and HIP_QUANT_DEVICE"
            )
        return raw.decode("utf-8", errors="replace")

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

    def quantize_e5m2_stochastic(self, arr, seed=0):
        """Quantize float32 array to FP8 E5M2 with stochastic rounding.

        Uses unbiased stochastic rounding between adjacent E5M2 bins
        for better gradient convergence in training.
        """
        if self._quantize_stochastic_e5m2 is None:
            raise RuntimeError("Stochastic E5M2 not available in loaded DLL")
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        nrows, n_per_row = arr.shape
        dst = np.empty(nrows * n_per_row, dtype=np.uint8)
        result = self._quantize_stochastic_e5m2(
            arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            nrows, n_per_row, int(seed),
        )
        if result != dst.size:
            raise RuntimeError(f"Stochastic E5M2 returned {result} bytes, expected {dst.size}")
        return dst.reshape(nrows, n_per_row)

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
        arch = self.gcn_arch
        if not arch.startswith("gfx12"):
            raise RuntimeError(
                f"FP8 WMMA test requires the gfx12/RDNA4 w32 intrinsic path; current device arch is {arch or 'unknown'}. "
                "CDNA may support FP8/BF16 through MFMA/rocBLASLt paths, but not this RDNA4-specific kernel."
            )
        runtime_version = self.hip_runtime_version
        if runtime_version and runtime_version < 70200000:
            raise RuntimeError(
                f"FP8 WMMA test requires ROCm/HIP 7.2+; current runtime is {runtime_version}."
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

    def wmma_diag(self):
        """Run WMMA stability diagnostics on the GPU.

        Executes four kernel tests to determine whether the
        __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12 intrinsic
        is functioning correctly. Detects hangs, zero outputs, and
        register decay under repeated use.

        Returns:
            dict with keys:
                test1_value: max WMMA output for known inputs (milliscale int,
                             value/1000 ≈ expected K). Should be > 0.
                test2_repeated_ok: 0 = pass (100 iterations OK),
                                   < 0 = failed at iteration |value|.
                test3_lds_value: max output from LDS-staged WMMA (milliscale).
                test4_multiwave: number of waves (0-16) that produced non-zero
                                 output. Should be 16 on healthy hardware.
                ok: True if all four tests pass, False otherwise.
            None on DLL error or missing diagnostic support.
        """
        if self._wmma_diag is None:
            return None
        results = (ctypes.c_int * 4)()
        ret = self._wmma_diag(results)
        if ret not in (0, 2, 3, 4):
            return None
        r = list(results)
        t1_ok = r[0] > 0
        t2_ok = r[1] == 0
        t3_ok = r[2] > 0
        t4_ok = r[3] == 16
        return {
            "test1_value": r[0],
            "test1_ok": t1_ok,
            "test2_repeated_ok": r[1],
            "test2_ok": t2_ok,
            "test3_lds_value": r[2],
            "test3_ok": t3_ok,
            "test4_multiwave": r[3],
            "test4_ok": t4_ok,
            "ok": t1_ok and t2_ok and t3_ok and t4_ok,
            "gated": ret != 0,
            "gate_reason": {
                2: "not gfx12 device",
                3: "ROCm < 7.2",
                4: "WMMA disabled by env var",
            }.get(ret, "None"),
        }

    def quantize_numpy(self, arr, type_num, imatrix=None, hq2_iterations=4,
                       aq2_iterations=8):
        """Quantize a float32 numpy array to the given GGML type.

        Args:
            arr: 2-D float32 numpy array (nrows, n_per_row) or 1-D.
            type_num: GGML type number (use GGML_TYPE dict).
            imatrix: Optional importance matrix (same shape as arr).
            hq2_iterations: Lloyd iterations for HQ2 (1-16; default 4).
            aq2_iterations: Lloyd iterations for AQ2 (1-16; default 8).
                AQ2 should normally use an attention-derived imatrix.

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
        if int(type_num) in {GGML_TYPE["AQ2"], GGML_TYPE["AQ2_QK"], GGML_TYPE["AQ2_VO"]}:
            if not 1 <= int(aq2_iterations) <= 16:
                raise ValueError("aq2_iterations must be in [1, 16]")
            if self._quantize_tensor_aq2_iters is None:
                raise RuntimeError(
                    "Loaded hip_quantize library does not support AQ2; rebuild it"
                )
            result = self._quantize_tensor_aq2_iters(
                int(type_num), src_ptr, dst_ptr, nrows, n_per_row, im_ptr, int(aq2_iterations)
            )
        elif int(type_num) == GGML_TYPE["HQ2"] and hq2_iterations != 4:
            if not 1 <= int(hq2_iterations) <= 16:
                raise ValueError("hq2_iterations must be in [1, 16]")
            if self._quantize_tensor_hq2_iters is None:
                raise RuntimeError(
                    "Loaded hip_quantize library does not support configurable HQ2 iterations; rebuild it"
                )
            result = self._quantize_tensor_hq2_iters(
                int(type_num), src_ptr, dst_ptr, nrows, n_per_row, im_ptr, int(hq2_iterations)
            )
        else:
            result = self._dll.quantize_tensor(
                int(type_num), src_ptr, dst_ptr, nrows, n_per_row, im_ptr
            )
        if result != out_nbytes:
            raise RuntimeError(
                f"Quantize returned {result} bytes, expected {out_nbytes}"
            )
        return dst

    def benchmark_quantize_kernel(
        self, arr, type_num, hq2_iterations=4, warmup_iterations=10, timed_iterations=50
    ):
        """Return average HIP-event kernel time in milliseconds.

        Host-to-device copies, allocations, output copies, and optional
        importance weights are deliberately excluded.  This is for a fair
        baseline comparison of the native quantization kernels.
        """
        if self._benchmark_quantize_kernel is None:
            raise RuntimeError(
                "Loaded hip_quantize library does not support kernel benchmarking; rebuild it"
            )
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError("arr must be a 1-D or 2-D float32 array")
        nrows, n_per_row = arr.shape
        block_size = self.blck_size(type_num)
        if block_size <= 0:
            raise ValueError(f"Unsupported type: {type_num}")
        if n_per_row % block_size != 0:
            raise ValueError(
                f"n_per_row ({n_per_row}) must be multiple of block size ({block_size})"
            )
        if int(type_num) == GGML_TYPE["HQ2"] and not 1 <= int(hq2_iterations) <= 16:
            raise ValueError("hq2_iterations must be in [1, 16]")
        if int(warmup_iterations) < 0 or int(timed_iterations) < 1:
            raise ValueError("warmup_iterations must be >= 0 and timed_iterations must be >= 1")

        average_ms = ctypes.c_float()
        result = self._benchmark_quantize_kernel(
            int(type_num),
            arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            nrows,
            n_per_row,
            int(hq2_iterations),
            int(warmup_iterations),
            int(timed_iterations),
            ctypes.byref(average_ms),
        )
        if result != 0:
            raise RuntimeError("Native quantization kernel benchmark failed")
        return float(average_ms.value)

    def quantize_numpy_to(self, arr, type_num, dst, imatrix=None, hq2_iterations=4,
                          aq2_iterations=8):
        """Quantize into a pre-allocated uint8 buffer. Modifies dst in-place."""
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        nrows, n_per_row = arr.shape
        block_size = self.blck_size(type_num)
        if block_size <= 0:
            raise ValueError(f"Unsupported type: {type_num}")
        if n_per_row % block_size != 0:
            raise ValueError(
                f"n_per_row ({n_per_row}) must be multiple of block size ({block_size})"
            )
        out_nbytes = nrows * (self.type_size(type_num) * (n_per_row // block_size))
        if not isinstance(dst, np.ndarray):
            raise TypeError("dst must be a writable C-contiguous numpy array")
        if dst.dtype != np.uint8 or not dst.flags.c_contiguous or not dst.flags.writeable:
            raise ValueError("dst must be a writable C-contiguous uint8 numpy array")
        if dst.size != out_nbytes:
            raise ValueError(
                f"dst has {dst.size} bytes, expected {out_nbytes} for type {type_num}"
            )
        src_ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        dst_ptr = dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        im_ptr = None
        if imatrix is not None:
            imatrix = np.ascontiguousarray(imatrix, dtype=np.float32)
            if imatrix.shape != arr.shape:
                raise ValueError(f"imatrix shape {imatrix.shape} != arr shape {arr.shape}")
            im_ptr = imatrix.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        if int(type_num) in {GGML_TYPE["AQ2"], GGML_TYPE["AQ2_QK"], GGML_TYPE["AQ2_VO"]}:
            if not 1 <= int(aq2_iterations) <= 16:
                raise ValueError("aq2_iterations must be in [1, 16]")
            if self._quantize_tensor_aq2_iters is None:
                raise RuntimeError(
                    "Loaded hip_quantize library does not support AQ2; rebuild it"
                )
            result = self._quantize_tensor_aq2_iters(
                int(type_num), src_ptr, dst_ptr, nrows, n_per_row, im_ptr, int(aq2_iterations)
            )
        elif int(type_num) == GGML_TYPE["HQ2"] and hq2_iterations != 4:
            if not 1 <= int(hq2_iterations) <= 16:
                raise ValueError("hq2_iterations must be in [1, 16]")
            if self._quantize_tensor_hq2_iters is None:
                raise RuntimeError(
                    "Loaded hip_quantize library does not support configurable HQ2 iterations; rebuild it"
                )
            result = self._quantize_tensor_hq2_iters(
                int(type_num), src_ptr, dst_ptr, nrows, n_per_row, im_ptr, int(hq2_iterations)
            )
        else:
            result = self._dll.quantize_tensor(
                int(type_num), src_ptr, dst_ptr, nrows, n_per_row, im_ptr
            )
        if result != out_nbytes:
            raise RuntimeError(f"Quantize returned {result} bytes, expected {out_nbytes}")
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

    def dequantize_to_fp8(self, packed, type_num, n_per_row, output_format="E4M3"):
        """Dequantize packed GGML Q blocks directly to raw FP8 bytes.

        The conversion and FP8 encode execute in one GPU kernel, so no F32
        tensor is allocated or copied back to the host.  Supports all legacy
        Q4/Q5/Q8, K-Quant (Q2_K-Q6_K), I-Quant (IQ1_S-IQ4_XS), T-Quant
        (TQ1_0, TQ2_0), H-Quant (HQ2), and IQ4_NL formats.

        Args:
            packed: Flat packed ``uint8`` Q-type data, such as returned by
                :meth:`quantize_numpy` or :meth:`quantize_from_fp8`.
            type_num: Source GGML Q type number (for example,
                ``GGML_TYPE["Q4_K"]``).
            n_per_row: Logical number of elements in each source row.  It
                must be a multiple of the source type's block size.
            output_format: ``"E4M3"`` / ``"F8_E4M3"`` or ``"E5M2"`` /
                ``"F8_E5M2"``.  The result contains one raw FP8 byte per
                logical element.

        Returns:
            A C-contiguous ``uint8`` array with shape ``(nrows, n_per_row)``.
        """
        if self._dequantize_tensor_to_fp8 is None:
            raise RuntimeError(
                "Loaded hip_quantize library does not support Q-to-FP8 dequantization; rebuild hip_quantize.dll"
            )

        type_num = int(type_num)
        n_per_row = int(n_per_row)
        block_size = self.blck_size(type_num)
        type_size = self.type_size(type_num)
        if block_size <= 0 or type_size <= 0:
            raise ValueError(f"Unsupported source type: {type_num}")
        if n_per_row <= 0 or n_per_row % block_size != 0:
            raise ValueError(
                f"n_per_row ({n_per_row}) must be a positive multiple of block size ({block_size})"
            )

        if isinstance(output_format, str):
            output_format = output_format.upper()
        if output_format in ("E4M3", "F8_E4M3", GGML_TYPE["F8_E4M3"]):
            output_type = GGML_TYPE["F8_E4M3"]
        elif output_format in ("E5M2", "F8_E5M2", GGML_TYPE["F8_E5M2"]):
            output_type = GGML_TYPE["F8_E5M2"]
        else:
            raise ValueError(f"Unsupported FP8 output_format: {output_format}")

        packed = np.ascontiguousarray(packed, dtype=np.uint8).reshape(-1)
        row_bytes = type_size * (n_per_row // block_size)
        if packed.size == 0 or packed.size % row_bytes != 0:
            raise ValueError(
                f"packed size ({packed.size}) must be a non-zero multiple of row size ({row_bytes})"
            )
        nrows = packed.size // row_bytes
        dst = np.empty((nrows, n_per_row), dtype=np.uint8)

        result = self._dequantize_tensor_to_fp8(
            type_num,
            packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            nrows,
            n_per_row,
            output_type,
        )
        if result != dst.size:
            raise RuntimeError(
                f"Q-to-FP8 dequantization returned {result} bytes, expected {dst.size}"
            )
        return dst

    def dequantize_to_e4m3(self, packed, type_num, n_per_row):
        """Shortcut for :meth:`dequantize_to_fp8` with E4M3 output."""
        return self.dequantize_to_fp8(packed, type_num, n_per_row, "E4M3")

    def dequantize_to_e5m2(self, packed, type_num, n_per_row):
        """Shortcut for :meth:`dequantize_to_fp8` with E5M2 output."""
        return self.dequantize_to_fp8(packed, type_num, n_per_row, "E5M2")


_default_instance = None

def get_hip_quant(dll_path=None):
    global _default_instance
    if dll_path is not None:
        return HipQuant(dll_path)
    if _default_instance is None:
        _default_instance = HipQuant()
    return _default_instance


def quantize(arr, type_num, imatrix=None, hq2_iterations=4, aq2_iterations=8):
    return get_hip_quant().quantize_numpy(
        arr, type_num, imatrix=imatrix, hq2_iterations=hq2_iterations,
        aq2_iterations=aq2_iterations)


def bf16_to_fp32(values, *, shape=None):
    """Vectorized CPU conversion of raw IEEE bfloat16 values to ``float32``.

    ``hip_quant`` has GPU kernels for its supported packed GGML quantization
    formats, but BF16 is an ordinary 16-bit floating-point storage format, not
    one of those packed Q formats. Decode a BF16 tensor on the CPU at a model
    loader boundary with this helper, then leave the loader's BF16 weight policy
    unchanged. No HIP runtime, DLL, or PyTorch extension is loaded here.

    Args:
        values: A ``numpy.uint16`` array containing raw BF16 bit patterns, or
            a bytes-like buffer in little-endian BF16 order.
        shape: Optional logical output shape when ``values`` is a flat buffer.

    Returns:
        A new ``numpy.float32`` array. Conversion is exact: the BF16 bits are
        placed in the upper 16 bits of each FP32 value, preserving signed zero,
        infinities, and NaN payload bits.
    """
    if isinstance(values, (bytes, bytearray, memoryview)):
        try:
            if memoryview(values).nbytes % 2:
                raise ValueError("BF16 byte buffer length must be a multiple of 2")
            words = np.frombuffer(values, dtype="<u2")
        except (BufferError, TypeError) as exc:
            raise TypeError(
                "bf16_to_fp32: values must be a contiguous bytes-like BF16 buffer"
            ) from exc
    else:
        words = np.asarray(values)
        if words.dtype.kind != "u" or words.dtype.itemsize != 2:
            raise TypeError(
                "bf16_to_fp32: values must be a numpy.uint16 array of raw BF16 bit patterns"
            )
        # Convert non-native-endian uint16 arrays by numeric value before the
        # bit shift. Native uint16 input remains zero-copy until the required
        # FP32 output allocation below.
        words = words.astype(np.uint16, copy=False)

    fp32_bits = words.astype(np.uint32, copy=False) << np.uint32(16)
    result = fp32_bits.view(np.float32)

    if shape is None:
        return result
    try:
        shape = tuple(int(dim) for dim in shape)
    except TypeError as exc:
        raise TypeError("bf16_to_fp32: shape must be an iterable of dimensions") from exc
    if any(dim < 0 for dim in shape):
        raise ValueError("bf16_to_fp32: shape dimensions must be non-negative")
    expected_size = int(np.prod(shape, dtype=np.int64))
    if expected_size != result.size:
        raise ValueError(
            f"bf16_to_fp32: shape {shape} requires {expected_size} values, "
            f"but the input contains {result.size}"
        )
    return result.reshape(shape)


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


def cpu_reference_quantize(arr, type_name, imatrix=None, hq2_iterations=4,
                           aq2_iterations=8):
    """CPU-based reference quantization for testing without a GPU.

    Args:
        arr: float32 numpy array
        type_name: str like "Q4_0", "Q8_0"
        imatrix: optional float32 importance matrix
        hq2_iterations: Lloyd iterations for HQ2 (default 4)
        aq2_iterations: Lloyd iterations for AQ2 (default 8)

    Returns:
        uint8 numpy array
    """
    from .cdna_compat import cpu_reference_quantize as _cpu_ref
    return _cpu_ref(
        arr, type_name, imatrix=imatrix, hq2_iterations=hq2_iterations,
        aq2_iterations=aq2_iterations)


def __getattr__(name):
    if name in _TORCH_EXPORTS:
        # drop-in helpers live in hip_quant.wave_helpers (lighter import)
        if name in (
            "wave_available",
            "is_wave_compatible",
            "wave_sdpa",
            "patch_sdpa",
            "unpatch_sdpa",
            "is_patched",
            "patch_transformers",
        ):
            try:
                from . import wave_helpers as _attn_mod  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    f"hip_quant.{name} requires PyTorch. Install torch with ROCm support."
                ) from exc
            value = getattr(_attn_mod, name)
            globals()[name] = value
            return value
        if name in ("query", "query_one", "brief", "status", "format_gpu", "format_table", "GpuMonitor"):
            try:
                from . import smi as _smi_mod  # type: ignore
            except ImportError as exc:
                raise ImportError(f"hip_quant.{name} requires hip_quant.smi") from exc
            value = getattr(_smi_mod, name)
            globals()[name] = value
            return value
        try:
            from . import torch_api
        except ImportError as exc:
            raise ImportError(
                f"hip_quant.{name} requires the PyTorch extension.\n\n"
                "To build it:\n"
                '  cd C:\\path\\to\\hip_quant\n'
                '  & "C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools\\VC\\Auxiliary\\Build\\vcvars64.bat"\n'
                '  python setup_torch.py build_ext --inplace\n\n'
                "Requires: ROCm 7.x + PyTorch 2.x with ROCm support."
            ) from exc
        value = getattr(torch_api, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def info(dll_path=None):
    """Print a diagnostic report: ROCm version, GPU info, DLL status, torch extension status.

    Call without arguments for a quick system check.  Equivalent to ``hip-quant --info``.
    """
    try:
        from . import report_device
        print(report_device(dll_path))
    except Exception as e:
        print(f"NumPy / DLL backend:  unavailable ({e})")

    try:
        from . import torch_api as ta
        ext = ta._load_extension()
        print(f"\nPyTorch extension:    loaded ({ext.__module__})")
        print(f"  wave_attn_forward:  {'OK' if hasattr(ext, 'wave_attn_forward') else 'missing — rebuild setup_torch.py'}")
        print(f"  wave_attn_backward: {'OK' if hasattr(ext, 'wave_attn_backward') else 'missing — rebuild setup_torch.py'}")
        print(f"  quantize_e4m3:      {'OK' if hasattr(ext, 'quantize_e4m3') else 'missing'}")
    except Exception as e:
        print(f"\nPyTorch extension:    not loaded ({e})")

    try:
        import torch
        print(f"\nPyTorch:              {torch.__version__}")
        if hasattr(torch.version, 'hip'):
            print(f"ROCm:                 {torch.version.hip}")
        if torch.cuda.is_available():
            print(f"GPU:                  {torch.cuda.get_device_name(0)}")
        else:
            print(f"GPU:                  none visible")
    except ImportError:
        print(f"\nPyTorch:              not installed")

    print(f"\nhip_quant version:    {__version__}")


def available():
    """Return True if hip_quant can run FP8 operations on this machine."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        props = torch.cuda.get_device_properties(0)
        arch = getattr(props, "gcnArchName", "")
        if not arch.startswith("gfx12"):
            return False
        from . import torch_api
        torch_api._load_extension()
        return True
    except Exception:
        return False


_DEMO_LOADED = False
_DEMO_MOD = None


def _load_demo():
    global _DEMO_LOADED, _DEMO_MOD
    if not _DEMO_LOADED:
        import importlib
        _DEMO_MOD = importlib.import_module(".demo", __package__)
        _DEMO_LOADED = True
    return _DEMO_MOD


def demo(name=None):
    """Run beginner-friendly demos of FP8 quantization, WaveAttention, MXFP4, and more.

    >>> import hip_quant
    >>> hip_quant.demo()               # run all demos
    >>> hip_quant.demo("attention")     # run attention demo only
    """
    _load_demo().run(name)
