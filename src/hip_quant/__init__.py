import ctypes
import os
from pathlib import Path

import numpy as np

_PKG_DIR = Path(__file__).resolve().parent

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
}

GGML_TYPE_NAME = {v: k for k, v in GGML_TYPE.items()}

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
}

IMATRIX_REQUIRED_TYPES = {16, 17, 19}

_DEFAULT_ROCM_BIN = r"C:\Program Files\AMD\ROCm\7.1\bin"


def normalize_type(type_id):
    if isinstance(type_id, str):
        key = type_id.upper()
        if key not in GGML_TYPE:
            choices = ", ".join(sorted(GGML_TYPE))
            raise ValueError(f"Unsupported GGML type {type_id!r}. Supported: {choices}")
        return GGML_TYPE[key]
    type_num = int(type_id)
    if type_num not in GGML_TYPE_NAME:
        choices = ", ".join(f"{name}={num}" for name, num in sorted(GGML_TYPE.items()))
        raise ValueError(f"Unsupported GGML type {type_num}. Supported: {choices}")
    return type_num


def type_name(type_id):
    return GGML_TYPE_NAME[normalize_type(type_id)]


def supported_types():
    return dict(GGML_TYPE)


class HipQuant:
    def __init__(self, dll_path=None, rocm_bin=None):
        rocm_bin = rocm_bin or os.environ.get("HIP_QUANT_ROCM_BIN") or _DEFAULT_ROCM_BIN
        if os.name == "nt" and os.path.isdir(rocm_bin):
            os.add_dll_directory(rocm_bin)

        if dll_path is None:
            candidates = [
                _PKG_DIR / "hip_quantize.dll",
                _PKG_DIR.parent / "hip_quantize.dll",
                _PKG_DIR.parent.parent / "src" / "hip_quant" / "hip_quantize.dll",
            ]
            dll_path = next((p for p in candidates if p.is_file()), None)
            if dll_path is None:
                tried = "\n".join(str(p) for p in candidates)
                raise FileNotFoundError(
                    "hip_quantize.dll not found. Build it with `python -m hip_quant.build` "
                    f"or pass dll_path explicitly. Tried:\n{tried}"
                )

        self._dll_path = str(dll_path)
        self._dll = ctypes.CDLL(self._dll_path)
        self._dll.quantize_tensor.restype = ctypes.c_size_t
        self._dll.quantize_tensor.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_float),
        ]
        self._dll.ggml_type_size_for.restype = ctypes.c_size_t
        self._dll.ggml_type_size_for.argtypes = [ctypes.c_int]
        self._dll.ggml_blck_size_for.restype = ctypes.c_size_t
        self._dll.ggml_blck_size_for.argtypes = [ctypes.c_int]
        self._dll.ggml_row_size_for.restype = ctypes.c_size_t
        self._dll.ggml_row_size_for.argtypes = [ctypes.c_int, ctypes.c_int64]
        self._dll.get_device_name.restype = ctypes.c_char_p
        self._dll.get_device_count.restype = ctypes.c_int

    @property
    def dll_path(self):
        return self._dll_path

    @property
    def device_name(self):
        raw = self._dll.get_device_name()
        return raw.decode("utf-8", errors="replace") if raw else "unknown"

    @property
    def device_count(self):
        return int(self._dll.get_device_count())

    def type_size(self, type_id):
        return int(self._dll.ggml_type_size_for(normalize_type(type_id)))

    def block_size(self, type_id):
        return int(self._dll.ggml_blck_size_for(normalize_type(type_id)))

    def blck_size(self, type_id):
        return self.block_size(type_id)

    def row_size(self, type_id, n_per_row):
        return int(self._dll.ggml_row_size_for(normalize_type(type_id), int(n_per_row)))

    def output_nbytes(self, type_id, nrows, n_per_row):
        type_num = normalize_type(type_id)
        nrows = int(nrows)
        n_per_row = int(n_per_row)
        self._validate_dimensions(type_num, nrows, n_per_row)
        return nrows * self.row_size(type_num, n_per_row)

    def output_shape(self, type_id, nrows, n_per_row):
        return (int(nrows), self.row_size(type_id, int(n_per_row)))

    def quantize_numpy(self, arr, type_id, imatrix=None, require_imatrix=True):
        arr = self._prepare_array(arr)
        type_num = normalize_type(type_id)
        nrows, n_per_row = arr.shape
        self._validate_dimensions(type_num, nrows, n_per_row)
        imatrix = self._prepare_imatrix(arr, type_num, imatrix, require_imatrix)

        out_nbytes = self.output_nbytes(type_num, nrows, n_per_row)
        dst = np.empty(out_nbytes, dtype=np.uint8)
        self._quantize_into(arr, type_num, dst, imatrix)
        return dst

    def quantize_rows(self, arr, type_id, imatrix=None, require_imatrix=True):
        arr = self._prepare_array(arr)
        out = self.quantize_numpy(arr, type_id, imatrix=imatrix, require_imatrix=require_imatrix)
        return out.reshape(self.output_shape(type_id, arr.shape[0], arr.shape[1]))

    def quantize_numpy_to(self, arr, type_id, dst, imatrix=None, require_imatrix=True):
        arr = self._prepare_array(arr)
        type_num = normalize_type(type_id)
        nrows, n_per_row = arr.shape
        self._validate_dimensions(type_num, nrows, n_per_row)
        imatrix = self._prepare_imatrix(arr, type_num, imatrix, require_imatrix)

        dst = np.asarray(dst)
        if dst.dtype != np.uint8:
            raise ValueError(f"dst must have dtype uint8, got {dst.dtype}")
        if not dst.flags.c_contiguous:
            raise ValueError("dst must be C-contiguous")
        expected = self.output_nbytes(type_num, nrows, n_per_row)
        if dst.size != expected:
            raise ValueError(f"dst has {dst.size} bytes, expected {expected}")
        return self._quantize_into(arr, type_num, dst.reshape(-1), imatrix)

    def quantize_to_file(self, arr, type_id, path, imatrix=None, require_imatrix=True):
        out = self.quantize_numpy(arr, type_id, imatrix=imatrix, require_imatrix=require_imatrix)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(out.tobytes())
        return path

    def _quantize_into(self, arr, type_num, dst, imatrix):
        src_ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        dst_ptr = dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        im_ptr = imatrix.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if imatrix is not None else None
        nrows, n_per_row = arr.shape
        result = self._dll.quantize_tensor(type_num, src_ptr, dst_ptr, nrows, n_per_row, im_ptr)
        if result != dst.size:
            raise RuntimeError(f"quantize_tensor returned {result} bytes, expected {dst.size}")
        return dst

    def _prepare_array(self, arr):
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(f"arr must be 1-D or 2-D, got shape {arr.shape}")
        if arr.dtype != np.float32 or not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr, dtype=np.float32)
        return arr

    def _prepare_imatrix(self, arr, type_num, imatrix, require_imatrix):
        if imatrix is None:
            if require_imatrix and type_num in IMATRIX_REQUIRED_TYPES:
                name = type_name(type_num)
                raise ValueError(f"{name} requires an imatrix for GGML-compatible quantization")
            return None
        imatrix = np.asarray(imatrix)
        if imatrix.ndim == 1 and arr.shape[0] == 1:
            imatrix = imatrix.reshape(1, -1)
        if imatrix.shape != arr.shape:
            raise ValueError(f"imatrix shape {imatrix.shape} != arr shape {arr.shape}")
        if imatrix.dtype != np.float32 or not imatrix.flags.c_contiguous:
            imatrix = np.ascontiguousarray(imatrix, dtype=np.float32)
        return imatrix

    def _validate_dimensions(self, type_num, nrows, n_per_row):
        if nrows <= 0:
            raise ValueError(f"nrows must be positive, got {nrows}")
        if n_per_row <= 0:
            raise ValueError(f"n_per_row must be positive, got {n_per_row}")
        block = self.block_size(type_num)
        if block <= 0:
            raise ValueError(f"unsupported GGML type {type_num}")
        if n_per_row % block != 0:
            raise ValueError(f"n_per_row ({n_per_row}) must be a multiple of block size ({block})")


_default_instance = None


def get_hip_quant(dll_path=None):
    global _default_instance
    if dll_path is not None:
        return HipQuant(dll_path)
    if _default_instance is None:
        _default_instance = HipQuant()
    return _default_instance


def quantize(arr, type_id, imatrix=None, require_imatrix=True):
    return get_hip_quant().quantize_numpy(arr, type_id, imatrix=imatrix, require_imatrix=require_imatrix)


def quantize_rows(arr, type_id, imatrix=None, require_imatrix=True):
    return get_hip_quant().quantize_rows(arr, type_id, imatrix=imatrix, require_imatrix=require_imatrix)
