import ctypes
import numpy as np
import os

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
}

_ROCM_BIN = r"C:\Program Files\AMD\ROCm\7.1\bin"

class HipQuant:
    def __init__(self, dll_path=None):
        # Ensure ROCm 7.1 runtime is searched before PyTorch's ROCm 7.2 venv
        if os.path.isdir(_ROCM_BIN):
            os.add_dll_directory(_ROCM_BIN)
        if dll_path is None:
            candidates = [
                os.path.join(_PKG_DIR, "hip_quantize.dll"),
                os.path.join(_PKG_DIR, "..", "hip_quantize.dll"),
                os.path.join(_PKG_DIR, "..", "..", "src", "hip_quantize.dll"),
            ]
            for p in candidates:
                p = os.path.normpath(p)
                if os.path.isfile(p):
                    dll_path = p
                    break
            if dll_path is None:
                raise FileNotFoundError(
                    f"hip_quantize.dll not found. Tried: {candidates}"
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
        self._dll.ggml_type_size_for.restype = ctypes.c_size_t
        self._dll.ggml_type_size_for.argtypes = [ctypes.c_int]
        self._dll.ggml_blck_size_for.restype = ctypes.c_size_t
        self._dll.ggml_blck_size_for.argtypes = [ctypes.c_int]
        self._dll.ggml_row_size_for.restype = ctypes.c_size_t
        self._dll.ggml_row_size_for.argtypes = [ctypes.c_int, ctypes.c_int64]
        self._dll.get_device_name.restype = ctypes.c_char_p
        self._dll.get_device_count.restype = ctypes.c_int
        self._dll.get_device_count.argtypes = []

    @property
    def device_count(self):
        return self._dll.get_device_count()

    @property
    def device_name(self):
        return self._dll.get_device_name().decode("utf-8")

    def type_size(self, type_num):
        return self._dll.ggml_type_size_for(int(type_num))

    def blck_size(self, type_num):
        return self._dll.ggml_blck_size_for(int(type_num))

    def row_size(self, type_num, n_per_row):
        return self._dll.ggml_row_size_for(int(type_num), n_per_row)

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
