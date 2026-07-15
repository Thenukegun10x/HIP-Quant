"""No-GPU validation for the HQ2 quant type.

HQ2 is a TurboQuant-style learned-codebook 2-bit weight quant:
a per-block, importance-weighted 4-level codebook fit by k-means.
These tests exercise the CPU reference path (cdna_compat) and the
Python type registration, so they run without a GPU or the HIP DLL.

Run:
    python tests/test_hq2.py -v
"""

import os
import sys

# Ensure the parent directory is on the path so we can import hip_quant
# (the repo root directory *is* the hip_quant package).
_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_root = os.path.dirname(_src)
for _p in (_root, _src):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from hip_quant import GGML_TYPE, GGML_TYPE_BLOCK_SIZE, GGML_TYPE_BLOCK_BYTES
from hip_quant.cdna_compat import cpu_reference_quantize, _dequantize_hq2


def _uniform2_baseline(x: np.ndarray) -> np.ndarray:
    """Symmetric 4-level uniform quant — the HQ2 initialization levels.

    k-means (HQ2) should never be worse than this on the same weights.
    """
    x = x.astype(np.float64)
    amax = float(np.max(np.abs(x)))
    if amax < 1e-12:
        return np.zeros_like(x)
    levels = np.array([-amax, -amax / 3.0, amax / 3.0, amax])
    d = x[:, None] - levels[None, :]
    c = np.argmin(d * d, axis=1)
    return levels[c].astype(np.float32)


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def test_hq2_registration():
    assert GGML_TYPE["HQ2"] == 38
    assert GGML_TYPE_BLOCK_SIZE[38] == 256
    assert GGML_TYPE_BLOCK_BYTES[38] == 72


def test_hq2_block_size_and_roundtrip():
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((4, 256 * 8)).astype(np.float32)
    packed = cpu_reference_quantize(arr, "HQ2")
    assert packed.dtype == np.uint8
    assert len(packed) == 4 * 8 * 72

    rec = _dequantize_hq2(packed, 256 * 8)
    assert rec.shape == arr.shape

    # k-means is monotonically non-worse than its uniform initialization
    err_hq2 = _mse(rec, arr)
    err_base = _mse(_uniform2_baseline(arr.reshape(-1)), arr.reshape(-1))
    assert err_hq2 <= err_base * 1.05, (err_hq2, err_base)


def test_hq2_with_importance_matrix():
    rng = np.random.default_rng(1)
    arr = rng.standard_normal((2, 256 * 4)).astype(np.float32)
    # Emphasize the largest-magnitude weights (TurboQuant-style saliency).
    imatrix = np.abs(arr).astype(np.float32)
    packed = cpu_reference_quantize(arr, "HQ2", imatrix=imatrix)
    assert len(packed) == 2 * 4 * 72

    rec = _dequantize_hq2(packed, 256 * 4)
    assert rec.shape == arr.shape
    assert np.all(np.isfinite(rec))

    # Importance weighting should not make the overall fit wildly worse than
    # the uniform baseline (it trades uniform error for salient-weight error).
    err_hq2 = _mse(rec, arr)
    err_base = _mse(_uniform2_baseline(arr.reshape(-1)), arr.reshape(-1))
    assert err_hq2 < err_base * 1.5, (err_hq2, err_base)


def test_hq2_zero_block():
    arr = np.zeros((1, 256), dtype=np.float32)
    packed = cpu_reference_quantize(arr, "HQ2")
    assert len(packed) == 72
    rec = _dequantize_hq2(packed, 256)
    assert np.all(rec == 0.0)


if __name__ == "__main__":
    test_hq2_registration()
    test_hq2_block_size_and_roundtrip()
    test_hq2_with_importance_matrix()
    test_hq2_zero_block()
    print("All HQ2 CPU-reference tests passed.")
