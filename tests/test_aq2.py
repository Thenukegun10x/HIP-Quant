"""CPU and optional ROCm checks for the first AQ2 implementation."""

import os
import sys

import numpy as np
import pytest

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_root = os.path.dirname(_src)
for _path in (_root, _src):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from hip_quant import GGML_TYPE, GGML_TYPE_BLOCK_BYTES, GGML_TYPE_BLOCK_SIZE
from hip_quant.cdna_compat import _dequantize_hq2, cpu_reference_quantize


def test_aq2_registration_and_wire_budget():
    assert GGML_TYPE["AQ2"] == 39
    assert GGML_TYPE_BLOCK_SIZE[39] == 256
    assert GGML_TYPE_BLOCK_BYTES[39] == 72
    assert GGML_TYPE_BLOCK_BYTES[39] * 8 / 256 == 2.25
    for name, type_id in (("AQ2_QK", 40), ("AQ2_VO", 41)):
        assert GGML_TYPE[name] == type_id
        assert GGML_TYPE_BLOCK_SIZE[type_id] == 256
        assert GGML_TYPE_BLOCK_BYTES[type_id] == 72
        assert GGML_TYPE_BLOCK_BYTES[type_id] * 8 / 256 == 2.25


def test_aq2_cpu_reference_uses_attention_map_and_is_finite():
    rng = np.random.default_rng(42)
    values = rng.standard_normal((2, 512)).astype(np.float32)
    # A role-aware calibration map is represented as a full per-weight map at
    # this ABI boundary. In practice this is generated separately for Q/K, V,
    # and O from the attention capture harness.
    importance = (0.1 + 4.0 * np.abs(values)).astype(np.float32)
    packed = cpu_reference_quantize(
        values, "AQ2", imatrix=importance, aq2_iterations=8
    )
    assert packed.dtype == np.uint8
    assert packed.size == 2 * 2 * 72
    decoded = _dequantize_hq2(packed, values.shape[1])
    assert decoded.shape == values.shape
    assert np.all(np.isfinite(decoded))


def test_aq2_zero_block_and_iteration_determinism():
    zero = np.zeros((1, 256), dtype=np.float32)
    assert np.all(_dequantize_hq2(cpu_reference_quantize(zero, "AQ2"), 256) == 0.0)

    rng = np.random.default_rng(7)
    values = rng.standard_normal((1, 256)).astype(np.float32)
    importance = np.linspace(0.1, 2.0, 256, dtype=np.float32).reshape(1, -1)
    first = cpu_reference_quantize(
        values, "AQ2", imatrix=importance, aq2_iterations=8
    )
    second = cpu_reference_quantize(
        values, "AQ2", imatrix=importance, aq2_iterations=8
    )
    assert np.array_equal(first, second)


def test_aq2_role_types_share_cpu_wire_contract():
    rng = np.random.default_rng(17)
    values = rng.standard_normal((1, 256)).astype(np.float32)
    importance = (0.25 + np.abs(values)).astype(np.float32)
    qk = cpu_reference_quantize(values, "AQ2_QK", imatrix=importance, aq2_iterations=8)
    vo = cpu_reference_quantize(values, "AQ2_VO", imatrix=importance, aq2_iterations=8)
    assert qk.size == vo.size == 72
    assert np.array_equal(qk, vo)


@pytest.mark.gpu
def test_aq2_rocm_kernel_matches_cpu_reference_shape():
    if os.environ.get("AQ2_TEST_GPU") != "1":
        pytest.skip("set AQ2_TEST_GPU=1 for the intentional ROCm check")

    from hip_quant import HipQuant

    quantizer = HipQuant()
    if quantizer.type_size(GGML_TYPE["AQ2"]) != 72:
        pytest.skip("loaded DLL predates AQ2; rebuild hip_quantize.dll first")
    rng = np.random.default_rng(9)
    values = rng.standard_normal((2, 512)).astype(np.float32)
    importance = (0.2 + np.abs(values)).astype(np.float32)
    for type_name in ("AQ2", "AQ2_QK", "AQ2_VO"):
        if quantizer.type_size(GGML_TYPE[type_name]) != 72:
            pytest.fail(f"loaded DLL does not register {type_name}")
        packed = quantizer.quantize_numpy(
            values, GGML_TYPE[type_name], imatrix=importance, aq2_iterations=8
        )
        assert packed.size == 2 * 2 * 72
        decoded = _dequantize_hq2(packed, values.shape[1])
        assert np.all(np.isfinite(decoded))
