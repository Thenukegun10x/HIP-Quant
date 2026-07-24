"""Portable tests for the HQ8_G128 reference codec."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest


_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_root = os.path.dirname(_src)
for _path in (_root, _src):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import hq2
from hq2.hq8 import (
    HQ8_G128_BITS_PER_WEIGHT,
    HQ8_G128_BLOCK_BYTES,
    HQ8_G128_BLOCK_SIZE,
    decode_hq8_g128_numpy,
)


def _reference_q8_decode(values: np.ndarray, group_size: int) -> np.ndarray:
    """Same FP16-scale symmetric int8 math, parameterized for Q8 group screens."""

    values = np.ascontiguousarray(values, dtype=np.float32)
    assert values.shape[-1] % group_size == 0
    groups = values.reshape(-1, group_size)
    stored_scales = (np.max(np.abs(groups), axis=1) / 127.0).astype(np.float16)
    scales = stored_scales.astype(np.float32)
    normalized = np.zeros_like(groups)
    nonzero = scales > 0.0
    normalized[nonzero] = groups[nonzero] / scales[nonzero, None]
    codes = np.clip(np.rint(normalized), -127, 127).astype(np.int8)
    return (scales[:, None] * codes.astype(np.float32)).reshape(values.shape)


def test_hq8_g128_roundtrip_has_expected_layout_and_quality_screen(tmp_path):
    rng = np.random.default_rng(8128)
    values = rng.normal(0.0, 0.7, size=(7, 3840)).astype(np.float32)
    values[0, :HQ8_G128_BLOCK_SIZE] = 0.0

    packed = hq2.quantize(values, backend="cpu", format="hq8_g128")
    assert isinstance(packed, hq2.HQ8Tensor)
    assert packed.packed.shape == (values.size // HQ8_G128_BLOCK_SIZE, HQ8_G128_BLOCK_BYTES)
    assert packed.bits_per_weight == HQ8_G128_BITS_PER_WEIGHT == 8.125
    assert packed.nbytes == values.size // HQ8_G128_BLOCK_SIZE * HQ8_G128_BLOCK_BYTES

    decoded = packed.dequantize()
    np.testing.assert_array_equal(decoded, decode_hq8_g128_numpy(packed.packed, values.shape))
    assert np.isfinite(decoded).all()
    np.testing.assert_array_equal(decoded[0, :HQ8_G128_BLOCK_SIZE], np.zeros(HQ8_G128_BLOCK_SIZE, dtype=np.float32))

    q8_0_like = _reference_q8_decode(values, 32)
    g128_mse = float(np.mean(np.square(values - decoded), dtype=np.float64))
    q8_0_like_mse = float(np.mean(np.square(values - q8_0_like), dtype=np.float64))
    assert q8_0_like_mse > 0.0
    assert g128_mse > 0.0
    # The wider group trades some local dynamic-range adaptation for fewer
    # scale operations. It must remain in the same high-bit error regime.
    assert g128_mse / q8_0_like_mse < 4.0

    portable = packed.save(tmp_path / "reference.hq8.npz")
    reloaded = hq2.load(portable)
    assert isinstance(reloaded, hq2.HQ8Tensor)
    np.testing.assert_array_equal(reloaded.packed, packed.packed)
    np.testing.assert_array_equal(reloaded.dequantize(), decoded)


def test_hq8_g128_archive_and_mixed_decoder_are_exact(tmp_path):
    values = np.linspace(-3.0, 3.0, 2 * 3840, dtype=np.float32).reshape(2, 3840)
    packed = hq2.quantize(values, backend="cpu", format="HQ8_G128")
    path = packed.save(tmp_path / "reference.hq8")

    model = hq2.load_model(path)
    descriptor = model.descriptor("__hq8_g128_tensor__")
    assert descriptor.format == hq2.HQ8_G128_FORMAT
    assert descriptor.format.layout == "linear_out_in_row_major_blocks128"
    assert descriptor.format.parameters == {
        "group_size": HQ8_G128_BLOCK_SIZE,
        "quantization": "symmetric-maxabs-int8",
    }
    loaded = model.hq8_g128_tensor("__hq8_g128_tensor__")
    assert loaded.backend == "hq8-g128-file"
    np.testing.assert_array_equal(loaded.packed, packed.packed)
    np.testing.assert_array_equal(hq2.decode_archive_weight(model, "__hq8_g128_tensor__"), packed.dequantize())

    analysis = hq2.analyze_model(path)
    assert analysis.storage["payload_bits_per_weight"] == pytest.approx(8.125)
    assert analysis.formats[0]["format"]["name"] == "HQ8_G128"
    assert analysis.integrity["deep_scan_not_implemented_for"] == []


def test_hq8_g128_rejects_unaligned_nonfinite_and_importance_inputs():
    with pytest.raises(ValueError, match="final dimension"):
        hq2.quantize(np.zeros((1, 129), dtype=np.float32), backend="cpu", format="hq8_g128")
    with pytest.raises(ValueError, match="finite"):
        hq2.quantize(np.array([[np.nan] * 128], dtype=np.float32), backend="cpu", format="hq8_g128")
    with pytest.raises(ValueError, match="importance"):
        hq2.quantize(
            np.zeros((1, 128), dtype=np.float32),
            importance=np.ones((1, 128), dtype=np.float32),
            backend="cpu",
            format="hq8_g128",
        )


def test_hq8_g128_torch_cpu_linear_matches_decoded_reference():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    values = torch.linspace(-2.0, 2.0, 2 * 128, dtype=torch.float32).reshape(2, 128)
    packed = hq2.quantize(values, backend="torch", format="hq8_g128")
    assert packed.backend == "torch-cpu"
    assert packed.packed.dtype == torch.uint8
    layer = hq2.HQ8Linear(packed)
    input = torch.linspace(-1.0, 1.0, 3 * 128, dtype=torch.float32).reshape(3, 128)
    actual = layer(input)
    expected = F.linear(input, packed.dequantize())
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.gpu
def test_hq8_g128_rocm_fused_matches_decoded_reference_when_explicitly_enabled():
    if os.environ.get("HQ2_TEST_GPU") != "1":
        pytest.skip("set HQ2_TEST_GPU=1 to run the intentional accelerator check")
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    if not torch.cuda.is_available() or not torch.version.hip:
        pytest.skip("requires ROCm Torch")
    if not hq2.rocm_hq8_g128_fused_available():
        pytest.skip("rebuild hip_quant._C with setup_torch.py to enable the HQ8_G128 fused kernel")

    rng = np.random.default_rng(8129)
    packed = hq2.quantize(
        rng.normal(size=(37, 512)).astype(np.float32), backend="cpu", format="hq8_g128"
    ).to("cuda")
    layer = hq2.HQ8Linear(packed)
    input = torch.randn((3, 512), device="cuda", dtype=torch.float32)
    actual = layer(input)
    expected = F.linear(input, packed.dequantize())
    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=6e-5)
