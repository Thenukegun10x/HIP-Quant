"""CPU coverage for exact Q4/Q8 mixed-archive decoders and error ranking."""

from __future__ import annotations

import os
import sys

import numpy as np


_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_root = os.path.dirname(_src)
for _path in (_root, _src):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from hq2.mixed_policy import Q4_0_FORMAT, Q8_0_FORMAT
from hq2.sensitivity import decode_q4_0_numpy, decode_q8_0_numpy, tensor_error


def test_q4_0_decoder_preserves_ggml_low_then_high_nibble_order():
    packed = np.zeros((1, 18), dtype=np.uint8)
    packed[:, :2] = np.asarray([2.0], dtype="<f2").view(np.uint8).reshape(1, 2)
    low = np.arange(16, dtype=np.uint8)
    high = 15 - low
    packed[:, 2:] = (high << 4) | low
    decoded = decode_q4_0_numpy(packed, (1, 32))
    assert np.array_equal(decoded[0, :16], 2.0 * (low.astype(np.float32) - 8.0))
    assert np.array_equal(decoded[0, 16:], 2.0 * (high.astype(np.float32) - 8.0))


def test_q8_0_decoder_preserves_signed_payload():
    packed = np.zeros((1, 34), dtype=np.uint8)
    packed[:, :2] = np.asarray([0.5], dtype="<f2").view(np.uint8).reshape(1, 2)
    values = np.arange(-16, 16, dtype=np.int8)
    packed[:, 2:] = values.view(np.uint8)
    decoded = decode_q8_0_numpy(packed, (1, 32))
    assert np.array_equal(decoded, (values.astype(np.float32) * 0.5).reshape(1, 32))


def test_activation_weighted_error_uses_input_channel_energy():
    source = np.asarray([[2.0, 3.0], [5.0, 7.0]], dtype=np.float32)
    restored = source.copy()
    restored[:, 1] -= 1.0
    result = tensor_error(
        source,
        restored,
        name="layer.weight",
        format=Q4_0_FORMAT,
        payload_bytes=18,
        importance=np.asarray([0.0, 4.0], dtype=np.float32),
    )
    assert result.mse == 0.5
    assert result.activation_weighted_sse == 8.0
    assert result.error_per_payload_byte == 8.0 / 18.0
    assert result.calibration == "activation-diagonal"
    assert tensor_error(source, source, name="q8.weight", format=Q8_0_FORMAT, payload_bytes=34, importance=None).mse == 0.0
