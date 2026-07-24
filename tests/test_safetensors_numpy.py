from __future__ import annotations

import json
import struct

import numpy as np

from hq2.safetensors_numpy import SafeTensorNumpyFile


def _write_safetensors(path, entries, payload):
    header = json.dumps(entries, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)


def test_reads_bf16_without_torch(tmp_path):
    # BF16 encodings for [1.0, -2.5, +inf, nan].
    values = np.array([0x3F80, 0xC020, 0x7F80, 0x7FC0], dtype="<u2")
    path = tmp_path / "bf16.safetensors"
    _write_safetensors(
        path,
        {"weight": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]}},
        values.tobytes(),
    )
    actual = SafeTensorNumpyFile(path).array("weight")
    assert actual.dtype == np.float32
    assert np.array_equal(actual[:1], np.array([[1.0, -2.5]], dtype=np.float32))
    assert np.isposinf(actual[1, 0])
    assert np.isnan(actual[1, 1])


def test_reads_f16_and_validates_payload_length(tmp_path):
    values = np.array([1.0, -0.5], dtype="<f2")
    path = tmp_path / "f16.safetensors"
    _write_safetensors(
        path,
        {"weight": {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]}},
        values.tobytes(),
    )
    assert np.array_equal(SafeTensorNumpyFile(path).array("weight"), values)
