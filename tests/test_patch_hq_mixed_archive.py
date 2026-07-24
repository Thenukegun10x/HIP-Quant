from __future__ import annotations

import numpy as np

import hq2
from tools.patch_hq_mixed_archive import _verify_copied_payloads


def test_byte_preserving_verification_accepts_copied_payloads(tmp_path):
    packed = hq2.quantize(np.linspace(-1.0, 1.0, 256, dtype=np.float32).reshape(1, 256), backend="cpu")
    source_path = tmp_path / "source.hq"
    copied_path = tmp_path / "copy.hq"
    with hq2.HQModelWriter(source_path) as writer:
        writer.add("linear.weight", packed)
    source = hq2.load_model(source_path)
    descriptor = source.descriptor("linear.weight")
    with hq2.HQModelWriter(copied_path) as writer:
        writer.add_raw(
            "linear.weight",
            source.payload("linear.weight"),
            shape=descriptor.shape,
            format=descriptor.format,
            iterations=descriptor.iterations,
            importance_weighted=descriptor.importance_weighted,
        )
    copied = hq2.load_model(copied_path)
    assert _verify_copied_payloads(source, copied, ["linear.weight"]) == []
