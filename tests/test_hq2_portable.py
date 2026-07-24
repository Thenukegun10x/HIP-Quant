"""Public HQ2 library tests that do not require a native HIP installation."""

from __future__ import annotations

import os
import sys
import json
import struct

import numpy as np
import pytest


# The repository root is the hip_quant package, while hq2 is a normal child
# package.  Put both import views on sys.path for source-tree test runs.
_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_root = os.path.dirname(_src)
for _path in (_root, _src):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import hq2
from hq2.archive import LEGACY_MODEL_MAGIC, LEGACY_MODEL_VERSION, PAYLOAD_ALIGNMENT
from hq2.format import BLOCK_BYTES, decode_numpy
from hq2.hq3 import HQ3_BLOCK_BYTES, decode_hq3_numpy


def test_cpu_layout_roundtrip_and_portable_file(tmp_path):
    rng = np.random.default_rng(22)
    values = rng.normal(size=(3, 512)).astype(np.float32)
    importance = (0.1 + rng.random(values.shape)).astype(np.float32)

    packed = hq2.quantize(values, importance=importance, backend="cpu", iterations=8)
    assert packed.packed.shape == (6, BLOCK_BYTES)
    assert packed.packed.dtype == np.uint8
    assert packed.nbytes == 6 * BLOCK_BYTES
    assert packed.bits_per_weight == 2.25
    assert packed.importance_weighted

    decoded = packed.dequantize()
    np.testing.assert_array_equal(decoded, decode_numpy(packed.packed, values.shape))
    assert np.isfinite(decoded).all()

    path = packed.save(tmp_path / "weights.hq2.npz")
    loaded = hq2.load(path)
    np.testing.assert_array_equal(loaded.packed, packed.packed)
    np.testing.assert_array_equal(loaded.dequantize(), decoded)
    assert loaded.shape == values.shape

    hq2_path = packed.save(tmp_path / "packed_tensor.hq2")
    assert hq2_path.exists()
    assert not (tmp_path / "packed_tensor.hq2.npz").exists()
    from_hq2 = hq2.load(hq2_path)
    np.testing.assert_array_equal(from_hq2.packed, packed.packed)
    assert from_hq2.backend == "hq2-file"


def test_hq2_model_archive_is_lazy_aligned_and_never_requantizes(tmp_path):
    rng = np.random.default_rng(29)
    first = hq2.quantize(rng.normal(size=(2, 512)).astype(np.float32), backend="cpu")
    second = hq2.quantize(rng.normal(size=(3, 256)).astype(np.float32), backend="cpu")
    path = tmp_path / "gemma-hq2.hq2"

    model = hq2.save_model(
        path,
        {"model.layers.0.mlp.up_proj.weight": first, "model.layers.0.mlp.down_proj.weight": second},
        metadata={"architecture": "gemma", "quantization": "HQ2", "selected_layers": "mlp"},
    )
    assert model.tensor_names == (
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    )
    assert model.metadata["architecture"] == "gemma"
    assert all(model._entries[name]["offset"] % 4096 == 0 for name in model.tensor_names)

    loaded_first = hq2.load_model(path).tensor("model.layers.0.mlp.up_proj.weight")
    assert isinstance(loaded_first.packed, np.memmap)
    np.testing.assert_array_equal(loaded_first.packed, first.packed)
    np.testing.assert_array_equal(loaded_first.dequantize(), first.dequantize())
    with pytest.raises(ValueError, match="use hq2.load_model"):
        hq2.load(path)


def test_hq3_cpu_layout_roundtrip_portable_file_and_archive(tmp_path):
    rng = np.random.default_rng(23)
    values = rng.normal(size=(3, 512)).astype(np.float32)
    importance = (0.1 + rng.random(values.shape)).astype(np.float32)

    packed = hq2.quantize(values, importance=importance, backend="cpu", format="hq3", iterations=8)
    assert isinstance(packed, hq2.HQ3Tensor)
    assert packed.packed.shape == (6, HQ3_BLOCK_BYTES)
    assert packed.bits_per_weight == 3.5
    decoded = packed.dequantize()
    np.testing.assert_array_equal(decoded, decode_hq3_numpy(packed.packed, values.shape))
    assert np.isfinite(decoded).all()

    path = packed.save(tmp_path / "weights.hq3.npz")
    loaded = hq2.load(path)
    assert isinstance(loaded, hq2.HQ3Tensor)
    np.testing.assert_array_equal(loaded.packed, packed.packed)

    archive_path = packed.save(tmp_path / "packed_tensor.hq3")
    archive = hq2.load_model(archive_path)
    restored = archive.hq3_tensor("__hq3_tensor__")
    assert restored.backend == "hq3-file"
    np.testing.assert_array_equal(restored.packed, packed.packed)
    np.testing.assert_array_equal(restored.dequantize(), decoded)


def test_v2_archive_uses_compact_footer_index_and_hq2_payload_contract(tmp_path):
    packed = hq2.quantize(np.arange(512, dtype=np.float32).reshape(1, 512), backend="cpu")
    path = tmp_path / "compact.hq2"
    hq2.save_model(path, {"weight": packed}, metadata={"family": "HQ"})

    raw = path.read_bytes()
    # The v2 envelope begins payload data at the first page, not after v1's
    # former 1 MiB reservation.  The index lives after the packed tensor.
    assert raw[:8] == b"HQMODL2\x00"
    assert len(raw) < (1 << 20)
    model = hq2.load_model(path)
    entry = model.descriptor("weight")
    assert entry.offset == PAYLOAD_ALIGNMENT
    assert entry.format.name == "HQ2"
    assert entry.format.packing == "centroids-f16-le+selectors-u2-lsb0"
    assert entry.nbytes == packed.nbytes
    np.testing.assert_array_equal(model.payload("weight"), packed.numpy().reshape(-1))


def test_analyzer_reports_storage_layout_deep_hq2_health_and_cli_json(tmp_path, capsys):
    packed = hq2.quantize(np.arange(2 * 512, dtype=np.float32).reshape(2, 512), backend="cpu", iterations=8)
    path = tmp_path / "inspect-me.hq2"
    hq2.save_model(
        path,
        {"model.layers.0.mlp.up_proj.weight": packed},
        metadata={"architecture": "test-model", "quantization": "HQ2"},
    )

    analysis = hq2.analyze_model(path, deep=True, checksums=True)
    assert analysis.container_version == 2
    assert analysis.storage["logical_value_count"] == 1024
    assert analysis.storage["payload_bits_per_weight"] == 2.25
    assert analysis.formats[0]["declared_bits_per_weight"] == 2.25
    row = analysis.tensors[0]
    codec = row["codec_analysis"]
    assert codec["nonfinite_centroid_count"] == 0
    assert sum(codec["selector_histogram"]) == 1024
    assert len(row["sha256"]) == 64
    assert analysis.integrity["hq2_deep_scan_mode"] == "full"
    assert "Container: HQ v2" in hq2.render_analysis(analysis)

    from hq2.__main__ import main

    assert main(["analyze", str(path), "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["container"] == {"family": "HQ", "version": 2}
    assert output["tensors"][0]["name"] == "model.layers.0.mlp.up_proj.weight"


def test_archive_mixes_hq2_hq3_without_changing_hq2(tmp_path):
    hq2_weight = hq2.quantize(np.arange(256, dtype=np.float32).reshape(1, 256), backend="cpu")
    hq3_weight = hq2.quantize(np.arange(256, dtype=np.float32).reshape(1, 256), backend="cpu", format="hq3")
    path = tmp_path / "mixed-family.hq"
    with hq2.HQModelWriter(path, metadata={"family": "HQ"}) as writer:
        writer.add("hq2.weight", hq2_weight)
        writer.add("hq3.weight", hq3_weight)

    model = hq2.load_model(path)
    assert {format.name for format in model.formats} == {"HQ2", "HQ3"}
    from hq2.transformers_loader import _packed_names

    assert _packed_names(model) == ("hq2.weight", "hq3.weight")
    hq3 = model.descriptor("hq3.weight")
    assert hq3.format == hq2.HQ3_FORMAT
    np.testing.assert_array_equal(model.payload("hq3.weight"), hq3_weight.numpy().reshape(-1))
    np.testing.assert_array_equal(model.tensor("hq2.weight").packed, hq2_weight.packed)
    np.testing.assert_array_equal(model.tensor("hq3.weight").packed, hq3_weight.packed)

    analysis = hq2.analyze_model(path, deep=True)
    assert len(analysis.formats) == 2
    assert analysis.integrity["hq3_payloads_deep_scanned"] == 1
    assert analysis.integrity["hq3_nonfinite_centroid_count"] == 0
    assert sum(analysis.integrity["hq3_selector_histogram"]) == 256
    assert analysis.integrity["deep_scan_not_implemented_for"] == []


def test_v1_hq2_archive_stays_readable(tmp_path):
    """The first Gemma archive remains valid after the compact v2 upgrade."""
    packed = hq2.quantize(np.arange(256, dtype=np.float32).reshape(1, 256), backend="cpu")
    path = tmp_path / "legacy.hq2"
    header = struct.Struct("<8sIQQ")
    manifest = {
        "format": "HQ2MODEL",
        "version": LEGACY_MODEL_VERSION,
        "layout": "linear_out_in_row_major_blocks256",
        "block_size": 256,
        "block_bytes": 72,
        "metadata": {"legacy": True},
        "tensors": {
            "weight": {
                "shape": [1, 256],
                "offset": PAYLOAD_ALIGNMENT,
                "nbytes": packed.nbytes,
                "iterations": packed.iterations,
                "importance_weighted": packed.importance_weighted,
            }
        },
    }
    encoded = json.dumps(manifest, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as file:
        file.write(b"\0" * PAYLOAD_ALIGNMENT)
        file.seek(PAYLOAD_ALIGNMENT)
        packed.numpy().tofile(file)
        file.seek(0)
        file.write(header.pack(LEGACY_MODEL_MAGIC, LEGACY_MODEL_VERSION, len(encoded), PAYLOAD_ALIGNMENT))
        file.write(encoded)

    model = hq2.load_model(path)
    assert model.metadata == {"legacy": True}
    assert model.descriptor("weight").format == hq2.HQ2_FORMAT
    np.testing.assert_array_equal(model.tensor("weight").packed, packed.packed)

    migrated_path = tmp_path / "migrated.hq2"
    migrated = hq2.repack_model(path, migrated_path)
    assert migrated_path.read_bytes()[:8] == b"HQMODL2\x00"
    assert migrated.metadata == model.metadata
    np.testing.assert_array_equal(migrated.tensor("weight").packed, packed.packed)


def test_cpu_is_independent_of_native_rocm_package():
    values = np.linspace(-2.0, 2.0, 256, dtype=np.float32)
    packed = hq2.quantize(values, backend="cpu")
    assert packed.backend == "cpu"
    assert packed.dequantize().shape == values.shape


def test_invalid_shape_and_unimplemented_vulkan_are_clear():
    with pytest.raises(ValueError, match="final dimension"):
        hq2.quantize(np.zeros((1, 257), dtype=np.float32))
    with pytest.raises(hq2.BackendUnavailable, match="Vulkan"):
        hq2.quantize(np.zeros((1, 256), dtype=np.float32), backend="vulkan")


def test_torch_cpu_roundtrip_matches_packed_cpu_decode():
    torch = pytest.importorskip("torch")
    values = torch.linspace(-3, 3, 512, dtype=torch.float32).reshape(1, 512)
    packed = hq2.quantize(values, backend="torch", iterations=8)
    assert packed.backend == "torch-cpu"
    assert packed.packed.dtype == torch.uint8

    decoded = packed.dequantize()
    expected = decode_numpy(packed.numpy(), tuple(values.shape))
    np.testing.assert_array_equal(decoded.numpy(), expected)


def test_hq3_torch_cpu_roundtrip_and_linear_reference():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    values = torch.linspace(-3, 3, 2 * 512, dtype=torch.float32).reshape(2, 512)
    packed = hq2.quantize(values, backend="torch", format="hq3", iterations=8)
    assert packed.backend == "torch-cpu"
    assert packed.packed.shape == (4, HQ3_BLOCK_BYTES)
    decoded = packed.dequantize()
    expected = decode_hq3_numpy(packed.numpy(), tuple(values.shape))
    np.testing.assert_array_equal(decoded.numpy(), expected)

    layer = hq2.HQ3Linear(packed)
    input = torch.randn(3, 512, dtype=torch.float32)
    torch.testing.assert_close(layer(input), F.linear(input, decoded), rtol=0.0, atol=0.0)


def test_torch_low_sum_importance_does_not_shrink_hq_centroids():
    """Uniform importance scaling must leave each learned codebook unchanged."""
    torch = pytest.importorskip("torch")
    values = torch.linspace(-1.0, 1.0, 512, dtype=torch.float32).reshape(2, 256)
    # Every populated centroid has total weight below one. This specifically
    # guards against accidentally dividing weighted totals by 1.0.
    importance = torch.full_like(values, 1.0e-4)
    for format_name in ("hq2", "hq3"):
        baseline = hq2.quantize(values, backend="torch", format=format_name, iterations=8)
        weighted = hq2.quantize(values, importance=importance, backend="torch", format=format_name, iterations=8)
        baseline_error = (values - baseline.dequantize()).square().mean()
        weighted_error = (values - weighted.dequantize()).square().mean()
        # The exact FP16 centroid bytes can differ at a rounding boundary, but
        # uniform scaling must preserve the fitted quality. The old bug made
        # this error many orders of magnitude worse by dividing by 1.0.
        assert weighted_error <= baseline_error * 1.01


def test_hq2linear_cpu_reference_matches_decoded_linear():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    values = np.linspace(-2.0, 2.0, 2 * 256, dtype=np.float32).reshape(2, 256)
    packed = hq2.quantize(values, backend="cpu")
    layer = hq2.HQ2Linear(packed)
    input = torch.linspace(-1.0, 1.0, 3 * 256, dtype=torch.float32).reshape(3, 256)
    actual = layer(input)
    expected = F.linear(input, torch.from_numpy(packed.dequantize()))
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_raw_tensor_roundtrip_preserves_bfloat16_bits_for_standalone_packages(tmp_path):
    torch = pytest.importorskip("torch")
    original = torch.tensor([[-1.5, 0.0, 0.125], [3.0, -0.25, 8.0]], dtype=torch.bfloat16)
    path = tmp_path / "raw-package.hq2"
    with hq2.HQModelWriter(path, metadata={"standalone_package": True}) as writer:
        writer.add_raw(
            "exact.weight",
            original.contiguous().view(torch.uint8).numpy(),
            shape=tuple(original.shape),
            format=hq2.raw_format_for_torch(original.dtype),
        )

    model = hq2.load_model(path)
    assert model.descriptor("exact.weight").format.name == "RAW"
    stored = model.raw_tensor("exact.weight")
    try:
        restored = stored.to_torch()
    finally:
        stored.close()
    torch.testing.assert_close(restored, original, rtol=0.0, atol=0.0)
    assert torch.equal(restored.view(torch.uint8), original.view(torch.uint8))
    analysis = hq2.analyze_model(path)
    assert analysis.formats[0]["format"]["name"] == "RAW"
    assert analysis.formats[0]["payload_bits_per_weight"] == 16.0


@pytest.mark.gpu
def test_torch_accelerator_roundtrip_when_explicitly_enabled():
    """Run intentionally on CUDA or ROCm with ``HQ2_TEST_GPU=1``."""
    if os.environ.get("HQ2_TEST_GPU") != "1":
        pytest.skip("set HQ2_TEST_GPU=1 to run the intentional accelerator check")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm Torch device visible")

    values = torch.randn((2, 512), device="cuda", dtype=torch.bfloat16)
    packed = hq2.quantize(values, backend="torch", iterations=8)
    assert packed.packed.device == values.device
    decoded = packed.dequantize()
    assert decoded.device == values.device
    assert torch.isfinite(decoded).all()
    expected = decode_numpy(packed.numpy(), (2, 512))
    np.testing.assert_array_equal(decoded.cpu().numpy(), expected)

    # A CPU-generated portable block may be moved without a re-quantization.
    cpu_packed = hq2.quantize(values.float().cpu().numpy(), backend="cpu", iterations=8)
    moved = cpu_packed.to(values.device)
    assert moved.packed.device == values.device
    np.testing.assert_array_equal(moved.dequantize().cpu().numpy(), cpu_packed.dequantize())

    # Asking for BF16 is a final output cast, so its values are intentionally
    # rounded relative to the exact FP16-centroid decode above.
    decoded_bf16 = packed.dequantize(dtype=torch.bfloat16)
    np.testing.assert_array_equal(
        decoded_bf16.float().cpu().numpy(),
        torch.from_numpy(expected).to(torch.bfloat16).float().numpy(),
    )


@pytest.mark.gpu
def test_hq2linear_rocm_fused_matches_direct_decode_when_explicitly_enabled():
    if os.environ.get("HQ2_TEST_GPU") != "1":
        pytest.skip("set HQ2_TEST_GPU=1 to run the intentional accelerator check")
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    if not torch.cuda.is_available() or not torch.version.hip:
        pytest.skip("requires ROCm Torch")
    if not hq2.rocm_fused_available():
        pytest.skip("rebuild hip_quant._C with setup_torch.py to enable the HQ2 fused kernel")

    rng = np.random.default_rng(47)
    packed = hq2.quantize(rng.normal(size=(37, 512)).astype(np.float32), backend="cpu").to("cuda")
    layer = hq2.HQ2Linear(packed)
    input = torch.randn((3, 512), device="cuda", dtype=torch.float32)
    actual = layer(input)
    expected = F.linear(input, packed.dequantize())
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=4e-5)


@pytest.mark.gpu
def test_hq3linear_rocm_fused_matches_direct_decode_when_explicitly_enabled():
    if os.environ.get("HQ2_TEST_GPU") != "1":
        pytest.skip("set HQ2_TEST_GPU=1 to run the intentional accelerator check")
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    if not torch.cuda.is_available() or not torch.version.hip:
        pytest.skip("requires ROCm Torch")
    if not hq2.rocm_hq3_fused_available():
        pytest.skip("rebuild hip_quant._C with setup_torch.py to enable the HQ3 fused kernel")

    rng = np.random.default_rng(48)
    packed = hq2.quantize(
        rng.normal(size=(37, 512)).astype(np.float32), backend="cpu", format="hq3"
    ).to("cuda")
    layer = hq2.HQ3Linear(packed)
    input = torch.randn((3, 512), device="cuda", dtype=torch.float32)
    actual = layer(input)
    expected = F.linear(input, packed.dequantize())
    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=6e-5)
