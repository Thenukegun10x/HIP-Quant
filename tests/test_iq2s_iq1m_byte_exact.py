"""GPU tests for the IQ2_S / IQ1_M on-device quantizers.

Structural tests run whenever the hip_quant DLL loads. The byte-exact tests
round-trip deterministic float32 tensors through llama-quantize.exe and
compare the quantized bytes to hip_quant's kernels; they are skipped when
llama-quantize.exe is not present.

Requirements: AMD GPU with hip_quant DLL loaded.
"""
import os
import sys
import gc
import struct
import subprocess
import tempfile
import pathlib
import unittest

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ROOT.parent))

_FRESH_DLL = _ROOT / "hip_quantize.dll"
if _FRESH_DLL.exists():
    os.environ["HIP_QUANT_DLL"] = str(_FRESH_DLL)

try:
    from hip_quant import (
        get_hip_quant,
        GGML_TYPE,
        GGML_TYPE_BLOCK_SIZE,
        GGML_TYPE_BLOCK_BYTES,
    )
    HAS_HIP_QUANT = True
except ImportError:
    HAS_HIP_QUANT = False

try:
    import gguf
    HAS_GGUF = True
except ImportError:
    HAS_GGUF = False

requires_dll = unittest.skipUnless(HAS_HIP_QUANT, "hip_quant DLL required")

_QUANTIZE_CANDIDATES = [
    _ROOT / "Own Quant" / "llama_cpp_stock_vulkan_build" / "bin" / "llama-quantize.exe",
    _ROOT / "Own Quant" / "llama_cpp_stock" / "bin" / "llama-quantize.exe",
]
_QUANTIZE_EXE = next((p for p in _QUANTIZE_CANDIDATES if p.exists()), None)

requires_llama = unittest.skipUnless(
    _QUANTIZE_EXE is not None and HAS_GGUF, "llama-quantize.exe / gguf not found"
)

_SEED = 20260708
_IQ2_S = GGML_TYPE["IQ2_S"]
_IQ1_M = GGML_TYPE["IQ1_M"]
_Q1_0 = GGML_TYPE["Q1_0"]
_Q2_0 = GGML_TYPE["Q2_0"]
_Q8_K = GGML_TYPE["Q8_K"]
_BF16 = GGML_TYPE["BF16"]


def _nearest_int_rne(f):
    """Bit-exact port of ggml nearest_int (ties to even)."""
    f = f.astype(np.float32) + np.float32(12582912.0)
    return (f.view(np.int32) & 0x007fffff) - 0x00400000


def _q8_k_ref(arr):
    """Port of quantize_row_q8_K_ref from ggml-quants.c."""
    out = []
    nb = arr.shape[1] // 256
    for r in range(arr.shape[0]):
        for b in range(nb):
            blk = arr[r, b * 256:(b + 1) * 256]
            amax = np.abs(blk).max()
            if amax == 0:
                d = np.float32(0)
                qs = np.zeros(256, dtype=np.int8)
                bs = np.zeros(16, dtype=np.int16)
            else:
                iscale = np.float32(-127.0) / np.float32(amax)
                v = _nearest_int_rne(iscale * blk)
                v = np.minimum(v, 127)
                qs = v.astype(np.int8)
                bs = qs.reshape(16, 16).sum(axis=1, dtype=np.int32).astype(np.int16)
                d = np.float32(1.0) / iscale
            out.append(d.tobytes() + qs.tobytes() + bs.tobytes())
    return b"".join(out)


def _make_tensors(rows, cols, rng):
    return {
        "tok_embeddings.weight": rng.randn(rows, cols).astype(np.float32),
        "norm.weight": (rng.randn(rows, cols).astype(np.float32) * 0.5),
        "blk.0.attn_q.weight": (rng.randn(rows, cols).astype(np.float32) * 3.0 - 0.5),
        "blk.0.attn_k.weight": (rng.randn(rows, cols).astype(np.float32) + 2.0),
    }


def _write_imatrix(path, tensors):
    """Legacy .n imatrix: one float per COLUMN (shared across rows)."""
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(tensors)))
        for name, arr in tensors.items():
            b = name.encode("utf-8")
            f.write(struct.pack("<i", len(b)))
            f.write(b)
            f.write(struct.pack("<ii", 1, arr.shape[1]))
            f.write(np.ones(arr.shape[1], dtype=np.float32).tobytes())


def _write_gguf(path, tensors, cols):
    w = gguf.GGUFWriter(path, "llama")
    w.add_name("e2e")
    w.add_context_length(128)
    w.add_embedding_length(cols)
    w.add_block_count(1)
    w.add_head_count(2)
    w.add_head_count_kv(1)
    w.add_feed_forward_length(64)
    w.add_vocab_size(300)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_tokenizer_model("llama")
    w.add_token_list(["\u0000"] * 300)
    w.add_token_scores([0.0] * 300)
    w.add_bos_token_id(0)
    w.add_eos_token_id(0)
    for name, arr in tensors.items():
        w.add_tensor(name, arr, raw_dtype=gguf.GGMLQuantizationType.F32)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def _run_llama_quantize(src, out, imatrix_path, dtype, force_types):
    cmd = [str(_QUANTIZE_EXE)]
    if imatrix_path:
        cmd += ["--imatrix", imatrix_path]
    for opt in (force_types or []):
        cmd += ["--tensor-type", opt]
    cmd += [src, out, dtype, "1"]
    env = dict(os.environ)
    env["PATH"] = str(_QUANTIZE_EXE.parent) + os.pathsep + env.get("PATH", "")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            f"llama-quantize failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
        )


def _probe(hq):
    """Verify all supported types dispatch on the loaded DLL; skip otherwise."""
    probe = np.ones((1, 256), dtype=np.float32)
    imat = np.ones((1, 256), dtype=np.float32)
    for qtype in (_IQ2_S, _IQ1_M, _Q1_0, _Q2_0, _Q8_K, _BF16):
        hq.quantize_numpy(probe, qtype, imatrix=imat)


@requires_dll
class TestIQStruct(unittest.TestCase):
    """Structural invariants: block sizes, dispatch, output lengths."""

    @classmethod
    def setUpClass(cls):
        cls.hq = get_hip_quant()
        try:
            _probe(cls.hq)
        except Exception as exc:
            raise unittest.SkipTest(f"Q types not available: {exc}")

    def test_block_sizes(self):
        self.assertEqual(GGML_TYPE_BLOCK_SIZE[_IQ2_S], 256)
        self.assertEqual(GGML_TYPE_BLOCK_BYTES[_IQ2_S], 82)
        self.assertEqual(GGML_TYPE_BLOCK_SIZE[_IQ1_M], 256)
        self.assertEqual(GGML_TYPE_BLOCK_BYTES[_IQ1_M], 56)
        self.assertEqual(GGML_TYPE_BLOCK_SIZE[_Q1_0], 128)
        self.assertEqual(GGML_TYPE_BLOCK_BYTES[_Q1_0], 18)
        self.assertEqual(GGML_TYPE_BLOCK_SIZE[_Q2_0], 64)
        self.assertEqual(GGML_TYPE_BLOCK_BYTES[_Q2_0], 18)
        self.assertEqual(GGML_TYPE_BLOCK_SIZE[_Q8_K], 256)
        self.assertEqual(GGML_TYPE_BLOCK_BYTES[_Q8_K], 292)
        self.assertEqual(GGML_TYPE_BLOCK_SIZE[_BF16], 1)
        self.assertEqual(GGML_TYPE_BLOCK_BYTES[_BF16], 2)

    def test_quantize_output_lengths(self):
        rows, cols = 4, 512
        rng = np.random.RandomState(_SEED)
        x = rng.randn(rows, cols).astype(np.float32)
        imat = np.ones((rows, cols), dtype=np.float32)
        for qtype, blk, bbytes in ((_IQ2_S, 256, 82), (_IQ1_M, 256, 56),
                                   (_Q1_0, 128, 18), (_Q2_0, 64, 18),
                                   (_Q8_K, 256, 292), (_BF16, 1, 2)):
            n_blocks = rows * (cols // blk)
            self.assertEqual(len(self.hq.quantize_numpy(x, qtype, imatrix=imat)), n_blocks * bbytes)
            self.assertEqual(len(self.hq.quantize_numpy(x, qtype)), n_blocks * bbytes)


@requires_llama
@requires_dll
class TestIQByteExact(unittest.TestCase):
    """Byte-exact comparison of on-device kernels vs llama-quantize.exe."""

    ROWS, COLS = 4, 256

    @classmethod
    def setUpClass(cls):
        cls.hq = get_hip_quant()
        try:
            _probe(cls.hq)
        except Exception as exc:
            raise unittest.SkipTest(f"Q types not available: {exc}")

    def _roundtrip(self, dtype, force_types):
        rng = np.random.RandomState(_SEED)
        tensors = _make_tensors(self.ROWS, self.COLS, rng)
        qtype = GGML_TYPE[dtype]
        with tempfile.TemporaryDirectory(prefix="test_iq_") as tmp:
            src = os.path.join(tmp, "in.gguf")
            out = os.path.join(tmp, "out.gguf")
            im = os.path.join(tmp, "im.n")
            _write_imatrix(im, tensors)
            _write_gguf(src, tensors, self.COLS)
            _run_llama_quantize(src, out, im, dtype, force_types)
            reader = gguf.GGUFReader(out)
            by_name = {t.name: np.array(t.data) for t in reader.tensors}
            del reader
            gc.collect()
            imat = np.ones((self.ROWS, self.COLS), dtype=np.float32)
            for name, arr in tensors.items():
                ref = np.ascontiguousarray(by_name[name]).reshape(-1).view(np.uint8)
                got = np.frombuffer(
                    self.hq.quantize_numpy(arr, qtype, imatrix=imat), dtype=np.uint8
                )
                self.assertEqual(len(got), len(ref), f"{dtype} {name}: size")
                self.assertTrue(np.array_equal(got, ref), f"{dtype} {name}: bytes differ")

    def test_iq2_s_byte_exact(self):
        self._roundtrip("IQ2_S", [f"{n}=iq2_s" for n in ("tok_embeddings.weight", "norm.weight",
                                                         "blk.0.attn_q.weight", "blk.0.attn_k.weight")])

    def test_iq1_m_byte_exact(self):
        self._roundtrip("IQ1_M", None)

    def test_q1_0_byte_exact(self):
        self._roundtrip("Q1_0", None)

    def test_q2_0_byte_exact(self):
        self._roundtrip("Q2_0", None)

    def test_bf16_byte_exact(self):
        self._roundtrip("BF16", None)

    def test_q8_K_vs_ref_replica(self):
        rng = np.random.RandomState(_SEED)
        x = rng.randn(4, 256).astype(np.float32)
        imat = np.ones((4, 256), dtype=np.float32)
        got = np.ascontiguousarray(self.hq.quantize_numpy(x, _Q8_K, imatrix=imat)).view(np.uint8)
        ref = np.frombuffer(_q8_k_ref(x), dtype=np.uint8)
        self.assertEqual(len(got), len(ref))
        self.assertTrue(np.array_equal(got, ref), "Q8_K bytes differ from q8_K_ref replica")

    def test_dequant_to_fp8_smoke(self):
        rng = np.random.RandomState(_SEED)
        x = rng.randn(4, 256).astype(np.float32)
        imat = np.ones((4, 256), dtype=np.float32)
        for qtype in (_Q1_0, _Q2_0, _Q8_K, _BF16):
            packed = self.hq.quantize_numpy(x, qtype, imatrix=imat)
            out = self.hq.dequantize_to_fp8(packed, qtype, 256, "E4M3")
            self.assertEqual(out.shape, (4, 256), f"{qtype}: dequant shape")
            self.assertEqual(out.dtype, np.uint8)

    def test_dequant_iq2s_iq1m_roundtrip(self):
        rng = np.random.RandomState(_SEED)
        x = (rng.randn(1, 256).astype(np.float32) * 0.8)
        imat = np.ones((1, 256), dtype=np.float32)
        golden = {
            _IQ2_S: [165, 160, 19, 147, 19, 19, 37, 32, 147, 147, 19, 19, 147, 160, 160, 165,
                     14, 14, 14, 142, 33, 142, 142, 161, 27, 14, 14, 155, 142, 161, 14, 14,
                     35, 145, 145, 157, 145, 163, 17, 29, 35, 35, 145, 35, 157, 157, 145, 35,
                     13, 141, 13, 141, 154, 141, 26, 160, 13, 13, 154, 141, 13, 32, 154, 141,
                     36, 36, 159, 18, 146, 18, 146, 31, 146, 146, 164, 164, 159, 146, 146, 159,
                     17, 163, 29, 29, 157, 145, 29, 17, 145, 17, 29, 17, 17, 29, 29, 29,
                     147, 19, 147, 147, 161, 147, 161, 19, 161, 147, 166, 161, 19, 147, 33, 33,
                     144, 144, 144, 144, 34, 162, 144, 16, 28, 16, 162, 16, 144, 16, 156, 144,
                     14, 142, 14, 161, 14, 33, 142, 14, 155, 161, 142, 142, 142, 142, 27, 14,
                     149, 149, 149, 21, 21, 21, 21, 149, 21, 149, 162, 149, 40, 162, 21, 149,
                     19, 147, 147, 19, 19, 19, 19, 38, 166, 147, 147, 19, 19, 33, 147, 33,
                     142, 142, 33, 27, 14, 33, 14, 27, 142, 161, 14, 142, 27, 161, 14, 155,
                     19, 19, 161, 147, 33, 38, 147, 147, 147, 161, 147, 161, 19, 147, 147, 19,
                     37, 147, 19, 32, 160, 147, 37, 147, 19, 32, 19, 165, 37, 19, 19, 160,
                     144, 16, 162, 16, 28, 16, 16, 156, 144, 162, 16, 162, 16, 156, 28, 16,
                     144, 28, 28, 144, 28, 28, 16, 16, 34, 16, 16, 28, 144, 16, 28, 144],
            _IQ1_M: [61, 61, 73, 73, 73, 73, 79, 73, 73, 73, 73, 73, 73, 61, 61, 61,
                     69, 69, 69, 69, 75, 57, 69, 57, 75, 69, 69, 57, 69, 57, 69, 69,
                     80, 74, 74, 74, 74, 60, 74, 80, 80, 80, 74, 80, 74, 74, 74, 80,
                     65, 65, 65, 65, 51, 65, 72, 51, 66, 66, 55, 66, 66, 72, 55, 66,
                     77, 77, 59, 72, 59, 72, 72, 77, 71, 71, 57, 57, 71, 71, 71, 57,
                     69, 57, 75, 75, 57, 69, 75, 69, 69, 69, 75, 69, 69, 75, 69, 75,
                     72, 72, 72, 72, 59, 59, 59, 72, 57, 71, 57, 71, 76, 71, 76, 76,
                     71, 71, 71, 71, 76, 57, 71, 71, 72, 72, 59, 72, 72, 72, 59, 72,
                     71, 71, 71, 57, 71, 76, 71, 71, 71, 57, 71, 71, 71, 71, 76, 71,
                     59, 72, 72, 72, 72, 77, 72, 59, 72, 72, 59, 72, 77, 59, 77, 59,
                     72, 59, 72, 72, 72, 72, 77, 77, 57, 71, 71, 76, 76, 76, 71, 76,
                     71, 71, 76, 71, 71, 76, 71, 76, 71, 57, 71, 71, 76, 57, 71, 71,
                     72, 72, 59, 72, 77, 77, 72, 72, 71, 57, 71, 57, 71, 71, 71, 76,
                     79, 61, 73, 73, 61, 73, 79, 61, 73, 79, 73, 61, 79, 73, 73, 73,
                     69, 69, 57, 69, 75, 69, 69, 57, 68, 55, 68, 55, 68, 55, 74, 74,
                     66, 72, 72, 55, 72, 72, 66, 66, 72, 72, 66, 72, 55, 66, 72, 66],
        }
        for qtype in (_IQ2_S, _IQ1_M):
            packed = self.hq.quantize_numpy(x, qtype, imatrix=imat)
            out = self.hq.dequantize_to_fp8(packed, qtype, 256, "E4M3").reshape(-1)
            self.assertEqual(out.shape, (256,), f"{qtype}: dequant shape")
            self.assertEqual(out.dtype, np.uint8)
            self.assertEqual(
                out.tolist(), golden[qtype],
                f"{qtype}: dequant bytes changed vs golden fixture (GPU decode)",
            )


if __name__ == "__main__":
    unittest.main()
