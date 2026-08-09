"""E2E byte-exact comparison of hip_quant IQ2_S / IQ1_M kernels vs llama.cpp CPU.

Writes a minimal GGUF with random float32 tensors, quantizes it with
llama-quantize.exe, then compares the quantized tensor bytes to the output
of hip_quant's DLL on the same data.

Usage:
  python e2e_iq2s_iq1m.py [--rows N] [--cols M] [--types IQ2_S,IQ1_M] [--keep]
"""
import os
import sys
import struct
import subprocess
import tempfile
import argparse
import shutil

os.environ["HIP_QUANT_DLL"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "hip_quantize.dll"
)

import numpy as np
import gguf

_repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_repo_root))

from hip_quant import get_hip_quant, GGML_TYPE, GGML_TYPE_BLOCK_SIZE, GGML_TYPE_BLOCK_BYTES

QUANTIZE_DIR = r"Own Quant\llama_cpp_stock_vulkan_build\bin"
QUANTIZE_EXE = os.path.join(QUANTIZE_DIR, "llama-quantize.exe")


def write_legacy_imatrix(path, tensors):
    """Legacy .n imatrix: one float per COLUMN (shared across rows)."""
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(tensors)))
        for name, arr in tensors.items():
            b = name.encode("utf-8")
            f.write(struct.pack("<i", len(b)))
            f.write(b)
            f.write(struct.pack("<ii", 1, arr.shape[1]))
            f.write(np.ones(arr.shape[1], dtype=np.float32).tobytes())


def quantize_gguf(src_gguf, out_gguf, imatrix_path, qtype_name, rows, cols, force_types=None):
    cmd = [QUANTIZE_EXE]
    if imatrix_path:
        cmd += ["--imatrix", imatrix_path]
    for opt in (force_types or []):
        cmd += ["--tensor-type", opt]
    cmd += [src_gguf, out_gguf, qtype_name, "1"]
    env = dict(os.environ)
    env["PATH"] = os.path.abspath(QUANTIZE_DIR) + os.pathsep + env.get("PATH", "")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"llama-quantize failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--cols", type=int, default=256)
    ap.add_argument("--types", default="IQ2_S,IQ1_M")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    type_names = [t.strip() for t in args.types.split(",")]
    rng = np.random.RandomState(20260708)
    cols, rows = args.cols, args.rows

    tensors = {
        "tok_embeddings.weight": rng.randn(rows, cols).astype(np.float32),
        "norm.weight": (rng.randn(rows, cols).astype(np.float32) * 0.5),
        "blk.0.attn_q.weight": (rng.randn(rows, cols).astype(np.float32) * 3.0 - 0.5),
        "blk.0.attn_k.weight": (rng.randn(rows, cols).astype(np.float32) + 2.0),
    }

    hq = get_hip_quant()
    tmp = tempfile.mkdtemp(prefix="e2e_iq_")
    all_ok = True
    for dtype in type_names:
        qtype = GGML_TYPE[dtype]
        blk = GGML_TYPE_BLOCK_SIZE[qtype]
        blk_bytes = GGML_TYPE_BLOCK_BYTES[qtype]
        if cols % blk != 0:
            print(f"[{dtype}] cols {cols} not multiple of block {blk} - skipping")
            continue

        src_gguf = os.path.join(tmp, f"in_{dtype}.gguf")
        out_gguf = os.path.join(tmp, f"out_{dtype}.gguf")
        imatrix_path = os.path.join(tmp, f"imatrix_{dtype}.n")
        write_legacy_imatrix(imatrix_path, tensors)

        w = gguf.GGUFWriter(src_gguf, "llama")
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

        force = None
        if dtype == "IQ2_S":
            force = [f"{n}=iq2_s" for n in tensors]
        quantize_gguf(src_gguf, out_gguf, imatrix_path, dtype, rows, cols, force_types=force)

        r = gguf.GGUFReader(out_gguf)
        by_name = {t.name: t.data for t in r.tensors}

        rowbytes = (cols // blk) * blk_bytes
        imatrix_gpu = np.ones((rows, cols), dtype=np.float32)
        for name, arr in tensors.items():
            ref = np.ascontiguousarray(by_name[name])
            if ref.size != rows * rowbytes:
                print(f"[{dtype}] {name}: llama bytes {ref.size} != expected {rows * rowbytes}")
                all_ok = False
                continue
            got = hq.quantize_numpy(arr, qtype, imatrix=imatrix_gpu)
            got = np.frombuffer(got, dtype=np.uint8)
            refb = np.ascontiguousarray(ref).reshape(-1).view(np.uint8)
            if np.array_equal(got, refb):
                print(f"[{dtype}] {name}: BYTE-EXACT ({len(got)} bytes)")
            else:
                n = min(len(got), len(refb))
                diff = np.nonzero(got[:n] != refb[:n])[0]
                all_ok = False
                print(f"[{dtype}] {name}: MISMATCH {len(diff)}/{n} bytes, first diffs at {diff[:8]}")

    if not args.keep:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ALL_PASS" if all_ok else "FAILURES")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
