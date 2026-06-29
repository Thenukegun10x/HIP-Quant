import ctypes
import numpy as np
import sys, os

# Add src/ to path
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_repo_root, "src"))

from hip_quant import HipQuant, GGML_TYPE, GGML_TYPE_BLOCK_SIZE, GGML_TYPE_BLOCK_BYTES

# Optional: load CPU reference DLL for byte-exact comparison
_cpu_dll = None
_cpu_hints = [
    os.path.join(_repo_root, "src", "quantize_wrapper.dll"),
    os.path.join(_repo_root, "..", "quantize_wrapper.dll"),
]
for p in _cpu_hints:
    p = os.path.normpath(p)
    if os.path.exists(p):
        _cpu_dll_path = p
        _cpu_dll = ctypes.CDLL(p)
        _cpu_dll.quantize_tensor.argtypes = [
            ctypes.c_int, ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_float),
        ]
        _cpu_dll.quantize_tensor.restype = ctypes.c_size_t
        _cpu_dll.ggml_type_size_for.restype = ctypes.c_size_t
        _cpu_dll.ggml_type_size_for.argtypes = [ctypes.c_int]
        _cpu_dll.ggml_blck_size_for.restype = ctypes.c_size_t
        _cpu_dll.ggml_blck_size_for.argtypes = [ctypes.c_int]
        print(f"INFO: CPU reference loaded from {_cpu_dll_path}")
        break

if _cpu_dll is None:
    print("INFO: No CPU reference DLL found — checking HIP-only (no byte-exact comparison).")
    print("  Build quantize_wrapper.dll from llama.cpp/ggml for CPU comparison.")

# Types that ggml_quantize_chunk does NOT support (HIP-only)
HIP_ONLY_TYPES = {9}  # Q8_1 is internal-only in ggml

type_info = [
    ("Q4_0",   2),
    ("Q4_1",   3),
    ("Q5_0",   6),
    ("Q5_1",   7),
    ("Q8_0",   8),
    ("Q8_1",   9),
    ("Q2_K",  10),
    ("Q3_K",  11),
    ("Q4_K",  12),
    ("Q5_K",  13),
    ("Q6_K",  14),
    ("IQ3_XXS", 18),
    ("IQ4_NL", 20),
    ("IQ4_XS", 23),
]

rng = np.random.RandomState(42)
n_per_row = 1024
nrows = 10
src = np.zeros((nrows, n_per_row), dtype=np.float32)
for row in range(nrows):
    base = np.sin(np.linspace(0, 4 * np.pi, n_per_row)) * 2.0
    noise = rng.randn(n_per_row) * 0.3
    outliers = np.zeros(n_per_row)
    outlier_idx = rng.randint(0, n_per_row, size=5)
    outliers[outlier_idx] = rng.choice([-10, 10, -8, 8], size=5)
    src[row] = base + noise + outliers

src_flat = src.ravel().astype(np.float32)
src_ptr = src_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
imatrix_ptr = ctypes.POINTER(ctypes.c_float)()

hip = HipQuant()
print(f"Device: {hip.device_name}\n")

all_pass = True
for name, type_num in type_info:
    print(f"{'='*60}")
    print(f"Testing HIP {name} (type {type_num})")
    print(f"{'='*60}")

    blk_bytes = GGML_TYPE_BLOCK_BYTES[type_num]
    blk_size = GGML_TYPE_BLOCK_SIZE[type_num]

    if type_num in HIP_ONLY_TYPES:
        hip_out = hip.quantize_numpy(src_flat, type_num)
        expected = nrows * int(GGML_TYPE_BLOCK_BYTES[type_num]) * (n_per_row // int(GGML_TYPE_BLOCK_SIZE[type_num]))
        if len(hip_out) == expected:
            print(f"  HIP-only type: {len(hip_out)} bytes [OK]")
            print(f"  RESULT: PASS")
        else:
            print(f"  ERROR: HIP returned {len(hip_out)} bytes, expected {expected}")
            all_pass = False
        continue

    total_size = nrows * blk_bytes * (n_per_row // blk_size)
    n_blocks = total_size // blk_bytes

    if _cpu_dll is not None:
        cpu_out = (ctypes.c_uint8 * total_size)()
        _cpu_dll.quantize_tensor(type_num, src_ptr, cpu_out, nrows, n_per_row, imatrix_ptr)
        cpu_np = np.frombuffer(cpu_out, dtype=np.uint8).copy()
    else:
        cpu_np = None

    hip_out = hip.quantize_numpy(src_flat, type_num)

    if len(hip_out) != total_size:
        print(f"  ERROR: HIP returned {len(hip_out)} bytes, expected {total_size}")
        all_pass = False
        continue

    if cpu_np is not None:
        n_diff = int(np.sum(cpu_np != hip_out))
        pct = 100.0 * n_diff / len(cpu_np)
        print(f"  Blocks: {n_blocks}  Bytes diff: {n_diff}/{len(cpu_np)} ({pct:.2f}%)")

        if n_diff > 0:
            cpu_blocks = cpu_np.reshape(-1, blk_bytes)
            hip_blocks = hip_out.reshape(-1, blk_bytes)
            for b in range(n_blocks):
                if not np.array_equal(cpu_blocks[b], hip_blocks[b]):
                    n_diff_b = int(np.sum(cpu_blocks[b] != hip_blocks[b]))
                    print(f"  First diff block #{b}: {n_diff_b}/{blk_bytes} bytes")
                    show = min(blk_bytes, 136)
                    print(f"    CPU: {' '.join(f'{cpu_blocks[b,i]:02x}' for i in range(show))}")
                    print(f"    HIP: {' '.join(f'{hip_blocks[b,i]:02x}' for i in range(show))}")
                    break

        thresholds = {10: 40, 11: 10, 14: 10, 18: 10}
        max_ok = thresholds.get(type_num, 10)
        if n_diff == 0:
            print(f"  RESULT: PASS (byte-exact)")
        elif pct <= max_ok:
            print(f"  RESULT: PASS (within fp tolerance, {pct:.1f}% <= {max_ok}%)")
        else:
            print(f"  RESULT: FAIL ({pct:.1f}% > {max_ok}%)")
            all_pass = False
    else:
        print(f"  HIP output: {len(hip_out)} bytes across {n_blocks} blocks [OK]")
        print(f"  RESULT: PASS (HIP-only smoke test)")

print(f"\n{'='*60}")
if all_pass:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED")

sys.exit(0 if all_pass else 1)
