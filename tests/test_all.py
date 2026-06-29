import ctypes
import numpy as np
import sys, os

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_repo_root, "src"))

from hip_quant import HipQuant, GGML_TYPE, GGML_TYPE_BLOCK_SIZE, GGML_TYPE_BLOCK_BYTES

# Load CPU quantize_wrapper DLL
_cpu_hints = [
    os.path.join(_repo_root, "src", "quantize_wrapper.dll"),
    os.path.join(os.path.dirname(__file__), "..", "quantize_wrapper.dll"),
]
_cpu_dll_path = None
for p in _cpu_hints:
    if os.path.exists(p):
        _cpu_dll_path = p
        break
if _cpu_dll_path is None:
    print("ERROR: Cannot find quantize_wrapper.dll")
    sys.exit(1)

_llama_bin = r"C:\Users\armor\Desktop\llamma.cpp server\llama.cpp\build\bin"
for d in [os.path.join(_repo_root, "src"), _llama_bin, os.path.dirname(os.path.abspath(__file__))]:
    if os.path.isdir(d):
        os.add_dll_directory(d)

cpu_dll = ctypes.CDLL(_cpu_dll_path)
cpu_dll.quantize_tensor.argtypes = [
    ctypes.c_int, ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_float),
]
cpu_dll.quantize_tensor.restype = ctypes.c_size_t
cpu_dll.ggml_type_size_for.restype = ctypes.c_size_t
cpu_dll.ggml_type_size_for.argtypes = [ctypes.c_int]
cpu_dll.ggml_blck_size_for.restype = ctypes.c_size_t
cpu_dll.ggml_blck_size_for.argtypes = [ctypes.c_int]

# Types that ggml_quantize_chunk does NOT support (HIP-only)
HIP_ONLY_TYPES = {9}  # Q8_1 is internal-only in ggml

# Type info: (name, type_num)
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
    ("IQ2_XXS", 16),
    ("IQ2_XS", 17),
    ("IQ3_XXS", 18),
    ("IQ1_S", 19),
    ("IQ4_NL", 20),
    ("IQ3_S", 21),
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
im = np.abs(np.sin(np.linspace(0, 4*np.pi, n_per_row))).astype(np.float32) + 0.1
im = np.tile(im.reshape(1, -1), (nrows, 1)).astype(np.float32)
im_flat = im.ravel().astype(np.float32)
im_ptr = im_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

hip = HipQuant()

all_pass = True
for name, type_num in type_info:
    print(f"\n{'='*60}")
    print(f"Testing HIP {name} (type {type_num})")
    print(f"{'='*60}")

    blk_bytes = GGML_TYPE_BLOCK_BYTES[type_num]
    blk_size = GGML_TYPE_BLOCK_SIZE[type_num]

    if type_num in HIP_ONLY_TYPES:
        # HIP-only type: verify HIP runs correctly, no CPU comparison
        hip_out = hip.quantize_numpy(src_flat, type_num)
        expected = int(cpu_dll.ggml_row_size_for(type_num, n_per_row)) * nrows
        if len(hip_out) == expected:
            print(f"  HIP-only type: {len(hip_out)} bytes [OK]")
            print(f"  RESULT: PASS")
        else:
            print(f"  ERROR: HIP returned {len(hip_out)} bytes, expected {expected}")
            all_pass = False
        continue

    total_size = int(cpu_dll.ggml_row_size_for(type_num, n_per_row)) * nrows
    n_blocks = total_size // blk_bytes

    cpu_out = (ctypes.c_uint8 * total_size)()
    cpu_imatrix_ptr = im_ptr if type_num in {16, 17, 19} else imatrix_ptr
    cpu_dll.quantize_tensor(type_num, src_ptr, cpu_out, nrows, n_per_row, cpu_imatrix_ptr)
    cpu_np = np.frombuffer(cpu_out, dtype=np.uint8).copy()

    hip_arr = src if type_num in {16, 17, 19} else src_flat
    hip_out = hip.quantize_numpy(hip_arr, type_num, imatrix=im if type_num in {16, 17, 19} else None)

    if len(hip_out) != total_size:
        print(f"  ERROR: HIP returned {len(hip_out)} bytes, expected {total_size}")
        all_pass = False
        continue

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
                # For IQ4_XS: print per-component breakdown
                if type_num == 23:
                    print(f"    CPU d(16bit): {cpu_blocks[b,0]:02x}{cpu_blocks[b,1]:02x}")
                    print(f"    HIP d(16bit): {hip_blocks[b,0]:02x}{hip_blocks[b,1]:02x}")
                    print(f"    CPU scales_h: {cpu_blocks[b,2]:02x}{cpu_blocks[b,3]:02x}")
                    print(f"    HIP scales_h: {hip_blocks[b,2]:02x}{hip_blocks[b,3]:02x}")
                    print(f"    CPU scales_l[0..3]: {' '.join(f'{cpu_blocks[b,4+i]:02x}' for i in range(4))}")
                    print(f"    HIP scales_l[0..3]: {' '.join(f'{hip_blocks[b,4+i]:02x}' for i in range(4))}")
                    print(f"    CPU qs[0..15]:     {' '.join(f'{cpu_blocks[b,8+i]:02x}' for i in range(16))}")
                    print(f"    HIP qs[0..15]:     {' '.join(f'{hip_blocks[b,8+i]:02x}' for i in range(16))}")
                break

    thresholds = {10: 40, 11: 10, 14: 10}
    max_ok = thresholds.get(type_num, 10)
    if n_diff == 0:
        print(f"  RESULT: PASS (byte-exact)")
    elif pct <= max_ok:
        print(f"  RESULT: PASS (within fp tolerance, {pct:.1f}% <= {max_ok}%)")
    else:
        print(f"  RESULT: FAIL ({pct:.1f}% > {max_ok}%)")
        all_pass = False

    if type_num == 20:
        hip_out_iq4nl_noim = hip.quantize_numpy(src, type_num)
    if type_num == 23:
        hip_out_iq4xs_noim = hip.quantize_numpy(src, type_num)

print(f"\n{'='*60}")
if all_pass:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED")

# Imatrix smoke test for IQ4_NL
print(f"\n{'='*60}")
print("Testing IQ4_NL with imatrix (HIP only)")
hip_out_im = hip.quantize_numpy(src, 20, imatrix=im)
n_diff_im = int(np.sum(hip_out_im != hip_out_iq4nl_noim))
print(f"  {len(hip_out_im)} bytes, differs from no-imatrix: {n_diff_im}/{len(hip_out_im)} bytes")
if n_diff_im > 0:
    print(f"  RESULT: PASS (imatrix changes output as expected)")
else:
    print(f"  RESULT: FAIL (imatrix had no effect)")
    all_pass = False

sys.exit(0 if all_pass else 1)
