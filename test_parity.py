import numpy as np
from __init__ import get_hip_quant

# Python port of llama.cpp's CPU reference: quantize_row_tq1_0_ref
def quantize_tq1_0_cpu_ref(x):
    assert len(x) == 256
    amax = float(np.max(np.abs(x)))
    d = amax
    id_val = 1.0 / d if d != 0.0 else 0.0
    
    qs = np.zeros(48, dtype=np.uint8)
    qh = np.zeros(4, dtype=np.uint8)
    
    # 5 elements per byte, along 32 bytes
    for m in range(32):
        q = 0
        for n in range(5):
            val = x[m + n*32] * id_val
            # python round() matches C lroundf (ties to even)
            xi = int(round(val)) + 1
            q = q * 3 + xi
        qs[m] = (q * 256 + 242) // 243
        
    # along 16 bytes
    for m in range(16):
        q = 0
        for n in range(5):
            val = x[160 + m + n*16] * id_val
            xi = int(round(val)) + 1
            q = q * 3 + xi
        qs[32 + m] = (q * 256 + 242) // 243
        
    # 4 elements per byte
    for j in range(4):
        q = 0
        for m in range(4):
            val = x[240 + j + m*4] * id_val
            xi = int(round(val)) + 1
            q = q * 3 + xi
        q *= 3
        qh[j] = (q * 256 + 242) // 243
        
    d_fp16 = np.array([d], dtype=np.float16)
    return qs.tobytes() + qh.tobytes() + d_fp16.tobytes()


# Python port of llama.cpp's CPU reference: quantize_row_tq2_0_ref
def quantize_tq2_0_cpu_ref(x):
    assert len(x) == 256
    amax = float(np.max(np.abs(x)))
    d = amax
    id_val = 1.0 / d if d != 0.0 else 0.0
    
    qs = np.zeros(64, dtype=np.uint8)
    
    # Block 1
    for m in range(32):
        q = 0
        for n in range(4):
            val = x[m + n*32] * id_val
            xi = int(round(val)) + 1
            q += (xi & 3) << (2*n)
        qs[m] = q
        
    # Block 2
    for m in range(32):
        q = 0
        for n in range(4):
            val = x[128 + m + n*32] * id_val
            xi = int(round(val)) + 1
            q += (xi & 3) << (2*n)
        qs[32 + m] = q
        
    d_fp16 = np.array([d], dtype=np.float16)
    return qs.tobytes() + d_fp16.tobytes()

def test_bit_parity():
    hq = get_hip_quant()
    
    # Generate some random floats that are representative of real weights/activations
    np.random.seed(1337)
    arr = np.random.randn(256).astype(np.float32)
    
    print("Testing TQ1_0 Parity...")
    gpu_tq1 = bytes(hq.quantize_numpy(arr[None, :], 34))
    cpu_tq1 = quantize_tq1_0_cpu_ref(arr)
    
    if gpu_tq1 == cpu_tq1:
        print("TQ1_0: 100% BIT-FOR-BIT IDENTICAL!")
    else:
        print("TQ1_0: MISMATCH!")
        for i in range(len(gpu_tq1)):
            if gpu_tq1[i] != cpu_tq1[i]:
                print(f"Byte {i} diff: GPU={gpu_tq1[i]}, CPU={cpu_tq1[i]}")

    print("\nTesting TQ2_0 Parity...")
    gpu_tq2 = bytes(hq.quantize_numpy(arr[None, :], 35))
    cpu_tq2 = quantize_tq2_0_cpu_ref(arr)
    
    if gpu_tq2 == cpu_tq2:
        print("TQ2_0: 100% BIT-FOR-BIT IDENTICAL!")
    else:
        print("TQ2_0: MISMATCH!")
        for i in range(len(gpu_tq2)):
            if gpu_tq2[i] != cpu_tq2[i]:
                print(f"Byte {i} diff: GPU={gpu_tq2[i]}, CPU={cpu_tq2[i]}")

if __name__ == "__main__":
    test_bit_parity()
