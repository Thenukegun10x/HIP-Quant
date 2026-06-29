import numpy as np
from __init__ import get_hip_quant

def test_tq_quant():
    hq = get_hip_quant()
    print(f"--- Initialization ---")
    print(f"Device Name: {hq.device_name}")
    print(f"Device Count: {hq.device_count}")
    print()

    # Generate test data: 4 blocks of 256 elements
    # Using small floats so they map nicely to the -1, 0, 1 ternary constraints
    np.random.seed(42)
    arr = np.random.randn(4, 256).astype(np.float32)

    print("--- Testing TQ1_0 (Type 34) ---")
    # TQ1_0 size per block is 54 bytes
    out_tq1 = hq.quantize_numpy(arr, 34)
    expected_size_tq1 = 4 * 54
    assert len(out_tq1) == expected_size_tq1, f"Expected {expected_size_tq1} bytes, got {len(out_tq1)}"
    print(f"SUCCESS: TQ1_0 Quantization successful! Compressed 4x256 fp32 block (4096 bytes) into {len(out_tq1)} bytes (1.68 bpw)")

    print("\n--- Testing TQ2_0 (Type 35) ---")
    # TQ2_0 size per block is 66 bytes
    out_tq2 = hq.quantize_numpy(arr, 35)
    expected_size_tq2 = 4 * 66
    assert len(out_tq2) == expected_size_tq2, f"Expected {expected_size_tq2} bytes, got {len(out_tq2)}"
    print(f"SUCCESS: TQ2_0 Quantization successful! Compressed 4x256 fp32 block (4096 bytes) into {len(out_tq2)} bytes (2.06 bpw)")

    print("\nSUCCESS: All HIP Ternary Quantization tests passed successfully!")

if __name__ == "__main__":
    test_tq_quant()
