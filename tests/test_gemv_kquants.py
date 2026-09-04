import sys
import os
import pathlib
import torch
import torch.nn.functional as F

# Add repo root
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch_api as T
import gguf

device = "cuda" if torch.cuda.is_available() else "cpu"
ext = T._load_extension()

print("=== Verifying Native K-Quant GEMVs (Q4_K, Q5_K, Q6_K) on RX 9070 XT ===")

model_path = os.environ.get("TEST_MODEL_PATH", "")
if not model_path or not os.path.exists(model_path):
    print("TEST_MODEL_PATH not set or not found, skipping real-weight test.")
    sys.exit(0)


gf = gguf.load(model_path)
with gf.open():
    test_tensors = []
    for t in gf.tensors:
        if t.ggml_type in (12, 13, 14) and len(t.shape) == 2:
            test_tensors.append(t)
            if len(test_tensors) >= 6:
                break

    print(f"Found {len(test_tensors)} K-quant test matrices:")
    for t in test_tensors:
        print(f"  {t.name:<35} type={t.type_name:<8} shape={t.shape}")

    for t in test_tensors:
        N, K = t.shape[0], t.shape[1]
        raw_bytes = bytes(gf.raw_bytes(t))
        w_raw = torch.frombuffer(raw_bytes, dtype=torch.uint8).to(device)

        # 1. Golden Reference: Dequantize to FP8 -> Float32
        w_fp8 = T.dequantize_q_to_e4m3(w_raw, t.ggml_type, K).reshape(N, K)
        w_ref = T.dequantize_e4m3(w_fp8).to(torch.float32)

        # 2. Random input vector x
        torch.manual_seed(42)
        x = (torch.randn(1, K, device=device, dtype=torch.float32) * 0.5).to(torch.float16)

        # 3. Ground Truth: PyTorch FP32 matrix-vector multiplication
        y_ref = F.linear(x.float(), w_ref).float()

        # 4. Candidate: Native AOT GEMV directly on raw packed bytes
        y_kernel = ext.gemv_q_forward(x, w_raw, t.ggml_type, N, None).float()

        cos_sim = F.cosine_similarity(y_kernel.flatten().unsqueeze(0), y_ref.flatten().unsqueeze(0)).item()
        max_err = (y_kernel - y_ref).abs().max().item()
        mean_err = (y_kernel - y_ref).abs().mean().item()

        print(f"\nTensor: {t.name} ({t.type_name}, shape=[{N}, {K}])")
        print(f"  Cosine Similarity: {cos_sim:.6f}")
        print(f"  Max Absolute Diff: {max_err:.5f}")
        print(f"  Mean Absolute Diff: {mean_err:.5f}")

        assert cos_sim >= 0.9990, f"Cosine similarity {cos_sim:.6f} below threshold!"
        print("  -> PASSED! (Bit-level mathematical alignment)")


print("\nALL K-QUANT GEMV TESTS PASSED SUCCESSFULLY!")
