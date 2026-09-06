import os
import sys
import time
import torch

from hip_inference.models import load_model

def main():
    model_path = r"C:\Users\armor\Downloads\Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf"
    runner = load_model(model_path, device="cuda")
    rec = runner.layers[0]

    from hip_inference.core.linear import dispatch_linear, _GEMM_Q_TYPES, _GEMM_PREFILL_ON
    from hip_inference.core.norm import rms_norm
    from hip_inference.models.qwen35 import swiglu

    print(f"_GEMM_PREFILL_ON: {_GEMM_PREFILL_ON}")
    print(f"19 in _GEMM_Q_TYPES: {19 in _GEMM_Q_TYPES}")

    gate_w = rec["ffn_gate_w"]
    up_w = rec["ffn_up_w"]
    down_w = rec["ffn_down_w"]
    print(f"gate ggml_type: {getattr(gate_w, 'ggml_type', None)}")
    print(f"up ggml_type: {getattr(up_w, 'ggml_type', None)}")
    print(f"down ggml_type: {getattr(down_w, 'ggml_type', None)}")

    S = 512
    h = torch.randn(1, S, runner.hidden, dtype=torch.float16, device="cuda")
    torch.cuda.synchronize()

    # Warmup
    _ = dispatch_linear(rms_norm(h, rec["post_attn_norm_w"], runner.eps), gate_w)
    torch.cuda.synchronize()

    for it in range(3):
        t0 = time.time()
        x = rms_norm(h, rec["post_attn_norm_w"], runner.eps)
        torch.cuda.synchronize()
        t_norm = (time.time() - t0) * 1000

        t0 = time.time()
        g = dispatch_linear(x, gate_w)
        torch.cuda.synchronize()
        t_gate = (time.time() - t0) * 1000

        t0 = time.time()
        u = dispatch_linear(x, up_w)
        torch.cuda.synchronize()
        t_up = (time.time() - t0) * 1000

        t0 = time.time()
        mid = swiglu(g, u, out=g)
        torch.cuda.synchronize()
        t_swiglu = (time.time() - t0) * 1000

        t0 = time.time()
        d = dispatch_linear(mid, down_w)
        torch.cuda.synchronize()
        t_down = (time.time() - t0) * 1000

        print(f"Iter {it} S={S} FFN layer 0 breakdown:")
        print(f"  norm:   {t_norm:.2f} ms")
        print(f"  gate:   {t_gate:.2f} ms")
        print(f"  up:     {t_up:.2f} ms")
        print(f"  swiglu: {t_swiglu:.2f} ms")
        print(f"  down:   {t_down:.2f} ms")
        print(f"  total:  {t_norm + t_gate + t_up + t_swiglu + t_down:.2f} ms")

if __name__ == "__main__":
    try:
        main()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        os._exit(0)
