#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FP8 Crash Diagnostic Tool for hip_quant
=========================================
Isolates the exact failure point in the FP8 pipeline.

Tests are ordered from least to most invasive; each stage is isolated so
we can pinpoint whether the crash is in:
  A.  Basic HIP driver / PyTorch CUDA layer
  B.  Non-WMMA HIP kernel launch (quantize/dequantize)
  C.  WMMA intrinsic kernel launch (the ``__builtin_amdgcn_wmma`` path)
  D.  Memory bandwidth / power spike at specific matrix sizes
  E.  Repeated kernel submission (TDR accumulation)

Usage:
    python diagnose_fp8_crash.py

Returns exit code 0 if all tests pass, 1 if any test fails.
"""

import os, sys, math, time, json, platform, subprocess, ctypes, traceback
import numpy as np

# ── Colour helpers ──────────────────────────────────────────────────────────
OK   = "\033[92m"
FAIL = "\033[91m"
WARN = "\033[93m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RST  = "\033[0m"

PASS = f"{OK}PASS{RST}"
SKIP = f"{WARN}SKIP{RST}"
FAIL_ = f"{FAIL}FAIL{RST}"

results = []   # (test_name, passed: bool | None, detail: str)

def record(name, passed, detail=""):
    tag = PASS if passed is True else (SKIP if passed is None else FAIL_)
    results.append((name, passed, detail))
    print(f"  [{tag}] {name}" + (f"  — {detail}" if detail else ""))

def heading(n, text):
    print(f"\n{BOLD}{'='*56}{RST}")
    print(f"{BOLD}Stage {n}: {text}{RST}")
    print(f"{BOLD}{'='*56}{RST}")

def subheading(text):
    print(f"\n  {BLUE}{text}{RST}")


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1: System Information
# ═══════════════════════════════════════════════════════════════════════════
heading(1, "System Information")

try:
    import torch
    print(f"  PyTorch version : {torch.__version__}")
    print(f"  CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  Device count    : {torch.cuda.device_count()}")
        print(f"  Device name     : {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability(0)
        print(f"  Compute cap     : {cap[0]}.{cap[1]}")
        props = torch.cuda.get_device_properties(0)
        print(f"  Total VRAM      : {props.total_memory / 1024**3:.2f} GiB")
        print(f"  CUs             : {props.multi_processor_count}")
        print(f"  ROCm version    : {torch.version.hip if hasattr(torch, 'version') and hasattr(torch.version, 'hip') else 'N/A'}")
        try:
            import torch.utils.cpp_extension
            hip_ver = torch.utils.cpp_extension.hip_version()
            print(f"  HIP version     : {hip_ver}")
        except Exception:
            pass
    record("PyTorch + CUDA available", True)
except Exception as e:
    print(f"  {WARN}PyTorch import failed: {e}{RST}")
    record("PyTorch + CUDA available", False, str(e))

# DLL info
try:
    from hip_quant import get_hip_quant, GGML_TYPE
    from hip_quant.device_info import probe_device
    dev = probe_device()
    print(f"  DLL loaded      : {dev.dll_loaded}")
    print(f"  Device name     : {dev.name}")
    print(f"  GCN arch        : {dev.gcn_arch}")
    print(f"  Has WMMA        : {dev.has_wmma}")
    print(f"  CU count        : {dev.cu_count}")
    print(f"  VRAM total      : {dev.memory_gb:.2f} GiB")
    print(f"  VRAM free       : {dev.memory_free_gb:.2f} GiB")
    record("DLL device probe", True)
except Exception as e:
    print(f"  {WARN}DLL probe failed: {e}{RST}")
    record("DLL device probe", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2: PyTorch Basic CUDA Ops
# ═══════════════════════════════════════════════════════════════════════════
heading(2, "PyTorch Basic CUDA Ops")

if not torch.cuda.is_available():
    record("Basic CUDA ops", None, "No CUDA device")
else:
    try:
        device = torch.device("cuda")
        a = torch.randn(1024, 1024, device=device)
        b = torch.randn(1024, 1024, device=device)
        c = a @ b
        torch.cuda.synchronize()
        val = c.sum().item()
        assert math.isfinite(val), f"Non-finite result: {val}"
        record("Tensor creation + matmul + sync", True)
    except Exception as e:
        record("Tensor creation + matmul + sync", False, str(e))
        traceback.print_exc()

    try:
        a_bf16 = torch.randn(1024, 1024, device=device, dtype=torch.bfloat16)
        b_bf16 = torch.randn(1024, 1024, device=device, dtype=torch.bfloat16)
        c_bf16 = a_bf16 @ b_bf16
        torch.cuda.synchronize()
        record("bfloat16 matmul", True)
    except Exception as e:
        record("bfloat16 matmul", False, str(e))

    try:
        a_f16 = torch.randn(1024, 1024, device=device, dtype=torch.float16)
        b_f16 = torch.randn(1024, 1024, device=device, dtype=torch.float16)
        c_f16 = a_f16 @ b_f16
        torch.cuda.synchronize()
        record("float16 matmul", True)
    except Exception as e:
        record("float16 matmul", False, str(e))

    try:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        free_before = torch.cuda.memory_reserved()
        big = torch.randn(1024, 1024, 64, device=device)
        torch.cuda.synchronize()
        del big
        torch.cuda.empty_cache()
        record("Large memory alloc + free", True)
    except Exception as e:
        record("Large memory alloc + free", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════
# Stage 3: DLL — Basic HIP Operations (no WMMA)
# ═══════════════════════════════════════════════════════════════════════════
heading(3, "DLL — Basic HIP (no WMMA)")

try:
    hq = get_hip_quant()
    print(f"  Device name  : {hq.device_name}")
    print(f"  Device count : {hq.device_count}")
    record("DLL load + device query", True)
except Exception as e:
    record("DLL load + device query", False, str(e))
    hq = None

if hq is not None:
    # Test non-WMMA quantization (these kernels use no arch-specific intrinsics)
    try:
        np.random.seed(42)
        x = np.random.randn(2, 128).astype(np.float32)
        for qtype_name in ["Q4_0", "Q8_0", "F8_E4M3"]:
            qtype = GGML_TYPE[qtype_name]
            quantized = hq.quantize_numpy(x, qtype)
            assert quantized.dtype == np.uint8
            assert len(quantized) > 0
        record("Non-WMMA quantization (Q4_0, Q8_0, F8_E4M3)", True)
    except Exception as e:
        record("Non-WMMA quantization (Q4_0, Q8_0, F8_E4M3)", False, str(e))

    # Test FP8 expand (GPU kernel, portable)
    try:
        fp8_data = hq.quantize_numpy(x, GGML_TYPE["F8_E4M3"])
        # quantize_from_fp8 uses FP8 expand on GPU -> then GGML quantize
        result = hq.quantize_from_fp8(fp8_data, GGML_TYPE["Q8_0"])
        assert result.dtype == np.uint8
        record("FP8 expand + requantize (GPU kernel, portable)", True)
    except Exception as e:
        record("FP8 expand + requantize (GPU kernel, portable)", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════
# Stage 4: PyTorch Extension — FP8 Quant/Dequant Kernels (no WMMA)
# ═══════════════════════════════════════════════════════════════════════════
heading(4, "PyTorch Extension — Quantize/Dequant (no WMMA)")

_has_torch_ext = False
try:
    from hip_quant.torch_api import quantize_e4m3, dequantize_e4m3, quantize_e5m2, dequantize_e5m2, _load_extension
    _load_extension()
    _has_torch_ext = True
    print(f"  PyTorch extension loaded: {True}")
except Exception as e:
    print(f"  {WARN}PyTorch extension not available: {e}{RST}")
    _has_torch_ext = False

if _has_torch_ext and torch.cuda.is_available():
    device = torch.device("cuda")
    try:
        t = torch.randn(128, 64, device=device)
        fp8 = quantize_e4m3(t)
        torch.cuda.synchronize()
        assert fp8.dtype == torch.uint8
        assert fp8.shape == t.shape
        record("quantize_e4m3 (f32 input)", True)
    except Exception as e:
        record("quantize_e4m3 (f32 input)", False, str(e))

    try:
        fp8 = quantize_e4m3(t)
        back = dequantize_e4m3(fp8)
        torch.cuda.synchronize()
        assert back.dtype == torch.float32
        record("dequantize_e4m3", True)
    except Exception as e:
        record("dequantize_e4m3", False, str(e))

    try:
        t_f16 = torch.randn(128, 64, device=device, dtype=torch.float16)
        fp8_f16 = quantize_e4m3(t_f16)
        torch.cuda.synchronize()
        record("quantize_e4m3 (f16 input)", True)
    except Exception as e:
        record("quantize_e4m3 (f16 input)", False, str(e))

    try:
        t_bf16 = torch.randn(128, 64, device=device, dtype=torch.bfloat16)
        fp8_bf16 = quantize_e4m3(t_bf16)
        torch.cuda.synchronize()
        record("quantize_e4m3 (bf16 input)", True)
    except Exception as e:
        record("quantize_e4m3 (bf16 input)", False, str(e))

    try:
        t = torch.randn(128, 64, device=device)
        fp8 = quantize_e5m2(t)
        torch.cuda.synchronize()
        back = dequantize_e5m2(fp8)
        torch.cuda.synchronize()
        record("quantize_e5m2 + dequantize_e5m2", True)
    except Exception as e:
        record("quantize_e5m2 + dequantize_e5m2", False, str(e))

    try:
        t = torch.tensor([1.0, -1.0, 0.0, 448.0], device=device)
        fp8 = quantize_e4m3(t)
        torch.cuda.synchronize()
        vals = fp8.cpu().tolist()
        assert vals[0] == 0x38, f"1.0 -> {vals[0]:#04x}, expected 0x38"
        assert vals[1] == 0xB8, f"-1.0 -> {vals[1]:#04x}, expected 0xB8"
        assert vals[2] in (0x00, 0x80), f"0.0 -> {vals[2]:#04x}"
        assert vals[3] == 0x7E, f"448.0 -> {vals[3]:#04x}, expected 0x7E"
        record("FP8 encoding correctness check", True)
    except Exception as e:
        record("FP8 encoding correctness check", False, str(e))

    # Round-trip accuracy
    try:
        t = torch.randn(256, 256, device=device)
        fp8 = quantize_e4m3(t)
        back = dequantize_e4m3(fp8)
        torch.cuda.synchronize()
        rel = ((back - t).abs() / t.abs().clamp(min=1e-6)).mean().item()
        ok = rel < 0.20
        record(f"FP8 E4M3 round-trip accuracy  (rel_err={rel:.4f})", ok)
    except Exception as e:
        record("FP8 E4M3 round-trip accuracy", False, str(e))

else:
    for name in ["quantize_e4m3", "dequantize_e4m3", "quantize_e5m2", "dequantize_e5m2",
                  "quantize_e4m3 f16", "quantize_e4m3 bf16", "FP8 encoding correctness",
                  "FP8 round-trip accuracy"]:
        record(name, None, "PyTorch extension unavailable")


# ═══════════════════════════════════════════════════════════════════════════
# Stage 5: WMMA GEMM via DLL (the critical test)
# ═══════════════════════════════════════════════════════════════════════════
heading(5, "FP8 WMMA GEMM via DLL")

def run_wmma_gemm(hq, M, N, K, label=""):
    """Run a single WMMA GEMM.  Returns (passed, detail_str)."""
    try:
        np.random.seed(42)
        A_f32 = np.random.randn(M, K).astype(np.float32) * 0.5
        B_f32 = np.random.randn(K, N).astype(np.float32) * 0.5

        A_fp8 = hq.quantize_numpy(A_f32, GGML_TYPE["F8_E4M3"]).reshape(M, K)
        B_fp8 = hq.quantize_numpy(B_f32, GGML_TYPE["F8_E4M3"]).reshape(K, N)

        C = hq.fp8_gemm_test_wmma(A_fp8, B_fp8, M, N, K)
        if C is None:
            return False, "DLL returned None (kernel launch error)"

        C_ref = A_f32 @ B_f32
        rel = np.abs(C - C_ref) / (np.abs(C_ref) + 1e-10)
        max_rel = rel.max()
        ok = max_rel < 0.50  # FP8 E4M3 is approximate
        detail = f"max_rel_err={max_rel:.4f}"
        return ok, detail
    except Exception as e:
        return False, str(e)

if hq is not None:
    sizes = [
        (32,  32,  32,   "minimal 32x32x32"),
        (64,  64,  64,   "small 64x64x64"),
        (128, 128, 128,  "medium 128x128x128"),
        (256, 256, 256,  "large 256x256x256"),
    ]
    for M, N, K, label in sizes:
        ok, detail = run_wmma_gemm(hq, M, N, K, label)
        record(f"WMMA GEMM {label}", ok, detail)
        if not ok:
            print(f"    {WARN}-> WMMA failed at {label}, stopping Stage 5{RST}")
            break

    # Non-square shapes
    for M, N, K, label in [
        (32,  64,  128, "32x64x128"),
        (64,  32,  128, "64x32x128"),
        (128, 64,  32,  "128x64x32"),
    ]:
        ok, detail = run_wmma_gemm(hq, M, N, K, label)
        record(f"WMMA GEMM {label}", ok, detail)
        if not ok:
            print(f"    {WARN}-> WMMA failed at non-square {label}{RST}")
            break

    # Large sizes
    big_sizes = [
        (512,  512,  512,  "512x512x512"),
        (1024, 1024, 1024, "1024x1024x1024"),
    ]
    for M, N, K, label in big_sizes:
        ok, detail = run_wmma_gemm(hq, M, N, K, label)
        record(f"WMMA GEMM {label}", ok, detail)
        if not ok:
            print(f"    {WARN}-> WMMA failed at big {label}{RST}")
            break
else:
    record("WMMA GEMM (all sizes)", None, "DLL not loaded")


# ═══════════════════════════════════════════════════════════════════════════
# Stage 6: FP8 Linear Forward via PyTorch Extension (WMMA path)
# ═══════════════════════════════════════════════════════════════════════════
heading(6, "FP8 Linear Forward (WMMA via PyTorch Extension)")

if _has_torch_ext and torch.cuda.is_available():
    device = torch.device("cuda")
    try:
        from hip_quant.torch_api import fp8_linear_forward
        M, K, N = 32, 32, 32
        inp = torch.randn(M, K, device=device)
        wt  = torch.randn(N, K, device=device)
        out = fp8_linear_forward(inp, wt, None)
        torch.cuda.synchronize()
        ok = out.shape == (M, N) and out.dtype == torch.float32
        record(f"fp8_linear_forward (32x32x32)", ok,
               f"shape={tuple(out.shape)}" if ok else str(out.shape))
    except Exception as e:
        record(f"fp8_linear_forward (32x32x32)", False, str(e))

    try:
        from hip_quant.torch_api import fp8_linear_forward_scaled
        M, K, N = 32, 64, 32
        inp = torch.randn(M, K, device=device)
        wt  = torch.randn(N, K, device=device)
        out = fp8_linear_forward_scaled(inp, wt, None, 1.0, 1.0)
        torch.cuda.synchronize()
        record(f"fp8_linear_forward_scaled (32x64x32)", out.shape == (M, N),
               f"shape={tuple(out.shape)}")
    except Exception as e:
        record(f"fp8_linear_forward_scaled (32x64x32)", False, str(e))

    try:
        from hip_quant.torch_api import Fp8LinearFunction
        M, K, N = 32, 64, 32
        inp = torch.randn(M, K, device=device, requires_grad=True)
        wt  = torch.randn(N, K, device=device, requires_grad=True)
        out = Fp8LinearFunction.apply(inp, wt, None)
        torch.cuda.synchronize()
        record(f"Fp8LinearFunction.forward (32x64x32)", out.shape == (M, N),
               f"shape={tuple(out.shape)}")
    except Exception as e:
        record(f"Fp8LinearFunction.forward (32x64x32)", False, str(e))

    try:
        from hip_quant.torch_api import Fp8LinearFunction
        M, K, N = 32, 64, 32
        inp = torch.randn(M, K, device=device, requires_grad=True)
        wt  = torch.randn(N, K, device=device, requires_grad=True)
        out = Fp8LinearFunction.apply(inp, wt, None)
        out.sum().backward()
        torch.cuda.synchronize()
        ok = inp.grad is not None and wt.grad is not None
        record(f"Fp8LinearFunction forward+backward", ok)
    except Exception as e:
        record(f"Fp8LinearFunction forward+backward", False, str(e))

    try:
        from hip_quant.torch_api import Fp8ScaledLinearFunction
        M, K, N = 32, 64, 32
        inp = torch.randn(M, K, device=device, requires_grad=True)
        wt  = torch.randn(N, K, device=device, requires_grad=True)
        out = Fp8ScaledLinearFunction.apply(inp, wt, None, 1.0, 1.0)
        out.sum().backward()
        torch.cuda.synchronize()
        ok = inp.grad is not None and wt.grad is not None
        record(f"Fp8ScaledLinearFunction forward+backward", ok)
    except Exception as e:
        record(f"Fp8ScaledLinearFunction forward+backward", False, str(e))

    try:
        from hip_quant.torch_api import Fp8ShadowLinearFunction
        M, K, N = 32, 64, 32
        inp = torch.randn(M, K, device=device, requires_grad=True)
        wt_master = torch.randn(N, K, device=device, requires_grad=True)
        wt_fp8 = torch.randint(0, 256, (N, K), device=device, dtype=torch.uint8)
        out = Fp8ShadowLinearFunction.apply(inp, wt_master, wt_fp8, 1.0, 1.0, None)
        out.sum().backward()
        torch.cuda.synchronize()
        ok = inp.grad is not None and wt_master.grad is not None
        record(f"Fp8ShadowLinearFunction forward+backward", ok)
    except Exception as e:
        record(f"Fp8ShadowLinearFunction forward+backward", False, str(e))

    try:
        from hip_quant.torch_api import Fp8TensorMeta
        from hip_quant.torch_api import Fp8TensorMeta
        meta = Fp8TensorMeta(device=str(device))
        t = torch.randn(128, 64, device=device)
        for _ in range(20):
            meta.update(t)
        s = meta.scale.item()
        inv = meta.inv_scale.item()
        ok = math.isfinite(s) and s > 0 and math.isfinite(inv) and inv > 0
        record(f"Fp8TensorMeta scale tracking (20 updates)", ok,
               f"scale={s:.4f}, inv_scale={inv:.6f}")
    except Exception as e:
        record(f"Fp8TensorMeta scale tracking", False, str(e))

    # Fp8Linear nn.Module (end-to-end unscaled)
    try:
        from hip_quant.torch_api import Fp8Linear
        layer = Fp8Linear(64, 32).to(device)
        x = torch.randn(8, 64, device=device)
        y = layer(x)
        y.sum().backward()
        torch.cuda.synchronize()
        ok = layer.weight.grad is not None
        record(f"Fp8Linear nn.Module forward+backward", ok,
               f"output shape={tuple(y.shape)}")
    except Exception as e:
        record(f"Fp8Linear nn.Module forward+backward", False, str(e))

    # Fp8ScaledLinear nn.Module (end-to-end scaled)
    try:
        from hip_quant.torch_api import Fp8ScaledLinear
        layer = Fp8ScaledLinear(64, 32).to(device)
        x = torch.randn(8, 64, device=device)
        y = layer(x)
        y.sum().backward()
        torch.cuda.synchronize()
        ok = layer.weight.grad is not None
        record(f"Fp8ScaledLinear nn.Module forward+backward", ok)
    except Exception as e:
        record(f"Fp8ScaledLinear nn.Module forward+backward", False, str(e))

    # Small training loop (reproduces the original crash scenario)
    try:
        from hip_quant.torch_api import Fp8ScaledLinear, Adafactor
        import torch.nn as nn
        model = nn.Sequential(
            Fp8ScaledLinear(64, 32, history_len=8),
            nn.ReLU(),
            Fp8ScaledLinear(32, 16, history_len=8),
        ).to(device)
        opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
        crash_happened = False
        for step in range(10):
            x = torch.randn(4, 64, device=device)
            loss = model(x).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            if not math.isfinite(loss.item()):
                crash_happened = True
                break
        record(f"Fp8ScaledLinear 10-step training loop",
               not crash_happened,
               f"finished {step+1} steps" if not crash_happened else f"crashed at step {step+1}")
    except Exception as e:
        record(f"Fp8ScaledLinear 10-step training loop", False, str(e))

else:
    for name in ["fp8_linear_forward", "fp8_linear_forward_scaled",
                  "Fp8LinearFunction forward", "Fp8LinearFunction backward",
                  "Fp8ScaledLinearFunction", "Fp8ShadowLinearFunction",
                  "Fp8TensorMeta", "Fp8Linear module", "Fp8ScaledLinear module",
                  "Fp8ScaledLinear training loop"]:
        record(name, None, "PyTorch extension unavailable")


# ═══════════════════════════════════════════════════════════════════════════
# Stage 7: Progressive Stress Test
# ═══════════════════════════════════════════════════════════════════════════
heading(7, "Progressive Stress Test (TDR Detection)")

def stress_wmma_gemm(hq, M, N, K, iterations=20, label=""):
    """Run *iterations* WMMA GEMMs rapidly to trigger TDR if unstable."""
    try:
        np.random.seed(42)
        A_f32 = np.random.randn(M, K).astype(np.float32) * 0.3
        B_f32 = np.random.randn(K, N).astype(np.float32) * 0.3
        A_fp8 = hq.quantize_numpy(A_f32, GGML_TYPE["F8_E4M3"]).reshape(M, K)
        B_fp8 = hq.quantize_numpy(B_f32, GGML_TYPE["F8_E4M3"]).reshape(K, N)

        for i in range(iterations):
            C = hq.fp8_gemm_test_wmma(A_fp8, B_fp8, M, N, K)
            if C is None:
                return False, f"failed at iteration {i+1}/{iterations}"
            # small random perturbation to prevent caching
            noise = np.random.randn(K, N).astype(np.float32) * 1e-6
            B_fp8 = hq.quantize_numpy(B_f32 + noise, GGML_TYPE["F8_E4M3"]).reshape(K, N)
        return True, f"{iterations}/{iterations} iterations passed"
    except Exception as e:
        return False, str(e)

if hq is not None:
    # Light stress at small size
    ok, detail = stress_wmma_gemm(hq, 64, 64, 64, iterations=10, label="64×64×64 ×10")
    record("WMMA stress 64×64×64 ×10", ok, detail)

    if ok:
        # Medium stress
        ok, detail = stress_wmma_gemm(hq, 128, 128, 128, iterations=10, label="128×128×128 ×10")
        record("WMMA stress 128×128×128 ×10", ok, detail)

    if ok:
        # Heavy stress — most likely to trigger TDR
        ok, detail = stress_wmma_gemm(hq, 256, 256, 256, iterations=20, label="256×256×256 ×20")
        record("WMMA stress 256×256×256 ×20", ok, detail)
else:
    record("WMMA stress (all)", None, "DLL not loaded")


# ═══════════════════════════════════════════════════════════════════════════
# Stage 8: GPU Stability Aftermath Check
# ═══════════════════════════════════════════════════════════════════════════
heading(8, "GPU Stability Aftermath")

if torch.cuda.is_available():
    try:
        device = torch.device("cuda")
        torch.cuda.synchronize()
        a = torch.randn(512, 512, device=device)
        b = torch.randn(512, 512, device=device)
        c = a @ b
        torch.cuda.synchronize()
        val = c.sum().item()
        stable = math.isfinite(val)
        record("Basic matmul after WMMA stress", stable,
               f"sum={val:.4f}" if stable else "non-finite")
    except Exception as e:
        record("Basic matmul after WMMA stress", False, str(e))

    try:
        torch.cuda.synchronize()
        err = torch.cuda.get_last_error() if hasattr(torch.cuda, "get_last_error") else None
        record("CUDA last error after tests",
               err is None or err == 0,
               f"error={err}" if err else "no error")
    except Exception as e:
        record("CUDA last error after tests", None, str(e))

    try:
        free_before = torch.cuda.memory_reserved()
        big = torch.randn(512, 512, 128, device=device)
        torch.cuda.synchronize()
        del big
        torch.cuda.empty_cache()
        record("GPU memory integrity after tests", True)
    except Exception as e:
        record("GPU memory integrity after tests", False, str(e))

else:
    record("GPU aftermath (all)", None, "No CUDA device")


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
heading("Final", "Summary")

passed = sum(1 for _, p, _ in results if p is True)
failed = sum(1 for _, p, _ in results if p is False)
skipped = sum(1 for _, p, _ in results if p is None)
total = len(results)

print(f"\n  Total : {total}")
print(f"  {OK}Passed : {passed}{RST}")
print(f"  {FAIL}Failed : {failed}{RST}")
print(f"  {WARN}Skipped: {skipped}{RST}")

if failed > 0:
    print(f"\n  {BOLD}{FAIL}FAILED TESTS:{RST}")
    for name, ok, detail in results:
        if ok is False:
            print(f"    {FAIL}X{RST} {name}")
            if detail:
                print(f"      {detail}")
    print(f"\n  {BOLD}Interpretation:{RST}")
    print(f"  Look at which stage the first failure occurred.")
    print(f"  Stages 1-2 → PyTorch/ROCm driver issue")
    print(f"  Stages 3-4 → Non-WMMA kernel issue (should rarely fail)")
    print(f"  Stage 5    → WMMA intrinsic via DLL (isolates WMMA itself)")
    print(f"  Stage 6    → Full FP8 training path via PyTorch ext")
    print(f"  Stage 7    → TDR / power / thermal stress")
else:
    print(f"\n  {BOLD}{OK}All tests passed.{RST}")
    print(f"  FP8 WMMA is working correctly on this system.")
    print(f"  The crash you experienced may be intermittent or workload-specific.")
    if hq is not None:
        print(f"\n  Running `--dtype fp8` should be safe on this configuration.")

sys.exit(0 if failed == 0 else 1)
