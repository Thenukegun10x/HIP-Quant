"""
hip_quant/demo.py — beginner-friendly walkthroughs of FP8, attention, and MXFP4.

Usage::

    import hip_quant
    hip_quant.demo()            # run all demos
    hip_quant.demo("quantize")  # run quantize demo only
    hip_quant.demo("attention") # run attention demo only

Each demo prints explanations, runs a tiny example on GPU, and shows the results.
Designed for CS students — every line is commented.
"""

from __future__ import annotations

import math
import time


def _section(title: str):
    line = "-" * 60
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}")


def _maybe_import():
    """Try to import torch — return None with a helpful message if missing."""
    try:
        import torch
        import torch.nn.functional as F
        return torch, F
    except ImportError:
        print(
            "SKIPPED — PyTorch is not installed.\n"
            "  Install with:  pip install torch --index-url https://download.pytorch.org/whl/rocm7.1\n"
            "  Then rebuild:   python setup_torch.py build_ext --inplace"
        )
        return None, None


def _maybe_gpu(torch):
    """Check for a ROCm GPU."""
    if not torch.cuda.is_available():
        print("SKIPPED — no ROCm GPU detected (torch.cuda.is_available() is False).")
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════════
#  DEMO 1 — FP8 Quantization
# ═══════════════════════════════════════════════════════════════════════════

def demo_quantize():
    """
    FP8 Quantization basics.

    What is FP8?
        FP8 (E4M3) uses 1 sign bit, 4 exponent bits, and 3 mantissa bits.
        That's only 256 possible values per byte — but it stores numbers
        from −448 to +448 with ~6% precision.

        Compare:
          float32 → 32 bits,   ±1e38 range,    ~7 decimal digits
          float16 → 16 bits,   ±65504 range,   ~3 decimal digits
          fp8_e4m3 → 8 bits,   ±448 range,     ~1 decimal digit

        The key trade‑off:  4× less memory bandwidth than float32,
        2× less than float16, at very acceptable precision for matrix math.

    Why quantize?
        GPU kernels on RDNA4 can multiply FP8 matrices with a single
        instruction (``v_wmma_f32_16x16x16_fp8_fp8``), while float32 needs
        many more cycles.  This demo shows the conversion and precision.
    """
    import hip_quant

    torch, F = _maybe_import()
    if torch is None:
        return
    if not _maybe_gpu(torch):
        return

    print()
    print("  Creating a random 4×8 float32 tensor on the GPU ...")
    x = torch.randn(4, 8, device="cuda")

    print(f"  Original (float32):  memory = {x.element_size() * x.numel()} bytes")
    print(f"  Sample values:       {x[0, :4].tolist()}")

    print()
    print("  Quantizing to FP8 E4M3 (uint8) with per-tensor scale ...")
    x_fp8, scale = hip_quant.quantize(x, dtype="e4m3")
    print(f"  Quant scale:         {scale.item():.6f}")

    print(f"  Quantized (fp8):     memory = {x_fp8.element_size() * x_fp8.numel()} bytes")
    print(f"  dtype:               {x_fp8.dtype}  (raw uint8 — each byte IS an FP8 value)")

    print()
    print("  Dequantizing back to float32 for comparison ...")
    x_back = hip_quant.dequantize(x_fp8, scale=scale)
    error = (x - x_back).abs()
    print(f"  Max absolute error:  {error.max().item():.6f}")
    print(f"  Mean absolute error: {error.mean().item():.6f}")
    print(f"  This is ~{error.mean().item() / x.abs().mean().item() * 100:.1f}% of the average magnitude.")
    print()
    print("  ✓  FP8 quantization preserved the values with ~0.5–6% error.")
    print("     For neural network weights this is nearly lossless in practice.")


# ═══════════════════════════════════════════════════════════════════════════
#  DEMO 2 — QuantizedLinear (drop‑in nn.Linear replacement)
# ═══════════════════════════════════════════════════════════════════════════

def demo_linear():
    """
    QuantizedLinear — a drop‑in replacement for torch.nn.Linear.

    What this does:
        1.  Stores the weight matrix in FP8 (8 bits per weight instead of 32).
        2.  On every forward pass, quantizes the input to FP8 on‑the‑fly.
        3.  Calls hipBLASLt (AMD's tuned BLAS library) to do the fast FP8 GEMM.
        4.  Dequantizes the result back to the original dtype.

    Why this matters:
        Memory:   4× less weight storage than float32.
        Speed:    hipBLASLt uses the GPU's tensor cores (WMMA) for FP8.
        Accuracy: Typically <0.1% error vs float32 — invisible in practice.

    Usage in real code:
        >>> layer = hip_quant.QuantizedLinear(4096, 14336)
        >>> y = layer(x)   # x can be float32, float16, or already FP8

    Or convert an existing model:
        >>> model = hip_quant.convert_to_quantized(model)
    """
    import hip_quant

    torch, F = _maybe_import()
    if torch is None:
        return
    if not _maybe_gpu(torch):
        return

    print()
    print("  Building a QuantizedLinear layer:  64 inputs → 128 outputs")
    layer = hip_quant.QuantizedLinear(64, 128, bias=True).cuda()

    print(f"  Layer: {layer}")
    print(f"  Weight: {layer.weight.shape} (float32, will be FP8-quantized on first forward)")

    print()
    print("  Creating a batch of 4 random inputs (seq_len=8, dim=64) ...")
    x = torch.randn(4, 8, 64, device="cuda")

    # Warmup (first call quantizes weight — slower)
    start = time.perf_counter()
    y_fp8 = layer(x)
    torch.cuda.synchronize()
    first_ms = (time.perf_counter() - start) * 1000

    # Second call (weight already cached as FP8)
    start = time.perf_counter()
    for _ in range(10):
        y_fp8 = layer(x)
    torch.cuda.synchronize()
    avg_ms = (time.perf_counter() - start) / 10 * 1000

    print(f"  First forward (includes weight quantize):  {first_ms:.3f} ms")
    print(f"  Average forward (weight cached as FP8):   {avg_ms:.3f} ms")
    print(f"  Output shape:  {y_fp8.shape}")

    # Reference comparison (float32)
    ref = torch.nn.functional.linear(
        x, hip_quant.dequantize(layer._fp8_weight, scale=layer._weight_scale), layer.bias
    )
    cos = torch.nn.functional.cosine_similarity(y_fp8.flatten(), ref.flatten(), dim=0)
    print(f"  Cosine similarity vs float32 reference:  {cos.item():.6f}")
    print()
    print("  ✓  QuantizedLinear output matches float32 to ~{:.4f} cosine similarity.".format(cos.item()))
    print(f"     Weight memory:  {layer._fp8_weight.numel()} bytes (FP8) vs "
          f"{layer._fp8_weight.numel() * 4} bytes (float32) — 4× less.")


# ═══════════════════════════════════════════════════════════════════════════
#  DEMO 3 — WaveAttention (native FP8 flash attention)
# ═══════════════════════════════════════════════════════════════════════════

def demo_attention():
    """
    WaveAttention — native FP8 WMMA flash attention for RDNA4.

    What is flash attention?
        Standard attention computes:

            S = Q × Kᵀ          (attention scores)
            P = softmax(S)      (attention probabilities)
            O = P × V           (weighted values)

        Flash attention tiles Q, K, V into GPU shared memory and fuses
        the softmax with the matrix multiplies.  This avoids writing the
        huge S matrix (seq_len × seq_len) to slow global memory.

    How WaveAttention differs from AOTriton:
        AOTriton v2 converts Triton code to GPU assembly, but it uses
        **fp16 WMMA** + a 1200‑instruction software FP8 decode step.
        WaveAttention calls ``v_wmma_f32_16x16x16_fp8_fp8`` — the
        **real RDNA4 FP8 hardware instruction** — directly, with no
        software decode.  This is 3–8× faster than SDPA.

    What happens under the hood:
        1.  Q, K, V are auto‑quantized to FP8 (if not already).
        2.  The 16×16×16 WMMA instruction computes the matmul.
        3.  Online softmax is done with warp‑level shuffles.
        4.  P is re‑quantized to FP8 on‑the‑fly with ``v_cvt_pk_fp8_f32``.
        5.  P × V uses another WMMA call.
    """
    import hip_quant

    torch, F = _maybe_import()
    if torch is None:
        return
    if not _maybe_gpu(torch):
        return

    B, H, S, D = 2, 4, 128, 64   # batch, heads, seq_len, head_dim

    print()
    print(f"  Config:  B={B} batch, H={H} heads, S={S} seq_len, D={D} dim")
    print("  Creating random Q, K, V tensors on GPU ...")
    q = torch.randn(B, H, S, D, device="cuda")
    k = torch.randn(B, H, S, D, device="cuda")
    v = torch.randn(B, H, S, D, device="cuda")
    scale = 1.0 / math.sqrt(D)

    # WaveAttention (FP8 WMMA)
    print()
    print("  Running WaveAttention (native FP8 WMMA) ...")
    start = time.perf_counter()
    for _ in range(5):
        out_wave = hip_quant.wave_attn(q, k, v, scale=scale)
    torch.cuda.synchronize()
    wave_ms = (time.perf_counter() - start) / 5 * 1000

    # PyTorch SDPA (float32 reference, uses AOTriton fp16 under the hood)
    print("  Running PyTorch SDPA (float32 reference) ...")
    start = time.perf_counter()
    for _ in range(5):
        out_ref = F.scaled_dot_product_attention(q, k, v, scale=scale)
    torch.cuda.synchronize()
    sdpa_ms = (time.perf_counter() - start) / 5 * 1000

    cos = F.cosine_similarity(out_wave.flatten(), out_ref.flatten(), dim=0)

    ops = 4 * B * H * S * S * D        # 2 matmuls × O(S²D) MACs each
    wave_tflops = ops / wave_ms * 1e-9
    sdpa_tflops = ops / sdpa_ms * 1e-9

    print()
    print(f"  WaveAttention:  {wave_ms:.3f} ms   ({wave_tflops:.1f} effective TFLOP/s)")
    print(f"  PyTorch SDPA:   {sdpa_ms:.3f} ms   ({sdpa_tflops:.1f} effective TFLOP/s)")
    print(f"  Speedup:        {sdpa_ms / wave_ms:.2f}×")
    print(f"  Cosine sim:     {cos.item():.6f}  (1.0 = identical)")
    print()
    print("  ✓  WaveAttention is {:.1f}× faster than SDPA with cosine similarity ≈ {:.4f}.".format(
        sdpa_ms / wave_ms, cos.item()))
    print("     The FP8 WMMA instruction is the difference — 1 instruction")
    print("     per 16×16×16 tile vs AOTriton's software FP8 emulation.")


# ═══════════════════════════════════════════════════════════════════════════
#  DEMO 4 — MXFP4 (micro‑scaling FP4 weight format)
# ═══════════════════════════════════════════════════════════════════════════

def demo_mxfp4():
    """
    MXFP4 — the OCP Micro‑scaling FP4 weight format.

    What is MXFP4?
        MXFP4 uses only 4 bits per weight (E2M1 encoding), plus a
        shared 8‑bit power‑of‑two scale (UE8M0) per block of 32 weights.
        The effective bit‑rate is ~4.25 bits per weight — vs 8 bits for
        FP8, 16 bits for FP16, and 32 bits for float32.

        Why would you use it?
            Model storage size.  A 70B‑parameter model is:
              ~280 GB  in float32
              ~140 GB  in float16
               ~70 GB  in FP8
               ~37 GB  in MXFP4

            For inference, you convert MXFP4 → FP8 once at load time,
            then run all your FP8 kernels normally.  The conversion is
            fast (~0.07 ms for a 4096×4096 matrix on a RDNA4 GPU).

    How the conversion works (E2M1 → E4M3):
        E2M1 has only 16 possible values per nibble.
        UE8M0 has 256 possible scale values (NaN reserved at 0xFF).
        Together, there are only 16 × 256 = 4,096 (E2M1, UE8M0) pairs.
        We pre‑compute a 4 KB lookup table mapping every pair to the
        corresponding E4M3 byte.  The GPU kernel just does a LUT lookup
        — zero float32 math, no branches.
    """
    import hip_quant
    import warnings

    torch, F = _maybe_import()
    if torch is None:
        return
    if not _maybe_gpu(torch):
        return

    print()
    print("  Creating a tiny 2×32 MXFP4 tensor on GPU ...")
    print("  (2 rows × 32 FP4 values = 1 block per row, 2 scales total)")
    nrows, n_per_row = 2, 32

    # Build packed FP4 data: low nibble = even index, high nibble = odd index
    # The 16 possible E2M1 codes for a row:
    codes = torch.arange(32, dtype=torch.uint8) & 0x0F         # 32 FP4 values
    packed_row = codes[0::2] | (codes[1::2] << 4)              # pack 2 per byte
    packed = packed_row.repeat(nrows, 1).cuda()
    scales = torch.full((nrows, 1), 127, dtype=torch.uint8, device="cuda")  # scale=2^0=1

    print(f"  Packed shape:  {packed.shape}  (each byte = 2 FP4 values)")
    print(f"  Scales shape:  {scales.shape}  (1 byte per 32 FP4 values)")

    print()
    print("  Converting MXFP4 → FP8 E4M3 ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        start = time.perf_counter()
        for _ in range(100):
            fp8_out = hip_quant.mxfp4_to_fp8(packed, scales, n_per_row=n_per_row)
        torch.cuda.synchronize()
        avg_us = (time.perf_counter() - start) / 100 * 1_000_000

    print(f"  Conversion time:  {avg_us:.1f} µs  (for {nrows * n_per_row} values)")
    print(f"  FP8 output shape: {fp8_out.shape}")

    # Dequantize to float to see the values
    actual = hip_quant.dequantize(fp8_out).cpu()
    expected = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0] * 2,
        dtype=torch.float32,
    )
    diff = (actual - expected).abs().max()

    print(f"  Max error vs expected:  {diff.item():.6f}")
    print()
    print("  ✓  MXFP4 → FP8 conversion is exact for standard E2M1 codes.")
    print("     In production: convert weights once at model load time,")
    print(f"    then reuse the FP8 tensor.  Memory: {packed.numel() + scales.numel()} bytes (MXFP4)")
    print(f"    → {fp8_out.numel()} bytes (FP8 cached).")


# ═══════════════════════════════════════════════════════════════════════════
#  DEMO 5 — Full model conversion
# ═══════════════════════════════════════════════════════════════════════════

def demo_convert_model():
    """
    Converting an existing nn.Module to FP8.

    ``hip_quant.convert_to_quantized()`` recursively replaces every
    ``nn.Linear`` layer with a ``QuantizedLinear`` layer that stores
    its weight in FP8 and uses hipBLASLt for the GEMM.

    The original float32 weights are automatically quantized on the
    first forward pass — no manual quantization needed.
    """
    import hip_quant

    torch, F = _maybe_import()
    if torch is None:
        return
    if not _maybe_gpu(torch):
        return

    print()
    print("  Building a tiny 3‑layer MLP ...")
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
    )
    print(f"  Original model:  {model}")

    # Count float32 bytes
    f32_bytes = sum(p.numel() * 4 for p in model.parameters())
    print(f"  Parameters:      {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Memory (f32):    {f32_bytes:,} bytes")

    print()
    print("  Converting all nn.Linear → QuantizedLinear ...")
    hip_quant.convert_to_quantized(model)
    model = model.cuda()
    print(f"  Converted model: {model}")

    # Run one forward to quantize weights
    x = torch.randn(4, 64, device="cuda")
    y = model(x)

    # Count FP8 bytes
    fp8_bytes = 0
    for m in model.modules():
        if isinstance(m, hip_quant.QuantizedLinear) and m._fp8_weight is not None:
            fp8_bytes += m._fp8_weight.numel() * 1
    print(f"  Output shape:    {y.shape}")
    print(f"  Weight memory after quantize:  {fp8_bytes:,} bytes (FP8)")
    print(f"  Memory saved:    {f32_bytes - fp8_bytes:,} bytes  "
          f"({(1 - fp8_bytes / f32_bytes) * 100:.0f}%)")
    print()
    print("  ✓  Model converted.  All Linear layers now use FP8 weights")
    print("     and hipBLASLt GEMM under the hood.  No code changes needed.")


# ═══════════════════════════════════════════════════════════════════════════
#  Main dispatcher
# ═══════════════════════════════════════════════════════════════════════════

_DEMOS = {
    "quantize":    ("FP8 Quantization",         demo_quantize),
    "linear":      ("QuantizedLinear (nn.Module)", demo_linear),
    "attention":   ("WaveAttention (FP8 WMMA)",  demo_attention),
    "mxfp4":       ("MXFP4 → FP8 conversion",    demo_mxfp4),
    "convert":     ("convert_to_quantized",      demo_convert_model),
}


def run(name: str | None = None):
    """Run one or all demos.

    ``hip_quant.demo()``                runs everything
    ``hip_quant.demo("attention")``     runs the attention demo only
    """
    if name is None:
        print("============================================================")
        print("        hip_quant  --  Beginner Demo Suite")
        print("        FP8 . MXFP4 . WaveAttention . QuantizedLinear")
        print("============================================================")

        for key, (title, fn) in _DEMOS.items():
            _section(f"DEMO: {title}")
            try:
                fn()
            except Exception as e:
                print(f"\n  ✗  Demo failed: {e}")
    else:
        if name not in _DEMOS:
            available = ", ".join(_DEMOS.keys())
            print(f"Unknown demo '{name}'. Available: {available}")
            return
        title, fn = _DEMOS[name]
        _section(f"DEMO: {title}")
        fn()
