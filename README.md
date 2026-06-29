# hip-quant

HIP/ROCm-based tensor quantization library for AMD GPUs.

`hip-quant` provides blazing fast on-device quantization of float32 numpy arrays into GGML/llama.cpp compatible formats. It is specifically optimized for ROCm 7.1 and the `gfx1201` architecture.

## Supported Quantization Formats

- **Legacy/Standard:** `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`
- **K-Quants:** `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`
- **I-Quants (Importance Matrix):** `IQ1_S`, `IQ2_XXS`, `IQ2_XS`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL`, `IQ4_XS`
- **Ternary Quants:** `TQ1_0` (1.69 bpw), `TQ2_0` (2.06 bpw)

*Note: Ternary quants (`TQ1_0` and `TQ2_0`) map weights to strictly 3 states (-1, 0, 1) and pack them in Base-3 memory layouts. They are incredibly memory and compute efficient but are intended exclusively for models trained to be ternary (e.g., BitNet, TriLM).*
