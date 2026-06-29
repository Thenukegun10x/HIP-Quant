// torch_ext/fp8_quant_kernels.hip
//
// Element-wise FP8 quantize / dequantize kernels for the PyTorch extension.
// These kernels operate directly on device memory; no CPU transfers needed.
// They reuse the device helpers already defined in hip_quant_util.h.
//
// Kernel launch convention (matches plan §Phase 1):
//   blocks  = (numel + 255) / 256
//   threads = 256
//   Each thread processes exactly one element.

#include <hip/hip_runtime.h>
#include "../hip_quant_util.h"

// ---------------------------------------------------------------------------
// Quantize: float32 → FP8 E4M3
// ---------------------------------------------------------------------------
__global__
__launch_bounds__(256, 4)
void quant_e4m3_kernel(
    const float * __restrict__ src,
    uint8_t      * __restrict__ dst,
    int64_t       numel
) {
    int64_t idx = (int64_t)blockIdx.x * 256 + threadIdx.x;
    if (idx >= numel) return;
    dst[idx] = fp32_to_fp8_e4m3(src[idx]);
}

// ---------------------------------------------------------------------------
// Quantize: float32 → FP8 E5M2
// ---------------------------------------------------------------------------
__global__
__launch_bounds__(256, 4)
void quant_e5m2_kernel(
    const float * __restrict__ src,
    uint8_t      * __restrict__ dst,
    int64_t       numel
) {
    int64_t idx = (int64_t)blockIdx.x * 256 + threadIdx.x;
    if (idx >= numel) return;
    dst[idx] = fp32_to_fp8_e5m2(src[idx]);
}

// ---------------------------------------------------------------------------
// Dequantize: FP8 E4M3 → float32
// ---------------------------------------------------------------------------
__global__
__launch_bounds__(256, 4)
void dequant_e4m3_kernel(
    const uint8_t * __restrict__ src,
    float         * __restrict__ dst,
    int64_t        numel
) {
    int64_t idx = (int64_t)blockIdx.x * 256 + threadIdx.x;
    if (idx >= numel) return;
    dst[idx] = fp8_e4m3_to_fp32(src[idx]);
}

// ---------------------------------------------------------------------------
// Dequantize: FP8 E5M2 → float32
// ---------------------------------------------------------------------------
__global__
__launch_bounds__(256, 4)
void dequant_e5m2_kernel(
    const uint8_t * __restrict__ src,
    float         * __restrict__ dst,
    int64_t        numel
) {
    int64_t idx = (int64_t)blockIdx.x * 256 + threadIdx.x;
    if (idx >= numel) return;
    dst[idx] = fp8_e5m2_to_fp32(src[idx]);
}

// ---------------------------------------------------------------------------
// C-linkage wrappers called from pytorch_bindings.cpp
// These are NOT __global__; they are host-callable launch helpers.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Internal helper: compute grid dimension, guarding against UB edge cases.
// Asserts numel > 0 (callers skip launch when numel == 0).
// Asserts the resulting block count fits in uint32 (limits tensor to ~274 GB
// at float32, which is far beyond any real GPU VRAM today).
// ---------------------------------------------------------------------------
static inline uint32_t safe_blocks(int64_t numel) {
    // numel must be positive; caller must handle the zero case.
    assert(numel > 0);
    int64_t b = (numel + 255) / 256;
    // Sanity check: block count must fit in uint32 for dim3.
    // 2^32-1 blocks * 256 threads = ~1.1 trillion elements (>> any GPU VRAM).
    assert(b <= (int64_t)0xFFFFFFFFu);
    return (uint32_t)b;
}

extern "C" {

void launch_quant_e4m3(const float* src, uint8_t* dst, int64_t numel,
                       hipStream_t stream) {
    if (numel == 0) return;  // Hazard A: launching dim3(0) is UB in HIP
    hipLaunchKernelGGL(quant_e4m3_kernel,
                       dim3(safe_blocks(numel)), dim3(256), 0, stream,
                       src, dst, numel);
}

void launch_quant_e5m2(const float* src, uint8_t* dst, int64_t numel,
                       hipStream_t stream) {
    if (numel == 0) return;
    hipLaunchKernelGGL(quant_e5m2_kernel,
                       dim3(safe_blocks(numel)), dim3(256), 0, stream,
                       src, dst, numel);
}

void launch_dequant_e4m3(const uint8_t* src, float* dst, int64_t numel,
                         hipStream_t stream) {
    if (numel == 0) return;
    hipLaunchKernelGGL(dequant_e4m3_kernel,
                       dim3(safe_blocks(numel)), dim3(256), 0, stream,
                       src, dst, numel);
}

void launch_dequant_e5m2(const uint8_t* src, float* dst, int64_t numel,
                         hipStream_t stream) {
    if (numel == 0) return;
    hipLaunchKernelGGL(dequant_e5m2_kernel,
                       dim3(safe_blocks(numel)), dim3(256), 0, stream,
                       src, dst, numel);
}

} // extern "C"
