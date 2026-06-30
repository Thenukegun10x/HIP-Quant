// torch_ext/fp8_quant_kernels.hip
//
// Element-wise FP8 quantize / dequantize kernels for the PyTorch extension.
// These kernels operate directly on device memory; no CPU transfers needed.
// They reuse the device helpers already defined in hip_quant_util.h.
//
// Kernel launch convention (matches plan §Phase 1):
//   blocks  = (numel + 255) / 256
//   threads = 256
//   Each thread processes four contiguous elements.

#include <hip/hip_runtime.h>
#include <assert.h>
#include "../hip_quant_util.h"

#define HIP_QUANT_DTYPE_F32  0
#define HIP_QUANT_DTYPE_F16  1
#define HIP_QUANT_DTYPE_BF16 2
#define HIP_QUANT_ELEMS_PER_THREAD 4

template <int DTYPE>
static __device__ inline float load_float_input_t(const void* src, int64_t idx) {
    if constexpr (DTYPE == HIP_QUANT_DTYPE_F16) {
        return fp16_to_fp32(((const uint16_t*)src)[idx]);
    }
    if constexpr (DTYPE == HIP_QUANT_DTYPE_BF16) {
        return bf16_to_fp32(((const uint16_t*)src)[idx]);
    }
    return ((const float*)src)[idx];
}

template <int DTYPE>
static __device__ inline uint8_t quant_input_e4m3_t(const void* src, int64_t idx) {
    if constexpr (DTYPE == HIP_QUANT_DTYPE_BF16) {
        return bf16_to_fp8_e4m3(((const uint16_t*)src)[idx]);
    }
    return fp32_to_fp8_e4m3(load_float_input_t<DTYPE>(src, idx));
}

template <int DTYPE>
static __device__ inline uint8_t quant_input_e5m2_t(const void* src, int64_t idx) {
    if constexpr (DTYPE == HIP_QUANT_DTYPE_BF16) {
        return bf16_to_fp8_e5m2(((const uint16_t*)src)[idx]);
    }
    return fp32_to_fp8_e5m2(load_float_input_t<DTYPE>(src, idx));
}

// ---------------------------------------------------------------------------
// Quantize: float32 → FP8 E4M3
// ---------------------------------------------------------------------------
template <int DTYPE>
__global__
__launch_bounds__(256, 4)
void quant_e4m3_kernel_t(
    const void  * __restrict__ src,
    uint8_t      * __restrict__ dst,
    int64_t       numel
) {
    int64_t idx = ((int64_t)blockIdx.x * 256 + threadIdx.x) * HIP_QUANT_ELEMS_PER_THREAD;
    #pragma unroll
    for (int i = 0; i < HIP_QUANT_ELEMS_PER_THREAD; i++) {
        int64_t cur = idx + i;
        if (cur < numel) {
            dst[cur] = quant_input_e4m3_t<DTYPE>(src, cur);
        }
    }
}

// ---------------------------------------------------------------------------
// Quantize: float32 → FP8 E5M2
// ---------------------------------------------------------------------------
template <int DTYPE>
__global__
__launch_bounds__(256, 4)
void quant_e5m2_kernel_t(
    const void  * __restrict__ src,
    uint8_t      * __restrict__ dst,
    int64_t       numel
) {
    int64_t idx = ((int64_t)blockIdx.x * 256 + threadIdx.x) * HIP_QUANT_ELEMS_PER_THREAD;
    #pragma unroll
    for (int i = 0; i < HIP_QUANT_ELEMS_PER_THREAD; i++) {
        int64_t cur = idx + i;
        if (cur < numel) {
            dst[cur] = quant_input_e5m2_t<DTYPE>(src, cur);
        }
    }
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
    int64_t idx = ((int64_t)blockIdx.x * 256 + threadIdx.x) * HIP_QUANT_ELEMS_PER_THREAD;
    #pragma unroll
    for (int i = 0; i < HIP_QUANT_ELEMS_PER_THREAD; i++) {
        int64_t cur = idx + i;
        if (cur < numel) {
            dst[cur] = fp8_e4m3_to_fp32(src[cur]);
        }
    }
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
    int64_t idx = ((int64_t)blockIdx.x * 256 + threadIdx.x) * HIP_QUANT_ELEMS_PER_THREAD;
    #pragma unroll
    for (int i = 0; i < HIP_QUANT_ELEMS_PER_THREAD; i++) {
        int64_t cur = idx + i;
        if (cur < numel) {
            dst[cur] = fp8_e5m2_to_fp32(src[cur]);
        }
    }
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
    int64_t b = (numel + 256 * HIP_QUANT_ELEMS_PER_THREAD - 1) / (256 * HIP_QUANT_ELEMS_PER_THREAD);
    // Sanity check: block count must fit in uint32 for dim3.
    // 2^32-1 blocks * 256 threads = ~1.1 trillion elements (>> any GPU VRAM).
    assert(b <= (int64_t)0xFFFFFFFFu);
    return (uint32_t)b;
}

extern "C" {

void launch_quant_e4m3(const void* src, uint8_t* dst, int64_t numel,
                       int dtype, hipStream_t stream) {
    if (numel == 0) return;  // Hazard A: launching dim3(0) is UB in HIP
    if (dtype == HIP_QUANT_DTYPE_F16) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(quant_e4m3_kernel_t<HIP_QUANT_DTYPE_F16>),
                           dim3(safe_blocks(numel)), dim3(256), 0, stream,
                           src, dst, numel);
    } else if (dtype == HIP_QUANT_DTYPE_BF16) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(quant_e4m3_kernel_t<HIP_QUANT_DTYPE_BF16>),
                           dim3(safe_blocks(numel)), dim3(256), 0, stream,
                           src, dst, numel);
    } else {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(quant_e4m3_kernel_t<HIP_QUANT_DTYPE_F32>),
                           dim3(safe_blocks(numel)), dim3(256), 0, stream,
                           src, dst, numel);
    }
}

void launch_quant_e5m2(const void* src, uint8_t* dst, int64_t numel,
                       int dtype, hipStream_t stream) {
    if (numel == 0) return;
    if (dtype == HIP_QUANT_DTYPE_F16) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(quant_e5m2_kernel_t<HIP_QUANT_DTYPE_F16>),
                           dim3(safe_blocks(numel)), dim3(256), 0, stream,
                           src, dst, numel);
    } else if (dtype == HIP_QUANT_DTYPE_BF16) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(quant_e5m2_kernel_t<HIP_QUANT_DTYPE_BF16>),
                           dim3(safe_blocks(numel)), dim3(256), 0, stream,
                           src, dst, numel);
    } else {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(quant_e5m2_kernel_t<HIP_QUANT_DTYPE_F32>),
                           dim3(safe_blocks(numel)), dim3(256), 0, stream,
                           src, dst, numel);
    }
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
