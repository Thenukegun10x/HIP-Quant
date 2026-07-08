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

template <int DTYPE>
static __device__ inline uint8_t quant_input_e5m2_stochastic_t(const void* src, int64_t idx, uint64_t seed) {
    return fp32_to_fp8_e5m2_stochastic(load_float_input_t<DTYPE>(src, idx), seed, (uint64_t)idx);
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
// Quantize: float32 -> FP8 E5M2 with stochastic rounding
// ---------------------------------------------------------------------------
template <int DTYPE>
__global__
__launch_bounds__(256, 4)
void quant_e5m2_stochastic_kernel_t(
    const void  * __restrict__ src,
    uint8_t      * __restrict__ dst,
    int64_t       numel,
    uint64_t      seed
) {
    int64_t idx = ((int64_t)blockIdx.x * 256 + threadIdx.x) * HIP_QUANT_ELEMS_PER_THREAD;
    #pragma unroll
    for (int i = 0; i < HIP_QUANT_ELEMS_PER_THREAD; i++) {
        int64_t cur = idx + i;
        if (cur < numel) {
            dst[cur] = quant_input_e5m2_stochastic_t<DTYPE>(src, cur, seed);
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
// Block-wise quantize/dequantize over the last logical tensor dimension.
// One HIP block handles one scale block. Scales are dequant scales:
//   real ~= fp8_value * scale
// ---------------------------------------------------------------------------
template <int DTYPE, bool E5M2>
__global__
__launch_bounds__(256, 4)
void quant_blockwise_kernel_t(
    const void  * __restrict__ src,
    uint8_t      * __restrict__ dst,
    float        * __restrict__ scales,
    int64_t       rows,
    int64_t       cols,
    int64_t       blocks_per_row,
    int           block_size
) {
    int64_t block_id = (int64_t)blockIdx.x;
    int64_t total_blocks = rows * blocks_per_row;
    if (block_id >= total_blocks) return;

    int64_t row = block_id / blocks_per_row;
    int64_t block_col = block_id - row * blocks_per_row;
    int64_t col_start = block_col * (int64_t)block_size;
    int64_t len = cols - col_start;
    if (len > block_size) len = block_size;
    if (len <= 0) return;

    __shared__ float shared_max[256];
    float local_max = 0.0f;
    int64_t base = row * cols + col_start;

    for (int64_t i = threadIdx.x; i < len; i += blockDim.x) {
        float v = load_float_input_t<DTYPE>(src, base + i);
        float av = fabsf(v);
        if (isfinite(av) && av > local_max) local_max = av;
    }

    shared_max[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float other = shared_max[threadIdx.x + stride];
            if (other > shared_max[threadIdx.x]) shared_max[threadIdx.x] = other;
        }
        __syncthreads();
    }

    float amax = shared_max[0];
    float scale = amax > 0.0f ? (amax / (E5M2 ? 57344.0f : 448.0f)) : 1.0f;
    if (threadIdx.x == 0) scales[block_id] = scale;
    __syncthreads();

    float inv_scale = 1.0f / scale;
    for (int64_t i = threadIdx.x; i < len; i += blockDim.x) {
        float v = load_float_input_t<DTYPE>(src, base + i) * inv_scale;
        dst[base + i] = E5M2 ? fp32_to_fp8_e5m2(v) : fp32_to_fp8_e4m3(v);
    }
}

template <int DTYPE>
__global__
__launch_bounds__(256, 4)
void quant_e5m2_blockwise_stochastic_kernel_t(
    const void  * __restrict__ src,
    uint8_t      * __restrict__ dst,
    float        * __restrict__ scales,
    int64_t       rows,
    int64_t       cols,
    int64_t       blocks_per_row,
    int           block_size,
    uint64_t      seed
) {
    int64_t block_id = (int64_t)blockIdx.x;
    int64_t total_blocks = rows * blocks_per_row;
    if (block_id >= total_blocks) return;

    int64_t row = block_id / blocks_per_row;
    int64_t block_col = block_id - row * blocks_per_row;
    int64_t col_start = block_col * (int64_t)block_size;
    int64_t len = cols - col_start;
    if (len > block_size) len = block_size;
    if (len <= 0) return;

    __shared__ float shared_max[256];
    float local_max = 0.0f;
    int64_t base = row * cols + col_start;

    for (int64_t i = threadIdx.x; i < len; i += blockDim.x) {
        float av = fabsf(load_float_input_t<DTYPE>(src, base + i));
        if (isfinite(av) && av > local_max) local_max = av;
    }

    shared_max[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float other = shared_max[threadIdx.x + stride];
            if (other > shared_max[threadIdx.x]) shared_max[threadIdx.x] = other;
        }
        __syncthreads();
    }

    float amax = shared_max[0];
    float scale = amax > 0.0f ? (amax / 57344.0f) : 1.0f;
    if (threadIdx.x == 0) scales[block_id] = scale;
    __syncthreads();

    float inv_scale = 1.0f / scale;
    for (int64_t i = threadIdx.x; i < len; i += blockDim.x) {
        int64_t idx = base + i;
        float v = load_float_input_t<DTYPE>(src, idx) * inv_scale;
        dst[idx] = fp32_to_fp8_e5m2_stochastic(v, seed, (uint64_t)idx);
    }
}

template <bool E5M2>
__global__
__launch_bounds__(256, 4)
void dequant_blockwise_kernel_t(
    const uint8_t * __restrict__ src,
    const float   * __restrict__ scales,
    float         * __restrict__ dst,
    int64_t        rows,
    int64_t        cols,
    int64_t        blocks_per_row,
    int            block_size
) {
    int64_t block_id = (int64_t)blockIdx.x;
    int64_t total_blocks = rows * blocks_per_row;
    if (block_id >= total_blocks) return;

    int64_t row = block_id / blocks_per_row;
    int64_t block_col = block_id - row * blocks_per_row;
    int64_t col_start = block_col * (int64_t)block_size;
    int64_t len = cols - col_start;
    if (len > block_size) len = block_size;
    if (len <= 0) return;

    float scale = scales[block_id];
    int64_t base = row * cols + col_start;
    for (int64_t i = threadIdx.x; i < len; i += blockDim.x) {
        float v = E5M2 ? fp8_e5m2_to_fp32(src[base + i]) : fp8_e4m3_to_fp32(src[base + i]);
        dst[base + i] = v * scale;
    }
}

template <int DTYPE>
__global__
__launch_bounds__(256, 4)
void adafactor_row_mean_square_kernel_t(
    const void * __restrict__ grad,
    float      * __restrict__ row_out,
    int64_t      rows,
    int64_t      cols,
    float        eps
) {
    int64_t row = (int64_t)blockIdx.x;
    if (row >= rows) return;

    __shared__ float shared_sum[256];
    float local = 0.0f;
    int64_t base = row * cols;
    for (int64_t col = threadIdx.x; col < cols; col += blockDim.x) {
        float v = load_float_input_t<DTYPE>(grad, base + col);
        local += v * v;
    }
    shared_sum[threadIdx.x] = local;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) row_out[row] = shared_sum[0] / (float)cols + eps;
}

template <int DTYPE>
__global__
__launch_bounds__(256, 4)
void adafactor_col_mean_square_kernel_t(
    const void * __restrict__ grad,
    float      * __restrict__ col_out,
    int64_t      rows,
    int64_t      cols,
    float        eps
) {
    int64_t col = (int64_t)blockIdx.x;
    if (col >= cols) return;

    __shared__ float shared_sum[256];
    float local = 0.0f;
    for (int64_t row = threadIdx.x; row < rows; row += blockDim.x) {
        float v = load_float_input_t<DTYPE>(grad, row * cols + col);
        local += v * v;
    }
    shared_sum[threadIdx.x] = local;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) col_out[col] = shared_sum[0] / (float)rows + eps;
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

void launch_quant_e5m2_stochastic(const void* src, uint8_t* dst, int64_t numel,
                                  int dtype, uint64_t seed, hipStream_t stream) {
    if (numel == 0) return;
    if (dtype == HIP_QUANT_DTYPE_F16) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(quant_e5m2_stochastic_kernel_t<HIP_QUANT_DTYPE_F16>),
                           dim3(safe_blocks(numel)), dim3(256), 0, stream,
                           src, dst, numel, seed);
    } else if (dtype == HIP_QUANT_DTYPE_BF16) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(quant_e5m2_stochastic_kernel_t<HIP_QUANT_DTYPE_BF16>),
                           dim3(safe_blocks(numel)), dim3(256), 0, stream,
                           src, dst, numel, seed);
    } else {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(quant_e5m2_stochastic_kernel_t<HIP_QUANT_DTYPE_F32>),
                           dim3(safe_blocks(numel)), dim3(256), 0, stream,
                           src, dst, numel, seed);
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

static inline uint32_t safe_scale_blocks(int64_t rows, int64_t blocks_per_row) {
    assert(rows >= 0);
    assert(blocks_per_row > 0);
    int64_t b = rows * blocks_per_row;
    assert(b <= (int64_t)0xFFFFFFFFu);
    return (uint32_t)b;
}

void launch_quant_e4m3_blockwise(const void* src, uint8_t* dst, float* scales,
                                 int64_t rows, int64_t cols, int64_t blocks_per_row,
                                 int block_size, int dtype, hipStream_t stream) {
    if (rows == 0) return;
    uint32_t grid = safe_scale_blocks(rows, blocks_per_row);
    if (dtype == HIP_QUANT_DTYPE_F16) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(quant_blockwise_kernel_t<HIP_QUANT_DTYPE_F16, false>),
                           dim3(grid), dim3(256), 0, stream,
                           src, dst, scales, rows, cols, blocks_per_row, block_size);
    } else if (dtype == HIP_QUANT_DTYPE_BF16) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(quant_blockwise_kernel_t<HIP_QUANT_DTYPE_BF16, false>),
                           dim3(grid), dim3(256), 0, stream,
                           src, dst, scales, rows, cols, blocks_per_row, block_size);
    } else {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(quant_blockwise_kernel_t<HIP_QUANT_DTYPE_F32, false>),
                           dim3(grid), dim3(256), 0, stream,
                           src, dst, scales, rows, cols, blocks_per_row, block_size);
    }
}

void launch_quant_e5m2_blockwise(const void* src, uint8_t* dst, float* scales,
                                 int64_t rows, int64_t cols, int64_t blocks_per_row,
                                 int block_size, int dtype, hipStream_t stream) {
    if (rows == 0) return;
    uint32_t grid = safe_scale_blocks(rows, blocks_per_row);
    if (dtype == HIP_QUANT_DTYPE_F16) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(quant_blockwise_kernel_t<HIP_QUANT_DTYPE_F16, true>),
                           dim3(grid), dim3(256), 0, stream,
                           src, dst, scales, rows, cols, blocks_per_row, block_size);
    } else if (dtype == HIP_QUANT_DTYPE_BF16) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(quant_blockwise_kernel_t<HIP_QUANT_DTYPE_BF16, true>),
                           dim3(grid), dim3(256), 0, stream,
                           src, dst, scales, rows, cols, blocks_per_row, block_size);
    } else {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(quant_blockwise_kernel_t<HIP_QUANT_DTYPE_F32, true>),
                           dim3(grid), dim3(256), 0, stream,
                           src, dst, scales, rows, cols, blocks_per_row, block_size);
    }
}

void launch_quant_e5m2_blockwise_stochastic(const void* src, uint8_t* dst, float* scales,
                                            int64_t rows, int64_t cols, int64_t blocks_per_row,
                                            int block_size, int dtype, uint64_t seed,
                                            hipStream_t stream) {
    if (rows == 0) return;
    uint32_t grid = safe_scale_blocks(rows, blocks_per_row);
    if (dtype == HIP_QUANT_DTYPE_F16) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(quant_e5m2_blockwise_stochastic_kernel_t<HIP_QUANT_DTYPE_F16>),
                           dim3(grid), dim3(256), 0, stream,
                           src, dst, scales, rows, cols, blocks_per_row, block_size, seed);
    } else if (dtype == HIP_QUANT_DTYPE_BF16) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(quant_e5m2_blockwise_stochastic_kernel_t<HIP_QUANT_DTYPE_BF16>),
                           dim3(grid), dim3(256), 0, stream,
                           src, dst, scales, rows, cols, blocks_per_row, block_size, seed);
    } else {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(quant_e5m2_blockwise_stochastic_kernel_t<HIP_QUANT_DTYPE_F32>),
                           dim3(grid), dim3(256), 0, stream,
                           src, dst, scales, rows, cols, blocks_per_row, block_size, seed);
    }
}

void launch_dequant_e4m3_blockwise(const uint8_t* src, const float* scales, float* dst,
                                   int64_t rows, int64_t cols, int64_t blocks_per_row,
                                   int block_size, hipStream_t stream) {
    if (rows == 0) return;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(dequant_blockwise_kernel_t<false>),
                       dim3(safe_scale_blocks(rows, blocks_per_row)), dim3(256), 0, stream,
                       src, scales, dst, rows, cols, blocks_per_row, block_size);
}

void launch_dequant_e5m2_blockwise(const uint8_t* src, const float* scales, float* dst,
                                   int64_t rows, int64_t cols, int64_t blocks_per_row,
                                   int block_size, hipStream_t stream) {
    if (rows == 0) return;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(dequant_blockwise_kernel_t<true>),
                       dim3(safe_scale_blocks(rows, blocks_per_row)), dim3(256), 0, stream,
                       src, scales, dst, rows, cols, blocks_per_row, block_size);
}

void launch_adafactor_row_col_mean_square(const void* grad, float* row_out, float* col_out,
                                          int64_t rows, int64_t cols, int dtype,
                                          float eps, hipStream_t stream) {
    if (rows == 0 || cols == 0) return;
    assert(rows <= (int64_t)0xFFFFFFFFu);
    assert(cols <= (int64_t)0xFFFFFFFFu);
    if (dtype == HIP_QUANT_DTYPE_F16) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(adafactor_row_mean_square_kernel_t<HIP_QUANT_DTYPE_F16>),
                           dim3((uint32_t)rows), dim3(256), 0, stream,
                           grad, row_out, rows, cols, eps);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(adafactor_col_mean_square_kernel_t<HIP_QUANT_DTYPE_F16>),
                           dim3((uint32_t)cols), dim3(256), 0, stream,
                           grad, col_out, rows, cols, eps);
    } else if (dtype == HIP_QUANT_DTYPE_BF16) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(adafactor_row_mean_square_kernel_t<HIP_QUANT_DTYPE_BF16>),
                           dim3((uint32_t)rows), dim3(256), 0, stream,
                           grad, row_out, rows, cols, eps);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(adafactor_col_mean_square_kernel_t<HIP_QUANT_DTYPE_BF16>),
                           dim3((uint32_t)cols), dim3(256), 0, stream,
                           grad, col_out, rows, cols, eps);
    } else {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(adafactor_row_mean_square_kernel_t<HIP_QUANT_DTYPE_F32>),
                           dim3((uint32_t)rows), dim3(256), 0, stream,
                           grad, row_out, rows, cols, eps);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(adafactor_col_mean_square_kernel_t<HIP_QUANT_DTYPE_F32>),
                           dim3((uint32_t)cols), dim3(256), 0, stream,
                           grad, col_out, rows, cols, eps);
    }
}

} // extern "C"
