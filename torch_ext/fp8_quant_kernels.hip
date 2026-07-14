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

// Quantize buffers are normally consumed by a later kernel rather than by the
// kernel that writes them.  Tell the compiler that these one-pass accesses do
// not need to displace useful cache lines.  Define HIP_QUANT_DISABLE_NONTEMPORAL
// for a target/compiler where profiling shows ordinary cached accesses win.
template <typename T>
static __device__ __forceinline__ T hip_quant_stream_load(const T* ptr) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(HIP_QUANT_ENABLE_NONTEMPORAL) && !defined(HIP_QUANT_DISABLE_NONTEMPORAL)
    return __builtin_nontemporal_load(ptr);
#else
    return *ptr;
#endif
}

template <typename T>
static __device__ __forceinline__ void hip_quant_stream_store(T* ptr, T value) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(HIP_QUANT_ENABLE_NONTEMPORAL) && !defined(HIP_QUANT_DISABLE_NONTEMPORAL)
    __builtin_nontemporal_store(value, ptr);
#else
    *ptr = value;
#endif
}

template <int DTYPE>
static __device__ inline float load_float_input_t(const void* src, int64_t idx) {
    if constexpr (DTYPE == HIP_QUANT_DTYPE_F16) {
        return fp16_to_fp32(hip_quant_stream_load(((const uint16_t*)src) + idx));
    }
    if constexpr (DTYPE == HIP_QUANT_DTYPE_BF16) {
        return bf16_to_fp32(hip_quant_stream_load(((const uint16_t*)src) + idx));
    }
    return hip_quant_stream_load(((const float*)src) + idx);
}

template <int DTYPE>
static __device__ inline uint8_t quant_input_e4m3_t(const void* src, int64_t idx) {
    if constexpr (DTYPE == HIP_QUANT_DTYPE_BF16) {
        return bf16_to_fp8_e4m3(hip_quant_stream_load(((const uint16_t*)src) + idx));
    }
    return fp32_to_fp8_e4m3(load_float_input_t<DTYPE>(src, idx));
}

template <int DTYPE>
static __device__ inline uint8_t quant_input_e5m2_t(const void* src, int64_t idx) {
    if constexpr (DTYPE == HIP_QUANT_DTYPE_BF16) {
        return bf16_to_fp8_e5m2(hip_quant_stream_load(((const uint16_t*)src) + idx));
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
            hip_quant_stream_store(dst + cur, quant_input_e4m3_t<DTYPE>(src, cur));
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
            hip_quant_stream_store(dst + cur, quant_input_e5m2_t<DTYPE>(src, cur));
        }
    }
}

// ---------------------------------------------------------------------------
// Fused quantize + transpose: [rows, cols] floating point -> [cols, rows]
// FP8.  The 32x32 shared-memory tile keeps source reads and transposed writes
// coalesced, avoiding a materialised full-precision transpose.
// ---------------------------------------------------------------------------
template <int DTYPE, bool E5M2>
__global__
__launch_bounds__(256, 2)
void quant_transpose_kernel_t(
    const void* __restrict__ src,
    uint8_t* __restrict__ dst,
    int64_t rows,
    int64_t cols
) {
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;
    __shared__ uint8_t tile[TILE_DIM][TILE_DIM + 1];

    const int64_t input_col = (int64_t)blockIdx.x * TILE_DIM + threadIdx.x;
    const int64_t input_row_base = (int64_t)blockIdx.y * TILE_DIM + threadIdx.y;

    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        const int64_t input_row = input_row_base + j;
        uint8_t value = 0;
        if (input_row < rows && input_col < cols) {
            const int64_t offset = input_row * cols + input_col;
            value = E5M2 ? quant_input_e5m2_t<DTYPE>(src, offset)
                         : quant_input_e4m3_t<DTYPE>(src, offset);
        }
        tile[threadIdx.y + j][threadIdx.x] = value;
    }
    __syncthreads();

    // Swapping the tile indices makes adjacent lanes write adjacent columns
    // of the transposed tensor. The +1 tile stride prevents LDS bank aliasing.
    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        const int64_t output_row = (int64_t)blockIdx.x * TILE_DIM + threadIdx.y + j;
        const int64_t output_col = (int64_t)blockIdx.y * TILE_DIM + threadIdx.x;
        if (output_row < cols && output_col < rows) {
            hip_quant_stream_store(
                dst + output_row * rows + output_col,
                tile[threadIdx.x][threadIdx.y + j]);
        }
    }
}

// ---------------------------------------------------------------------------
// Delayed-scaling E4M3 activation quantization.
//
// The current scale is intentionally consumed before the observed amax is
// committed.  This is the usual delayed-scaling contract: the reduction feeds
// the *next* invocation, while this invocation has one streaming read of the
// activation and one streaming write of its FP8 bytes.  A tiny follow-up
// kernel updates the ring-buffer-derived scale after all blocks have finished
// their atomic amax reduction.
// ---------------------------------------------------------------------------
template <int DTYPE>
__global__
__launch_bounds__(256, 4)
void quant_e4m3_delayed_amax_kernel_t(
    const void  * __restrict__ src,
    uint8_t      * __restrict__ dst,
    const float  * __restrict__ quant_scale,
    float        * __restrict__ amax_scratch,
    int          * __restrict__ sample_nonfinite,
    int64_t       numel
) {
    const int64_t idx = ((int64_t)blockIdx.x * 256 + threadIdx.x)
                      * HIP_QUANT_ELEMS_PER_THREAD;
    const float scale = quant_scale[0];
    float local_amax = 0.0f;
    bool nonfinite = false;

    #pragma unroll
    for (int i = 0; i < HIP_QUANT_ELEMS_PER_THREAD; ++i) {
        const int64_t cur = idx + i;
        if (cur >= numel) continue;
        const float value = load_float_input_t<DTYPE>(src, cur);
        const float abs_value = fabsf(value);
        if (isfinite(abs_value)) {
            if (abs_value > local_amax) local_amax = abs_value;
        } else {
            nonfinite = true;
        }
        hip_quant_stream_store(dst + cur, fp32_to_fp8_e4m3(value * scale));
    }

    // IEEE-positive floats retain their numerical order when reinterpreted as
    // unsigned integers, so atomicMax supplies a fast global amax reduction.
    if (local_amax > 0.0f) {
        atomicMax(reinterpret_cast<unsigned int*>(amax_scratch),
                  __float_as_uint(local_amax));
    }
    if (nonfinite) {
        // ROCm 7.1 does not expose byte-sized atomics, so use an int32 scratch
        // flag and let the single-block finalize kernel set the public bool.
        atomicExch(sample_nonfinite, 1);
    }
}

__global__
__launch_bounds__(256, 1)
void finalize_delayed_amax_kernel(
    const float   * __restrict__ amax_scratch,
    const int     * __restrict__ sample_nonfinite,
    uint8_t       * __restrict__ found_nonfinite,
    float         * __restrict__ scale,
    float         * __restrict__ inv_scale,
    float         * __restrict__ amax_history,
    int            history_len,
    int            history_slot
) {
    // A non-finite sample must not poison either the delayed history or the
    // scale used by subsequent training steps. The sticky flag is already set
    // by the streaming kernel for the caller to inspect.
    if (sample_nonfinite[0] != 0) {
        if (threadIdx.x == 0) found_nonfinite[0] = (uint8_t)1;
        return;
    }

    __shared__ float partial_max[256];
    float local_max = 0.0f;
    for (int i = threadIdx.x; i < history_len; i += blockDim.x) {
        float value = amax_history[i];
        if (i == history_slot) value = amax_scratch[0];
        if (value > local_max) local_max = value;
    }
    partial_max[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float other = partial_max[threadIdx.x + stride];
            if (other > partial_max[threadIdx.x]) partial_max[threadIdx.x] = other;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const float observed_max = fmaxf(partial_max[0], 1.0e-12f);
        const float next_scale = 448.0f / observed_max;
        amax_history[history_slot] = amax_scratch[0];
        scale[0] = next_scale;
        inv_scale[0] = 1.0f / next_scale;
    }
}

// Quantize a scaled floating-point tensor directly into a persistent FP8
// shadow. This avoids both the temporary uint8 allocation and the following
// device-to-device copy used by the old refresh path.
template <int DTYPE>
__global__
__launch_bounds__(256, 4)
void refresh_e4m3_shadow_kernel_t(
    const void  * __restrict__ src,
    uint8_t      * __restrict__ shadow,
    const float  * __restrict__ quant_scale,
    int64_t       numel
) {
    const int64_t idx = ((int64_t)blockIdx.x * 256 + threadIdx.x)
                      * HIP_QUANT_ELEMS_PER_THREAD;
    const float scale = quant_scale[0];
    #pragma unroll
    for (int i = 0; i < HIP_QUANT_ELEMS_PER_THREAD; ++i) {
        const int64_t cur = idx + i;
        if (cur < numel) {
            hip_quant_stream_store(shadow + cur,
                                  fp32_to_fp8_e4m3(load_float_input_t<DTYPE>(src, cur) * scale));
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

void launch_quant_e4m3_delayed_amax(
    const void* src, uint8_t* dst, const float* quant_scale,
    float* amax_scratch, int* sample_nonfinite, uint8_t* found_nonfinite,
    float* scale, float* inv_scale, float* amax_history,
    int64_t numel, int history_len, int history_slot, int dtype,
    hipStream_t stream
) {
    if (numel == 0) return;
    if (dtype == HIP_QUANT_DTYPE_F16) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(quant_e4m3_delayed_amax_kernel_t<HIP_QUANT_DTYPE_F16>),
                           dim3(safe_blocks(numel)), dim3(256), 0, stream,
                           src, dst, quant_scale, amax_scratch, sample_nonfinite, numel);
    } else if (dtype == HIP_QUANT_DTYPE_BF16) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(quant_e4m3_delayed_amax_kernel_t<HIP_QUANT_DTYPE_BF16>),
                           dim3(safe_blocks(numel)), dim3(256), 0, stream,
                           src, dst, quant_scale, amax_scratch, sample_nonfinite, numel);
    } else {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(quant_e4m3_delayed_amax_kernel_t<HIP_QUANT_DTYPE_F32>),
                           dim3(safe_blocks(numel)), dim3(256), 0, stream,
                           src, dst, quant_scale, amax_scratch, sample_nonfinite, numel);
    }
    hipLaunchKernelGGL(finalize_delayed_amax_kernel, dim3(1), dim3(256), 0, stream,
                       amax_scratch, sample_nonfinite, found_nonfinite, scale, inv_scale,
                       amax_history, history_len, history_slot);
}

void launch_refresh_e4m3_shadow(
    const void* src, uint8_t* shadow, const float* quant_scale,
    int64_t numel, int dtype, hipStream_t stream
) {
    if (numel == 0) return;
    if (dtype == HIP_QUANT_DTYPE_F16) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(refresh_e4m3_shadow_kernel_t<HIP_QUANT_DTYPE_F16>),
                           dim3(safe_blocks(numel)), dim3(256), 0, stream,
                           src, shadow, quant_scale, numel);
    } else if (dtype == HIP_QUANT_DTYPE_BF16) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(refresh_e4m3_shadow_kernel_t<HIP_QUANT_DTYPE_BF16>),
                           dim3(safe_blocks(numel)), dim3(256), 0, stream,
                           src, shadow, quant_scale, numel);
    } else {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(refresh_e4m3_shadow_kernel_t<HIP_QUANT_DTYPE_F32>),
                           dim3(safe_blocks(numel)), dim3(256), 0, stream,
                           src, shadow, quant_scale, numel);
    }
}

static inline uint32_t safe_transpose_blocks(int64_t extent) {
    assert(extent >= 0);
    const int64_t blocks = (extent + 31) / 32;
    assert(blocks <= (int64_t)0xFFFFFFFFu);
    return (uint32_t)blocks;
}

void launch_quant_e4m3_transpose(const void* src, uint8_t* dst,
                                 int64_t rows, int64_t cols, int dtype,
                                 hipStream_t stream) {
    if (rows == 0 || cols == 0) return;
    const dim3 grid(safe_transpose_blocks(cols), safe_transpose_blocks(rows));
    const dim3 block(32, 8);
    if (dtype == HIP_QUANT_DTYPE_F16) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(quant_transpose_kernel_t<HIP_QUANT_DTYPE_F16, false>),
                           grid, block, 0, stream, src, dst, rows, cols);
    } else if (dtype == HIP_QUANT_DTYPE_BF16) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(quant_transpose_kernel_t<HIP_QUANT_DTYPE_BF16, false>),
                           grid, block, 0, stream, src, dst, rows, cols);
    } else {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(quant_transpose_kernel_t<HIP_QUANT_DTYPE_F32, false>),
                           grid, block, 0, stream, src, dst, rows, cols);
    }
}

void launch_quant_e5m2_transpose(const void* src, uint8_t* dst,
                                 int64_t rows, int64_t cols, int dtype,
                                 hipStream_t stream) {
    if (rows == 0 || cols == 0) return;
    const dim3 grid(safe_transpose_blocks(cols), safe_transpose_blocks(rows));
    const dim3 block(32, 8);
    if (dtype == HIP_QUANT_DTYPE_F16) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(quant_transpose_kernel_t<HIP_QUANT_DTYPE_F16, true>),
                           grid, block, 0, stream, src, dst, rows, cols);
    } else if (dtype == HIP_QUANT_DTYPE_BF16) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(quant_transpose_kernel_t<HIP_QUANT_DTYPE_BF16, true>),
                           grid, block, 0, stream, src, dst, rows, cols);
    } else {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(quant_transpose_kernel_t<HIP_QUANT_DTYPE_F32, true>),
                           grid, block, 0, stream, src, dst, rows, cols);
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
