// torch_ext/q_to_fp8_kernels.hip
//
// GPU-resident GGML Q-type -> FP8 dequantization kernels for the PyTorch
// extension.  These reuse the exact same device-side math as
// kernels/dequant_to_fp8.cu but are compiled into the _C extension so they
// operate on torch CUDA tensors with zero CPU round-trips.

#include <hip/hip_runtime.h>
#include <assert.h>
#include "../hip_quant_types.h"
#include "../hip_quant_util.h"
#include "../hip_iquant_util.h"

// -----------------------------------------------------------------------
// Device helpers (duplicated from kernels/dequant_to_fp8.cu so this
// translation unit is self-contained within the torch extension build)
// -----------------------------------------------------------------------

template <bool E5M2>
__device__ inline uint8_t fp32_to_output_fp8(float value) {
    return E5M2 ? fp32_to_fp8_e5m2(value) : fp32_to_fp8_e4m3(value);
}

__device__ inline uint8_t unpack_k4_scale(const uint8_t *scales, int group) {
    if (group < 4) return scales[group] & 63;
    return (scales[group + 4] & 0x0f) | ((scales[group - 4] >> 6) << 4);
}

__device__ inline uint8_t unpack_k4_min(const uint8_t *scales, int group) {
    if (group < 4) return scales[group + 4] & 63;
    return (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4);
}

// -----------------------------------------------------------------------
// Per-type dequant-to-FP8 kernels (identical logic to dequant_to_fp8.cu)
// -----------------------------------------------------------------------

template <bool E5M2>
__global__ __launch_bounds__(32, 8)
void dequant_q4_0_to_fp8_kernel(
    const block_q4_0 * __restrict__ src, uint8_t * __restrict__ dst,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int row = (int)blockIdx.x + (int)blockIdx.z * (int)gridDim.x;
    const int block = blockIdx.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= 32) return;
    const block_q4_0 q = src[row * blocks_per_row + block];
    const uint8_t nibble = i < 16 ? (q.qs[i] & 0x0f) : (q.qs[i - 16] >> 4);
    dst[row * n_per_row + block * 32 + i] = fp32_to_output_fp8<E5M2>(
        fp16_to_fp32(q.d) * ((int)nibble - 8));
}

template <bool E5M2>
__global__ __launch_bounds__(32, 8)
void dequant_q4_1_to_fp8_kernel(
    const block_q4_1 * __restrict__ src, uint8_t * __restrict__ dst,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int row = (int)blockIdx.x + (int)blockIdx.z * (int)gridDim.x;
    const int block = blockIdx.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= 32) return;
    const block_q4_1 q = src[row * blocks_per_row + block];
    const uint8_t nibble = i < 16 ? (q.qs[i] & 0x0f) : (q.qs[i - 16] >> 4);
    dst[row * n_per_row + block * 32 + i] = fp32_to_output_fp8<E5M2>(
        fp16_to_fp32(q.d) * (float)nibble + fp16_to_fp32(q.m));
}

template <bool E5M2>
__global__ __launch_bounds__(32, 8)
void dequant_q5_0_to_fp8_kernel(
    const block_q5_0 * __restrict__ src, uint8_t * __restrict__ dst,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int row = (int)blockIdx.x + (int)blockIdx.z * (int)gridDim.x;
    const int block = blockIdx.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= 32) return;
    const block_q5_0 q = src[row * blocks_per_row + block];
    const uint8_t low = i < 16 ? (q.qs[i] & 0x0f) : (q.qs[i - 16] >> 4);
    const int value = (int)low | (((q.qh[i >> 3] >> (i & 7)) & 1) << 4);
    dst[row * n_per_row + block * 32 + i] = fp32_to_output_fp8<E5M2>(
        fp16_to_fp32(q.d) * (value - 16));
}

template <bool E5M2>
__global__ __launch_bounds__(32, 8)
void dequant_q5_1_to_fp8_kernel(
    const block_q5_1 * __restrict__ src, uint8_t * __restrict__ dst,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int row = (int)blockIdx.x + (int)blockIdx.z * (int)gridDim.x;
    const int block = blockIdx.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= 32) return;
    const block_q5_1 q = src[row * blocks_per_row + block];
    const uint8_t low = i < 16 ? (q.qs[i] & 0x0f) : (q.qs[i - 16] >> 4);
    const int value = (int)low | (((q.qh[i >> 3] >> (i & 7)) & 1) << 4);
    dst[row * n_per_row + block * 32 + i] = fp32_to_output_fp8<E5M2>(
        fp16_to_fp32(q.d) * (float)value + fp16_to_fp32(q.m));
}

template <bool E5M2>
__global__ __launch_bounds__(32, 8)
void dequant_q8_0_to_fp8_kernel(
    const block_q8_0 * __restrict__ src, uint8_t * __restrict__ dst,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int row = (int)blockIdx.x + (int)blockIdx.z * (int)gridDim.x;
    const int block = blockIdx.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= 32) return;
    const block_q8_0 q = src[row * blocks_per_row + block];
    dst[row * n_per_row + block * 32 + i] = fp32_to_output_fp8<E5M2>(
        fp16_to_fp32(q.d) * (float)q.qs[i]);
}

template <bool E5M2>
__global__ __launch_bounds__(32, 8)
void dequant_q8_1_to_fp8_kernel(
    const block_q8_1 * __restrict__ src, uint8_t * __restrict__ dst,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int row = (int)blockIdx.x + (int)blockIdx.z * (int)gridDim.x;
    const int block = blockIdx.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= 32) return;
    const block_q8_1 q = src[row * blocks_per_row + block];
    dst[row * n_per_row + block * 32 + i] = fp32_to_output_fp8<E5M2>(
        fp16_to_fp32(q.d) * (float)q.qs[i]);
}

template <bool E5M2>
__global__ __launch_bounds__(256, 4)
void dequant_q2_k_to_fp8_kernel(
    const block_q2_K * __restrict__ src, uint8_t * __restrict__ dst,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int row = (int)blockIdx.x + (int)blockIdx.z * (int)gridDim.x;
    const int block = blockIdx.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= QK_K) return;
    const block_q2_K q = src[row * blocks_per_row + block];
    const int segment = (i & 127) >> 5;
    const int packed = (i >> 7) * 32 + (i & 31);
    const int value = (q.qs[packed] >> (2 * segment)) & 3;
    const uint8_t scale_min = q.scales[i >> 4];
    const float scale = fp16_to_fp32(q.d) * (float)(scale_min & 0x0f);
    const float min = fp16_to_fp32(q.dmin) * (float)(scale_min >> 4);
    dst[row * n_per_row + block * QK_K + i] = fp32_to_output_fp8<E5M2>(
        scale * (float)value - min);
}

template <bool E5M2>
__global__ __launch_bounds__(256, 4)
void dequant_q3_k_to_fp8_kernel(
    const block_q3_K * __restrict__ src, uint8_t * __restrict__ dst,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int row = (int)blockIdx.x + (int)blockIdx.z * (int)gridDim.x;
    const int block = blockIdx.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= QK_K) return;
    const block_q3_K q = src[row * blocks_per_row + block];
    const int segment = (i & 127) >> 5;
    const int packed = (i >> 7) * 32 + (i & 31);
    int value = (q.qs[packed] >> (2 * segment)) & 3;
    if (q.hmask[i & 31] & (1u << (i >> 5))) value += 4;
    value -= 4;
    const int group = i >> 4;
    const uint8_t low = group < 8 ? (q.scales[group] & 0x0f) : (q.scales[group - 8] >> 4);
    const uint8_t high = (q.scales[8 + (group & 3)] >> (2 * (group >> 2))) & 3;
    const int scale = ((int)high << 4 | low) - 32;
    dst[row * n_per_row + block * QK_K + i] = fp32_to_output_fp8<E5M2>(
        fp16_to_fp32(q.d) * (float)scale * (float)value);
}

template <bool E5M2>
__global__ __launch_bounds__(256, 4)
void dequant_q4_k_to_fp8_kernel(
    const block_q4_K * __restrict__ src, uint8_t * __restrict__ dst,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int row = (int)blockIdx.x + (int)blockIdx.z * (int)gridDim.x;
    const int block = blockIdx.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= QK_K) return;
    const block_q4_K q = src[row * blocks_per_row + block];
    const int group = i >> 5;
    const int packed = (i >> 6) * 32 + (i & 31);
    const uint8_t value = (q.qs[packed] >> (((i >> 5) & 1) * 4)) & 0x0f;
    const float scale = fp16_to_fp32(q.d) * (float)unpack_k4_scale(q.scales, group);
    const float min = fp16_to_fp32(q.dmin) * (float)unpack_k4_min(q.scales, group);
    dst[row * n_per_row + block * QK_K + i] = fp32_to_output_fp8<E5M2>(
        scale * (float)value - min);
}

template <bool E5M2>
__global__ __launch_bounds__(256, 4)
void dequant_q5_k_to_fp8_kernel(
    const block_q5_K * __restrict__ src, uint8_t * __restrict__ dst,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int row = (int)blockIdx.x + (int)blockIdx.z * (int)gridDim.x;
    const int block = blockIdx.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= QK_K) return;
    const block_q5_K q = src[row * blocks_per_row + block];
    const int group = i >> 5;
    const int chunk = i >> 6;
    const int lane = i & 31;
    const int half = (i >> 5) & 1;
    const int packed = chunk * 32 + lane;
    const int high = (q.qh[lane] >> (chunk * 2 + half)) & 1;
    const int value = ((q.qs[packed] >> (half * 4)) & 0x0f) | (high << 4);
    const float scale = fp16_to_fp32(q.d) * (float)unpack_k4_scale(q.scales, group);
    const float min = fp16_to_fp32(q.dmin) * (float)unpack_k4_min(q.scales, group);
    dst[row * n_per_row + block * QK_K + i] = fp32_to_output_fp8<E5M2>(
        scale * (float)value - min);
}

template <bool E5M2>
__global__ __launch_bounds__(256, 4)
void dequant_q6_k_to_fp8_kernel(
    const block_q6_K * __restrict__ src, uint8_t * __restrict__ dst,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int row = (int)blockIdx.x + (int)blockIdx.z * (int)gridDim.x;
    const int block = blockIdx.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= QK_K) return;
    const block_q6_K q = src[row * blocks_per_row + block];
    const int chunk = i >> 7;
    const int section = (i & 127) >> 5;
    const int lane = i & 31;
    const int ql_index = chunk * 64 + (section & 1 ? 32 : 0) + lane;
    const int low = (q.ql[ql_index] >> ((section >> 1) * 4)) & 0x0f;
    const int high = (q.qh[chunk * 32 + lane] >> (section * 2)) & 3;
    const int value = (low | (high << 4)) - 32;
    const float scale = fp16_to_fp32(q.d) * (float)q.scales[i >> 4];
    dst[row * n_per_row + block * QK_K + i] = fp32_to_output_fp8<E5M2>(
        scale * (float)value);
}

template <bool E5M2>
__global__ __launch_bounds__(32, 8)
void dequant_iq4_nl_to_fp8_kernel(
    const block_iq4_nl * __restrict__ src, uint8_t * __restrict__ dst,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int row = (int)blockIdx.x + (int)blockIdx.z * (int)gridDim.x;
    const int block = blockIdx.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= QK4_NL) return;
    const block_iq4_nl q = src[row * blocks_per_row + block];
    const uint8_t nibble = i < 16 ? (q.qs[i] & 0x0f) : (q.qs[i - 16] >> 4);
    dst[row * n_per_row + block * QK4_NL + i] = fp32_to_output_fp8<E5M2>(
        fp16_to_fp32(q.d) * (float)d_kvalues_iq4nl[nibble]);
}

template <bool E5M2>
__global__ __launch_bounds__(256, 4)
void dequant_iq4_xs_to_fp8_kernel(
    const block_iq4_xs * __restrict__ src, uint8_t * __restrict__ dst,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int row = (int)blockIdx.x + (int)blockIdx.z * (int)gridDim.x;
    const int block = blockIdx.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= QK_K) return;
    const block_iq4_xs q = src[row * blocks_per_row + block];

    const int ib = i / 32;          // sub-block (0..7)
    const int pos_in_sub = i % 32;  // position (0..31)
    const int nibble = (pos_in_sub < 16)
        ? (q.qs[16 * ib + pos_in_sub] & 0x0f)
        : (q.qs[16 * ib + (pos_in_sub - 16)] >> 4);

    const int ls_low = (q.scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf;
    const int ls_high = (q.scales_h >> (2 * ib)) & 3;
    const int ls = ls_low | (ls_high << 4);
    const float dl = fp16_to_fp32(q.d) * (float)(ls - 32);

    dst[row * n_per_row + block * QK_K + i] = fp32_to_output_fp8<E5M2>(
        dl * (float)d_kvalues_iq4nl[nibble]);
}

// -----------------------------------------------------------------------
// Dispatch helper: given GGML type number, launch the correct kernel.
// Returns 1 on success, 0 if the type is unsupported.
// -----------------------------------------------------------------------
template <bool E5M2>
static int dispatch_dequant_q_to_fp8(
    int type, const uint8_t *src, uint8_t *dst,
    int nrows, int blocks_per_row, int n_per_row, int block_size,
    hipStream_t stream
) {
    const unsigned int row_grid = (unsigned int)((nrows < 65535) ? nrows : 65535);
    const unsigned int row_layers = ((unsigned int)nrows + row_grid - 1u) / row_grid;
    dim3 grid(row_grid, (unsigned int)blocks_per_row, row_layers);
    int threads = (block_size <= 32) ? 32 : 256;
    (void)threads;

    switch (type) {
        case 2:
            hipLaunchKernelGGL(HIP_KERNEL_NAME(dequant_q4_0_to_fp8_kernel<E5M2>), grid, 32, 0, stream,
                (const block_q4_0 *)src, dst, nrows, blocks_per_row, n_per_row);
            break;
        case 3:
            hipLaunchKernelGGL(HIP_KERNEL_NAME(dequant_q4_1_to_fp8_kernel<E5M2>), grid, 32, 0, stream,
                (const block_q4_1 *)src, dst, nrows, blocks_per_row, n_per_row);
            break;
        case 6:
            hipLaunchKernelGGL(HIP_KERNEL_NAME(dequant_q5_0_to_fp8_kernel<E5M2>), grid, 32, 0, stream,
                (const block_q5_0 *)src, dst, nrows, blocks_per_row, n_per_row);
            break;
        case 7:
            hipLaunchKernelGGL(HIP_KERNEL_NAME(dequant_q5_1_to_fp8_kernel<E5M2>), grid, 32, 0, stream,
                (const block_q5_1 *)src, dst, nrows, blocks_per_row, n_per_row);
            break;
        case 8:
            hipLaunchKernelGGL(HIP_KERNEL_NAME(dequant_q8_0_to_fp8_kernel<E5M2>), grid, 32, 0, stream,
                (const block_q8_0 *)src, dst, nrows, blocks_per_row, n_per_row);
            break;
        case 9:
            hipLaunchKernelGGL(HIP_KERNEL_NAME(dequant_q8_1_to_fp8_kernel<E5M2>), grid, 32, 0, stream,
                (const block_q8_1 *)src, dst, nrows, blocks_per_row, n_per_row);
            break;
        case 10:
            hipLaunchKernelGGL(HIP_KERNEL_NAME(dequant_q2_k_to_fp8_kernel<E5M2>), grid, 256, 0, stream,
                (const block_q2_K *)src, dst, nrows, blocks_per_row, n_per_row);
            break;
        case 11:
            hipLaunchKernelGGL(HIP_KERNEL_NAME(dequant_q3_k_to_fp8_kernel<E5M2>), grid, 256, 0, stream,
                (const block_q3_K *)src, dst, nrows, blocks_per_row, n_per_row);
            break;
        case 12:
            hipLaunchKernelGGL(HIP_KERNEL_NAME(dequant_q4_k_to_fp8_kernel<E5M2>), grid, 256, 0, stream,
                (const block_q4_K *)src, dst, nrows, blocks_per_row, n_per_row);
            break;
        case 13:
            hipLaunchKernelGGL(HIP_KERNEL_NAME(dequant_q5_k_to_fp8_kernel<E5M2>), grid, 256, 0, stream,
                (const block_q5_K *)src, dst, nrows, blocks_per_row, n_per_row);
            break;
        case 14:
            hipLaunchKernelGGL(HIP_KERNEL_NAME(dequant_q6_k_to_fp8_kernel<E5M2>), grid, 256, 0, stream,
                (const block_q6_K *)src, dst, nrows, blocks_per_row, n_per_row);
            break;
        case 20:
            hipLaunchKernelGGL(HIP_KERNEL_NAME(dequant_iq4_nl_to_fp8_kernel<E5M2>), grid, 32, 0, stream,
                (const block_iq4_nl *)src, dst, nrows, blocks_per_row, n_per_row);
            break;
        case 23:
            hipLaunchKernelGGL(HIP_KERNEL_NAME(dequant_iq4_xs_to_fp8_kernel<E5M2>), grid, 256, 0, stream,
                (const block_iq4_xs *)src, dst, nrows, blocks_per_row, n_per_row);
            break;
        default:
            return 0;
    }
    return 1;
}

// -----------------------------------------------------------------------
// C-linkage launcher called from pytorch_bindings.cpp
// -----------------------------------------------------------------------
extern "C" {

void launch_dequant_q_to_fp8(
    const uint8_t* src, uint8_t* dst,
    int type_num, int nrows, int blocks_per_row, int n_per_row,
    int block_size, int e5m2,
    hipStream_t stream
) {
    if (nrows == 0 || blocks_per_row == 0) return;
    if (e5m2) {
        dispatch_dequant_q_to_fp8<true>(type_num, src, dst, nrows, blocks_per_row, n_per_row, block_size, stream);
    } else {
        dispatch_dequant_q_to_fp8<false>(type_num, src, dst, nrows, blocks_per_row, n_per_row, block_size, stream);
    }
}

} // extern "C"
