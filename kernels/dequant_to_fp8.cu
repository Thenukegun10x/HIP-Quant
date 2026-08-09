#include <hip/hip_runtime.h>
#include "../hip_quant_types.h"
#include "../hip_quant_util.h"
#include "../hip_iquant_util.h"

// IQ2_XXS / IQ3_XXS grid tables (declared in their quant kernels, in constant memory)
extern __constant__ int8_t d_iq2xxs_grid[256][8];
extern __constant__ int8_t d_iq3xxs_grid[256][4];

// GGML-compatible sign lookup table for IQ2_XXS / IQ2_XS / IQ3_XXS
// Maps 7-bit sign patterns to 8-bit expanded sign bytes.
__device__ const uint8_t d_ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

// Bit masks for testing sign bits (positions 0-7)
__device__ const uint8_t d_kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

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
__global__ __launch_bounds__(128, 8)
void dequant_q1_0_to_fp8_kernel(
    const block_q1_0 * __restrict__ src, uint8_t * __restrict__ dst,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int row = (int)blockIdx.x + (int)blockIdx.z * (int)gridDim.x;
    const int block = blockIdx.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= QK1_0) return;
    const block_q1_0 q = src[row * blocks_per_row + block];
    const bool bit = (q.qs[i / 8] >> (i % 8)) & 1;
    const float d = fp16_to_fp32(q.d);
    dst[row * n_per_row + block * QK1_0 + i] = fp32_to_output_fp8<E5M2>(bit ? d : -d);
}

template <bool E5M2>
__global__ __launch_bounds__(64, 8)
void dequant_q2_0_to_fp8_kernel(
    const block_q2_0 * __restrict__ src, uint8_t * __restrict__ dst,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int row = (int)blockIdx.x + (int)blockIdx.z * (int)gridDim.x;
    const int block = blockIdx.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= QK2_0) return;
    const block_q2_0 q = src[row * blocks_per_row + block];
    const int code = (q.qs[i / 4] >> (2 * (i % 4))) & 0x03;
    dst[row * n_per_row + block * QK2_0 + i] = fp32_to_output_fp8<E5M2>(
        (float)(code - 1) * fp16_to_fp32(q.d));
}

template <bool E5M2>
__global__ __launch_bounds__(QK_K, 4)
void dequant_q8_k_to_fp8_kernel(
    const block_q8_K * __restrict__ src, uint8_t * __restrict__ dst,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int row = (int)blockIdx.x + (int)blockIdx.z * (int)gridDim.x;
    const int block = blockIdx.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= QK_K) return;
    const block_q8_K q = src[row * blocks_per_row + block];
    dst[row * n_per_row + block * QK_K + i] = fp32_to_output_fp8<E5M2>(
        q.d * (float)q.qs[i]);
}

template <bool E5M2>
__global__ __launch_bounds__(256, 8)
void dequant_bf16_to_fp8_kernel(
    const uint16_t * __restrict__ src, uint8_t * __restrict__ dst,
    int64_t n_elements
) {
    int64_t idx = (int64_t)blockIdx.x * 256 + threadIdx.x;
    if (idx >= n_elements) return;
    const float v = __int_as_float((int)((uint32_t)src[idx] << 16));
    dst[idx] = fp32_to_output_fp8<E5M2>(v);
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

// =========================================================================
// I-Quant dequant-to-FP8 kernels (GGML reference formulas)
// =========================================================================

template <bool E5M2>
__global__ __launch_bounds__(256, 4)
void dequant_iq2_xxs_to_fp8_kernel(
    const block_iq2_xxs * __restrict__ src, uint8_t * __restrict__ dst,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int row = (int)blockIdx.x + (int)blockIdx.z * (int)gridDim.x;
    const int block = blockIdx.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= QK_K) return;
    const block_iq2_xxs q = src[row * blocks_per_row + block];

    const int ib32 = i / 32;        // sub-block index (0..7)
    const int l = (i % 32) / 8;     // group within sub-block (0..3)
    const int j = i % 8;            // position within group (0..7)

    const uint8_t *aux8 = (const uint8_t *)(q.qs + 4 * ib32);
    const uint8_t *grid = (const uint8_t *)(d_iq2xxs_grid + aux8[l]);
    const uint32_t aux32_1 = ((const uint32_t *)(q.qs + 4 * ib32))[1];
    const uint8_t signs = d_ksigns_iq2xs[(aux32_1 >> (7 * l)) & 127];
    const float db = fp16_to_fp32(q.d) * (0.5f + (float)(aux32_1 >> 28)) * 0.25f;

    float val = db * (float)((int8_t)grid[j]);
    dst[row * n_per_row + block * QK_K + i] = fp32_to_output_fp8<E5M2>(
        (signs & d_kmask_iq2xs[j]) ? -val : val);
}

template <bool E5M2>
__global__ __launch_bounds__(256, 4)
void dequant_iq2_xs_to_fp8_kernel(
    const block_iq2_xs * __restrict__ src, uint8_t * __restrict__ dst,
    const int8_t * __restrict__ grid,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int row = (int)blockIdx.x + (int)blockIdx.z * (int)gridDim.x;
    const int block = blockIdx.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= QK_K) return;
    const block_iq2_xs q = src[row * blocks_per_row + block];

    const int ib32 = i / 32;
    const int l = (i % 32) / 8;
    const int j = i % 8;

    const uint16_t qs_entry = q.qs[4 * ib32 + l];
    const uint8_t *grid_row = (const uint8_t *)(grid + (qs_entry & 511));
    const uint8_t signs = d_ksigns_iq2xs[qs_entry >> 9];

    const int nibble = q.scales[ib32];
    const float db_l = fp16_to_fp32(q.d) * (0.5f + (float)(nibble & 0xf)) * 0.25f;
    const float db_h = fp16_to_fp32(q.d) * (0.5f + (float)(nibble >> 4)) * 0.25f;
    const float db = (l / 2) == 0 ? db_l : db_h;

    float val = db * (float)((int8_t)grid_row[j]);
    dst[row * n_per_row + block * QK_K + i] = fp32_to_output_fp8<E5M2>(
        (signs & d_kmask_iq2xs[j]) ? -val : val);
}

template <bool E5M2>
__global__ __launch_bounds__(256, 4)
void dequant_iq3_xxs_to_fp8_kernel(
    const block_iq3_xxs * __restrict__ src, uint8_t * __restrict__ dst,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int row = (int)blockIdx.x + (int)blockIdx.z * (int)gridDim.x;
    const int block = blockIdx.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= QK_K) return;
    const block_iq3_xxs q = src[row * blocks_per_row + block];

    const int ib32 = i / 32;
    const int l = (i % 32) / 8;
    const int j = i % 8;

    const uint8_t *grid_qs = q.qs + 8 * ib32;
    const uint8_t *scales_and_signs = q.qs + QK_K / 4;
    const uint32_t aux32 = ((const uint32_t *)scales_and_signs)[ib32];
    const float db = fp16_to_fp32(q.d) * (0.5f + (float)(aux32 >> 28)) * 0.5f;
    const uint8_t signs = d_ksigns_iq2xs[(aux32 >> (7 * l)) & 127];

    const int grid_idx = (j < 4)
        ? (int)grid_qs[2 * l + 0]
        : (int)grid_qs[2 * l + 1];
    const int grid_pos = j & 3;

    const uint8_t *grid_row = (const uint8_t *)(d_iq3xxs_grid + grid_idx);
    float val = db * (float)((int8_t)grid_row[grid_pos]);
    dst[row * n_per_row + block * QK_K + i] = fp32_to_output_fp8<E5M2>(
        (signs & d_kmask_iq2xs[j]) ? -val : val);
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

template <bool E5M2>
__global__ __launch_bounds__(256, 4)
void dequant_iq1_s_to_fp8_kernel(
    const block_iq1_s * __restrict__ src, uint8_t * __restrict__ dst,
    const int8_t * __restrict__ iq1s_grid,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int row = (int)blockIdx.x + (int)blockIdx.z * (int)gridDim.x;
    const int block = blockIdx.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= QK_K) return;
    const block_iq1_s q = src[row * blocks_per_row + block];

    const int ib = i / 32;
    const int l = (i % 32) / 8;
    const int j = i % 8;

    const float d = fp16_to_fp32(q.d);
    const uint16_t qh_entry = q.qh[ib];
    const float dl = d * (float)(2 * ((qh_entry >> 12) & 7) + 1);
    const float delta = (qh_entry & 0x8000) ? -0.125f : 0.125f;

    const int grid_high = (qh_entry >> (3 * l)) & 7;
    const int grid_idx = (int)q.qs[4 * ib + l] | (grid_high << 8);
    const int8_t *grid_row = iq1s_grid + 8 * grid_idx;

    float val = dl * ((float)grid_row[j] + delta);
    dst[row * n_per_row + block * QK_K + i] = fp32_to_output_fp8<E5M2>(val);
}

template <bool E5M2>
__global__ __launch_bounds__(256, 4)
void dequant_iq2_s_to_fp8_kernel(
    const block_iq2_s * __restrict__ src, uint8_t * __restrict__ dst,
    const int8_t * __restrict__ iq2s_grid,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int row = (int)blockIdx.x + (int)blockIdx.z * (int)gridDim.x;
    const int block = blockIdx.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= QK_K) return;
    const block_iq2_s q = src[row * blocks_per_row + block];

    const int ib = i / 32;
    const int l = (i % 32) / 8;
    const int j = i % 8;

    const float d = fp16_to_fp32(q.d);
    const float db0 = d * (0.5f + (float)(q.scales[ib] & 0x0f)) * 0.25f;
    const float db1 = d * (0.5f + (float)(q.scales[ib] >> 4)) * 0.25f;
    const float dl = l < 2 ? db0 : db1;

    const int grid_idx = (int)q.qs[4 * ib + l] | ((int)(q.qh[ib] << (8 - 2 * l)) & 0x300);
    const int8_t *grid_row = iq2s_grid + 8 * grid_idx;

    const uint8_t sign_byte = q.qs[32 + 4 * ib + l];
    const float sign = (sign_byte & d_kmask_iq2xs[j]) ? -1.0f : 1.0f;

    float val = dl * (float)grid_row[j] * sign;
    dst[row * n_per_row + block * QK_K + i] = fp32_to_output_fp8<E5M2>(val);
}

template <bool E5M2>
__global__ __launch_bounds__(256, 4)
void dequant_iq1_m_to_fp8_kernel(
    const block_iq1_m * __restrict__ src, uint8_t * __restrict__ dst,
    const int8_t * __restrict__ iq1s_grid,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int row = (int)blockIdx.x + (int)blockIdx.z * (int)gridDim.x;
    const int block = blockIdx.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= QK_K) return;
    const block_iq1_m q = src[row * blocks_per_row + block];

    const int ib = i / 32;
    const int l = (i % 32) / 8;
    const int j = i % 8;

    const uint16_t * sc = (const uint16_t *)q.scales;
    const uint16_t scale_u16 =
        (uint16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
    const float d = fp16_to_fp32(scale_u16);

    const int scw = sc[ib / 2];
    const int lvl0 = (scw >> (6 * (ib % 2) + 0)) & 7;
    const int lvl1 = (scw >> (6 * (ib % 2) + 3)) & 7;
    const float dl = l < 2 ? d * (float)(2 * lvl0 + 1) : d * (float)(2 * lvl1 + 1);

    const int qh_byte = q.qh[2 * ib + l / 2];
    const int grid_idx = (int)q.qs[4 * ib + l] | ((qh_byte << (8 - 4 * (l % 2))) & 0x700);
    const int8_t *grid_row = iq1s_grid + 8 * grid_idx;

    const float delta = (qh_byte & (l % 2 == 0 ? 0x08 : 0x80)) ? -0.125f : 0.125f;

    float val = dl * ((float)grid_row[j] + delta);
    dst[row * n_per_row + block * QK_K + i] = fp32_to_output_fp8<E5M2>(val);
}

template <bool E5M2>
__global__ __launch_bounds__(256, 4)
void dequant_iq3_s_to_fp8_kernel(
    const block_iq3_s * __restrict__ src, uint8_t * __restrict__ dst,
    const int8_t * __restrict__ grid,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int row = (int)blockIdx.x + (int)blockIdx.z * (int)gridDim.x;
    const int block = blockIdx.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= QK_K) return;
    const block_iq3_s q = src[row * blocks_per_row + block];

    const int ib32 = i / 32;
    const int l = (i % 32) / 8;
    const int j = i % 8;

    const int pair = ib32 / 2;
    const int is_first = (ib32 % 2) == 0;
    const float db = fp16_to_fp32(q.d) * (float)(1 + 2 * (
        is_first ? (q.scales[pair] & 0xf) : (q.scales[pair] >> 4)));
    const int qh_byte = q.qh[ib32];
    const int grid_idx = (j < 4)
        ? (int)(q.qs[8 * ib32 + 2 * l + 0] | (((qh_byte << (8 - 2 * l)) & 256)))
        : (int)(q.qs[8 * ib32 + 2 * l + 1] | (((qh_byte << (7 - 2 * l)) & 256)));
    const int grid_pos = j & 3;
    const uint8_t signs_byte = q.signs[4 * ib32 + l];

    const uint8_t *grid_row = (const uint8_t *)(grid + grid_idx);
    float val = db * (float)((int8_t)grid_row[grid_pos]);
    dst[row * n_per_row + block * QK_K + i] = fp32_to_output_fp8<E5M2>(
        (signs_byte & d_kmask_iq2xs[j]) ? -val : val);
}

// =========================================================================
// T-Quant dequant-to-FP8 kernels (GGML reference formulas)
// =========================================================================

template <bool E5M2>
__global__ __launch_bounds__(256, 4)
void dequant_tq1_0_to_fp8_kernel(
    const block_tq1_0 * __restrict__ src, uint8_t * __restrict__ dst,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int row = (int)blockIdx.x + (int)blockIdx.z * (int)gridDim.x;
    const int block = blockIdx.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= QK_K) return;
    const block_tq1_0 q = src[row * blocks_per_row + block];

    const float d = fp16_to_fp32(q.d);
    // Unpack ternary values from 5-per-byte (qs) and 4-per-byte (qh) encoding
    // qs stores 240 values (48 bytes × 5), qh stores remaining 16 (4 bytes × 4)
    // Base-3 unpacking: digit = (byte * 3^n) * 3 / 256
    float val;
    if (i < 240) {
        const int byte_idx = i / 5;
        const int n = i % 5;
        const uint8_t qb = q.qs[byte_idx];
        const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};
        int16_t xi = (int16_t)(((uint16_t)qb * pow3[n]) * 3) >> 8;
        val = (float)(xi - 1) * d;
    } else {
        const int idx = i - 240;
        const int byte_idx = idx / 4;
        const int n = idx % 4;
        const uint8_t qb = q.qh[byte_idx];
        const uint8_t pow3[5] = {1, 3, 9, 27, 81};
        int16_t xi = (int16_t)(((uint16_t)qb * pow3[n]) * 3) >> 8;
        val = (float)(xi - 1) * d;
    }
    dst[row * n_per_row + block * QK_K + i] = fp32_to_output_fp8<E5M2>(val);
}

template <bool E5M2>
__global__ __launch_bounds__(256, 4)
void dequant_tq2_0_to_fp8_kernel(
    const block_tq2_0 * __restrict__ src, uint8_t * __restrict__ dst,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int row = (int)blockIdx.x + (int)blockIdx.z * (int)gridDim.x;
    const int block = blockIdx.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= QK_K) return;
    const block_tq2_0 q = src[row * blocks_per_row + block];

    const float d = fp16_to_fp32(q.d);
    const int byte_idx = i / 4;
    const int l = i % 4;
    const int8_t qi = (q.qs[byte_idx] >> (2 * l)) & 3;

    float val = (float)(qi - 1) * d;
    dst[row * n_per_row + block * QK_K + i] = fp32_to_output_fp8<E5M2>(val);
}

// =========================================================================
// HQ2 dequant-to-FP8 (TurboQuant-style learned-codebook 2-bit weight quant)
// =========================================================================

template <bool E5M2>
__global__ __launch_bounds__(256, 4)
void dequant_hq2_to_fp8_kernel(
    const block_hq2 * __restrict__ src, uint8_t * __restrict__ dst,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int row = (int)blockIdx.x + (int)blockIdx.z * (int)gridDim.x;
    const int block = blockIdx.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= HQ2_K) return;
    const block_hq2 * q = src + row * blocks_per_row + block;

    const int c = (q->qs[i >> 2] >> (2 * (i & 3))) & 3;
    const float val = fp16_to_fp32(q->levels[c]);
    dst[row * n_per_row + block * HQ2_K + i] = fp32_to_output_fp8<E5M2>(val);
}
