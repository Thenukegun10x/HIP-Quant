// torch_ext/dequant_q_to_fp16.hip
//
// High-throughput GPU-resident GGML Q-type and I-quant -> FP16 dequantization kernels.
// Optimized for zero VGPR spilling, on-demand field loads, and 100% coalesced 16-bit stores.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include "../hip_quant_types.h"
#include "../hip_quant_util.h"
#include "iq_grid_tables.h"

#define GGML_TYPE_Q4_0 2
#define GGML_TYPE_Q4_1 3
#define GGML_TYPE_Q5_0 6
#define GGML_TYPE_Q5_1 7
#define GGML_TYPE_Q8_0 8
#define GGML_TYPE_Q8_1 9
#define GGML_TYPE_Q2_K 10
#define GGML_TYPE_Q3_K 11
#define GGML_TYPE_Q4_K 12
#define GGML_TYPE_Q5_K 13
#define GGML_TYPE_Q6_K 14
#define GGML_TYPE_IQ2_XXS 16
#define GGML_TYPE_IQ2_XS 17
#define GGML_TYPE_IQ3_XXS 18
#define GGML_TYPE_IQ1_S 19
#define GGML_TYPE_IQ4_NL 20
#define GGML_TYPE_IQ3_S 21
#define GGML_TYPE_IQ2_S 22
#define GGML_TYPE_IQ4_XS 23
#define GGML_TYPE_IQ1_M 29

namespace {

__device__ inline uint8_t unpack_k4_scale(const uint8_t *scales, int group) {
    if (group < 4) return scales[group] & 63;
    return (scales[group + 4] & 0x0f) | ((scales[group - 4] >> 6) << 4);
}

__device__ inline uint8_t unpack_k4_min(const uint8_t *scales, int group) {
    if (group < 4) return scales[group + 4] & 63;
    return (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4);
}

// -------------------------------------------------------------------------
// 32-element blocks
// -------------------------------------------------------------------------

__global__ __launch_bounds__(32, 8)
void dequant_q4_0_to_fp16_kernel(
    const block_q4_0 * __restrict__ src, half * __restrict__ dst,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int block = blockIdx.x;
    const int row = (int)blockIdx.y + (int)blockIdx.z * (int)gridDim.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= 32) return;
    const block_q4_0* q = &src[row * blocks_per_row + block];
    const uint8_t nibble = i < 16 ? (q->qs[i] & 0x0f) : (q->qs[i - 16] >> 4);
    float val = fp16_to_fp32(q->d) * ((int)nibble - 8);
    dst[row * n_per_row + block * 32 + i] = __float2half(val);
}

__global__ __launch_bounds__(32, 8)
void dequant_q4_1_to_fp16_kernel(
    const block_q4_1 * __restrict__ src, half * __restrict__ dst,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int block = blockIdx.x;
    const int row = (int)blockIdx.y + (int)blockIdx.z * (int)gridDim.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= 32) return;
    const block_q4_1* q = &src[row * blocks_per_row + block];
    const uint8_t nibble = i < 16 ? (q->qs[i] & 0x0f) : (q->qs[i - 16] >> 4);
    float val = fp16_to_fp32(q->d) * (float)nibble + fp16_to_fp32(q->m);
    dst[row * n_per_row + block * 32 + i] = __float2half(val);
}

__global__ __launch_bounds__(32, 8)
void dequant_q5_0_to_fp16_kernel(
    const block_q5_0 * __restrict__ src, half * __restrict__ dst,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int block = blockIdx.x;
    const int row = (int)blockIdx.y + (int)blockIdx.z * (int)gridDim.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= 32) return;
    const block_q5_0* q = &src[row * blocks_per_row + block];
    const uint8_t low = i < 16 ? (q->qs[i] & 0x0f) : (q->qs[i - 16] >> 4);
    const int value = (int)low | (((q->qh[i >> 3] >> (i & 7)) & 1) << 4);
    float val = fp16_to_fp32(q->d) * (float)(value - 16);
    dst[row * n_per_row + block * 32 + i] = __float2half(val);
}

__global__ __launch_bounds__(32, 8)
void dequant_q5_1_to_fp16_kernel(
    const block_q5_1 * __restrict__ src, half * __restrict__ dst,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int block = blockIdx.x;
    const int row = (int)blockIdx.y + (int)blockIdx.z * (int)gridDim.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= 32) return;
    const block_q5_1* q = &src[row * blocks_per_row + block];
    const uint8_t low = i < 16 ? (q->qs[i] & 0x0f) : (q->qs[i - 16] >> 4);
    const int value = (int)low | (((q->qh[i >> 3] >> (i & 7)) & 1) << 4);
    float val = fp16_to_fp32(q->d) * (float)value + fp16_to_fp32(q->m);
    dst[row * n_per_row + block * 32 + i] = __float2half(val);
}

__global__ __launch_bounds__(32, 8)
void dequant_q8_0_to_fp16_kernel(
    const block_q8_0 * __restrict__ src, half * __restrict__ dst,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int block = blockIdx.x;
    const int row = (int)blockIdx.y + (int)blockIdx.z * (int)gridDim.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= 32) return;
    const block_q8_0* q = &src[row * blocks_per_row + block];
    float val = fp16_to_fp32(q->d) * (float)q->qs[i];
    dst[row * n_per_row + block * 32 + i] = __float2half(val);
}

__global__ __launch_bounds__(32, 8)
void dequant_q8_1_to_fp16_kernel(
    const block_q8_1 * __restrict__ src, half * __restrict__ dst,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int block = blockIdx.x;
    const int row = (int)blockIdx.y + (int)blockIdx.z * (int)gridDim.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= 32) return;
    const block_q8_1* q = &src[row * blocks_per_row + block];
    float val = fp16_to_fp32(q->d) * (float)q->qs[i];
    dst[row * n_per_row + block * 32 + i] = __float2half(val);
}

__global__ __launch_bounds__(32, 8)
void dequant_iq4_nl_to_fp16_kernel(
    const block_iq4_nl * __restrict__ src, half * __restrict__ dst,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int block = blockIdx.x;
    const int row = (int)blockIdx.y + (int)blockIdx.z * (int)gridDim.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= 32) return;
    const block_iq4_nl* q = &src[row * blocks_per_row + block];
    const uint8_t nibble = i < 16 ? (q->qs[i] & 0x0f) : (q->qs[i - 16] >> 4);
    float val = fp16_to_fp32(q->d) * (float)d_kvalues_iq4nl[nibble];
    dst[row * n_per_row + block * 32 + i] = __float2half(val);
}

// -------------------------------------------------------------------------
// 256-element blocks (K-quants)
// -------------------------------------------------------------------------

__global__ __launch_bounds__(256, 4)
void dequant_q2_k_to_fp16_kernel(
    const block_q2_K * __restrict__ src, half * __restrict__ dst,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int block = blockIdx.x;
    const int row = (int)blockIdx.y + (int)blockIdx.z * (int)gridDim.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= 256) return;
    const block_q2_K* q = &src[row * blocks_per_row + block];
    const int segment = (i & 127) >> 5;
    const int packed = (i >> 7) * 32 + (i & 31);
    const int value = (q->qs[packed] >> (2 * segment)) & 3;
    const uint8_t scale_min = q->scales[i >> 4];
    const float scale = fp16_to_fp32(q->d) * (float)(scale_min & 0x0f);
    const float min = fp16_to_fp32(q->dmin) * (float)(scale_min >> 4);
    float val = scale * (float)value - min;
    dst[row * n_per_row + block * 256 + i] = __float2half(val);
}

__global__ __launch_bounds__(256, 4)
void dequant_q3_k_to_fp16_kernel(
    const block_q3_K * __restrict__ src, half * __restrict__ dst,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int block = blockIdx.x;
    const int row = (int)blockIdx.y + (int)blockIdx.z * (int)gridDim.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= 256) return;
    const block_q3_K* q = &src[row * blocks_per_row + block];
    const int segment = (i & 127) >> 5;
    const int packed = (i >> 7) * 32 + (i & 31);
    int value = (q->qs[packed] >> (2 * segment)) & 3;
    if (q->hmask[i & 31] & (1u << (i >> 5))) value += 4;
    value -= 4;
    const int group = i >> 4;
    const uint8_t low = group < 8 ? (q->scales[group] & 0x0f) : (q->scales[group - 8] >> 4);
    const uint8_t high = (q->scales[8 + (group & 3)] >> (2 * (group >> 2))) & 3;
    const int scale = ((int)high << 4 | low) - 32;
    float val = fp16_to_fp32(q->d) * (float)scale * (float)value;
    dst[row * n_per_row + block * 256 + i] = __float2half(val);
}

__global__ __launch_bounds__(256, 4)
void dequant_q4_k_to_fp16_kernel(
    const block_q4_K * __restrict__ src, half * __restrict__ dst,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int block = blockIdx.x;
    const int row = (int)blockIdx.y + (int)blockIdx.z * (int)gridDim.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= 256) return;
    const block_q4_K* q = &src[row * blocks_per_row + block];
    const int group = i >> 5;
    const int packed = (i >> 6) * 32 + (i & 31);
    const uint8_t value = (q->qs[packed] >> (((i >> 5) & 1) * 4)) & 0x0f;
    const float scale = fp16_to_fp32(q->d) * (float)unpack_k4_scale(q->scales, group);
    const float min = fp16_to_fp32(q->dmin) * (float)unpack_k4_min(q->scales, group);
    float val = scale * (float)value - min;
    dst[row * n_per_row + block * 256 + i] = __float2half(val);
}

__global__ __launch_bounds__(256, 4)
void dequant_q5_k_to_fp16_kernel(
    const block_q5_K * __restrict__ src, half * __restrict__ dst,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int block = blockIdx.x;
    const int row = (int)blockIdx.y + (int)blockIdx.z * (int)gridDim.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= 256) return;
    const block_q5_K* q = &src[row * blocks_per_row + block];
    const int group = i >> 5;
    const int chunk = i >> 6;
    const int lane = i & 31;
    const int half_idx = (i >> 5) & 1;
    const int packed = chunk * 32 + lane;
    const int high = (q->qh[lane] >> (chunk * 2 + half_idx)) & 1;
    const int value = ((q->qs[packed] >> (half_idx * 4)) & 0x0f) | (high << 4);
    const float scale = fp16_to_fp32(q->d) * (float)unpack_k4_scale(q->scales, group);
    const float min = fp16_to_fp32(q->dmin) * (float)unpack_k4_min(q->scales, group);
    float val = scale * (float)value - min;
    dst[row * n_per_row + block * 256 + i] = __float2half(val);
}

__global__ __launch_bounds__(256, 4)
void dequant_q6_k_to_fp16_kernel(
    const block_q6_K * __restrict__ src, half * __restrict__ dst,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int block = blockIdx.x;
    const int row = (int)blockIdx.y + (int)blockIdx.z * (int)gridDim.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= 256) return;
    const block_q6_K* q = &src[row * blocks_per_row + block];
    const int chunk = i >> 7;
    const int section = (i & 127) >> 5;
    const int lane = i & 31;
    const int ql_index = chunk * 64 + (section & 1 ? 32 : 0) + lane;
    const int low = (q->ql[ql_index] >> ((section >> 1) * 4)) & 0x0f;
    const int high = (q->qh[chunk * 32 + lane] >> (section * 2)) & 3;
    const int value = (low | (high << 4)) - 32;
    const float scale = fp16_to_fp32(q->d) * (float)q->scales[i >> 4];
    float val = scale * (float)value;
    dst[row * n_per_row + block * 256 + i] = __float2half(val);
}

// -------------------------------------------------------------------------
// 256-element blocks (I-quants)
// -------------------------------------------------------------------------

__global__ __launch_bounds__(256, 4)
void dequant_iq1_s_to_fp16_kernel(
    const block_iq1_s * __restrict__ src, half * __restrict__ dst,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int block = blockIdx.x;
    const int row = (int)blockIdx.y + (int)blockIdx.z * (int)gridDim.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= 256) return;
    const block_iq1_s* q = &src[row * blocks_per_row + block];

    const int ib = i / 32;
    const int l = (i % 32) / 8;
    const int j = i % 8;

    const float d = fp16_to_fp32(q->d);
    const uint16_t qh_entry = q->qh[ib];
    const float dl = d * (float)(2 * ((qh_entry >> 12) & 7) + 1);
    const float delta = (qh_entry & 0x8000) ? -0.125f : 0.125f;

    const int grid_high = (qh_entry >> (3 * l)) & 7;
    const int grid_idx = (int)q->qs[4 * ib + l] | (grid_high << 8);
    const uint64_t grid_u64 = d_iq1s_grid[grid_idx];
    const int8_t grid_val = (int8_t)((grid_u64 >> (j * 8)) & 0xff);

    float val = dl * ((float)grid_val + delta);
    dst[row * n_per_row + block * 256 + i] = __float2half(val);
}

__global__ __launch_bounds__(256, 4)
void dequant_iq1_m_to_fp16_kernel(
    const block_iq1_m * __restrict__ src, half * __restrict__ dst,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int block = blockIdx.x;
    const int row = (int)blockIdx.y + (int)blockIdx.z * (int)gridDim.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= 256) return;
    const block_iq1_m* q = &src[row * blocks_per_row + block];

    const int ib = i / 32;
    const int l = (i % 32) / 8;
    const int j = i % 8;

    const uint16_t * sc = (const uint16_t *)q->scales;
    const uint16_t scale_u16 =
        (uint16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
    const float d = fp16_to_fp32(scale_u16);

    const int scw = sc[ib / 2];
    const int lvl0 = (scw >> (6 * (ib % 2) + 0)) & 7;
    const int lvl1 = (scw >> (6 * (ib % 2) + 3)) & 7;
    const float dl = l < 2 ? d * (float)(2 * lvl0 + 1) : d * (float)(2 * lvl1 + 1);

    const int qh_byte = q->qh[2 * ib + l / 2];
    const int grid_idx = (int)q->qs[4 * ib + l] | ((qh_byte << (8 - 4 * (l % 2))) & 0x700);
    const uint64_t grid_u64 = d_iq1s_grid[grid_idx];
    const int8_t grid_val = (int8_t)((grid_u64 >> (j * 8)) & 0xff);

    const float delta = (qh_byte & (l % 2 == 0 ? 0x08 : 0x80)) ? -0.125f : 0.125f;
    float val = dl * ((float)grid_val + delta);
    dst[row * n_per_row + block * 256 + i] = __float2half(val);
}

__global__ __launch_bounds__(256, 4)
void dequant_iq2_xs_to_fp16_kernel(
    const block_iq2_xs * __restrict__ src, half * __restrict__ dst,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int block = blockIdx.x;
    const int row = (int)blockIdx.y + (int)blockIdx.z * (int)gridDim.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= 256) return;
    const block_iq2_xs* q = &src[row * blocks_per_row + block];

    const int ib32 = i / 32;
    const int l = (i % 32) / 8;
    const int j = i % 8;

    const uint16_t qs_entry = q->qs[4 * ib32 + l];
    const uint8_t signs = d_ksigns_iq2xs[qs_entry >> 9];

    const int nibble = q->scales[ib32];
    const float d = fp16_to_fp32(q->d);
    const float db_l = d * (0.5f + (float)(nibble & 0xf)) * 0.25f;
    const float db_h = d * (0.5f + (float)(nibble >> 4)) * 0.25f;
    const float db = (l < 2) ? db_l : db_h;

    uint64_t grid_u64 = d_iq2xs_grid[qs_entry & 511];
    int8_t grid_val = (int8_t)((grid_u64 >> (j * 8)) & 0xff);
    float sign = ((signs >> j) & 1) ? -db : db;
    float val = sign * (float)grid_val;
    dst[row * n_per_row + block * 256 + i] = __float2half(val);
}

__global__ __launch_bounds__(256, 4)
void dequant_iq2_xxs_to_fp16_kernel(
    const block_iq2_xxs * __restrict__ src, half * __restrict__ dst,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int block = blockIdx.x;
    const int row = (int)blockIdx.y + (int)blockIdx.z * (int)gridDim.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= 256) return;
    const block_iq2_xxs* q = &src[row * blocks_per_row + block];

    const int ib32 = i / 32;
    const int l = (i % 32) / 8;
    const int j = i % 8;

    const uint8_t *aux8 = (const uint8_t *)(q->qs + 4 * ib32);
    const uint32_t aux32_1 = ((const uint32_t *)(q->qs + 4 * ib32))[1];
    const uint8_t signs = d_ksigns_iq2xs[(aux32_1 >> (7 * l)) & 127];
    const float db = fp16_to_fp32(q->d) * (0.5f + (float)(aux32_1 >> 28)) * 0.25f;

    uint64_t grid_u64 = d_iq2xxs_grid[aux8[l]];
    int8_t grid_val = (int8_t)((grid_u64 >> (j * 8)) & 0xff);
    float sign = ((signs >> j) & 1) ? -db : db;
    float val = sign * (float)grid_val;
    dst[row * n_per_row + block * 256 + i] = __float2half(val);
}

__global__ __launch_bounds__(256, 4)
void dequant_iq2_s_to_fp16_kernel(
    const block_iq2_s * __restrict__ src, half * __restrict__ dst,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int block = blockIdx.x;
    const int row = (int)blockIdx.y + (int)blockIdx.z * (int)gridDim.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= 256) return;
    const block_iq2_s* q = &src[row * blocks_per_row + block];

    const int ib = i / 32;
    const int l = (i % 32) / 8;
    const int j = i % 8;

    const float d = fp16_to_fp32(q->d);
    const float db0 = d * (0.5f + (float)(q->scales[ib] & 0x0f)) * 0.25f;
    const float db1 = d * (0.5f + (float)(q->scales[ib] >> 4)) * 0.25f;
    const float dl = l < 2 ? db0 : db1;

    const int grid_idx = (int)q->qs[4 * ib + l] | ((int)(q->qh[ib] << (8 - 2 * l)) & 0x300);
    uint64_t grid_u64 = d_iq2s_grid[grid_idx];
    int8_t grid_val = (int8_t)((grid_u64 >> (j * 8)) & 0xff);

    const uint8_t sign_byte = q->qs[32 + 4 * ib + l];
    float sign = ((sign_byte >> j) & 1) ? -dl : dl;
    float val = sign * (float)grid_val;
    dst[row * n_per_row + block * 256 + i] = __float2half(val);
}

__global__ __launch_bounds__(256, 4)
void dequant_iq3_xxs_to_fp16_kernel(
    const block_iq3_xxs * __restrict__ src, half * __restrict__ dst,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int block = blockIdx.x;
    const int row = (int)blockIdx.y + (int)blockIdx.z * (int)gridDim.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= 256) return;
    const block_iq3_xxs* q = &src[row * blocks_per_row + block];

    const int ib32 = i / 32;
    const int l = (i % 32) / 8;
    const int j = i % 8;

    const uint8_t *grid_qs = q->qs + 8 * ib32;
    const uint8_t *scales_and_signs = q->qs + 256 / 4;
    const uint32_t aux32 = ((const uint32_t *)scales_and_signs)[ib32];
    const float db = fp16_to_fp32(q->d) * (0.5f + (float)(aux32 >> 28)) * 0.5f;
    const uint8_t signs = d_ksigns_iq2xs[(aux32 >> (7 * l)) & 127];

    const int grid_idx = (j < 4)
        ? (int)grid_qs[2 * l + 0]
        : (int)grid_qs[2 * l + 1];
    const int grid_pos = j & 3;

    uint32_t grid_u32 = d_iq3xxs_grid[grid_idx];
    int8_t grid_val = (int8_t)((grid_u32 >> (grid_pos * 8)) & 0xff);
    float sign = ((signs >> j) & 1) ? -db : db;
    float val = sign * (float)grid_val;
    dst[row * n_per_row + block * 256 + i] = __float2half(val);
}

__global__ __launch_bounds__(256, 4)
void dequant_iq3_s_to_fp16_kernel(
    const block_iq3_s * __restrict__ src, half * __restrict__ dst,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int block = blockIdx.x;
    const int row = (int)blockIdx.y + (int)blockIdx.z * (int)gridDim.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= 256) return;
    const block_iq3_s* q = &src[row * blocks_per_row + block];

    const int ib32 = i / 32;
    const int l = (i % 32) / 8;
    const int j = i % 8;

    const int pair = ib32 / 2;
    const int is_first = (ib32 % 2) == 0;
    const float db = fp16_to_fp32(q->d) * (float)(1 + 2 * (
        is_first ? (q->scales[pair] & 0xf) : (q->scales[pair] >> 4)));
    const int qh_byte = q->qh[ib32];
    const int grid_idx = (j < 4)
        ? (int)(q->qs[8 * ib32 + 2 * l + 0] | (((qh_byte << (8 - 2 * l)) & 256)))
        : (int)(q->qs[8 * ib32 + 2 * l + 1] | (((qh_byte << (7 - 2 * l)) & 256)));
    const int grid_pos = j & 3;
    const uint8_t signs_byte = q->signs[4 * ib32 + l];

    uint32_t grid_u32 = d_iq3s_grid[grid_idx];
    int8_t grid_val = (int8_t)((grid_u32 >> (grid_pos * 8)) & 0xff);
    float sign = ((signs_byte >> j) & 1) ? -db : db;
    float val = sign * (float)grid_val;
    dst[row * n_per_row + block * 256 + i] = __float2half(val);
}

__global__ __launch_bounds__(256, 4)
void dequant_iq4_xs_to_fp16_kernel(
    const block_iq4_xs * __restrict__ src, half * __restrict__ dst,
    int nrows, int blocks_per_row, int n_per_row
) {
    const int block = blockIdx.x;
    const int row = (int)blockIdx.y + (int)blockIdx.z * (int)gridDim.y;
    const int i = threadIdx.x;
    if (row >= nrows || block >= blocks_per_row || i >= 256) return;
    const block_iq4_xs* q = &src[row * blocks_per_row + block];

    const int ib = i / 32;          // sub-block (0..7)
    const int pos_in_sub = i % 32;  // position (0..31)
    const int nibble = (pos_in_sub < 16)
        ? (q->qs[16 * ib + pos_in_sub] & 0x0f)
        : (q->qs[16 * ib + (pos_in_sub - 16)] >> 4);

    const int ls_low = (q->scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf;
    const int ls_high = (q->scales_h >> (2 * ib)) & 3;
    const int ls = ls_low | (ls_high << 4);
    const float dl = fp16_to_fp32(q->d) * (float)(ls - 32);

    float val = dl * (float)d_kvalues_iq4nl[nibble];
    dst[row * n_per_row + block * 256 + i] = __float2half(val);
}

} // anonymous namespace

// -------------------------------------------------------------------------
// C-linkage launcher
// -------------------------------------------------------------------------
extern "C" {

void launch_dequant_q_to_fp16(
    const void* src,
    void* dst,
    int type_num,
    int nrows,
    int blocks_per_row,
    int n_per_row,
    hipStream_t stream
) {
    if (nrows <= 0 || blocks_per_row <= 0 || n_per_row <= 0) return;

    unsigned int y_dim = (unsigned int)nrows;
    unsigned int z_dim = 1;
    if (y_dim > 32768) {
        z_dim = (y_dim + 32767) / 32768;
        y_dim = 32768;
    }
    dim3 grid((unsigned int)blocks_per_row, y_dim, z_dim);
    half* dst_h = (half*)dst;

    switch (type_num) {
        case GGML_TYPE_Q4_0:
            hipLaunchKernelGGL(dequant_q4_0_to_fp16_kernel, grid, 32, 0, stream,
                               (const block_q4_0*)src, dst_h, nrows, blocks_per_row, n_per_row);
            break;
        case GGML_TYPE_Q4_1:
            hipLaunchKernelGGL(dequant_q4_1_to_fp16_kernel, grid, 32, 0, stream,
                               (const block_q4_1*)src, dst_h, nrows, blocks_per_row, n_per_row);
            break;
        case GGML_TYPE_Q5_0:
            hipLaunchKernelGGL(dequant_q5_0_to_fp16_kernel, grid, 32, 0, stream,
                               (const block_q5_0*)src, dst_h, nrows, blocks_per_row, n_per_row);
            break;
        case GGML_TYPE_Q5_1:
            hipLaunchKernelGGL(dequant_q5_1_to_fp16_kernel, grid, 32, 0, stream,
                               (const block_q5_1*)src, dst_h, nrows, blocks_per_row, n_per_row);
            break;
        case GGML_TYPE_Q8_0:
            hipLaunchKernelGGL(dequant_q8_0_to_fp16_kernel, grid, 32, 0, stream,
                               (const block_q8_0*)src, dst_h, nrows, blocks_per_row, n_per_row);
            break;
        case GGML_TYPE_Q8_1:
            hipLaunchKernelGGL(dequant_q8_1_to_fp16_kernel, grid, 32, 0, stream,
                               (const block_q8_1*)src, dst_h, nrows, blocks_per_row, n_per_row);
            break;
        case GGML_TYPE_IQ4_NL:
            hipLaunchKernelGGL(dequant_iq4_nl_to_fp16_kernel, grid, 32, 0, stream,
                               (const block_iq4_nl*)src, dst_h, nrows, blocks_per_row, n_per_row);
            break;
        case GGML_TYPE_Q2_K:
            hipLaunchKernelGGL(dequant_q2_k_to_fp16_kernel, grid, 256, 0, stream,
                               (const block_q2_K*)src, dst_h, nrows, blocks_per_row, n_per_row);
            break;
        case GGML_TYPE_Q3_K:
            hipLaunchKernelGGL(dequant_q3_k_to_fp16_kernel, grid, 256, 0, stream,
                               (const block_q3_K*)src, dst_h, nrows, blocks_per_row, n_per_row);
            break;
        case GGML_TYPE_Q4_K:
            hipLaunchKernelGGL(dequant_q4_k_to_fp16_kernel, grid, 256, 0, stream,
                               (const block_q4_K*)src, dst_h, nrows, blocks_per_row, n_per_row);
            break;
        case GGML_TYPE_Q5_K:
            hipLaunchKernelGGL(dequant_q5_k_to_fp16_kernel, grid, 256, 0, stream,
                               (const block_q5_K*)src, dst_h, nrows, blocks_per_row, n_per_row);
            break;
        case GGML_TYPE_Q6_K:
            hipLaunchKernelGGL(dequant_q6_k_to_fp16_kernel, grid, 256, 0, stream,
                               (const block_q6_K*)src, dst_h, nrows, blocks_per_row, n_per_row);
            break;
        case GGML_TYPE_IQ1_S:
            hipLaunchKernelGGL(dequant_iq1_s_to_fp16_kernel, grid, 256, 0, stream,
                               (const block_iq1_s*)src, dst_h, nrows, blocks_per_row, n_per_row);
            break;
        case GGML_TYPE_IQ1_M:
            hipLaunchKernelGGL(dequant_iq1_m_to_fp16_kernel, grid, 256, 0, stream,
                               (const block_iq1_m*)src, dst_h, nrows, blocks_per_row, n_per_row);
            break;
        case GGML_TYPE_IQ2_XS:
            hipLaunchKernelGGL(dequant_iq2_xs_to_fp16_kernel, grid, 256, 0, stream,
                               (const block_iq2_xs*)src, dst_h, nrows, blocks_per_row, n_per_row);
            break;
        case GGML_TYPE_IQ2_XXS:
            hipLaunchKernelGGL(dequant_iq2_xxs_to_fp16_kernel, grid, 256, 0, stream,
                               (const block_iq2_xxs*)src, dst_h, nrows, blocks_per_row, n_per_row);
            break;
        case GGML_TYPE_IQ2_S:
            hipLaunchKernelGGL(dequant_iq2_s_to_fp16_kernel, grid, 256, 0, stream,
                               (const block_iq2_s*)src, dst_h, nrows, blocks_per_row, n_per_row);
            break;
        case GGML_TYPE_IQ3_XXS:
            hipLaunchKernelGGL(dequant_iq3_xxs_to_fp16_kernel, grid, 256, 0, stream,
                               (const block_iq3_xxs*)src, dst_h, nrows, blocks_per_row, n_per_row);
            break;
        case GGML_TYPE_IQ3_S:
            hipLaunchKernelGGL(dequant_iq3_s_to_fp16_kernel, grid, 256, 0, stream,
                               (const block_iq3_s*)src, dst_h, nrows, blocks_per_row, n_per_row);
            break;
        case GGML_TYPE_IQ4_XS:
            hipLaunchKernelGGL(dequant_iq4_xs_to_fp16_kernel, grid, 256, 0, stream,
                               (const block_iq4_xs*)src, dst_h, nrows, blocks_per_row, n_per_row);
            break;
        default:
            break;
    }
}

} // extern "C"
