// torch_ext/gemv_q_kernels.hip
//
// Dedicated AOT-specialized GEMV (Matrix-Vector) inference kernels for
// single-token decode (M=1) on AMD RDNA 4 (gfx1201 / Wave32).
//
// Features:
// - Direct streaming of native GGUF blocks (Q8_0, Q4_0, Q4_K, Q5_K, Q6_K, IQ3_S, IQ3_XXS) from VRAM.
// - Cooperative LDS caching of activation vector x with automatic global-memory fallback for huge K.
// - 1 Wave (32 threads) per output row N; fast 5-cycle __shfl_down reduction.
// - Zero host-device synchronization, zero shared-memory bank conflicts.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include "../hip_quant_types.h"
#include "../hip_quant_util.h"
#include "iq_grid_tables.h"

#define GGML_TYPE_Q4_0 2
#define GGML_TYPE_Q8_0 8
#define GGML_TYPE_Q2_K 10
#define GGML_TYPE_Q4_K 12
#define GGML_TYPE_Q5_K 13
#define GGML_TYPE_Q6_K 14
#define GGML_TYPE_IQ2_XXS 16
#define GGML_TYPE_IQ2_XS 17
#define GGML_TYPE_IQ3_XXS 18
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
// 32-element blocks (Q8_0, Q4_0)
// -------------------------------------------------------------------------

struct dot_q8_0 {
    static constexpr int BLOCK_SIZE = 32;
    __device__ __forceinline__ float operator()(const block_q8_0& blk, const half* x) const {
        float d = fp16_to_fp32(blk.d);
        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            sum += (float)blk.qs[i] * __half2float(x[i]);
        }
        return sum * d;
    }
};

struct dot_q4_0 {
    static constexpr int BLOCK_SIZE = 32;
    __device__ __forceinline__ float operator()(const block_q4_0& blk, const half* x) const {
        float d = fp16_to_fp32(blk.d);
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            uint8_t byte = blk.qs[i];
            int v0 = (byte & 0x0F) - 8;
            int v1 = (byte >> 4)   - 8;
            sum0 += (float)v0 * __half2float(x[i]);
            sum1 += (float)v1 * __half2float(x[i + 16]);
        }
        return (sum0 + sum1) * d;
    }
};

template <typename BlockType, typename DotOp, int ROWS_PER_BLOCK = 4, int M = 1>
__global__ void gemv_q_kernel(
    const half* __restrict__ x,
    const BlockType* __restrict__ weights,
    half* __restrict__ y,
    const half* __restrict__ bias,
    int N, int K,
    int K_full, int row_stride_blocks,
    bool has_bias,
    bool use_lds
) {
    extern __shared__ half x_shared[];
    const half* x0_vec = x;
    const half* x1_vec = (M == 2) ? (x + K_full) : nullptr;
    if (use_lds) {
        // Chunk-aware staging: x0's slice is contiguous at x, but x1's slice
        // lives at x1_full + start, i.e. x + K_full for every chunk (for an
        // unchunked M=2 GEMV, K_full == K and this is the same copy as before).
        int tid = threadIdx.y * 32 + threadIdx.x;
        int block_threads = 32 * ROWS_PER_BLOCK;
        for (int i = tid; i < K; i += block_threads) {
            x_shared[i] = x[i];
        }
        if constexpr (M == 2) {
            for (int i = tid; i < K; i += block_threads) {
                x_shared[K + i] = x[K_full + i];
            }
        }
        __syncthreads();
        x0_vec = x_shared;
        if constexpr (M == 2) {
            x1_vec = x_shared + K;
        }
    }

    int wave_id = threadIdx.y;
    int lane_id = threadIdx.x;
    int n = blockIdx.x * ROWS_PER_BLOCK + wave_id;
    if (n >= N) return;

    int num_blocks = K / DotOp::BLOCK_SIZE;
    const BlockType* row = weights + (size_t)n * row_stride_blocks;

    DotOp dot_op;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (int b = lane_id; b < num_blocks; b += 32) {
        BlockType blk = row[b];
        sum0 += dot_op(blk, &x0_vec[b * DotOp::BLOCK_SIZE]);
        if constexpr (M == 2) {
            sum1 += dot_op(blk, &x1_vec[b * DotOp::BLOCK_SIZE]);
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum0 += __shfl_down(sum0, offset, 32);
        if constexpr (M == 2) {
            sum1 += __shfl_down(sum1, offset, 32);
        }
    }

    if (lane_id == 0) {
        if (has_bias) {
            float b_val = __half2float(bias[n]);
            sum0 += b_val;
            if constexpr (M == 2) sum1 += b_val;
        }
        y[n] = __float2half(sum0);
        if constexpr (M == 2) {
            y[N + n] = __float2half(sum1);
        }
    }
}

// -------------------------------------------------------------------------
// 256-element super-blocks (Q4_K, Q5_K, Q6_K, IQ3_S, IQ3_XXS)
// -------------------------------------------------------------------------

struct dot_q4_K {
    __device__ __forceinline__ float operator()(const block_q4_K& blk, const half* x_sb, int lane_id) const {
        float d = fp16_to_fp32(blk.d);
        float dmin = fp16_to_fp32(blk.dmin);
        float sc[8], m[8];
        #pragma unroll
        for (int g = 0; g < 8; ++g) {
            sc[g] = d * (float)unpack_k4_scale(blk.scales, g);
            m[g]  = dmin * (float)unpack_k4_min(blk.scales, g);
        }

        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            uint32_t byte = blk.qs[k * 32 + lane_id];
            float v0 = (float)(byte & 0x0f);
            float v1 = (float)(byte >> 4);
            float w0 = fmaf(sc[2 * k + 0], v0, -m[2 * k + 0]);
            float w1 = fmaf(sc[2 * k + 1], v1, -m[2 * k + 1]);
            sum = fmaf(w0, __half2float(x_sb[(2 * k + 0) * 32 + lane_id]), sum);
            sum = fmaf(w1, __half2float(x_sb[(2 * k + 1) * 32 + lane_id]), sum);
        }
        return sum;
    }
};

struct dot_q5_K {
    __device__ __forceinline__ float operator()(const block_q5_K& blk, const half* x_sb, int lane_id) const {
        float d = fp16_to_fp32(blk.d);
        float dmin = fp16_to_fp32(blk.dmin);
        float sc[8], m[8];
        #pragma unroll
        for (int g = 0; g < 8; ++g) {
            sc[g] = d * (float)unpack_k4_scale(blk.scales, g);
            m[g]  = dmin * (float)unpack_k4_min(blk.scales, g);
        }

        uint8_t qh = blk.qh[lane_id];
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            uint8_t byte = blk.qs[k * 32 + lane_id];
            int h0 = (qh >> (2 * k + 0)) & 1;
            int h1 = (qh >> (2 * k + 1)) & 1;
            float v0 = (float)((byte & 0x0f) | (h0 << 4));
            float v1 = (float)((byte >> 4)   | (h1 << 4));
            float w0 = sc[2 * k + 0] * v0 - m[2 * k + 0];
            float w1 = sc[2 * k + 1] * v1 - m[2 * k + 1];
            sum += w0 * __half2float(x_sb[(2 * k + 0) * 32 + lane_id]);
            sum += w1 * __half2float(x_sb[(2 * k + 1) * 32 + lane_id]);
        }
        return sum;
    }
};

struct dot_q6_K {
    __device__ __forceinline__ float operator()(const block_q6_K& blk, const half* x_sb, int lane_id) const {
        float d = fp16_to_fp32(blk.d);
        float sum = 0.0f;
        int lane_hi = (lane_id >= 16) ? 1 : 0;

        #pragma unroll
        for (int chunk = 0; chunk < 2; ++chunk) {
            uint8_t qh = blk.qh[chunk * 32 + lane_id];
            uint8_t ql0 = blk.ql[chunk * 64 + lane_id];
            uint8_t ql1 = blk.ql[chunk * 64 + 32 + lane_id];

            int v0 = (int)((ql0 & 0x0f) | ((qh & 3) << 4)) - 32;
            int v1 = (int)((ql1 & 0x0f) | (((qh >> 2) & 3) << 4)) - 32;
            int v2 = (int)((ql0 >> 4)   | (((qh >> 4) & 3) << 4)) - 32;
            int v3 = (int)((ql1 >> 4)   | (((qh >> 6) & 3) << 4)) - 32;

            float s0 = d * (float)blk.scales[(chunk * 4 + 0) * 2 + lane_hi];
            float s1 = d * (float)blk.scales[(chunk * 4 + 1) * 2 + lane_hi];
            float s2 = d * (float)blk.scales[(chunk * 4 + 2) * 2 + lane_hi];
            float s3 = d * (float)blk.scales[(chunk * 4 + 3) * 2 + lane_hi];

            sum += s0 * (float)v0 * __half2float(x_sb[chunk * 128 + 0 * 32 + lane_id]);
            sum += s1 * (float)v1 * __half2float(x_sb[chunk * 128 + 1 * 32 + lane_id]);
            sum += s2 * (float)v2 * __half2float(x_sb[chunk * 128 + 2 * 32 + lane_id]);
            sum += s3 * (float)v3 * __half2float(x_sb[chunk * 128 + 3 * 32 + lane_id]);
        }
        return sum;
    }
};

struct dot_iq3_s {
    const uint32_t* grid;
    __device__ __forceinline__ dot_iq3_s(const uint32_t* g = d_iq3s_grid) : grid(g) {}
    __device__ __forceinline__ float operator()(const block_iq3_s& blk, const half* x_sb, int lane_id) const {
        float d = fp16_to_fp32(blk.d);
        const int l = lane_id >> 3; // 0..3
        const int j = lane_id & 7;  // 0..7
        const int grid_pos = j & 3;

        float sum = 0.0f;
        #pragma unroll
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            int pair = ib32 >> 1;
            bool is_first = ((ib32 & 1) == 0);
            uint8_t sc_raw = is_first ? (blk.scales[pair] & 0x0f) : (blk.scales[pair] >> 4);
            float db = d * (float)(1 + 2 * sc_raw);

            int qh_byte = (int)blk.qh[ib32];
            int grid_idx = (j < 4)
                ? (int)(blk.qs[8 * ib32 + 2 * l + 0] | (((qh_byte << (8 - 2 * l)) & 256)))
                : (int)(blk.qs[8 * ib32 + 2 * l + 1] | (((qh_byte << (7 - 2 * l)) & 256)));

            uint32_t grid_u32 = grid[grid_idx];
            int8_t grid_val = (int8_t)((grid_u32 >> (grid_pos * 8)) & 0xff);
            uint8_t signs_byte = blk.signs[4 * ib32 + l];
            float sign = ((signs_byte >> j) & 1) ? -db : db;
            float w = sign * (float)grid_val;

            sum = fmaf(w, __half2float(x_sb[ib32 * 32 + lane_id]), sum);
        }
        return sum;
    }
};

struct dot_iq3_xxs {
    const uint32_t* grid;
    __device__ __forceinline__ dot_iq3_xxs(const uint32_t* g = d_iq3xxs_grid) : grid(g) {}
    __device__ __forceinline__ float operator()(const block_iq3_xxs& blk, const half* x_sb, int lane_id) const {
        float d = fp16_to_fp32(blk.d);
        const int l = lane_id >> 3; // 0..3
        const int j = lane_id & 7;  // 0..7
        const int grid_pos = j & 3;

        const uint8_t *scales_and_signs = blk.qs + QK_K / 4; // offset 64
        const uint32_t *aux32_ptr = (const uint32_t *)scales_and_signs;

        float sum = 0.0f;
        #pragma unroll
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            uint32_t aux32 = aux32_ptr[ib32];
            float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
            uint8_t signs = d_ksigns_iq2xs[(aux32 >> (7 * l)) & 127];

            const uint8_t *grid_qs = blk.qs + 8 * ib32;
            int grid_idx = (j < 4)
                ? (int)grid_qs[2 * l + 0]
                : (int)grid_qs[2 * l + 1];

            uint32_t grid_u32 = grid[grid_idx];
            int8_t grid_val = (int8_t)((grid_u32 >> (grid_pos * 8)) & 0xff);
            float sign = ((signs >> j) & 1) ? -db : db;
            float w = sign * (float)grid_val;

            sum = fmaf(w, __half2float(x_sb[ib32 * 32 + lane_id]), sum);
        }
        return sum;
    }
};

struct dot_q2_K {
    __device__ __forceinline__ float operator()(const block_q2_K& blk, const half* x_sb, int lane_id) const {
        float d = fp16_to_fp32(blk.d);
        float dmin = fp16_to_fp32(blk.dmin);

        float sum = 0.0f;
        #pragma unroll
        for (int ib = 0; ib < 8; ++ib) {
            int segment = ib & 3;
            int packed = (ib >> 2) * 32 + lane_id;
            int value = (blk.qs[packed] >> (2 * segment)) & 3;

            int lane_hi = (lane_id >= 16) ? 1 : 0;
            uint8_t scale_min = blk.scales[ib * 2 + lane_hi];
            float scale = d * (float)(scale_min & 0x0f);
            float min = dmin * (float)(scale_min >> 4);

            float w = scale * (float)value - min;
            sum += w * __half2float(x_sb[ib * 32 + lane_id]);
        }
        return sum;
    }
};

struct dot_iq4_xs {
    __device__ __forceinline__ float operator()(const block_iq4_xs& blk, const half* x_sb, int lane_id) const {
        float d = fp16_to_fp32(blk.d);
        float sum = 0.0f;

        #pragma unroll
        for (int ib = 0; ib < 8; ++ib) {
            int nibble = (lane_id < 16)
                ? (blk.qs[16 * ib + lane_id] & 0x0f)
                : (blk.qs[16 * ib + (lane_id - 16)] >> 4);

            int ls_low = (blk.scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf;
            int ls_high = (blk.scales_h >> (2 * ib)) & 3;
            int ls = ls_low | (ls_high << 4);
            float dl = d * (float)(ls - 32);

            float w = dl * (float)d_kvalues_iq4nl[nibble];
            sum += w * __half2float(x_sb[ib * 32 + lane_id]);
        }
        return sum;
    }
};

struct dot_iq2_xxs {
    __device__ __forceinline__ float operator()(const block_iq2_xxs& blk, const half* x_sb, int lane_id) const {
        float d = fp16_to_fp32(blk.d);
        const int l = lane_id >> 3; // 0..3
        const int j = lane_id & 7;  // 0..7

        float sum = 0.0f;
        #pragma unroll
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const uint8_t *aux8 = (const uint8_t *)(blk.qs + 4 * ib32);
            const uint32_t aux32_1 = ((const uint32_t *)(blk.qs + 4 * ib32))[1];
            const uint8_t signs = d_ksigns_iq2xs[(aux32_1 >> (7 * l)) & 127];
            const float db = d * (0.5f + (float)(aux32_1 >> 28)) * 0.25f;

            uint64_t grid_u64 = d_iq2xxs_grid[aux8[l]];
            int8_t grid_val = (int8_t)((grid_u64 >> (j * 8)) & 0xff);
            float sign = ((signs >> j) & 1) ? -db : db;
            float w = sign * (float)grid_val;

            sum = fmaf(w, __half2float(x_sb[ib32 * 32 + lane_id]), sum);
        }
        return sum;
    }
};

struct dot_iq2_xs {
    __device__ __forceinline__ float operator()(const block_iq2_xs& blk, const half* x_sb, int lane_id) const {
        float d = fp16_to_fp32(blk.d);
        const int l = lane_id >> 3; // 0..3
        const int j = lane_id & 7;  // 0..7

        float sum = 0.0f;
        #pragma unroll
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const uint16_t qs_entry = blk.qs[4 * ib32 + l];
            const uint8_t signs = d_ksigns_iq2xs[qs_entry >> 9];

            const int nibble = blk.scales[ib32];
            const float db_l = d * (0.5f + (float)(nibble & 0xf)) * 0.25f;
            const float db_h = d * (0.5f + (float)(nibble >> 4)) * 0.25f;
            const float db = (l < 2) ? db_l : db_h;

            uint64_t grid_u64 = d_iq2xs_grid[qs_entry & 511];
            int8_t grid_val = (int8_t)((grid_u64 >> (j * 8)) & 0xff);
            float sign = ((signs >> j) & 1) ? -db : db;
            float w = sign * (float)grid_val;

            sum = fmaf(w, __half2float(x_sb[ib32 * 32 + lane_id]), sum);
        }
        return sum;
    }
};

struct dot_iq2_s {
    __device__ __forceinline__ float operator()(const block_iq2_s& blk, const half* x_sb, int lane_id) const {
        float d = fp16_to_fp32(blk.d);
        const int l = lane_id >> 3; // 0..3
        const int j = lane_id & 7;  // 0..7

        float sum = 0.0f;
        #pragma unroll
        for (int ib = 0; ib < 8; ++ib) {
            const float db0 = d * (0.5f + (float)(blk.scales[ib] & 0x0f)) * 0.25f;
            const float db1 = d * (0.5f + (float)(blk.scales[ib] >> 4)) * 0.25f;
            const float dl = (l < 2) ? db0 : db1;

            const int grid_idx = (int)blk.qs[4 * ib + l] | ((int)(blk.qh[ib] << (8 - 2 * l)) & 0x300);
            uint64_t grid_u64 = d_iq2s_grid[grid_idx];
            int8_t grid_val = (int8_t)((grid_u64 >> (j * 8)) & 0xff);

            const uint8_t sign_byte = blk.qs[32 + 4 * ib + l];
            float sign = ((sign_byte >> j) & 1) ? -dl : dl;
            float w = sign * (float)grid_val;

            sum = fmaf(w, __half2float(x_sb[ib * 32 + lane_id]), sum);
        }
        return sum;
    }
};

struct dot_iq1_m {
    __device__ __forceinline__ float operator()(const block_iq1_m& blk, const half* x_sb, int lane_id) const {
        const int l = lane_id >> 3; // 0..3
        const int j = lane_id & 7;  // 0..7

        const uint16_t * sc = (const uint16_t *)blk.scales;
        const uint16_t scale_u16 =
            (uint16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
        const float d = fp16_to_fp32(scale_u16);

        float sum = 0.0f;
        #pragma unroll
        for (int ib = 0; ib < 8; ++ib) {
            const int scw = sc[ib / 2];
            const int lvl0 = (scw >> (6 * (ib % 2) + 0)) & 7;
            const int lvl1 = (scw >> (6 * (ib % 2) + 3)) & 7;
            const float dl = (l < 2) ? d * (float)(2 * lvl0 + 1) : d * (float)(2 * lvl1 + 1);

            const int qh_byte = blk.qh[2 * ib + l / 2];
            const int grid_idx = (int)blk.qs[4 * ib + l] | ((qh_byte << (8 - 4 * (l % 2))) & 0x700);
            uint64_t grid_u64 = d_iq1s_grid[grid_idx];
            int8_t grid_val = (int8_t)((grid_u64 >> (j * 8)) & 0xff);

            const float delta = (qh_byte & (l % 2 == 0 ? 0x08 : 0x80)) ? -0.125f : 0.125f;
            float w = dl * ((float)grid_val + delta);

            sum = fmaf(w, __half2float(x_sb[ib * 32 + lane_id]), sum);
        }
        return sum;
    }
};

template <typename BlockType, typename DotOp, int ROWS_PER_BLOCK = 4, int M = 1>
__global__ void gemv_qk_kernel(
    const half* __restrict__ x,
    const BlockType* __restrict__ weights,
    half* __restrict__ y,
    const half* __restrict__ bias,
    int N, int K,
    int K_full, int row_stride_sb,
    bool has_bias,
    bool use_lds
) {
    extern __shared__ half x_shared[];
    const half* x0_vec = x;
    const half* x1_vec = (M == 2) ? (x + K_full) : nullptr;
    if (use_lds) {
        // Chunk-aware staging (see gemv_q_kernel): x1's slice is at x + K_full.
        int tid = threadIdx.y * 32 + threadIdx.x;
        int block_threads = 32 * ROWS_PER_BLOCK;
        for (int i = tid; i < K; i += block_threads) {
            x_shared[i] = x[i];
        }
        if constexpr (M == 2) {
            for (int i = tid; i < K; i += block_threads) {
                x_shared[K + i] = x[K_full + i];
            }
        }
        __syncthreads();
        x0_vec = x_shared;
        if constexpr (M == 2) {
            x1_vec = x_shared + K;
        }
    }

    int wave_id = threadIdx.y;
    int lane_id = threadIdx.x;
    int n = blockIdx.x * ROWS_PER_BLOCK + wave_id;
    if (n >= N) return;

    int num_superblocks = K / 256;
    const BlockType* row = weights + (size_t)n * row_stride_sb;

    DotOp dot_op;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (int b = 0; b < num_superblocks; ++b) {
        BlockType blk = row[b];
        sum0 += dot_op(blk, &x0_vec[b * 256], lane_id);
        if constexpr (M == 2) {
            sum1 += dot_op(blk, &x1_vec[b * 256], lane_id);
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum0 += __shfl_down(sum0, offset, 32);
        if constexpr (M == 2) {
            sum1 += __shfl_down(sum1, offset, 32);
        }
    }

    if (lane_id == 0) {
        if (has_bias) {
            float b_val = __half2float(bias[n]);
            sum0 += b_val;
            if constexpr (M == 2) sum1 += b_val;
        }
        y[n] = __float2half(sum0);
        if constexpr (M == 2) {
            y[N + n] = __float2half(sum1);
        }
    }
}

template <int ROWS_PER_BLOCK = 4, int M = 1>
__global__ void gemv_iq3s_kernel(
    const half* __restrict__ x,
    const block_iq3_s* __restrict__ weights,
    half* __restrict__ y,
    const half* __restrict__ bias,
    int N, int K,
    int K_full, int row_stride_sb,
    bool has_bias,
    bool use_lds
) {
    __shared__ uint32_t s_grid[512];
    extern __shared__ half x_shared[];
    const half* x0_vec = x;
    const half* x1_vec = (M == 2) ? (x + K_full) : nullptr;
    int tid = threadIdx.y * 32 + threadIdx.x;

    // Cooperatively load 512 entries (2 KB) of grid table into LDS
    if (tid < 128) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            s_grid[tid * 4 + i] = d_iq3s_grid[tid * 4 + i];
        }
    }

    if (use_lds) {
        // Chunk-aware staging (see gemv_q_kernel): x1's slice is at x + K_full.
        int block_threads = 32 * ROWS_PER_BLOCK;
        for (int i = tid; i < K; i += block_threads) {
            x_shared[i] = x[i];
        }
        if constexpr (M == 2) {
            for (int i = tid; i < K; i += block_threads) {
                x_shared[K + i] = x[K_full + i];
            }
        }
        x0_vec = x_shared;
        if constexpr (M == 2) {
            x1_vec = x_shared + K;
        }
    }
    __syncthreads();

    int wave_id = threadIdx.y;
    int lane_id = threadIdx.x;
    int n = blockIdx.x * ROWS_PER_BLOCK + wave_id;
    if (n >= N) return;

    int num_superblocks = K / 256;
    const block_iq3_s* row = weights + (size_t)n * row_stride_sb;

    dot_iq3_s dot_op(s_grid);
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (int b = 0; b < num_superblocks; ++b) {
        block_iq3_s blk = row[b];
        sum0 += dot_op(blk, &x0_vec[b * 256], lane_id);
        if constexpr (M == 2) {
            sum1 += dot_op(blk, &x1_vec[b * 256], lane_id);
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum0 += __shfl_down(sum0, offset, 32);
        if constexpr (M == 2) {
            sum1 += __shfl_down(sum1, offset, 32);
        }
    }

    if (lane_id == 0) {
        if (has_bias) {
            float b_val = __half2float(bias[n]);
            sum0 += b_val;
            if constexpr (M == 2) sum1 += b_val;
        }
        y[n] = __float2half(sum0);
        if constexpr (M == 2) {
            y[N + n] = __float2half(sum1);
        }
    }
}

template <int ROWS_PER_BLOCK = 4, int M = 1>
__global__ void gemv_iq3xxs_kernel(
    const half* __restrict__ x,
    const block_iq3_xxs* __restrict__ weights,
    half* __restrict__ y,
    const half* __restrict__ bias,
    int N, int K,
    int K_full, int row_stride_sb,
    bool has_bias,
    bool use_lds
) {
    __shared__ uint32_t s_grid[256];
    extern __shared__ half x_shared[];
    const half* x0_vec = x;
    const half* x1_vec = (M == 2) ? (x + K_full) : nullptr;
    int tid = threadIdx.y * 32 + threadIdx.x;

    // Cooperatively load 256 entries (1 KB) of grid table into LDS
    if (tid < 128) {
        s_grid[tid * 2 + 0] = d_iq3xxs_grid[tid * 2 + 0];
        s_grid[tid * 2 + 1] = d_iq3xxs_grid[tid * 2 + 1];
    }

    if (use_lds) {
        // Chunk-aware staging (see gemv_q_kernel): x1's slice is at x + K_full.
        int block_threads = 32 * ROWS_PER_BLOCK;
        for (int i = tid; i < K; i += block_threads) {
            x_shared[i] = x[i];
        }
        if constexpr (M == 2) {
            for (int i = tid; i < K; i += block_threads) {
                x_shared[K + i] = x[K_full + i];
            }
        }
        x0_vec = x_shared;
        if constexpr (M == 2) {
            x1_vec = x_shared + K;
        }
    }
    __syncthreads();

    int wave_id = threadIdx.y;
    int lane_id = threadIdx.x;
    int n = blockIdx.x * ROWS_PER_BLOCK + wave_id;
    if (n >= N) return;

    int num_superblocks = K / 256;
    const block_iq3_xxs* row = weights + (size_t)n * row_stride_sb;

    dot_iq3_xxs dot_op(s_grid);
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (int b = 0; b < num_superblocks; ++b) {
        block_iq3_xxs blk = row[b];
        sum0 += dot_op(blk, &x0_vec[b * 256], lane_id);
        if constexpr (M == 2) {
            sum1 += dot_op(blk, &x1_vec[b * 256], lane_id);
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum0 += __shfl_down(sum0, offset, 32);
        if constexpr (M == 2) {
            sum1 += __shfl_down(sum1, offset, 32);
        }
    }

    if (lane_id == 0) {
        if (has_bias) {
            float b_val = __half2float(bias[n]);
            sum0 += b_val;
            if constexpr (M == 2) sum1 += b_val;
        }
        y[n] = __float2half(sum0);
        if constexpr (M == 2) {
            y[N + n] = __float2half(sum1);
        }
    }
}

} // namespace

template <int M>
static void dispatch_gemv_q(
    const void* input,
    const void* packed_weight,
    void* output,
    const void* bias,
    int ggml_type,
    int N,
    int K,
    int K_full, int row_stride_sb,
    bool has_bias,
    hipStream_t stream
) {
    constexpr int ROWS_PER_BLOCK = 4;
    dim3 block(32, ROWS_PER_BLOCK); // 128 threads per block (4 waves)
    dim3 grid((N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);

    bool use_lds = ((size_t)M * K <= 8192);
    size_t shared_bytes = use_lds ? ((size_t)M * K * sizeof(half)) : 0;

    if (ggml_type == GGML_TYPE_Q8_0) {
        hipLaunchKernelGGL(
            (gemv_q_kernel<block_q8_0, dot_q8_0, ROWS_PER_BLOCK, M>),
            grid, block, shared_bytes, stream,
            (const half*)input, (const block_q8_0*)packed_weight, (half*)output, (const half*)bias,
            N, K, K_full, row_stride_sb, has_bias, use_lds
        );
    } else if (ggml_type == GGML_TYPE_Q4_0) {
        hipLaunchKernelGGL(
            (gemv_q_kernel<block_q4_0, dot_q4_0, ROWS_PER_BLOCK, M>),
            grid, block, shared_bytes, stream,
            (const half*)input, (const block_q4_0*)packed_weight, (half*)output, (const half*)bias,
            N, K, K_full, row_stride_sb, has_bias, use_lds
        );
    } else if (ggml_type == GGML_TYPE_Q4_K) {
        hipLaunchKernelGGL(
            (gemv_qk_kernel<block_q4_K, dot_q4_K, ROWS_PER_BLOCK, M>),
            grid, block, shared_bytes, stream,
            (const half*)input, (const block_q4_K*)packed_weight, (half*)output, (const half*)bias,
            N, K, K_full, row_stride_sb, has_bias, use_lds
        );
    } else if (ggml_type == GGML_TYPE_Q5_K) {
        hipLaunchKernelGGL(
            (gemv_qk_kernel<block_q5_K, dot_q5_K, ROWS_PER_BLOCK, M>),
            grid, block, shared_bytes, stream,
            (const half*)input, (const block_q5_K*)packed_weight, (half*)output, (const half*)bias,
            N, K, K_full, row_stride_sb, has_bias, use_lds
        );
    } else if (ggml_type == GGML_TYPE_Q6_K) {
        hipLaunchKernelGGL(
            (gemv_qk_kernel<block_q6_K, dot_q6_K, ROWS_PER_BLOCK, M>),
            grid, block, shared_bytes, stream,
            (const half*)input, (const block_q6_K*)packed_weight, (half*)output, (const half*)bias,
            N, K, K_full, row_stride_sb, has_bias, use_lds
        );
    } else if (ggml_type == GGML_TYPE_Q2_K) {
        hipLaunchKernelGGL(
            (gemv_qk_kernel<block_q2_K, dot_q2_K, ROWS_PER_BLOCK, M>),
            grid, block, shared_bytes, stream,
            (const half*)input, (const block_q2_K*)packed_weight, (half*)output, (const half*)bias,
            N, K, K_full, row_stride_sb, has_bias, use_lds
        );
    } else if (ggml_type == GGML_TYPE_IQ3_S) {
        hipLaunchKernelGGL(
            (gemv_iq3s_kernel<ROWS_PER_BLOCK, M>),
            grid, block, shared_bytes, stream,
            (const half*)input, (const block_iq3_s*)packed_weight, (half*)output, (const half*)bias,
            N, K, K_full, row_stride_sb, has_bias, use_lds
        );
    } else if (ggml_type == GGML_TYPE_IQ3_XXS) {
        hipLaunchKernelGGL(
            (gemv_iq3xxs_kernel<ROWS_PER_BLOCK, M>),
            grid, block, shared_bytes, stream,
            (const half*)input, (const block_iq3_xxs*)packed_weight, (half*)output, (const half*)bias,
            N, K, K_full, row_stride_sb, has_bias, use_lds
        );
    } else if (ggml_type == GGML_TYPE_IQ4_XS) {
        hipLaunchKernelGGL(
            (gemv_qk_kernel<block_iq4_xs, dot_iq4_xs, ROWS_PER_BLOCK, M>),
            grid, block, shared_bytes, stream,
            (const half*)input, (const block_iq4_xs*)packed_weight, (half*)output, (const half*)bias,
            N, K, K_full, row_stride_sb, has_bias, use_lds
        );
    } else if (ggml_type == GGML_TYPE_IQ2_XXS) {
        hipLaunchKernelGGL(
            (gemv_qk_kernel<block_iq2_xxs, dot_iq2_xxs, ROWS_PER_BLOCK, M>),
            grid, block, shared_bytes, stream,
            (const half*)input, (const block_iq2_xxs*)packed_weight, (half*)output, (const half*)bias,
            N, K, K_full, row_stride_sb, has_bias, use_lds
        );
    } else if (ggml_type == GGML_TYPE_IQ2_XS) {
        hipLaunchKernelGGL(
            (gemv_qk_kernel<block_iq2_xs, dot_iq2_xs, ROWS_PER_BLOCK, M>),
            grid, block, shared_bytes, stream,
            (const half*)input, (const block_iq2_xs*)packed_weight, (half*)output, (const half*)bias,
            N, K, K_full, row_stride_sb, has_bias, use_lds
        );
    } else if (ggml_type == GGML_TYPE_IQ2_S) {
        hipLaunchKernelGGL(
            (gemv_qk_kernel<block_iq2_s, dot_iq2_s, ROWS_PER_BLOCK, M>),
            grid, block, shared_bytes, stream,
            (const half*)input, (const block_iq2_s*)packed_weight, (half*)output, (const half*)bias,
            N, K, K_full, row_stride_sb, has_bias, use_lds
        );
    } else if (ggml_type == GGML_TYPE_IQ1_M) {
        hipLaunchKernelGGL(
            (gemv_qk_kernel<block_iq1_m, dot_iq1_m, ROWS_PER_BLOCK, M>),
            grid, block, shared_bytes, stream,
            (const half*)input, (const block_iq1_m*)packed_weight, (half*)output, (const half*)bias,
            N, K, K_full, row_stride_sb, has_bias, use_lds
        );
    }
}

// Superblock (row-chunk) width per type: 32-elem blocks for Q8_0/Q4_0,
// 256-elem superblocks for all K-quants and I-quants.
static int gemv_q_sb_elems(int ggml_type) {
    if (ggml_type == GGML_TYPE_Q8_0 || ggml_type == GGML_TYPE_Q4_0) return 32;
    return 256;
}

// Packed bytes per superblock. Must match the table in
// pytorch_bindings.cpp gemv_q_forward.
static int gemv_q_block_bytes(int ggml_type) {
    switch (ggml_type) {
        case GGML_TYPE_Q8_0: return 34;
        case GGML_TYPE_Q4_0: return 18;
        case 10: return 84;   // Q2_K
        case 12: return 144;  // Q4_K
        case 13: return 176;  // Q5_K
        case 14: return 210;  // Q6_K
        case 16: return 66;   // IQ2_XXS
        case 17: return 74;   // IQ2_XS
        case 18: return 98;   // IQ3_XXS
        case 21: return 110;  // IQ3_S
        case 22: return 82;   // IQ2_S
        case 23: return 136;  // IQ4_XS
        case 29: return 56;   // IQ1_M
        default: return 0;
    }
}

// LDS staging cap (elements, incl. M rows) mirrors the use_lds policy in
// dispatch_gemv_q. Chunks are multiples of 256 elems, which is divisible by
// every block width (32..256), so no chunk ever splits a packed block.
static int gemv_q_chunk_cap(int M) {
    int cap = 8192 / (M > 0 ? M : 1);
    return (cap / 256) * 256;
}

// Number of K-chunks for a (M, K) GEMV. Returns 1 when the single LDS path
// applies (no split needed).
extern "C" {

int gemv_q_split_chunks(int M, int K) {
    int cap = gemv_q_chunk_cap(M);
    if (cap <= 0 || K <= cap) return 1;
    return (K + cap - 1) / cap;
}

void launch_gemv_q_forward(
    const void* input,
    const void* packed_weight,
    void* output,
    const void* bias,
    int ggml_type,
    int M,
    int N,
    int K,
    bool has_bias,
    hipStream_t stream
) {
    int sb = gemv_q_sb_elems(ggml_type);
    int row_stride_sb = (sb > 0) ? (K / sb) : 0;
    if (M == 1) {
        dispatch_gemv_q<1>(input, packed_weight, output, bias, ggml_type, N, K, K, row_stride_sb, has_bias, stream);
    } else if (M == 2) {
        dispatch_gemv_q<2>(input, packed_weight, output, bias, ggml_type, N, K, K, row_stride_sb, has_bias, stream);
    }
}

} // extern "C"

// Sum of partials over K-chunks (+ optional bias), fp32 accumulate.
__global__ void gemv_q_reduce_kernel(
    const half* __restrict__ partials,
    half* __restrict__ y,
    const half* __restrict__ bias,
    int MN, int N, int C,
    bool has_bias
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= MN) return;
    float s = 0.0f;
    for (int c = 0; c < C; ++c) {
        s += __half2float(partials[(size_t)c * MN + i]);
    }
    if (has_bias) {
        s += __half2float(bias[i % N]);
    }
    y[i] = __float2half(s);
}

// K-split GEMV for M*K > 8192 (x no longer fits the LDS staging path).
// Each chunk runs the standard kernel over its K-slice with full LDS reuse;
// partials [C, M*N] are reduced afterwards. scratch must hold C*M*N halves.
// Bias is applied once in the reduce (chunks run with has_bias=false).
template <int M>
static void dispatch_gemv_q_split(
    const void* input,
    const void* packed_weight,
    void* partials,
    int ggml_type,
    int N,
    int K,
    int C,
    hipStream_t stream
) {
    int sb_elems = gemv_q_sb_elems(ggml_type);
    int blk_bytes = gemv_q_block_bytes(ggml_type);
    int row_stride_sb = K / sb_elems;
    // Distribute K over C chunks in units of sb_elems (never splits a block).
    int base = (K / C / sb_elems) * sb_elems;
    int rem = (K - base * C) / sb_elems;
    int start = 0;
    for (int c = 0; c < C; ++c) {
        int kc = base + (c < rem ? sb_elems : 0);
        const half* x_c = (const half*)input + start;
        const void* w_c = (const uint8_t*)packed_weight +
            (size_t)(start / sb_elems) * blk_bytes;
        half* y_c = (half*)partials + (size_t)c * M * N;
        dispatch_gemv_q<M>(x_c, w_c, y_c, nullptr, ggml_type, N, kc, K,
                           row_stride_sb, false, stream);
        start += kc;
    }
}

extern "C" {

void launch_gemv_q_split_forward(
    const void* input,
    const void* packed_weight,
    void* partials_scratch,
    void* output,
    const void* bias,
    int ggml_type,
    int M,
    int N,
    int K,
    int C,
    bool has_bias,
    hipStream_t stream
) {
    if (M == 1) {
        dispatch_gemv_q_split<1>(input, packed_weight, partials_scratch,
                                 ggml_type, N, K, C, stream);
    } else if (M == 2) {
        dispatch_gemv_q_split<2>(input, packed_weight, partials_scratch,
                                 ggml_type, N, K, C, stream);
    } else {
        return;
    }
    int MN = M * N;
    dim3 block(256);
    dim3 grid((MN + 255) / 256);
    hipLaunchKernelGGL(gemv_q_reduce_kernel, grid, block, 0, stream,
        (const half*)partials_scratch, (half*)output, (const half*)bias,
        MN, N, C, has_bias);
}

} // extern "C"

// ---------------------------------------------------------------------------
// Embedding Lookup Decoders: Zero-Copy on-the-fly dequantization (C++ linkage)
// ---------------------------------------------------------------------------
struct decode_iq2_s {
    __device__ __forceinline__ void operator()(const block_iq2_s& blk, half* out_sb, int lane_id) const {
        float d = fp16_to_fp32(blk.d);
        const int l = lane_id >> 3;
        const int j = lane_id & 7;
        #pragma unroll
        for (int ib = 0; ib < 8; ++ib) {
            const float db0 = d * (0.5f + (float)(blk.scales[ib] & 0x0f)) * 0.25f;
            const float db1 = d * (0.5f + (float)(blk.scales[ib] >> 4)) * 0.25f;
            const float dl = (l < 2) ? db0 : db1;
            const int grid_idx = (int)blk.qs[4 * ib + l] | ((int)(blk.qh[ib] << (8 - 2 * l)) & 0x300);
            uint64_t grid_u64 = d_iq2s_grid[grid_idx];
            int8_t grid_val = (int8_t)((grid_u64 >> (j * 8)) & 0xff);
            const uint8_t sign_byte = blk.qs[32 + 4 * ib + l];
            float sign = ((sign_byte >> j) & 1) ? -dl : dl;
            float w = sign * (float)grid_val;
            out_sb[ib * 32 + lane_id] = __float2half(w);
        }
    }
};

struct decode_q4_K {
    __device__ __forceinline__ void operator()(const block_q4_K& blk, half* out_sb, int lane_id) const {
        float d = fp16_to_fp32(blk.d);
        float dmin = fp16_to_fp32(blk.dmin);
        float sc[8], m[8];
        #pragma unroll
        for (int g = 0; g < 8; ++g) {
            sc[g] = d * (float)unpack_k4_scale(blk.scales, g);
            m[g]  = dmin * (float)unpack_k4_min(blk.scales, g);
        }
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            uint32_t byte = blk.qs[k * 32 + lane_id];
            float v0 = (float)(byte & 0x0f);
            float v1 = (float)(byte >> 4);
            float w0 = fmaf(sc[2 * k + 0], v0, -m[2 * k + 0]);
            float w1 = fmaf(sc[2 * k + 1], v1, -m[2 * k + 1]);
            out_sb[(2 * k + 0) * 32 + lane_id] = __float2half(w0);
            out_sb[(2 * k + 1) * 32 + lane_id] = __float2half(w1);
        }
    }
};

struct decode_q8_0 {
    __device__ __forceinline__ void operator()(const block_q8_0& blk, half* out_sb, int lane_id) const {
        float d = fp16_to_fp32(blk.d);
        out_sb[lane_id] = __float2half((float)blk.qs[lane_id] * d);
    }
};

template <typename BlockType, typename DecodeOp, int BLOCK_ELEMENTS = 256>
__global__ void dequant_embedding_kernel(
    const BlockType* __restrict__ weights,
    const int64_t* __restrict__ token_ids,
    half* __restrict__ out,
    int K,
    int blocks_per_row
) {
    int token_idx = blockIdx.x;
    int b = blockIdx.y;
    int lane_id = threadIdx.x;
    int64_t row_idx = token_ids[token_idx];

    const BlockType& blk = weights[row_idx * blocks_per_row + b];
    DecodeOp decode_op;
    decode_op(blk, &out[token_idx * K + b * BLOCK_ELEMENTS], lane_id);
}

extern "C" {

void launch_dequant_embedding_forward(
    const void* packed_weight,
    const int64_t* token_ids,
    void* output,
    int ggml_type,
    int num_tokens,
    int K,
    hipStream_t stream
) {
    if (ggml_type == GGML_TYPE_IQ2_S) {
        int blocks_per_row = K / 256;
        dim3 grid(num_tokens, blocks_per_row);
        hipLaunchKernelGGL(
            (dequant_embedding_kernel<block_iq2_s, decode_iq2_s, 256>),
            grid, dim3(32), 0, stream,
            (const block_iq2_s*)packed_weight, token_ids, (half*)output, K, blocks_per_row
        );
    } else if (ggml_type == GGML_TYPE_Q4_K) {
        int blocks_per_row = K / 256;
        dim3 grid(num_tokens, blocks_per_row);
        hipLaunchKernelGGL(
            (dequant_embedding_kernel<block_q4_K, decode_q4_K, 256>),
            grid, dim3(32), 0, stream,
            (const block_q4_K*)packed_weight, token_ids, (half*)output, K, blocks_per_row
        );
    } else if (ggml_type == GGML_TYPE_Q8_0) {
        int blocks_per_row = K / 32;
        dim3 grid(num_tokens, blocks_per_row);
        hipLaunchKernelGGL(
            (dequant_embedding_kernel<block_q8_0, decode_q8_0, 32>),
            grid, dim3(32), 0, stream,
            (const block_q8_0*)packed_weight, token_ids, (half*)output, K, blocks_per_row
        );
    }
}

} // extern "C"
