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
    __device__ __forceinline__ float operator()(const block_q8_0* __restrict__ blk_ptr, const half* x) const {
        const block_q8_0& blk = *blk_ptr; // alias: fields load on demand, no struct copy
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
    __device__ __forceinline__ float operator()(const block_q4_0* __restrict__ blk_ptr, const half* x) const {
        const block_q4_0& blk = *blk_ptr; // alias: fields load on demand, no struct copy
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
        sum0 += dot_op(row + b, &x0_vec[b * DotOp::BLOCK_SIZE]);
        if constexpr (M == 2) {
            sum1 += dot_op(row + b, &x1_vec[b * DotOp::BLOCK_SIZE]);
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
    __device__ __forceinline__ float operator()(const block_q4_K* __restrict__ blk_ptr, const half* x_sb, int lane_id) const {
        const block_q4_K& blk = *blk_ptr; // alias: fields load on demand, no struct copy
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
    __device__ __forceinline__ float operator()(const block_q5_K* __restrict__ blk_ptr, const half* x_sb, int lane_id) const {
        const block_q5_K& blk = *blk_ptr; // alias: fields load on demand, no struct copy
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
    __device__ __forceinline__ float operator()(const block_q6_K* __restrict__ blk_ptr, const half* x_sb, int lane_id) const {
        const block_q6_K& blk = *blk_ptr; // alias: fields load on demand, no struct copy
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
    // Vector L1 cached codebook
    const uint32_t* grid;
    __device__ __forceinline__ dot_iq3_s(const uint32_t* g = d_iq3s_grid) : grid(g) {}
    __device__ __forceinline__ float operator()(const block_iq3_s* __restrict__ blk_ptr, const half* x_sb, int lane_id) const {
        const block_iq3_s& blk = *blk_ptr;
        float d = fp16_to_fp32(blk.d);
        const int l = lane_id >> 3; // 0..3
        const int j = lane_id & 7;  // 0..7
        const int j_mod_4 = j & 3;
        const int qh_bit_pos = 2 * l + (j >= 4 ? 1 : 0);
        const int qs_sub_offset = (j >= 4 ? 1 : 0);

        // 1. Single 32-bit load for 4-byte scales: precompute all 8 db values
        uint32_t scales_u32;
        memcpy(&scales_u32, blk.scales, 4);
        float db[8];
        #pragma unroll
        for (int ib = 0; ib < 8; ++ib) {
            uint8_t sc_raw = (scales_u32 >> (ib * 4)) & 0x0f;
            db[ib] = d * (float)(1 + 2 * sc_raw);
        }

        // 2. Single 64-bit load for 8-byte qh
        uint64_t qh_u64;
        memcpy(&qh_u64, blk.qh, 8);

        // 3. Load 32-byte signs array into registers once per superblock
        uint32_t signs_words[8];
        memcpy(signs_words, blk.signs, 32);

        const int8_t* grid_i8 = reinterpret_cast<const int8_t*>(grid);
        float sum = 0.0f;

        // Stage 1: Issue all 8 qs loads up-front into the memory pipeline concurrently
        int qs_bytes[8];
        #pragma unroll
        for (int ib = 0; ib < 8; ++ib) {
            qs_bytes[ib] = (int)blk.qs[8 * ib + 2 * l + qs_sub_offset];
        }

        // Stage 2: Compute grid indices and issue all grid table lookups into L1 concurrently
        int8_t grid_vals[8];
        #pragma unroll
        for (int ib = 0; ib < 8; ++ib) {
            int qh_byte = (int)((qh_u64 >> (8 * ib)) & 0xff);
            int qh_bit = (qh_byte >> qh_bit_pos) & 1;
            int grid_idx = qs_bytes[ib] | (qh_bit << 8);
            uint32_t grid_u32 = grid[grid_idx];
            grid_vals[ib] = (int8_t)((grid_u32 >> (j_mod_4 * 8)) & 0xff);
        }

        // Stage 3: Compute signed weights and accumulate FMAs
        #pragma unroll
        for (int ib = 0; ib < 8; ++ib) {
            int sign_bit = (signs_words[ib] >> lane_id) & 1;
            float sign = sign_bit ? -db[ib] : db[ib];
            float w = sign * (float)grid_vals[ib];
            sum = fmaf(w, __half2float(x_sb[ib * 32 + lane_id]), sum);
        }
        return sum;
    }
};

struct dot_iq3_xxs {
    const uint32_t* grid;
    __device__ __forceinline__ dot_iq3_xxs(const uint32_t* g = d_iq3xxs_grid) : grid(g) {}
    __device__ __forceinline__ float operator()(const block_iq3_xxs* __restrict__ blk_ptr, const half* x_sb, int lane_id) const {
        const block_iq3_xxs& blk = *blk_ptr; // alias: fields load on demand, no struct copy
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
    __device__ __forceinline__ float operator()(const block_q2_K* __restrict__ blk_ptr, const half* x_sb, int lane_id) const {
        const block_q2_K& blk = *blk_ptr; // alias: fields load on demand, no struct copy
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
    __device__ __forceinline__ float operator()(const block_iq4_xs* __restrict__ blk_ptr, const half* x_sb, int lane_id) const {
        uint64_t header = *reinterpret_cast<const uint64_t*>(blk_ptr);
        uint16_t d_u16 = (uint16_t)(header & 0xffff);
        float d = fp16_to_fp32(d_u16);
        uint32_t scales_h = (uint32_t)((header >> 16) & 0xffff);
        uint32_t scales_l = (uint32_t)(header >> 32);

        float sum = 0.0f;
        #pragma unroll
        for (int ib = 0; ib < 8; ++ib) {
            int ls_low = (scales_l >> (4 * ib)) & 0x0f;
            int ls_high = (scales_h >> (2 * ib)) & 0x03;
            int ls = ls_low | (ls_high << 4);
            float dl = d * (float)(ls - 32);

            int byte_idx = 16 * ib + (lane_id & 15);
            uint8_t q_byte = blk_ptr->qs[byte_idx];
            int nibble = (lane_id < 16) ? (q_byte & 0x0f) : (q_byte >> 4);

            float w = dl * (float)d_kvalues_iq4nl[nibble];
            sum += w * __half2float(x_sb[ib * 32 + lane_id]);
        }
        return sum;
    }
};

struct dot_iq2_xxs {
    __device__ __forceinline__ float operator()(const block_iq2_xxs* __restrict__ blk_ptr, const half* x_sb, int lane_id) const {
        const block_iq2_xxs& blk = *blk_ptr; // alias: fields load on demand, no struct copy
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
    __device__ __forceinline__ float operator()(const block_iq2_xs* __restrict__ blk_ptr, const half* x_sb, int lane_id) const {
        const block_iq2_xs& blk = *blk_ptr; // alias: fields load on demand, no struct copy
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
    // NOTE: block arrives by pointer and fields load on demand. Materializing
    // the whole 82-byte struct per thread blew the VGPR budget and spilled
    // (20 static scratch ops); all lanes read the same block, so these loads
    // broadcast through L2 instead of living in registers.
    __device__ __forceinline__ float operator()(const block_iq2_s* __restrict__ blk, const half* x_sb, int lane_id) const {
        float d = fp16_to_fp32(blk->d);
        const int l = lane_id >> 3; // 0..3
        const int j = lane_id & 7;  // 0..7

        float sum = 0.0f;
        #pragma unroll 2
        for (int ib = 0; ib < 8; ++ib) {
            const float db0 = d * (0.5f + (float)(blk->scales[ib] & 0x0f)) * 0.25f;
            const float db1 = d * (0.5f + (float)(blk->scales[ib] >> 4)) * 0.25f;
            const float dl = (l < 2) ? db0 : db1;

            const int grid_idx = (int)blk->qs[4 * ib + l] | ((int)(blk->qh[ib] << (8 - 2 * l)) & 0x300);
            uint64_t grid_u64 = d_iq2s_grid[grid_idx];
            int8_t grid_val = (int8_t)((grid_u64 >> (j * 8)) & 0xff);

            const uint8_t sign_byte = blk->qs[32 + 4 * ib + l];
            float sign = ((sign_byte >> j) & 1) ? -dl : dl;
            float w = sign * (float)grid_val;

            sum = fmaf(w, __half2float(x_sb[ib * 32 + lane_id]), sum);
        }
        return sum;
    }
};

struct dot_iq1_m {
    __device__ __forceinline__ float operator()(const block_iq1_m* __restrict__ blk_ptr, const half* x_sb, int lane_id) const {
        const block_iq1_m& blk = *blk_ptr; // alias: fields load on demand, no struct copy
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
        sum0 += dot_op(row + b, &x0_vec[b * 256], lane_id);
        if constexpr (M == 2) {
            sum1 += dot_op(row + b, &x1_vec[b * 256], lane_id);
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

// ---------------------------------------------------------------------------
// Batched GEMM over quantized weights (P-prefill-1). Decode kernels above
// are sacred (M=1/2, LDS-staged x); these are NEW kernels for S>1 prefill.
//
// Design: one block handles ROWS_PER_BLOCK N-rows x one MTILE slab of M.
// Weight blocks stream global->LDS ONCE per K-chunk and fan out to all
// MTILE dots (the entire win: row-loop GEMV re-streams weights per token).
// x rows stay in global (L2-cached across N-blocks; x is small vs weights).
// Per-thread float sum[MTILE]; lane 0 writes y[m*N+n] (strided, tiny).
//
// LDS layout: extern bytes, weight area 16-aligned, per-row stride padded
// to 16B (Q6_K's 210B blocks would otherwise misalign rows>0).
// ---------------------------------------------------------------------------

template <typename BlockType, typename DotOp, int ROWS_PER_BLOCK = 4, int MTILE = 16>
__global__ void gemm_qk_kernel(
    const half* __restrict__ x,       // [M, K_full] row-major
    const BlockType* __restrict__ weights, // [N, row_stride_sb]
    half* __restrict__ y,            // [M, N]
    const half* __restrict__ bias,
    int N, int K, int Mrows,
    int K_full, int row_stride_sb,
    int blk_bytes,
    bool has_bias
) {
    extern __shared__ uint8_t ws_base[];
    const int tid = threadIdx.y * 32 + threadIdx.x;
    const int block_threads = 32 * ROWS_PER_BLOCK;
    const int wave_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int n = blockIdx.x * ROWS_PER_BLOCK + wave_id;
    const int m_base = blockIdx.y * MTILE;
    const bool n_valid = (n < N);

    // K-chunk sized so ROWS rows of blocks fit ~48KB LDS.
    const int num_sb = K / 256;
    int chunk_sb = (49152 / ROWS_PER_BLOCK) / blk_bytes;
    if (chunk_sb < 1) chunk_sb = 1;
    if (chunk_sb > num_sb) chunk_sb = num_sb;
    const size_t row_stride_B = ((size_t)chunk_sb * (size_t)blk_bytes + 15) & ~(size_t)15;

    DotOp dot_op;
    float sum[MTILE];
    #pragma unroll
    for (int j = 0; j < MTILE; ++j) sum[j] = 0.0f;

    for (int sb0 = 0; sb0 < num_sb; sb0 += chunk_sb) {
        int cur_sb = num_sb - sb0;
        if (cur_sb > chunk_sb) cur_sb = chunk_sb;
        // Cooperative global->LDS stage of ROWS rows x cur_sb blocks,
        // row by row (division-free): packed block widths (98..210B) are
        // rarely 16-aligned, so vectorized staging would fault; staging
        // traffic is negligible next to the dot FLOPs -- keep it simple.
        for (int r = 0; r < ROWS_PER_BLOCK; ++r) {
            int n_r = blockIdx.x * ROWS_PER_BLOCK + r;
            size_t valid = (n_r < N) ? (size_t)cur_sb * (size_t)blk_bytes : 0;
            size_t base = (size_t)r * row_stride_B;
            const uint8_t* src_r = (const uint8_t*)weights
                + ((size_t)n_r * row_stride_sb + sb0) * blk_bytes;
            for (size_t i = tid; i < valid; i += (size_t)block_threads) {
                ws_base[base + i] = src_r[i];
            }
        }
        __syncthreads();

        if (n_valid) {
            const BlockType* wrow = (const BlockType*)(ws_base + (size_t)wave_id * row_stride_B);
            const int x_col0 = sb0 * 256;
            #pragma unroll
            for (int j = 0; j < MTILE; ++j) {
                int m = m_base + j;
                if (m >= Mrows) continue;
                const half* xm = x + (size_t)m * K_full + x_col0;
                float acc = 0.0f;
                for (int b = 0; b < cur_sb; ++b) {
                    acc += dot_op(wrow + b, &xm[b * 256], lane_id);
                }
                sum[j] += acc;
            }
        }
        __syncthreads();
    }

    if (!n_valid) return;
    #pragma unroll
    for (int j = 0; j < MTILE; ++j) {
        float s = sum[j];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            s += __shfl_down(s, offset, 32);
        }
        int m = m_base + j;
        if (lane_id == 0 && m < Mrows) {
            if (has_bias) s += __half2float(bias[n]);
            y[(size_t)m * N + n] = __float2half(s);
        }
    }
}

// Same batched-GEMM design for the grid-table I-quants (IQ3_S / IQ3_XXS).
// Dynamic shared starts AFTER static: ws_dyn[0] already sits past s_grid,
// so NO extra offset (an earlier +GRID_BYTES revision double-counted and
// overflowed LDS by 2KB -> cross-block corruption, nondeterministic garbage;
// decode's s_grid+x_shared pattern proves the rule).
template <typename BlockType, typename DotOp, int ROWS_PER_BLOCK = 4, int MTILE = 16>
__global__ void gemm_iq3_kernel(
    const half* __restrict__ x,
    const BlockType* __restrict__ weights,
    half* __restrict__ y,
    const half* __restrict__ bias,
    int N, int K, int Mrows,
    int K_full, int row_stride_sb,
    int blk_bytes,
    bool has_bias
) {
    __shared__ uint32_t s_grid[512];  // max(512 iq3s, 256 iq3xxs) words
    extern __shared__ uint8_t ws_dyn[];
    uint8_t* ws_base = ws_dyn;  // dynamic base is already past static s_grid
    const int tid = threadIdx.y * 32 + threadIdx.x;
    const int block_threads = 32 * ROWS_PER_BLOCK;
    const int wave_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int n = blockIdx.x * ROWS_PER_BLOCK + wave_id;
    const int m_base = blockIdx.y * MTILE;
    const bool n_valid = (n < N);

    if constexpr (std::is_same_v<BlockType, block_iq3_s>) {
        if (tid < 128) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                s_grid[tid * 4 + i] = d_iq3s_grid[tid * 4 + i];
            }
        }
    } else {
        if (tid < 128) {
            s_grid[tid * 2 + 0] = d_iq3xxs_grid[tid * 2 + 0];
            s_grid[tid * 2 + 1] = d_iq3xxs_grid[tid * 2 + 1];
        }
    }
    __syncthreads();

    const int num_sb = K / 256;
    int chunk_sb = (49152 / ROWS_PER_BLOCK) / blk_bytes;
    if (chunk_sb < 1) chunk_sb = 1;
    if (chunk_sb > num_sb) chunk_sb = num_sb;
    const size_t row_stride_B = ((size_t)chunk_sb * (size_t)blk_bytes + 15) & ~(size_t)15;

    DotOp dot_op(s_grid);
    float sum[MTILE];
    #pragma unroll
    for (int j = 0; j < MTILE; ++j) sum[j] = 0.0f;

    for (int sb0 = 0; sb0 < num_sb; sb0 += chunk_sb) {
        int cur_sb = num_sb - sb0;
        if (cur_sb > chunk_sb) cur_sb = chunk_sb;
        for (int r = 0; r < ROWS_PER_BLOCK; ++r) {
            int n_r = blockIdx.x * ROWS_PER_BLOCK + r;
            size_t valid = (n_r < N) ? (size_t)cur_sb * (size_t)blk_bytes : 0;
            size_t base = (size_t)r * row_stride_B;
            const uint8_t* src_r = (const uint8_t*)weights
                + ((size_t)n_r * row_stride_sb + sb0) * blk_bytes;
            for (size_t i = tid; i < valid; i += (size_t)block_threads) {
                ws_base[base + i] = src_r[i];
            }
        }
        __syncthreads();

        if (n_valid) {
            const BlockType* wrow = (const BlockType*)(ws_base + (size_t)wave_id * row_stride_B);
            const int x_col0 = sb0 * 256;
            #pragma unroll
            for (int j = 0; j < MTILE; ++j) {
                int m = m_base + j;
                if (m >= Mrows) continue;
                const half* xm = x + (size_t)m * K_full + x_col0;
                float acc = 0.0f;
                for (int b = 0; b < cur_sb; ++b) {
                    acc += dot_op(wrow + b, &xm[b * 256], lane_id);
                }
                sum[j] += acc;
            }
        }
        __syncthreads();
    }

    if (!n_valid) return;
    #pragma unroll
    for (int j = 0; j < MTILE; ++j) {
        float s = sum[j];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            s += __shfl_down(s, offset, 32);
        }
        int m = m_base + j;
        if (lane_id == 0 && m < Mrows) {
            if (has_bias) s += __half2float(bias[n]);
            y[(size_t)m * N + n] = __float2half(s);
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
    extern __shared__ half x_shared[];
    const half* x0_vec = x;
    const half* x1_vec = (M == 2) ? (x + K_full) : nullptr;
    int tid = threadIdx.y * 32 + threadIdx.x;

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
        __syncthreads();
    }

    int wave_id = threadIdx.y;
    int lane_id = threadIdx.x;
    int n = blockIdx.x * ROWS_PER_BLOCK + wave_id;
    if (n >= N) return;

    int num_superblocks = K / 256;
    const block_iq3_s* row = weights + (size_t)n * row_stride_sb;

    dot_iq3_s dot_op(d_iq3s_grid);
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (int b = 0; b < num_superblocks; ++b) {
        sum0 += dot_op(row + b, &x0_vec[b * 256], lane_id);
        if constexpr (M == 2) {
            sum1 += dot_op(row + b, &x1_vec[b * 256], lane_id);
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
        sum0 += dot_op(row + b, &x0_vec[b * 256], lane_id);
        if constexpr (M == 2) {
            sum1 += dot_op(row + b, &x1_vec[b * 256], lane_id);
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
__global__ void gemv_iq4xs_kernel(
    const half* __restrict__ x,
    const block_iq4_xs* __restrict__ weights,
    half* __restrict__ y,
    const half* __restrict__ bias,
    int N, int K,
    int K_full, int row_stride_sb,
    bool has_bias,
    bool use_lds
) {
    __shared__ float s_kvalues[32];
    extern __shared__ half x_shared[];
    const half* x0_vec = x;
    const half* x1_vec = (M == 2) ? (x + K_full) : nullptr;
    int tid = threadIdx.y * 32 + threadIdx.x;

    if (tid < 16) {
        float val = (float)d_kvalues_iq4nl[tid];
        s_kvalues[tid] = val;
        s_kvalues[tid + 16] = val;
    }

    if (use_lds) {
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
    const block_iq4_xs* row = weights + (size_t)n * row_stride_sb;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (int b = 0; b < num_superblocks; ++b) {
        const block_iq4_xs* blk_ptr = row + b;
        uint64_t header = *reinterpret_cast<const uint64_t*>(blk_ptr);
        uint16_t d_u16 = (uint16_t)(header & 0xffff);
        float d = fp16_to_fp32(d_u16);
        uint32_t scales_h = (uint32_t)((header >> 16) & 0xffff);
        uint32_t scales_l = (uint32_t)(header >> 32);

        const half* x0_sb = &x0_vec[b * 256];
        const half* x1_sb = (M == 2) ? &x1_vec[b * 256] : nullptr;

        #pragma unroll
        for (int ib = 0; ib < 8; ++ib) {
            int ls_low = (scales_l >> (4 * ib)) & 0x0f;
            int ls_high = (scales_h >> (2 * ib)) & 0x03;
            int ls = ls_low | (ls_high << 4);
            float dl = d * (float)(ls - 32);

            int byte_idx = 16 * ib + (lane_id & 15);
            uint8_t q_byte = blk_ptr->qs[byte_idx];
            int nibble = (lane_id < 16) ? (q_byte & 0x0f) : (q_byte >> 4);

            float w = dl * s_kvalues[nibble];
            sum0 += w * __half2float(x0_sb[ib * 32 + lane_id]);
            if constexpr (M == 2) {
                sum1 += w * __half2float(x1_sb[ib * 32 + lane_id]);
            }
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
        constexpr int ROWS_8 = 8;
        dim3 block_8(32, ROWS_8);
        dim3 grid_8((N + ROWS_8 - 1) / ROWS_8);
        hipLaunchKernelGGL(
            (gemv_iq3s_kernel<ROWS_8, M>),
            grid_8, block_8, 0, stream,
            (const half*)input, (const block_iq3_s*)packed_weight, (half*)output, (const half*)bias,
            N, K, K_full, row_stride_sb, has_bias, false
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
            (gemv_iq4xs_kernel<ROWS_PER_BLOCK, M>),
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

// Batched-GEMM dispatch (P-prefill-1, qk flavor). MTILE=16 slabs over M;
// shared_bytes must match the kernel's chunk formula exactly.
// (gemv_q_sb_elems / gemv_q_block_bytes are defined below with decode.)
static int gemv_q_sb_elems(int ggml_type);
static int gemv_q_block_bytes(int ggml_type);
template <int MTILE>
static void dispatch_gemm_qk(
    const void* input,
    const void* packed_weight,
    void* output,
    const void* bias,
    int ggml_type,
    int N, int K, int Mrows,
    int K_full, int row_stride_sb,
    int blk_bytes,
    bool has_bias,
    hipStream_t stream
) {
    constexpr int ROWS_PER_BLOCK = 4;
    dim3 block(32, ROWS_PER_BLOCK);
    dim3 grid((N + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK,
              (Mrows + MTILE - 1) / MTILE);
    int num_sb = K / 256;
    int chunk_sb = (49152 / ROWS_PER_BLOCK) / blk_bytes;
    if (chunk_sb < 1) chunk_sb = 1;
    if (chunk_sb > num_sb) chunk_sb = num_sb;
    size_t row_stride_B = ((size_t)chunk_sb * (size_t)blk_bytes + 15) & ~(size_t)15;
    size_t shared_bytes = (size_t)ROWS_PER_BLOCK * row_stride_B;

    if (ggml_type == GGML_TYPE_Q4_K) {
        hipLaunchKernelGGL(
            (gemm_qk_kernel<block_q4_K, dot_q4_K, ROWS_PER_BLOCK, MTILE>),
            grid, block, shared_bytes, stream,
            (const half*)input, (const block_q4_K*)packed_weight, (half*)output, (const half*)bias,
            N, K, Mrows, K_full, row_stride_sb, blk_bytes, has_bias);
    } else if (ggml_type == GGML_TYPE_Q5_K) {
        hipLaunchKernelGGL(
            (gemm_qk_kernel<block_q5_K, dot_q5_K, ROWS_PER_BLOCK, MTILE>),
            grid, block, shared_bytes, stream,
            (const half*)input, (const block_q5_K*)packed_weight, (half*)output, (const half*)bias,
            N, K, Mrows, K_full, row_stride_sb, blk_bytes, has_bias);
    } else if (ggml_type == GGML_TYPE_Q6_K) {
        hipLaunchKernelGGL(
            (gemm_qk_kernel<block_q6_K, dot_q6_K, ROWS_PER_BLOCK, MTILE>),
            grid, block, shared_bytes, stream,
            (const half*)input, (const block_q6_K*)packed_weight, (half*)output, (const half*)bias,
            N, K, Mrows, K_full, row_stride_sb, blk_bytes, has_bias);
    } else if (ggml_type == GGML_TYPE_Q2_K) {
        hipLaunchKernelGGL(
            (gemm_qk_kernel<block_q2_K, dot_q2_K, ROWS_PER_BLOCK, MTILE>),
            grid, block, shared_bytes, stream,
            (const half*)input, (const block_q2_K*)packed_weight, (half*)output, (const half*)bias,
            N, K, Mrows, K_full, row_stride_sb, blk_bytes, has_bias);
    } else if (ggml_type == GGML_TYPE_IQ4_XS) {
        hipLaunchKernelGGL(
            (gemm_qk_kernel<block_iq4_xs, dot_iq4_xs, ROWS_PER_BLOCK, MTILE>),
            grid, block, shared_bytes, stream,
            (const half*)input, (const block_iq4_xs*)packed_weight, (half*)output, (const half*)bias,
            N, K, Mrows, K_full, row_stride_sb, blk_bytes, has_bias);
    } else if (ggml_type == GGML_TYPE_IQ2_XXS) {
        hipLaunchKernelGGL(
            (gemm_qk_kernel<block_iq2_xxs, dot_iq2_xxs, ROWS_PER_BLOCK, MTILE>),
            grid, block, shared_bytes, stream,
            (const half*)input, (const block_iq2_xxs*)packed_weight, (half*)output, (const half*)bias,
            N, K, Mrows, K_full, row_stride_sb, blk_bytes, has_bias);
    } else if (ggml_type == GGML_TYPE_IQ2_XS) {
        hipLaunchKernelGGL(
            (gemm_qk_kernel<block_iq2_xs, dot_iq2_xs, ROWS_PER_BLOCK, MTILE>),
            grid, block, shared_bytes, stream,
            (const half*)input, (const block_iq2_xs*)packed_weight, (half*)output, (const half*)bias,
            N, K, Mrows, K_full, row_stride_sb, blk_bytes, has_bias);
    } else if (ggml_type == GGML_TYPE_IQ2_S) {
        hipLaunchKernelGGL(
            (gemm_qk_kernel<block_iq2_s, dot_iq2_s, ROWS_PER_BLOCK, MTILE>),
            grid, block, shared_bytes, stream,
            (const half*)input, (const block_iq2_s*)packed_weight, (half*)output, (const half*)bias,
            N, K, Mrows, K_full, row_stride_sb, blk_bytes, has_bias);
    } else if (ggml_type == GGML_TYPE_IQ1_M) {
        hipLaunchKernelGGL(
            (gemm_qk_kernel<block_iq1_m, dot_iq1_m, ROWS_PER_BLOCK, MTILE>),
            grid, block, shared_bytes, stream,
            (const half*)input, (const block_iq1_m*)packed_weight, (half*)output, (const half*)bias,
            N, K, Mrows, K_full, row_stride_sb, blk_bytes, has_bias);
    }
}

extern "C" {

void launch_gemm_q_forward(
    const void* input,
    const void* packed_weight,
    void* output,
    const void* bias,
    int ggml_type,
    int Mrows,
    int N,
    int K,
    bool has_bias,
    hipStream_t stream
) {
    int sb = gemv_q_sb_elems(ggml_type);
    int blk = gemv_q_block_bytes(ggml_type);
    if (sb <= 0 || blk <= 0 || Mrows <= 0 || N <= 0 || K <= 0) return;
    int row_stride_sb = K / sb;
    if (ggml_type == GGML_TYPE_IQ3_S || ggml_type == GGML_TYPE_IQ3_XXS) {
        constexpr int ROWS = 4, MT = 16;
        dim3 block(32, ROWS);
        dim3 grid((N + ROWS - 1) / ROWS, (Mrows + MT - 1) / MT);
        int num_sb = K / 256;
        int chunk_sb = (49152 / ROWS) / blk;
        if (chunk_sb < 1) chunk_sb = 1;
        if (chunk_sb > num_sb) chunk_sb = num_sb;
        size_t row_stride_B = ((size_t)chunk_sb * (size_t)blk + 15) & ~(size_t)15;
        size_t shared_bytes = (size_t)ROWS * row_stride_B;
        if (ggml_type == GGML_TYPE_IQ3_S) {
            hipLaunchKernelGGL(
                (gemm_iq3_kernel<block_iq3_s, dot_iq3_s, ROWS, MT>),
                grid, block, shared_bytes, stream,
                (const half*)input, (const block_iq3_s*)packed_weight, (half*)output,
                (const half*)bias, N, K, Mrows, K, row_stride_sb, blk, has_bias);
        } else {
            hipLaunchKernelGGL(
                (gemm_iq3_kernel<block_iq3_xxs, dot_iq3_xxs, ROWS, MT>),
                grid, block, shared_bytes, stream,
                (const half*)input, (const block_iq3_xxs*)packed_weight, (half*)output,
                (const half*)bias, N, K, Mrows, K, row_stride_sb, blk, has_bias);
        }
        return;
    }
    dispatch_gemm_qk<16>(input, packed_weight, output, bias, ggml_type,
                         N, K, Mrows, K, row_stride_sb, blk, has_bias, stream);
}

} // extern "C"

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

// ---------------------------------------------------------------------------
// Fused SwiGLU activation: out = silu(g) * u (fp16, vectorized 8x half)
// Supports out == g or out == u for zero-allocation in-place execution.
// ---------------------------------------------------------------------------
__global__ void swiglu_kernel(
    const half* __restrict__ g,
    const half* __restrict__ u,
    half* __restrict__ out,
    int64_t total_elements
) {
    int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t idx = t * 8;
    if (idx >= total_elements) return;

    if (idx + 8 <= total_elements) {
        float4 g_raw = *reinterpret_cast<const float4*>(g + idx);
        float4 u_raw = *reinterpret_cast<const float4*>(u + idx);
        const half* gh = reinterpret_cast<const half*>(&g_raw);
        const half* uh = reinterpret_cast<const half*>(&u_raw);
        float4 out_raw;
        half* outh = reinterpret_cast<half*>(&out_raw);

        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float gf = __half2float(gh[k]);
            float uf = __half2float(uh[k]);
            float s = gf / (1.0f + expf(-gf));
            outh[k] = __float2half(s * uf);
        }
        *reinterpret_cast<float4*>(out + idx) = out_raw;
    } else {
        for (int64_t i = idx; i < total_elements; ++i) {
            float gf = __half2float(g[i]);
            float uf = __half2float(u[i]);
            float s = gf / (1.0f + expf(-gf));
            out[i] = __float2half(s * uf);
        }
    }
}

void launch_swiglu_forward(
    const void* g,
    const void* u,
    void* out,
    int64_t total_elements,
    hipStream_t stream
) {
    if (total_elements <= 0) return;
    int threads = 256;
    int64_t n_threads = (total_elements + 7) / 8;
    int blocks = (int)((n_threads + threads - 1) / threads);
    hipLaunchKernelGGL(swiglu_kernel, dim3(blocks), dim3(threads), 0, stream,
                       (const half*)g, (const half*)u, (half*)out, total_elements);
}

} // extern "C"
