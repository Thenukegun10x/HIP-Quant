// torch_ext/wave_attn_backward.hip
//
// WaveAttention native backward kernels for AMD GFX12 (RDNA4).
//
// Computes dQ, dK, dV given O, dO and saved Q/K/V/LSE from forward.
// Fully accelerated with FP8 WMMA intrinsics for all matrix multiplications:
//   - S  = Q @ K^T   (FP8 WMMA)
//   - dP = dO @ V^T  (FP8 WMMA)
//   - dQ = dS @ K    (FP8 WMMA)
//   - dK = dS^T @ Q  (FP8 WMMA)
//   - dV = P^T @ dO  (FP8 WMMA)
//
// Follows FlashAttention-2 exactly, including its separate preprocess pass for
//   D_i = sum_j P_ij * dP_ij = sum_d dO_id * O_id
// D is a full-row reduction, so it cannot be formed inside a K-tile loop; the
// preprocess kernel below materialises it up front as `Delta`.

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include "../hip_quant_util.h"

namespace cg = cooperative_groups;

typedef float v8f __attribute__((ext_vector_type(8)));
typedef int   v2i __attribute__((ext_vector_type(2)));

__device__ inline float fast_expf_bwd(float x) {
    return __builtin_exp2f(x * 1.4426950408889634f);
}

__device__ inline uint32_t pack_2float_to_fp8_e4m3_bwd(float f0, float f1) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    uint32_t res;
    asm volatile("v_cvt_pk_fp8_f32 %0, %1, %2" : "=v"(res) : "v"(f0), "v"(f1));
    return res & 0xFFFF;
#else
    uint8_t b0 = fp32_bits_to_fp8_e4m3(__float_as_int(f0));
    uint8_t b1 = fp32_bits_to_fp8_e4m3(__float_as_int(f1));
    return (uint32_t)b0 | ((uint32_t)b1 << 8);
#endif
}

__device__ inline uint32_t pack_4float_to_fp8_e4m3_bwd(float f0, float f1, float f2, float f3) {
    uint32_t p0 = pack_2float_to_fp8_e4m3_bwd(f0, f1);
    uint32_t p1 = pack_2float_to_fp8_e4m3_bwd(f2, f3);
    return (p0 & 0xFFFF) | (p1 << 16);
}

// LDS load helpers
// LDS bank-conflict padding.
//
// LDS is 32 banks x 4B = 128 bytes per bank row, so a tile stored at its
// natural pitch (Dim = 128, or K_TILE / 16 for the transposed staging buffers)
// makes the bank index independent of -- or only 2- to 8-valued in -- the
// lane's row index.  The 16 lanes of a WMMA operand fetch then collide on one
// or two banks.  An 8B pad makes the bank index (lw*2 + const) mod 32, giving
// 16 distinct banks whose two dwords each cover all 32 exactly once.
//
// Rows end up 8B- but not 16B-aligned, so the loader below writes two uint2s
// rather than one uint4.  Tiles are written once and re-read many times, so
// that trade strongly favours the read side.
#define WAVE_ATTN_LDS_PAD 8

__device__ inline void load_fp8_tile_to_lds(
    const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
    int g_row_start, int num_rows, int dim, int total_g_rows,
    int tid, int tid_stride, int dst_row_stride
) {
    int total_bytes = num_rows * dim;
    int num_u4 = total_bytes / 16;
    const uint4* g_u4 = (const uint4*)src;
    int shift = (dim == 128) ? 3 : 2;
    int mask  = (dim == 128) ? 7 : 3;
    for (int i = tid; i < num_u4; i += tid_stride) {
        int r = i >> shift;
        int c = (i & mask) << 4;
        int g_r = g_row_start + r;
        uint4 val = (g_r < total_g_rows && c < dim)
            ? g_u4[(g_r * dim + c) >> 4]
            : make_uint4(0,0,0,0);
        uint2* d2 = (uint2*)(dst + r * dst_row_stride + c);
        d2[0] = make_uint2(val.x, val.y);
        d2[1] = make_uint2(val.z, val.w);
    }
}

// FP8 E4M3 saturates at 448.  Target a little below that so the hardware
// v_cvt_pk_fp8_f32 rounding cannot push a near-amax value over the top.
#define WAVE_ATTN_FP8_TARGET 240.0f

// dS = P * (dP - D) is linear in dO, so a fixed dS scale cannot work: halving
// the gradient magnitude halves dS, and below |dO| ~ 1e-3 the whole tensor
// falls through E4M3's subnormal floor and quantises to zero (dV is unaffected
// -- it never passes through dS -- which is what makes the failure look
// selective).  Tying the dS scale to do_scale makes it track the gradient
// magnitude, so accuracy no longer depends on how far into training we are.
// Both factors are powers of two, so the product introduces no extra rounding.
// 0.5 reproduces the previously hand-tuned value of 16 at the |dO| ~ 1 that
// the original constant was chosen for.
#define WAVE_ATTN_DS_REL 0.5f

// dO is quantised to E4M3 before feeding the dP and dV WMMAs.  E4M3's smallest
// normal is 2^-6 and its smallest subnormal 2^-9, so an unscaled gradient
// tensor (values commonly <= 1e-3 in real training) would land almost entirely
// in subnormals or flush to zero.  Rescale by amax so the tensor spans E4M3's
// normal range; the factor is divided back out of dP and dV.
//
// The scale must be uniform over every row that contributes to a given output
// element -- dV[k,:] sums over all queries -- so it is per (batch, head), not
// per row.
// The factor is rounded down to a power of two.  Scaling a float by 2^n only
// shifts the exponent, so the E4M3 rounding of dO is bit-identical to what an
// unscaled cast would produce -- the rescale can only move values out of the
// subnormal range, never add a second rounding step.
__device__ inline float wave_attn_do_scale(const unsigned int* __restrict__ dO_amax_bits, int bh_idx) {
    float amax = __uint_as_float(dO_amax_bits[bh_idx]);
    if (!(amax > 0.0f && amax < 3.0e38f)) return 1.0f;
    int exp2_of;
    frexpf(WAVE_ATTN_FP8_TARGET / amax, &exp2_of);  // s = m * 2^e, m in [0.5, 1)
    return ldexpf(1.0f, exp2_of - 1);               // largest power of two <= s
}

// ═══════════════════════════════════════════════════════════════════════════
// Preprocess — Delta[b,h,q] = sum_d dO[q,d] * O[q,d], plus amax(|dO|) per (b,h)
// ═══════════════════════════════════════════════════════════════════════════
template <int THREADS>
__global__ void wave_attn_bwd_preprocess_kernel(
    const float*  __restrict__ O,
    const float*  __restrict__ dO,
    float*        __restrict__ Delta,
    unsigned int* __restrict__ dO_amax_bits,
    int Seq_Q, int Dim
) {
    int q_idx  = blockIdx.x;
    int bh_idx = blockIdx.y;

    int64_t row = ((int64_t)bh_idx * Seq_Q + q_idx) * Dim;

    float sum  = 0.0f;
    float amax = 0.0f;
    for (int d = threadIdx.x; d < Dim; d += THREADS) {
        float g = dO[row + d];
        sum  += g * O[row + d];
        amax  = fmaxf(amax, fabsf(g));
    }

    __shared__ float s_sum[THREADS];
    __shared__ float s_max[THREADS];
    s_sum[threadIdx.x] = sum;
    s_max[threadIdx.x] = amax;
    __syncthreads();

    #pragma unroll
    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
            s_max[threadIdx.x]  = fmaxf(s_max[threadIdx.x], s_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        Delta[(int64_t)bh_idx * Seq_Q + q_idx] = s_sum[0];
        // |dO| >= 0, and the IEEE-754 bit pattern of a non-negative float is
        // monotonic in its value, so an integer atomicMax is a valid float max.
        atomicMax(&dO_amax_bits[bh_idx], __float_as_uint(s_max[0]));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Quantise dO to E4M3 once, up front
// ═══════════════════════════════════════════════════════════════════════════
// Both main kernels consume dO as FP8.  Converting it inside them meant every
// (k-block, Q-tile) pair re-read the fp32 tensor and re-ran the cast through a
// scalar byte loop -- for Seq_K = 2048 that is 32x redundant work and 4x the
// global traffic of the FP8 form.  Doing it once here also lets both kernels
// use the vectorised uint4 LDS loader instead of a per-byte loop.
__global__ void wave_attn_bwd_quant_do_kernel(
    const float4* __restrict__ dO4,
    const unsigned int* __restrict__ dO_amax_bits,
    uchar4*       __restrict__ dO_fp8_4,
    int64_t total4, int64_t row_stride4
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total4) return;

    float do_scale = wave_attn_do_scale(dO_amax_bits, (int)(i / row_stride4));

    float4 g = dO4[i];
    uint32_t p = pack_4float_to_fp8_e4m3_bwd(g.x * do_scale, g.y * do_scale,
                                             g.z * do_scale, g.w * do_scale);
    uchar4 o;
    o.x = (uint8_t)(p & 0xFF);
    o.y = (uint8_t)((p >> 8)  & 0xFF);
    o.z = (uint8_t)((p >> 16) & 0xFF);
    o.w = (uint8_t)((p >> 24) & 0xFF);
    dO_fp8_4[i] = o;
}

// ═══════════════════════════════════════════════════════════════════════════
// dQ kernel — Q-tile outer (Full WMMA)
// ═══════════════════════════════════════════════════════════════════════════
template <int Q_TILE, int K_TILE, int THREADS_PER_BLOCK>
__global__ void wave_attn_bwd_dq_kernel(
    const uint8_t* __restrict__ Q_fp8,
    const uint8_t* __restrict__ K_fp8,
    const uint8_t* __restrict__ V_fp8,
    const uint8_t* __restrict__ dO_fp8,
    const float*   __restrict__ LSE,
    const float*   __restrict__ Delta,
    const unsigned int* __restrict__ dO_amax_bits,
    float*         __restrict__ dQ,
    int B, int H, int Seq_Q, int Seq_K, int Dim,
    float softmax_scale, float q_scale, float k_scale, float v_scale,
    bool is_causal
) {
    constexpr int NUM_K_SUBS = K_TILE / 16;

    extern __shared__ uint8_t smem_base[];
    // Padded LDS pitches -- see WAVE_ATTN_LDS_PAD.
    const int LD  = Dim + WAVE_ATTN_LDS_PAD;              // row-major tiles
    constexpr int LDT = K_TILE + WAVE_ATTN_LDS_PAD;       // transposed K

    uint8_t* smem_Q       = smem_base;
    uint8_t* smem_K       = smem_Q + Q_TILE * LD;
    uint8_t* smem_V       = smem_K + K_TILE * LD;
    uint8_t* smem_K_trans = smem_V + K_TILE * LD;
    uint8_t* smem_dO_fp8  = smem_K_trans + Dim * LDT;

    const cg::thread_block cta = cg::this_thread_block();

    int bh_idx        = blockIdx.x;
    int q_block_idx   = blockIdx.y;
    int q_block_start = q_block_idx * Q_TILE;
    if (q_block_start >= Seq_Q) return;

    int tid     = threadIdx.x;
    int wave_id = tid >> 5;
    int lane_id = tid & 31;
    int lw      = lane_id & 15;
    int lg      = lane_id >> 4;

    int b = bh_idx / H;
    int h = bh_idx % H;

    int64_t q_off   = ((int64_t)b * H + h) * Seq_Q * Dim;
    int64_t k_off   = ((int64_t)b * H + h) * Seq_K * Dim;
    int64_t v_off   = ((int64_t)b * H + h) * Seq_K * Dim;
    int64_t do_off  = ((int64_t)b * H + h) * Seq_Q * Dim;
    int64_t dq_off  = ((int64_t)b * H + h) * Seq_Q * Dim;
    int64_t lse_off = ((int64_t)b * H + h) * Seq_Q;

    float cscale = softmax_scale * q_scale * k_scale;
    int   num_d  = (Dim + 15) / 16;
    int   num_q_waves = Q_TILE >> 4;

    float do_scale = wave_attn_do_scale(dO_amax_bits, bh_idx);
    float ds_scale = WAVE_ATTN_DS_REL * do_scale;
    // dP = dO_true @ V_true^T, recovered from the FP8 operands:
    //   dO_fp8 = do_scale * dO_true,  V_true = v_scale * V_fp8
    float dp_scale = v_scale / do_scale;

    // 1. Load Q and (pre-quantised) dO tiles into LDS
    load_fp8_tile_to_lds(Q_fp8 + q_off, smem_Q, q_block_start, Q_TILE, Dim, Seq_Q, tid, THREADS_PER_BLOCK, LD);
    load_fp8_tile_to_lds(dO_fp8 + do_off, smem_dO_fp8, q_block_start, Q_TILE, Dim, Seq_Q, tid, THREADS_PER_BLOCK, LD);
    __syncthreads();

    v8f dq_acc[8];
    #pragma unroll
    for (int d = 0; d < 8; d++) dq_acc[d] = (v8f){};

    int total_k_tiles = (Seq_K + K_TILE - 1) / K_TILE;
    int q_r = q_block_start + wave_id * 16 + lw;
    // q_valid gates global loads/stores only.  It must never gate control flow:
    // it is divergent within a wave on the sequence tail, so branching on it
    // would desynchronise __syncthreads() and corrupt the shared-memory tiles.
    // Out-of-range lanes compute against zero-padded LDS and drop their stores.
    bool q_valid = wave_id < num_q_waves && q_r < Seq_Q;
    float lse_q   = q_valid ? LSE[lse_off + q_r]   : 0.0f;
    float delta_q = q_valid ? Delta[lse_off + q_r] : 0.0f;

    for (int kt = 0; kt < total_k_tiles; kt++) {
        int ks = kt * K_TILE;

        // Load K and V tiles
        load_fp8_tile_to_lds(K_fp8 + k_off, smem_K, ks, K_TILE, Dim, Seq_K, tid, THREADS_PER_BLOCK, LD);
        load_fp8_tile_to_lds(V_fp8 + v_off, smem_V, ks, K_TILE, Dim, Seq_K, tid, THREADS_PER_BLOCK, LD);
        __syncthreads();

        // Transpose smem_K -> smem_K_trans
        int total_k_u4 = (K_TILE * Dim) / 16;
        for (int i = tid; i < total_k_u4; i += THREADS_PER_BLOCK) {
            int r = i / (Dim / 16);
            int c = (i % (Dim / 16)) * 16;
            const uint4* k_u4 = (const uint4*)&smem_K[r * LD + c];
            uint4 val = *k_u4;
            uint8_t* bytes = (uint8_t*)&val;
            #pragma unroll
            for (int b_idx = 0; b_idx < 16; b_idx++) {
                smem_K_trans[(c + b_idx) * LDT + r] = bytes[b_idx];
            }
        }
        __syncthreads();

        // ── S = Q @ K^T (WMMA) ──────────────────────────────────────
        v8f sacc[NUM_K_SUBS];
        #pragma unroll
        for (int k_sub = 0; k_sub < NUM_K_SUBS; k_sub++) sacc[k_sub] = (v8f){};

        int q_lds_row = (wave_id < num_q_waves) ? (wave_id * 16 + lw) : 0;

        for (int dt = 0; dt < num_d; dt++) {
            int doff = dt * 16;
            v2i qv = (v2i){0, 0};
            if (doff + lg * 8 < Dim) {
                uint64_t q64 = *(const uint64_t*)&smem_Q[q_lds_row * LD + doff + lg * 8];
                qv = *(const v2i*)&q64;
            }
            #pragma unroll
            for (int k_sub = 0; k_sub < NUM_K_SUBS; k_sub++) {
                int kr = ks + k_sub * 16 + lw;
                v2i kv = (v2i){0, 0};
                if (kr < Seq_K && doff + lg * 8 < Dim) {
                    uint64_t k64 = *(const uint64_t*)&smem_K[(k_sub * 16 + lw) * LD + doff + lg * 8];
                    kv = *(const v2i*)&k64;
                }
                #if defined(__gfx1200__) || defined(__gfx1201__)
                sacc[k_sub] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(kv, qv, sacc[k_sub]);
                #endif
            }
        }

        // ── P = softmax(S, LSE) ──────────────────────────────────────
        float P_vals[NUM_K_SUBS][8];
        #pragma unroll
        for (int k_sub = 0; k_sub < NUM_K_SUBS; k_sub++) {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int kc = ks + k_sub * 16 + lg * 8 + i;
                float s = sacc[k_sub][i] * cscale;
                if (q_r >= Seq_Q || kc >= Seq_K || (is_causal && kc > q_r)) s = -1e30f;
                P_vals[k_sub][i] = (s > -1e29f) ? fast_expf_bwd(s - lse_q) : 0.0f;
            }
        }

        // ── dP = v_scale * dO @ V^T (WMMA using smem_dO_fp8) ────────
        v8f dP_acc[NUM_K_SUBS];
        #pragma unroll
        for (int k_sub = 0; k_sub < NUM_K_SUBS; k_sub++) dP_acc[k_sub] = (v8f){};

        for (int dt = 0; dt < num_d; dt++) {
            int doff = dt * 16;
            v2i do_fp8 = (v2i){0, 0};
            if (doff + lg * 8 < Dim) {
                uint64_t do64 = *(const uint64_t*)&smem_dO_fp8[q_lds_row * LD + doff + lg * 8];
                do_fp8 = *(const v2i*)&do64;
            }

            #pragma unroll
            for (int k_sub = 0; k_sub < NUM_K_SUBS; k_sub++) {
                int kr = ks + k_sub * 16 + lw;
                v2i vv = (v2i){0, 0};
                if (kr < Seq_K && doff + lg * 8 < Dim) {
                    uint64_t v64 = *(const uint64_t*)&smem_V[(k_sub * 16 + lw) * LD + doff + lg * 8];
                    vv = *(const v2i*)&v64;
                }
                #if defined(__gfx1200__) || defined(__gfx1201__)
                dP_acc[k_sub] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(vv, do_fp8, dP_acc[k_sub]);
                #endif
            }
        }

        #pragma unroll
        for (int k_sub = 0; k_sub < NUM_K_SUBS; k_sub++) {
            #pragma unroll
            for (int i = 0; i < 8; i++) dP_acc[k_sub][i] *= dp_scale;
        }

        // ── dS = P * (dP - D),  D = Delta[q] from the preprocess pass ─
        // D is a reduction over the whole key axis, so it must come from
        // outside this K-tile loop.
        v2i ds_fp8[NUM_K_SUBS];
        #pragma unroll
        for (int k_sub = 0; k_sub < NUM_K_SUBS; k_sub++) {
            float ds_vals[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int kc = ks + k_sub * 16 + lg * 8 + i;
                float ds = P_vals[k_sub][i] * (dP_acc[k_sub][i] - delta_q);
                if (q_r >= Seq_Q || kc >= Seq_K || (is_causal && kc > q_r)) ds = 0.0f;
                ds_vals[i] = ds * ds_scale;
            }
            uint32_t u0 = pack_4float_to_fp8_e4m3_bwd(ds_vals[0], ds_vals[1], ds_vals[2], ds_vals[3]);
            uint32_t u1 = pack_4float_to_fp8_e4m3_bwd(ds_vals[4], ds_vals[5], ds_vals[6], ds_vals[7]);
            ds_fp8[k_sub] = (v2i){(int)u0, (int)u1};
        }

        // ── dQ += cscale * dS @ K (WMMA!) ────────────────────────────
        for (int dt = 0; dt < num_d; dt++) {
            int doff = dt * 16;
            int vc = doff + lw;
            #pragma unroll
            for (int k_sub = 0; k_sub < NUM_K_SUBS; k_sub++) {
                v2i kv_trans = (v2i){0, 0};
                if (vc < Dim) {
                    uint64_t k64 = *(const uint64_t*)&smem_K_trans[vc * LDT + k_sub * 16 + lg * 8];
                    kv_trans = *(const v2i*)&k64;
                }
                #if defined(__gfx1200__) || defined(__gfx1201__)
                dq_acc[dt] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(ds_fp8[k_sub], kv_trans, dq_acc[dt]);
                #endif
            }
        }

        __syncthreads();
    }

    // dQ_true = softmax_scale * dS @ K_true, and K_true = k_scale * K_fp8.
    // No q_scale here: this is the gradient w.r.t. Q_true, not w.r.t. Q_fp8.
    float out_scale = (softmax_scale * k_scale) / ds_scale;
    if (wave_id >= num_q_waves) return;
    for (int dt = 0; dt < num_d; dt++) {
        int doff = dt * 16;
        #pragma unroll
        for (int d = 0; d < 8; d++) {
            int oc = doff + lw;
            int qr = q_block_start + wave_id * 16 + lg * 8 + d;
            if (qr < Seq_Q && oc < Dim)
                dQ[dq_off + (int64_t)qr * Dim + oc] = dq_acc[dt][d] * out_scale;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// dK/dV kernel — K-tile outer (Full WMMA)
// ═══════════════════════════════════════════════════════════════════════════
template <int K_TILE, int Q_TILE, int THREADS_PER_BLOCK>
__global__ void wave_attn_bwd_dkdv_kernel(
    const uint8_t* __restrict__ Q_fp8,
    const uint8_t* __restrict__ K_fp8,
    const uint8_t* __restrict__ V_fp8,
    const uint8_t* __restrict__ dO_fp8,
    const float*   __restrict__ LSE,
    const float*   __restrict__ Delta,
    const unsigned int* __restrict__ dO_amax_bits,
    float*         __restrict__ dK,
    float*         __restrict__ dV,
    int B, int H, int Seq_Q, int Seq_K, int Dim,
    float softmax_scale, float q_scale, float k_scale, float v_scale,
    bool is_causal
) {
    constexpr int NUM_K_SUBS = K_TILE / 16;
    constexpr int NUM_Q_SUBS = Q_TILE / 16;

    // Each wave owns one 16-key sub-tile and is the only writer of that slice
    // of dK/dV, so the block must have exactly one wave per sub-tile.  Fewer
    // waves silently leaves the uncovered keys at zero.
    static_assert(THREADS_PER_BLOCK / 32 == NUM_K_SUBS,
                  "dK/dV block needs exactly one wave per 16-key sub-tile");

    extern __shared__ uint8_t smem_base[];
    // Padded LDS pitches -- see WAVE_ATTN_LDS_PAD.  LDS16 is the pitch of the
    // 16-wide staging buffers (Q^T, dO^T, P^T, dS^T), whose natural 16B pitch
    // gives a 2-way conflict.
    const int LD = Dim + WAVE_ATTN_LDS_PAD;
    constexpr int LDS16 = 16 + WAVE_ATTN_LDS_PAD;

    uint8_t* smem_K        = smem_base;
    uint8_t* smem_V        = smem_K + K_TILE * LD;
    uint8_t* smem_Q        = smem_V + K_TILE * LD;
    uint8_t* smem_Q_trans  = smem_Q + Q_TILE * LD;
    uint8_t* smem_dO_fp8   = smem_Q_trans + Dim * LDS16;
    uint8_t* smem_dO_trans = smem_dO_fp8 + Q_TILE * LD;
    uint8_t* lds_P_trans   = smem_dO_trans + Dim * LDS16;
    uint8_t* lds_dS_trans  = lds_P_trans + K_TILE * LDS16;

    const cg::thread_block cta = cg::this_thread_block();

    int bh_idx       = blockIdx.x;
    int k_block_idx  = blockIdx.y;
    int k_block_start = k_block_idx * K_TILE;
    if (k_block_start >= Seq_K) return;

    int tid     = threadIdx.x;
    int wave_id = tid >> 5;
    int lane_id = tid & 31;
    int lw      = lane_id & 15;
    int lg      = lane_id >> 4;

    int b = bh_idx / H;
    int h = bh_idx % H;

    int64_t q_off   = ((int64_t)b * H + h) * Seq_Q * Dim;
    int64_t k_off   = ((int64_t)b * H + h) * Seq_K * Dim;
    int64_t v_off   = ((int64_t)b * H + h) * Seq_K * Dim;
    int64_t do_off  = ((int64_t)b * H + h) * Seq_Q * Dim;
    int64_t dk_off  = ((int64_t)b * H + h) * Seq_K * Dim;
    int64_t dv_off  = ((int64_t)b * H + h) * Seq_K * Dim;
    int64_t lse_off = ((int64_t)b * H + h) * Seq_Q;

    float cscale = softmax_scale * q_scale * k_scale;
    int   num_d  = (Dim + 15) / 16;
    int   num_k_waves  = K_TILE >> 4;

    constexpr float pscale = 16.0f;   // P is in [0, 1], so this stays fixed

    float do_scale = wave_attn_do_scale(dO_amax_bits, bh_idx);
    float ds_scale = WAVE_ATTN_DS_REL * do_scale;
    float dp_scale = v_scale / do_scale;

    // Load K and V tiles into LDS
    load_fp8_tile_to_lds(K_fp8 + k_off, smem_K, k_block_start, K_TILE, Dim, Seq_K, tid, THREADS_PER_BLOCK, LD);
    load_fp8_tile_to_lds(V_fp8 + v_off, smem_V, k_block_start, K_TILE, Dim, Seq_K, tid, THREADS_PER_BLOCK, LD);
    __syncthreads();

    // Out-of-range keys are handled purely by the `kr < Seq_K` guard on the
    // epilogue stores — see the matching note in the dQ kernel.  Every thread
    // must reach every __syncthreads() and take part in the LDS transposes
    // below, which are striped across the whole block, so key validity must
    // never gate control flow.

    v8f dk_acc[8], dv_acc[8];
    #pragma unroll
    for (int d = 0; d < 8; d++) { dk_acc[d] = (v8f){}; dv_acc[d] = (v8f){}; }

    int total_q_tiles = (Seq_Q + Q_TILE - 1) / Q_TILE;

    for (int qt = 0; qt < total_q_tiles; qt++) {
        int qs = qt * Q_TILE;

        // Load Q and (pre-quantised) dO tiles into LDS
        load_fp8_tile_to_lds(Q_fp8 + q_off, smem_Q, qs, Q_TILE, Dim, Seq_Q, tid, THREADS_PER_BLOCK, LD);
        load_fp8_tile_to_lds(dO_fp8 + do_off, smem_dO_fp8, qs, Q_TILE, Dim, Seq_Q, tid, THREADS_PER_BLOCK, LD);
        __syncthreads();

        #pragma unroll
        for (int q_sub = 0; q_sub < NUM_Q_SUBS; q_sub++) {
            int q_sub_start = qs + q_sub * 16;

            // Each wave owns exactly one 16-key sub-tile -- the same slice it
            // reads back for the dK/dV accumulation below.  Previously every
            // wave computed all NUM_K_SUBS sub-tiles and wrote byte-identical
            // results to the shared P/dS staging buffers, doing NUM_K_SUBS
            // times the necessary S and dP work.
            const int k_sub = wave_id;
            const int k_base = k_sub * 16;

            // ── S = Q @ K^T (WMMA) ──────────────────────────────────
            v8f sacc = (v8f){};

            for (int dt = 0; dt < num_d; dt++) {
                int doff = dt * 16;
                v2i qv = (v2i){0, 0};
                if (q_sub_start + lw < Seq_Q && doff + lg * 8 < Dim) {
                    uint64_t q64 = *(const uint64_t*)&smem_Q[(q_sub * 16 + lw) * LD + doff + lg * 8];
                    qv = *(const v2i*)&q64;
                }
                int kr = k_block_start + k_base + lw;
                v2i kv = (v2i){0, 0};
                if (kr < Seq_K && doff + lg * 8 < Dim) {
                    uint64_t k64 = *(const uint64_t*)&smem_K[(k_base + lw) * LD + doff + lg * 8];
                    kv = *(const v2i*)&k64;
                }
                #if defined(__gfx1200__) || defined(__gfx1201__)
                sacc = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(kv, qv, sacc);
                #endif
            }

            // ── P = softmax(S, LSE) ──────────────────────────────────
            int   qr_lane = q_sub_start + lw;
            float lse_q   = (qr_lane < Seq_Q) ? LSE[lse_off + qr_lane] : 0.0f;

            float P_vals[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int kc = k_block_start + k_base + lg * 8 + i;
                float s = sacc[i] * cscale;
                if (qr_lane >= Seq_Q || kc >= Seq_K || (is_causal && kc > qr_lane)) s = -1e30f;
                P_vals[i] = (s > -1e29f) ? fast_expf_bwd(s - lse_q) : 0.0f;
            }

            // ── dP = v_scale * dO @ V^T (WMMA using smem_dO_fp8) ────────
            v8f dP_acc = (v8f){};

            for (int dt = 0; dt < num_d; dt++) {
                int doff = dt * 16;
                v2i do_fp8 = (v2i){0, 0};
                if (q_sub_start + lw < Seq_Q && doff + lg * 8 < Dim) {
                    uint64_t do64 = *(const uint64_t*)&smem_dO_fp8[(q_sub * 16 + lw) * LD + doff + lg * 8];
                    do_fp8 = *(const v2i*)&do64;
                }
                int kr = k_block_start + k_base + lw;
                v2i vv = (v2i){0, 0};
                if (kr < Seq_K && doff + lg * 8 < Dim) {
                    uint64_t v64 = *(const uint64_t*)&smem_V[(k_base + lw) * LD + doff + lg * 8];
                    vv = *(const v2i*)&v64;
                }
                #if defined(__gfx1200__) || defined(__gfx1201__)
                dP_acc = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(vv, do_fp8, dP_acc);
                #endif
            }

            #pragma unroll
            for (int i = 0; i < 8; i++) dP_acc[i] *= dp_scale;

            // ── dS = P * (dP - D),  D = Delta[q] from the preprocess pass ─
            // This block only holds K_TILE keys, so the reduction can never be
            // formed here; Delta is the full-row value.
            float delta_q = (qr_lane < Seq_Q) ? Delta[lse_off + qr_lane] : 0.0f;

            // ── Store P^T and dS^T into LDS for WMMA ──────────────────
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int k_idx = k_base + lg * 8 + i;
                float p_val  = P_vals[i] * pscale;
                float ds_val = P_vals[i] * (dP_acc[i] - delta_q) * ds_scale;
                int kc = k_block_start + k_idx;
                if (qr_lane >= Seq_Q || kc >= Seq_K || (is_causal && kc > qr_lane)) {
                    p_val  = 0.0f;
                    ds_val = 0.0f;
                }
                uint32_t p_packed  = pack_2float_to_fp8_e4m3_bwd(p_val, 0.0f);
                uint32_t ds_packed = pack_2float_to_fp8_e4m3_bwd(ds_val, 0.0f);
                lds_P_trans [k_idx * LDS16 + lw] = (uint8_t)(p_packed & 0xFF);
                lds_dS_trans[k_idx * LDS16 + lw] = (uint8_t)(ds_packed & 0xFF);
            }

            // Transpose smem_Q sub-block -> smem_Q_trans
            int q_sub_u4 = (16 * Dim) / 16;
            for (int i = tid; i < q_sub_u4; i += THREADS_PER_BLOCK) {
                int r = i / (Dim / 16);
                int c = (i % (Dim / 16)) * 16;
                const uint4* q_u4 = (const uint4*)&smem_Q[(q_sub * 16 + r) * LD + c];
                uint4 val = *q_u4;
                uint8_t* bytes = (uint8_t*)&val;
                #pragma unroll
                for (int b_idx = 0; b_idx < 16; b_idx++) {
                    smem_Q_trans[(c + b_idx) * LDS16 + r] = bytes[b_idx];
                }
            }

            // Transpose smem_dO_fp8 sub-block -> smem_dO_trans
            for (int i = tid; i < q_sub_u4; i += THREADS_PER_BLOCK) {
                int r = i / (Dim / 16);
                int c = (i % (Dim / 16)) * 16;
                const uint4* do_u4 = (const uint4*)&smem_dO_fp8[(q_sub * 16 + r) * LD + c];
                uint4 val = *do_u4;
                uint8_t* bytes = (uint8_t*)&val;
                #pragma unroll
                for (int b_idx = 0; b_idx < 16; b_idx++) {
                    smem_dO_trans[(c + b_idx) * LDS16 + r] = bytes[b_idx];
                }
            }
            __syncthreads();

            // ── Accumulate dK and dV via WMMA! ───────────────────────
            v2i p_fp8  = (v2i){0, 0};
            v2i ds_fp8 = (v2i){0, 0};
            int k_idx_wave = wave_id * 16 + lw;
            if (k_idx_wave < K_TILE) {
                uint64_t p64  = *(const uint64_t*)&lds_P_trans [k_idx_wave * LDS16 + lg * 8];
                uint64_t ds64 = *(const uint64_t*)&lds_dS_trans[k_idx_wave * LDS16 + lg * 8];
                p_fp8  = *(const v2i*)&p64;
                ds_fp8 = *(const v2i*)&ds64;
            }

            for (int dt = 0; dt < num_d; dt++) {
                int doff = dt * 16;
                int vc = doff + lw;

                v2i do_trans = (v2i){0, 0};
                v2i qv_trans = (v2i){0, 0};
                if (vc < Dim) {
                    uint64_t do64 = *(const uint64_t*)&smem_dO_trans[vc * LDS16 + lg * 8];
                    uint64_t q64  = *(const uint64_t*)&smem_Q_trans [vc * LDS16 + lg * 8];
                    do_trans = *(const v2i*)&do64;
                    qv_trans = *(const v2i*)&q64;
                }

                #if defined(__gfx1200__) || defined(__gfx1201__)
                dv_acc[dt] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(p_fp8,  do_trans, dv_acc[dt]);
                dk_acc[dt] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(ds_fp8, qv_trans, dk_acc[dt]);
                #endif
            }

            __syncthreads();

        } // end Q sub-block

    } // end Q tile loop

    // dV_true = P^T @ dO_true, and dO_fp8 = do_scale * dO_true.  No v_scale:
    // V only enters dV through the softmax, which is already in P.
    float dv_out_scale = 1.0f / (pscale * do_scale);
    // dK_true = softmax_scale * dS^T @ Q_true, and Q_true = q_scale * Q_fp8.
    float dk_out_scale = (softmax_scale * q_scale) / ds_scale;

    if (wave_id >= num_k_waves) return;
    for (int dt = 0; dt < num_d; dt++) {
        int doff = dt * 16;
        #pragma unroll
        for (int d = 0; d < 8; d++) {
            int oc = doff + lw;
            int kr = k_block_start + wave_id * 16 + lg * 8 + d;
            if (kr < Seq_K && oc < Dim) {
                dK[dk_off + (int64_t)kr * Dim + oc] = dk_acc[dt][d] * dk_out_scale;
                dV[dv_off + (int64_t)kr * Dim + oc] = dv_acc[dt][d] * dv_out_scale;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Host launcher
// ═══════════════════════════════════════════════════════════════════════════
extern "C" {
void launch_wave_attn_backward(
    const uint8_t* Q_fp8, const uint8_t* K_fp8, const uint8_t* V_fp8,
    const float* O, const float* dO, const float* LSE,
    float* dQ, float* dK, float* dV,
    float* delta_scratch, unsigned int* do_amax_scratch, uint8_t* do_fp8_scratch,
    int B, int H, int Seq_Q, int Seq_K, int Dim,
    float softmax_scale, float q_scale, float k_scale, float v_scale,
    bool is_causal, hipStream_t stream
) {
    int Q_TILE = (Seq_Q <= 16) ? 16 : ((Seq_Q <= 32) ? 32 : 64);
    int K_TILE = 64;
    int THREADS = (Q_TILE <= 16) ? 32 : ((Q_TILE <= 32) ? 64 : 128);

    int num_q_blocks = (Seq_Q + Q_TILE - 1) / Q_TILE;
    int num_k_blocks = (Seq_K + K_TILE - 1) / K_TILE;

    const int LD    = Dim + WAVE_ATTN_LDS_PAD;
    const int LDT   = K_TILE + WAVE_ATTN_LDS_PAD;
    const int LDS16 = 16 + WAVE_ATTN_LDS_PAD;
    size_t smem_dq   = (size_t)(Q_TILE * LD + K_TILE * LD + K_TILE * LD + Dim * LDT + Q_TILE * LD);
    size_t smem_dkdv = (size_t)(K_TILE * LD + K_TILE * LD + Q_TILE * LD + Dim * LDS16
                              + Q_TILE * LD + Dim * LDS16 + K_TILE * LDS16 * 2);

    // ── Preprocess: Delta + amax(|dO|) ────────────────────────────────
    // Must complete before either kernel below reads them; same stream, so
    // the dependency is ordered without an explicit sync.
    {
        constexpr int PRE_THREADS = 128;
        dim3 grid(Seq_Q, B * H);
        dim3 block(PRE_THREADS);
        hipLaunchKernelGGL((wave_attn_bwd_preprocess_kernel<PRE_THREADS>),
            grid, block, 0, stream,
            O, dO, delta_scratch, do_amax_scratch, Seq_Q, Dim);
    }

    // ── Quantise dO once (needs the amax the pass above accumulated) ──
    {
        int64_t total4 = ((int64_t)B * H * Seq_Q * Dim) / 4;
        int64_t row_stride4 = ((int64_t)Seq_Q * Dim) / 4;
        constexpr int QT = 256;
        int blocks = (int)((total4 + QT - 1) / QT);
        hipLaunchKernelGGL(wave_attn_bwd_quant_do_kernel, dim3(blocks), dim3(QT), 0, stream,
            (const float4*)dO, do_amax_scratch, (uchar4*)do_fp8_scratch,
            total4, row_stride4);
    }

    // ── dQ kernel ─────────────────────────────────────────────────────
    {
        dim3 grid(B * H, num_q_blocks);
        dim3 block(THREADS);
        auto launch = [&](auto kernel) {
            hipLaunchKernelGGL(kernel, grid, block, smem_dq, stream,
                Q_fp8, K_fp8, V_fp8, do_fp8_scratch, LSE, delta_scratch, do_amax_scratch, dQ,
                B, H, Seq_Q, Seq_K, Dim,
                softmax_scale, q_scale, k_scale, v_scale, is_causal);
        };
        if      (Q_TILE == 16) launch(wave_attn_bwd_dq_kernel<16, 64, 32>);
        else if (Q_TILE == 32) launch(wave_attn_bwd_dq_kernel<32, 64, 64>);
        else                   launch(wave_attn_bwd_dq_kernel<64, 64, 128>);
    }

    // ── dK/dV kernel ──────────────────────────────────────────────────
    // Thread count is pinned to K_TILE/16 waves and is deliberately NOT
    // derived from Q_TILE: each wave owns one key sub-tile, so a short Seq_Q
    // must not be allowed to shrink the block below K_TILE/16 waves.
    {
        constexpr int DKDV_THREADS = (64 / 16) * 32;   // K_TILE / 16 waves
        dim3 grid(B * H, num_k_blocks);
        dim3 block(DKDV_THREADS);
        auto launch = [&](auto kernel) {
            hipLaunchKernelGGL(kernel, grid, block, smem_dkdv, stream,
                Q_fp8, K_fp8, V_fp8, do_fp8_scratch, LSE, delta_scratch, do_amax_scratch, dK, dV,
                B, H, Seq_Q, Seq_K, Dim,
                softmax_scale, q_scale, k_scale, v_scale, is_causal);
        };
        if      (Q_TILE == 16) launch(wave_attn_bwd_dkdv_kernel<64, 16, DKDV_THREADS>);
        else if (Q_TILE == 32) launch(wave_attn_bwd_dkdv_kernel<64, 32, DKDV_THREADS>);
        else                   launch(wave_attn_bwd_dkdv_kernel<64, 64, DKDV_THREADS>);
    }
}
} // extern "C"
