// torch_ext/wave_attn.hip
//
// WaveAttention: Native FP8 WMMA Flash Attention for AMD GFX12 (RDNA4).
//
// Optimizations:
//   1. Native v_wmma_f32_16x16x16_fp8_fp8 (bypasses AOTriton software decode)
//   2. fast_expf via __builtin_exp2f (replaces software expf)
//   3. Parallel K/V tile loads (split threads, 2× load throughput)
//   4. Unrolled V transpose load via uint4 global reads
//   5. Adaptive Q_TILE/K_TILE selection per sequence length
//   6. Online softmax with warp-level shuffle reduction

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include "../hip_quant_util.h"

#define HIP_QUANT_DTYPE_F32  0
#define HIP_QUANT_DTYPE_F16  1
#define HIP_QUANT_DTYPE_BF16 2
#define HIP_QUANT_DTYPE_FP8  3

namespace cg = cooperative_groups;

typedef float v8f __attribute__((ext_vector_type(8)));
typedef int   v2i __attribute__((ext_vector_type(2)));
typedef int   v4i __attribute__((ext_vector_type(4)));
typedef int   v8i __attribute__((ext_vector_type(8)));

__device__ inline float fast_expf(float x) {
    return __builtin_exp2f(x * 1.4426950408889634f);
}

static __device__ inline void store_attn_output_wave(void* dst, int dtype, int64_t idx, float value) {
    if (dtype == HIP_QUANT_DTYPE_F16) {
        ((uint16_t*)dst)[idx] = fp32_to_fp16(value);
    } else if (dtype == HIP_QUANT_DTYPE_BF16) {
        ((uint16_t*)dst)[idx] = fp32_to_bf16(value);
    } else {
        ((float*)dst)[idx] = value;
    }
}

// Native GFX12 1-cycle FP32 → FP8 E4M3 hardware packing
__device__ inline uint32_t fast_pack_2float_to_fp8_e4m3_wave(float f0, float f1) {
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

__device__ inline uint32_t pack_4float_to_fp8_e4m3_fast_wave(float f0, float f1, float f2, float f3) {
    uint32_t p0 = fast_pack_2float_to_fp8_e4m3_wave(f0, f1);
    uint32_t p1 = fast_pack_2float_to_fp8_e4m3_wave(f2, f3);
    return (p0 & 0xFFFF) | (p1 << 16);
}

// LDS bank-conflict padding.
//
// LDS is 32 banks x 4B = 128 bytes per bank row.  A tile stored with its
// natural row stride (Dim = 128, or K_TILE = 64 for the transposed V) makes the
// bank index independent of -- or only 2-valued in -- the lane's row index, so
// the 16 lanes of a WMMA operand fetch collide on one or two banks.  That is a
// 16-way (K) and 8-way (V) conflict on the single hottest load in the kernel.
//
// Padding the stride by 8B makes the bank index (lw*2 + const) mod 32 for K and
// (lw*18 + const) mod 32 for V -- 16 distinct banks, each lane's two dwords
// covering all 32 exactly once, i.e. fully conflict-free.  8B (rather than 16B)
// means LDS rows are only 8B-aligned, so the tile store below writes two uint2s
// instead of one uint4.  That trade is heavily favourable: each tile is written
// once but its WMMA operands are re-read ~8x.
#define WAVE_ATTN_LDS_PAD 8

// Division-free vectorized load from global → LDS using 128-bit (uint4) loads.
// `dst_row_stride` is the padded LDS row pitch in bytes (multiple of 16).
__device__ inline void load_tile_to_lds_wave_fast(
    const uint8_t* __restrict__ src_global,
    uint8_t* __restrict__ dst_lds,
    int g_row_start, int num_rows, int dim, int total_g_rows,
    int tid_start, int tid_stride, int tid_count, int dst_row_stride
) {
    int total_bytes = num_rows * dim;
    int num_u4 = total_bytes / 16;
    const uint4* g_u4 = (const uint4*)src_global;

    int u4_per_row_shift = (dim == 128) ? 3 : 2;
    int u4_per_row_mask  = (dim == 128) ? 7 : 3;

    for (int i = tid_start; i < num_u4; i += tid_stride) {
        int r = i >> u4_per_row_shift;
        int c = (i & u4_per_row_mask) << 4;
        int g_r = g_row_start + r;

        uint4 val;
        if (g_r < total_g_rows && c < dim) {
            val = g_u4[(g_r * dim + c) >> 4];
        } else {
            val = make_uint4(0, 0, 0, 0);
        }
        // Padded rows are 8B- but not 16B-aligned; split into two 8B stores.
        uint2* dst = (uint2*)(dst_lds + r * dst_row_stride + c);
        dst[0] = make_uint2(val.x, val.y);
        dst[1] = make_uint2(val.z, val.w);
    }
}

// INT4 Q/K tile load - packed 2 per byte, LD4 = Dim/2 + PAD
template <int K_TILE>
__device__ inline void load_int4_tile_to_lds(
    const uint8_t* __restrict__ src_global,
    uint8_t* __restrict__ dst_lds,
    int g_row_start, int num_rows, int dim, int total_g_rows,
    int tid_start, int tid_stride, int dst_row_stride
) {
    int dim2 = dim >> 1;
    int total_bytes = num_rows * dim2;
    int num_u4 = total_bytes >> 4;
    const uint4* g_u4 = (const uint4*)src_global;
    int u4_per_row_shift = (dim == 128) ? 2 : 1;
    int u4_per_row_mask  = (dim == 128) ? 3 : 1;
    for (int i = tid_start; i < num_u4; i += tid_stride) {
        int r = i >> u4_per_row_shift;
        int c_bytes = (i & u4_per_row_mask) << 4;
        int g_r = g_row_start + r;
        uint4 val;
        if (g_r < total_g_rows && (c_bytes<<1) < dim) {
            val = g_u4[((g_r * dim2 + c_bytes) >> 4)];
        } else {
            val = make_uint4(0,0,0,0);
        }
        uint2* dst = (uint2*)(dst_lds + r * dst_row_stride + c_bytes);
        dst[0] = make_uint2(val.x, val.y);
        dst[1] = make_uint2(val.z, val.w);
    }
}

// INT4 -> FP8 E4M3 lookup table (int4 0..15 where 0=0, 7=7, 8=-8, 15=-1)
__device__ const uint8_t INT4_TO_FP8_LUT[16] = {0,56,64,68,72,74,76,78,208,206,204,202,200,196,192,184};

__device__ inline void unpack_packed32_to_fp8_pair(uint32_t packed, uint32_t &out0, uint32_t &out1) {
    uint8_t b0 =  packed        & 0xFF;
    uint8_t b1 = (packed >> 8)  & 0xFF;
    uint8_t b2 = (packed >> 16) & 0xFF;
    uint8_t b3 = (packed >> 24) & 0xFF;
    uint8_t lo0 = INT4_TO_FP8_LUT[b0 & 0xF];
    uint8_t hi0 = INT4_TO_FP8_LUT[b0 >> 4];
    uint8_t lo1 = INT4_TO_FP8_LUT[b1 & 0xF];
    uint8_t hi1 = INT4_TO_FP8_LUT[b1 >> 4];
    uint8_t lo2 = INT4_TO_FP8_LUT[b2 & 0xF];
    uint8_t hi2 = INT4_TO_FP8_LUT[b2 >> 4];
    uint8_t lo3 = INT4_TO_FP8_LUT[b3 & 0xF];
    uint8_t hi3 = INT4_TO_FP8_LUT[b3 >> 4];
    out0 = (uint32_t)lo0 | ((uint32_t)hi0 << 8) | ((uint32_t)lo1 << 16) | ((uint32_t)hi1 << 24);
    out1 = (uint32_t)lo2 | ((uint32_t)hi2 << 8) | ((uint32_t)lo3 << 16) | ((uint32_t)hi3 << 24);
}

// Vectorized unpack of a packed INT4 tile (LD4) to FP8 tile (LD) - processes 4 packed bytes (8 FP8) per iter
__device__ inline void unpack_int4_tile_to_fp8_vec(
    uint8_t* __restrict__ dst_lds, const uint8_t* __restrict__ src_lds,
    int num_rows, int dim, int dst_stride, int src_stride,
    int tid, int stride) {
    int dim2 = dim >> 1;
    for (int r = tid; r < num_rows; r += stride) {
        int c = 0;
        // bulk 4-packed-bytes -> 8 FP8 bytes using 1x uint32 load / 2x uint32 store
        for (; c + 4 <= dim2; c += 4) {
            uint32_t packed = *(const uint32_t*)&src_lds[r * src_stride + c];
            uint32_t o0, o1;
            unpack_packed32_to_fp8_pair(packed, o0, o1);
            *(uint32_t*)&dst_lds[r * dst_stride + c*2    ] = o0;
            *(uint32_t*)&dst_lds[r * dst_stride + c*2 + 4] = o1;
        }
        // remainder (dim2 % 4 == 0 for Dim 64/128 so never runs, but keep for safety)
        for (; c < dim2; ++c) {
            uint8_t packed = src_lds[r * src_stride + c];
            dst_lds[r * dst_stride + c*2    ] = INT4_TO_FP8_LUT[packed & 0xF];
            dst_lds[r * dst_stride + c*2 + 1] = INT4_TO_FP8_LUT[packed >> 4];
        }
    }
}

// Transposed V load: uint4 global read → unrolled LDS stores.
template <int K_TILE, int LDV>
__device__ inline void load_v_tile_transposed_wave_fast(
    const uint8_t* __restrict__ src_global,
    uint8_t* __restrict__ dst_lds,
    int g_row_start, int num_rows, int dim, int total_g_rows,
    int tid_start, int tid_stride, int tid_count
) {
    int total_bytes = num_rows * dim;
    int num_u4 = total_bytes / 16;
    const uint4* g_u4 = (const uint4*)src_global;

    int u4_per_row_shift = (dim == 128) ? 3 : 2;
    int u4_per_row_mask  = (dim == 128) ? 7 : 3;

    for (int i = tid_start; i < num_u4; i += tid_stride) {
        int r = i >> u4_per_row_shift;
        int c = (i & u4_per_row_mask) << 4;
        int g_r = g_row_start + r;

        uint4 u4;
        if (g_r < total_g_rows && c < dim) {
            u4 = g_u4[(g_r * dim + c) >> 4];
        } else {
            u4 = make_uint4(0, 0, 0, 0);
        }

        uint8_t* bytes = (uint8_t*)&u4;
        int col0 = c * LDV + r;
        #pragma unroll
        for (int k = 0; k < 16; k += 4) {
            dst_lds[col0 + (k+0) * LDV] = bytes[k+0];
            dst_lds[col0 + (k+1) * LDV] = bytes[k+1];
            dst_lds[col0 + (k+2) * LDV] = bytes[k+2];
            dst_lds[col0 + (k+3) * LDV] = bytes[k+3];
        }
    }
}

// Vectorized V transpose via 4x4 byte shuffle + ds_write_b32/b128.
template <int K_TILE, int LDV>
__device__ inline void load_v_tile_transposed_wave_fast_vec(
    const uint8_t* __restrict__ src_global,
    uint8_t* __restrict__ dst_lds,
    int g_row_start, int num_rows, int dim, int total_g_rows,
    int tid_start, int tid_stride, int tid_count
) {
    const uint4* g_u4 = (const uint4*)src_global;
    int blocks_per_row = dim >> 4;
    int block_rows = K_TILE >> 2;
    int total_blocks = block_rows * blocks_per_row;
    for (int b = tid_start; b < total_blocks; b += tid_stride) {
        int block_r = b / blocks_per_row;
        int block_c = b % blocks_per_row;
        int r0 = block_r * 4;
        int c0 = block_c * 16;
        int g_r0 = g_row_start + r0;
        uint4 u4_0, u4_1, u4_2, u4_3;
        if (g_r0 < total_g_rows) {
            u4_0 = (c0 < dim) ? g_u4[((g_r0)*dim + c0)>>4] : make_uint4(0,0,0,0);
        } else u4_0 = make_uint4(0,0,0,0);
        if (g_r0+1 < total_g_rows) {
            u4_1 = (c0 < dim) ? g_u4[((g_r0+1)*dim + c0)>>4] : make_uint4(0,0,0,0);
        } else u4_1 = make_uint4(0,0,0,0);
        if (g_r0+2 < total_g_rows) {
            u4_2 = (c0 < dim) ? g_u4[((g_r0+2)*dim + c0)>>4] : make_uint4(0,0,0,0);
        } else u4_2 = make_uint4(0,0,0,0);
        if (g_r0+3 < total_g_rows) {
            u4_3 = (c0 < dim) ? g_u4[((g_r0+3)*dim + c0)>>4] : make_uint4(0,0,0,0);
        } else u4_3 = make_uint4(0,0,0,0);
        {
            uint32_t a = u4_0.x, b = u4_1.x, c = u4_2.x, d = u4_3.x;
            uint32_t t0 = (a & 0xFF) | ((b & 0xFF) << 8) | ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
            uint32_t t1 = ((a>>8)&0xFF) | (((b>>8)&0xFF)<<8) | (((c>>8)&0xFF)<<16) | (((d>>8)&0xFF)<<24);
            uint32_t t2 = ((a>>16)&0xFF) | (((b>>16)&0xFF)<<8) | (((c>>16)&0xFF)<<16) | (((d>>16)&0xFF)<<24);
            uint32_t t3 = ((a>>24)&0xFF) | (((b>>24)&0xFF)<<8) | (((c>>24)&0xFF)<<16) | (((d>>24)&0xFF)<<24);
            *(uint32_t*)&dst_lds[(c0+0)*LDV + r0] = t0;
            *(uint32_t*)&dst_lds[(c0+1)*LDV + r0] = t1;
            *(uint32_t*)&dst_lds[(c0+2)*LDV + r0] = t2;
            *(uint32_t*)&dst_lds[(c0+3)*LDV + r0] = t3;
        }
        {
            uint32_t a = u4_0.y, b = u4_1.y, c = u4_2.y, d = u4_3.y;
            uint32_t t0 = (a & 0xFF) | ((b & 0xFF) << 8) | ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
            uint32_t t1 = ((a>>8)&0xFF) | (((b>>8)&0xFF)<<8) | (((c>>8)&0xFF)<<16) | (((d>>8)&0xFF)<<24);
            uint32_t t2 = ((a>>16)&0xFF) | (((b>>16)&0xFF)<<8) | (((c>>16)&0xFF)<<16) | (((d>>16)&0xFF)<<24);
            uint32_t t3 = ((a>>24)&0xFF) | (((b>>24)&0xFF)<<8) | (((c>>24)&0xFF)<<16) | (((d>>24)&0xFF)<<24);
            *(uint32_t*)&dst_lds[(c0+4)*LDV + r0] = t0;
            *(uint32_t*)&dst_lds[(c0+5)*LDV + r0] = t1;
            *(uint32_t*)&dst_lds[(c0+6)*LDV + r0] = t2;
            *(uint32_t*)&dst_lds[(c0+7)*LDV + r0] = t3;
        }
        {
            uint32_t a = u4_0.z, b = u4_1.z, c = u4_2.z, d = u4_3.z;
            uint32_t t0 = (a & 0xFF) | ((b & 0xFF) << 8) | ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
            uint32_t t1 = ((a>>8)&0xFF) | (((b>>8)&0xFF)<<8) | (((c>>8)&0xFF)<<16) | (((d>>8)&0xFF)<<24);
            uint32_t t2 = ((a>>16)&0xFF) | (((b>>16)&0xFF)<<8) | (((c>>16)&0xFF)<<16) | (((d>>16)&0xFF)<<24);
            uint32_t t3 = ((a>>24)&0xFF) | (((b>>24)&0xFF)<<8) | (((c>>24)&0xFF)<<16) | (((d>>24)&0xFF)<<24);
            *(uint32_t*)&dst_lds[(c0+8)*LDV + r0] = t0;
            *(uint32_t*)&dst_lds[(c0+9)*LDV + r0] = t1;
            *(uint32_t*)&dst_lds[(c0+10)*LDV + r0] = t2;
            *(uint32_t*)&dst_lds[(c0+11)*LDV + r0] = t3;
        }
        {
            uint32_t a = u4_0.w, b = u4_1.w, c = u4_2.w, d = u4_3.w;
            uint32_t t0 = (a & 0xFF) | ((b & 0xFF) << 8) | ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
            uint32_t t1 = ((a>>8)&0xFF) | (((b>>8)&0xFF)<<8) | (((c>>8)&0xFF)<<16) | (((d>>8)&0xFF)<<24);
            uint32_t t2 = ((a>>16)&0xFF) | (((b>>16)&0xFF)<<8) | (((c>>16)&0xFF)<<16) | (((d>>16)&0xFF)<<24);
            uint32_t t3 = ((a>>24)&0xFF) | (((b>>24)&0xFF)<<8) | (((c>>24)&0xFF)<<16) | (((d>>24)&0xFF)<<24);
            *(uint32_t*)&dst_lds[(c0+12)*LDV + r0] = t0;
            *(uint32_t*)&dst_lds[(c0+13)*LDV + r0] = t1;
            *(uint32_t*)&dst_lds[(c0+14)*LDV + r0] = t2;
            *(uint32_t*)&dst_lds[(c0+15)*LDV + r0] = t3;
        }
    }
}


template <int Q_TILE, int K_TILE, int THREADS_PER_BLOCK>
__global__ void wave_attn_kernel_templated(
    const uint8_t * __restrict__ Q_fp8,
    const uint8_t * __restrict__ K_fp8,
    const uint8_t * __restrict__ V_fp8,
    void          * __restrict__ Out,
    float         * __restrict__ LSE,
    float         * __restrict__ Partial_Max,
    float         * __restrict__ Partial_Sum,
    float         * __restrict__ Partial_Out,
    int B, int H, int Seq_Q, int Seq_K, int Dim,
    float softmax_scale, float q_scale, float k_scale, float v_scale,
    int out_dtype, bool is_causal, int num_splits
) {
    constexpr int NUM_K_SUBS = K_TILE / 16;
    constexpr int NUM_Q_SUBS = Q_TILE / 16;

    // The Q tile is never shared between waves -- wave w reads only LDS rows
    // w*16 .. w*16+15 and nothing else ever touches them -- so keeping it
    // resident in LDS for the whole K loop just burns the scarcest resource.
    // It is staged through the K/V region during setup, pulled into registers,
    // and the space is then reused.  LDS drops from
    //     Q_TILE*Dim + 2*K_TILE*Dim   to   max(Q_TILE*Dim, 2*K_TILE*Dim).
    // Padded LDS pitches -- see WAVE_ATTN_LDS_PAD.  LDK is the K row pitch;
    // LDV is the column pitch of the transposed V tile.
    const int LDK = Dim + WAVE_ATTN_LDS_PAD;
    constexpr int LDV = K_TILE + WAVE_ATTN_LDS_PAD;

    // Sequential K/V loading: K tile is loaded first, then V tile reuses
    // the same LDS region after S/softmax.  All threads participate in
    // each load phase (no split), so each thread does half as many uint4
    // loads as before.  LDS footprint is max(K, V) instead of K+V.
    extern __shared__ uint8_t smem_base[];
    uint8_t* smem_K    = smem_base;       // K tile (Phase 1), then V tile (Phase 3)

    const cg::thread_block cta = cg::this_thread_block();
    const cg::thread_group wmma_wave = cg::tiled_partition(cta, 32);

    int split_idx = blockIdx.x % num_splits;
    int bh_idx = blockIdx.x / num_splits;
    int q_block_idx = blockIdx.y;
    int q_block_start = q_block_idx * Q_TILE;
    if (q_block_start >= Seq_Q) return;

    int tid = threadIdx.x;
    int wave_id = tid >> 5;
    int lane_id = tid & 31;
    int lw = lane_id & 15;
    int lg = lane_id >> 4;

    int b = bh_idx / H;
    int h = bh_idx % H;

    int64_t q_off = ((int64_t)b * H + h) * Seq_Q * Dim;
    int64_t k_off = ((int64_t)b * H + h) * Seq_K * Dim;
    int64_t v_off = ((int64_t)b * H + h) * Seq_K * Dim;
    int64_t o_off = ((int64_t)b * H + h) * Seq_Q * Dim;

    float cscale = softmax_scale * q_scale * k_scale;
    const float log_pscale = 2.77258872224f;
    float pscale = 16.0f;
    int num_d = (Dim + 15) / 16;
    int num_q_waves = Q_TILE >> 4;

    // Q is reloaded from global memory (FP8, L2-cached) on each WMMA
    // iteration instead of pre-loaded into a 16-VGPR q_reg[8] array.
    // This keeps total register pressure within the wave32 budget and
    // avoids the LDS occupancy cost of a separate Q tile region.
    const uint8_t* __restrict__ Q_base = Q_fp8 + q_off;

    v8f acc_o[8];
    #pragma unroll
    for (int d = 0; d < 8; d++) acc_o[d] = (v8f){};

    float row_max = -1e30f;
    float row_sum = 0.0f;

    int total_k_tiles = (Seq_K + K_TILE - 1) / K_TILE;
    int tiles_per_split = (total_k_tiles + num_splits - 1) / num_splits;
    int kt_start = split_idx * tiles_per_split;
    int kt_end = (split_idx + 1) * tiles_per_split;
    if (kt_end > total_k_tiles) kt_end = total_k_tiles;

    int q_r = q_block_start + wave_id * 16 + lw;

    // Decode support: when Seq_Q < Seq_K the Q tile covers the LAST Seq_Q
    // keys (absolute positions q_start .. Seq_K-1). q_start is block-uniform
    // (kernel args only); the mask below stays a per-lane expression, so no
    // new divergent control flow around __syncthreads/__shfl_sync.
    const int q_start = (Seq_K > Seq_Q) ? (Seq_K - Seq_Q) : 0;

    if (is_causal) {
        int max_q_in_block = q_start + q_block_start + Q_TILE - 1;
        int m = (max_q_in_block + K_TILE - 1) / K_TILE;
        if (kt_end > m) kt_end = m;
    }

    // ─── outer loop over K/V tiles ──────────────────────────────────
    for (int kt = kt_start; kt < kt_end; kt++) {
        int ks = kt * K_TILE;

        // ── Load K tile (all threads) ────────────────────────────
        load_tile_to_lds_wave_fast(K_fp8 + k_off, smem_K, ks, K_TILE, Dim, Seq_K, tid, THREADS_PER_BLOCK, THREADS_PER_BLOCK, LDK);
        cta.sync();

        // ── Phase 1: S = Q @ K^T  (FP8 WMMA + scale + running max) ──
        v8f sacc[NUM_K_SUBS];
        float tile_max = -1e30f;
        #pragma unroll
        for (int k_sub = 0; k_sub < NUM_K_SUBS; k_sub++) sacc[k_sub] = (v8f){};

        // Q is reloaded from global memory (FP8, L2-cached) on each
        // d-sub-tile iteration instead of pre-loaded into q_reg[8].
        // This saves 14 VGPRs (16→2), bringing register pressure below
        // 128 and eliminating scratch spilling.
        #pragma unroll
        for (int dt = 0; dt < 8; dt++) {
            if (dt >= num_d) break;
            int doff = dt * 16;
            v2i qv = (v2i){0, 0};
            if (q_r < Seq_Q && doff + lg * 8 < Dim) {
                uint64_t q64 = *(const uint64_t*)&Q_base[(int64_t)q_r * Dim + doff + lg * 8];
                qv = *(const v2i*)&q64;
            }

            #pragma unroll
            for (int k_sub = 0; k_sub < NUM_K_SUBS; k_sub++) {
                int kr = ks + k_sub * 16 + lw;
                v2i kv = (v2i){0, 0};
                if (kr < Seq_K && doff + lg * 8 < Dim) {
                    uint64_t k64 = *(const uint64_t*)&smem_K[(k_sub * 16 + lw) * LDK + doff + lg * 8];
                    kv = *(const v2i*)&k64;
                }
                #if defined(__gfx1200__) || defined(__gfx1201__)
                sacc[k_sub] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(kv, qv, sacc[k_sub]);
                #endif
            }
        }

        // ── Phase 2: scale + running max (single fused pass) ────────
        // Reuse sacc for scaled values — sacc is dead after Phase 1,
        // so writing the scaled scores back into it saves 32 VGPRs
        // vs a separate scaled[NUM_K_SUBS][8] array.
        #pragma unroll
        for (int k_sub = 0; k_sub < NUM_K_SUBS; k_sub++) {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int kc = ks + k_sub * 16 + lg * 8 + i;
                float s = sacc[k_sub][i] * cscale;
                if (kc >= Seq_K || (is_causal && kc > q_start + q_r)) s = -1e30f;
                sacc[k_sub][i] = s;  // reuse sacc as scaled storage
                tile_max = fmaxf(tile_max, s);
            }
        }

        // ── Online softmax (warp shuffle reduction + exp + sum) ─────
        float r_max = fmaxf(tile_max, __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, tile_max, 16));
        float nmax = fmaxf(row_max, r_max);
        float alp = (row_max > -1e29f) ? fast_expf(row_max - nmax) : 0.0f;
        float alpha = alp;
        row_max = nmax;

        float row_sum_new = row_sum * alp;

        v2i pv[NUM_K_SUBS];
        #pragma unroll
        for (int k_sub = 0; k_sub < NUM_K_SUBS; k_sub++) {
            float p_vals[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                float pv_f = (sacc[k_sub][i] > -1e29f) ? fast_expf(sacc[k_sub][i] - nmax) : 0.0f;
                p_vals[i] = pv_f;
                row_sum_new += pv_f;
            }
            uint32_t u0 = pack_4float_to_fp8_e4m3_fast_wave(p_vals[0] * pscale, p_vals[1] * pscale, p_vals[2] * pscale, p_vals[3] * pscale);
            uint32_t u1 = pack_4float_to_fp8_e4m3_fast_wave(p_vals[4] * pscale, p_vals[5] * pscale, p_vals[6] * pscale, p_vals[7] * pscale);
            pv[k_sub] = (v2i){(int)u0, (int)u1};
        }
        row_sum = row_sum_new;

        // ── Alpha rescaling: fuse into P@V WMMA operand preparation ──
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float corr_alpha = __shfl_sync(0xFFFFFFFFFFFFFFFFULL, alpha, lg * 8 + i);
            #pragma unroll
            for (int d = 0; d < 8; d++) {
                acc_o[d][i] *= corr_alpha;
            }
        }

        // ── Load V tile (all threads, reuses K's LDS) ──────────
        load_v_tile_transposed_wave_fast<K_TILE, LDV>(V_fp8 + v_off, smem_K, ks, K_TILE, Dim, Seq_K,
            tid, THREADS_PER_BLOCK, THREADS_PER_BLOCK);
        cta.sync();

        // ── Phase 3: O += P @ V  (FP8 WMMA) ────────────────────────
        for (int dt = 0; dt < num_d; dt++) {
            int doff = dt * 16;
            int vc = doff + lw;

            #pragma unroll
            for (int k_sub = 0; k_sub < NUM_K_SUBS; k_sub++) {
                v2i vv = (v2i){0, 0};
                if (vc < Dim) {
                    uint64_t v64 = *(const uint64_t*)&smem_K[vc * LDV + k_sub * 16 + lg * 8];
                    vv = *(const v2i*)&v64;
                }
                #if defined(__gfx1200__) || defined(__gfx1201__)
                acc_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(pv[k_sub], vv, acc_o[dt]);
                #endif
            }
        }

        cta.sync();
    }

    // Wave-uniform, so this early-out is safe ahead of the broadcasts below.
    if (wave_id >= num_q_waves) return;

    float total_row_sum = row_sum + __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, row_sum, 16);

    if (num_splits == 1) {
        for (int dt = 0; dt < num_d; dt++) {
            int doff = dt * 16;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int orr = q_block_start + wave_id * 16 + lg * 8 + i;
                int oc = doff + lw;
                float rsum = __shfl_sync(0xFFFFFFFFFFFFFFFFULL, total_row_sum, lg * 8 + i);
                float rmax = __shfl_sync(0xFFFFFFFFFFFFFFFFULL, row_max, lg * 8 + i);
                float inv = (rsum > 0.0f) ? (1.0f / (rsum * pscale)) : 0.0f;
                if (orr < Seq_Q && oc < Dim) {
                    float fval = acc_o[dt][i] * inv * v_scale;
                    store_attn_output_wave(Out, out_dtype, o_off + (int64_t)orr * Dim + oc, fval);
                }
                if (LSE != nullptr && oc == 0 && dt == 0 && orr < Seq_Q) {
                    float lse_val = rmax + logf(rsum + 1e-12f);
                    LSE[(int64_t)(b * H + h) * Seq_Q + orr] = lse_val;
                }
            }
        }
    } else {
        for (int dt = 0; dt < num_d; dt++) {
            int doff = dt * 16;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int orr = q_block_start + wave_id * 16 + lg * 8 + i;
                int oc = doff + lw;
                float rmax = __shfl_sync(0xFFFFFFFFFFFFFFFFULL, row_max, lg * 8 + i);
                float rsum = __shfl_sync(0xFFFFFFFFFFFFFFFFULL, total_row_sum, lg * 8 + i);
                if (orr < Seq_Q && oc < Dim) {
                    int64_t global_q_idx = (int64_t)(b * H + h) * Seq_Q + orr;
                    if (oc == 0) {
                        Partial_Max[global_q_idx * num_splits + split_idx] = rmax;
                        Partial_Sum[global_q_idx * num_splits + split_idx] = rsum;
                    }
                    int64_t out_elem_idx = global_q_idx * Dim + oc;
                    Partial_Out[out_elem_idx * num_splits + split_idx] = acc_o[dt][i];
                }
            }
        }
    }
}


template <int Q_TILE, int K_TILE, int THREADS_PER_BLOCK>
__global__ void wave_attn_kernel_templated_int4(
    const uint8_t* __restrict__ Q_int4,
    const uint8_t* __restrict__ K_int4,
    const uint8_t* __restrict__ V_fp8,
    void* __restrict__ Out,
    float* __restrict__ LSE,
    float* __restrict__ Partial_Max,
    float* __restrict__ Partial_Sum,
    float* __restrict__ Partial_Out,
    int B, int H, int Seq_Q, int Seq_K, int Dim,
    float softmax_scale, float q_scale, float k_scale, float v_scale,
    int out_dtype, bool is_causal, int num_splits
) {
    constexpr int NUM_K_SUBS = K_TILE / 16;
    const int LD  = Dim + WAVE_ATTN_LDS_PAD;
    const int LD4 = (Dim>>1) + WAVE_ATTN_LDS_PAD;
    constexpr int LDV = K_TILE + WAVE_ATTN_LDS_PAD;
    extern __shared__ uint8_t smem_base[];
    size_t q4_bytes = (size_t)Q_TILE * LD4;
    size_t k4_bytes = (size_t)K_TILE * LD4;
    size_t q_bytes = (size_t)Q_TILE * LD;
    size_t k_bytes = (size_t)K_TILE * LD;
    uint8_t* smem_Q4 = smem_base;
    uint8_t* smem_Q  = smem_Q4 + q4_bytes;
    uint8_t* smem_K40_4 = smem_Q + q_bytes;
    uint8_t* smem_K41_4 = smem_K40_4 + k4_bytes;
    uint8_t* smem_K0 = smem_K41_4 + k4_bytes;
    uint8_t* smem_K1 = smem_K0 + k_bytes;
    uint8_t* smem_V  = smem_K1 + k_bytes;
    const cg::thread_block cta = cg::this_thread_block();
    int tid = threadIdx.x; int wave_id = tid>>5; int lane_id = tid&31; int lw=lane_id&15; int lg=lane_id>>4;
    int bh_idx = blockIdx.x / num_splits; int b = bh_idx / H; int h = bh_idx % H;
    int q_block_idx = blockIdx.y; int q_block_start = q_block_idx*Q_TILE;
    if (q_block_start >= Seq_Q) return;
    int num_q_waves = Q_TILE>>4; int q_r = q_block_start + wave_id*16 + lw;
    // Decode support: Q tile covers absolute positions q_start... (uniform).
    const int q_start = (Seq_K > Seq_Q) ? (Seq_K - Seq_Q) : 0;
    int64_t q_off_packed = ((int64_t)b * H + h) * Seq_Q * Dim / 2;
    int64_t k_off_packed = ((int64_t)b * H + h) * Seq_K * Dim / 2;
    int64_t v_off = ((int64_t)b * H + h) * Seq_K * Dim;
    int64_t o_off = ((int64_t)b * H + h) * Seq_Q * Dim;
    float cscale = softmax_scale * q_scale * k_scale;
    float pscale = 16.0f;
    int num_d = (Dim + 15) / 16;
    v8f acc_o[8]; for(int d=0;d<8;d++) acc_o[d]={0};
    v2i q_reg[8]; for(int i=0;i<8;i++) q_reg[i]={0,0};
    load_int4_tile_to_lds<K_TILE>(Q_int4 + q_off_packed, smem_Q4, q_block_start, Q_TILE, Dim, Seq_Q, tid, THREADS_PER_BLOCK, LD4);
    cta.sync();
    // Vectorized INT4->FP8 unpack for Q tile (4 packed bytes -> 8 FP8 bytes per iter)
    unpack_int4_tile_to_fp8_vec(smem_Q, smem_Q4, Q_TILE, Dim, LD, LD4, tid, THREADS_PER_BLOCK);
    cta.sync();
    {
        int q_lds_row = wave_id*16 + lw;
        #pragma unroll
        for (int dt = 0; dt < 8; dt++) {
            if (dt >= num_d) break;
            int doff = dt*16;
            if (wave_id < num_q_waves && q_lds_row < Q_TILE) {
                int g_q_row = q_block_start + q_lds_row;
                if (g_q_row < Seq_Q && doff + lg*8 < Dim) {
                    uint64_t q64 = *(const uint64_t*)&smem_Q[q_lds_row * LD + doff + lg*8];
                    q_reg[dt] = *(const v2i*)&q64;
                }
            }
        }
    }
    cta.sync();
    float row_max = -1e30f, row_sum = 0.0f;
    int total_k_tiles = (Seq_K + K_TILE - 1) / K_TILE;
    int tiles_per_split = (total_k_tiles + num_splits - 1) / num_splits;
    int kt_start = (blockIdx.x % num_splits) * tiles_per_split;
    int kt_end = kt_start + tiles_per_split; if (kt_end > total_k_tiles) kt_end = total_k_tiles;
    if (total_k_tiles > 0) {
        int ks0 = kt_start * K_TILE;
        load_int4_tile_to_lds<K_TILE>(K_int4 + k_off_packed, smem_K40_4, ks0, K_TILE, Dim, Seq_K, tid, THREADS_PER_BLOCK, LD4);
        cta.sync();
        unpack_int4_tile_to_fp8_vec(smem_K0, smem_K40_4, K_TILE, Dim, LD, LD4, tid, THREADS_PER_BLOCK);
        cta.sync();
    }
    for (int kt = kt_start; kt < kt_end; kt++) {
        int ks = kt * K_TILE;
        // The first tile of every split is loaded into K0.  Use parity
        // relative to the split rather than the global K-tile index; odd
        // split starts would otherwise consume the uninitialized K1 buffer.
        int local_kt = kt - kt_start;
        bool use_k0 = ((local_kt & 1) == 0);
        uint8_t* cur_K = use_k0 ? smem_K0 : smem_K1;
        uint8_t* next_K = use_k0 ? smem_K1 : smem_K0;
        uint8_t* next_K4 = use_k0 ? smem_K41_4 : smem_K40_4;
        v8f sacc[NUM_K_SUBS]; for(int k=0;k<NUM_K_SUBS;k++) sacc[k]={0};
        #pragma unroll
        for (int dt = 0; dt < 8; dt++) {
            if (dt >= num_d) break;
            int doff = dt*16;
            v2i qv = q_reg[dt];
            #pragma unroll
            for (int k_sub = 0; k_sub < NUM_K_SUBS; k_sub++) {
                int kr = ks + k_sub*16 + lw;
                v2i kv = {0,0};
                if (kr < Seq_K && doff + lg*8 < Dim) {
                    uint64_t k64 = *(const uint64_t*)&cur_K[(k_sub*16+lw)*LD + doff + lg*8];
                    kv = *(const v2i*)&k64;
                }
                #if defined(__gfx1200__) || defined(__gfx1201__)
                sacc[k_sub] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(kv, qv, sacc[k_sub]);
                #endif
            }
        }
        bool wave_fully_unmasked = !is_causal || (ks + K_TILE <= q_start + q_block_start + wave_id*16);
        float tile_max = -1e30f;
        #pragma unroll
        for (int k_sub = 0; k_sub < NUM_K_SUBS; k_sub++) {
            #pragma unroll
            for (int i=0;i<8;i++) {
                int kc = ks + k_sub*16 + lg*8 + i;
                float s = sacc[k_sub][i] * cscale;
                if (wave_fully_unmasked) { if (kc >= Seq_K) s = -1e30f; } else { if (kc >= Seq_K || kc > q_start + q_r) s = -1e30f; }
                sacc[k_sub][i]=s; tile_max = fmaxf(tile_max,s);
            }
        }
        float r_max = fmaxf(tile_max, __shfl_xor_sync(0xFFFFFFFFULL, tile_max, 16));
        float nmax = fmaxf(row_max, r_max);
        float alp = (row_max > -1e29f) ? fast_expf(row_max - nmax) : 0.0f;
        float alpha = alp; row_max = nmax;
        float row_sum_new = row_sum * alp;
        v2i pv[NUM_K_SUBS];
        #pragma unroll
        for (int k_sub=0;k_sub<NUM_K_SUBS;k_sub++) {
            float p_vals[8];
            #pragma unroll
            for(int i=0;i<8;i++){ p_vals[i]=(sacc[k_sub][i]>-1e29f)?fast_expf(sacc[k_sub][i]-nmax):0.0f; row_sum_new+=p_vals[i]; }
            uint32_t u0=pack_4float_to_fp8_e4m3_fast_wave(p_vals[0]*pscale,p_vals[1]*pscale,p_vals[2]*pscale,p_vals[3]*pscale);
            uint32_t u1=pack_4float_to_fp8_e4m3_fast_wave(p_vals[4]*pscale,p_vals[5]*pscale,p_vals[6]*pscale,p_vals[7]*pscale);
            pv[k_sub]=(v2i){(int)u0,(int)u1};
        }
        row_sum=row_sum_new;
        #pragma unroll
        for(int i=0;i<8;i++){
            float corr=__shfl_sync(0xFFFFFFFFULL,alpha,lg*8+i);
            if(corr!=1.0f) {
                #pragma unroll
                for(int d=0;d<8;d++) acc_o[d][i]*=corr;
            }
        }
        load_v_tile_transposed_wave_fast_vec<K_TILE,LDV>(V_fp8+v_off, smem_V, ks, K_TILE, Dim, Seq_K, tid, THREADS_PER_BLOCK, THREADS_PER_BLOCK);
        cta.sync();
        bool has_next = (kt+1 < kt_end);
        if (has_next) {
            int ks_next=(kt+1)*K_TILE;
            load_int4_tile_to_lds<K_TILE>(K_int4 + k_off_packed, next_K4, ks_next, K_TILE, Dim, Seq_K, tid, THREADS_PER_BLOCK, LD4);
            cta.sync();
            unpack_int4_tile_to_fp8_vec(next_K, next_K4, K_TILE, Dim, LD, LD4, tid, THREADS_PER_BLOCK);
            cta.sync();
        }
        for(int dt=0;dt<num_d;dt++){
            int doff=dt*16;int vc=doff+lw;
            #pragma unroll
            for(int k_sub=0;k_sub<NUM_K_SUBS;k_sub++){
                v2i vv={0,0};
                if(vc<Dim){
                    uint64_t v64=*(const uint64_t*)&smem_V[vc*LDV + k_sub*16 + lg*8];
                    vv=*(const v2i*)&v64;
                }
                #if defined(__gfx1200__)||defined(__gfx1201__)
                acc_o[dt]=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(pv[k_sub],vv,acc_o[dt]);
                #endif
            }
        }
    }
    if (wave_id >= num_q_waves) return;
    float total_row_sum = row_sum + __shfl_xor_sync(0xFFFFFFFFULL, row_sum, 16);
    if (num_splits==1) {
        for(int dt=0;dt<num_d;dt++){
            int doff=dt*16;
            #pragma unroll
            for(int i=0;i<8;i++){
                int orr=q_block_start+wave_id*16+lg*8+i;int oc=doff+lw;
                float rsum=__shfl_sync(0xFFFFFFFFULL,total_row_sum,lg*8+i);
                float rmax=__shfl_sync(0xFFFFFFFFULL,row_max,lg*8+i);
                float inv=(rsum>0)?(1.0f/(rsum*pscale)):0;
                if(orr<Seq_Q&&oc<Dim){float fval=acc_o[dt][i]*inv*v_scale; store_attn_output_wave(Out,out_dtype,o_off+(int64_t)orr*Dim+oc,fval);}
                if(LSE&&oc==0&&dt==0&&orr<Seq_Q){float lse=(rsum>0)?(rmax+logf(rsum)):__int_as_float(0xFF800000); LSE[(int64_t)(b*H+h)*Seq_Q+orr]=lse;}
            }
        }
    } else {
        for(int dt=0;dt<num_d;dt++){
            int doff=dt*16;
            #pragma unroll
            for(int i=0;i<8;i++){
                int orr=q_block_start+wave_id*16+lg*8+i;int oc=doff+lw;
                float rmax=__shfl_sync(0xFFFFFFFFULL,row_max,lg*8+i);
                float rsum=__shfl_sync(0xFFFFFFFFULL,total_row_sum,lg*8+i);
                if(orr<Seq_Q&&oc<Dim){
                    int64_t gidx=(int64_t)(b*H+h)*Seq_Q+orr;
                    if(oc==0){Partial_Max[gidx*num_splits+ (blockIdx.x%num_splits)]=rmax; Partial_Sum[gidx*num_splits+ (blockIdx.x%num_splits)]=rsum;}
                    int64_t out_idx=gidx*Dim+oc;
                    Partial_Out[out_idx*num_splits+ (blockIdx.x%num_splits)]=acc_o[dt][i];
                }
            }
        }
    }
}

template <int Q_TILE, int K_TILE, int THREADS_PER_BLOCK>
__global__ void wave_attn_kernel_templated_int4_native(
    const uint8_t* __restrict__ Q_int4,
    const uint8_t* __restrict__ K_int4,
    const uint8_t* __restrict__ V_fp8,
    void* __restrict__ Out,
    float* __restrict__ LSE,
    float* __restrict__ Partial_Max,
    float* __restrict__ Partial_Sum,
    float* __restrict__ Partial_Out,
    int B, int H, int Seq_Q, int Seq_K, int Dim,
    float softmax_scale, float q_scale, float k_scale, float v_scale,
    int out_dtype, bool is_causal, int num_splits
) {
    constexpr int NUM_K_SUBS = K_TILE / 16;
    const int LD4 = (Dim>>1) + WAVE_ATTN_LDS_PAD;
    constexpr int LDV = K_TILE + WAVE_ATTN_LDS_PAD;
    extern __shared__ uint8_t smem_base[];
    size_t q4_bytes = (size_t)Q_TILE * LD4;
    size_t k4_bytes = (size_t)K_TILE * LD4;
    uint8_t* smem_Q4 = smem_base;
    uint8_t* smem_K40_4 = smem_Q4 + q4_bytes;
    uint8_t* smem_K41_4 = smem_K40_4 + k4_bytes;
    uint8_t* smem_V  = smem_K41_4 + k4_bytes;
    const cg::thread_block cta = cg::this_thread_block();
    int tid = threadIdx.x; int wave_id = tid>>5; int lane_id = tid&31; int lw=lane_id&15; int lg=lane_id>>4;
    int bh_idx = blockIdx.x / num_splits; int b = bh_idx / H; int h = bh_idx % H;
    int q_block_idx = blockIdx.y; int q_block_start = q_block_idx*Q_TILE;
    if (q_block_start >= Seq_Q) return;
    int num_q_waves = Q_TILE>>4; int q_r = q_block_start + wave_id*16 + lw;
    // Decode support: Q tile covers absolute positions q_start... (uniform).
    const int q_start = (Seq_K > Seq_Q) ? (Seq_K - Seq_Q) : 0;
    int64_t q_off_packed = ((int64_t)b * H + h) * Seq_Q * Dim / 2;
    int64_t k_off_packed = ((int64_t)b * H + h) * Seq_K * Dim / 2;
    int64_t v_off = ((int64_t)b * H + h) * Seq_K * Dim;
    int64_t o_off = ((int64_t)b * H + h) * Seq_Q * Dim;
    float cscale = softmax_scale * q_scale * k_scale;
    float pscale = 16.0f;
    int num_d = (Dim + 15) / 16;
    int num_d_i4 = (Dim + 31) / 32;
    v8f acc_o[8]; for(int d=0;d<8;d++) acc_o[d]={0};
    // Load Q packed tile
    load_int4_tile_to_lds<K_TILE>(Q_int4 + q_off_packed, smem_Q4, q_block_start, Q_TILE, Dim, Seq_Q, tid, THREADS_PER_BLOCK, LD4);
    cta.sync();
    float row_max = -1e30f, row_sum = 0.0f;
    int total_k_tiles = (Seq_K + K_TILE - 1) / K_TILE;
    int tiles_per_split = (total_k_tiles + num_splits - 1) / num_splits;
    int kt_start = (blockIdx.x % num_splits) * tiles_per_split;
    int kt_end = kt_start + tiles_per_split; if (kt_end > total_k_tiles) kt_end = total_k_tiles;
    if (total_k_tiles > 0) {
        int ks0 = kt_start * K_TILE;
        load_int4_tile_to_lds<K_TILE>(K_int4 + k_off_packed, smem_K40_4, ks0, K_TILE, Dim, Seq_K, tid, THREADS_PER_BLOCK, LD4);
        cta.sync();
    }
    for (int kt = kt_start; kt < kt_end; kt++) {
        int ks = kt * K_TILE;
        int local_kt = kt - kt_start;
        bool use_k0 = ((local_kt & 1) == 0);
        uint8_t* cur_K4 = use_k0 ? smem_K40_4 : smem_K41_4;
        uint8_t* next_K4 = use_k0 ? smem_K41_4 : smem_K40_4;
        // INT32 WMMA for QK^T : 16x16x32 with int4
        v8i sacc_i[NUM_K_SUBS]; for(int k=0;k<NUM_K_SUBS;k++) sacc_i[k]={0,0,0,0,0,0,0,0};
        #pragma unroll
        for (int dt = 0; dt < 4; dt++) {
            if (dt >= num_d_i4) break;
            int doff_packed = dt * 16 + lg * 8; // 16 packed bytes per dt (32 int4)
            v2i qv = {0,0};
            int q_lds_row = wave_id * 16 + lw;
            if (wave_id < num_q_waves && q_lds_row < Q_TILE) {
                int g_q_row = q_block_start + q_lds_row;
                if (g_q_row < Seq_Q && doff_packed < LD4) {
                    // guard Dim bound: doff*2 < Dim
                    if (dt*32 + lg*16 < Dim) {
                        uint64_t q64 = *(const uint64_t*)&smem_Q4[q_lds_row * LD4 + doff_packed];
                        qv = *(const v2i*)&q64;
                    }
                }
            }
            #pragma unroll
            for (int k_sub = 0; k_sub < NUM_K_SUBS; k_sub++) {
                int kr = ks + k_sub*16 + lw;
                v2i kv = {0,0};
                if (kr < Seq_K && dt*32 + lg*16 < Dim) {
                    uint64_t k64 = *(const uint64_t*)&cur_K4[(k_sub*16+lw)*LD4 + doff_packed];
                    kv = *(const v2i*)&k64;
                }
                #if defined(__gfx1200__) || defined(__gfx1201__)
                sacc_i[k_sub] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, kv, true, qv, sacc_i[k_sub], false);
                #endif
            }
        }
        // Convert int32 accumulator to float scores
        v8f sacc[NUM_K_SUBS];
        for(int k=0;k<NUM_K_SUBS;k++){
            #pragma unroll
            for(int i=0;i<8;i++) sacc[k][i] = float(sacc_i[k][i]) * cscale;
        }
        bool wave_fully_unmasked = !is_causal || (ks + K_TILE <= q_start + q_block_start + wave_id*16);
        float tile_max = -1e30f;
        #pragma unroll
        for (int k_sub = 0; k_sub < NUM_K_SUBS; k_sub++) {
            #pragma unroll
            for (int i=0;i<8;i++) {
                int kc = ks + k_sub*16 + lg*8 + i;
                float s = sacc[k_sub][i];
                if (wave_fully_unmasked) { if (kc >= Seq_K) s = -1e30f; } else { if (kc >= Seq_K || kc > q_start + q_r) s = -1e30f; }
                sacc[k_sub][i]=s; tile_max = fmaxf(tile_max,s);
            }
        }
        float r_max = fmaxf(tile_max, __shfl_xor_sync(0xFFFFFFFFULL, tile_max, 16));
        float nmax = fmaxf(row_max, r_max);
        float alp = (row_max > -1e29f) ? fast_expf(row_max - nmax) : 0.0f;
        float alpha = alp; row_max = nmax;
        float row_sum_new = row_sum * alp;
        v2i pv[NUM_K_SUBS];
        #pragma unroll
        for (int k_sub=0;k_sub<NUM_K_SUBS;k_sub++) {
            float p_vals[8];
            #pragma unroll
            for(int i=0;i<8;i++){ p_vals[i]=(sacc[k_sub][i]>-1e29f)?fast_expf(sacc[k_sub][i]-nmax):0.0f; row_sum_new+=p_vals[i]; }
            uint32_t u0=pack_4float_to_fp8_e4m3_fast_wave(p_vals[0]*pscale,p_vals[1]*pscale,p_vals[2]*pscale,p_vals[3]*pscale);
            uint32_t u1=pack_4float_to_fp8_e4m3_fast_wave(p_vals[4]*pscale,p_vals[5]*pscale,p_vals[6]*pscale,p_vals[7]*pscale);
            pv[k_sub]=(v2i){(int)u0,(int)u1};
        }
        row_sum=row_sum_new;
        #pragma unroll
        for(int i=0;i<8;i++){
            float corr=__shfl_sync(0xFFFFFFFFULL,alpha,lg*8+i);
            if(corr!=1.0f) {
                #pragma unroll
                for(int d=0;d<8;d++) acc_o[d][i]*=corr;
            }
        }
        load_v_tile_transposed_wave_fast_vec<K_TILE,LDV>(V_fp8+v_off, smem_V, ks, K_TILE, Dim, Seq_K, tid, THREADS_PER_BLOCK, THREADS_PER_BLOCK);
        cta.sync();
        bool has_next = (kt+1 < kt_end);
        if (has_next) {
            int ks_next=(kt+1)*K_TILE;
            load_int4_tile_to_lds<K_TILE>(K_int4 + k_off_packed, next_K4, ks_next, K_TILE, Dim, Seq_K, tid, THREADS_PER_BLOCK, LD4);
            cta.sync();
        }
        for(int dt=0;dt<num_d;dt++){
            int doff=dt*16;int vc=doff+lw;
            #pragma unroll
            for(int k_sub=0;k_sub<NUM_K_SUBS;k_sub++){
                v2i vv={0,0};
                if(vc<Dim){
                    uint64_t v64=*(const uint64_t*)&smem_V[vc*LDV + k_sub*16 + lg*8];
                    vv=*(const v2i*)&v64;
                }
                #if defined(__gfx1200__)||defined(__gfx1201__)
                acc_o[dt]=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(pv[k_sub],vv,acc_o[dt]);
                #endif
            }
        }
    }
    if (wave_id >= num_q_waves) return;
    float total_row_sum = row_sum + __shfl_xor_sync(0xFFFFFFFFULL, row_sum, 16);
    if (num_splits==1) {
        for(int dt=0;dt<num_d;dt++){
            int doff=dt*16;
            #pragma unroll
            for(int i=0;i<8;i++){
                int orr=q_block_start+wave_id*16+lg*8+i;int oc=doff+lw;
                float rsum=__shfl_sync(0xFFFFFFFFULL,total_row_sum,lg*8+i);
                float rmax=__shfl_sync(0xFFFFFFFFULL,row_max,lg*8+i);
                float inv=(rsum>0)?(1.0f/(rsum*pscale)):0;
                if(orr<Seq_Q&&oc<Dim){float fval=acc_o[dt][i]*inv*v_scale; store_attn_output_wave(Out,out_dtype,o_off+(int64_t)orr*Dim+oc,fval);}
                if(LSE&&oc==0&&dt==0&&orr<Seq_Q){float lse=(rsum>0)?(rmax+logf(rsum)):__int_as_float(0xFF800000); LSE[(int64_t)(b*H+h)*Seq_Q+orr]=lse;}
            }
        }
    } else {
        for(int dt=0;dt<num_d;dt++){
            int doff=dt*16;
            #pragma unroll
            for(int i=0;i<8;i++){
                int orr=q_block_start+wave_id*16+lg*8+i;int oc=doff+lw;
                float rmax=__shfl_sync(0xFFFFFFFFULL,row_max,lg*8+i);
                float rsum=__shfl_sync(0xFFFFFFFFULL,total_row_sum,lg*8+i);
                if(orr<Seq_Q&&oc<Dim){
                    int64_t gidx=(int64_t)(b*H+h)*Seq_Q+orr;
                    if(oc==0){Partial_Max[gidx*num_splits+ (blockIdx.x%num_splits)]=rmax; Partial_Sum[gidx*num_splits+ (blockIdx.x%num_splits)]=rsum;}
                    int64_t out_idx=gidx*Dim+oc;
                    Partial_Out[out_idx*num_splits+ (blockIdx.x%num_splits)]=acc_o[dt][i];
                }
            }
        }
    }
}

__global__ void wave_attn_split_reduce_kernel(
    const float * __restrict__ Partial_Max,
    const float * __restrict__ Partial_Sum,
    const float * __restrict__ Partial_Out,
    void        * __restrict__ Out,
    float       * __restrict__ LSE,
    int B, int H, int Seq_Q, int Dim, int num_splits,
    float v_scale, int out_dtype
) {
    int64_t global_q_idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_q_rows = (int64_t)B * H * Seq_Q;
    if (global_q_idx >= total_q_rows) return;

    float gmax = -1e30f;
    for (int s = 0; s < num_splits; s++) {
        float m = Partial_Max[global_q_idx * num_splits + s];
        gmax = fmaxf(gmax, m);
    }

    float gsum = 0.0f;
    for (int s = 0; s < num_splits; s++) {
        float m = Partial_Max[global_q_idx * num_splits + s];
        float psum = Partial_Sum[global_q_idx * num_splits + s];
        float alpha = (m > -1e29f) ? expf(m - gmax) : 0.0f;
        gsum += psum * alpha;
    }

    float pscale = 16.0f;
    float inv_sum = (gsum > 0.0f) ? (1.0f / (gsum * pscale)) : 0.0f;

    for (int d = 0; d < Dim; d++) {
        float merged_acc = 0.0f;
        for (int s = 0; s < num_splits; s++) {
            float m = Partial_Max[global_q_idx * num_splits + s];
            float alpha = (m > -1e29f) ? expf(m - gmax) : 0.0f;
            int64_t out_elem_idx = global_q_idx * Dim + d;
            float o_val = Partial_Out[out_elem_idx * num_splits + s];
            merged_acc += o_val * alpha;
        }
        float fval = merged_acc * inv_sum * v_scale;
        store_attn_output_wave(Out, out_dtype, global_q_idx * Dim + d, fval);
    }
    if (LSE != nullptr) {
        float lse = gmax + logf(gsum + 1e-12f);
        LSE[global_q_idx] = lse;
    }
}


extern "C" {
void launch_wave_attn(
    const uint8_t* Q_fp8, const uint8_t* K_fp8, const uint8_t* V_fp8,
    void* Out, float* LSE, float* partial_max, float* partial_sum, float* partial_out,
    int B, int H, int Seq_Q, int Seq_K, int Dim,
    float softmax_scale, float q_scale, float k_scale, float v_scale,
    int out_dtype, bool is_causal, hipStream_t stream
) {
    // ── Tile selection ──────────────────────────────────────────────
    // K_TILE=128 for D≤64 and Seq_K≥128: halves K-tile iterations.
    // K_TILE=64  otherwise (small sequences have only 1 iter anyway
    //   and the larger sacc[8] VGPR footprint hurts more than it helps).
    int Q_TILE, K_TILE, THREADS;
    const bool large_k = (Dim <= 64 && Seq_K >= 128);

    if (Seq_Q <= 16) {
        Q_TILE = 16;
        K_TILE = large_k ? 128 : 64;
        THREADS = 32;
    } else if (Seq_Q <= 32) {
        Q_TILE = 32;
        K_TILE = large_k ? 128 : 64;
        THREADS = 64;
    } else if (Seq_Q <= 64) {
        Q_TILE = 64;
        K_TILE = large_k ? 128 : 64;
        THREADS = 128;
    } else {
        Q_TILE = 128;
        K_TILE = large_k ? 128 : 64;
        THREADS = 256;
    }

    int num_blocks_q = (Seq_Q + Q_TILE - 1) / Q_TILE;
    int base_grid = B * H * num_blocks_q;
    int num_k_tiles = (Seq_K + K_TILE - 1) / K_TILE;

    int num_splits = 1;
    if (base_grid <= 2 && num_k_tiles >= 4 && partial_max != nullptr) {
        num_splits = (16 + base_grid - 1) / base_grid;
        if (num_splits > num_k_tiles) num_splits = num_k_tiles;
        if (num_splits > 8) num_splits = 8;
    }

    dim3 grid(B * H * num_splits, num_blocks_q);
    dim3 block(THREADS);
    // Sequential K/V loading: only one tile in LDS at a time.
    size_t smem_bytes = (size_t)max(
        (int64_t)K_TILE * (Dim + WAVE_ATTN_LDS_PAD),
        (int64_t)Dim    * (K_TILE + WAVE_ATTN_LDS_PAD));

    if (Q_TILE == 16 && K_TILE == 128) {
        hipLaunchKernelGGL((wave_attn_kernel_templated<16, 128, 32>), grid, block, smem_bytes, stream,
            Q_fp8, K_fp8, V_fp8, Out, LSE, partial_max, partial_sum, partial_out,
            B, H, Seq_Q, Seq_K, Dim, softmax_scale, q_scale, k_scale, v_scale,
            out_dtype, is_causal, num_splits);
    } else if (Q_TILE == 16 && K_TILE == 64) {
        hipLaunchKernelGGL((wave_attn_kernel_templated<16, 64, 32>), grid, block, smem_bytes, stream,
            Q_fp8, K_fp8, V_fp8, Out, LSE, partial_max, partial_sum, partial_out,
            B, H, Seq_Q, Seq_K, Dim, softmax_scale, q_scale, k_scale, v_scale,
            out_dtype, is_causal, num_splits);
    } else if (Q_TILE == 32 && K_TILE == 128) {
        hipLaunchKernelGGL((wave_attn_kernel_templated<32, 128, 64>), grid, block, smem_bytes, stream,
            Q_fp8, K_fp8, V_fp8, Out, LSE, partial_max, partial_sum, partial_out,
            B, H, Seq_Q, Seq_K, Dim, softmax_scale, q_scale, k_scale, v_scale,
            out_dtype, is_causal, num_splits);
    } else if (Q_TILE == 32 && K_TILE == 64) {
        hipLaunchKernelGGL((wave_attn_kernel_templated<32, 64, 64>), grid, block, smem_bytes, stream,
            Q_fp8, K_fp8, V_fp8, Out, LSE, partial_max, partial_sum, partial_out,
            B, H, Seq_Q, Seq_K, Dim, softmax_scale, q_scale, k_scale, v_scale,
            out_dtype, is_causal, num_splits);
    } else if (Q_TILE == 64 && K_TILE == 128) {
        hipLaunchKernelGGL((wave_attn_kernel_templated<64, 128, 128>), grid, block, smem_bytes, stream,
            Q_fp8, K_fp8, V_fp8, Out, LSE, partial_max, partial_sum, partial_out,
            B, H, Seq_Q, Seq_K, Dim, softmax_scale, q_scale, k_scale, v_scale,
            out_dtype, is_causal, num_splits);
    } else if (Q_TILE == 64 && K_TILE == 64) {
        hipLaunchKernelGGL((wave_attn_kernel_templated<64, 64, 128>), grid, block, smem_bytes, stream,
            Q_fp8, K_fp8, V_fp8, Out, LSE, partial_max, partial_sum, partial_out,
            B, H, Seq_Q, Seq_K, Dim, softmax_scale, q_scale, k_scale, v_scale,
            out_dtype, is_causal, num_splits);
    } else if (Q_TILE == 128 && K_TILE == 128) {
        hipLaunchKernelGGL((wave_attn_kernel_templated<128, 128, 256>), grid, block, smem_bytes, stream,
            Q_fp8, K_fp8, V_fp8, Out, LSE, partial_max, partial_sum, partial_out,
            B, H, Seq_Q, Seq_K, Dim, softmax_scale, q_scale, k_scale, v_scale,
            out_dtype, is_causal, num_splits);
    } else {
        // Q_TILE == 128 && K_TILE == 64
        hipLaunchKernelGGL((wave_attn_kernel_templated<128, 64, 256>), grid, block, smem_bytes, stream,
            Q_fp8, K_fp8, V_fp8, Out, LSE, partial_max, partial_sum, partial_out,
            B, H, Seq_Q, Seq_K, Dim, softmax_scale, q_scale, k_scale, v_scale,
            out_dtype, is_causal, num_splits);
    }

    if (num_splits > 1) {
        int total_q_rows = B * H * Seq_Q;
        dim3 r_grid((total_q_rows + 31) / 32);
        dim3 r_block(32);
        hipLaunchKernelGGL(wave_attn_split_reduce_kernel, r_grid, r_block, 0, stream,
            partial_max, partial_sum, partial_out, Out, LSE,
            B, H, Seq_Q, Dim, num_splits, v_scale, out_dtype);
    }
}
void launch_wave_attn_int4(
    const uint8_t* Q_int4, const uint8_t* K_int4, const uint8_t* V_fp8,
    void* Out, float* LSE, float* partial_max, float* partial_sum, float* partial_out,
    int B, int H, int Seq_Q, int Seq_K, int Dim,
    float softmax_scale, float q_scale, float k_scale, float v_scale,
    int out_dtype, bool is_causal, hipStream_t stream
) {
    int Q_TILE, K_TILE, THREADS;
    const bool large_k = (Dim <= 64 && Seq_K >= 128);
    if (Seq_Q <= 16) { Q_TILE=16; K_TILE= large_k?128:64; THREADS=32; }
    else if (Seq_Q <= 32) { Q_TILE=32; K_TILE= large_k?128:64; THREADS=64; }
    else if (Seq_Q <= 64) { Q_TILE=64; K_TILE= large_k?128:64; THREADS=128; }
    else { Q_TILE=128; K_TILE= large_k?128:64; THREADS=256; }
    int num_blocks_q = (Seq_Q + Q_TILE - 1) / Q_TILE;
    int base_grid = B * H * num_blocks_q;
    int num_k_tiles = (Seq_K + K_TILE - 1) / K_TILE;
    int num_splits = 1;
    if (base_grid <= 2 && num_k_tiles >= 4 && partial_max != nullptr) {
        num_splits = (16 + base_grid - 1) / base_grid;
        if (num_splits > num_k_tiles) num_splits = num_k_tiles;
        if (num_splits > 8) num_splits = 8;
    }
    dim3 grid(B * H * num_splits, num_blocks_q);
    dim3 block(THREADS);
    // Native INT4 path: only packed Q4/K4 + V are live (no unpacked Q/K).  This
    // halves LDS and halves K-loop iterations (K=32 vs 16).
    size_t ld4 = (size_t)(Dim >> 1) + WAVE_ATTN_LDS_PAD;
    size_t ldv = (size_t)K_TILE + WAVE_ATTN_LDS_PAD;
    size_t smem_bytes =
        (size_t)Q_TILE * ld4 + (size_t)K_TILE * ld4 * 2 +
        (size_t)Dim * ldv;
    if (Q_TILE==16 && K_TILE==128) hipLaunchKernelGGL((wave_attn_kernel_templated_int4_native<16,128,32>),grid,block,smem_bytes,stream,Q_int4,K_int4,V_fp8,Out,LSE,partial_max,partial_sum,partial_out,B,H,Seq_Q,Seq_K,Dim,softmax_scale,q_scale,k_scale,v_scale,out_dtype,is_causal,num_splits);
    else if (Q_TILE==16 && K_TILE==64) hipLaunchKernelGGL((wave_attn_kernel_templated_int4_native<16,64,32>),grid,block,smem_bytes,stream,Q_int4,K_int4,V_fp8,Out,LSE,partial_max,partial_sum,partial_out,B,H,Seq_Q,Seq_K,Dim,softmax_scale,q_scale,k_scale,v_scale,out_dtype,is_causal,num_splits);
    else if (Q_TILE==32 && K_TILE==128) hipLaunchKernelGGL((wave_attn_kernel_templated_int4_native<32,128,64>),grid,block,smem_bytes,stream,Q_int4,K_int4,V_fp8,Out,LSE,partial_max,partial_sum,partial_out,B,H,Seq_Q,Seq_K,Dim,softmax_scale,q_scale,k_scale,v_scale,out_dtype,is_causal,num_splits);
    else if (Q_TILE==32 && K_TILE==64) hipLaunchKernelGGL((wave_attn_kernel_templated_int4_native<32,64,64>),grid,block,smem_bytes,stream,Q_int4,K_int4,V_fp8,Out,LSE,partial_max,partial_sum,partial_out,B,H,Seq_Q,Seq_K,Dim,softmax_scale,q_scale,k_scale,v_scale,out_dtype,is_causal,num_splits);
    else if (Q_TILE==64 && K_TILE==128) hipLaunchKernelGGL((wave_attn_kernel_templated_int4_native<64,128,128>),grid,block,smem_bytes,stream,Q_int4,K_int4,V_fp8,Out,LSE,partial_max,partial_sum,partial_out,B,H,Seq_Q,Seq_K,Dim,softmax_scale,q_scale,k_scale,v_scale,out_dtype,is_causal,num_splits);
    else if (Q_TILE==64 && K_TILE==64) hipLaunchKernelGGL((wave_attn_kernel_templated_int4_native<64,64,128>),grid,block,smem_bytes,stream,Q_int4,K_int4,V_fp8,Out,LSE,partial_max,partial_sum,partial_out,B,H,Seq_Q,Seq_K,Dim,softmax_scale,q_scale,k_scale,v_scale,out_dtype,is_causal,num_splits);
    else if (Q_TILE==128 && K_TILE==128) hipLaunchKernelGGL((wave_attn_kernel_templated_int4_native<128,128,256>),grid,block,smem_bytes,stream,Q_int4,K_int4,V_fp8,Out,LSE,partial_max,partial_sum,partial_out,B,H,Seq_Q,Seq_K,Dim,softmax_scale,q_scale,k_scale,v_scale,out_dtype,is_causal,num_splits);
    else hipLaunchKernelGGL((wave_attn_kernel_templated_int4_native<128,64,256>),grid,block,smem_bytes,stream,Q_int4,K_int4,V_fp8,Out,LSE,partial_max,partial_sum,partial_out,B,H,Seq_Q,Seq_K,Dim,softmax_scale,q_scale,k_scale,v_scale,out_dtype,is_causal,num_splits);
    if (num_splits > 1) {
        int total_q_rows = B * H * Seq_Q;
        dim3 r_grid((total_q_rows + 31) / 32);
        dim3 r_block(32);
        hipLaunchKernelGGL(wave_attn_split_reduce_kernel, r_grid, r_block, 0, stream, partial_max, partial_sum, partial_out, Out, LSE, B, H, Seq_Q, Dim, num_splits, v_scale, out_dtype);
    }
}

} // extern "C"
