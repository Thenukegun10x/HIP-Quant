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

// Division-free vectorized load from global → LDS using 128-bit (uint4) loads.
__device__ inline void load_tile_to_lds_wave_fast(
    const uint8_t* __restrict__ src_global,
    uint8_t* __restrict__ dst_lds,
    int g_row_start, int num_rows, int dim, int total_g_rows,
    int tid_start, int tid_stride, int tid_count
) {
    int total_bytes = num_rows * dim;
    int num_u4 = total_bytes / 16;
    uint4* lds_u4 = (uint4*)dst_lds;
    const uint4* g_u4 = (const uint4*)src_global;

    int u4_per_row_shift = (dim == 128) ? 3 : 2;
    int u4_per_row_mask  = (dim == 128) ? 7 : 3;

    for (int i = tid_start; i < num_u4; i += tid_stride) {
        int r = i >> u4_per_row_shift;
        int c = (i & u4_per_row_mask) << 4;
        int g_r = g_row_start + r;

        if (g_r < total_g_rows && c < dim) {
            int g_u4_idx = (g_r * dim + c) >> 4;
            lds_u4[i] = g_u4[g_u4_idx];
        } else {
            lds_u4[i] = make_uint4(0, 0, 0, 0);
        }
    }
}

// Transposed V load: uint4 global read → unrolled LDS stores.
template <int K_TILE>
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
        int col0 = c * K_TILE + r;
        #pragma unroll
        for (int k = 0; k < 16; k += 4) {
            dst_lds[col0 + (k+0) * K_TILE] = bytes[k+0];
            dst_lds[col0 + (k+1) * K_TILE] = bytes[k+1];
            dst_lds[col0 + (k+2) * K_TILE] = bytes[k+2];
            dst_lds[col0 + (k+3) * K_TILE] = bytes[k+3];
        }
    }
}


template <int Q_TILE, int K_TILE, int THREADS_PER_BLOCK>
__global__ void wave_attn_kernel_templated(
    const uint8_t * __restrict__ Q_fp8,
    const uint8_t * __restrict__ K_fp8,
    const uint8_t * __restrict__ V_fp8,
    void          * __restrict__ Out,
    float         * __restrict__ Partial_Max,
    float         * __restrict__ Partial_Sum,
    float         * __restrict__ Partial_Out,
    int B, int H, int Seq_Q, int Seq_K, int Dim,
    float softmax_scale, float q_scale, float k_scale, float v_scale,
    int out_dtype, bool is_causal, int num_splits
) {
    constexpr int NUM_K_SUBS = K_TILE / 16;
    constexpr int NUM_Q_SUBS = Q_TILE / 16;

    extern __shared__ uint8_t smem_base[];
    uint8_t* smem_Q = smem_base;
    uint8_t* smem_K = smem_Q + Q_TILE * Dim;
    uint8_t* smem_V = smem_K + K_TILE * Dim;

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
    int half_threads = THREADS_PER_BLOCK >> 1;

    // Load Q tile (Q_TILE × Dim) into LDS — all threads participate
    load_tile_to_lds_wave_fast(Q_fp8 + q_off, smem_Q, q_block_start, Q_TILE, Dim, Seq_Q, tid, THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    __syncthreads();

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

    if (is_causal) {
        int max_q_in_block = q_block_start + Q_TILE - 1;
        int m = (max_q_in_block + K_TILE - 1) / K_TILE;
        if (kt_end > m) kt_end = m;
    }

    int q_lds_row = wave_id * 16 + lw;
    bool valid_wave = wave_id < num_q_waves && q_r < Seq_Q;

    // ─── outer loop over K/V tiles ──────────────────────────────────
    for (int kt = kt_start; kt < kt_end; kt++) {
        int ks = kt * K_TILE;

        // ── Parallel K/V tile load (split threads, 2× throughput) ──
        // Low half loads K (row-major), high half loads V (transposed)
        if (tid < half_threads) {
            load_tile_to_lds_wave_fast(K_fp8 + k_off, smem_K, ks, K_TILE, Dim, Seq_K, tid, half_threads, THREADS_PER_BLOCK);
        } else {
            load_v_tile_transposed_wave_fast<K_TILE>(V_fp8 + v_off, smem_V, ks, K_TILE, Dim, Seq_K,
                tid - half_threads, half_threads, THREADS_PER_BLOCK);
        }
        cta.sync();

        if (!valid_wave) { cta.sync(); continue; }

        // ── Phase 1: S = Q @ K^T  (FP8 WMMA + scale + running max) ──
        v8f sacc[NUM_K_SUBS];
        float tile_max = -1e30f;
        #pragma unroll
        for (int k_sub = 0; k_sub < NUM_K_SUBS; k_sub++) sacc[k_sub] = (v8f){};

        for (int dt = 0; dt < num_d; dt++) {
            int doff = dt * 16;
            v2i qv = (v2i){0, 0};
            if (doff + lg * 8 < Dim) {
                uint64_t q64 = *(const uint64_t*)&smem_Q[q_lds_row * Dim + doff + lg * 8];
                qv = *(const v2i*)&q64;
            }

            #pragma unroll
            for (int k_sub = 0; k_sub < NUM_K_SUBS; k_sub++) {
                int kr = ks + k_sub * 16 + lw;
                v2i kv = (v2i){0, 0};
                if (kr < Seq_K && doff + lg * 8 < Dim) {
                    uint64_t k64 = *(const uint64_t*)&smem_K[(k_sub * 16 + lw) * Dim + doff + lg * 8];
                    kv = *(const v2i*)&k64;
                }
                #if defined(__gfx1200__) || defined(__gfx1201__)
                sacc[k_sub] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(kv, qv, sacc[k_sub]);
                #endif
            }
        }

        // ── Phase 2: scale + running max (single fused pass) ────────
        float scaled[NUM_K_SUBS][8];
        #pragma unroll
        for (int k_sub = 0; k_sub < NUM_K_SUBS; k_sub++) {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int kc = ks + k_sub * 16 + lg * 8 + i;
                float s = sacc[k_sub][i] * cscale;
                if (kc >= Seq_K || (is_causal && kc > q_r)) s = -1e30f;
                scaled[k_sub][i] = s;
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
                float pv_f = (scaled[k_sub][i] > -1e29f) ? fast_expf(scaled[k_sub][i] - nmax) : 0.0f;
                p_vals[i] = pv_f;
                row_sum_new += pv_f;
            }
            uint32_t u0 = pack_4float_to_fp8_e4m3_fast_wave(p_vals[0] * pscale, p_vals[1] * pscale, p_vals[2] * pscale, p_vals[3] * pscale);
            uint32_t u1 = pack_4float_to_fp8_e4m3_fast_wave(p_vals[4] * pscale, p_vals[5] * pscale, p_vals[6] * pscale, p_vals[7] * pscale);
            pv[k_sub] = (v2i){(int)u0, (int)u1};
        }
        row_sum = row_sum_new;

        // ── Alpha rescaling: fuse into P@V WMMA operand preparation ──
        // Apply alpha to acc_o via vector multiply, then proceed to P@V
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float corr_alpha = __shfl_sync(0xFFFFFFFFFFFFFFFFULL, alpha, lg * 8 + i);
            #pragma unroll
            for (int d = 0; d < 8; d++) {
                acc_o[d][i] *= corr_alpha;
            }
        }

        // ── Phase 3: O += P @ V  (FP8 WMMA) ────────────────────────
        for (int dt = 0; dt < num_d; dt++) {
            int doff = dt * 16;
            int vc = doff + lw;

            #pragma unroll
            for (int k_sub = 0; k_sub < NUM_K_SUBS; k_sub++) {
                v2i vv = (v2i){0, 0};
                if (vc < Dim) {
                    uint64_t v64 = *(const uint64_t*)&smem_V[vc * K_TILE + k_sub * 16 + lg * 8];
                    vv = *(const v2i*)&v64;
                }
                #if defined(__gfx1200__) || defined(__gfx1201__)
                acc_o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(pv[k_sub], vv, acc_o[dt]);
                #endif
            }
        }

        cta.sync();
    }

    if (!valid_wave) return;

    float total_row_sum = row_sum + __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, row_sum, 16);

    if (num_splits == 1) {
        for (int dt = 0; dt < num_d; dt++) {
            int doff = dt * 16;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int orr = q_block_start + wave_id * 16 + lg * 8 + i;
                int oc = doff + lw;
                float rsum = __shfl_sync(0xFFFFFFFFFFFFFFFFULL, total_row_sum, lg * 8 + i);
                float inv = (rsum > 0.0f) ? (1.0f / (rsum * pscale)) : 0.0f;
                if (orr < Seq_Q && oc < Dim) {
                    float fval = acc_o[dt][i] * inv * v_scale;
                    store_attn_output_wave(Out, out_dtype, o_off + (int64_t)orr * Dim + oc, fval);
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
                if (orr < Seq_Q && oc < Dim) {
                    int64_t global_q_idx = (int64_t)(b * H + h) * Seq_Q + orr;
                    float rmax = __shfl_sync(0xFFFFFFFFFFFFFFFFULL, row_max, lg * 8 + i);
                    float rsum = __shfl_sync(0xFFFFFFFFFFFFFFFFULL, total_row_sum, lg * 8 + i);
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


__global__ void wave_attn_split_reduce_kernel(
    const float * __restrict__ Partial_Max,
    const float * __restrict__ Partial_Sum,
    const float * __restrict__ Partial_Out,
    void        * __restrict__ Out,
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
}


extern "C" {
void launch_wave_attn(
    const uint8_t* Q_fp8, const uint8_t* K_fp8, const uint8_t* V_fp8,
    void* Out, float* partial_max, float* partial_sum, float* partial_out,
    int B, int H, int Seq_Q, int Seq_K, int Dim,
    float softmax_scale, float q_scale, float k_scale, float v_scale,
    int out_dtype, bool is_causal, hipStream_t stream
) {
    // ── Tile selection ──────────────────────────────────────────────
    // Q_TILE capped at 64: smaller Q → bigger K_TILE → fewer outer iterations
    // K_TILE=128 for D<128, K_TILE=64 for D≥128 (LDS budget)
    int Q_TILE, K_TILE, THREADS;

    // ── Adaptive tile selection (v7-style with K_TILE bias toward 128) ─
    if (Seq_Q <= 16) {
        Q_TILE = 16;
        K_TILE = (Dim >= 128 || Seq_K < 128) ? 64 : 128;
        THREADS = 32;
    } else if (Seq_Q <= 32) {
        Q_TILE = 32;
        K_TILE = (Dim >= 128 || Seq_K < 128) ? 64 : 128;
        THREADS = 64;
    } else if (Seq_Q <= 64) {
        Q_TILE = 64;
        K_TILE = (Dim >= 128 || Seq_K < 128) ? 64 : 128;
        THREADS = 128;
    } else {
        // Long sequences: use Q_TILE=128 to keep block count down and reuse K/V loads
        Q_TILE = 128;
        K_TILE = (Dim >= 128) ? 64 : ((Seq_K >= 128) ? 128 : 64);
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
    size_t smem_bytes = (size_t)(Q_TILE * Dim + K_TILE * Dim + Dim * K_TILE) * sizeof(uint8_t);

    if (Q_TILE == 16) {
        hipLaunchKernelGGL((wave_attn_kernel_templated<16, 64, 32>), grid, block, smem_bytes, stream,
            Q_fp8, K_fp8, V_fp8, Out, partial_max, partial_sum, partial_out,
            B, H, Seq_Q, Seq_K, Dim, softmax_scale, q_scale, k_scale, v_scale,
            out_dtype, is_causal, num_splits);
    } else if (Q_TILE == 32) {
        hipLaunchKernelGGL((wave_attn_kernel_templated<32, 64, 64>), grid, block, smem_bytes, stream,
            Q_fp8, K_fp8, V_fp8, Out, partial_max, partial_sum, partial_out,
            B, H, Seq_Q, Seq_K, Dim, softmax_scale, q_scale, k_scale, v_scale,
            out_dtype, is_causal, num_splits);
    } else if (Q_TILE == 64 && K_TILE == 128) {
        hipLaunchKernelGGL((wave_attn_kernel_templated<64, 128, 128>), grid, block, smem_bytes, stream,
            Q_fp8, K_fp8, V_fp8, Out, partial_max, partial_sum, partial_out,
            B, H, Seq_Q, Seq_K, Dim, softmax_scale, q_scale, k_scale, v_scale,
            out_dtype, is_causal, num_splits);
    } else if (Q_TILE == 64 && K_TILE == 64) {
        hipLaunchKernelGGL((wave_attn_kernel_templated<64, 64, 128>), grid, block, smem_bytes, stream,
            Q_fp8, K_fp8, V_fp8, Out, partial_max, partial_sum, partial_out,
            B, H, Seq_Q, Seq_K, Dim, softmax_scale, q_scale, k_scale, v_scale,
            out_dtype, is_causal, num_splits);
    } else if (Q_TILE == 128 && K_TILE == 128) {
        hipLaunchKernelGGL((wave_attn_kernel_templated<128, 128, 256>), grid, block, smem_bytes, stream,
            Q_fp8, K_fp8, V_fp8, Out, partial_max, partial_sum, partial_out,
            B, H, Seq_Q, Seq_K, Dim, softmax_scale, q_scale, k_scale, v_scale,
            out_dtype, is_causal, num_splits);
    } else {
        // Q_TILE == 128 && K_TILE == 64
        hipLaunchKernelGGL((wave_attn_kernel_templated<128, 64, 256>), grid, block, smem_bytes, stream,
            Q_fp8, K_fp8, V_fp8, Out, partial_max, partial_sum, partial_out,
            B, H, Seq_Q, Seq_K, Dim, softmax_scale, q_scale, k_scale, v_scale,
            out_dtype, is_causal, num_splits);
    }

    if (num_splits > 1) {
        int total_q_rows = B * H * Seq_Q;
        dim3 r_grid((total_q_rows + 31) / 32);
        dim3 r_block(32);
        hipLaunchKernelGGL(wave_attn_split_reduce_kernel, r_grid, r_block, 0, stream,
            partial_max, partial_sum, partial_out, Out,
            B, H, Seq_Q, Dim, num_splits, v_scale, out_dtype);
    }
}
} // extern "C"
