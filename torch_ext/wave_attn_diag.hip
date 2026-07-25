// torch_ext/wave_attn_diag.hip
//
// Diagnostic kernel: dumps forward S and backward dP/dS intermediates
// for a single (Q tile, K tile) pair to verify WMMA data layout.
//
// Writes:
//   S_diag  [Q_TILE * K_TILE] — raw Q@K^T scores (no scale)
//   dP_diag [Q_TILE * K_TILE] — dO @ V^T  (no v_scale)
//   dS_diag [Q_TILE * K_TILE] — softmax backward
//
// Each blockIdx.y selects one Q-tile row within the Q tile for dumping.

#include <hip/hip_runtime.h>
#include "../hip_quant_util.h"

typedef float v8f __attribute__((ext_vector_type(8)));
typedef int   v2i __attribute__((ext_vector_type(2)));

__device__ inline float fast_expf_diag(float x) {
    return __builtin_exp2f(x * 1.4426950408889634f);
}

__device__ inline uint32_t pack_4float_to_fp8_e4m3_diag(float f0, float f1, float f2, float f3) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    uint32_t res0, res1;
    asm volatile("v_cvt_pk_fp8_f32 %0, %1, %2" : "=v"(res0) : "v"(f0), "v"(f1));
    asm volatile("v_cvt_pk_fp8_f32 %0, %1, %2" : "=v"(res1) : "v"(f2), "v"(f3));
    return ((res0 & 0xFFFF) | ((res1 & 0xFFFF) << 16));
#else
    uint8_t b0 = fp32_bits_to_fp8_e4m3(__float_as_int(f0));
    uint8_t b1 = fp32_bits_to_fp8_e4m3(__float_as_int(f1));
    uint8_t b2 = fp32_bits_to_fp8_e4m3(__float_as_int(f2));
    uint8_t b3 = fp32_bits_to_fp8_e4m3(__float_as_int(f3));
    return (uint32_t)b0 | ((uint32_t)b1 << 8) | ((uint32_t)b2 << 16) | ((uint32_t)b3 << 24);
#endif
}

template <int Q_TILE, int K_TILE, int THREADS>
__global__ void wave_attn_diag_kernel(
    const uint8_t* __restrict__ Q_fp8,
    const uint8_t* __restrict__ K_fp8,
    const uint8_t* __restrict__ V_fp8,
    const float*   __restrict__ dO,
    float*         __restrict__ S_diag,     // [Q_TILE * K_TILE]
    float*         __restrict__ dP_diag,    // [Q_TILE * K_TILE]
    float*         __restrict__ dS_diag,    // [Q_TILE * K_TILE]
    float          softmax_scale,
    int            q_tile_row,               // which Q row within the Q tile to dump (global index)
    int            Dim
) {
    constexpr int NUM_K_SUBS = K_TILE / 16;
    constexpr int NUM_Q_SUBS = Q_TILE / 16;

    extern __shared__ uint8_t smem[];

    int tid = threadIdx.x;
    int wave_id = tid >> 5;
    int lane_id = tid & 31;
    int lw = lane_id & 15;
    int lg = lane_id >> 4;

    int num_d = (Dim + 15) / 16;
    int half_t = THREADS >> 1;

    int q_lds_row = wave_id * 16 + lw;
    if (q_lds_row >= Q_TILE) return;

    // ── Load Q tile to LDS ──
    int total_qb = Q_TILE * Dim;
    int num_q_u4 = total_qb / 16;
    uint4* q_lds = (uint4*)smem;
    const uint4* q_g = (const uint4*)Q_fp8;
    {
        int shift = (Dim == 128) ? 3 : 2;
        int mask  = (Dim == 128) ? 7 : 3;
        for (int i = tid; i < num_q_u4; i += THREADS) {
            int r = i >> shift;
            int c = (i & mask) << 4;
            q_lds[i] = (r < Q_TILE && c < Dim) ? q_g[(r * Dim + c) >> 4] : make_uint4(0,0,0,0);
        }
    }
    __syncthreads();

    // ── Load K tile to LDS (after Q in smem) ──
    uint8_t* smem_K = (uint8_t*)smem + Q_TILE * Dim;
    {
        int total_kb = K_TILE * Dim;
        int num_k_u4 = total_kb / 16;
        uint4* k_lds = (uint4*)smem_K;
        const uint4* k_g = (const uint4*)K_fp8;
        int shift = (Dim == 128) ? 3 : 2;
        int mask  = (Dim == 128) ? 7 : 3;
        for (int i = tid; i < num_k_u4; i += THREADS) {
            int r = i >> shift;
            int c = (i & mask) << 4;
            k_lds[i] = (r < K_TILE && c < Dim) ? k_g[(r * Dim + c) >> 4] : make_uint4(0,0,0,0);
        }
    }
    __syncthreads();

    // ── Determine which Q row this thread dumps ──
    bool is_dump_wave   = (q_tile_row >= wave_id * 16) && (q_tile_row < (wave_id + 1) * 16);
    int  dump_row_local = q_tile_row - wave_id * 16;  // 0..15 within this wave

    v8f sacc[NUM_K_SUBS];
    #pragma unroll
    for (int ks = 0; ks < NUM_K_SUBS; ks++) sacc[ks] = (v8f){};

    // ── S = Q @ K^T via WMMA ──
    for (int dt = 0; dt < num_d; dt++) {
        int doff = dt * 16;
        v2i qv = (v2i){0, 0};
        if (doff + lg * 8 < Dim) {
            uint64_t q64 = *(const uint64_t*)&smem[q_lds_row * Dim + doff + lg * 8];
            qv = *(const v2i*)&q64;
        }
        #pragma unroll
        for (int ks = 0; ks < NUM_K_SUBS; ks++) {
            v2i kv = (v2i){0, 0};
            int kr = ks * 16 + lw;
            if (kr < K_TILE && doff + lg * 8 < Dim) {
                uint64_t k64 = *(const uint64_t*)&smem_K[(ks * 16 + lw) * Dim + doff + lg * 8];
                kv = *(const v2i*)&k64;
            }
            #if defined(__gfx1200__) || defined(__gfx1201__)
            sacc[ks] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(kv, qv, sacc[ks]);
            #endif
        }
    }

    // ── Dump S (raw scores, no scale, no softmax) ──
    if (is_dump_wave && lw == dump_row_local) {
        #pragma unroll
        for (int ks = 0; ks < NUM_K_SUBS; ks++) {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int kc = ks * 16 + lg * 8 + i;
                int idx = q_tile_row * K_TILE + kc;
                S_diag[idx] = sacc[ks][i];   // raw WMMA output
            }
        }
    }
    __syncthreads();

    // ── Load V tile ──
    uint8_t* smem_V = smem_K + K_TILE * Dim;
    {
        int total_vb = K_TILE * Dim;
        int num_v_u4 = total_vb / 16;
        uint4* v_lds = (uint4*)smem_V;
        const uint4* v_g = (const uint4*)V_fp8;
        int shift = (Dim == 128) ? 3 : 2;
        int mask  = (Dim == 128) ? 7 : 3;
        for (int i = tid; i < num_v_u4; i += THREADS) {
            int r = i >> shift;
            int c = (i & mask) << 4;
            v_lds[i] = (r < K_TILE && c < Dim) ? v_g[(r * Dim + c) >> 4] : make_uint4(0,0,0,0);
        }
    }
    __syncthreads();

    // ── Load dO tile (float, after V) ──
    float* smem_dO = (float*)(smem_V + K_TILE * Dim);
    {
        int total_do = Q_TILE * Dim;
        for (int i = tid; i < total_do; i += THREADS) {
            int r = i / Dim;
            int c = i % Dim;
            smem_dO[i] = (r < Q_TILE && c < Dim) ? dO[(int64_t)r * Dim + c] : 0.0f;
        }
    }
    __syncthreads();

    // ── dP = dO @ V^T via WMMA ──
    v8f dP_acc[NUM_K_SUBS];
    #pragma unroll
    for (int ks = 0; ks < NUM_K_SUBS; ks++) dP_acc[ks] = (v8f){};

    for (int dt = 0; dt < num_d; dt++) {
        int doff = dt * 16;

        // Convert dO row to FP8
        float do_row[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int col = doff + lg * 8 + i;
            float v = (q_lds_row < Q_TILE && col < Dim)
                ? smem_dO[q_lds_row * Dim + col] : 0.0f;
            do_row[i] = v;
        }
        uint32_t u0 = pack_4float_to_fp8_e4m3_diag(do_row[0], do_row[1], do_row[2], do_row[3]);
        uint32_t u1 = pack_4float_to_fp8_e4m3_diag(do_row[4], do_row[5], do_row[6], do_row[7]);
        v2i do_fp8 = (v2i){(int)u0, (int)u1};

        #pragma unroll
        for (int ks = 0; ks < NUM_K_SUBS; ks++) {
            v2i vv = (v2i){0, 0};
            int kr = ks * 16 + lw;
            if (kr < K_TILE && doff + lg * 8 < Dim) {
                uint64_t v64 = *(const uint64_t*)&smem_V[(ks * 16 + lw) * Dim + doff + lg * 8];
                vv = *(const v2i*)&v64;
            }
            #if defined(__gfx1200__) || defined(__gfx1201__)
            dP_acc[ks] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(vv, do_fp8, dP_acc[ks]);
            #endif
        }
    }

    // ── Dump dP (raw WMMA output) ──
    if (is_dump_wave && lw == dump_row_local) {
        #pragma unroll
        for (int ks = 0; ks < NUM_K_SUBS; ks++) {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int kc = ks * 16 + lg * 8 + i;
                int idx = q_tile_row * K_TILE + kc;
                dP_diag[idx] = dP_acc[ks][i];  // raw WMMA output, no v_scale
            }
        }
    }

    // ── Compute dS = P * (dP - rowsum(P * dP)) ──
    // Reconstruct P: need LSE.  For diagnosis, compute from raw S here.
    float s_row[K_TILE];  // S values for this Q row
    #pragma unroll
    for (int ks = 0; ks < NUM_K_SUBS; ks++) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int kc = ks * 16 + lg * 8 + i;
            s_row[kc] = sacc[ks][i] * softmax_scale;
        }
    }

    // Warp-level reduction for rowmax
    float row_max = -1e30f;
    #pragma unroll
    for (int ks = 0; ks < NUM_K_SUBS; ks++) {
        #pragma unroll
        for (int i = 0; i < 8; i++) row_max = fmaxf(row_max, s_row[ks * 16 + lg * 8 + i]);
    }
    row_max = fmaxf(row_max, __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, row_max, 16));

    // Compute P (on the actual dump row only)
    float p_row[K_TILE];
    float row_sum = 0.0f;
    #pragma unroll
    for (int ks = 0; ks < NUM_K_SUBS; ks++) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int kc = ks * 16 + lg * 8 + i;
            float pv = (s_row[kc] > -1e29f) ? fast_expf_diag(s_row[kc] - row_max) : 0.0f;
            p_row[kc] = pv;
            row_sum += pv;
        }
    }
    row_sum += __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, row_sum, 16);
    #pragma unroll
    for (int ks = 0; ks < NUM_K_SUBS; ks++) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int kc = ks * 16 + lg * 8 + i;
            p_row[kc] /= (row_sum + 1e-12f);
        }
    }

    // Compute rowsum(P * dP) for this Q row
    float p_dp_sum = 0.0f;
    #pragma unroll
    for (int ks = 0; ks < NUM_K_SUBS; ks++) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int kc = ks * 16 + lg * 8 + i;
            p_dp_sum += p_row[kc] * dP_acc[ks][i];
        }
    }
    p_dp_sum += __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, p_dp_sum, 16);

    // Compute dS and dump
    if (is_dump_wave && lw == dump_row_local) {
        #pragma unroll
        for (int ks = 0; ks < NUM_K_SUBS; ks++) {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int kc = ks * 16 + lg * 8 + i;
                float ds_val = p_row[kc] * (dP_acc[ks][i] - p_dp_sum);
                dS_diag[q_tile_row * K_TILE + kc] = ds_val;
            }
        }
    }
}


// ── Host launcher ──
extern "C" {
void launch_wave_attn_diag(
    const uint8_t* Q_fp8, const uint8_t* K_fp8, const uint8_t* V_fp8,
    const float* dO,
    float* S_diag, float* dP_diag, float* dS_diag,
    int Q_TILE, int K_TILE, int Dim,
    float softmax_scale, int q_tile_row,
    hipStream_t stream
) {
    auto launch = [&](auto kernel) {
        dim3 grid(1, 1);  // single block
        dim3 block(Q_TILE <= 16 ? 32 : (Q_TILE <= 32 ? 64 : 128));
        size_t smem = (size_t)(Q_TILE * Dim + K_TILE * Dim + K_TILE * Dim + Q_TILE * Dim * sizeof(float));
        hipLaunchKernelGGL(kernel, grid, block, smem, stream,
            Q_fp8, K_fp8, V_fp8, dO, S_diag, dP_diag, dS_diag,
            softmax_scale, q_tile_row, Dim);
    };

    if (Q_TILE == 16 && K_TILE == 64)       launch(wave_attn_diag_kernel<16, 64, 32>);
    else if (Q_TILE == 32 && K_TILE == 64)  launch(wave_attn_diag_kernel<32, 64, 64>);
    else if (Q_TILE == 64 && K_TILE == 64)  launch(wave_attn_diag_kernel<64, 64, 128>);
}
} // extern "C"
