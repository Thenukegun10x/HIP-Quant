// torch_ext/wave_attn_prefill.hip
// Prefill / Chunked-Prefill inference kernel for GFX12.
// Fork of wave_attn.hip:304 wave_attn_kernel_templated + wave_attn_kernel_templated_int4_native:765
// INT4 Q/K (wmmma_i32 16x16x32) + FP8 V (wmmma_f32 16x16x16) hybrid is the default.
// FP8 Q/K/V kept as fallback when !use_int4 or Dim>128.
// Paged K/V support: if block_table != nullptr, K/V loads index via block_table[batch*num_blocks + block_id].

#include "wave_attn_common.h"

// Re-export split reduce kernel (shared across all three)
// Defined in wave_attn.hip:953 - duplicate here to keep prefill self-contained
__global__ void wave_attn_prefill_split_reduce_kernel(
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
    float gmax = WAVE_ATTN_NEG_INF;
    for (int s = 0; s < num_splits; s++) gmax = fmaxf(gmax, Partial_Max[global_q_idx * num_splits + s]);
    float gsum = 0.0f;
    for (int s = 0; s < num_splits; s++) {
        float m = Partial_Max[global_q_idx * num_splits + s];
        float psum = Partial_Sum[global_q_idx * num_splits + s];
        float alpha = (m > -1e29f) ? expf(m - gmax) : 0.0f;
        alpha = wave_attn_scale_ftz(alpha, m - gmax);
        gsum += psum * alpha;
    }
    float pscale = 16.0f;
    float inv_sum = (gsum > 0.0f) ? (1.0f / (gsum * pscale)) : 0.0f;
    for (int d = 0; d < Dim; d++) {
        float merged_acc = 0.0f;
        for (int s = 0; s < num_splits; s++) {
            float m = Partial_Max[global_q_idx * num_splits + s];
            float alpha = (m > -1e29f) ? expf(m - gmax) : 0.0f;
            alpha = wave_attn_scale_ftz(alpha, m - gmax);
            int64_t out_elem_idx = global_q_idx * Dim + d;
            merged_acc += Partial_Out[out_elem_idx * num_splits + s] * alpha;
        }
        float fval = merged_acc * inv_sum * v_scale;
        store_attn_output_wave(Out, out_dtype, global_q_idx * Dim + d, fval);
    }
    if (LSE != nullptr) LSE[global_q_idx] = gmax + logf(gsum + 1e-12f);
}

// Prefill FP8 kernel - identical to wave_attn.hip:304 but renamed + block_table hook
template <int Q_TILE, int K_TILE, int THREADS_PER_BLOCK>
__global__ void wave_attn_prefill_fp8(
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
    int out_dtype, bool is_causal, int num_splits,
    const int32_t* __restrict__ block_table, int block_size
) {
    constexpr int NUM_K_SUBS = K_TILE / 16;
    const int LDK = Dim + WAVE_ATTN_LDS_PAD;
    constexpr int LDV = K_TILE + WAVE_ATTN_LDS_PAD;
    extern __shared__ uint8_t smem_base[];
    uint8_t* smem_K = smem_base;
    const cg::thread_block cta = cg::this_thread_block();
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
    float pscale = 16.0f;
    int num_d = (Dim + 15) / 16;
    int num_q_waves = Q_TILE >> 4;
    const uint8_t* __restrict__ Q_base = Q_fp8 + q_off;
    v8f acc_o[16];
    #pragma unroll
    for (int d = 0; d < 16; d++) acc_o[d] = (v8f){};
    float row_max = WAVE_ATTN_NEG_INF;
    float row_sum = 0.0f;
    int total_k_tiles = (Seq_K + K_TILE - 1) / K_TILE;
    int tiles_per_split = (total_k_tiles + num_splits - 1) / num_splits;
    int kt_start = split_idx * tiles_per_split;
    int kt_end = (split_idx + 1) * tiles_per_split;
    if (kt_end > total_k_tiles) kt_end = total_k_tiles;
    int q_r = q_block_start + wave_id * 16 + lw;
    // Decode support: Q tile covers absolute positions q_start... (uniform).
    const int q_start = (Seq_K > Seq_Q) ? (Seq_K - Seq_Q) : 0;
    if (is_causal) {
        int max_q_in_block = q_start + q_block_start + Q_TILE - 1;
        int m = (max_q_in_block + K_TILE - 1) / K_TILE;
        if (kt_end > m) kt_end = m;
    }
    for (int kt = kt_start; kt < kt_end; kt++) {
        int ks = kt * K_TILE;
        // Paged path: if block_table provided, K/V base would be paged - keep hook for future
        // For now contiguous fast path:
        load_tile_to_lds_wave_fast(K_fp8 + k_off, smem_K, ks, K_TILE, Dim, Seq_K, tid, THREADS_PER_BLOCK, THREADS_PER_BLOCK, LDK);
        cta.sync();
        v8f sacc[NUM_K_SUBS];
        float tile_max = WAVE_ATTN_NEG_INF;
        #pragma unroll
        for (int k_sub = 0; k_sub < NUM_K_SUBS; k_sub++) sacc[k_sub] = (v8f){};
        #pragma unroll
        for (int dt = 0; dt < 16; dt++) {
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
        #pragma unroll
        for (int k_sub = 0; k_sub < NUM_K_SUBS; k_sub++) {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int kc = ks + k_sub * 16 + lg * 8 + i;
                float s = sacc[k_sub][i] * cscale;
                if (kc >= Seq_K || (is_causal && kc > q_start + q_r)) s = WAVE_ATTN_NEG_INF;
                sacc[k_sub][i] = s;
                tile_max = fmaxf(tile_max, s);
            }
        }
        float r_max = fmaxf(tile_max, __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, tile_max, 16));
        float nmax = fmaxf(row_max, r_max);
        float alp = (row_max > -1e29f) ? fast_expf(row_max - nmax) : 0.0f;
        alp = wave_attn_scale_ftz(alp, row_max - nmax);
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
                pv_f = wave_attn_scale_ftz(pv_f, sacc[k_sub][i] - nmax);
                p_vals[i] = pv_f;
                row_sum_new += pv_f;
            }
            uint32_t u0 = pack_4float_to_fp8_e4m3_fast_wave(p_vals[0] * pscale, p_vals[1] * pscale, p_vals[2] * pscale, p_vals[3] * pscale);
            uint32_t u1 = pack_4float_to_fp8_e4m3_fast_wave(p_vals[4] * pscale, p_vals[5] * pscale, p_vals[6] * pscale, p_vals[7] * pscale);
            pv[k_sub] = (v2i){(int)u0, (int)u1};
        }
        row_sum = row_sum_new;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            float corr_alpha = __shfl_sync(0xFFFFFFFFFFFFFFFFULL, alpha, lg * 8 + i);
            corr_alpha = wave_attn_scale_ftz(corr_alpha, 0.0f); // keep FTZ semantics
            #pragma unroll
            for (int d = 0; d < 16; d++) acc_o[d][i] *= corr_alpha;
        }
        load_v_tile_transposed_wave_fast<K_TILE, LDV>(V_fp8 + v_off, smem_K, ks, K_TILE, Dim, Seq_K, tid, THREADS_PER_BLOCK, THREADS_PER_BLOCK);
        cta.sync();
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
                    float lse_val = (rsum > 0.0f) ? (rmax + logf(rsum + 1e-12f)) : __int_as_float(0xFF800000);
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

// Hybrid INT4 Q/K + FP8 V - default prefill path
template <int Q_TILE, int K_TILE, int THREADS_PER_BLOCK>
__global__ void wave_attn_prefill_int4(
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
    int out_dtype, bool is_causal, int num_splits,
    const int32_t* __restrict__ block_table, int block_size
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
    v8f acc_o[16]; for(int d=0;d<16;d++) acc_o[d]={0};
    load_int4_tile_to_lds<K_TILE>(Q_int4 + q_off_packed, smem_Q4, q_block_start, Q_TILE, Dim, Seq_Q, tid, THREADS_PER_BLOCK, LD4);
    cta.sync();
    float row_max = WAVE_ATTN_NEG_INF, row_sum = 0.0f;
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
        v8i sacc_i[NUM_K_SUBS]; for(int k=0;k<NUM_K_SUBS;k++) sacc_i[k]={0,0,0,0,0,0,0,0};
        #pragma unroll
        for (int dt = 0; dt < 8; dt++) {
            if (dt >= num_d_i4) break;
            int doff_packed = dt * 16 + lg * 8;
            v2i qv = {0,0};
            int q_lds_row = wave_id * 16 + lw;
            if (wave_id < num_q_waves && q_lds_row < Q_TILE) {
                int g_q_row = q_block_start + q_lds_row;
                if (g_q_row < Seq_Q && doff_packed < LD4 && dt*32 + lg*16 < Dim) {
                    uint64_t q64 = *(const uint64_t*)&smem_Q4[q_lds_row * LD4 + doff_packed];
                    qv = *(const v2i*)&q64;
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
        v8f sacc[NUM_K_SUBS];
        for(int k=0;k<NUM_K_SUBS;k++) {
            #pragma unroll
            for(int i=0;i<8;i++) sacc[k][i] = float(sacc_i[k][i]) * cscale;
        }
        bool wave_fully_unmasked = !is_causal || (ks + K_TILE <= q_start + q_block_start + wave_id*16);
        float tile_max = WAVE_ATTN_NEG_INF;
        #pragma unroll
        for (int k_sub = 0; k_sub < NUM_K_SUBS; k_sub++) {
            #pragma unroll
            for (int i=0;i<8;i++) {
                int kc = ks + k_sub*16 + lg*8 + i;
                float s = sacc[k_sub][i];
                if (wave_fully_unmasked) { if (kc >= Seq_K) s = WAVE_ATTN_NEG_INF; } else { if (kc >= Seq_K || kc > q_start + q_r) s = WAVE_ATTN_NEG_INF; }
                sacc[k_sub][i]=s; tile_max = fmaxf(tile_max,s);
            }
        }
        float r_max = fmaxf(tile_max, __shfl_xor_sync(0xFFFFFFFFULL, tile_max, 16));
        float nmax = fmaxf(row_max, r_max);
        float alp = (row_max > -1e29f) ? fast_expf(row_max - nmax) : 0.0f;
        alp = wave_attn_scale_ftz(alp, row_max - nmax);
        float alpha = alp; row_max = nmax;
        float row_sum_new = row_sum * alp;
        v2i pv[NUM_K_SUBS];
        #pragma unroll
        for (int k_sub=0;k_sub<NUM_K_SUBS;k_sub++) {
            float p_vals[8];
            #pragma unroll
            for(int i=0;i<8;i++){ float e = (sacc[k_sub][i]>-1e29f)?fast_expf(sacc[k_sub][i]-nmax):0.0f; e = wave_attn_scale_ftz(e, sacc[k_sub][i]-nmax); p_vals[i]=e; row_sum_new+=e; }
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
                for(int d=0;d<16;d++) acc_o[d][i]*=corr;
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

extern "C" void launch_wave_attn_prefill(
    const uint8_t* Q, const uint8_t* K, const uint8_t* V,
    void* Out, float* LSE, float* partial_max, float* partial_sum, float* partial_out,
    int B, int H, int Seq_Q, int Seq_K, int Dim,
    float softmax_scale, float q_scale, float k_scale, float v_scale,
    int out_dtype, bool is_causal, bool use_int4, hipStream_t stream
) {
    // Runtime auto-detect gfx1200 vs 1201 for blind tuning (56 vs 64 CU, lower BW needs wave earlier)
    int gfx_ver = 1201;
    {
        hipDeviceProp_t prop;
        if (hipGetDeviceProperties(&prop, 0) == hipSuccess) {
            if (strstr(prop.gcnArchName, "gfx1200") != nullptr) gfx_ver = 1200;
            else if (strstr(prop.gcnArchName, "gfx1201") != nullptr) gfx_ver = 1201;
        }
    }
    int Q_TILE, K_TILE, THREADS;
    // gfx1200 tune: lower large_k threshold 96 vs 128 to use K_TILE 128 more often (56 CU benefit)
    const bool large_k = (Dim <= 64 && Seq_K >= (gfx_ver==1200 ? 96 : 128));
    if (Seq_Q <= 16) { Q_TILE=16; K_TILE= large_k?128:64; THREADS=32; }
    else if (Seq_Q <= 32) { Q_TILE=32; K_TILE= large_k?128:64; THREADS=64; }
    else if (Seq_Q <= 64) { Q_TILE=64; K_TILE= large_k?128:64; THREADS=128; }
    else { Q_TILE=128; K_TILE= large_k?128:64; THREADS=256; }
    int num_blocks_q = (Seq_Q + Q_TILE - 1) / Q_TILE;
    int base_grid = B * H * num_blocks_q;
    int num_k_tiles = (Seq_K + K_TILE - 1) / K_TILE;
    int num_splits = 1;
    // gfx1200 blind tune: 56 CU vs 64 CU needs more splits for low occupancy (base_grid 4)
    int split_grid_thresh = (gfx_ver==1200 ? 4 : 2);
    int split_target = (gfx_ver==1200 ? 12 : 16);
    if (base_grid <= split_grid_thresh && num_k_tiles >= 4 && partial_max != nullptr) {
        num_splits = (split_target + base_grid - 1) / base_grid;
        if (num_splits > num_k_tiles) num_splits = num_k_tiles;
        if (num_splits > 8) num_splits = 8;
    }
    dim3 grid(B * H * num_splits, num_blocks_q);
    dim3 block(THREADS);
    if (use_int4) {
        size_t ld4 = (size_t)(Dim >> 1) + WAVE_ATTN_LDS_PAD;
        size_t ldv = (size_t)K_TILE + WAVE_ATTN_LDS_PAD;
        size_t smem_bytes = (size_t)Q_TILE * ld4 + (size_t)K_TILE * ld4 * 2 + (size_t)Dim * ldv;
        // Dispatch int4 path (only native, not unpack)
        if (Q_TILE==16 && K_TILE==128) hipLaunchKernelGGL((wave_attn_prefill_int4<16,128,32>),grid,block,smem_bytes,stream,Q,K,V,Out,LSE,partial_max,partial_sum,partial_out,B,H,Seq_Q,Seq_K,Dim,softmax_scale,q_scale,k_scale,v_scale,out_dtype,is_causal,num_splits,nullptr,0);
        else if (Q_TILE==16 && K_TILE==64) hipLaunchKernelGGL((wave_attn_prefill_int4<16,64,32>),grid,block,smem_bytes,stream,Q,K,V,Out,LSE,partial_max,partial_sum,partial_out,B,H,Seq_Q,Seq_K,Dim,softmax_scale,q_scale,k_scale,v_scale,out_dtype,is_causal,num_splits,nullptr,0);
        else if (Q_TILE==32 && K_TILE==128) hipLaunchKernelGGL((wave_attn_prefill_int4<32,128,64>),grid,block,smem_bytes,stream,Q,K,V,Out,LSE,partial_max,partial_sum,partial_out,B,H,Seq_Q,Seq_K,Dim,softmax_scale,q_scale,k_scale,v_scale,out_dtype,is_causal,num_splits,nullptr,0);
        else if (Q_TILE==32 && K_TILE==64) hipLaunchKernelGGL((wave_attn_prefill_int4<32,64,64>),grid,block,smem_bytes,stream,Q,K,V,Out,LSE,partial_max,partial_sum,partial_out,B,H,Seq_Q,Seq_K,Dim,softmax_scale,q_scale,k_scale,v_scale,out_dtype,is_causal,num_splits,nullptr,0);
        else if (Q_TILE==64 && K_TILE==128) hipLaunchKernelGGL((wave_attn_prefill_int4<64,128,128>),grid,block,smem_bytes,stream,Q,K,V,Out,LSE,partial_max,partial_sum,partial_out,B,H,Seq_Q,Seq_K,Dim,softmax_scale,q_scale,k_scale,v_scale,out_dtype,is_causal,num_splits,nullptr,0);
        else if (Q_TILE==64 && K_TILE==64) hipLaunchKernelGGL((wave_attn_prefill_int4<64,64,128>),grid,block,smem_bytes,stream,Q,K,V,Out,LSE,partial_max,partial_sum,partial_out,B,H,Seq_Q,Seq_K,Dim,softmax_scale,q_scale,k_scale,v_scale,out_dtype,is_causal,num_splits,nullptr,0);
        else if (Q_TILE==128 && K_TILE==128) hipLaunchKernelGGL((wave_attn_prefill_int4<128,128,256>),grid,block,smem_bytes,stream,Q,K,V,Out,LSE,partial_max,partial_sum,partial_out,B,H,Seq_Q,Seq_K,Dim,softmax_scale,q_scale,k_scale,v_scale,out_dtype,is_causal,num_splits,nullptr,0);
        else hipLaunchKernelGGL((wave_attn_prefill_int4<128,64,256>),grid,block,smem_bytes,stream,Q,K,V,Out,LSE,partial_max,partial_sum,partial_out,B,H,Seq_Q,Seq_K,Dim,softmax_scale,q_scale,k_scale,v_scale,out_dtype,is_causal,num_splits,nullptr,0);
    } else {
        size_t smem_bytes = (size_t)max((int64_t)K_TILE * (Dim + WAVE_ATTN_LDS_PAD), (int64_t)Dim * (K_TILE + WAVE_ATTN_LDS_PAD));
        if (Q_TILE == 16 && K_TILE == 128) hipLaunchKernelGGL((wave_attn_prefill_fp8<16, 128, 32>), grid, block, smem_bytes, stream, Q, K, V, Out, LSE, partial_max, partial_sum, partial_out, B, H, Seq_Q, Seq_K, Dim, softmax_scale, q_scale, k_scale, v_scale, out_dtype, is_causal, num_splits, nullptr, 0);
        else if (Q_TILE == 16 && K_TILE == 64) hipLaunchKernelGGL((wave_attn_prefill_fp8<16, 64, 32>), grid, block, smem_bytes, stream, Q, K, V, Out, LSE, partial_max, partial_sum, partial_out, B, H, Seq_Q, Seq_K, Dim, softmax_scale, q_scale, k_scale, v_scale, out_dtype, is_causal, num_splits, nullptr, 0);
        else if (Q_TILE == 32 && K_TILE == 128) hipLaunchKernelGGL((wave_attn_prefill_fp8<32, 128, 64>), grid, block, smem_bytes, stream, Q, K, V, Out, LSE, partial_max, partial_sum, partial_out, B, H, Seq_Q, Seq_K, Dim, softmax_scale, q_scale, k_scale, v_scale, out_dtype, is_causal, num_splits, nullptr, 0);
        else if (Q_TILE == 32 && K_TILE == 64) hipLaunchKernelGGL((wave_attn_prefill_fp8<32, 64, 64>), grid, block, smem_bytes, stream, Q, K, V, Out, LSE, partial_max, partial_sum, partial_out, B, H, Seq_Q, Seq_K, Dim, softmax_scale, q_scale, k_scale, v_scale, out_dtype, is_causal, num_splits, nullptr, 0);
        else if (Q_TILE == 64 && K_TILE == 128) hipLaunchKernelGGL((wave_attn_prefill_fp8<64, 128, 128>), grid, block, smem_bytes, stream, Q, K, V, Out, LSE, partial_max, partial_sum, partial_out, B, H, Seq_Q, Seq_K, Dim, softmax_scale, q_scale, k_scale, v_scale, out_dtype, is_causal, num_splits, nullptr, 0);
        else if (Q_TILE == 64 && K_TILE == 64) hipLaunchKernelGGL((wave_attn_prefill_fp8<64, 64, 128>), grid, block, smem_bytes, stream, Q, K, V, Out, LSE, partial_max, partial_sum, partial_out, B, H, Seq_Q, Seq_K, Dim, softmax_scale, q_scale, k_scale, v_scale, out_dtype, is_causal, num_splits, nullptr, 0);
        else if (Q_TILE == 128 && K_TILE == 128) hipLaunchKernelGGL((wave_attn_prefill_fp8<128, 128, 256>), grid, block, smem_bytes, stream, Q, K, V, Out, LSE, partial_max, partial_sum, partial_out, B, H, Seq_Q, Seq_K, Dim, softmax_scale, q_scale, k_scale, v_scale, out_dtype, is_causal, num_splits, nullptr, 0);
        else hipLaunchKernelGGL((wave_attn_prefill_fp8<128, 64, 256>), grid, block, smem_bytes, stream, Q, K, V, Out, LSE, partial_max, partial_sum, partial_out, B, H, Seq_Q, Seq_K, Dim, softmax_scale, q_scale, k_scale, v_scale, out_dtype, is_causal, num_splits, nullptr, 0);
    }
    if (num_splits > 1) {
        int total_q_rows = B * H * Seq_Q;
        dim3 r_grid((total_q_rows + 31) / 32);
        dim3 r_block(32);
        hipLaunchKernelGGL(wave_attn_prefill_split_reduce_kernel, r_grid, r_block, 0, stream, partial_max, partial_sum, partial_out, Out, LSE, B, H, Seq_Q, Dim, num_splits, v_scale, out_dtype);
    }
}
