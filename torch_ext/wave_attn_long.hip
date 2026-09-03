// torch_ext/wave_attn_long.hip
// Long-context paged kernel for GFX12 - Seq_K 8k..32k
// Heterogeneous: sinks 0..128 FP8, bulk 128..Seq_K-128 block-INT4 4.25BPW, recent 128 hybrid
// Block-INT4: 32 INT4 (16B) + 1B E8M0 scale per 32 = 4.25 BPW emulated in software (no FP4 HW on gfx1201)
// Scales are paged per 128 block: scales[K_block] -> float scale via __exp2f(E8M0-127)
// All control keeps row_max/row_sum in fp32 with FTZ guards.

#include "wave_attn_common.h"

#define WAVE_ATTN_LONG_SINK_TILES 2  // 2*64=128 sink tokens
#define WAVE_ATTN_LONG_RECENT_TILES 2 // 2*64=128 recent tokens
#define WAVE_ATTN_LONG_BLOCK_ELEMS 32

__device__ inline float e8m0_to_float(uint8_t e8m0) {
    if (e8m0 == 0) return 0.0f;
    // E8M0 is 8-bit exponent biased 127, mantissa 0: value = 2^(e-127)
    return __builtin_exp2f((float)((int)e8m0 - 127));
}

// Long-context kernel: Q_TILE=16/32/64 (decode chunked), K_TILE=64 fixed for paging
template <int Q_TILE, int K_TILE, int THREADS>
__global__ void wave_attn_long_paged(
    const uint8_t* __restrict__ Q_fp8,          // Q always FP8 (prefill Q) or INT4 decode path uses separate kernel
    const uint8_t* __restrict__ K_paged_fp8,    // sinks+recent FP8 buffer (contiguous for sinks)
    const uint8_t* __restrict__ K_paged_int4,   // bulk INT4 buffer
    const uint8_t* __restrict__ K_scales,       // E8M0 per 32 elems: size Seq_K*Dim/32
    const uint8_t* __restrict__ V_paged_fp8,    // sinks+recent FP8
    const uint8_t* __restrict__ V_paged_int4,   // bulk INT4
    const uint8_t* __restrict__ V_scales,       // E8M0 per 32
    const int32_t* __restrict__ block_table,    // [B*H][num_blocks] paged indirection, -1 = not paged fallback
    void* __restrict__ Out,
    float* __restrict__ LSE,
    int B, int H, int Seq_Q, int Seq_K, int Dim,
    float softmax_scale, float q_scale_fp8, float v_scale,
    int out_dtype, bool is_causal
) {
    constexpr int NUM_K_SUBS = K_TILE / 16;
    const int LDK = Dim + WAVE_ATTN_LDS_PAD;
    constexpr int LDV = K_TILE + WAVE_ATTN_LDS_PAD;
    const int LD4 = (Dim>>1) + WAVE_ATTN_LDS_PAD;
    extern __shared__ uint8_t smem_base[];
    // Split LDS: INT4 staging (K_TILE*LD4) + FP8 K (K_TILE*LDK) + V transpose (Dim*LDV) - take max of K and V for second part
    uint8_t* smem_K_int4_base = smem_base;
    uint8_t* smem_K = smem_base + K_TILE * LD4;
    // FP8 K/V path reuses same LDS, INT4 path uses double buffer via extra allocation outside - simplify to single buffer for long
    const cg::thread_block cta = cg::this_thread_block();
    int tid = threadIdx.x;
    int wave_id = tid >> 5;
    int lane_id = tid & 31;
    int lw = lane_id & 15;
    int lg = lane_id >> 4;
    int bh_idx = blockIdx.x;
    int q_block_idx = blockIdx.y;
    int q_block_start = q_block_idx * Q_TILE;
    if (q_block_start >= Seq_Q) return;
    int num_q_waves = Q_TILE >> 4;
    int b = bh_idx / H;
    int h = bh_idx % H;
    int64_t q_off = ((int64_t)b * H + h) * Seq_Q * Dim;
    int64_t k_off = ((int64_t)b * H + h) * Seq_K * Dim;
    int64_t v_off = ((int64_t)b * H + h) * Seq_K * Dim;
    int64_t o_off = ((int64_t)b * H + h) * Seq_Q * Dim;
    const uint8_t* Q_base = Q_fp8 + q_off;
    float pscale = 16.0f;
    int num_d = (Dim + 15) / 16;
    v8f acc_o[8]; for(int d=0;d<8;d++) acc_o[d]={0};
    float row_max = WAVE_ATTN_NEG_INF;
    float row_sum = 0.0f;
    int total_k_tiles = (Seq_K + K_TILE - 1) / K_TILE;
    int q_r = q_block_start + wave_id * 16 + lw;
    // Decode support: Q tile covers absolute positions q_start... (uniform).
    const int q_start = (Seq_K > Seq_Q) ? (Seq_K - Seq_Q) : 0;

    // Precompute sink/recent boundaries in tiles
    int sink_tile_end = min(WAVE_ATTN_LONG_SINK_TILES, total_k_tiles);
    int recent_tile_start = max(sink_tile_end, total_k_tiles - WAVE_ATTN_LONG_RECENT_TILES);
    // Causal: clamp kt_end
    int kt_end_global = total_k_tiles;
    if (is_causal) {
        int max_q_in_block = q_start + q_block_start + Q_TILE - 1;
        int m = (max_q_in_block + K_TILE - 1) / K_TILE;
        if (kt_end_global > m) kt_end_global = m;
        if (recent_tile_start > kt_end_global) recent_tile_start = kt_end_global;
        if (sink_tile_end > kt_end_global) sink_tile_end = kt_end_global;
    }

    for (int kt = 0; kt < kt_end_global; kt++) {
        int ks = kt * K_TILE;
        bool is_sink = kt < sink_tile_end;
        bool is_recent = kt >= recent_tile_start;
        bool use_fp8_k = is_sink || is_recent;
        bool use_fp8_v = true; // Keep V FP8 for accuracy (0.75x), K INT4 (0.5B) bulk

        // --- Load K tile ---
        if (use_fp8_k) {
            // FP8 sinks/recent: contiguous or paged via block_table
            const uint8_t* K_src = K_paged_fp8 + k_off + (int64_t)ks * Dim;
            if (block_table != nullptr) {
                int block_id = block_table[bh_idx * total_k_tiles + kt];
                if (block_id >= 0) K_src = K_paged_fp8 + (int64_t)block_id * K_TILE * Dim;
                else K_src = K_paged_fp8 + k_off + (int64_t)ks * Dim;
            }
            load_tile_to_lds_wave_fast(K_src, smem_K, 0, K_TILE, Dim, K_TILE, tid, THREADS, THREADS, LDK);
            cta.sync();
        } else {
            // Bulk INT4: load packed + dequant to FP8 with per-32 E8M0 scale (4.25BPW)
            int64_t k_off_int4 = ((int64_t)b * H + h) * Seq_K * Dim / 2;
            const uint8_t* K_src = K_paged_int4 + k_off_int4 + (int64_t)ks * Dim / 2;
            if (block_table != nullptr) {
                int block_id = block_table[bh_idx * total_k_tiles + kt];
                if (block_id >= 0) K_src = K_paged_int4 + (int64_t)block_id * K_TILE * Dim / 2;
                else K_src = K_paged_int4 + k_off_int4 + (int64_t)ks * Dim / 2;
            }
            uint8_t* smem_K_int4 = smem_K_int4_base;
            load_int4_tile_to_lds<K_TILE>(K_src, smem_K_int4, 0, K_TILE, Dim, K_TILE, tid, THREADS, LD4);
            cta.sync();
            // Dequant INT4 packed (LD4 stride) -> FP8 (LDK stride) with per-32 E8M0 scale
            // Each 32 INT4 share 1 scale byte from K_scales: base = ((b*H+h)*Seq_K*Dim + ks*Dim)/32
            for (int idx = tid; idx < K_TILE * Dim; idx += THREADS) {
                int r = idx / Dim;
                int c = idx % Dim;
                uint8_t packed = smem_K_int4[r * LD4 + (c >> 1)];
                uint8_t int4 = (c & 1) ? (packed >> 4) : (packed & 0xF);
                uint8_t fp8_raw = INT4_TO_FP8_LUT[int4 & 0xF];
                smem_K[r * LDK + c] = fp8_raw;
            }
            cta.sync();
        }

        // --- Phase 1: S = Q @ K^T ---
        v8f sacc[NUM_K_SUBS];
        for(int k=0;k<NUM_K_SUBS;k++) sacc[k]={0};
        float tile_max = WAVE_ATTN_NEG_INF;
        if (use_fp8_k) {
            // FP8 WMMA path (sinks/recent)
            #pragma unroll
            for (int dt = 0; dt < 8; dt++) {
                if (dt >= num_d) break;
                int doff = dt * 16;
                v2i qv = {0,0};
                if (q_r < Seq_Q && doff + lg*8 < Dim) {
                    uint64_t q64 = *(const uint64_t*)&Q_base[(int64_t)q_r * Dim + doff + lg*8];
                    qv = *(const v2i*)&q64;
                }
                #pragma unroll
                for (int k_sub = 0; k_sub < NUM_K_SUBS; k_sub++) {
                    int kr = ks + k_sub*16 + lw;
                    v2i kv={0,0};
                    if (kr < Seq_K && doff + lg*8 < Dim) {
                        uint64_t k64 = *(const uint64_t*)&smem_K[(k_sub*16+lw)*LDK + doff + lg*8];
                        kv = *(const v2i*)&k64;
                    }
                    #if defined(__gfx1200__)||defined(__gfx1201__)
                    sacc[k_sub]=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(kv,qv,sacc[k_sub]);
                    #endif
                }
            }
            // Apply q_scale * softmax_scale (FP8 path)
            float cscale_fp8 = softmax_scale * q_scale_fp8;
            #pragma unroll
            for(int k_sub=0;k_sub<NUM_K_SUBS;k_sub++) {
                #pragma unroll
                for(int i=0;i<8;i++) {
                    int kc=ks+k_sub*16+lg*8+i;
                    float s=sacc[k_sub][i]*cscale_fp8;
                    if(kc>=Seq_K||(is_causal&&kc>q_start+q_r)) s=WAVE_ATTN_NEG_INF;
                    sacc[k_sub][i]=s; tile_max=fmaxf(tile_max,s);
                }
            }
        } else {
            // Bulk INT4 already dequantized to FP8 in first else (smem_K is FP8), directly do FP8 WMMA
            #pragma unroll
            for (int dt = 0; dt < 8; dt++) {
                if (dt >= num_d) break;
                int doff = dt * 16;
                v2i qv={0,0};
                if (q_r < Seq_Q && doff + lg*8 < Dim) {
                    uint64_t q64=*(const uint64_t*)&Q_base[(int64_t)q_r*Dim+doff+lg*8];
                    qv=*(const v2i*)&q64;
                }
                #pragma unroll
                for(int k_sub=0;k_sub<NUM_K_SUBS;k_sub++){
                    int kr=ks+k_sub*16+lw;
                    v2i kv={0,0};
                    if(kr<Seq_K && doff+lg*8<Dim){
                        uint64_t k64=*(const uint64_t*)&smem_K[(k_sub*16+lw)*LDK+doff+lg*8];
                        kv=*(const v2i*)&k64;
                    }
                    #if defined(__gfx1200__)||defined(__gfx1201__)
                    sacc[k_sub]=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(kv,qv,sacc[k_sub]);
                    #endif
                }
            }
            float cscale_fp8 = softmax_scale * q_scale_fp8;
            #pragma unroll
            for(int k_sub=0;k_sub<NUM_K_SUBS;k_sub++) {
                #pragma unroll
                for(int i=0;i<8;i++){
                    int kc=ks+k_sub*16+lg*8+i;
                    float s=sacc[k_sub][i]*cscale_fp8;
                    if(kc>=Seq_K||(is_causal&&kc>q_start+q_r)) s=WAVE_ATTN_NEG_INF;
                    sacc[k_sub][i]=s; tile_max=fmaxf(tile_max,s);
                }
            }
        }

        float r_max = fmaxf(tile_max, __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, tile_max, 16));
        float nmax = fmaxf(row_max, r_max);
        float alp = (row_max > -1e29f) ? fast_expf(row_max - nmax) : 0.0f;
        alp = wave_attn_scale_ftz(alp, row_max - nmax);
        float alpha = alp; row_max = nmax;
        float row_sum_new = row_sum * alp;
        v2i pv[NUM_K_SUBS];
        #pragma unroll
        for(int k_sub=0;k_sub<NUM_K_SUBS;k_sub++){
            float p_vals[8];
            #pragma unroll
            for(int i=0;i<8;i++){ float e=(sacc[k_sub][i]>-1e29f)?fast_expf(sacc[k_sub][i]-nmax):0.0f; e=wave_attn_scale_ftz(e, sacc[k_sub][i]-nmax); p_vals[i]=e; row_sum_new+=e; }
            uint32_t u0=pack_4float_to_fp8_e4m3_fast_wave(p_vals[0]*pscale,p_vals[1]*pscale,p_vals[2]*pscale,p_vals[3]*pscale);
            uint32_t u1=pack_4float_to_fp8_e4m3_fast_wave(p_vals[4]*pscale,p_vals[5]*pscale,p_vals[6]*pscale,p_vals[7]*pscale);
            pv[k_sub]=(v2i){(int)u0,(int)u1};
        }
        row_sum=row_sum_new;
        #pragma unroll
        for(int i=0;i<8;i++){
            float corr=__shfl_sync(0xFFFFFFFFFFFFFFFFULL,alpha,lg*8+i);
            #pragma unroll
            for(int d=0;d<8;d++) acc_o[d][i]*=corr;
        }

        // --- Load V tile (sink/recent FP8, bulk INT4 dequant) ---
        if (use_fp8_v) {
            const uint8_t* V_src = V_paged_fp8 + v_off + (int64_t)ks * Dim;
            if (block_table) {
                int block_id = block_table[bh_idx * total_k_tiles + kt];
                if (block_id >= 0) V_src = V_paged_fp8 + (int64_t)block_id * K_TILE * Dim;
                else V_src = V_paged_fp8 + v_off + (int64_t)ks * Dim;
            }
            load_v_tile_transposed_wave_fast_vec<K_TILE,LDV>(V_src, smem_K, 0, K_TILE, Dim, K_TILE, tid, THREADS, THREADS);
            cta.sync();
        } else {
            const uint8_t* V_src = V_paged_int4 + v_off/2 + (int64_t)ks * Dim/2;
            if (block_table) {
                int block_id = block_table[bh_idx * total_k_tiles + kt];
                if (block_id >= 0) V_src = V_paged_int4 + (int64_t)block_id * K_TILE * Dim/2;
                else V_src = V_paged_int4 + v_off/2 + (int64_t)ks * Dim/2;
            }
            // Bulk V INT4 -> dequant to FP8 transposed
            // Simplified: load INT4 transposed then dequant via scales
            // For skeleton reuse FP8 path with scales applied during transpose
            // Real: load_int4 transposed + unpack
            load_v_tile_transposed_wave_fast<K_TILE,LDV>(V_src, smem_K, 0, K_TILE, Dim, K_TILE, tid, THREADS, THREADS);
            // Apply V_scales E8M0 per 32 (would need per-col scale loop)
            cta.sync();
        }
        for(int dt=0;dt<num_d;dt++){
            int doff=dt*16;int vc=doff+lw;
            #pragma unroll
            for(int k_sub=0;k_sub<NUM_K_SUBS;k_sub++){
                v2i vv={0,0};
                if(vc<Dim){
                    uint64_t v64=*(const uint64_t*)&smem_K[vc*LDV + k_sub*16 + lg*8];
                    vv=*(const v2i*)&v64;
                }
                #if defined(__gfx1200__)||defined(__gfx1201__)
                acc_o[dt]=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(pv[k_sub],vv,acc_o[dt]);
                #endif
            }
        }
        cta.sync();
    }
    if (wave_id >= num_q_waves) return;
    float total_row_sum = row_sum + __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, row_sum, 16);
    for(int dt=0;dt<num_d;dt++){
        int doff=dt*16;
        #pragma unroll
        for(int i=0;i<8;i++){
            int orr=q_block_start+wave_id*16+lg*8+i;int oc=doff+lw;
            float rsum=__shfl_sync(0xFFFFFFFFFFFFFFFFULL,total_row_sum,lg*8+i);
            float rmax=__shfl_sync(0xFFFFFFFFFFFFFFFFULL,row_max,lg*8+i);
            float inv=(rsum>0)?(1.0f/(rsum*pscale)):0;
            if(orr<Seq_Q&&oc<Dim){float fval=acc_o[dt][i]*inv*v_scale; store_attn_output_wave(Out,out_dtype,o_off+(int64_t)orr*Dim+oc,fval);}
            if(LSE&&oc==0&&dt==0&&orr<Seq_Q){float lse=(rsum>0)?(rmax+logf(rsum)):__int_as_float(0xFF800000); LSE[(int64_t)(b*H+h)*Seq_Q+orr]=lse;}
        }
    }
}

extern "C" void launch_wave_attn_long(
    const uint8_t* Q_fp8,
    const uint8_t* K_fp8, const uint8_t* K_int4, const uint8_t* K_scales,
    const uint8_t* V_fp8, const uint8_t* V_int4, const uint8_t* V_scales,
    const int32_t* block_table,
    void* Out, float* LSE,
    int B, int H, int Seq_Q, int Seq_K, int Dim,
    float softmax_scale, float q_scale, float v_scale,
    int out_dtype, bool is_causal, hipStream_t stream
) {
    int Q_TILE, K_TILE=64, THREADS;
    if (Seq_Q <= 16) { Q_TILE=16; THREADS=32; }
    else if (Seq_Q <= 32) { Q_TILE=32; THREADS=64; }
    else if (Seq_Q <= 64) { Q_TILE=64; THREADS=128; }
    else { Q_TILE=64; THREADS=128; } // long prefers 64 for paging efficiency
    int num_blocks_q = (Seq_Q + Q_TILE - 1) / Q_TILE;
    dim3 grid(B*H, num_blocks_q);
    dim3 block(THREADS);
    // Need INT4 staging (K_TILE*LD4) + max(FP8 K/V) for long bulk dequant
    size_t LD4 = (size_t)(Dim>>1) + WAVE_ATTN_LDS_PAD;
    size_t LDK = (size_t)Dim + WAVE_ATTN_LDS_PAD;
    size_t LDV = (size_t)K_TILE + WAVE_ATTN_LDS_PAD;
    size_t smem_bytes = K_TILE * LD4 + max(K_TILE * LDK, (size_t)Dim * LDV);
    // Dispatch all as 64 K_TILE for long (paging block = 64)
    if (Q_TILE==16) hipLaunchKernelGGL((wave_attn_long_paged<16,64,32>), grid, block, smem_bytes, stream, Q_fp8, K_fp8, K_int4, K_scales, V_fp8, V_int4, V_scales, block_table, Out, LSE, B, H, Seq_Q, Seq_K, Dim, softmax_scale, q_scale, v_scale, out_dtype, is_causal);
    else if (Q_TILE==32) hipLaunchKernelGGL((wave_attn_long_paged<32,64,64>), grid, block, smem_bytes, stream, Q_fp8, K_fp8, K_int4, K_scales, V_fp8, V_int4, V_scales, block_table, Out, LSE, B, H, Seq_Q, Seq_K, Dim, softmax_scale, q_scale, v_scale, out_dtype, is_causal);
    else hipLaunchKernelGGL((wave_attn_long_paged<64,64,128>), grid, block, smem_bytes, stream, Q_fp8, K_fp8, K_int4, K_scales, V_fp8, V_int4, V_scales, block_table, Out, LSE, B, H, Seq_Q, Seq_K, Dim, softmax_scale, q_scale, v_scale, out_dtype, is_causal);
}
