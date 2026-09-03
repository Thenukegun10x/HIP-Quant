// torch_ext/wave_attn_decode.hip
// Decode inference kernel for GFX12 - specialized for Seq_Q = 1..4 (BW-bound).
// Q_TILE=16 THREADS=32 only, Q in regs broadcast via shuffle, double-buffered K.
// INT4 Q/K (wmma_i32 16x16x32) + INT4 V (dequantized into transposed LDS).

#include "wave_attn_common.h"

// All-INT4 V transpose dequant: V INT4 packed 2 per byte -> FP8 transposed LDS.
// Load four rows and 32 columns per work item.  The uint4 reads avoid the
// scalar byte loads and div/mod operations of the original implementation.
template<int K_TILE, int LDV>
__device__ inline void load_v_int4_transposed_dequant(
    const uint8_t* __restrict__ src_int4,
    uint8_t* __restrict__ dst_fp8_lds,
    int g_row_start, int num_rows, int dim, int total_g_rows,
    int tid, int threads)
{
    // Dim is restricted to 64/128 by the binding, so each block is complete.
    const int blocks_per_row = dim >> 5;
    const int block_rows = K_TILE >> 2;
    const int total_blocks = block_rows * blocks_per_row;
    const int packed_row_stride = dim >> 1;
    const uint4* src_u4 = reinterpret_cast<const uint4*>(src_int4);

    for (int block = tid; block < total_blocks; block += threads) {
        const int r0 = (block / blocks_per_row) << 2;
        const int c0 = (block % blocks_per_row) << 5;
        const int g_r0 = g_row_start + r0;
        uint4 p0 = make_uint4(0, 0, 0, 0);
        uint4 p1 = make_uint4(0, 0, 0, 0);
        uint4 p2 = make_uint4(0, 0, 0, 0);
        uint4 p3 = make_uint4(0, 0, 0, 0);
        if (c0 < dim) {
            if (g_r0 + 0 < total_g_rows) p0 = src_u4[((g_r0 + 0) * packed_row_stride + (c0 >> 1)) >> 4];
            if (g_r0 + 1 < total_g_rows) p1 = src_u4[((g_r0 + 1) * packed_row_stride + (c0 >> 1)) >> 4];
            if (g_r0 + 2 < total_g_rows) p2 = src_u4[((g_r0 + 2) * packed_row_stride + (c0 >> 1)) >> 4];
            if (g_r0 + 3 < total_g_rows) p3 = src_u4[((g_r0 + 3) * packed_row_stride + (c0 >> 1)) >> 4];
        }
        const uint8_t* b0 = reinterpret_cast<const uint8_t*>(&p0);
        const uint8_t* b1 = reinterpret_cast<const uint8_t*>(&p1);
        const uint8_t* b2 = reinterpret_cast<const uint8_t*>(&p2);
        const uint8_t* b3 = reinterpret_cast<const uint8_t*>(&p3);
        #pragma unroll
        for (int c = 0; c < 32; ++c) {
            const int shift = (c & 1) << 2;
            const int packed_c = c >> 1;
            const uint32_t t =
                (uint32_t)INT4_TO_FP8_LUT[(b0[packed_c] >> shift) & 0xF] |
                ((uint32_t)INT4_TO_FP8_LUT[(b1[packed_c] >> shift) & 0xF] << 8) |
                ((uint32_t)INT4_TO_FP8_LUT[(b2[packed_c] >> shift) & 0xF] << 16) |
                ((uint32_t)INT4_TO_FP8_LUT[(b3[packed_c] >> shift) & 0xF] << 24);
            *reinterpret_cast<uint32_t*>(&dst_fp8_lds[(c0 + c) * LDV + r0]) = t;
        }
    }
}

// Decode: Q=1 specialized kernel
// Q_TILE is fixed 16 (1 wave = 32 threads covers K_TILE/16 sub-tiles). K_TILE templated 64/128.
// Q is loaded once to regs (not per-dt global reload) - saves L2 + VGPR reloads in wave_attn.hip:418.
template <int K_TILE>
__global__ __launch_bounds__(32, 4) void wave_attn_decode_int4(
    const uint8_t* __restrict__ Q_int4,
    const uint8_t* __restrict__ K_int4,
    const uint8_t* __restrict__ V_int4,
    void* __restrict__ Out,
    float* __restrict__ LSE,
    int B, int H, int Seq_Q, int Seq_K, int Dim,
    float softmax_scale, float q_scale, float k_scale, float v_scale,
    int out_dtype, bool is_causal
) {
    static_assert(K_TILE == 64 || K_TILE == 128, "K_TILE 64/128 only");
    constexpr int NUM_K_SUBS = K_TILE / 16;
    constexpr int Q_TILE = 16;
    constexpr int THREADS = 32;
    const int LD4 = (Dim>>1) + WAVE_ATTN_LDS_PAD;
    constexpr int LDV = K_TILE + WAVE_ATTN_LDS_PAD;
    extern __shared__ uint8_t smem_base[];
    // Single-buffered K (prefetch removed) to allow K_TILE 128 for Dim128 within 32KB: Q 16*LD4 + K*LD4 + Dim*LDV = 27KB
    size_t q4_bytes = (size_t)Q_TILE * LD4;
    size_t k4_bytes = (size_t)K_TILE * LD4;
    uint8_t* smem_Q4 = smem_base;
    uint8_t* smem_K4 = smem_Q4 + q4_bytes;
    uint8_t* smem_V  = smem_K4 + k4_bytes;
    const cg::thread_block cta = cg::this_thread_block();
    int tid = threadIdx.x;
    int lane_id = tid & 31;
    int lw = lane_id & 15;
    int lg = lane_id >> 4;
    int bh_idx = blockIdx.x;
    int q_block_idx = blockIdx.y;
    // Decode grid: B*H, Q blocks cover Seq_Q (usually 1 block)
    int q_block_start = q_block_idx * Q_TILE;
    if (q_block_start >= Seq_Q) return;
    int b = bh_idx / H;
    int h = bh_idx % H;
    int64_t q_off_packed = ((int64_t)b * H + h) * Seq_Q * Dim / 2;
    int64_t k_off_packed = ((int64_t)b * H + h) * Seq_K * Dim / 2;
    int64_t v_off = ((int64_t)b * H + h) * Seq_K * Dim;
    int64_t o_off = ((int64_t)b * H + h) * Seq_Q * Dim;
    float cscale = softmax_scale * q_scale * k_scale;
    float pscale = 16.0f;
    int num_d = (Dim + 15) / 16;
    int num_d_i4 = (Dim + 31) / 32;
    // Load Q once -> regs (decode unique: no per-dt reload)
    // Q_TILE=16 so wave_id=0 always; q_r = q_block_start + lw
    int q_r = q_block_start + lw;
    // Decode support: Q tile covers absolute positions q_start... (uniform).
    const int q_start = (Seq_K > Seq_Q) ? (Seq_K - Seq_Q) : 0;
    // Load Q packed tile
    load_int4_tile_to_lds<K_TILE>(Q_int4 + q_off_packed, smem_Q4, q_block_start, Q_TILE, Dim, Seq_Q, tid, THREADS, LD4);
    cta.sync();
    // Pull Q into regs: 4 dt_i4 * v2i (covers Dim up to 128)
    v2i q_reg[4] = {{0,0},{0,0},{0,0},{0,0}};
    #pragma unroll
    for (int dt = 0; dt < 4; dt++) {
        if (dt >= num_d_i4) break;
        int doff_packed = dt * 16 + lg * 8;
        if (q_r < Seq_Q && doff_packed < LD4 && dt*32 + lg*16 < Dim) {
            uint64_t q64 = *(const uint64_t*)&smem_Q4[lw * LD4 + doff_packed];
            q_reg[dt] = *(const v2i*)&q64;
        }
    }
    cta.sync(); // Q regs ready, LDS can be reused for K prefetch

    v8f acc_o[8];
    for(int d=0;d<8;d++) acc_o[d] = (v8f){};
    float row_max = WAVE_ATTN_NEG_INF;
    float row_sum = 0.0f;
    int total_k_tiles = (Seq_K + K_TILE - 1) / K_TILE;
    for (int kt = 0; kt < total_k_tiles; kt++) {
        int ks = kt * K_TILE;
        // Load K tile (no double buffer, single sync)
        load_int4_tile_to_lds<K_TILE>(K_int4 + k_off_packed, smem_K4, ks, K_TILE, Dim, Seq_K, tid, THREADS, LD4);
        cta.sync();
        uint8_t* cur_K4 = smem_K4;
        v8i sacc_i[NUM_K_SUBS];
        for(int k=0;k<NUM_K_SUBS;k++) sacc_i[k]={0,0,0,0,0,0,0,0};
        #pragma unroll
        for (int dt = 0; dt < 4; dt++) {
            if (dt >= num_d_i4) break;
            v2i qv = q_reg[dt];
            #pragma unroll
            for (int k_sub = 0; k_sub < NUM_K_SUBS; k_sub++) {
                int kr = ks + k_sub*16 + lw;
                v2i kv = {0,0};
                if (kr < Seq_K) {
                    int doff_packed = dt * 16 + lg * 8;
                    if (dt*32 + lg*16 < Dim) {
                        uint64_t k64 = *(const uint64_t*)&cur_K4[(k_sub*16+lw)*LD4 + doff_packed];
                        kv = *(const v2i*)&k64;
                    }
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
        // Causal + Seq_K mask - wave_id 0 only, q_r is lane uniform per sub-tile
        float tile_max = WAVE_ATTN_NEG_INF;
        bool wave_fully_unmasked = !is_causal || (ks + K_TILE <= q_start + q_r);
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
            for(int i=0;i<8;i++){ float e=(sacc[k_sub][i]>-1e29f)?fast_expf(sacc[k_sub][i]-nmax):0.0f; e=wave_attn_scale_ftz(e, sacc[k_sub][i]-nmax); p_vals[i]=e; row_sum_new+=e; }
            uint32_t u0=pack_4float_to_fp8_e4m3_fast_wave(p_vals[0]*pscale,p_vals[1]*pscale,p_vals[2]*pscale,p_vals[3]*pscale);
            uint32_t u1=pack_4float_to_fp8_e4m3_fast_wave(p_vals[4]*pscale,p_vals[5]*pscale,p_vals[6]*pscale,p_vals[7]*pscale);
            pv[k_sub]=(v2i){(int)u0,(int)u1};
        }
        row_sum=row_sum_new;
        // Alpha rescale - decode has single row so broadcast is cheap
        float corr = __shfl_sync(0xFFFFFFFFULL, alpha, 0);
        if (corr != 1.0f) {
            #pragma unroll
            for(int d=0;d<8;d++) {
                #pragma unroll
                for(int i=0;i<8;i++) acc_o[d][i] *= corr;
            }
        }

        load_v_int4_transposed_dequant<K_TILE,LDV>(V_int4 + (v_off>>1), smem_V, ks, K_TILE, Dim, Seq_K, tid, THREADS);
        cta.sync();
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
        // V tile done, loop continues - K prefetch already in next buffer
    }
    // Output - only q_r < Seq_Q threads write
    float total_row_sum = row_sum + __shfl_xor_sync(0xFFFFFFFFULL, row_sum, 16);
    for(int dt=0;dt<num_d;dt++){
        int doff=dt*16;
        #pragma unroll
        for(int i=0;i<8;i++){
            int orr=q_block_start + lg*8 + i; // decode: wave_id 0, so lg*8 + i is Q row offset within tile
            int oc=doff+lw;
            // For Seq_Q=1, only lw holds column, i indexes Q row
            if (q_block_start==0 && lw==0) {} // keep wave uniform for shuffle
            float rsum=__shfl_sync(0xFFFFFFFFULL,total_row_sum,lg*8+i);
            float rmax=__shfl_sync(0xFFFFFFFFULL,row_max,lg*8+i);
            float inv=(rsum>0)?(1.0f/(rsum*pscale)):0;
            // Map: Q row is q_block_start + lg*8 + i when Q_TILE=16 (lg 0/1, i 0..7 => 16 rows)
            int q_idx = q_block_start + lg*8 + i;
            if(q_idx < Seq_Q && oc < Dim){
                float fval=acc_o[dt][i]*inv*v_scale;
                store_attn_output_wave(Out,out_dtype,o_off+(int64_t)q_idx*Dim+oc,fval);
            }
            if(LSE && oc==0 && dt==0 && q_idx < Seq_Q){
                float lse=(rsum>0)?(rmax+logf(rsum)):__int_as_float(0xFF800000);
                LSE[(int64_t)(b*H+h)*Seq_Q+q_idx]=lse;
            }
        }
    }
}

extern "C" void launch_wave_attn_decode(
    const uint8_t* Q_int4, const uint8_t* K_int4, const uint8_t* V_int4,
    void* Out, float* LSE,
    int B, int H, int Seq_Q, int Seq_K, int Dim,
    float softmax_scale, float q_scale, float k_scale, float v_scale,
    int out_dtype, bool is_causal, hipStream_t stream
) {
    int gfx_ver = 1201;
    { hipDeviceProp_t prop; if (hipGetDeviceProperties(&prop, 0)==hipSuccess) { if (strstr(prop.gcnArchName,"gfx1200")!=nullptr) gfx_ver=1200; } }
    // All-INT4 V to hit 3.5x (0.5B KV) 1B total vs hybrid 1.5B.
    bool use128 = (Seq_K >= 1024) || (Dim <= 64 && Seq_K >= (gfx_ver==1200 ? 96 : 128));
    bool large_k = use128;
    int K_TILE = large_k ? 128 : 64;
    int num_blocks_q = (Seq_Q + 15) / 16;
    dim3 grid(B * H, num_blocks_q);
    dim3 block(32);
    size_t ld4 = (size_t)(Dim >> 1) + WAVE_ATTN_LDS_PAD;
    size_t ldv = (size_t)K_TILE + WAVE_ATTN_LDS_PAD;
    size_t smem_bytes = 16 * ld4 + (size_t)K_TILE * ld4 + (size_t)Dim * ldv; // single K buffer for 128
    if (K_TILE == 128) hipLaunchKernelGGL((wave_attn_decode_int4<128>), grid, block, smem_bytes, stream, Q_int4, K_int4, V_int4, Out, LSE, B, H, Seq_Q, Seq_K, Dim, softmax_scale, q_scale, k_scale, v_scale, out_dtype, is_causal);
    else hipLaunchKernelGGL((wave_attn_decode_int4<64>), grid, block, smem_bytes, stream, Q_int4, K_int4, V_int4, Out, LSE, B, H, Seq_Q, Seq_K, Dim, softmax_scale, q_scale, k_scale, v_scale, out_dtype, is_causal);
}
