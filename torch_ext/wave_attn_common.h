#pragma once
// torch_ext/wave_attn_common.h
// Shared helpers for WaveAttention inference kernels.
// Extracted from torch_ext/wave_attn.hip to keep prefill/decode/long in sync.
// Keep in sync with hip_quant_util.h conversions.

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

// Native GFX12 1-cycle FP32 -> FP8 E4M3 packing
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

// LDS bank-conflict padding: 8B pad makes bank = (lw*2+const)%32 for K and (lw*18+const)%32 for V -> 16 distinct banks
#define WAVE_ATTN_LDS_PAD 8

// Division-free uint4 global -> LDS with 8B-split stores (padded rows 8B-aligned)
__device__ inline void load_tile_to_lds_wave_fast(
    const uint8_t* __restrict__ src_global,
    uint8_t* __restrict__ dst_lds,
    int g_row_start, int num_rows, int dim, int total_g_rows,
    int tid_start, int tid_stride, int tid_count, int dst_row_stride
) {
    int total_bytes = num_rows * dim;
    int num_u4 = total_bytes / 16;
    const uint4* g_u4 = (const uint4*)src_global;
    int u4_per_row_shift = (dim == 256) ? 4 : ((dim == 128) ? 3 : 2);
    int u4_per_row_mask  = (dim == 256) ? 15 : ((dim == 128) ? 7 : 3);
    for (int i = tid_start; i < num_u4; i += tid_stride) {
        int r = i >> u4_per_row_shift;
        int c = (i & u4_per_row_mask) << 4;
        int g_r = g_row_start + r;
        uint4 val;
        if (g_r < total_g_rows && c < dim) val = g_u4[(g_r * dim + c) >> 4];
        else val = make_uint4(0, 0, 0, 0);
        uint2* dst = (uint2*)(dst_lds + r * dst_row_stride + c);
        dst[0] = make_uint2(val.x, val.y);
        dst[1] = make_uint2(val.z, val.w);
    }
}

// INT4 Q/K tile load - packed 2 per byte, dim2=Dim/2 + PAD
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
    int u4_per_row_shift = (dim == 256) ? 3 : ((dim == 128) ? 2 : 1);
    int u4_per_row_mask  = (dim == 256) ? 7 : ((dim == 128) ? 3 : 1);
    for (int i = tid_start; i < num_u4; i += tid_stride) {
        int r = i >> u4_per_row_shift;
        int c_bytes = (i & u4_per_row_mask) << 4;
        int g_r = g_row_start + r;
        uint4 val;
        if (g_r < total_g_rows && (c_bytes<<1) < dim) val = g_u4[((g_r * dim2 + c_bytes) >> 4)];
        else val = make_uint4(0,0,0,0);
        uint2* dst = (uint2*)(dst_lds + r * dst_row_stride + c_bytes);
        dst[0] = make_uint2(val.x, val.y);
        dst[1] = make_uint2(val.z, val.w);
    }
}

// INT4 -> FP8 E4M3 LUT where int4 0..15 maps: 0=0,7=7,8=-8,15=-1
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

__device__ inline void unpack_int4_tile_to_fp8_vec(
    uint8_t* __restrict__ dst_lds, const uint8_t* __restrict__ src_lds,
    int num_rows, int dim, int dst_stride, int src_stride,
    int tid, int stride) {
    int dim2 = dim >> 1;
    for (int r = tid; r < num_rows; r += stride) {
        int c = 0;
        for (; c + 4 <= dim2; c += 4) {
            uint32_t packed = *(const uint32_t*)&src_lds[r * src_stride + c];
            uint32_t o0, o1;
            unpack_packed32_to_fp8_pair(packed, o0, o1);
            *(uint32_t*)&dst_lds[r * dst_stride + c*2    ] = o0;
            *(uint32_t*)&dst_lds[r * dst_stride + c*2 + 4] = o1;
        }
        for (; c < dim2; ++c) {
            uint8_t packed = src_lds[r * src_stride + c];
            dst_lds[r * dst_stride + c*2    ] = INT4_TO_FP8_LUT[packed & 0xF];
            dst_lds[r * dst_stride + c*2 + 1] = INT4_TO_FP8_LUT[packed >> 4];
        }
    }
}

// Transposed V load variants
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
    int u4_per_row_shift = (dim == 256) ? 4 : ((dim == 128) ? 3 : 2);
    int u4_per_row_mask  = (dim == 256) ? 15 : ((dim == 128) ? 7 : 3);
    for (int i = tid_start; i < num_u4; i += tid_stride) {
        int r = i >> u4_per_row_shift;
        int c = (i & u4_per_row_mask) << 4;
        int g_r = g_row_start + r;
        uint4 u4;
        if (g_r < total_g_rows && c < dim) u4 = g_u4[(g_r * dim + c) >> 4];
        else u4 = make_uint4(0, 0, 0, 0);
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
        if (g_r0 < total_g_rows) u4_0 = (c0 < dim) ? g_u4[((g_r0)*dim + c0)>>4] : make_uint4(0,0,0,0);
        else u4_0 = make_uint4(0,0,0,0);
        if (g_r0+1 < total_g_rows) u4_1 = (c0 < dim) ? g_u4[((g_r0+1)*dim + c0)>>4] : make_uint4(0,0,0,0);
        else u4_1 = make_uint4(0,0,0,0);
        if (g_r0+2 < total_g_rows) u4_2 = (c0 < dim) ? g_u4[((g_r0+2)*dim + c0)>>4] : make_uint4(0,0,0,0);
        else u4_2 = make_uint4(0,0,0,0);
        if (g_r0+3 < total_g_rows) u4_3 = (c0 < dim) ? g_u4[((g_r0+3)*dim + c0)>>4] : make_uint4(0,0,0,0);
        else u4_3 = make_uint4(0,0,0,0);
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

// NaN/Inf guards
#define WAVE_ATTN_NEG_INF -1e30f
#define WAVE_ATTN_FTZ_THRESH -75.0f
__device__ inline float wave_attn_scale_ftz(float scale, float diff) {
    // FTZ: zero out exp(diff) contribution when diff < FTZ_THRESH to avoid denorm
    return diff >= WAVE_ATTN_FTZ_THRESH ? scale : 0.0f;
}
