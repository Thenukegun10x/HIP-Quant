#include <hip/hip_runtime.h>
#include "../hip_quant_types.h"
#include "../hip_quant_util.h"

#define QK_K 256
#define GROUP_MAX_EPS 1e-15f

// make_qx_quants for n=16, nmax=32, rmse_type=1, qw=NULL
__device__ inline float make_qx_quants_q6(
    int grp, const float *s_x, int8_t *s_L
) {
    int base = grp * 16;
    float max_val = 0;
    float amax = 0;

    for (int k = 0; k < 16; ++k) {
        float ax = fabsf(s_x[base + k]);
        if (ax > amax) { amax = ax; max_val = s_x[base + k]; }
    }

    if (amax < GROUP_MAX_EPS) {
        for (int k = 0; k < 16; ++k) s_L[base + k] = 0;
        return 0.0f;
    }

    float iscale = -32.0f / max_val;
    float sumlx = 0, suml2 = 0;

    for (int k = 0; k < 16; ++k) {
        float xv = s_x[base + k];
        int l = nearest_int(iscale * xv);
        l = max(-32, min(31, l));
        s_L[base + k] = (int8_t)(l + 32);
        float w = xv * xv;
        sumlx += w * xv * l;
        suml2 += w * l * l;
    }

    float scale = suml2 > 0 ? sumlx / suml2 : 0.0f;
    float best = scale * sumlx;

    for (int is = -9; is <= 9; ++is) {
        if (is == 0) continue;
        iscale = -(32.0f + 0.1f * (float)is) / max_val;
        sumlx = suml2 = 0;
        int8_t Ltmp[16];
        for (int k = 0; k < 16; ++k) {
            float xv = s_x[base + k];
            int l = nearest_int(iscale * xv);
            l = max(-32, min(31, l));
            Ltmp[k] = (int8_t)(l + 32);
            float w = xv * xv;
            sumlx += w * xv * l;
            suml2 += w * l * l;
        }
        if (suml2 > 0 && sumlx * sumlx > best * suml2) {
            for (int k = 0; k < 16; ++k) s_L[base + k] = Ltmp[k];
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }

    return scale;
}

extern "C" __global__
__launch_bounds__(256, 4)
void quantize_q6_K_kernel(
    const float * __restrict__ src,
    uint8_t * __restrict__ dst,
    const float * __restrict__ imatrix,
    int nrows,
    int n_per_row
) {
    int row = blockIdx.x;
    int sb = blockIdx.y;
    int tid = threadIdx.x;
    int lane = tid & 15;
    int grp  = tid >> 4;

    int base = row * n_per_row + sb * QK_K + grp * 16 + lane;
    if (base > (row + 1) * n_per_row - 1) return;

    __shared__ float s_x[QK_K];
    __shared__ int8_t s_L[QK_K];
    __shared__ float s_scales[16];
    __shared__ float s_d_val;
    __shared__ int8_t s_sc[16];

    s_x[grp * 16 + lane] = src[base];
    __syncthreads();

    if (lane == 0) {
        float scale_val = make_qx_quants_q6(grp, s_x, s_L);
        s_scales[grp] = scale_val;
    }
    __syncthreads();

    if (tid == 0) {
        float max_scale = 0, max_abs_scale = 0;
        for (int j = 0; j < 16; ++j) {
            float abs_scale = fabsf(s_scales[j]);
            if (abs_scale > max_abs_scale) {
                max_abs_scale = abs_scale;
                max_scale = s_scales[j];
            }
        }

        block_q6_K *blk = (block_q6_K*)(dst + (row * (n_per_row / QK_K) + sb) * sizeof(block_q6_K));

        if (max_abs_scale < GROUP_MAX_EPS) {
            for (int j = 0; j < 128; ++j) blk->ql[j] = 0;
            for (int j = 0; j < 64; ++j) blk->qh[j] = 0;
            for (int j = 0; j < 16; ++j) blk->scales[j] = 0;
            blk->d = fp32_to_fp16(0.0f);
            s_d_val = 0.0f;
        } else {
            float iscale = -128.0f / max_scale;
            for (int j = 0; j < 16; ++j) {
                int l = nearest_int(iscale * s_scales[j]);
                s_sc[j] = (int8_t)min(127, l);
                blk->scales[j] = s_sc[j];
            }
            blk->d = fp32_to_fp16(1.0f / iscale);
            s_d_val = fp16_to_fp32(blk->d);
        }
    }
    __syncthreads();

    // Requantize
    float d_scale = s_d_val * (float)s_sc[grp];
    if (d_scale != 0.0f) {
        float xv = s_x[grp * 16 + lane];
        int l = nearest_int(xv / d_scale);
        l = max(-32, min(31, l));
        s_L[grp * 16 + lane] = (int8_t)(l + 32);
    }
    __syncthreads();

    // Pack ql[128] and qh[64]: process 128-element chunks
    block_q6_K *blk = (block_q6_K*)(dst + (row * (n_per_row / QK_K) + sb) * sizeof(block_q6_K));

    // Chunk 0: elements 0-127, chunk 1: elements 128-255
    // Each chunk: 32 threads write ql[0..31] and ql[32..63] and qh[0..31]
    if (tid < 64) {
        int chunk = tid / 32;
        int l = tid % 32;
        int base = chunk * 128;

        int8_t q1 = s_L[base + l];
        int8_t q2 = s_L[base + l + 32];
        int8_t q3 = s_L[base + l + 64];
        int8_t q4 = s_L[base + l + 96];

        // ql[0..31] = low nibble of q1 and q3
        blk->ql[chunk * 64 + l] = (uint8_t)((q1 & 0xF) | ((q3 & 0xF) << 4));
        // ql[32..63] = low nibble of q2 and q4
        blk->ql[chunk * 64 + l + 32] = (uint8_t)((q2 & 0xF) | ((q4 & 0xF) << 4));
        // qh[0..31] = high 2 bits of q1, q2, q3, q4
        blk->qh[chunk * 32 + l] = (uint8_t)(
            ((q1 >> 4) & 3) |
            (((q2 >> 4) & 3) << 2) |
            (((q3 >> 4) & 3) << 4) |
            (((q4 >> 4) & 3) << 6)
        );
    }
}
