#include <hip/hip_runtime.h>
#include "../hip_quant_types.h"
#include "../hip_quant_util.h"

#define QK_K 256

// make_qkx2_quants for n=16, nmax=3, use_mad=true
// Disable FMA contraction to match x86 SSE float evaluation order
#pragma STDC FP_CONTRACT OFF
__device__ inline float make_qkx2_quants_q2(
    int grp, const float *s_x, uint8_t *s_L,
    float *out_min
) {
    int base = grp * 16;
    float min_v = s_x[base];
    float max_v = s_x[base];
    float sum_w = fabsf(s_x[base]);
    float sum_x = sum_w * s_x[base];

    for (int k = 1; k < 16; ++k) {
        float xv = s_x[base + k];
        float wv = fabsf(xv);
        if (xv < min_v) min_v = xv;
        if (xv > max_v) max_v = xv;
        sum_w += wv;
        sum_x += wv * xv;
    }

    if (min_v > 0) min_v = 0;
    if (max_v <= min_v) {
        for (int k = 0; k < 16; ++k) s_L[base + k] = 0;
        *out_min = -min_v;
        return 0.0f;
    }

    float iscale = 3.0f / (max_v - min_v);
    float scale = 1.0f / iscale;
    float best_error = 0.0f;

    for (int k = 0; k < 16; ++k) {
        float xv = s_x[base + k];
        int l = nearest_int(iscale * (xv - min_v));
        l = max(0, min(3, l));
        s_L[base + k] = (uint8_t)l;
        float diff = fabsf(scale * l + min_v - xv);
        float wv = fabsf(xv);
        best_error += wv * diff;
    }

    uint8_t Laux[16];
    for (int is = 0; is <= 15; ++is) {
        iscale = (-0.5f + 0.1f * (float)is + 3.0f) / (max_v - min_v);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int k = 0; k < 16; ++k) {
            float xv = s_x[base + k];
            int l = nearest_int(iscale * (xv - min_v));
            l = max(0, min(3, l));
            Laux[k] = (uint8_t)l;
            float wv = fabsf(xv);
            float wl = wv * (float)l;
            sum_l += wl;
            sum_l2 += wl * (float)l;
            sum_xl += wl * xv;
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
            float this_min   = (sum_l2 * sum_x  - sum_l * sum_xl) / D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float cur_error = 0.0f;
            for (int k = 0; k < 16; ++k) {
                float diff = fabsf(this_scale * Laux[k] + this_min - s_x[base + k]);
                float wv = fabsf(s_x[base + k]);
                cur_error += wv * diff;
            }
            if (cur_error < best_error) {
                for (int k = 0; k < 16; ++k) s_L[base + k] = Laux[k];
                best_error = cur_error;
                scale = this_scale;
                min_v = this_min;
            }
        }
    }

    *out_min = -min_v;
    return scale;
}

extern "C" __global__
__launch_bounds__(256, 4)
void quantize_q2_K_kernel(
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
    __shared__ uint8_t s_L[QK_K];
    __shared__ float s_scales[16];
    __shared__ float s_mins[16];
    __shared__ float s_d_val;
    __shared__ float s_dm_val;
    __shared__ uint8_t s_sc[16];

    s_x[grp * 16 + lane] = src[base];
    __syncthreads();

    if (lane == 0) {
        float min_val;
        float scale_val = make_qkx2_quants_q2(grp, s_x, s_L, &min_val);
        s_scales[grp] = scale_val;
        s_mins[grp] = min_val;
    }
    __syncthreads();

    if (tid == 0) {
        float max_scale = 0.0f, max_min = 0.0f;
        for (int j = 0; j < 16; ++j) {
            if (s_scales[j] > max_scale) max_scale = s_scales[j];
            if (s_mins[j] > max_min) max_min = s_mins[j];
        }

        float q4scale = 15.0f;
        block_q2_K *blk = (block_q2_K*)(dst + (row * (n_per_row / QK_K) + sb) * sizeof(block_q2_K));

        if (max_scale > 0) {
            float iscale = q4scale / max_scale;
            for (int j = 0; j < 16; ++j) {
                int l = nearest_int(iscale * s_scales[j]);
                s_sc[j] = (uint8_t)l;
                blk->scales[j] = s_sc[j];
            }
            blk->d = fp32_to_fp16(max_scale / q4scale);
        } else {
            for (int j = 0; j < 16; ++j) { s_sc[j] = 0; blk->scales[j] = 0; }
            blk->d = fp32_to_fp16(0.0f);
        }
        if (max_min > 0) {
            float iscale = q4scale / max_min;
            for (int j = 0; j < 16; ++j) {
                int l = nearest_int(iscale * s_mins[j]);
                s_sc[j] |= (uint8_t)(l << 4);
                blk->scales[j] = s_sc[j];
            }
            blk->dmin = fp32_to_fp16(max_min / q4scale);
        } else {
            blk->dmin = fp32_to_fp16(0.0f);
        }

        s_d_val = fp16_to_fp32(blk->d);
        s_dm_val = fp16_to_fp32(blk->dmin);
    }
    __syncthreads();

    // Requantize with quantized scales (read from shared memory s_sc)
    uint8_t sc = s_sc[grp] & 0xF;
    uint8_t m  = s_sc[grp] >> 4;
    float d  = s_d_val * sc;
    if (d != 0.0f) {
        float dm = s_dm_val * m;
        int ql = nearest_int((s_x[grp * 16 + lane] + dm) / d);
        ql = max(0, min(3, ql));
        s_L[grp * 16 + lane] = (uint8_t)ql;
    }
    __syncthreads();

    // Pack qs: 4 values per byte (2 bits each), 128 elements at a time
    block_q2_K *blk = (block_q2_K*)(dst + (row * (n_per_row / QK_K) + sb) * sizeof(block_q2_K));

    // 128 elements per chunk → 32 bytes qs. 2 chunks total
    if (tid < 64) {
        int chunk = tid / 32;
        int l = tid % 32;
        int base_idx = chunk * 128;
        uint8_t byte = s_L[base_idx + l]
                     | (s_L[base_idx + l + 32] << 2)
                     | (s_L[base_idx + l + 64] << 4)
                     | (s_L[base_idx + l + 96] << 6);
        blk->qs[chunk * 32 + l] = byte;
    }
}
