#include "../hip_iquant_util.h"
#include "../hip_quant_types.h"
#include "../hip_quant_util.h"

#define QK_K 256
#define GROUP_MAX_EPS 1e-15f

extern "C" __global__
__launch_bounds__(256, 4)
void quantize_iq4_xs_kernel(
    const float * __restrict__ src,
    uint8_t * __restrict__ dst,
    const float * __restrict__ imatrix,
    int nrows,
    int n_per_row
) {
    int row = blockIdx.x;
    int sb = blockIdx.y;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int grp = tid >> 5;

    int base = row * n_per_row + sb * QK_K + grp * 32 + lane;
    if (base > (row + 1) * n_per_row - 1) return;

    __shared__ float s_x[QK_K];
    __shared__ float s_w[QK_K];
    __shared__ int s_L[QK_K];
    __shared__ float s_r[QK_K]; // reduction workspace
    __shared__ float s_scales[8];

    s_x[tid] = src[base];
    __syncthreads();

    // sigma2 = 2 * sum(x^2) / QK_K
    float xv = s_x[tid];
    float x2 = xv * xv;
    s_r[tid] = x2;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) s_r[tid] += s_r[tid + stride];
        __syncthreads();
    }
    float sigma2 = 2.0f * s_r[0] / (float)QK_K;
    __syncthreads();

    // weights
    if (imatrix != NULL) {
        float im_val = imatrix[base];
        s_w[tid] = im_val * sqrtf(sigma2 + x2);
    } else {
        s_w[tid] = x2;
    }
    __syncthreads();

    // Each warp processes its sub-block (grp = 0..7)
    // Find amax + signed max within sub-block
    int idx = grp * 32;
    s_r[idx + lane] = fabsf(xv);
    s_w[idx + lane] = xv; // reuse s_w for smax temporarily
    __syncthreads();

    for (int stride = 16; stride > 0; stride >>= 1) {
        if (lane < stride) {
            if (s_r[idx + lane + stride] > s_r[idx + lane]) {
                s_r[idx + lane] = s_r[idx + lane + stride];
                s_w[idx + lane] = s_w[idx + lane + stride];
            }
        }
        __syncthreads();
    }
    float amax = s_r[idx];
    float max_val = s_w[idx];
    __syncthreads();

    if (amax < GROUP_MAX_EPS) {
        if (lane == 0) s_scales[grp] = 0.0f;
        // s_w not needed for zero sub-blocks (skipped in trial loop)
        __syncthreads();
    } else {
        // Restore s_w to weights  
        if (imatrix != NULL) {
            float im_val = imatrix[base];
            s_w[tid] = im_val * sqrtf(sigma2 + x2);
        } else {
            s_w[tid] = x2;
        }
        __syncthreads();

        // Initial quantize — sequential within sub-block (match CPU order)
        if (lane == 0) {
            static const int ntry = 7;
            float d_val;
            float sumqx = 0.0f, sumq2 = 0.0f;
            float id_val = -d_kvalues_iq4nl[0] / max_val;  // = 127/max_val
            for (int j = 0; j < 32; ++j) {
                float al = id_val * s_x[idx + j];
                int l = best_index_int8_dev(16, d_kvalues_iq4nl, al);
                s_L[idx + j] = l;
                float q = (float)d_kvalues_iq4nl[l];
                float w = s_w[idx + j];
                sumqx += w * q * s_x[idx + j];
                sumq2 += w * q * q;
            }
            d_val = sumq2 > 0.0f ? sumqx / sumq2 : 0.0f;
            float best = d_val * sumqx;

            for (int itry = -ntry; itry <= ntry; ++itry) {
                id_val = ((float)itry + (float)d_kvalues_iq4nl[0]) / max_val;
                sumqx = 0.0f; sumq2 = 0.0f;
                for (int j = 0; j < 32; ++j) {
                    float al = id_val * s_x[idx + j];
                    int l = best_index_int8_dev(16, d_kvalues_iq4nl, al);
                    float q = (float)d_kvalues_iq4nl[l];
                    float w = s_w[idx + j];
                    sumqx += w * q * s_x[idx + j];
                    sumq2 += w * q * q;
                }
                if (sumq2 > 0.0f && sumqx * sumqx > best * sumq2) {
                    d_val = sumqx / sumq2;
                    best = d_val * sumqx;
                }
            }

            s_scales[grp] = d_val;
        }
        __syncthreads();
    }
    __syncthreads();

    // Phase 4: Thread 0 packs scales, then all threads requantize
    __shared__ int8_t s_scale_idx[8];
    __shared__ uint8_t s_scales_l[4];
    __shared__ uint16_t s_scales_h;
    __shared__ float s_d_val;

    if (tid == 0) {
        float max_scale = s_scales[0];
        float amax_scale = fabsf(s_scales[0]);
        for (int j = 1; j < 8; ++j) {
            float abs_s = fabsf(s_scales[j]);
            if (abs_s > amax_scale) {
                amax_scale = abs_s;
                max_scale = s_scales[j];
            }
        }

        block_iq4_xs *blk = (block_iq4_xs*)(dst + (row * (n_per_row / QK_K) + sb) * sizeof(block_iq4_xs));

        if (amax_scale < GROUP_MAX_EPS) {
            s_d_val = 0.0f;
            blk->d = 0;
            blk->scales_h = 0;
            for (int i = 0; i < 4; ++i) { s_scales_l[i] = 0; blk->scales_l[i] = 0; }
            for (int i = 0; i < 128; ++i) blk->qs[i] = 0;
            for (int j = 0; j < 8; ++j) s_scale_idx[j] = 0;
        } else {
            float d_val = -max_scale / 32.0f;
            s_d_val = d_val;
            blk->d = fp32_to_fp16(d_val);
            float id_val = d_val != 0.0f ? 1.0f / d_val : 0.0f;

            uint16_t scales_h = 0;
            for (int j = 0; j < 8; ++j) {
                int l = nearest_int(id_val * s_scales[j]);
                l = max(-32, min(31, l));
                s_scale_idx[j] = (int8_t)l;
                l += 32;
                uint8_t l_l = l & 0xF;
                uint8_t l_h = l >> 4;
                if (j % 2 == 0) s_scales_l[j/2] = l_l;
                else s_scales_l[j/2] |= (l_l << 4);
                scales_h |= (l_h << (2 * (j % 8)));
            }
            s_scales_h = scales_h;
            blk->scales_h = scales_h;
            for (int i = 0; i < 4; ++i) blk->scales_l[i] = s_scales_l[i];
        }
    }
    __syncthreads();

    // Requantize with final scales
    // Check both s_d_val and per-sub-block scale to avoid division by zero
    // for zero-value sub-blocks (s_scales[grp] == 0)
    float dl = s_d_val * (float)s_scale_idx[grp];
    if (dl != 0.0f) {
        float idl = 1.0f / dl;
        float al = idl * s_x[tid];
        s_L[tid] = best_index_int8_dev(16, d_kvalues_iq4nl, al);
    } else {
        s_L[tid] = 0;
    }
    __syncthreads();

    // Pack qs: per sub-block, 16 bytes per sub-block (32 elements: lo=first 16, hi=second 16)
    block_iq4_xs *blk = (block_iq4_xs*)(dst + (row * (n_per_row / QK_K) + sb) * sizeof(block_iq4_xs));
    if (tid < 128) {
        int ib = tid / 16;
        int j = tid % 16;
        int lo = s_L[ib * 32 + j];
        int hi = s_L[ib * 32 + j + 16];
        blk->qs[tid] = (uint8_t)(lo | (hi << 4));
    }
}
