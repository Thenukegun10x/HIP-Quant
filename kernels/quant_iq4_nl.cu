#include "../hip_iquant_util.h"
#include "../hip_quant_types.h"
#include "../hip_quant_util.h"

#define QK4_NL 32
#define GROUP_MAX_EPS 1e-15f

extern "C" __global__
__launch_bounds__(QK4_NL, 32)
void quantize_iq4_nl_kernel(
    const float * __restrict__ src,
    uint8_t * __restrict__ dst,
    const float * __restrict__ imatrix,
    int nrows,
    int n_per_row
) {
    int row = blockIdx.x;
    int sb = blockIdx.y;
    int tid = threadIdx.x;
    int base = row * n_per_row + sb * QK4_NL + tid;

    __shared__ float s_x[QK4_NL];
    __shared__ float s_w[QK4_NL];
    __shared__ int s_L[QK4_NL];
    __shared__ float s_reduce[QK4_NL];

    s_x[tid] = src[base];
    __syncthreads();

    // sigma2 = 2 * sum(x^2) / QK4_NL
    float xv = s_x[tid];
    float x2 = xv * xv;
    s_reduce[tid] = x2;
    __syncthreads();

    for (int stride = 16; stride > 0; stride >>= 1) {
        if (tid < stride) s_reduce[tid] += s_reduce[tid + stride];
        __syncthreads();
    }
    float sigma2 = 2.0f * s_reduce[0] / (float)QK4_NL;
    __syncthreads();

    // weights
    if (imatrix != NULL) {
        float im_val = imatrix[base];
        s_w[tid] = im_val * sqrtf(sigma2 + x2);
    } else {
        s_w[tid] = x2;
    }
    __syncthreads();

    // find amax + signed max (carry-along reduction)
    __shared__ float s_amax[QK4_NL];
    __shared__ float s_smax[QK4_NL];
    s_amax[tid] = fabsf(xv);
    s_smax[tid] = xv;
    __syncthreads();

    for (int stride = 16; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_amax[tid + stride] > s_amax[tid]) {
                s_amax[tid] = s_amax[tid + stride];
                s_smax[tid] = s_smax[tid + stride];
            }
        }
        __syncthreads();
    }
    float amax = s_amax[0];
    float max_val = s_smax[0];
    __syncthreads();

    if (amax < GROUP_MAX_EPS) {
        if (tid == 0) {
            block_iq4_nl *blk = (block_iq4_nl*)(dst + (row * (n_per_row / QK4_NL) + sb) * sizeof(block_iq4_nl));
            blk->d = 0;
            for (int i = 0; i < 16; ++i) blk->qs[i] = 0;
        }
        return;
    }

    // sequential quantize + trial loop (match CPU order)
    static const int ntry = 7;
    __shared__ float s_d_val;
    if (tid == 0) {
        float sumqx = 0.0f, sumq2 = 0.0f;
        float id_val = -d_kvalues_iq4nl[0] / max_val;  // = 127/max_val
        for (int j = 0; j < 32; ++j) {
            float al = id_val * s_x[j];
            int l = best_index_int8_dev(16, d_kvalues_iq4nl, al);
            s_L[j] = l;
            float q = (float)d_kvalues_iq4nl[l];
            float w = s_w[j];
            sumqx += w * q * s_x[j];
            sumq2 += w * q * q;
        }
        s_d_val = sumq2 > 0.0f ? sumqx / sumq2 : 0.0f;
        float best = s_d_val * sumqx;

        for (int itry = -ntry; itry <= ntry; ++itry) {
            id_val = ((float)itry + (float)d_kvalues_iq4nl[0]) / max_val;
            sumqx = 0.0f; sumq2 = 0.0f;
            for (int j = 0; j < 32; ++j) {
                float al = id_val * s_x[j];
                int l = best_index_int8_dev(16, d_kvalues_iq4nl, al);
                float q = (float)d_kvalues_iq4nl[l];
                float w = s_w[j];
                sumqx += w * q * s_x[j];
                sumq2 += w * q * q;
            }
            if (sumq2 > 0.0f && sumqx * sumqx > best * sumq2) {
                s_d_val = sumqx / sumq2;
                best = s_d_val * sumqx;
            }
        }

        // final requantize
        float id = s_d_val != 0.0f ? 1.0f / s_d_val : 0.0f;
        for (int j = 0; j < 32; ++j) {
            s_L[j] = best_index_int8_dev(16, d_kvalues_iq4nl, id * s_x[j]);
        }

        // store scale
        block_iq4_nl *blk = (block_iq4_nl*)(dst + (row * (n_per_row / QK4_NL) + sb) * sizeof(block_iq4_nl));
        blk->d = fp32_to_fp16(s_d_val);
    }
    __syncthreads();

    // pack qs: pairs of quants
    if (tid < 16) {
        int lo = s_L[tid];
        int hi = s_L[tid + 16];
        ((block_iq4_nl*)(dst + (row * (n_per_row / QK4_NL) + sb) * sizeof(block_iq4_nl)))->qs[tid] =
            (uint8_t)(lo | (hi << 4));
    }
}
