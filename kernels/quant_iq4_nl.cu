#include "../hip_iquant_util.h"
#include "../hip_quant_types.h"
#include "../hip_quant_util.h"

#define QK4_NL 32
#define GROUP_MAX_EPS 1e-15f

// IQ4_NL: non-linear 4-bit, uses warp shuffle for min/max/sum reductions

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

    __shared__ float s_w[QK4_NL];
    __shared__ int s_L[QK4_NL];
    __shared__ float s_d_val;

    float xv = src[base];
    float x2 = xv * xv;

    // sigma2 = 2 * sum(x^2) via warp shuffle
    float sum2 = x2;
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        sum2 += __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, sum2, s);
    }
    float sigma2 = __shfl_sync(0xFFFFFFFFFFFFFFFFull, sum2, 0) * 2.0f / (float)QK4_NL;

    // weights
    if (imatrix != NULL) {
        float im_val = imatrix[base];
        s_w[tid] = im_val * sqrtf(sigma2 + x2);
    } else {
        s_w[tid] = x2;
    }

    // find amax + signed max via warp shuffle
    float av = fabsf(xv);
    float sv = xv;
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        float other_a = __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, av, s);
        float other_s = __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, sv, s);
        if (other_a > av) { av = other_a; sv = other_s; }
    }
    float amax = __shfl_sync(0xFFFFFFFFFFFFFFFFull, av, 0);
    float max_val = __shfl_sync(0xFFFFFFFFFFFFFFFFull, sv, 0);

    __threadfence_block();  // ensure s_w[tid] visible before thread 0 scans it

    if (amax < GROUP_MAX_EPS) {
        if (tid == 0) {
            block_iq4_nl *blk = (block_iq4_nl*)(dst + (row * (n_per_row / QK4_NL) + sb) * sizeof(block_iq4_nl));
            blk->d = 0;
            for (int i = 0; i < 16; ++i) blk->qs[i] = 0;
        }
        return;
    }

    // Sequential quantize + trial loop (same as original, only thread 0)
    if (tid == 0) {
        float sumqx = 0.0f, sumq2 = 0.0f;
        float id_val = -d_kvalues_iq4nl[0] / max_val;
        for (int j = 0; j < 32; ++j) {
            float al = id_val * src[row * n_per_row + sb * QK4_NL + j];
            int l = best_index_int8_dev(16, d_kvalues_iq4nl, al);
            s_L[j] = l;
            float q = (float)d_kvalues_iq4nl[l];
            float w = s_w[j];
            sumqx += w * q * src[row * n_per_row + sb * QK4_NL + j];
            sumq2 += w * q * q;
        }
        s_d_val = sumq2 > 0.0f ? sumqx / sumq2 : 0.0f;
        float best = s_d_val * sumqx;

        static const int ntry = 7;
        for (int itry = -ntry; itry <= ntry; ++itry) {
            id_val = ((float)itry + (float)d_kvalues_iq4nl[0]) / max_val;
            sumqx = 0.0f; sumq2 = 0.0f;
            for (int j = 0; j < 32; ++j) {
                float al = id_val * src[row * n_per_row + sb * QK4_NL + j];
                int l = best_index_int8_dev(16, d_kvalues_iq4nl, al);
                float q = (float)d_kvalues_iq4nl[l];
                float w = s_w[j];
                sumqx += w * q * src[row * n_per_row + sb * QK4_NL + j];
                sumq2 += w * q * q;
            }
            if (sumq2 > 0.0f && sumqx * sumqx > best * sumq2) {
                s_d_val = sumqx / sumq2;
                best = s_d_val * sumqx;
            }
        }

        float id = s_d_val != 0.0f ? 1.0f / s_d_val : 0.0f;
        for (int j = 0; j < 32; ++j) {
            s_L[j] = best_index_int8_dev(16, d_kvalues_iq4nl, id * src[row * n_per_row + sb * QK4_NL + j]);
        }

        block_iq4_nl *blk = (block_iq4_nl*)(dst + (row * (n_per_row / QK4_NL) + sb) * sizeof(block_iq4_nl));
        blk->d = fp32_to_fp16(s_d_val);
    }
    __syncthreads();

    if (tid < 16) {
        int lo = s_L[tid];
        int hi = s_L[tid + 16];
        ((block_iq4_nl*)(dst + (row * (n_per_row / QK4_NL) + sb) * sizeof(block_iq4_nl)))->qs[tid] =
            (uint8_t)(lo | (hi << 4));
    }
}
