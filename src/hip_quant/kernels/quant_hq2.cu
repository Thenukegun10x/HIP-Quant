#include "../hip_quant_types.h"
#include "../hip_quant_util.h"

#define HQ2_EPS   1e-12f

extern "C" __global__
__launch_bounds__(256, 4)
void quantize_hq2_kernel(
    const float * __restrict__ src,
    uint8_t * __restrict__ dst,
    const float * __restrict__ imatrix,
    int nrows,
    int n_per_row,
    int hq2_iterations
) {
    int row = blockIdx.x;
    int sb = blockIdx.y;
    int tid = threadIdx.x;
    int base = row * n_per_row + sb * HQ2_K + tid;

    __shared__ float s_x[HQ2_K];
    __shared__ float s_w[HQ2_K];
    __shared__ float s_lev[4];
    __shared__ float s_acc[4];
    __shared__ float s_cnt[4];
    __shared__ float s_red[HQ2_K];
    __shared__ int s_assign[HQ2_K];

    float xv = src[base];
    s_x[tid] = xv;
    s_w[tid] = (imatrix != NULL) ? fmaxf(imatrix[base], 0.0f) : 1.0f;
    __syncthreads();

    s_red[tid] = fabsf(xv);
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) s_red[tid] = fmaxf(s_red[tid], s_red[tid + stride]);
        __syncthreads();
    }
    float amax = s_red[0];

    block_hq2 *blk = (block_hq2 *)(dst +
        (row * (n_per_row / HQ2_K) + sb) * sizeof(block_hq2));
    if (amax < HQ2_EPS) {
        if (tid == 0) {
            for (int i = 0; i < 4; ++i) blk->levels[i] = 0;
            for (int i = 0; i < 64; ++i) blk->qs[i] = 0;
        }
        return;
    }

    if (tid == 0) {
        s_lev[0] = -amax;
        s_lev[1] = -amax / 3.0f;
        s_lev[2] = amax / 3.0f;
        s_lev[3] = amax;
    }
    __syncthreads();

    for (int it = 0; it < hq2_iterations; ++it) {
        float best = 1e30f;
        int bc = 0;
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            float d = xv - s_lev[c];
            float e = d * d;
            if (e < best) { best = e; bc = c; }
        }
        if (tid < 4) { s_acc[tid] = 0.0f; s_cnt[tid] = 0.0f; }
        __syncthreads();
        atomicAdd(&s_acc[bc], s_w[tid] * xv);
        atomicAdd(&s_cnt[bc], s_w[tid]);
        __syncthreads();
        if (tid < 4 && s_cnt[tid] > 0.0f) {
            s_lev[tid] = s_acc[tid] / s_cnt[tid];
        }
        __syncthreads();
    }

    float best = 1e30f;
    int bc = 0;
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        float d = xv - s_lev[c];
        float e = d * d;
        if (e < best) { best = e; bc = c; }
    }
    s_assign[tid] = bc;
    __syncthreads();

    if (tid == 0) {
        for (int c = 0; c < 4; ++c) blk->levels[c] = fp32_to_fp16(s_lev[c]);
    }
    __syncthreads();

    if (tid < 64) {
        int b0 = s_assign[4 * tid + 0] & 3;
        int b1 = s_assign[4 * tid + 1] & 3;
        int b2 = s_assign[4 * tid + 2] & 3;
        int b3 = s_assign[4 * tid + 3] & 3;
        blk->qs[tid] = (uint8_t)(b0 | (b1 << 2) | (b2 << 4) | (b3 << 6));
    }
}
