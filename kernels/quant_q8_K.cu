#include <hip/hip_runtime.h>
#include "../hip_quant_types.h"
#include "../hip_quant_util.h"

// Q8_K: 256-element blocks, float scale d = -max/127, 8-bit quants
// v = nearest_int(-127/max * x), plus int16 bsums of quants in
// 16-element groups. Matches quantize_row_q8_K_ref in ggml-quants.c.

extern "C" __global__
__launch_bounds__(QK_K, 4)
void quantize_q8_K_kernel(
    const float * __restrict__ src,
    uint8_t * __restrict__ dst,
    const float * __restrict__ imatrix,
    int nrows,
    int n_per_row
) {
    int row = blockIdx.x;
    int blk = blockIdx.y;
    int tid = threadIdx.x;

    int base = row * n_per_row + blk * QK_K + tid;
    if (base >= (row + 1) * n_per_row) return;

    const float * x = src + row * n_per_row + blk * QK_K;
    block_q8_K *blk_out = (block_q8_K*)(dst + (row * (n_per_row / QK_K) + blk) * sizeof(block_q8_K));

    __shared__ float s_amax[QK_K];
    s_amax[tid] = fabsf(x[tid]);
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_amax[tid + s] > s_amax[tid]) {
                s_amax[tid] = s_amax[tid + s];
            }
        }
        __syncthreads();
    }
    float amax = s_amax[0];

    if (amax == 0.0f) {
        if (tid == 0) {
            blk_out->d = 0.0f;
            for (int j = 0; j < QK_K; ++j) blk_out->qs[j] = 0;
            for (int j = 0; j < QK_K / 16; ++j) blk_out->bsums[j] = 0;
        }
        return;
    }

    float iscale = -127.0f / amax;
    int v = nearest_int(iscale * x[tid]);
    if (v > 127) v = 127;
    blk_out->qs[tid] = (int8_t)v;
    __syncthreads();

    if (tid < QK_K / 16) {
        int sum = 0;
        #pragma unroll
        for (int k = 0; k < 16; ++k) sum += blk_out->qs[tid * 16 + k];
        blk_out->bsums[tid] = (int16_t)sum;
        if (tid == 0) blk_out->d = 1.0f / iscale;
    }
}
