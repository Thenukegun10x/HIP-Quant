#include <hip/hip_runtime.h>
#include "../hip_quant_types.h"
#include "../hip_quant_util.h"

// Q2_0: 64-element groups, 1 fp16 scale (max abs) + 2 bits per element.
// q = clamp(roundf(x*id) + 1, 0, 3): 00=-1, 01=0, 10=+1, 11=+2 (d).
// Matches quantize_row_q2_0_ref in ggml-quants.c.

extern "C" __global__
__launch_bounds__(QK2_0, 8)
void quantize_q2_0_kernel(
    const float * __restrict__ src,
    uint8_t * __restrict__ dst,
    const float * __restrict__ imatrix,
    int nrows,
    int n_per_row
) {
    int row = blockIdx.x;
    int blk = blockIdx.y;
    int tid = threadIdx.x;

    int base = row * n_per_row + blk * QK2_0 + tid;
    if (base >= (row + 1) * n_per_row) return;

    float v = fabsf(src[base]);
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, v, s));
    }
    __shared__ float s_max[2];
    if (tid % 32 == 0) s_max[tid / 32] = v;
    __syncthreads();

    float amax = fmaxf(s_max[0], s_max[1]);
    float id = amax > 0.0f ? 1.0f / amax : 0.0f;

    block_q2_0 *blk_out = (block_q2_0*)(dst + (row * (n_per_row / QK2_0) + blk) * sizeof(block_q2_0));

    if (tid == 0) {
        blk_out->d = fp32_to_fp16(amax);
    }

    if (tid < QK2_0 / 4) {
        const float * x = src + row * n_per_row + blk * QK2_0 + 4 * tid;
        uint8_t byte = 0;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int q = (int)roundf(x[k] * id) + 1;
            if (q < 0) q = 0;
            if (q > 3) q = 3;
            byte |= (uint8_t)(q << (2 * k));
        }
        blk_out->qs[tid] = byte;
    }
}
