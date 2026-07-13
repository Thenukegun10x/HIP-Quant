#include <hip/hip_runtime.h>
#include "../hip_quant_types.h"
#include "../hip_quant_util.h"

// Q8_1: 32-element blocks, symmetric 8-bit with sum
// Uses warp shuffle reduction (wave32-safe on gfx12/RDNA4)

extern "C" __global__
__launch_bounds__(32, 8)
void quantize_q8_1_kernel(
    const float * __restrict__ src,
    uint8_t * __restrict__ dst,
    const float * __restrict__ imatrix,
    int nrows,
    int n_per_row
) {
    int row = blockIdx.x;
    int blk = blockIdx.y;
    int tid = threadIdx.x;

    int base = row * n_per_row + blk * 32 + tid;
    if (base >= (row + 1) * n_per_row) return;

    float val = src[base];
    float v = fabsf(val);
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, v, s));
    }
    float amax = __shfl_sync(0xFFFFFFFFFFFFFFFFull, v, 0);

    float d = amax / 127.0f;
    float id = d > 0 ? 1.0f / d : 0.0f;

    int q = (int)roundf(val * id);
    if (q > 127) q = 127;
    if (q < -127) q = -127;

    block_q8_1 *blk_out = (block_q8_1*)(dst + (row * (n_per_row / 32) + blk) * sizeof(block_q8_1));

    if (tid == 0) {
        blk_out->d = fp32_to_fp16(d);
    }

    blk_out->qs[tid] = (int8_t)q;

    int sum = q;
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, sum, s);
    }

    if (tid == 0) {
        blk_out->s = fp32_to_fp16((float)__shfl_sync(0xFFFFFFFFFFFFFFFFull, sum, 0) * d);
    }
}
