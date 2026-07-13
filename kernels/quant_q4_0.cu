#include <hip/hip_runtime.h>
#include "../hip_quant_types.h"
#include "../hip_quant_util.h"

// Q4_0: 32-element blocks, symmetric 4-bit
// Uses warp shuffle reduction (wave32-safe on gfx12/RDNA4)

extern "C" __global__
__launch_bounds__(32, 8)
void quantize_q4_0_kernel(
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

    float v = val;
    float va = fabsf(val);
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, v, s);
        float other_a = __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, va, s);
        if (other_a > va) { v = other; va = other_a; }
    }

    float max_val = __shfl_sync(0xFFFFFFFFFFFFFFFFull, v, 0);
    float d = max_val / -8.0f;
    float id = d != 0 ? 1.0f / d : 0.0f;

    block_q4_0 *blk_out = (block_q4_0*)(dst + (row * (n_per_row / 32) + blk) * sizeof(block_q4_0));

    if (tid == 0) {
        blk_out->d = fp32_to_fp16(d);
    }

    int q = (int)(val * id + 8.5f);
    if (q < 0) q = 0;
    if (q > 15) q = 15;

    __shared__ uint8_t s_q[32];
    s_q[tid] = (uint8_t)q;
    __syncthreads();

    if (tid < 16) {
        blk_out->qs[tid] = s_q[tid] | (s_q[tid + 16] << 4);
    }
}
