#include <hip/hip_runtime.h>
#include "../hip_quant_types.h"
#include "../hip_quant_util.h"

// Q5_0: 32-element blocks, symmetric 5-bit
// Uses warp shuffle reduction (wave32-safe on gfx12/RDNA4)

extern "C" __global__
__launch_bounds__(32, 8)
void quantize_q5_0_kernel(
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
    float d = max_val / -16.0f;
    float id = d != 0 ? 1.0f / d : 0.0f;

    block_q5_0 *blk_out = (block_q5_0*)(dst + (row * (n_per_row / 32) + blk) * sizeof(block_q5_0));

    if (tid == 0) {
        blk_out->d = fp32_to_fp16(d);
    }

    int q = (int)(val * id + 16.5f);
    if (q < 0) q = 0;
    if (q > 31) q = 31;

    __shared__ uint8_t s_q[32];
    s_q[tid] = (uint8_t)q;
    __syncthreads();

    if (tid < 16) {
        uint8_t xi0 = s_q[tid];
        uint8_t xi1 = s_q[tid + 16];
        blk_out->qs[tid] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
    }

    // QH stores the fifth bit for eight consecutive elements per byte.
    // The previous cross-lane OR reduction produced an all-zero mask on
    // Wave32 gfx12, silently turning every Q5 value into its low 4 bits.
    // Pack the four bytes directly from the synchronised shared quants.
    if (tid < 4) {
        uint8_t high_bits = 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            high_bits |= ((s_q[tid * 8 + j] >> 4) & 1u) << j;
        }
        blk_out->qh[tid] = high_bits;
    }
}
