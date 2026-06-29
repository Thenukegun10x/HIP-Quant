#include <hip/hip_runtime.h>
#include "../hip_quant_types.h"
#include "../hip_quant_util.h"

// Q8_0: 32-element blocks, 1 fp16 scale + 32 int8 quants

extern "C" __global__
__launch_bounds__(32, 8)
void quantize_q8_0_kernel(
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

    // Shared memory tree reduction (safe on any wavefront size)
    __shared__ float s_av[32];
    __shared__ float s_vals[32];

    float val = src[base];
    s_vals[tid] = val;
    s_av[tid] = fabsf(val);
    __syncthreads();

    for (int s = 16; s > 0; s >>= 1) {
        if (tid < s) {
            s_av[tid] = fmaxf(s_av[tid], s_av[tid + s]);
        }
        __syncthreads();
    }

    float amax = s_av[0];
    float d = amax / 127.0f;
    float id = d > 0 ? 1.0f / d : 0.0f;

    block_q8_0 *blk_out = (block_q8_0*)(dst + (row * (n_per_row / 32) + blk) * sizeof(block_q8_0));

    if (tid == 0) {
        blk_out->d = fp32_to_fp16(d);
    }

    // roundf: round to nearest, ties away from zero (matches CPU)
    int q = (int)roundf(val * id);
    if (q > 127) q = 127;
    if (q < -127) q = -127;
    blk_out->qs[tid] = (int8_t)q;
}
