#include <hip/hip_runtime.h>
#include "../hip_quant_types.h"
#include "../hip_quant_util.h"

// Q8_1: 32-element blocks, symmetric 8-bit with sum
// d = amax/127, qs = round(val*id), s = d * sum(qs)

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

    __shared__ float s_av[32];
    __shared__ int s_sum[32];

    float val = src[base];
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

    int q = (int)roundf(val * id);
    if (q > 127) q = 127;
    if (q < -127) q = -127;

    // Wait for d to be computed before writing output
    // But we need d for sum calculation, and d doesn't depend on q
    // The sum is of q values, so we can compute it in parallel

    block_q8_1 *blk_out = (block_q8_1*)(dst + (row * (n_per_row / 32) + blk) * sizeof(block_q8_1));

    if (tid == 0) {
        blk_out->d = fp32_to_fp16(d);
    }

    blk_out->qs[tid] = (int8_t)q;

    s_sum[tid] = q;
    __syncthreads();

    for (int s = 16; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        blk_out->s = fp32_to_fp16((float)s_sum[0] * d);
    }
}
