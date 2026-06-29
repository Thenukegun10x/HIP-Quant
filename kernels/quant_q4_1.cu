#include <hip/hip_runtime.h>
#include "../hip_quant_types.h"
#include "../hip_quant_util.h"

// Q4_1: 32-element blocks, asymmetric 4-bit with min
// d = (max-min)/15, quant = (val-min)*id, clamped 0-15

extern "C" __global__
__launch_bounds__(32, 8)
void quantize_q4_1_kernel(
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

    __shared__ float s_min[32];
    __shared__ float s_max[32];

    float val = src[base];
    s_min[tid] = val;
    s_max[tid] = val;
    __syncthreads();

    for (int s = 16; s > 0; s >>= 1) {
        if (tid < s) {
            float v0 = s_min[tid];
            float v1 = s_min[tid + s];
            s_min[tid] = v0 < v1 ? v0 : v1;

            v0 = s_max[tid];
            v1 = s_max[tid + s];
            s_max[tid] = v0 > v1 ? v0 : v1;
        }
        __syncthreads();
    }

    float min_val = s_min[0];
    float max_val = s_max[0];
    float d = (max_val - min_val) / 15.0f;
    float id = d != 0 ? 1.0f / d : 0.0f;

    block_q4_1 *blk_out = (block_q4_1*)(dst + (row * (n_per_row / 32) + blk) * sizeof(block_q4_1));

    if (tid == 0) {
        blk_out->d = fp32_to_fp16(d);
        blk_out->m = fp32_to_fp16(min_val);
    }

    int q = (int)((val - min_val) * id + 0.5f);
    if (q < 0) q = 0;
    if (q > 15) q = 15;

    __shared__ uint8_t s_q[32];
    s_q[tid] = (uint8_t)q;
    __syncthreads();

    if (tid < 16) {
        blk_out->qs[tid] = s_q[tid] | (s_q[tid + 16] << 4);
    }
}
