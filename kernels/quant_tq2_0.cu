#include <hip/hip_runtime.h>
#include "../hip_quant_types.h"
#include "../hip_quant_util.h"

extern "C" __global__
__launch_bounds__(256, 1)
void quantize_tq2_0_kernel(
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

    __shared__ float s_vals[QK_K];

    float val = src[base];
    s_vals[tid] = fabsf(val);
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_vals[tid + s] > s_vals[tid]) {
                s_vals[tid] = s_vals[tid + s];
            }
        }
        __syncthreads();
    }

    float amax = s_vals[0];
    float d = amax;
    float id = d != 0.0f ? 1.0f / d : 0.0f;

    block_tq2_0 *blk_out = (block_tq2_0*)(dst + (row * (n_per_row / QK_K) + blk) * sizeof(block_tq2_0));

    if (tid == 0) {
        blk_out->d = fp32_to_fp16(d);
    }

    float scaled = val * id;
    int xi = nearest_int(scaled) + 1; // -1, 0, 1 -> 0, 1, 2
    if (xi < 0) xi = 0;
    if (xi > 2) xi = 2;

    __shared__ uint8_t s_q[QK_K];
    s_q[tid] = (uint8_t)(xi & 3);
    __syncthreads();

    // Pack 4 elements per byte
    if (tid < 64) {
        int chunk = tid / 32;
        int m = tid % 32;
        int src_idx = chunk * 128 + m;

        uint8_t q = 0;
        q += s_q[src_idx] << 0;
        q += s_q[src_idx + 32] << 2;
        q += s_q[src_idx + 64] << 4;
        q += s_q[src_idx + 96] << 6;

        blk_out->qs[tid] = q;
    }
}
