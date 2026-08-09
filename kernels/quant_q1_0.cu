#include <hip/hip_runtime.h>
#include "../hip_quant_types.h"
#include "../hip_quant_util.h"

// Q1_0: 128-element groups, 1 fp16 scale (mean abs) + 1 sign bit per element.
// Bit j is set when x[j] >= 0. Matches quantize_row_q1_0_ref in ggml-quants.c.

extern "C" __global__
__launch_bounds__(QK1_0, 8)
void quantize_q1_0_kernel(
    const float * __restrict__ src,
    uint8_t * __restrict__ dst,
    const float * __restrict__ imatrix,
    int nrows,
    int n_per_row
) {
    int row = blockIdx.x;
    int blk = blockIdx.y;
    int tid = threadIdx.x;

    int base = row * n_per_row + blk * QK1_0 + tid;
    if (base >= (row + 1) * n_per_row) return;

    block_q1_0 *blk_out = (block_q1_0*)(dst + (row * (n_per_row / QK1_0) + blk) * sizeof(block_q1_0));

    if (tid == 0) {
        // llama sums |x| sequentially in element order; a tree reduction
        // changes the F32 rounding and breaks byte-exactness of d.
        const float * x = src + row * n_per_row + blk * QK1_0;
        float sum_abs = 0.0f;
        #pragma unroll
        for (int j = 0; j < QK1_0; ++j) sum_abs += fabsf(x[j]);
        blk_out->d = fp32_to_fp16(sum_abs / QK1_0);
    }

    if (tid < QK1_0 / 8) {
        const float * x = src + row * n_per_row + blk * QK1_0 + 8 * tid;
        uint8_t byte = 0;
        #pragma unroll
        for (int bit = 0; bit < 8; ++bit) {
            if (x[bit] >= 0.0f) byte |= (uint8_t)(1 << bit);
        }
        blk_out->qs[tid] = byte;
    }
}
