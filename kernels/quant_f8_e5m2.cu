#include <hip/hip_runtime.h>
#include "../hip_quant_types.h"
#include "../hip_quant_util.h"

// F8_E5M2: 32-element blocks, 32 raw FP8 E5M2 values (no scale factor).
// Intended for gradients/backward-pass data where dynamic range matters more
// than mantissa precision.

extern "C" __global__
__launch_bounds__(32, 8)
void quantize_f8_e5m2_kernel(
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

    block_f8_e5m2 *blk_out = (block_f8_e5m2*)(dst + (row * (n_per_row / 32) + blk) * sizeof(block_f8_e5m2));
    blk_out->qs[tid] = fp32_to_fp8_e5m2(val);
}
