#include <hip/hip_runtime.h>
#include "../hip_quant_types.h"
#include "../hip_quant_util.h"

// BF16: f32 -> bf16 with round-to-nearest-even, NaN made quiet.
// Matches ggml_compute_fp32_to_bf16 (Google Brain float conversion).

extern "C" __global__
__launch_bounds__(256, 8)
void quantize_bf16_kernel(
    const float * __restrict__ src,
    uint8_t * __restrict__ dst,
    const float * __restrict__ imatrix,
    int nrows,
    int n_per_row
) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx >= nrows * n_per_row) return;

    uint32_t u = __float_as_int(src[idx]);
    uint16_t bits;
    if ((u & 0x7fffffff) > 0x7f800000) {
        bits = (uint16_t)((u >> 16) | 64); // force quiet NaN
    } else {
        bits = (uint16_t)((u + (0x7fff + ((u >> 16) & 1))) >> 16);
    }
    ((uint16_t*)dst)[idx] = bits;
}
