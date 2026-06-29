#include <hip/hip_runtime.h>
#include "../hip_quant_types.h"
#include "../hip_quant_util.h"

extern "C" __global__
__launch_bounds__(256, 1)
void quantize_tq1_0_kernel(
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

    // Warp-level parallel reduction for absolute maximum
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

    block_tq1_0 *blk_out = (block_tq1_0*)(dst + (row * (n_per_row / QK_K) + blk) * sizeof(block_tq1_0));

    if (tid == 0) {
        blk_out->d = fp32_to_fp16(d);
    }

    float scaled = val * id;
    int xi = nearest_int(scaled) + 1; // -1, 0, 1 -> 0, 1, 2
    if (xi < 0) xi = 0;
    if (xi > 2) xi = 2;

    __shared__ uint8_t s_q[QK_K];
    s_q[tid] = (uint8_t)xi;
    __syncthreads();

    // Pack into qs (first 32 bytes) -> 5 elements per byte
    if (tid < 32) {
        int m = tid;
        uint8_t q = 0;
        for (int n = 0; n < 5; ++n) {
            q *= 3;
            q += s_q[m + n*32];
        }
        q = ((uint16_t)q * 256 + 242) / 243;
        blk_out->qs[tid] = q;
    }
    // Pack into qs (next 16 bytes) -> 5 elements per byte
    else if (tid >= 32 && tid < 48) {
        int m = tid - 32;
        uint8_t q = 0;
        for (int n = 0; n < 5; ++n) {
            q *= 3;
            q += s_q[160 + m + n*16];
        }
        q = ((uint16_t)q * 256 + 242) / 243;
        blk_out->qs[tid] = q;
    }
    // Pack into qh (next 4 bytes) -> 4 elements per byte
    else if (tid >= 48 && tid < 52) {
        int j = tid - 48;
        uint8_t q = 0;
        for (int m = 0; m < 4; ++m) {
            q *= 3;
            q += s_q[240 + j + m*4];
        }
        q *= 3; // shift the first value to the most significant trit
        q = ((uint16_t)q * 256 + 242) / 243;
        blk_out->qh[j] = q;
    }
}
