#include <hip/hip_runtime.h>
#include "../hip_quant_types.h"
#include "../hip_quant_util.h"

// Q5_0: 32-element blocks, symmetric 5-bit
// d = max/-16, values range [-16, 15], stored as unsigned nibble + 16 bias
// 5th bit packed in qh[4] bitmask via shared memory reduction

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

    __shared__ float s_vals[32];

    float val = src[base];
    s_vals[tid] = val;
    __syncthreads();

    for (int s = 16; s > 0; s >>= 1) {
        if (tid < s) {
            float a0 = fabsf(s_vals[tid]);
            float a1 = fabsf(s_vals[tid + s]);
            if (a1 > a0) {
                s_vals[tid] = s_vals[tid + s];
            }
        }
        __syncthreads();
    }

    float max_val = s_vals[0];
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

    // Tree reduction for qh bitmask
    __shared__ uint32_t s_qh[32];
    s_qh[tid] = 0;
    if (tid < 16) {
        if (s_q[tid] & 0x10) s_qh[tid] |= (1u << (tid + 0));
        if (s_q[tid + 16] & 0x10) s_qh[tid] |= (1u << (tid + 16));
    }
    __syncthreads();

    for (int s = 16; s > 0; s >>= 1) {
        if (tid < s) {
            s_qh[tid] |= s_qh[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *(uint32_t*)blk_out->qh = s_qh[0];
    }
}
