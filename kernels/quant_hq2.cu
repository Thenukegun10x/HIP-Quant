#include "../hip_quant_types.h"
#include "../hip_quant_util.h"

// HQ2: 256-element block, 4-level TurboQuant-style learned codebook.
// One CTA of 256 threads owns the whole block (8 warps, Wave32-friendly on
// gfx12). Each thread holds one weight; a weighted in-kernel k-means fits the
// 4 non-uniform codebook levels, then the 2-bit assignments are packed.
//
// Importance weighting: when an importance matrix is supplied, each weight's
// contribution to the level fit is scaled by w_i (clamped >= 0). This steers
// the codebook toward the weights that matter most for the layer's output,
// which is the same idea behind TurboQuant's per-block codebook selection.

#define HQ2_EPS   1e-12f

extern "C" __global__
__launch_bounds__(256, 4)
void quantize_hq2_kernel(
    const float * __restrict__ src,
    uint8_t * __restrict__ dst,
    const float * __restrict__ imatrix,
    int nrows,
    int n_per_row,
    int hq2_iterations
) {
    int row = blockIdx.x;
    int sb  = blockIdx.y;
    int tid = threadIdx.x;

    int base = row * n_per_row + sb * HQ2_K + tid;
    if (base >= (row + 1) * n_per_row) return;

    __shared__ float s_lev[4];
    __shared__ float s_acc[4];
    __shared__ float s_cnt[4];
    __shared__ float s_red[HQ2_K];
    __shared__ int   s_assign[HQ2_K];

    float xv = src[base];
    const float weight = (imatrix != NULL) ? fmaxf(imatrix[base], 0.0f) : 1.0f;
    __syncthreads();

    // amax over the block (tree reduction in shared memory)
    s_red[tid] = fabsf(xv);
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) s_red[tid] = fmaxf(s_red[tid], s_red[tid + stride]);
        __syncthreads();
    }
    float amax = s_red[0];

    block_hq2 *blk = (block_hq2 *)(dst + (row * (n_per_row / HQ2_K) + sb) * sizeof(block_hq2));

    if (amax < HQ2_EPS) {
        if (tid == 0) {
            blk->levels[0] = blk->levels[1] = blk->levels[2] = blk->levels[3] = 0;
            for (int i = 0; i < 64; ++i) blk->qs[i] = 0;
        }
        return;
    }

    // Non-uniform, fine-near-zero initialization (TurboQuant-style):
    // symmetric 4 levels with finer spacing around zero.
    if (tid == 0) {
        s_lev[0] = -amax;
        s_lev[1] = -amax / 3.0f;
        s_lev[2] =  amax / 3.0f;
        s_lev[3] =  amax;
    }
    __syncthreads();

    // Weighted Lloyd's iterations (k-means). Assignment uses squared error;
    // the update step is importance-weighted so salient weights dominate the
    // fitted level. Shared-memory float atomics keep the reduction simple.
    for (int it = 0; it < hq2_iterations; ++it) {
        float best = 1e30f;
        int bc = 0;
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            float d = xv - s_lev[c];
            float e = d * d;
            if (e < best) { best = e; bc = c; }
        }

        if (tid < 4) { s_acc[tid] = 0.0f; s_cnt[tid] = 0.0f; }
        __syncthreads();
        atomicAdd(&s_acc[bc], weight * xv);
        atomicAdd(&s_cnt[bc], weight);
        __syncthreads();

        if (tid < 4) {
            if (s_cnt[tid] > 0.0f) s_lev[tid] = s_acc[tid] / s_cnt[tid];
        }
        __syncthreads();
    }

    // Final assignment -> store index in shared memory.
    float best = 1e30f;
    int bc = 0;
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        float d = xv - s_lev[c];
        float e = d * d;
        if (e < best) { best = e; bc = c; }
    }
    s_assign[tid] = bc;
    __syncthreads();

    if (tid == 0) {
        for (int c = 0; c < 4; ++c) blk->levels[c] = fp32_to_fp16(s_lev[c]);
    }
    __syncthreads();

    // Pack 4 indices per byte (conflict-free: one assembler thread per byte).
    if (tid < 64) {
        int b0 = s_assign[4 * tid + 0] & 3;
        int b1 = s_assign[4 * tid + 1] & 3;
        int b2 = s_assign[4 * tid + 2] & 3;
        int b3 = s_assign[4 * tid + 3] & 3;
        blk->qs[tid] = (uint8_t)(b0 | (b1 << 2) | (b2 << 4) | (b3 << 6));
    }
}
