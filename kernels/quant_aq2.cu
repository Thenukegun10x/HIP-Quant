#include "../hip_quant_types.h"
#include "../hip_quant_util.h"

// AQ2: attention-calibrated 2-bit learned-codebook quantization.
//
// AQ2 deliberately shares HQ2's 72-byte wire layout: four FP16 centroids and
// 256 two-bit selectors (2.25 BPW).  The difference is the calibration
// contract.  `imatrix` is expected to contain attention-derived per-weight
// saliency (for example a Q/K score, V context, or O residual sensitivity
// map), rather than only a generic activation-energy imatrix.
//
// This kernel is Wave32-oriented for gfx12/RDNA4.  One 256-thread CTA owns a
// complete quantization block, so it contains eight Wave32 waves.  The block
// maximum is reduced inside each wave and the eight wave leaders finish the
// reduction.  Lloyd updates likewise aggregate inside each wave before doing
// shared-memory atomics, reducing the update traffic from 256 to 8 writers per
// centroid.

#define AQ2_EPS        1e-12f
#define AQ2_WAVE_SIZE  32

static __device__ __forceinline__ float aq2_wave_max(float value) {
    for (int delta = AQ2_WAVE_SIZE / 2; delta > 0; delta >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, value, delta));
    }
    return value;
}

static __device__ __forceinline__ float aq2_wave_sum(float value) {
    for (int delta = AQ2_WAVE_SIZE / 2; delta > 0; delta >>= 1) {
        value += __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, value, delta);
    }
    return value;
}

extern "C" __global__
__launch_bounds__(256, 4)
void quantize_aq2_kernel(
    const float * __restrict__ src,
    uint8_t * __restrict__ dst,
    const float * __restrict__ imatrix,
    int nrows,
    int n_per_row,
    int aq2_iterations
) {
    const int row = blockIdx.x;
    const int block = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & (AQ2_WAVE_SIZE - 1);
    const int wave = tid / AQ2_WAVE_SIZE;
    const int base = row * n_per_row + block * AQ2_K + tid;

    // Dimensions are validated on the host as exact multiples of AQ2_K, so
    // every thread participates in every barrier.
    const float xv = src[base];
    const float weight = (imatrix != NULL) ? fmaxf(imatrix[base], 0.0f) : 1.0f;

    __shared__ float s_wave_max[256 / AQ2_WAVE_SIZE];
    __shared__ float s_amax;
    __shared__ float s_levels[4];
    __shared__ float s_acc[4];
    __shared__ float s_count[4];
    __shared__ int s_assign[AQ2_K];

    float block_max = aq2_wave_max(fabsf(xv));
    if (lane == 0) {
        s_wave_max[wave] = block_max;
    }
    __syncthreads();

    if (tid == 0) {
        float amax = s_wave_max[0];
        #pragma unroll
        for (int w = 1; w < 256 / AQ2_WAVE_SIZE; ++w) {
            amax = fmaxf(amax, s_wave_max[w]);
        }
        s_amax = amax;
        if (amax >= AQ2_EPS) {
            // Keep HQ2's useful near-zero spacing as a strong initialization,
            // then let the attention-weighted Lloyd pass adapt it.
            s_levels[0] = -amax;
            s_levels[1] = -amax / 3.0f;
            s_levels[2] =  amax / 3.0f;
            s_levels[3] =  amax;
        } else {
            s_levels[0] = s_levels[1] = s_levels[2] = s_levels[3] = 0.0f;
        }
    }
    __syncthreads();

    block_aq2 *out = (block_aq2 *)(dst +
        (row * (n_per_row / AQ2_K) + block) * sizeof(block_aq2));
    if (s_amax < AQ2_EPS) {
        if (tid == 0) {
            for (int c = 0; c < 4; ++c) out->levels[c] = 0;
            for (int i = 0; i < AQ2_K / 4; ++i) out->qs[i] = 0;
        }
        return;
    }

    for (int iteration = 0; iteration < aq2_iterations; ++iteration) {
        float best_error = 1e30f;
        int best_code = 0;
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const float delta = xv - s_levels[c];
            const float error = delta * delta;
            if (error < best_error) {
                best_error = error;
                best_code = c;
            }
        }

        if (tid < 4) {
            s_acc[tid] = 0.0f;
            s_count[tid] = 0.0f;
        }
        __syncthreads();

        // All lanes execute the same shuffle sequence.  Only one lane per
        // Wave32 performs each shared atomic update.
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const float selected = (best_code == c) ? weight : 0.0f;
            const float weighted_value = (best_code == c) ? weight * xv : 0.0f;
            const float wave_count = aq2_wave_sum(selected);
            const float wave_value = aq2_wave_sum(weighted_value);
            if (lane == 0) {
                atomicAdd(&s_count[c], wave_count);
                atomicAdd(&s_acc[c], wave_value);
            }
        }
        __syncthreads();

        if (tid < 4 && s_count[tid] > 0.0f) {
            s_levels[tid] = s_acc[tid] / s_count[tid];
        }
        __syncthreads();
    }

    float best_error = 1e30f;
    int best_code = 0;
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        const float delta = xv - s_levels[c];
        const float error = delta * delta;
        if (error < best_error) {
            best_error = error;
            best_code = c;
        }
    }
    s_assign[tid] = best_code;
    __syncthreads();

    if (tid == 0) {
        for (int c = 0; c < 4; ++c) {
            out->levels[c] = fp32_to_fp16(s_levels[c]);
        }
    }
    __syncthreads();

    if (tid < AQ2_K / 4) {
        const int i = tid * 4;
        const int b0 = s_assign[i + 0] & 3;
        const int b1 = s_assign[i + 1] & 3;
        const int b2 = s_assign[i + 2] & 3;
        const int b3 = s_assign[i + 3] & 3;
        out->qs[tid] = (uint8_t)(b0 | (b1 << 2) | (b2 << 4) | (b3 << 6));
    }
}
