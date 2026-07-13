#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include "../hip_quant_util.h"

namespace cg = cooperative_groups;

typedef float v8f __attribute__((ext_vector_type(8)));
typedef int   v2i __attribute__((ext_vector_type(2)));

// Test 1: Known input → verify that WMMA produces non-zero output.
// Feeds all-ones as FP8 E4M3 (0x38 = 1.0f) into both operands.
// Each 16x16 WMMA tile should compute C[i][j] = sum_k(1.0 * 1.0) = K.
// If the intrinsic is silently zeroing outputs, every result will be 0.0f.
extern "C" __global__
__launch_bounds__(32, 1)
void wmma_diag_known_values_kernel(
    const uint8_t * __restrict__ A_fp8,
    const uint8_t * __restrict__ B_fp8,
    float * __restrict__ C_out,
    float * __restrict__ max_abs,
    float * __restrict__ min_abs,
    int K
) {
    int tid = threadIdx.x;
    if (tid >= 32) return;

    int lane_wrapped = tid & 15;
    int lane_group   = tid >> 4;

    v8f acc = (v8f){};

    for (int k = 0; k < K; k += 16) {
        int a_packed[2] = {0, 0};
        int b_packed[2] = {0, 0};

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            ((uint8_t*)a_packed)[i] = A_fp8[k + lane_group * 8 + i];
            ((uint8_t*)b_packed)[i] = B_fp8[k + lane_group * 8 + i];
        }

        v2i a_vec = (v2i){ a_packed[0], a_packed[1] };
        v2i b_vec = (v2i){ b_packed[0], b_packed[1] };
        #if defined(__gfx1200__) || defined(__gfx1201__)
        acc = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a_vec, b_vec, acc);
        #endif
    }

    float local_max = 0.0f;
    float local_min = 1e30f;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int row = lane_group * 8 + i;
        int col = lane_wrapped;
        float val = acc[i];
        C_out[row * 16 + col] = val;
        if (val > local_max) local_max = val;
        if (val < local_min) local_min = val;
    }

    if (tid == 0) {
        *max_abs = local_max;
        *min_abs = local_min;
    }
}

// Test 2: Repeated WMMA accumulation to detect register decay or hangs.
// Accumulates N_ITERS times. If the intrinsic intermittently produces
// zeros or hangs on certain iterations, this catches it.
extern "C" __global__
__launch_bounds__(32, 1)
void wmma_diag_repeated_kernel(
    const uint8_t * __restrict__ A_fp8,
    const uint8_t * __restrict__ B_fp8,
    float * __restrict__ C_acc,
    int * __restrict__ ok_flag,
    int K,
    int n_iters
) {
    int tid = threadIdx.x;
    if (tid >= 32) { if (tid == 32) *ok_flag = 1; return; }

    int lane_wrapped = tid & 15;
    int lane_group   = tid >> 4;

    int local_ok = 1;

    for (int iter = 0; iter < n_iters && local_ok; iter++) {
        v8f acc = (v8f){};

        for (int k = 0; k < K; k += 16) {
            int a_packed[2] = {0, 0};
            int b_packed[2] = {0, 0};

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                ((uint8_t*)a_packed)[i] = A_fp8[k + lane_group * 8 + i];
                ((uint8_t*)b_packed)[i] = B_fp8[k + lane_group * 8 + i];
            }

            v2i a_vec = (v2i){ a_packed[0], a_packed[1] };
            v2i b_vec = (v2i){ b_packed[0], b_packed[1] };
            #if defined(__gfx1200__) || defined(__gfx1201__)
            acc = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a_vec, b_vec, acc);
            #endif
        }

        float val = acc[0];
        if (val == 0.0f || !isfinite(val)) {
            local_ok = 0;
            if (tid == 0) *ok_flag = -iter;
        } else if (iter == n_iters - 1 && tid == 0) {
            C_acc[0] = acc[0];
        }
    }

    __threadfence();
    if (tid == 0 && local_ok) {
        atomicMin(ok_flag, 0);  // 0 = no error seen
    }
}

// Test 3: WMMA after LDS load — tests that LDS→VGPR→WMMA store/load
// ordering is correct. Loads from global, copies through LDS, then WMMA.
extern "C" __global__
__launch_bounds__(64, 2)
void wmma_diag_lds_stage_kernel(
    const uint8_t * __restrict__ A_fp8,
    const uint8_t * __restrict__ B_fp8,
    float * __restrict__ C_out,
    int K
) {
    __shared__ uint8_t s_A[16 * 128];
    __shared__ uint8_t s_B[16 * 128];

    // Both waves participate in LDS staging and CTA barriers.  Only wave 0
    // owns the WMMA tile; returning wave 1 before a block barrier is invalid.
    const cg::thread_block cta = cg::this_thread_block();
    const cg::thread_group wmma_wave = cg::tiled_partition(cta, 32);
    const int cta_tid = cta.thread_rank();
    const bool is_wmma_wave = cta_tid < 32;
    const int tid = wmma_wave.thread_rank();

    int lane_wrapped = tid & 15;
    int lane_group   = tid >> 4;

    v8f acc = (v8f){};

    for (int k_start = 0; k_start < K; k_start += 128) {
        int k_stage = (K - k_start < 128) ? (K - k_start) : 128;

        for (int i = cta_tid; i < 16 * k_stage; i += cta.size()) {
            s_A[i] = A_fp8[k_start * 16 + i];
            s_B[i] = B_fp8[k_start * 16 + i];
        }
        cta.sync();

        if (is_wmma_wave) {
        wmma_wave.sync();
        for (int k = 0; k < k_stage; k += 16) {
            int a_packed[2] = {0, 0};
            int b_packed[2] = {0, 0};

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                ((uint8_t*)a_packed)[i] = s_A[(k + lane_group * 8 + i) * 16 + lane_wrapped];
                ((uint8_t*)b_packed)[i] = s_B[(k + lane_group * 8 + i) * 16 + lane_wrapped];
            }

            v2i a_vec = (v2i){ a_packed[0], a_packed[1] };
            v2i b_vec = (v2i){ b_packed[0], b_packed[1] };
            #if defined(__gfx1200__) || defined(__gfx1201__)
            acc = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a_vec, b_vec, acc);
            #endif
        }
        wmma_wave.sync();
        }
        cta.sync();
    }

    if (is_wmma_wave) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int row = lane_group * 8 + i;
            int col = lane_wrapped;
            C_out[row * 16 + col] = acc[i];
        }
    }
}

// Test 4: Multi-wave WMMA stress — 16 Wave32 blocks all doing WMMA.
// This keeps the stress test within the valid launch shape of the gfx12
// intrinsic.  Some Windows ROCm builds reject a 512-thread block containing
// sixteen WMMA waves even though each individual wave is valid.
extern "C" __global__
__launch_bounds__(32, 1)
void wmma_diag_multiwave_kernel(
    const uint8_t * __restrict__ A_fp8,
    const uint8_t * __restrict__ B_fp8,
    float * __restrict__ C_out,
    int * __restrict__ wave_results,
    int K
) {
    int wave = blockIdx.x;
    int tid  = threadIdx.x;
    if (wave >= 16 || tid >= 32) return;

    int lane_wrapped = tid & 15;
    int lane_group   = tid >> 4;

    v8f acc = (v8f){};

    for (int k = 0; k < K; k += 16) {
        int a_packed[2] = {0, 0};
        int b_packed[2] = {0, 0};

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            ((uint8_t*)a_packed)[i] = A_fp8[k + lane_group * 8 + i];
            ((uint8_t*)b_packed)[i] = B_fp8[k + lane_group * 8 + i];
        }

        v2i a_vec = (v2i){ a_packed[0], a_packed[1] };
        v2i b_vec = (v2i){ b_packed[0], b_packed[1] };
        #if defined(__gfx1200__) || defined(__gfx1201__)
        acc = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a_vec, b_vec, acc);
        #endif
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int row = lane_group * 8 + i;
        int col = lane_wrapped;
        C_out[wave * 16 * 16 + row * 16 + col] = acc[i];
    }

    __threadfence();
    if (tid == 0) {
        float max_val = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (acc[i] > max_val) max_val = acc[i];
        }
        wave_results[wave] = (max_val > 0.0f) ? 1 : 0;
    }
}
