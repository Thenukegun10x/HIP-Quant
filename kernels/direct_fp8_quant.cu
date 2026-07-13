#include <hip/hip_runtime.h>
#include "../hip_quant_types.h"
#include "../hip_quant_util.h"

// ============================================================
// Direct FP8→Quant fused kernels
//
// Eliminates the intermediate F32 buffer from the FP8 input
// quantization path. Each kernel reads FP8 bytes, converts to
// F32 inline, and proceeds with the standard quantization math.
//
// Two variants per GGML type: E4M3 (type 36) and E5M2 (type 37).
// ============================================================

#define QK4_0 32
#define QK4_1 32
#define QK5_0 32
#define QK5_1 32
#define QK8_0 32
#define QK8_1 32

__device__ inline void pack_q5_high_bits(uint8_t *qh, const uint8_t *q, int tid) {
    if (tid >= 4) return;
    uint8_t high_bits = 0;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        high_bits |= ((q[tid * 8 + j] >> 4) & 1u) << j;
    }
    qh[tid] = high_bits;
}

// ── Q4_0 from FP8 ──────────────────────────────────────────

extern "C" __global__
__launch_bounds__(32, 8)
void quantize_q4_0_from_fp8_e4m3_kernel(
    const uint8_t * __restrict__ src_fp8,
    uint8_t * __restrict__ dst,
    const float * __restrict__ imatrix,
    int nrows,
    int n_per_row
) {
    int row = blockIdx.x;
    int blk = blockIdx.y;
    int tid = threadIdx.x;
    int base = row * n_per_row + blk * QK4_0 + tid;
    if (base >= (row + 1) * n_per_row) return;

    float val = fp8_e4m3_to_fp32(src_fp8[base]);

    float v = val;
    float va = fabsf(val);
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, v, s);
        float other_a = __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, va, s);
        if (other_a > va) { v = other; va = other_a; }
    }
    float max_val = __shfl_sync(0xFFFFFFFFFFFFFFFFull, v, 0);
    float d = max_val / -8.0f;
    float id = d != 0 ? 1.0f / d : 0.0f;

    block_q4_0 *blk_out = (block_q4_0*)(dst + (row * (n_per_row / 32) + blk) * sizeof(block_q4_0));
    if (tid == 0) blk_out->d = fp32_to_fp16(d);

    int q = (int)(val * id + 8.5f);
    if (q < 0) q = 0;
    if (q > 15) q = 15;

    __shared__ uint8_t s_q[32];
    s_q[tid] = (uint8_t)q;
    __syncthreads();
    if (tid < 16) blk_out->qs[tid] = s_q[tid] | (s_q[tid + 16] << 4);
}

extern "C" __global__
__launch_bounds__(32, 8)
void quantize_q4_0_from_fp8_e5m2_kernel(
    const uint8_t * __restrict__ src_fp8,
    uint8_t * __restrict__ dst,
    const float * __restrict__ imatrix,
    int nrows,
    int n_per_row
) {
    int row = blockIdx.x;
    int blk = blockIdx.y;
    int tid = threadIdx.x;
    int base = row * n_per_row + blk * QK4_0 + tid;
    if (base >= (row + 1) * n_per_row) return;

    float val = fp8_e5m2_to_fp32(src_fp8[base]);

    float v = val;
    float va = fabsf(val);
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, v, s);
        float other_a = __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, va, s);
        if (other_a > va) { v = other; va = other_a; }
    }
    float max_val = __shfl_sync(0xFFFFFFFFFFFFFFFFull, v, 0);
    float d = max_val / -8.0f;
    float id = d != 0 ? 1.0f / d : 0.0f;

    block_q4_0 *blk_out = (block_q4_0*)(dst + (row * (n_per_row / 32) + blk) * sizeof(block_q4_0));
    if (tid == 0) blk_out->d = fp32_to_fp16(d);

    int q = (int)(val * id + 8.5f);
    if (q < 0) q = 0;
    if (q > 15) q = 15;

    __shared__ uint8_t s_q[32];
    s_q[tid] = (uint8_t)q;
    __syncthreads();
    if (tid < 16) blk_out->qs[tid] = s_q[tid] | (s_q[tid + 16] << 4);
}

// ── Q4_1 from FP8 ──────────────────────────────────────────

extern "C" __global__
__launch_bounds__(32, 8)
void quantize_q4_1_from_fp8_e4m3_kernel(
    const uint8_t * __restrict__ src_fp8,
    uint8_t * __restrict__ dst,
    const float * __restrict__ imatrix,
    int nrows,
    int n_per_row
) {
    int row = blockIdx.x;
    int blk = blockIdx.y;
    int tid = threadIdx.x;
    int base = row * n_per_row + blk * QK4_1 + tid;
    if (base >= (row + 1) * n_per_row) return;

    float val = fp8_e4m3_to_fp32(src_fp8[base]);

    float v_min = val, v_max = val;
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        v_min = fminf(v_min, __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, v_min, s));
        v_max = fmaxf(v_max, __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, v_max, s));
    }
    float min_val = __shfl_sync(0xFFFFFFFFFFFFFFFFull, v_min, 0);
    float max_val = __shfl_sync(0xFFFFFFFFFFFFFFFFull, v_max, 0);

    float d = (max_val - min_val) / 15.0f;
    float id = d > 0 ? 1.0f / d : 0.0f;

    block_q4_1 *blk_out = (block_q4_1*)(dst + (row * (n_per_row / 32) + blk) * sizeof(block_q4_1));
    if (tid == 0) { blk_out->d = fp32_to_fp16(d); blk_out->m = fp32_to_fp16(min_val); }

    int q = (int)((val - min_val) * id + 0.5f);
    if (q < 0) q = 0;
    if (q > 15) q = 15;

    __shared__ uint8_t s_q[32];
    s_q[tid] = (uint8_t)q;
    __syncthreads();
    if (tid < 16) blk_out->qs[tid] = s_q[tid] | (s_q[tid + 16] << 4);
}

extern "C" __global__
__launch_bounds__(32, 8)
void quantize_q4_1_from_fp8_e5m2_kernel(
    const uint8_t * __restrict__ src_fp8,
    uint8_t * __restrict__ dst,
    const float * __restrict__ imatrix,
    int nrows,
    int n_per_row
) {
    int row = blockIdx.x;
    int blk = blockIdx.y;
    int tid = threadIdx.x;
    int base = row * n_per_row + blk * QK4_1 + tid;
    if (base >= (row + 1) * n_per_row) return;

    float val = fp8_e5m2_to_fp32(src_fp8[base]);

    float v_min = val, v_max = val;
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        v_min = fminf(v_min, __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, v_min, s));
        v_max = fmaxf(v_max, __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, v_max, s));
    }
    float min_val = __shfl_sync(0xFFFFFFFFFFFFFFFFull, v_min, 0);
    float max_val = __shfl_sync(0xFFFFFFFFFFFFFFFFull, v_max, 0);

    float d = (max_val - min_val) / 15.0f;
    float id = d > 0 ? 1.0f / d : 0.0f;

    block_q4_1 *blk_out = (block_q4_1*)(dst + (row * (n_per_row / 32) + blk) * sizeof(block_q4_1));
    if (tid == 0) { blk_out->d = fp32_to_fp16(d); blk_out->m = fp32_to_fp16(min_val); }

    int q = (int)((val - min_val) * id + 0.5f);
    if (q < 0) q = 0;
    if (q > 15) q = 15;

    __shared__ uint8_t s_q[32];
    s_q[tid] = (uint8_t)q;
    __syncthreads();
    if (tid < 16) blk_out->qs[tid] = s_q[tid] | (s_q[tid + 16] << 4);
}

// ── Q5_0 from FP8 ──────────────────────────────────────────

extern "C" __global__
__launch_bounds__(32, 8)
void quantize_q5_0_from_fp8_e4m3_kernel(
    const uint8_t * __restrict__ src_fp8,
    uint8_t * __restrict__ dst,
    const float * __restrict__ imatrix,
    int nrows,
    int n_per_row
) {
    int row = blockIdx.x;
    int blk = blockIdx.y;
    int tid = threadIdx.x;
    int base = row * n_per_row + blk * QK5_0 + tid;
    if (base >= (row + 1) * n_per_row) return;

    float val = fp8_e4m3_to_fp32(src_fp8[base]);

    float v = val;
    float va = fabsf(val);
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, v, s);
        float other_a = __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, va, s);
        if (other_a > va) { v = other; va = other_a; }
    }
    float max_val = __shfl_sync(0xFFFFFFFFFFFFFFFFull, v, 0);
    float d = max_val / -16.0f;
    float id = d != 0 ? 1.0f / d : 0.0f;

    block_q5_0 *blk_out = (block_q5_0*)(dst + (row * (n_per_row / 32) + blk) * sizeof(block_q5_0));
    if (tid == 0) blk_out->d = fp32_to_fp16(d);

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

    pack_q5_high_bits(blk_out->qh, s_q, tid);
}

extern "C" __global__
__launch_bounds__(32, 8)
void quantize_q5_0_from_fp8_e5m2_kernel(
    const uint8_t * __restrict__ src_fp8,
    uint8_t * __restrict__ dst,
    const float * __restrict__ imatrix,
    int nrows,
    int n_per_row
) {
    int row = blockIdx.x;
    int blk = blockIdx.y;
    int tid = threadIdx.x;
    int base = row * n_per_row + blk * QK5_0 + tid;
    if (base >= (row + 1) * n_per_row) return;

    float val = fp8_e5m2_to_fp32(src_fp8[base]);

    float v = val;
    float va = fabsf(val);
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, v, s);
        float other_a = __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, va, s);
        if (other_a > va) { v = other; va = other_a; }
    }
    float max_val = __shfl_sync(0xFFFFFFFFFFFFFFFFull, v, 0);
    float d = max_val / -16.0f;
    float id = d != 0 ? 1.0f / d : 0.0f;

    block_q5_0 *blk_out = (block_q5_0*)(dst + (row * (n_per_row / 32) + blk) * sizeof(block_q5_0));
    if (tid == 0) blk_out->d = fp32_to_fp16(d);

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

    pack_q5_high_bits(blk_out->qh, s_q, tid);
}

// ── Q5_1 from FP8 ──────────────────────────────────────────

extern "C" __global__
__launch_bounds__(32, 8)
void quantize_q5_1_from_fp8_e4m3_kernel(
    const uint8_t * __restrict__ src_fp8,
    uint8_t * __restrict__ dst,
    const float * __restrict__ imatrix,
    int nrows,
    int n_per_row
) {
    int row = blockIdx.x;
    int blk = blockIdx.y;
    int tid = threadIdx.x;
    int base = row * n_per_row + blk * QK5_1 + tid;
    if (base >= (row + 1) * n_per_row) return;

    float val = fp8_e4m3_to_fp32(src_fp8[base]);

    float v_min = val, v_max = val;
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        v_min = fminf(v_min, __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, v_min, s));
        v_max = fmaxf(v_max, __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, v_max, s));
    }
    float min_val = __shfl_sync(0xFFFFFFFFFFFFFFFFull, v_min, 0);
    float max_val = __shfl_sync(0xFFFFFFFFFFFFFFFFull, v_max, 0);
    float d = (max_val - min_val) / 31.0f;
    float id = d != 0 ? 1.0f / d : 0.0f;

    block_q5_1 *blk_out = (block_q5_1*)(dst + (row * (n_per_row / 32) + blk) * sizeof(block_q5_1));
    if (tid == 0) { blk_out->d = fp32_to_fp16(d); blk_out->m = fp32_to_fp16(min_val); }

    int q = (int)((val - min_val) * id + 0.5f);
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

    pack_q5_high_bits(blk_out->qh, s_q, tid);
}

extern "C" __global__
__launch_bounds__(32, 8)
void quantize_q5_1_from_fp8_e5m2_kernel(
    const uint8_t * __restrict__ src_fp8,
    uint8_t * __restrict__ dst,
    const float * __restrict__ imatrix,
    int nrows,
    int n_per_row
) {
    int row = blockIdx.x;
    int blk = blockIdx.y;
    int tid = threadIdx.x;
    int base = row * n_per_row + blk * QK5_1 + tid;
    if (base >= (row + 1) * n_per_row) return;

    float val = fp8_e5m2_to_fp32(src_fp8[base]);

    float v_min = val, v_max = val;
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        v_min = fminf(v_min, __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, v_min, s));
        v_max = fmaxf(v_max, __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, v_max, s));
    }
    float min_val = __shfl_sync(0xFFFFFFFFFFFFFFFFull, v_min, 0);
    float max_val = __shfl_sync(0xFFFFFFFFFFFFFFFFull, v_max, 0);
    float d = (max_val - min_val) / 31.0f;
    float id = d != 0 ? 1.0f / d : 0.0f;

    block_q5_1 *blk_out = (block_q5_1*)(dst + (row * (n_per_row / 32) + blk) * sizeof(block_q5_1));
    if (tid == 0) { blk_out->d = fp32_to_fp16(d); blk_out->m = fp32_to_fp16(min_val); }

    int q = (int)((val - min_val) * id + 0.5f);
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

    pack_q5_high_bits(blk_out->qh, s_q, tid);
}

// ── Q8_0 from FP8 ──────────────────────────────────────────

extern "C" __global__
__launch_bounds__(32, 8)
void quantize_q8_0_from_fp8_e4m3_kernel(
    const uint8_t * __restrict__ src_fp8,
    uint8_t * __restrict__ dst,
    const float * __restrict__ imatrix,
    int nrows,
    int n_per_row
) {
    int row = blockIdx.x;
    int blk = blockIdx.y;
    int tid = threadIdx.x;
    int base = row * n_per_row + blk * QK8_0 + tid;
    if (base >= (row + 1) * n_per_row) return;

    float val = fp8_e4m3_to_fp32(src_fp8[base]);

    float v = fabsf(val);
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, v, s));
    }
    float amax = __shfl_sync(0xFFFFFFFFFFFFFFFFull, v, 0);

    float d = amax / 127.0f;
    float id = d > 0 ? 1.0f / d : 0.0f;

    block_q8_0 *blk_out = (block_q8_0*)(dst + (row * (n_per_row / 32) + blk) * sizeof(block_q8_0));
    if (tid == 0) blk_out->d = fp32_to_fp16(d);

    int q = (int)roundf(val * id);
    if (q > 127) q = 127;
    if (q < -127) q = -127;
    blk_out->qs[tid] = (int8_t)q;
}

extern "C" __global__
__launch_bounds__(32, 8)
void quantize_q8_0_from_fp8_e5m2_kernel(
    const uint8_t * __restrict__ src_fp8,
    uint8_t * __restrict__ dst,
    const float * __restrict__ imatrix,
    int nrows,
    int n_per_row
) {
    int row = blockIdx.x;
    int blk = blockIdx.y;
    int tid = threadIdx.x;
    int base = row * n_per_row + blk * QK8_0 + tid;
    if (base >= (row + 1) * n_per_row) return;

    float val = fp8_e5m2_to_fp32(src_fp8[base]);

    float v = fabsf(val);
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, v, s));
    }
    float amax = __shfl_sync(0xFFFFFFFFFFFFFFFFull, v, 0);

    float d = amax / 127.0f;
    float id = d > 0 ? 1.0f / d : 0.0f;

    block_q8_0 *blk_out = (block_q8_0*)(dst + (row * (n_per_row / 32) + blk) * sizeof(block_q8_0));
    if (tid == 0) blk_out->d = fp32_to_fp16(d);

    int q = (int)roundf(val * id);
    if (q > 127) q = 127;
    if (q < -127) q = -127;
    blk_out->qs[tid] = (int8_t)q;
}

// ── Q8_1 from FP8 ──────────────────────────────────────────

extern "C" __global__
__launch_bounds__(32, 8)
void quantize_q8_1_from_fp8_e4m3_kernel(
    const uint8_t * __restrict__ src_fp8,
    uint8_t * __restrict__ dst,
    const float * __restrict__ imatrix,
    int nrows,
    int n_per_row
) {
    int row = blockIdx.x;
    int blk = blockIdx.y;
    int tid = threadIdx.x;
    int base = row * n_per_row + blk * QK8_1 + tid;
    if (base >= (row + 1) * n_per_row) return;

    float val = fp8_e4m3_to_fp32(src_fp8[base]);

    float v = fabsf(val);
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, v, s));
    }
    float amax = __shfl_sync(0xFFFFFFFFFFFFFFFFull, v, 0);

    float d = amax / 127.0f;
    float id = d > 0 ? 1.0f / d : 0.0f;

    block_q8_1 *blk_out = (block_q8_1*)(dst + (row * (n_per_row / 32) + blk) * sizeof(block_q8_1));
    if (tid == 0) blk_out->d = fp32_to_fp16(d);

    int q = (int)roundf(val * id);
    if (q > 127) q = 127;
    if (q < -127) q = -127;
    blk_out->qs[tid] = (int8_t)q;

    int sum = q;
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, sum, s);
    }

    if (tid == 0) {
        blk_out->s = fp32_to_fp16((float)__shfl_sync(0xFFFFFFFFFFFFFFFFull, sum, 0) * d);
    }
}

extern "C" __global__
__launch_bounds__(32, 8)
void quantize_q8_1_from_fp8_e5m2_kernel(
    const uint8_t * __restrict__ src_fp8,
    uint8_t * __restrict__ dst,
    const float * __restrict__ imatrix,
    int nrows,
    int n_per_row
) {
    int row = blockIdx.x;
    int blk = blockIdx.y;
    int tid = threadIdx.x;
    int base = row * n_per_row + blk * QK8_1 + tid;
    if (base >= (row + 1) * n_per_row) return;

    float val = fp8_e5m2_to_fp32(src_fp8[base]);

    float v = fabsf(val);
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, v, s));
    }
    float amax = __shfl_sync(0xFFFFFFFFFFFFFFFFull, v, 0);

    float d = amax / 127.0f;
    float id = d > 0 ? 1.0f / d : 0.0f;

    block_q8_1 *blk_out = (block_q8_1*)(dst + (row * (n_per_row / 32) + blk) * sizeof(block_q8_1));
    if (tid == 0) blk_out->d = fp32_to_fp16(d);

    int q = (int)roundf(val * id);
    if (q > 127) q = 127;
    if (q < -127) q = -127;
    blk_out->qs[tid] = (int8_t)q;

    int sum = q;
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, sum, s);
    }

    if (tid == 0) {
        blk_out->s = fp32_to_fp16((float)__shfl_sync(0xFFFFFFFFFFFFFFFFull, sum, 0) * d);
    }
}
