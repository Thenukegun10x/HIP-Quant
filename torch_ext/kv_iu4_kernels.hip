// torch_ext/kv_iu4_kernels.hip
//
// INT4 KV-cache storage kernels (P1: V side; K side lands in P2).
//
// V layout per (head, token): D channels split into G groups (P0-locked G=4,
// 64 channels each at D=256). Each group carries an fp16 scale + u8
// zero-point; payload packs one byte per consecutive channel pair
// (low nibble = even channel, high nibble = odd channel).
//
//   packed [H, S, D/2] uint8, scales [H, S, G] fp16, zp [H, S, G] uint8
//   element (h, s, d): group g = d / (D/G),
//     byte = packed[(h*S + s)*(D/2) + d/2], nibble = d odd ? hi : lo
//     value = (nibble - zp[h,s,g]) * scale[h,s,g]
//
// Dequant decodes through a per-group 16-entry LUT built in LDS
// (1 LDS read vs 6-8 ALU ops per element). P1 reads dequantize to fp16 and
// reuse SDPA (fp-grade math); native WMMA fusion is P3.
//
// Rounding contract (verified Sep 2026, see iu4_p0/test_kv_kernels.py):
// on-device nearbyintf is round-half-even ([0,2,4,12] for [0.5,2.5,4.5,12.5]).
// Against an independent np.float32 reference over 32 shape/group/cases
// (~1M nibbles incl. real KV dumps): scales bit-exact, zp bit-exact,
// dequant bit-exact given the same payload, and exactly ONE nibble differed
// by +/-1 at an exact-.5 pre-rounding value (0x3f000000 both sides) -- a
// 1-ulp evaluation-order difference in v*inv+z (FMA-contraction latitude is
// explicitly IEEE-legal), deployment-irrelevant (<=1 LSB on one element;
// the live-kernel PPL gate is the arbiter). PyTorch CUDA scalar division
// lowers to reciprocal-multiply (observed 1-ulp vs true division at a zp
// .5-boundary: 0x3e971112 vs 0x3e971111) -- the KERNEL's true division is
// the more faithful one; torch-side sims inherit the 1-ulp noise.
//
// Launch convention: one HIP block (256 threads) per (head, token) row,
// strided loops over D. Supports D <= 512, G in {1,2,4,8} with D % G == 0.

#include <hip/hip_runtime.h>
#include <assert.h>
#include "../hip_quant_util.h"

#define KV_IU4_DTYPE_F32  0
#define KV_IU4_DTYPE_F16  1
#define KV_IU4_DTYPE_BF16 2
#define KV_IU4_MAX_D 512
#define KV_IU4_MAX_G 8

template <int DTYPE>
static __device__ inline float kv_i4_load(const void* src, int64_t idx) {
    if constexpr (DTYPE == KV_IU4_DTYPE_F16) {
        return fp16_to_fp32(((const uint16_t*)src)[idx]);
    }
    if constexpr (DTYPE == KV_IU4_DTYPE_BF16) {
        return bf16_to_fp32(((const uint16_t*)src)[idx]);
    }
    return ((const float*)src)[idx];
}

// ---------------------------------------------------------------------------
// Quantize V rows to asymmetric UINT4 nibbles.
// Grid: H*S blocks (one per head-token row). Each block:
//   pass 1: per-group amin/amax reduction -> scale/zp
//   pass 2: quantize row into shared bytes, pack pairs -> global
// ---------------------------------------------------------------------------
template <int DTYPE>
__global__
__launch_bounds__(256, 4)
void kv_quant_v_i4_kernel_t(
    const void* __restrict__ src,      // [H, S, D]
    uint8_t* __restrict__ packed,      // [H, S, D/2]
    uint16_t* __restrict__ scales,     // [H, S, G] fp16 bits
    uint8_t* __restrict__ zp,          // [H, S, G]
    int64_t H, int64_t S, int64_t D, int groups
) {
    int64_t row = (int64_t)blockIdx.x;          // head*S + s
    int64_t group_elems = D / groups;
    int64_t row_base = row * D;

    __shared__ float s_scale[KV_IU4_MAX_G];
    __shared__ float s_zp[KV_IU4_MAX_G];
    // Per-thread partial stats, reduced cross-thread through shared staging.
    float t_min[KV_IU4_MAX_G];
    float t_max[KV_IU4_MAX_G];
    #pragma unroll
    for (int g = 0; g < KV_IU4_MAX_G; g++) {
        t_min[g] = INFINITY;
        t_max[g] = -INFINITY;
    }
    for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
        float v = kv_i4_load<DTYPE>(src, row_base + d);
        float av = fabsf(v);
        // Match the blockwise convention: non-finite samples do not poison
        // the range (post-norm KV is clean; debug_decode health asserts it).
        if (!isfinite(av)) continue;
        int g = (int)(d / group_elems);
        if (v < t_min[g]) t_min[g] = v;
        if (v > t_max[g]) t_max[g] = v;
    }
    // Cross-thread reduction through shared memory (256 iterations x up to
    // 8 groups is trivial next to the D-wide pass).
    __shared__ float red_min[KV_IU4_MAX_G][256];
    __shared__ float red_max[KV_IU4_MAX_G][256];
    #pragma unroll
    for (int g = 0; g < KV_IU4_MAX_G; g++) {
        if (g < groups) {
            red_min[g][threadIdx.x] = t_min[g];
            red_max[g][threadIdx.x] = t_max[g];
        }
    }
    __syncthreads();
    if (threadIdx.x < groups) {
        int g = threadIdx.x;
        float mn = INFINITY, mx = -INFINITY;
        for (int t = 0; t < 256; t++) {
            float a = red_min[g][t], b = red_max[g][t];
            if (a < mn) mn = a;
            if (b > mx) mx = b;
        }
        if (!isfinite(mn)) { mn = 0.0f; mx = 0.0f; }  // all-nonfinite row: emit zeros
        float scale = (mx - mn) / 15.0f;
        if (!(scale > 0.0f)) scale = 1.0f;            // flat row (or inf-free zero range)
        float zpf = -mn / scale;
        int zpi = (int)nearbyintf(zpf);
        if (zpi < 0) zpi = 0;
        if (zpi > 15) zpi = 15;
        s_scale[g] = scale;
        s_zp[g] = (float)zpi;
        scales[row * groups + g] = fp32_to_fp16(scale);
        zp[row * groups + g] = (uint8_t)zpi;
    }
    __syncthreads();

    // Pass 2: quantize into shared staging, then pack pairs.
    __shared__ uint8_t qstage[KV_IU4_MAX_D];
    float inv_scale[KV_IU4_MAX_G];
    float zpf_[KV_IU4_MAX_G];
    #pragma unroll
    for (int g = 0; g < KV_IU4_MAX_G; g++) {
        if (g < groups) {
            inv_scale[g] = 1.0f / s_scale[g];
            zpf_[g] = s_zp[g];
        }
    }
    for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
        float v = kv_i4_load<DTYPE>(src, row_base + d);
        int g = (int)(d / group_elems);
        int q = (int)nearbyintf(v * inv_scale[g] + zpf_[g]);
        if (q < 0) q = 0;
        if (q > 15) q = 15;
        qstage[d] = (uint8_t)q;
    }
    __syncthreads();
    int64_t pairs = D / 2;
    int64_t pack_base = row * pairs;
    for (int64_t p = threadIdx.x; p < pairs; p += blockDim.x) {
        uint8_t lo = qstage[2 * p];
        uint8_t hi = qstage[2 * p + 1];
        packed[pack_base + p] = (uint8_t)(lo | (hi << 4));
    }
}

// ---------------------------------------------------------------------------
// Dequantize V rows through a per-group 16-entry LDS LUT: lut[g][q] =
// (q - zp[g]) * scale[g]. Grid: H*S blocks, 256 threads, strided over D.
// ---------------------------------------------------------------------------
__global__
__launch_bounds__(256, 4)
void kv_dequant_v_i4_kernel(
    const uint8_t* __restrict__ packed,   // [H, S, D/2]
    const uint16_t* __restrict__ scales,  // [H, S, G] fp16 bits
    const uint8_t* __restrict__ zp,       // [H, S, G]
    uint16_t* __restrict__ dst,           // [H, S, D] fp16 bits
    int64_t H, int64_t S, int64_t D, int groups
) {
    int64_t row = (int64_t)blockIdx.x;
    int64_t group_elems = D / groups;
    int64_t pairs = D / 2;

    __shared__ float lut[KV_IU4_MAX_G][16];
    for (int g = threadIdx.x; g < groups * 16; g += blockDim.x) {
        int gg = g / 16, q = g % 16;
        float scale = fp16_to_fp32(scales[row * groups + gg]);
        float z = (float)zp[row * groups + gg];
        lut[gg][q] = ((float)q - z) * scale;
    }
    // Padded pitch defeats LDS bank conflicts on the lookup path
    // (see wave_attn LDS_PAD invariant); 16-wide rows already stride cleanly.
    __syncthreads();

    int64_t pack_base = row * pairs;
    int64_t dst_base = row * D;
    for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
        uint8_t byte = packed[pack_base + d / 2];
        uint8_t nib = (d & 1) ? (uint8_t)(byte >> 4) : (uint8_t)(byte & 0xF);
        int g = (int)(d / group_elems);
        dst[dst_base + d] = fp32_to_fp16(lut[g][nib]);
    }
}

// ---------------------------------------------------------------------------
// C-linkage launch helpers (host-callable).
// ---------------------------------------------------------------------------
static inline uint32_t kv_i4_rows(int64_t H, int64_t S) {
    assert(H > 0 && S > 0);
    assert((int64_t)(H * S) <= (int64_t)0xFFFFFFFFu);
    return (uint32_t)(H * S);
}

static inline bool kv_i4_valid_shape(int64_t D, int groups) {
    if (D <= 0 || D > KV_IU4_MAX_D) return false;
    if (D & 1) return false;
    if (groups < 1 || groups > KV_IU4_MAX_G) return false;
    if (D % groups != 0) return false;
    return true;
}

extern "C" {

void launch_kv_quant_v_i4(const void* src, uint8_t* packed, uint16_t* scales,
                           uint8_t* zp, int64_t H, int64_t S, int64_t D,
                           int groups, int dtype, hipStream_t stream) {
    if (H == 0 || S == 0) return;
    assert(kv_i4_valid_shape(D, groups));
    uint32_t grid = kv_i4_rows(H, S);
    if (dtype == KV_IU4_DTYPE_F16) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(kv_quant_v_i4_kernel_t<KV_IU4_DTYPE_F16>),
                           dim3(grid), dim3(256), 0, stream,
                           src, packed, scales, zp, H, S, D, groups);
    } else if (dtype == KV_IU4_DTYPE_BF16) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(kv_quant_v_i4_kernel_t<KV_IU4_DTYPE_BF16>),
                           dim3(grid), dim3(256), 0, stream,
                           src, packed, scales, zp, H, S, D, groups);
    } else {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(kv_quant_v_i4_kernel_t<KV_IU4_DTYPE_F32>),
                           dim3(grid), dim3(256), 0, stream,
                           src, packed, scales, zp, H, S, D, groups);
    }
}

void launch_kv_dequant_v_i4(const uint8_t* packed, const uint16_t* scales,
                             const uint8_t* zp, uint16_t* dst,
                             int64_t H, int64_t S, int64_t D, int groups,
                             hipStream_t stream) {
    if (H == 0 || S == 0) return;
    assert(kv_i4_valid_shape(D, groups));
    hipLaunchKernelGGL(kv_dequant_v_i4_kernel,
                       dim3(kv_i4_rows(H, S)), dim3(256), 0, stream,
                       packed, scales, zp, dst, H, S, D, groups);
}

} // extern "C"
