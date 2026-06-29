#include <hip/hip_runtime.h>
#include "../hip_quant_types.h"
#include "../hip_quant_util.h"

#define QK_K 256
#define GROUP_MAX_EPS 1e-15f

// make_q3_quants for n=16, nmax=4, do_rmse=true
__device__ inline float make_q3_quants_q3(
    int grp, const float *s_x, int8_t *s_L
) {
    int base = grp * 16;
    float max_val = 0, amax = 0;

    for (int k = 0; k < 16; ++k) {
        float ax = fabsf(s_x[base + k]);
        if (ax > amax) { amax = ax; max_val = s_x[base + k]; }
    }

    if (amax < GROUP_MAX_EPS) {
        for (int k = 0; k < 16; ++k) s_L[base + k] = 0;
        return 0.0f;
    }

    float iscale = -4.0f / max_val;
    float sumlx = 0, suml2 = 0;

    for (int k = 0; k < 16; ++k) {
        float xv = s_x[base + k];
        int l = nearest_int(iscale * xv);
        l = max(-4, min(3, l));
        s_L[base + k] = (int8_t)l;
        float w = xv * xv;
        sumlx += w * xv * l;
        suml2 += w * l * l;
    }

    for (int itry = 0; itry < 5; ++itry) {
        int n_changed = 0;
        for (int k = 0; k < 16; ++k) {
            float xv = s_x[base + k];
            float w = xv * xv;
            float slx = sumlx - w * xv * (float)s_L[base + k];
            if (slx > 0) {
                float sl2 = suml2 - w * (float)s_L[base + k] * (float)s_L[base + k];
                int new_l = nearest_int(xv * sl2 / slx);
                new_l = max(-4, min(3, new_l));
                if (new_l != s_L[base + k]) {
                    slx += w * xv * new_l;
                    sl2 += w * (float)new_l * new_l;
                    if (sl2 > 0 && slx * slx * suml2 > sumlx * sumlx * sl2) {
                        s_L[base + k] = (int8_t)new_l;
                        sumlx = slx;
                        suml2 = sl2;
                        ++n_changed;
                    }
                }
            }
        }
        if (!n_changed) break;
    }

    for (int k = 0; k < 16; ++k) s_L[base + k] += 4;
    return suml2 > 0.0f ? sumlx / suml2 : 0.0f;
}

extern "C" __global__
__launch_bounds__(256, 4)
void quantize_q3_K_kernel(
    const float * __restrict__ src,
    uint8_t * __restrict__ dst,
    const float * __restrict__ imatrix,
    int nrows,
    int n_per_row
) {
    int row = blockIdx.x;
    int sb = blockIdx.y;
    int tid = threadIdx.x;
    int lane = tid & 15;
    int grp  = tid >> 4;

    int base = row * n_per_row + sb * QK_K + grp * 16 + lane;
    if (base > (row + 1) * n_per_row - 1) return;

    __shared__ float s_x[QK_K];
    __shared__ int8_t s_L[QK_K];
    __shared__ float s_scales[16];
    __shared__ float s_d_val;
    __shared__ uint8_t s_sc_packed[12];

    s_x[grp * 16 + lane] = src[base];
    __syncthreads();

    if (lane == 0) {
        float scale_val = make_q3_quants_q3(grp, s_x, s_L);
        s_scales[grp] = scale_val;
    }
    __syncthreads();

    // Initialize s_sc_packed to 0 (needed for OR operations)
    if (lane == 0) {
        s_sc_packed[0] = s_sc_packed[1] = s_sc_packed[2] = s_sc_packed[3] = 0;
        s_sc_packed[4] = s_sc_packed[5] = s_sc_packed[6] = s_sc_packed[7] = 0;
        s_sc_packed[8] = s_sc_packed[9] = s_sc_packed[10] = s_sc_packed[11] = 0;
    }
    __syncthreads();

    if (tid == 0) {
        float max_scale = 0, amax = 0;
        for (int j = 0; j < 16; ++j) {
            float scale = fabsf(s_scales[j]);
            if (scale > amax) { amax = scale; max_scale = s_scales[j]; }
        }

        block_q3_K *blk = (block_q3_K*)(dst + (row * (n_per_row / QK_K) + sb) * sizeof(block_q3_K));

        if (max_scale != 0) {
            float iscale = -32.0f / max_scale;
            for (int j = 0; j < 16; ++j) {
                int8_t l_val = (int8_t)(nearest_int(iscale * s_scales[j]));
                l_val = max(-32, min(31, l_val));
                int l_u = (int)(l_val + 32);  // 0-63

                if (j < 8) {
                    s_sc_packed[j] = (uint8_t)(l_u & 0xF);
                } else {
                    s_sc_packed[j - 8] |= (uint8_t)((l_u & 0xF) << 4);
                }
                l_u >>= 4;  // upper 2 bits
                s_sc_packed[(j % 4) + 8] |= (uint8_t)(l_u << (2 * (j / 4)));
            }
            for (int j = 0; j < 12; ++j) blk->scales[j] = s_sc_packed[j];
            blk->d = fp32_to_fp16(1.0f / iscale);
            s_d_val = fp16_to_fp32(blk->d);
        } else {
            for (int j = 0; j < 12; ++j) blk->scales[j] = 0;
            blk->d = fp32_to_fp16(0.0f);
            s_d_val = 0.0f;
        }
    }
    __syncthreads();

    // Requantize: unpack scale for this sub-block
    if (s_d_val != 0.0f) {
        int8_t sc_val;
        uint8_t raw = grp < 8 ? s_sc_packed[grp] & 0xF : s_sc_packed[grp - 8] >> 4;
        int upper = (s_sc_packed[8 + (grp % 4)] >> (2 * (grp / 4))) & 3;
        sc_val = (int8_t)(raw | (upper << 4)) - 32;

        float d = s_d_val * (float)sc_val;
        float xv = s_x[grp * 16 + lane];
        int l = nearest_int(xv / d);
        l = max(-4, min(3, l));
        s_L[grp * 16 + lane] = (int8_t)(l + 4);
    }
    __syncthreads();

    // Pack hmask[32] and qs[64]
    block_q3_K *blk = (block_q3_K*)(dst + (row * (n_per_row / QK_K) + sb) * sizeof(block_q3_K));

    if (tid == 0) {
        for (int k = 0; k < 32; ++k) blk->hmask[k] = 0;
        int m = 0;
        uint8_t hm = 1;
        for (int j = 0; j < QK_K; ++j) {
            if (s_L[j] > 3) {
                blk->hmask[m] |= hm;
                s_L[j] -= 4;
            }
            if (++m == 32) { m = 0; hm <<= 1; }
        }
    }
    __syncthreads();

    // Pack qs: 4 values per byte, same layout as Q2_K
    if (tid < 64) {
        int chunk = tid / 32;
        int l = tid % 32;
        int base_idx = chunk * 128;
        uint8_t byte = ((uint8_t)s_L[base_idx + l])
                     | ((uint8_t)(s_L[base_idx + l + 32]) << 2)
                     | ((uint8_t)(s_L[base_idx + l + 64]) << 4)
                     | ((uint8_t)(s_L[base_idx + l + 96]) << 6);
        blk->qs[chunk * 32 + l] = byte;
    }
}
