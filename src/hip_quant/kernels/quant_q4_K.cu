#include <hip/hip_runtime.h>
#include "../hip_quant_types.h"
#include "../hip_quant_util.h"

#define QK_K 256
#define GROUP_MAX_EPS 1e-15f

__device__ inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t &d, uint8_t &m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j+4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >> 4)   | ((q[j]   >> 6) << 4);
    }
}

__device__ inline void compute_weights_subblock(
    const float *s_x, float *s_w, int grp,
    float &sum_x2, float &sum_w
) {
    sum_x2 = 0; sum_w = 0;
    for (int k = 0; k < 32; ++k) {
        float val = s_x[grp * 32 + k];
        sum_x2 += val * val;
    }
    float av_x = sqrtf(sum_x2 / 32.0f);
    for (int k = 0; k < 32; ++k) {
        float v = av_x + fabsf(s_x[grp * 32 + k]);
        s_w[grp * 32 + k] = v;
        sum_w += v;
    }
}

// Serial make_qkx2_quants run by thread 0 of each group
// rmin=-1, rdelta=0.1, nstep=20, use_mad=false
__device__ inline float make_qkx2_quants_device(
    int grp, const float *s_x, const float *s_w, uint8_t *s_L,
    float *out_min, float rmin, float rdelta, int nstep
) {
    int base = grp * 32;
    float min_v = s_x[base];
    float max_v = s_x[base];
    float sum_w = s_w[base];
    float sum_x = sum_w * s_x[base];

    for (int k = 1; k < 32; ++k) {
        float xv = s_x[base + k];
        float wv = s_w[base + k];
        if (xv < min_v) min_v = xv;
        if (xv > max_v) max_v = xv;
        sum_w += wv;
        sum_x += wv * xv;
    }

    if (min_v > 0) min_v = 0;
    if (max_v <= min_v) {
        for (int k = 0; k < 32; ++k) s_L[base + k] = 0;
        *out_min = -min_v;
        return 0.0f;
    }

    float iscale = 15.0f / (max_v - min_v);
    float scale = 1.0f / iscale;
    float best_error = 0.0f;

    for (int k = 0; k < 32; ++k) {
        int l = nearest_int(iscale * (s_x[base + k] - min_v));
        l = max(0, min(15, l));
        s_L[base + k] = (uint8_t)l;
        float diff = scale * l + min_v - s_x[base + k];
        float wv = s_w[base + k];
        best_error += wv * diff * diff;
    }

    uint8_t Laux[32];
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta * (float)is + 15.0f) / (max_v - min_v);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int k = 0; k < 32; ++k) {
            int l = nearest_int(iscale * (s_x[base + k] - min_v));
            l = max(0, min(15, l));
            Laux[k] = (uint8_t)l;
            float wv = s_w[base + k];
            sum_l += wv * l;
            sum_l2 += wv * l * l;
            sum_xl += wv * l * s_x[base + k];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
            float this_min   = (sum_l2 * sum_x  - sum_l * sum_xl) / D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float cur_error = 0.0f;
            for (int k = 0; k < 32; ++k) {
                float diff = this_scale * Laux[k] + this_min - s_x[base + k];
                float wv = s_w[base + k];
                cur_error += wv * diff * diff;
            }
            if (cur_error < best_error) {
                for (int k = 0; k < 32; ++k) s_L[base + k] = Laux[k];
                best_error = cur_error;
                scale = this_scale;
                min_v = this_min;
            }
        }
    }

    *out_min = -min_v;
    return scale;
}

extern "C" __global__
__launch_bounds__(256, 4)
void quantize_q4_K_kernel(
    const float * __restrict__ src,
    uint8_t * __restrict__ dst,
    const float * __restrict__ imatrix,
    int nrows,
    int n_per_row
) {
    int row = blockIdx.x;
    int sb = blockIdx.y;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int grp  = tid >> 5;

    int base = row * n_per_row + sb * QK_K + grp * 32 + lane;
    if (base > (row + 1) * n_per_row - 1) return;

    __shared__ float s_x[QK_K];
    __shared__ float s_w[QK_K];
    __shared__ uint8_t s_L[QK_K];
    __shared__ float s_scales[8];
    __shared__ float s_mins[8];
    __shared__ uint8_t s_sc[12];
    __shared__ float s_d_val;
    __shared__ float s_dm_val;

    s_x[grp * 32 + lane] = src[base];
    __syncthreads();

    if (lane == 0) {
        float sum_x2, sum_w;
        compute_weights_subblock(s_x, s_w, grp, sum_x2, sum_w);

        float min_val;
        float scale_val = make_qkx2_quants_device(
            grp, s_x, s_w, s_L, &min_val, -1.0f, 0.1f, 20
        );
        s_scales[grp] = scale_val;
        s_mins[grp] = min_val;
    }
    __syncthreads();

    if (tid == 0) {
        float max_scale = 0.0f;
        float max_min = 0.0f;
        for (int j = 0; j < 8; ++j) {
            if (s_scales[j] > max_scale) max_scale = s_scales[j];
            if (s_mins[j] > max_min) max_min = s_mins[j];
        }

        float inv_scale = max_scale > 0 ? 63.0f / max_scale : 0.0f;
        float inv_min = max_min > 0 ? 63.0f / max_min : 0.0f;

        for (int j = 0; j < 8; ++j) {
            uint8_t ls = (uint8_t)min(63, nearest_int(inv_scale * s_scales[j]));
            uint8_t lm = (uint8_t)min(63, nearest_int(inv_min * s_mins[j]));
            if (j < 4) {
                s_sc[j] = ls;
                s_sc[j + 4] = lm;
            } else {
                s_sc[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                s_sc[j - 4] |= ((ls >> 4) << 6);
                s_sc[j]     |= ((lm >> 4) << 6);
            }
        }

        block_q4_K *blk = (block_q4_K*)(dst + (row * (n_per_row / QK_K) + sb) * sizeof(block_q4_K));
        blk->d = fp32_to_fp16(max_scale / 63.0f);
        blk->dmin = fp32_to_fp16(max_min / 63.0f);
        for (int i = 0; i < 12; ++i) blk->scales[i] = s_sc[i];

        s_d_val = fp16_to_fp32(blk->d);
        s_dm_val = fp16_to_fp32(blk->dmin);
    }
    __syncthreads();

    // Final requantize: each thread handles its own element
    uint8_t sc, m;
    get_scale_min_k4(grp, s_sc, sc, m);
    float d  = s_d_val * sc;
    float dm = s_dm_val * m;
    int ql = nearest_int((s_x[grp * 32 + lane] + dm) / d);
    ql = max(0, min(15, ql));
    s_L[grp * 32 + lane] = (uint8_t)ql;
    __syncthreads();

    // Pack quants into output: CPU pairs L[i] with L[i+32] for chunks of 64
    // qs[chunk*32 + l] = L[chunk*64 + l] | (L[chunk*64 + l + 32] << 4)
    block_q4_K *blk = (block_q4_K*)(dst + (row * (n_per_row / QK_K) + sb) * sizeof(block_q4_K));
    if (tid < 128) {
        int chunk = tid / 32;
        int l = tid % 32;
        int lo = s_L[chunk * 64 + l];
        int hi = s_L[chunk * 64 + l + 32];
        blk->qs[tid] = (uint8_t)(lo | (hi << 4));
    }
}
