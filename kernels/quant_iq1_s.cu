#include "../hip_quant_types.h"
#include "../hip_quant_util.h"
#include <float.h>

#define GROUP_MAX_EPS_IQ1_S 1e-12f
#define IQ1S_BLOCK_SIZE 32
#define IQ1S_DELTA 0.125f
#define NGRID_IQ1S 2048

static __device__ int iq1s_find_best_neighbour2(
    const uint16_t * neighbours, const int8_t * grid,
    const float * xval, const float * weight, float scale, const float * xg, int8_t * L
) {
    int num_neighbors = neighbours[0];
    float best_score = FLT_MAX;
    int grid_index = -1;
    for (int j = 1; j <= num_neighbors; ++j) {
        int idx = neighbours[j];
        const int8_t * pg = grid + 8 * idx;
        float d2 = 0;
        for (int i = 0; i < 8; ++i) {
            float q = xg[(pg[i] - 1) / 2];
            float w = weight[i];
            float diff = scale * q - xval[i];
            d2 += w * diff * diff;
        }
        if (d2 < best_score) {
            best_score = d2;
            grid_index = idx;
        }
    }
    if (grid_index < 0) grid_index = 0;
    const int8_t * pg = grid + 8 * grid_index;
    for (int i = 0; i < 8; ++i) L[i] = (pg[i] - 1) / 2;
    return grid_index;
}

__global__ void quantize_iq1_s_kernel(
    const float * __restrict__ src,
    uint8_t * __restrict__ dst,
    const float * __restrict__ imatrix,
    const int8_t * __restrict__ grid,
    const int * __restrict__ map,
    const uint16_t * __restrict__ neighbours_data,
    int nrows,
    int n_per_row
) {
    if (threadIdx.x != 0) return;

    int row = blockIdx.x;
    int ibl = blockIdx.y;
    if (row >= nrows) return;
    int nbl = n_per_row / QK_K;
    if (ibl >= nbl) return;

    const float * xbl = src + row * n_per_row + QK_K * ibl;
    block_iq1_s * y = (block_iq1_s *)(dst + (row * nbl + ibl) * sizeof(block_iq1_s));

    y->d = 0;
    for (int i = 0; i < QK_K / 8; ++i) y->qs[i] = 0;
    for (int i = 0; i < QK_K / 32; ++i) y->qh[i] = 0;

    const float x_p[3] = {-1.0f + IQ1S_DELTA, IQ1S_DELTA, 1.0f + IQ1S_DELTA};
    const float x_m[3] = {-1.0f - IQ1S_DELTA, -IQ1S_DELTA, 1.0f - IQ1S_DELTA};
    float scales[QK_K / IQ1S_BLOCK_SIZE];
    int8_t shifts[QK_K / IQ1S_BLOCK_SIZE];
    float max_scale = 0;

    float sumx2 = 0;
    for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i] * xbl[i];
    float sigma2 = 2.0f * sumx2 / QK_K;

    for (int ib = 0; ib < QK_K / IQ1S_BLOCK_SIZE; ++ib) {
        const float * xb = xbl + IQ1S_BLOCK_SIZE * ib;
        const float * qw = imatrix ? imatrix + row * n_per_row + QK_K * ibl + IQ1S_BLOCK_SIZE * ib : NULL;
        float weight[IQ1S_BLOCK_SIZE];
        int8_t L[IQ1S_BLOCK_SIZE];
        uint16_t index[IQ1S_BLOCK_SIZE / 8];
        float vals[IQ1S_BLOCK_SIZE];
        int idx[IQ1S_BLOCK_SIZE];
        float sumx[IQ1S_BLOCK_SIZE + 1], sumw[IQ1S_BLOCK_SIZE + 1];

        for (int i = 0; i < IQ1S_BLOCK_SIZE; ++i) {
            float wi = qw ? qw[i] * sqrtf(sigma2 + xb[i] * xb[i]) : xb[i] * xb[i];
            weight[i] = wi;
        }

        float max_v = fabsf(xb[0]);
        for (int i = 1; i < IQ1S_BLOCK_SIZE; ++i) {
            float ax = fabsf(xb[i]);
            if (ax > max_v) max_v = ax;
        }
        if (max_v < GROUP_MAX_EPS_IQ1_S) {
            scales[ib] = 0;
            shifts[ib] = 1;
            for (int i = 0; i < IQ1S_BLOCK_SIZE; ++i) L[i] = 1;
            continue;
        }

        for (int j = 0; j < IQ1S_BLOCK_SIZE; ++j) {
            vals[j] = xb[j];
            idx[j] = j;
        }
        for (int a = 1; a < IQ1S_BLOCK_SIZE; ++a) {
            float v = vals[a];
            int idv = idx[a];
            int b = a - 1;
            while (b >= 0 && vals[b] > v) {
                vals[b + 1] = vals[b];
                idx[b + 1] = idx[b];
                --b;
            }
            vals[b + 1] = v;
            idx[b + 1] = idv;
        }

        sumx[0] = 0;
        sumw[0] = 0;
        for (int j = 0; j < IQ1S_BLOCK_SIZE; ++j) {
            int i = idx[j];
            sumx[j + 1] = sumx[j] + weight[i] * xb[i];
            sumw[j + 1] = sumw[j] + weight[i];
        }

        float best_score = -FLT_MAX;
        float scale = max_v;
        int besti1 = -1, besti2 = -1, best_shift = 0;
        for (int i1 = 0; i1 <= IQ1S_BLOCK_SIZE; ++i1) {
            for (int i2 = i1; i2 <= IQ1S_BLOCK_SIZE; ++i2) {
                float sumqx = (sumx[i1] - sumx[0]) * x_p[0] + (sumx[i2] - sumx[i1]) * x_p[1] + (sumx[IQ1S_BLOCK_SIZE] - sumx[i2]) * x_p[2];
                float sumq2 = (sumw[i1] - sumw[0]) * x_p[0] * x_p[0] + (sumw[i2] - sumw[i1]) * x_p[1] * x_p[1] + (sumw[IQ1S_BLOCK_SIZE] - sumw[i2]) * x_p[2] * x_p[2];
                if (sumq2 > 0 && sumqx * sumqx > best_score * sumq2) {
                    scale = sumqx / sumq2;
                    best_score = scale * sumqx;
                    besti1 = i1;
                    besti2 = i2;
                    best_shift = 1;
                }
                sumqx = (sumx[i1] - sumx[0]) * x_m[0] + (sumx[i2] - sumx[i1]) * x_m[1] + (sumx[IQ1S_BLOCK_SIZE] - sumx[i2]) * x_m[2];
                sumq2 = (sumw[i1] - sumw[0]) * x_m[0] * x_m[0] + (sumw[i2] - sumw[i1]) * x_m[1] * x_m[1] + (sumw[IQ1S_BLOCK_SIZE] - sumw[i2]) * x_m[2] * x_m[2];
                if (sumq2 > 0 && sumqx * sumqx > best_score * sumq2) {
                    scale = sumqx / sumq2;
                    best_score = scale * sumqx;
                    besti1 = i1;
                    besti2 = i2;
                    best_shift = -1;
                }
            }
        }
        if (besti1 < 0 || besti2 < 0 || best_shift == 0) {
            scales[ib] = 0;
            shifts[ib] = 1;
            for (int i = 0; i < IQ1S_BLOCK_SIZE; ++i) L[i] = 1;
            continue;
        }

        for (int j = 0; j < besti1; ++j) L[idx[j]] = 0;
        for (int j = besti1; j < besti2; ++j) L[idx[j]] = 1;
        for (int j = besti2; j < IQ1S_BLOCK_SIZE; ++j) L[idx[j]] = 2;
        if (scale < 0) {
            for (int j = 0; j < IQ1S_BLOCK_SIZE; ++j) L[j] = 2 - L[j];
            scale = -scale;
            best_shift = -best_shift;
        }

        bool all_on_grid = true;
        const float * xx = best_shift == 1 ? x_p : x_m;
        for (int k = 0; k < IQ1S_BLOCK_SIZE / 8; ++k) {
            uint16_t u = 0;
            for (int j = 0; j < 8; ++j) u |= ((uint16_t)L[8 * k + j] << (2 * j));
            int grid_index = map[u];
            if (grid_index < 0) {
                all_on_grid = false;
                const uint16_t * neighbours = neighbours_data + (-grid_index - 1);
                grid_index = iq1s_find_best_neighbour2(neighbours, grid, xb + 8 * k, weight + 8 * k, scale, xx, L + 8 * k);
            }
            index[k] = (uint16_t)grid_index;
        }
        if (!all_on_grid) {
            float sumqx = 0, sumq2 = 0;
            for (int k = 0; k < IQ1S_BLOCK_SIZE / 8; ++k) {
                const int8_t * pg = grid + 8 * index[k];
                for (int j = 0; j < 8; ++j) {
                    float w = weight[8 * k + j];
                    float q = xx[(pg[j] - 1) / 2];
                    sumqx += w * q * xb[8 * k + j];
                    sumq2 += w * q * q;
                }
            }
            if (sumqx > 0 && sumq2 > 0) scale = sumqx / sumq2;
        }

        uint16_t h = 0;
        for (int k = 0; k < IQ1S_BLOCK_SIZE / 8; ++k) {
            y->qs[(IQ1S_BLOCK_SIZE / 8) * ib + k] = (uint8_t)(index[k] & 255);
            h |= (uint16_t)((index[k] >> 8) << (3 * k));
        }
        y->qh[ib] = h;
        scales[ib] = scale;
        shifts[ib] = (int8_t)best_shift;
        if (scale > max_scale) max_scale = scale;
    }

    if (!max_scale) return;
    float d = max_scale / 15.0f;
    y->d = fp32_to_fp16(d * 1.125f);
    float id = 1.0f / d;
    for (int ib = 0; ib < QK_K / IQ1S_BLOCK_SIZE; ++ib) {
        int l = nearest_int(0.5f * (id * scales[ib] - 1.0f));
        if (l < 0) l = 0;
        if (l > 7) l = 7;
        if (shifts[ib] == -1) l |= 8;
        y->qh[ib] |= (uint16_t)(l << 12);
    }
}
