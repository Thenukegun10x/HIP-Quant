#include "../hip_quant_types.h"
#include "../hip_quant_util.h"
#include <float.h>

#define GROUP_MAX_EPS 1e-15f

static __device__ int iq2xs_find_best_neighbour(
    const uint16_t * neighbours, const int8_t * grid,
    const float * xval, const float * weight, float scale, int8_t * L
) {
    int num_neighbors = neighbours[0];
    float best_d2 = FLT_MAX;
    int best_idx = neighbours[1];
    for (int j = 1; j <= num_neighbors; ++j) {
        int idx = neighbours[j];
        const int8_t * pg = grid + 8 * idx;
        float d2 = 0;
        for (int i = 0; i < 8; ++i) {
            float q = (float)pg[i];
            float diff = scale * q - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if (d2 < best_d2) {
            best_d2 = d2;
            best_idx = idx;
        }
    }
    const int8_t * pg = grid + 8 * best_idx;
    for (int i = 0; i < 8; ++i) L[i] = (pg[i] - 1) / 2;
    return best_idx;
}

__global__ void quantize_iq2_xs_kernel(
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
    block_iq2_xs * y = (block_iq2_xs *)(dst + (row * nbl + ibl) * sizeof(block_iq2_xs));

    float scales[QK_K / 16];
    uint16_t q2[2 * (QK_K / 16)];
    for (int i = 0; i < 2 * (QK_K / 16); ++i) q2[i] = 0;
    for (int i = 0; i < QK_K / 32; ++i) y->scales[i] = 0;

    float max_scale = 0;
    float sumx2 = 0;
    for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i] * xbl[i];
    float sigma2 = sumx2 / QK_K;

    for (int ib = 0; ib < QK_K / 16; ++ib) {
        const float * xb = xbl + 16 * ib;
        const float * qw = imatrix ? imatrix + row * n_per_row + QK_K * ibl + 16 * ib : NULL;
        float weight[16], waux[16], xval[16];
        int8_t L[16], Laux[16];
        bool is_on_grid[2], is_on_grid_aux[2];
        uint8_t block_signs[2];

        for (int i = 0; i < 16; ++i) {
            float wi = qw ? qw[i] * sqrtf(sigma2 + xb[i] * xb[i]) : xb[i] * xb[i];
            weight[i] = wi;
            waux[i] = sqrtf(wi);
        }
        for (int k = 0; k < 2; ++k) {
            int nflip = 0;
            uint8_t s = 0;
            for (int i = 0; i < 8; ++i) {
                if (xb[8 * k + i] >= 0) xval[8 * k + i] = xb[8 * k + i];
                else {
                    xval[8 * k + i] = -xb[8 * k + i];
                    ++nflip;
                    s |= (1u << i);
                }
            }
            if (nflip % 2) {
                int imin = 0;
                float min_v = weight[8 * k] * xb[8 * k] * xb[8 * k];
                for (int i = 1; i < 8; ++i) {
                    float ax = weight[8 * k + i] * xb[8 * k + i] * xb[8 * k + i];
                    if (ax < min_v) { min_v = ax; imin = i; }
                }
                xval[8 * k + imin] = -xval[8 * k + imin];
                s ^= (1u << imin);
            }
            block_signs[k] = s & 127;
        }

        float max_v = xval[0];
        for (int i = 1; i < 16; ++i) if (xval[i] > max_v) max_v = xval[i];
        for (int i = 0; i < 16; ++i) L[i] = 0;
        if (max_v < GROUP_MAX_EPS) { scales[ib] = 0; continue; }

        float best = 0;
        float scale = max_v / 5.0f;
        is_on_grid[0] = is_on_grid[1] = true;
        for (int is = -9; is <= 9; ++is) {
            float id = (5.0f + (float)is * 0.1f) / max_v;
            float this_scale = 1.0f / id;
            for (int k = 0; k < 2; ++k) {
                for (int i = 0; i < 8; ++i) {
                    int l = nearest_int(0.5f * (id * xval[8 * k + i] - 1.0f));
                    if (l < 0) l = 0;
                    if (l > 2) l = 2;
                    Laux[8 * k + i] = (int8_t)l;
                }
                uint16_t u = 0;
                for (int i = 0; i < 8; ++i) u |= ((uint16_t)Laux[8 * k + i] << (2 * i));
                int grid_index = map[u];
                is_on_grid_aux[k] = true;
                if (grid_index < 0) {
                    is_on_grid_aux[k] = false;
                    const uint16_t * neighbours = neighbours_data + (-grid_index - 1);
                    grid_index = iq2xs_find_best_neighbour(neighbours, grid, xval + 8 * k, waux + 8 * k, this_scale, Laux + 8 * k);
                }
            }
            float sumqx = 0, sumq2 = 0;
            for (int i = 0; i < 16; ++i) {
                float w = weight[i];
                float q = 2.0f * (float)Laux[i] + 1.0f;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
                scale = sumqx / sumq2;
                best = scale * sumqx;
                for (int i = 0; i < 16; ++i) L[i] = Laux[i];
                for (int k = 0; k < 2; ++k) is_on_grid[k] = is_on_grid_aux[k];
            }
        }

        int n_not_ongrid = 0;
        for (int k = 0; k < 2; ++k) if (!is_on_grid[k]) ++n_not_ongrid;
        if (n_not_ongrid > 0 && scale > 0) {
            float id = 1.0f / scale;
            for (int k = 0; k < 2; ++k) {
                if (is_on_grid[k]) continue;
                uint16_t u = 0;
                for (int i = 0; i < 8; ++i) {
                    int l = nearest_int(0.5f * (id * xval[8 * k + i] - 1.0f));
                    if (l < 0) l = 0;
                    if (l > 2) l = 2;
                    u |= ((uint16_t)l << (2 * i));
                    L[8 * k + i] = (int8_t)l;
                }
                int grid_index = map[u];
                if (grid_index < 0) {
                    const uint16_t * neighbours = neighbours_data + (-grid_index - 1);
                    grid_index = iq2xs_find_best_neighbour(neighbours, grid, xval + 8 * k, waux + 8 * k, scale, L + 8 * k);
                }
            }
            float sumqx = 0, sumq2 = 0;
            for (int i = 0; i < 16; ++i) {
                float w = weight[i];
                float q = 2.0f * (float)L[i] + 1.0f;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if (sumq2 > 0) scale = sumqx / sumq2;
        }

        if (scale < 0) {
            scale = -scale;
            for (int k = 0; k < 2; ++k) block_signs[k] = (~block_signs[k]) & 127;
        }
        for (int k = 0; k < 2; ++k) {
            uint16_t u = 0;
            for (int i = 0; i < 8; ++i) u |= ((uint16_t)L[8 * k + i] << (2 * i));
            int grid_index = map[u];
            if (grid_index < 0) { grid_index = 0; }
            q2[2 * ib + k] = (uint16_t)(grid_index | ((uint16_t)block_signs[k] << 9));
        }
        scales[ib] = scale;
        if (scale > max_scale) max_scale = scale;
    }

    if (!max_scale) {
        y->d = 0;
        for (int i = 0; i < QK_K / 8; ++i) y->qs[i] = 0;
        return;
    }
    float d = max_scale / 31.0f;
    y->d = fp32_to_fp16(d);
    float id = 1.0f / d;
    for (int ib = 0; ib < QK_K / 16; ++ib) {
        int l = nearest_int(0.5f * (id * scales[ib] - 1.0f));
        if (l < 0) l = 0;
        if (l > 15) l = 15;
        if (ib % 2 == 0) y->scales[ib / 2] = (uint8_t)l;
        else y->scales[ib / 2] |= (uint8_t)(l << 4);
    }
    uint8_t * qs = (uint8_t *)y->qs;
    const uint8_t * q2b = (const uint8_t *)q2;
    for (int i = 0; i < QK_K / 4; ++i) qs[i] = q2b[i];
}
