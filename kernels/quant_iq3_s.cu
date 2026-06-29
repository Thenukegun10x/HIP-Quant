#include "../hip_quant_types.h"
#include "../hip_quant_util.h"
#include <float.h>

#define IQ3S_BLOCK_SIZE 32

static __device__ int iq3s_find_best_neighbour(
    const uint16_t * neighbours, const int8_t * grid,
    const float * xval, const float * weight, float scale, int8_t * L
) {
    int num_neighbors = neighbours[0];
    float best_d2 = FLT_MAX;
    int best_idx = neighbours[1];
    for (int j = 1; j <= num_neighbors; ++j) {
        int idx = neighbours[j];
        const int8_t * pg = grid + 4 * idx;
        float d2 = 0;
        for (int i = 0; i < 4; ++i) {
            float q = (float)pg[i];
            float diff = scale * q - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if (d2 < best_d2) {
            best_d2 = d2;
            best_idx = idx;
        }
    }
    const int8_t * pg = grid + 4 * best_idx;
    for (int i = 0; i < 4; ++i) L[i] = (pg[i] - 1) / 2;
    return best_idx;
}

__global__ void quantize_iq3_s_kernel(
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
    block_iq3_s * y = (block_iq3_s *)(dst + (row * nbl + ibl) * sizeof(block_iq3_s));

    y->d = 0;
    for (int i = 0; i < QK_K / 4; ++i) y->qs[i] = 0;
    for (int i = 0; i < QK_K / 32; ++i) y->qh[i] = 0;
    for (int i = 0; i < QK_K / 8; ++i) y->signs[i] = 0;
    for (int i = 0; i < IQ3S_N_SCALE; ++i) y->scales[i] = 0;
    y->qs[0] = 123;

    float scales[QK_K / IQ3S_BLOCK_SIZE];
    float max_scale = 0;
    float sumx2 = 0;
    for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i] * xbl[i];
    float sigma2 = 2.0f * sumx2 / QK_K;

    int qs_pos = 0;
    int signs_pos = 0;
    for (int ib = 0; ib < QK_K / IQ3S_BLOCK_SIZE; ++ib) {
        const float * xb = xbl + IQ3S_BLOCK_SIZE * ib;
        const float * qw = imatrix ? imatrix + row * n_per_row + QK_K * ibl + IQ3S_BLOCK_SIZE * ib : NULL;
        float weight[IQ3S_BLOCK_SIZE], waux[IQ3S_BLOCK_SIZE], xval[IQ3S_BLOCK_SIZE];
        int8_t L[IQ3S_BLOCK_SIZE], Laux[IQ3S_BLOCK_SIZE];
        bool is_on_grid[IQ3S_BLOCK_SIZE / 4], is_on_grid_aux[IQ3S_BLOCK_SIZE / 4];
        uint8_t block_signs[IQ3S_BLOCK_SIZE / 8];

        for (int i = 0; i < IQ3S_BLOCK_SIZE; ++i) {
            float wi = qw ? qw[i] * sqrtf(sigma2 + xb[i] * xb[i]) : xb[i] * xb[i];
            weight[i] = wi;
            waux[i] = sqrtf(wi);
        }
        for (int k = 0; k < IQ3S_BLOCK_SIZE / 8; ++k) {
            uint8_t s = 0;
            for (int i = 0; i < 8; ++i) {
                if (xb[8 * k + i] >= 0) xval[8 * k + i] = xb[8 * k + i];
                else {
                    xval[8 * k + i] = -xb[8 * k + i];
                    s |= (1u << i);
                }
            }
            block_signs[k] = s;
        }

        float max_v = xval[0];
        for (int i = 1; i < IQ3S_BLOCK_SIZE; ++i) if (xval[i] > max_v) max_v = xval[i];
        for (int i = 0; i < IQ3S_BLOCK_SIZE; ++i) L[i] = 0;
        if (max_v == 0.0f) { scales[ib] = 0; continue; }

        float best = 0;
        float scale = max_v / 15.0f;
        for (int k = 0; k < IQ3S_BLOCK_SIZE / 4; ++k) is_on_grid[k] = false;
        for (int is = -9; is <= 9; ++is) {
            float id = (15.0f + (float)is * 0.2f) / max_v;
            float this_scale = 1.0f / id;
            for (int k = 0; k < IQ3S_BLOCK_SIZE / 4; ++k) {
                for (int i = 0; i < 4; ++i) {
                    int l = nearest_int(0.5f * (id * xval[4 * k + i] - 1.0f));
                    if (l < 0) l = 0;
                    if (l > 7) l = 7;
                    Laux[4 * k + i] = (int8_t)l;
                }
                uint16_t u = 0;
                for (int i = 0; i < 4; ++i) u |= ((uint16_t)Laux[4 * k + i] << (3 * i));
                int grid_index = map[u];
                is_on_grid_aux[k] = true;
                if (grid_index < 0) {
                    is_on_grid_aux[k] = false;
                    const uint16_t * neighbours = neighbours_data + (-grid_index - 1);
                    grid_index = iq3s_find_best_neighbour(neighbours, grid, xval + 4 * k, waux + 4 * k, this_scale, Laux + 4 * k);
                }
            }
            float sumqx = 0, sumq2 = 0;
            for (int i = 0; i < IQ3S_BLOCK_SIZE; ++i) {
                float w = weight[i];
                float q = 2.0f * (float)Laux[i] + 1.0f;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
                scale = sumqx / sumq2;
                best = scale * sumqx;
                for (int i = 0; i < IQ3S_BLOCK_SIZE; ++i) L[i] = Laux[i];
                for (int k = 0; k < IQ3S_BLOCK_SIZE / 4; ++k) is_on_grid[k] = is_on_grid_aux[k];
            }
        }

        int n_not_ongrid = 0;
        for (int k = 0; k < IQ3S_BLOCK_SIZE / 4; ++k) if (!is_on_grid[k]) ++n_not_ongrid;
        if (n_not_ongrid > 0 && scale > 0) {
            float id = 1.0f / scale;
            for (int k = 0; k < IQ3S_BLOCK_SIZE / 4; ++k) {
                uint16_t u = 0;
                for (int i = 0; i < 4; ++i) {
                    int l = nearest_int(0.5f * (id * xval[4 * k + i] - 1.0f));
                    if (l < 0) l = 0;
                    if (l > 7) l = 7;
                    u |= ((uint16_t)l << (3 * i));
                }
                int grid_index = map[u];
                if (grid_index < 0) {
                    const uint16_t * neighbours = neighbours_data + (-grid_index - 1);
                    grid_index = iq3s_find_best_neighbour(neighbours, grid, xval + 4 * k, waux + 4 * k, scale, L + 4 * k);
                }
                const int8_t * pg = grid + 4 * grid_index;
                for (int i = 0; i < 4; ++i) L[4 * k + i] = (pg[i] - 1) / 2;
            }
            float sumqx = 0, sumq2 = 0;
            for (int i = 0; i < IQ3S_BLOCK_SIZE; ++i) {
                float w = weight[i];
                float q = 2.0f * (float)L[i] + 1.0f;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if (sumq2 > 0) scale = sumqx / sumq2;
        }

        if (scale < 0) {
            scale = -scale;
            for (int k = 0; k < IQ3S_BLOCK_SIZE / 8; ++k) block_signs[k] = ~block_signs[k];
        }

        for (int k = 0; k < IQ3S_BLOCK_SIZE / 4; ++k) {
            uint16_t u = 0;
            for (int i = 0; i < 4; ++i) u |= ((uint16_t)L[4 * k + i] << (3 * i));
            int grid_index = map[u];
            if (grid_index < 0) { printf("error at type 21: map miss for u=%u\n", (unsigned)u); grid_index = 0; }
            y->qs[qs_pos + k] = (uint8_t)(grid_index & 255);
            y->qh[(ib * (IQ3S_BLOCK_SIZE / 4) + k) / 8] |= (uint8_t)((grid_index >> 8) << ((ib * (IQ3S_BLOCK_SIZE / 4) + k) % 8));
        }
        qs_pos += IQ3S_BLOCK_SIZE / 4;
        for (int k = 0; k < IQ3S_BLOCK_SIZE / 8; ++k) y->signs[signs_pos + k] = block_signs[k];
        signs_pos += IQ3S_BLOCK_SIZE / 8;
        scales[ib] = scale;
        if (scale > max_scale) max_scale = scale;
    }

    if (!max_scale) return;
    float d = max_scale / 31.0f;
    y->d = fp32_to_fp16(d * 1.033f);
    float id = 1.0f / d;
    for (int ib = 0; ib < QK_K / IQ3S_BLOCK_SIZE; ib += 2) {
        int l1 = nearest_int(0.5f * (id * scales[ib + 0] - 1.0f));
        if (l1 < 0) l1 = 0;
        if (l1 > 15) l1 = 15;
        int l2 = nearest_int(0.5f * (id * scales[ib + 1] - 1.0f));
        if (l2 < 0) l2 = 0;
        if (l2 > 15) l2 = 15;
        y->scales[ib / 2] = (uint8_t)(l1 | (l2 << 4));
    }
}
