#include "../hip_quant_types.h"
#include "../hip_quant_util.h"
#include "../hip_iquant_util.h"
#include <float.h>

#define GROUP_MAX_EPS_IQ3_XXS 1e-8f
#define KMAX_Q 8

// GPU constant tables (uploaded by host)
__constant__ int8_t d_iq3xxs_grid[256][4];
__constant__ int d_iq3xxs_map[4096];
__constant__ uint16_t d_iq3xxs_neighbours[22825];

static __device__ int iq3xxs_find_best_neighbour(
    const uint16_t * neighbours, int grid_index,
    const float * xval, const float * weight, float scale, int8_t * L
) {
    int num_neighbors = neighbours[0];
    float best_d2 = FLT_MAX;
    int best_idx = neighbours[1];
    for (int j = 1; j <= num_neighbors; ++j) {
        int idx = neighbours[j];
        const int8_t * pg = d_iq3xxs_grid[idx];
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
    const int8_t * pg = d_iq3xxs_grid[best_idx];
    for (int i = 0; i < 4; ++i) L[i] = (pg[i] - 1) / 2;
    return best_idx;
}

__global__ void quantize_iq3_xxs_kernel(
    const float * __restrict__ src,
    uint8_t * __restrict__ dst,
    const float * __restrict__ imatrix,
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
    block_iq3_xxs * y = (block_iq3_xxs *)(dst + (row * nbl + ibl) * sizeof(block_iq3_xxs));

    float scales[8];
    uint32_t scales_and_signs_arr[8];
    uint8_t q3[3 * (QK_K / 8) + QK_K / 32];

        float max_scale = 0;
        for (int i = 0; i < 3 * (QK_K / 8) + QK_K / 32; ++i) q3[i] = 0;

        float sumx2 = 0;
        for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i] * xbl[i];
        float sigma2 = 2.0f * sumx2 / QK_K;

        for (int ib = 0; ib < QK_K / 32; ++ib) {
            const float * xb = xbl + 32 * ib;
            float weight[32], waux[32], xval[32];
            int8_t L[32], Laux[32];
            bool is_on_grid[8], is_on_grid_aux[8];
            uint8_t block_signs[4];

            if (imatrix) {
                const float * qw = imatrix + row * n_per_row + QK_K * ibl + 32 * ib;
                for (int i = 0; i < 32; ++i)
                    weight[i] = qw[i] * sqrtf(sigma2 + xb[i] * xb[i]);
            } else {
                for (int i = 0; i < 32; ++i)
                    weight[i] = xb[i] * xb[i];
            }
            for (int i = 0; i < 32; ++i) waux[i] = sqrtf(weight[i]);

            for (int k = 0; k < 4; ++k) {
                int nflip = 0;
                uint8_t s = 0;
                for (int i = 0; i < 8; ++i) {
                    if (xb[8 * k + i] >= 0) {
                        xval[8 * k + i] = xb[8 * k + i];
                    } else {
                        xval[8 * k + i] = -xb[8 * k + i];
                        ++nflip;
                        s |= (1u << i);
                    }
                }
                if (nflip % 2) {
                    int imin = 0;
                    float min = weight[8 * k + 0] * xb[8 * k + 0] * xb[8 * k + 0];
                    for (int i = 1; i < 8; ++i) {
                        float ax = weight[8 * k + i] * xb[8 * k + i] * xb[8 * k + i];
                        if (ax < min) {
                            min = ax;
                            imin = i;
                        }
                    }
                    xval[8 * k + imin] = -xval[8 * k + imin];
                    s ^= (1u << imin);
                }
                block_signs[k] = s & 127;
            }

            float max_val = xval[0];
            for (int i = 1; i < 32; ++i)
                if (xval[i] > max_val) max_val = xval[i];

            if (max_val < GROUP_MAX_EPS_IQ3_XXS) {
                scales[ib] = 0;
                scales_and_signs_arr[ib] = 0;
                for (int i = 0; i < 32; ++i) L[i] = 0;
                continue;
            }

            float best = 0;
            float scale = max_val / (2.0f * KMAX_Q - 1.0f);
            for (int k = 0; k < 8; ++k) is_on_grid[k] = true;

            for (int is = -15; is <= 15; ++is) {
                float id = (2.0f * KMAX_Q - 1.0f + (float)is * 0.2f) / max_val;
                float this_scale = 1.0f / id;

                for (int k = 0; k < 8; ++k) {
                    for (int i = 0; i < 4; ++i) {
                        int l = nearest_int(0.5f * (id * xval[4 * k + i] - 1.0f));
                        if (l < 0) l = 0;
                        if (l > KMAX_Q - 1) l = KMAX_Q - 1;
                        Laux[4 * k + i] = (int8_t)l;
                    }
                    uint16_t u = 0;
                    for (int i = 0; i < 4; ++i)
                        u |= ((uint16_t)Laux[4 * k + i] << (3 * i));
                    int grid_index = d_iq3xxs_map[u];
                    is_on_grid_aux[k] = (grid_index >= 0);
                    if (grid_index < 0) {
                        const uint16_t * neighbours = d_iq3xxs_neighbours + (-grid_index - 1);
                        grid_index = iq3xxs_find_best_neighbour(
                            neighbours, grid_index,
                            xval + 4 * k, waux + 4 * k, this_scale, Laux + 4 * k);
                    }
                }

                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 32; ++i) {
                    float w = weight[i];
                    float qv = 2.0f * (float)Laux[i] + 1.0f;
                    sumqx += w * xval[i] * qv;
                    sumq2 += w * qv * qv;
                }
                if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
                    scale = sumqx / sumq2;
                    best = scale * sumqx;
                    for (int i = 0; i < 32; ++i) L[i] = Laux[i];
                    for (int k = 0; k < 8; ++k) is_on_grid[k] = is_on_grid_aux[k];
                }
            }

            {
                int n_not_ongrid = 0;
                for (int k = 0; k < 8; ++k)
                    if (!is_on_grid[k]) ++n_not_ongrid;
                if (n_not_ongrid > 0 && scale > 0) {
                    float id = 1.0f / scale;
                    for (int k = 0; k < 8; ++k) {
                        if (is_on_grid[k]) continue;
                        uint16_t u = 0;
                        for (int i = 0; i < 4; ++i) {
                            int l = nearest_int(0.5f * (id * xval[4 * k + i] - 1.0f));
                            if (l < 0) l = 0;
                            if (l > KMAX_Q - 1) l = KMAX_Q - 1;
                            u |= (l << (3 * i));
                        }
                        int grid_index = d_iq3xxs_map[u];
                        if (grid_index < 0) {
                            const uint16_t * neighbours = d_iq3xxs_neighbours + (-grid_index - 1);
                            grid_index = iq3xxs_find_best_neighbour(
                                neighbours, grid_index,
                                xval + 4 * k, waux + 4 * k, scale, L + 4 * k);
                        }
                        const int8_t * pg = d_iq3xxs_grid[grid_index];
                        for (int i = 0; i < 4; ++i)
                            L[4 * k + i] = (pg[i] - 1) / 2;
                    }
                    float sumqx = 0, sumq2 = 0;
                    for (int i = 0; i < 32; ++i) {
                        float w = weight[i];
                        float qv = 2.0f * (float)L[i] + 1.0f;
                        sumqx += w * xval[i] * qv;
                        sumq2 += w * qv * qv;
                    }
                    if (sumq2 > 0) scale = sumqx / sumq2;
                }
            }

            if (scale < 0) {
                scale = -scale;
                for (int k = 0; k < 4; ++k)
                    block_signs[k] = (~block_signs[k]) & 127;
            }

            for (int k = 0; k < 8; ++k) {
                uint16_t u = 0;
                for (int i = 0; i < 4; ++i)
                    u |= ((uint16_t)L[4 * k + i] << (3 * i));
                int grid_index = d_iq3xxs_map[u];
                if (grid_index < 0) {
                    printf("error at type 18: map miss for u=%u\n", (unsigned)u);
                    grid_index = 0;
                }
                q3[8 * ib + k] = (uint8_t)grid_index;
            }

            scales_and_signs_arr[ib] = (uint32_t)block_signs[0]
                | ((uint32_t)block_signs[1] << 7)
                | ((uint32_t)block_signs[2] << 14)
                | ((uint32_t)block_signs[3] << 21);
            scales[ib] = scale;
            if (scale > max_scale) max_scale = scale;

        }

        if (max_scale == 0) {
            y->d = 0;
            for (int i = 0; i < 3 * QK_K / 8; ++i) q3[i] = 0;
        } else {
            float d = max_scale / 31.0f;
            y->d = fp32_to_fp16(d * 1.0125f);
            float id = 1.0f / d;
            for (int ib = 0; ib < QK_K / 32; ++ib) {
                int l = nearest_int(0.5f * (id * scales[ib] - 1.0f));
                if (l < 0) l = 0;
                if (l > 15) l = 15;
                scales_and_signs_arr[ib] |= ((uint32_t)l << 28);
            }
        }

        // Pack q3 + scales_and_signs into qs
        uint8_t * qs = y->qs;
        for (int i = 0; i < QK_K / 4; ++i) qs[i] = q3[i];
        for (int i = 0; i < QK_K / 32; ++i) {
            ((uint32_t *)(qs + QK_K / 4))[i] = scales_and_signs_arr[i];
        }
}
