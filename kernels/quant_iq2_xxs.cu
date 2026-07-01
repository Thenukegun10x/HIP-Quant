#include "../hip_quant_types.h"
#include "../hip_quant_util.h"
#include <float.h>

#define GROUP_MAX_EPS 1e-15f
#define IQ2_XXS_KMAP_SIZE 43692

__constant__ int8_t d_iq2xxs_grid[256][8];

static __device__ float make_qp_quants_iq2xxs(
    int n, int nmax, const float * x, uint8_t * L, const float * quant_weights
) {
    float max_v = 0;
    for (int i = 0; i < n; ++i) {
        if (x[i] > max_v) max_v = x[i];
    }
    if (max_v < GROUP_MAX_EPS) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        return 0.0f;
    }
    float iscale = (float)nmax / max_v;
    for (int i = 0; i < n; ++i) {
        L[i] = (uint8_t)nearest_int(iscale * x[i]);
    }
    float scale = 1.0f / iscale;
    float best_mse = 0;
    for (int i = 0; i < n; ++i) {
        float diff = x[i] - scale * (float)L[i];
        float w = quant_weights[i];
        best_mse += w * diff * diff;
    }
    for (int is = -4; is <= 4; ++is) {
        if (is == 0) continue;
        float iscale_is = (0.1f * (float)is + (float)nmax) / max_v;
        float scale_is = 1.0f / iscale_is;
        float mse = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale_is * x[i]);
            if (l > nmax) l = nmax;
            float diff = x[i] - scale_is * (float)l;
            float w = quant_weights[i];
            mse += w * diff * diff;
        }
        if (mse < best_mse) {
            best_mse = mse;
            iscale = iscale_is;
        }
    }
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        if (l > nmax) l = nmax;
        L[i] = (uint8_t)l;
        float w = quant_weights[i];
        sumlx += w * x[i] * (float)l;
        suml2 += w * (float)l * (float)l;
    }
    for (int itry = 0; itry < 5; ++itry) {
        int n_changed = 0;
        for (int i = 0; i < n; ++i) {
            float w = quant_weights[i];
            float li = (float)L[i];
            float slx = sumlx - w * x[i] * li;
            float sl2 = suml2 - w * li * li;
            if (slx > 0 && sl2 > 0) {
                int new_l = nearest_int(x[i] * sl2 / slx);
                if (new_l > nmax) new_l = nmax;
                if (new_l != L[i]) {
                    slx += w * x[i] * (float)new_l;
                    sl2 += w * (float)new_l * (float)new_l;
                    if (slx * slx * suml2 > sumlx * sumlx * sl2) {
                        L[i] = (uint8_t)new_l;
                        sumlx = slx;
                        suml2 = sl2;
                        ++n_changed;
                    }
                }
            }
        }
        if (!n_changed) break;
    }
    return suml2 > 0.0f ? sumlx / suml2 : 0.0f;
}

static __device__ int iq2xxs_find_best_neighbour(
    const uint16_t * neighbours,
    const float * xval, const float * weight, float scale, int8_t * L
) {
    int num_neighbors = neighbours[0];
    float best_d2 = FLT_MAX;
    int best_idx = neighbours[1];
    for (int j = 1; j <= num_neighbors; ++j) {
        int idx = neighbours[j];
        const int8_t * pg = d_iq2xxs_grid[idx];
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
    const int8_t * pg = d_iq2xxs_grid[best_idx];
    for (int i = 0; i < 8; ++i) L[i] = (pg[i] - 1) / 2;
    return best_idx;
}

__global__ void quantize_iq2_xxs_kernel(
    const float * __restrict__ src,
    uint8_t * __restrict__ dst,
    const float * __restrict__ imatrix,
    const int * __restrict__ iq2xxs_map,
    const uint16_t * __restrict__ iq2xxs_neighbours,
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
    block_iq2_xxs * y = (block_iq2_xxs *)(dst + (row * nbl + ibl) * sizeof(block_iq2_xxs));

    float scales[QK_K / 32];
    uint32_t q2[2 * (QK_K / 32)];
    for (int i = 0; i < 2 * (QK_K / 32); ++i) q2[i] = 0;

    float max_scale = 0;
    float sumx2 = 0;
    for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i] * xbl[i];
    float sigma2 = sumx2 / QK_K;

    for (int ib = 0; ib < QK_K / 32; ++ib) {
        const float * xb = xbl + 32 * ib;
        const float * qw = imatrix ? imatrix + row * n_per_row + QK_K * ibl + 32 * ib : NULL;
        float weight[32], waux[32], xval[32];
        int8_t L[32], Laux[32];
        uint8_t Lu[32];
        uint8_t block_signs[4];

        for (int i = 0; i < 32; ++i) {
            float wi = qw ? qw[i] * sqrtf(sigma2 + xb[i] * xb[i]) : xb[i] * xb[i];
            weight[i] = wi;
            waux[i] = sqrtf(wi);
        }

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
                float min_v = weight[8 * k] * xb[8 * k] * xb[8 * k];
                for (int i = 1; i < 8; ++i) {
                    float ax = weight[8 * k + i] * xb[8 * k + i] * xb[8 * k + i];
                    if (ax < min_v) {
                        min_v = ax;
                        imin = i;
                    }
                }
                xval[8 * k + imin] = -xval[8 * k + imin];
                s ^= (1u << imin);
            }
            block_signs[k] = s & 127;
        }

        float max_v = xval[0];
        for (int i = 1; i < 32; ++i) if (xval[i] > max_v) max_v = xval[i];
        if (max_v < GROUP_MAX_EPS) {
            scales[ib] = 0;
            for (int i = 0; i < 32; ++i) L[i] = 0;
            continue;
        }

        float scale = make_qp_quants_iq2xxs(32, 4, xval, Lu, weight);
        for (int i = 0; i < 32; ++i) L[i] = (int8_t)Lu[i];
        float eff_max = scale * 3.0f;
        if (eff_max <= 0) {
            scales[ib] = 0;
            for (int i = 0; i < 32; ++i) L[i] = 0;
            continue;
        }

        float best = 0;
        for (int is = -6; is <= 6; ++is) {
            float id = (5.0f + (float)is * 0.1f) / eff_max;
            float this_scale = 1.0f / id;
            for (int k = 0; k < 4; ++k) {
                for (int i = 0; i < 8; ++i) {
                    int l = nearest_int(0.5f * (id * xval[8 * k + i] - 1.0f));
                    if (l < 0) l = 0;
                    if (l > 2) l = 2;
                    Laux[8 * k + i] = (int8_t)l;
                }
                uint16_t u = 0;
                for (int i = 0; i < 8; ++i) u |= ((uint16_t)Laux[8 * k + i] << (2 * i));
                int grid_index = iq2xxs_map[u];
                if (grid_index < 0) {
                    const uint16_t * neighbours = iq2xxs_neighbours + (-grid_index - 1);
                    grid_index = iq2xxs_find_best_neighbour(neighbours, xval + 8 * k, waux + 8 * k, this_scale, Laux + 8 * k);
                }
            }
            float sumqx = 0, sumq2 = 0;
            for (int i = 0; i < 32; ++i) {
                float w = weight[i];
                float q = 2.0f * (float)Laux[i] + 1.0f;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
                scale = sumqx / sumq2;
                best = scale * sumqx;
                for (int i = 0; i < 32; ++i) L[i] = Laux[i];
            }
        }

        if (scale > 0) {
            float id = 1.0f / scale;
            for (int k = 0; k < 4; ++k) {
                uint16_t u = 0;
                for (int i = 0; i < 8; ++i) {
                    int l = nearest_int(0.5f * (id * xval[8 * k + i] - 1.0f));
                    if (l < 0) l = 0;
                    if (l > 2) l = 2;
                    u |= ((uint16_t)l << (2 * i));
                }
                int grid_index = iq2xxs_map[u];
                if (grid_index < 0) {
                    const uint16_t * neighbours = iq2xxs_neighbours + (-grid_index - 1);
                    grid_index = iq2xxs_find_best_neighbour(neighbours, xval + 8 * k, waux + 8 * k, scale, L + 8 * k);
                }
                const int8_t * pg = d_iq2xxs_grid[grid_index];
                for (int i = 0; i < 8; ++i) L[8 * k + i] = (pg[i] - 1) / 2;
            }
            float sumqx = 0, sumq2 = 0;
            for (int i = 0; i < 32; ++i) {
                float w = weight[i];
                float q = 2.0f * (float)L[i] + 1.0f;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if (sumq2 > 0) scale = sumqx / sumq2;
        }

        if (scale < 0) {
            scale = -scale;
            for (int k = 0; k < 4; ++k) block_signs[k] = (~block_signs[k]) & 127;
        }

        for (int k = 0; k < 4; ++k) {
            uint16_t u = 0;
            for (int i = 0; i < 8; ++i) u |= ((uint16_t)L[8 * k + i] << (2 * i));
            int grid_index = iq2xxs_map[u];
            if (grid_index < 0) {
                grid_index = 0;
            }
            q2[2 * ib + 0] |= ((uint32_t)grid_index << (8 * k));
            q2[2 * ib + 1] |= ((uint32_t)block_signs[k] << (7 * k));
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
    for (int ib = 0; ib < QK_K / 32; ++ib) {
        int l = nearest_int(0.5f * (id * scales[ib] - 1.0f));
        if (l < 0) l = 0;
        if (l > 15) l = 15;
        q2[2 * ib + 1] |= ((uint32_t)l << 28);
    }

    uint8_t * qs = (uint8_t *)y->qs;
    const uint8_t * src_q2 = (const uint8_t *)q2;
    for (int i = 0; i < QK_K / 4; ++i) qs[i] = src_q2[i];
}
