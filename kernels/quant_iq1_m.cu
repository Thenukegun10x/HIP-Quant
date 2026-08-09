#include "../hip_quant_types.h"
#include "../hip_quant_util.h"
#include <float.h>

#define GROUP_MAX_EPS_IQ1_M 1e-7f
#define IQ1M_BLOCK_SIZE 16
#define IQ1M_DELTA 0.125f
#define NGRID_IQ1M 2048

static __device__ int iq1m_find_best_neighbour2(
    const uint16_t * neighbours, const int8_t * grid,
    const float * xval, const float * weight, float scale,
    const float * xg, int8_t * L
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

__global__ void quantize_iq1_m_kernel(
    const float * __restrict__ src,
    uint8_t * __restrict__ dst,
    const float * __restrict__ imatrix,
    const int8_t * __restrict__ grid,
    const int * __restrict__ map,
    const uint16_t * __restrict__ neighbours_data,
    int nrows,
    int n_per_row
) {
    const int n_sub_blocks = QK_K / IQ1M_BLOCK_SIZE;
    int row = blockIdx.x;
    int ibl = blockIdx.y;
    int ib = threadIdx.x;
    if (row >= nrows) return;
    int nbl = n_per_row / QK_K;
    if (ibl >= nbl) return;
    if (ib >= n_sub_blocks) return;

    const float * xbl = src + row * n_per_row + QK_K * ibl;
    block_iq1_m * y = (block_iq1_m *)(dst + (row * nbl + ibl) * sizeof(block_iq1_m));

    __shared__ float s_scales[QK_K / IQ1M_BLOCK_SIZE];
    __shared__ int8_t s_shifts[QK_K / IQ1M_BLOCK_SIZE];
    __shared__ uint8_t s_qs[QK_K / 8];
    __shared__ uint8_t s_qh[QK_K / 16];

    for (int i = threadIdx.x; i < QK_K / 8; i += n_sub_blocks) s_qs[i] = 0;
    for (int i = threadIdx.x; i < QK_K / 16; i += n_sub_blocks) s_qh[i] = 0;

    float local_scale;
    int8_t local_shift;
    const float x_p[3] = {-1.0f + IQ1M_DELTA, IQ1M_DELTA, 1.0f + IQ1M_DELTA};
    const float x_m[3] = {-1.0f - IQ1M_DELTA, -IQ1M_DELTA, 1.0f - IQ1M_DELTA};
    const uint8_t masks[4] = {0x00, 0x80, 0x08, 0x88};
    float sumx2 = 0;
    for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i] * xbl[i];
    float sigma2 = 2.0f * sumx2 / QK_K;

    {
        const float * xb = xbl + IQ1M_BLOCK_SIZE * ib;
        const float * qw = imatrix ? imatrix + row * n_per_row + QK_K * ibl + IQ1M_BLOCK_SIZE * ib : NULL;
        float weight[IQ1M_BLOCK_SIZE];
        int8_t L[IQ1M_BLOCK_SIZE];
        uint16_t index[IQ1M_BLOCK_SIZE / 8];
        float sumqx[4], sumq2[4];
        float vals[IQ1M_BLOCK_SIZE];
        int idx[IQ1M_BLOCK_SIZE];

        for (int i = 0; i < IQ1M_BLOCK_SIZE; ++i) {
            float wi = qw ? qw[i] * sqrtf(sigma2 + xb[i] * xb[i]) : xb[i] * xb[i];
            weight[i] = wi;
        }

        float max_v = fabsf(xb[0]);
        for (int i = 1; i < IQ1M_BLOCK_SIZE; ++i) {
            float ax = fabsf(xb[i]);
            if (ax > max_v) max_v = ax;
        }
        if (max_v < GROUP_MAX_EPS_IQ1_M) {
            local_scale = 0;
            local_shift = 0;
        } else {
            for (int j = 0; j < IQ1M_BLOCK_SIZE; ++j) {
                vals[j] = xb[j];
                idx[j] = j;
            }
            for (int a = 1; a < IQ1M_BLOCK_SIZE; ++a) {
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

            float best_score = -FLT_MAX;
            float scale = max_v;
            int besti1 = -1, besti2 = -1, best_k = -1;
            for (int i1 = 0; i1 <= IQ1M_BLOCK_SIZE; ++i1) {
                for (int i2 = i1; i2 <= IQ1M_BLOCK_SIZE; ++i2) {
                    for (int k = 0; k < 4; ++k) sumqx[k] = sumq2[k] = 0;
                    for (int j = 0; j < i1; ++j) {
                        int i = idx[j];
                        if (i < IQ1M_BLOCK_SIZE / 2) {
                            sumqx[0] += weight[i] * x_p[0] * xb[i]; sumqx[1] += weight[i] * x_p[0] * xb[i];
                            sumqx[2] += weight[i] * x_m[0] * xb[i]; sumqx[3] += weight[i] * x_m[0] * xb[i];
                            sumq2[0] += weight[i] * x_p[0] * x_p[0]; sumq2[1] += weight[i] * x_p[0] * x_p[0];
                            sumq2[2] += weight[i] * x_m[0] * x_m[0]; sumq2[3] += weight[i] * x_m[0] * x_m[0];
                        } else {
                            sumqx[0] += weight[i] * x_p[0] * xb[i]; sumqx[2] += weight[i] * x_p[0] * xb[i];
                            sumqx[1] += weight[i] * x_m[0] * xb[i]; sumqx[3] += weight[i] * x_m[0] * xb[i];
                            sumq2[0] += weight[i] * x_p[0] * x_p[0]; sumq2[2] += weight[i] * x_p[0] * x_p[0];
                            sumq2[1] += weight[i] * x_m[0] * x_m[0]; sumq2[3] += weight[i] * x_m[0] * x_m[0];
                        }
                    }
                    for (int j = i1; j < i2; ++j) {
                        int i = idx[j];
                        if (i < IQ1M_BLOCK_SIZE / 2) {
                            sumqx[0] += weight[i] * x_p[1] * xb[i]; sumqx[1] += weight[i] * x_p[1] * xb[i];
                            sumqx[2] += weight[i] * x_m[1] * xb[i]; sumqx[3] += weight[i] * x_m[1] * xb[i];
                            sumq2[0] += weight[i] * x_p[1] * x_p[1]; sumq2[1] += weight[i] * x_p[1] * x_p[1];
                            sumq2[2] += weight[i] * x_m[1] * x_m[1]; sumq2[3] += weight[i] * x_m[1] * x_m[1];
                        } else {
                            sumqx[0] += weight[i] * x_p[1] * xb[i]; sumqx[2] += weight[i] * x_p[1] * xb[i];
                            sumqx[1] += weight[i] * x_m[1] * xb[i]; sumqx[3] += weight[i] * x_m[1] * xb[i];
                            sumq2[0] += weight[i] * x_p[1] * x_p[1]; sumq2[2] += weight[i] * x_p[1] * x_p[1];
                            sumq2[1] += weight[i] * x_m[1] * x_m[1]; sumq2[3] += weight[i] * x_m[1] * x_m[1];
                        }
                    }
                    for (int j = i2; j < IQ1M_BLOCK_SIZE; ++j) {
                        int i = idx[j];
                        if (i < IQ1M_BLOCK_SIZE / 2) {
                            sumqx[0] += weight[i] * x_p[2] * xb[i]; sumqx[1] += weight[i] * x_p[2] * xb[i];
                            sumqx[2] += weight[i] * x_m[2] * xb[i]; sumqx[3] += weight[i] * x_m[2] * xb[i];
                            sumq2[0] += weight[i] * x_p[2] * x_p[2]; sumq2[1] += weight[i] * x_p[2] * x_p[2];
                            sumq2[2] += weight[i] * x_m[2] * x_m[2]; sumq2[3] += weight[i] * x_m[2] * x_m[2];
                        } else {
                            sumqx[0] += weight[i] * x_p[2] * xb[i]; sumqx[2] += weight[i] * x_p[2] * xb[i];
                            sumqx[1] += weight[i] * x_m[2] * xb[i]; sumqx[3] += weight[i] * x_m[2] * xb[i];
                            sumq2[0] += weight[i] * x_p[2] * x_p[2]; sumq2[2] += weight[i] * x_p[2] * x_p[2];
                            sumq2[1] += weight[i] * x_m[2] * x_m[2]; sumq2[3] += weight[i] * x_m[2] * x_m[2];
                        }
                    }
                    for (int k = 0; k < 4; ++k) {
                        if (sumq2[k] > 0 && sumqx[k] * sumqx[k] > best_score * sumq2[k]) {
                            scale = sumqx[k] / sumq2[k];
                            best_score = scale * sumqx[k];
                            besti1 = i1;
                            besti2 = i2;
                            best_k = k;
                        }
                    }
                }
            }
            if (besti1 < 0 || besti2 < 0 || best_k < 0) {
                local_scale = 0;
                local_shift = 0;
            } else {
                for (int j = 0; j < besti1; ++j) L[idx[j]] = 0;
                for (int j = besti1; j < besti2; ++j) L[idx[j]] = 1;
                for (int j = besti2; j < IQ1M_BLOCK_SIZE; ++j) L[idx[j]] = 2;
                if (scale < 0) {
                    for (int j = 0; j < IQ1M_BLOCK_SIZE; ++j) L[j] = 2 - L[j];
                    scale = -scale;
                    best_k = best_k == 0 ? 3 : best_k == 1 ? 2 : best_k == 2 ? 1 : 0;
                }

                bool all_on_grid = true;
                const float * xx;
                for (int k = 0; k < IQ1M_BLOCK_SIZE / 8; ++k) {
                    if (k == 0) xx = best_k < 2 ? x_p : x_m;
                    else xx = best_k % 2 == 0 ? x_p : x_m;
                    uint16_t u = 0;
                    for (int j = 0; j < 8; ++j) u |= ((uint16_t)L[8 * k + j] << (2 * j));
                    int grid_index = map[u];
                    if (grid_index < 0) {
                        all_on_grid = false;
                        const uint16_t * neighbours = neighbours_data + (-grid_index - 1);
                        grid_index = iq1m_find_best_neighbour2(neighbours, grid, xb + 8 * k, weight + 8 * k, scale, xx, L + 8 * k);
                    }
                    index[k] = (uint16_t)grid_index;
                }
                if (!all_on_grid) {
                    float sumqx_f = 0, sumq2_f = 0;
                    for (int k = 0; k < IQ1M_BLOCK_SIZE / 8; ++k) {
                        if (k == 0) xx = best_k < 2 ? x_p : x_m;
                        else xx = best_k % 2 == 0 ? x_p : x_m;
                        const int8_t * pg = grid + 8 * index[k];
                        for (int j = 0; j < 8; ++j) {
                            float w = weight[8 * k + j];
                            float q = xx[(pg[j] - 1) / 2];
                            sumqx_f += w * q * xb[8 * k + j];
                            sumq2_f += w * q * q;
                        }
                    }
                    if (sumqx_f > 0 && sumq2_f > 0) scale = sumqx_f / sumq2_f;
                }

                s_qs[2 * ib + 0] = (uint8_t)(index[0] & 255);
                s_qs[2 * ib + 1] = (uint8_t)(index[1] & 255);
                s_qh[ib] = (uint8_t)((index[0] >> 8) | ((index[1] >> 8) << 4));
                local_scale = scale;
                local_shift = (int8_t)best_k;
            }
        }
    }

    s_scales[ib] = local_scale;
    s_shifts[ib] = local_shift;
    __syncthreads();

    if (threadIdx.x == 0) {
        float max_scale = 0;
        for (int j = 0; j < n_sub_blocks; ++j)
            if (s_scales[j] > max_scale) max_scale = s_scales[j];

        if (max_scale == 0) {
            memset(y, 0, sizeof(block_iq1_m));
        } else {
            memset(y, 0, sizeof(block_iq1_m));
            uint16_t * sc = (uint16_t *)y->scales;
            float d = max_scale / 15.0f;
            float id = 1.0f / d;
            for (int j = 0; j < n_sub_blocks; ++j) {
                int l = nearest_int(0.5f * (id * s_scales[j] - 1.0f));
                if (l < 0) l = 0;
                if (l > 7) l = 7;
                sc[j / 4] |= (uint16_t)((uint16_t)l << (3 * (j % 4)));
                s_qh[j] |= masks[s_shifts[j]];
            }
            float sumqx_f = 0, sumq2_f = 0;
            for (int ib2 = 0; ib2 < n_sub_blocks; ++ib2) {
                const float * xb = xbl + IQ1M_BLOCK_SIZE * ib2;
                const float * qw2 = imatrix ? imatrix + row * n_per_row + QK_K * ibl + IQ1M_BLOCK_SIZE * ib2 : NULL;
                float weight2[IQ1M_BLOCK_SIZE];
                for (int i = 0; i < IQ1M_BLOCK_SIZE; ++i) {
                    float wi = qw2 ? qw2[i] * sqrtf(sigma2 + xb[i] * xb[i]) : xb[i] * xb[i];
                    weight2[i] = wi;
                }
                int l2 = nearest_int(0.5f * (id * s_scales[ib2] - 1.0f));
                if (l2 < 0) l2 = 0;
                if (l2 > 7) l2 = 7;
                for (int k = 0; k < IQ1M_BLOCK_SIZE / 8; ++k) {
                    const float * xx = (k == 0) ? (s_shifts[ib2] < 2 ? x_p : x_m)
                                                : (s_shifts[ib2] % 2 == 0 ? x_p : x_m);
                    const int8_t * pg = grid + 8 * (s_qs[2 * ib2 + k] | ((int)(s_qh[ib2] << (8 - 4 * k)) & 0x700));
                    for (int i = 0; i < 8; ++i) {
                        float w = weight2[8 * k + i];
                        float q = xx[(pg[i] - 1) / 2] * (2 * l2 + 1);
                        sumqx_f += w * q * xb[8 * k + i];
                        sumq2_f += w * q * q;
                    }
                }
            }
            float d_final = d;
            if (sumq2_f > 0) d_final = sumqx_f / sumq2_f;
            uint16_t du16 = fp32_to_fp16(d_final * 1.1125f);
            sc[0] |= (uint16_t)((du16 & 0x000f) << 12);
            sc[1] |= (uint16_t)((du16 & 0x00f0) << 8);
            sc[2] |= (uint16_t)((du16 & 0x0f00) << 4);
            sc[3] |= (uint16_t)((du16 & 0xf000) << 0);
            for (int i = 0; i < QK_K / 8; ++i) y->qs[i] = s_qs[i];
            for (int i = 0; i < QK_K / 16; ++i) y->qh[i] = s_qh[i];
        }
    }
}
