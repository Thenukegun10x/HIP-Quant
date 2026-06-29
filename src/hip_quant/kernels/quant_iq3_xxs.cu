#include <hip/hip_runtime.h>
#include "../hip_quant_types.h"
#include "../hip_quant_util.h"
#include "../hip_iquant_util.h"

#define QK_K 256
#define IQ3S_BLOCK_SIZE 32
#define NGRID_IQ3XXS 256
#define KMAP_SIZE 4096
#define KMAX_Q 8

// IQ3_XXS grid data (pre-computed from kgrid_256)
__constant__ int8_t d_iq3xxs_grid[NGRID_IQ3XXS][4];
__constant__ int d_iq3xxs_map[KMAP_SIZE];

static __device__ int iq3xxs_find_best_neighbour(
    const float *xval, const float *weight, float scale, int8_t *L
) {
    float best_d2 = __int_as_float(0x7F7FFFFF);
    int best_idx = -1;
    for (int j = 0; j < NGRID_IQ3XXS; ++j) {
        float d2 = 0.0f;
        for (int i = 0; i < 4; ++i) {
            float q = (float)d_iq3xxs_grid[j][i];
            float diff = scale * q - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if (d2 < best_d2) {
            best_d2 = d2;
            best_idx = j;
        }
    }
    for (int i = 0; i < 4; ++i) {
        L[i] = (d_iq3xxs_grid[best_idx][i] - 1) / 2;
    }
    return best_idx;
}

extern "C" __global__
__launch_bounds__(256, 4)
void quantize_iq3_xxs_kernel(
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

    int base = row * n_per_row + sb * QK_K + grp * IQ3S_BLOCK_SIZE + lane;
    if (base > (row + 1) * n_per_row - 1) return;

    __shared__ float s_x[QK_K];
    __shared__ float s_weight[QK_K];
    __shared__ float s_waux[QK_K];
    __shared__ float s_xval[QK_K];
    __shared__ int8_t s_L[QK_K];
    __shared__ int8_t s_Laux[QK_K];
    __shared__ float s_scales[8];
    __shared__ uint32_t s_ss[8];  // scales_and_signs
    __shared__ uint8_t s_qs_idx[64];  // grid indices (8 × 8)
    __shared__ float s_sigma2;
    __shared__ float s_max_scale;

    int idx = grp * IQ3S_BLOCK_SIZE + lane;
    s_x[idx] = src[base];
    __syncthreads();

    // Compute weights
    if (tid == 0) {
        float sumx2 = 0.0f;
        for (int i = 0; i < QK_K; ++i) sumx2 += s_x[i] * s_x[i];
        s_sigma2 = 2.0f * sumx2 / (float)QK_K;
    }
    __syncthreads();

    float sigma2 = s_sigma2;
    float xv = s_x[idx];
    if (imatrix) {
        // imatrix not passed per-element in this kernel design
        s_weight[idx] = xv * xv;
    } else {
        s_weight[idx] = xv * xv;
    }
    __syncthreads();

    // Each group of 32 threads: lane 0 does the heavy lifting
    if (lane == 0) {
        int ib = grp;  // sub-block index
        const float *xb = s_x + ib * IQ3S_BLOCK_SIZE;
        const float *weight = s_weight + ib * IQ3S_BLOCK_SIZE;
        float *xval = s_xval + ib * IQ3S_BLOCK_SIZE;
        float *waux = s_waux + ib * IQ3S_BLOCK_SIZE;
        int8_t *L = s_L + ib * IQ3S_BLOCK_SIZE;
        int8_t *Laux = s_Laux + ib * IQ3S_BLOCK_SIZE;
        uint8_t block_signs[4];

        for (int i = 0; i < IQ3S_BLOCK_SIZE; ++i) {
            waux[i] = sqrtf(weight[i]);
        }

        // Sign handling: 4 groups of 8 elements
        for (int k = 0; k < 4; ++k) {
            int nflip = 0;
            uint8_t s = 0;
            for (int i = 0; i < 8; ++i) {
                if (xb[8*k + i] >= 0) {
                    xval[8*k + i] = xb[8*k + i];
                } else {
                    xval[8*k + i] = -xb[8*k + i];
                    ++nflip;
                    s |= (1 << i);
                }
            }
            if (nflip & 1) {
                int imin = 0;
                float min = weight[8*k] * xb[8*k] * xb[8*k];
                for (int i = 1; i < 8; ++i) {
                    float ax = weight[8*k+i] * xb[8*k+i] * xb[8*k+i];
                    if (ax < min) { min = ax; imin = i; }
                }
                xval[8*k+imin] = -xval[8*k+imin];
                s ^= (1 << imin);
            }
            block_signs[k] = s & 127;
        }

        float max_v = xval[0];
        for (int i = 1; i < IQ3S_BLOCK_SIZE; ++i) {
            if (xval[i] > max_v) max_v = xval[i];
        }

        if (max_v < 1e-5f) {
            s_scales[ib] = 0.0f;
            s_ss[ib] = 0;
            for (int k = 0; k < 8; ++k) {
                s_qs_idx[ib * 8 + k] = 0;
            }
            return;
        }

        float best = 0.0f;
        float scale = max_v / (float)(2*KMAX_Q - 1);
        bool is_on_grid[8];
        bool is_on_grid_aux[8];
        for (int k = 0; k < 8; ++k) is_on_grid[k] = true;

        // 31 trials
        for (int is = -15; is <= 15; ++is) {
            float id = (float)(2*KMAX_Q - 1 + is * 20) * 0.05f / max_v;
            float this_scale = 1.0f / id;

            for (int k = 0; k < 8; ++k) {
                for (int i = 0; i < 4; ++i) {
                    int l = nearest_int(0.5f * (id * xval[4*k + i] - 1.0f));
                    l = max(0, min(KMAX_Q - 1, l));
                    Laux[4*k + i] = (int8_t)l;
                }
                uint16_t u = (uint16_t)Laux[4*k]
                           | ((uint16_t)Laux[4*k+1] << 3)
                           | ((uint16_t)Laux[4*k+2] << 6)
                           | ((uint16_t)Laux[4*k+3] << 9);
                int grid_idx = d_iq3xxs_map[(int)u];
                is_on_grid_aux[k] = true;
                if (grid_idx < 0) {
                    is_on_grid_aux[k] = false;
                    grid_idx = iq3xxs_find_best_neighbour(
                        xval + 4*k, waux + 4*k, this_scale, Laux + 4*k);
                }
            }

            float sumqx = 0.0f, sumq2 = 0.0f;
            for (int i = 0; i < IQ3S_BLOCK_SIZE; ++i) {
                float w = weight[i];
                float q = 2.0f * (float)Laux[i] + 1.0f;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }

            if (sumq2 > 0.0f && sumqx * sumqx > best * sumq2) {
                scale = sumqx / sumq2;
                best = scale * sumqx;
                for (int i = 0; i < IQ3S_BLOCK_SIZE; ++i) {
                    L[i] = Laux[i];
                }
                for (int k = 0; k < 8; ++k) {
                    is_on_grid[k] = is_on_grid_aux[k];
                }
            }
        }

        // Fallback for points not on grid
        int n_not_ongrid = 0;
        for (int k = 0; k < 8; ++k) {
            if (!is_on_grid[k]) ++n_not_ongrid;
        }
        if (n_not_ongrid > 0 && scale > 0.0f) {
            float id = 1.0f / scale;
            for (int k = 0; k < 8; ++k) {
                if (is_on_grid[k]) continue;
                uint16_t u = 0;
                for (int i = 0; i < 4; ++i) {
                    int l = nearest_int(0.5f * (id * xval[4*k + i] - 1.0f));
                    l = max(0, min(KMAX_Q - 1, l));
                    u |= (uint16_t)l << (3*i);
                    L[4*k + i] = (int8_t)l;
                }
                int grid_idx = d_iq3xxs_map[(int)u];
                if (grid_idx < 0) {
                    grid_idx = iq3xxs_find_best_neighbour(
                        xval + 4*k, waux + 4*k, scale, L + 4*k);
                } else {
                    const int8_t *pg = d_iq3xxs_grid[grid_idx];
                    for (int i = 0; i < 4; ++i) {
                        L[4*k + i] = (pg[i] - 1) / 2;
                    }
                }
            }
            float sumqx = 0.0f, sumq2 = 0.0f;
            for (int i = 0; i < IQ3S_BLOCK_SIZE; ++i) {
                float w = weight[i];
                float q = 2.0f * (float)L[i] + 1.0f;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if (sumq2 > 0.0f) scale = sumqx / sumq2;
        }

        if (scale < 0.0f) {
            scale = -scale;
            for (int k = 0; k < 4; ++k) block_signs[k] = (~block_signs[k]) & 127;
        }

        // Store grid indices and compute final scales_and_signs
        for (int k = 0; k < 8; ++k) {
            uint16_t u = 0;
            for (int i = 0; i < 4; ++i) {
                u |= (uint16_t)L[4*k + i] << (3*i);
            }
            int grid_idx = d_iq3xxs_map[(int)u];
            s_qs_idx[ib * 8 + k] = (uint8_t)grid_idx;
        }

        uint32_t ss = (uint32_t)block_signs[0]
                    | ((uint32_t)block_signs[1] << 7)
                    | ((uint32_t)block_signs[2] << 14)
                    | ((uint32_t)block_signs[3] << 21);
        s_scales[ib] = scale;
        s_ss[ib] = ss;
    }
    __syncthreads();

    // Find max_scale and compute super-block d
    if (tid == 0) {
        float max_scale = 0.0f;
        for (int j = 0; j < 8; ++j) {
            if (s_scales[j] > max_scale) max_scale = s_scales[j];
        }
        s_max_scale = max_scale;

        block_iq3_xxs *blk = (block_iq3_xxs*)(dst + (row * (n_per_row / QK_K) + sb) * sizeof(block_iq3_xxs));

        if (max_scale > 0.0f) {
            float d = max_scale / 31.0f;
            blk->d = fp32_to_fp16(d * 1.0125f);
            float id = 1.0f / d;
            for (int ib = 0; ib < 8; ++ib) {
                int l = nearest_int(0.5f * (id * s_scales[ib] - 1.0f));
                l = max(0, min(15, l));
                s_ss[ib] |= ((uint32_t)l << 28);
            }
        } else {
            blk->d = fp32_to_fp16(0.0f);
        }
    }
    __syncthreads();

    // Write output
    block_iq3_xxs *blk = (block_iq3_xxs*)(dst + (row * (n_per_row / QK_K) + sb) * sizeof(block_iq3_xxs));
    if (tid < 64) {
        blk->qs[tid] = s_qs_idx[tid];
    } else if (tid < 64 + 8) {
        int ss_idx = tid - 64;
        ((uint32_t*)(blk->qs + 64))[ss_idx] = s_ss[ss_idx];
    }
}
