#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "kernels/quant_q4_0.cu"
#include "kernels/quant_q4_1.cu"
#include "kernels/quant_q5_0.cu"
#include "kernels/quant_q5_1.cu"
#include "kernels/quant_q8_0.cu"
#include "kernels/quant_q8_1.cu"
#include "kernels/quant_q2_K.cu"
#include "kernels/quant_q3_K.cu"
#include "kernels/quant_q4_K.cu"
#include "kernels/quant_q5_K.cu"
#include "kernels/quant_q6_K.cu"
#include "kernels/quant_iq2_xxs.cu"
#include "kernels/quant_iq2_xs.cu"
#include "kernels/quant_iq1_s.cu"
#include "kernels/quant_iq4_nl.cu"
#include "kernels/quant_iq4_xs.cu"
#include "kernels/quant_iq3_xxs.cu"
#include "kernels/quant_iq3_s.cu"
#include "hip_quant_iq2xxs_data.h"
#include "hip_quant_iq2xs_data.h"
#include "hip_quant_iq1s_data.h"
#include "hip_quant_iq3xxs_data.h"
#include "hip_quant_iq3s_data.h"

#define QK_K 256
#define QK8_0 32
#define QK4_0 32
#define QK4_1 32
#define QK5_0 32
#define QK5_1 32
#define QK8_1 32
#define QK4_NL 32

static bool hip_initialized = false;
static bool iq3xxs_tables_loaded = false;
static bool iq2xxs_tables_loaded = false;
static bool iq2xs_tables_loaded = false;
static bool iq1s_tables_loaded = false;
static bool iq3s_tables_loaded = false;
static int device_id = 0;
static hipDeviceProp_t props;
static int *d_iq2xxs_map_data = NULL;
static uint16_t *d_iq2xxs_neighbours_data = NULL;
static int8_t *d_iq2xs_grid_data = NULL;
static int *d_iq2xs_map_data = NULL;
static uint16_t *d_iq2xs_neighbours_data = NULL;
static int8_t *d_iq1s_grid_data = NULL;
static int *d_iq1s_map_data = NULL;
static uint16_t *d_iq1s_neighbours_data = NULL;
static int8_t *d_iq3s_grid_data = NULL;
static int *d_iq3s_map_data = NULL;
static uint16_t *d_iq3s_neighbours_data = NULL;

#ifdef __cplusplus
extern "C" {
#endif

static size_t get_row_size(int type, int64_t n_per_row) {
    switch (type) {
        case 2:  return sizeof(block_q4_0) * (n_per_row / QK4_0);
        case 3:  return sizeof(block_q4_1) * (n_per_row / QK4_1);
        case 6:  return sizeof(block_q5_0) * (n_per_row / QK5_0);
        case 7:  return sizeof(block_q5_1) * (n_per_row / QK5_1);
        case 8:  return sizeof(block_q8_0) * (n_per_row / QK8_0);
        case 9:  return sizeof(block_q8_1) * (n_per_row / QK8_1);
        case 10: return sizeof(block_q2_K) * (n_per_row / QK_K);
        case 11: return sizeof(block_q3_K) * (n_per_row / QK_K);
        case 12: return sizeof(block_q4_K) * (n_per_row / QK_K);
        case 13: return sizeof(block_q5_K) * (n_per_row / QK_K);
        case 14: return sizeof(block_q6_K) * (n_per_row / QK_K);
        case 16: return sizeof(block_iq2_xxs) * (n_per_row / QK_K);
        case 17: return sizeof(block_iq2_xs) * (n_per_row / QK_K);
        case 18: return sizeof(block_iq3_xxs) * (n_per_row / QK_K);
        case 19: return sizeof(block_iq1_s) * (n_per_row / QK_K);
        case 20: return sizeof(block_iq4_nl) * (n_per_row / QK4_NL);
        case 21: return sizeof(block_iq3_s) * (n_per_row / QK_K);
        case 23: return sizeof(block_iq4_xs) * (n_per_row / QK_K);
        default: return 0;
    }
}

static int get_blocks_per_row(int type, int64_t n_per_row) {
    switch (type) {
        case 2:  case 3:  case 6:  case 7:  case 8:  case 9:
            return (int)(n_per_row / 32);
        case 10: case 11: case 12: case 13: case 14: case 16: case 17: case 18: case 19: case 21:
            return (int)(n_per_row / QK_K);
        case 20: return (int)(n_per_row / QK4_NL);
        case 23: return (int)(n_per_row / QK_K);
        default: return 0;
    }
}

__declspec(dllexport) size_t ggml_type_size_for(int type) {
    switch (type) {
        case 2:  return sizeof(block_q4_0);
        case 3:  return sizeof(block_q4_1);
        case 6:  return sizeof(block_q5_0);
        case 7:  return sizeof(block_q5_1);
        case 8:  return sizeof(block_q8_0);
        case 9:  return sizeof(block_q8_1);
        case 10: return sizeof(block_q2_K);
        case 11: return sizeof(block_q3_K);
        case 12: return sizeof(block_q4_K);
        case 13: return sizeof(block_q5_K);
        case 14: return sizeof(block_q6_K);
        case 16: return sizeof(block_iq2_xxs);
        case 17: return sizeof(block_iq2_xs);
        case 18: return sizeof(block_iq3_xxs);
        case 19: return sizeof(block_iq1_s);
        case 20: return sizeof(block_iq4_nl);
        case 21: return sizeof(block_iq3_s);
        case 23: return sizeof(block_iq4_xs);
        default: return 0;
    }
}

__declspec(dllexport) int ggml_blck_size_for(int type) {
    switch (type) {
        case 2:  case 3:  case 6:  case 7:  case 8:  case 9:  return 32;
        case 10: case 11: case 12: case 13: case 14: case 16: case 17: case 18: case 19: case 21: return 256;
        case 20: return QK4_NL;
        case 23: return QK_K;
        default: return 0;
    }
}

__declspec(dllexport) size_t ggml_row_size_for(int type, int64_t n_per_row) {
    int blck = ggml_blck_size_for(type);
    if (blck == 0) return 0;
    return (n_per_row / blck) * ggml_type_size_for(type);
}

__declspec(dllexport) const char* get_device_name() {
    static char name[256];
    if (!hip_initialized) {
        int count = 0;
        hipGetDeviceCount(&count);
        device_id = 0;
        int best_cu = 0;
        for (int i = 0; i < count; i++) {
            hipDeviceProp_t p;
            hipGetDeviceProperties(&p, i);
            if (p.multiProcessorCount > best_cu) {
                best_cu = p.multiProcessorCount;
                device_id = i;
            }
        }
        hipSetDevice(device_id);
        hipGetDeviceProperties(&props, device_id);
        hip_initialized = true;
    }
    strncpy(name, props.name, 255);
    return name;
}

__declspec(dllexport) size_t quantize_tensor(
    int type,
    const float* src,
    uint8_t* dst,
    int64_t nrows,
    int64_t n_per_row,
    const float* imatrix
) {
    if (!hip_initialized) {
        int count = 0;
        hipGetDeviceCount(&count);
        device_id = 0;
        int best_cu = 0;
        for (int i = 0; i < count; i++) {
            hipDeviceProp_t p;
            hipGetDeviceProperties(&p, i);
            if (p.multiProcessorCount > best_cu) {
                best_cu = p.multiProcessorCount;
                device_id = i;
            }
        }
        hipSetDevice(device_id);
        hipGetDeviceProperties(&props, device_id);
        hip_initialized = true;
    }
    if (!iq3xxs_tables_loaded) {
        hipMemcpyToSymbol(HIP_SYMBOL(d_iq3xxs_grid), h_iq3xxs_grid, sizeof(h_iq3xxs_grid));
        hipMemcpyToSymbol(HIP_SYMBOL(d_iq3xxs_map), h_iq3xxs_map, sizeof(h_iq3xxs_map));
        hipMemcpyToSymbol(HIP_SYMBOL(d_iq3xxs_neighbours), h_iq3xxs_neighbours, sizeof(h_iq3xxs_neighbours));
        iq3xxs_tables_loaded = true;
    }
    if (!iq2xxs_tables_loaded) {
        hipMemcpyToSymbol(HIP_SYMBOL(d_iq2xxs_grid), h_iq2xxs_grid, sizeof(h_iq2xxs_grid));
        hipMalloc(&d_iq2xxs_map_data, sizeof(h_iq2xxs_map));
        hipMalloc(&d_iq2xxs_neighbours_data, sizeof(h_iq2xxs_neighbours));
        hipMemcpy(d_iq2xxs_map_data, h_iq2xxs_map, sizeof(h_iq2xxs_map), hipMemcpyHostToDevice);
        hipMemcpy(d_iq2xxs_neighbours_data, h_iq2xxs_neighbours, sizeof(h_iq2xxs_neighbours), hipMemcpyHostToDevice);
        iq2xxs_tables_loaded = true;
    }
    if (!iq2xs_tables_loaded) {
        hipMalloc(&d_iq2xs_grid_data, sizeof(h_iq2xs_grid));
        hipMalloc(&d_iq2xs_map_data, sizeof(h_iq2xs_map));
        hipMalloc(&d_iq2xs_neighbours_data, sizeof(h_iq2xs_neighbours));
        hipMemcpy(d_iq2xs_grid_data, h_iq2xs_grid, sizeof(h_iq2xs_grid), hipMemcpyHostToDevice);
        hipMemcpy(d_iq2xs_map_data, h_iq2xs_map, sizeof(h_iq2xs_map), hipMemcpyHostToDevice);
        hipMemcpy(d_iq2xs_neighbours_data, h_iq2xs_neighbours, sizeof(h_iq2xs_neighbours), hipMemcpyHostToDevice);
        iq2xs_tables_loaded = true;
    }
    if (!iq1s_tables_loaded) {
        hipMalloc(&d_iq1s_grid_data, sizeof(h_iq1s_grid));
        hipMalloc(&d_iq1s_map_data, sizeof(h_iq1s_map));
        hipMalloc(&d_iq1s_neighbours_data, sizeof(h_iq1s_neighbours));
        hipMemcpy(d_iq1s_grid_data, h_iq1s_grid, sizeof(h_iq1s_grid), hipMemcpyHostToDevice);
        hipMemcpy(d_iq1s_map_data, h_iq1s_map, sizeof(h_iq1s_map), hipMemcpyHostToDevice);
        hipMemcpy(d_iq1s_neighbours_data, h_iq1s_neighbours, sizeof(h_iq1s_neighbours), hipMemcpyHostToDevice);
        iq1s_tables_loaded = true;
    }
    if (!iq3s_tables_loaded) {
        hipMalloc(&d_iq3s_grid_data, sizeof(h_iq3s_grid));
        hipMalloc(&d_iq3s_map_data, sizeof(h_iq3s_map));
        hipMalloc(&d_iq3s_neighbours_data, sizeof(h_iq3s_neighbours));
        hipMemcpy(d_iq3s_grid_data, h_iq3s_grid, sizeof(h_iq3s_grid), hipMemcpyHostToDevice);
        hipMemcpy(d_iq3s_map_data, h_iq3s_map, sizeof(h_iq3s_map), hipMemcpyHostToDevice);
        hipMemcpy(d_iq3s_neighbours_data, h_iq3s_neighbours, sizeof(h_iq3s_neighbours), hipMemcpyHostToDevice);
        iq3s_tables_loaded = true;
    }

    size_t row_size = get_row_size(type, n_per_row);
    if (row_size == 0) {
        fprintf(stderr, "hip_quantize: unsupported type %d\n", type);
        return 0;
    }
    size_t total_size = row_size * nrows;
    if (total_size == 0) return 0;

    int n_blocks_per_row = get_blocks_per_row(type, n_per_row);

    // Allocate device memory
    float *d_src = NULL;
    uint8_t *d_dst = NULL;
    float *d_imatrix = NULL;

    size_t src_bytes = (size_t)nrows * n_per_row * sizeof(float);
    size_t dst_bytes = total_size;
    size_t imatrix_bytes = imatrix ? src_bytes : 0;

    {
        hipError_t e = hipMalloc(&d_src, src_bytes);
        if (e != hipSuccess) { fprintf(stderr, "hip_quantize: hipMalloc(d_src, %zu) failed: %s\n", src_bytes, hipGetErrorString(e)); return 0; }
    }
    {
        hipError_t e = hipMalloc(&d_dst, dst_bytes);
        if (e != hipSuccess) { fprintf(stderr, "hip_quantize: hipMalloc(d_dst, %zu) failed: %s\n", dst_bytes, hipGetErrorString(e)); hipFree(d_src); return 0; }
    }
    if (imatrix) {
        hipError_t e = hipMalloc(&d_imatrix, imatrix_bytes);
        if (e != hipSuccess) { fprintf(stderr, "hip_quantize: hipMalloc(d_imatrix) failed: %s\n", hipGetErrorString(e)); hipFree(d_src); hipFree(d_dst); return 0; }
    }

    {
        hipError_t e = hipMemcpy(d_src, src, src_bytes, hipMemcpyHostToDevice);
        if (e != hipSuccess) { fprintf(stderr, "hip_quantize: hipMemcpy(d_src) failed: %s\n", hipGetErrorString(e)); hipFree(d_src); hipFree(d_dst); if (d_imatrix) hipFree(d_imatrix); return 0; }
    }
    if (imatrix) {
        hipError_t e = hipMemcpy(d_imatrix, imatrix, imatrix_bytes, hipMemcpyHostToDevice);
        if (e != hipSuccess) { fprintf(stderr, "hip_quantize: hipMemcpy(d_imatrix) failed: %s\n", hipGetErrorString(e)); hipFree(d_src); hipFree(d_dst); hipFree(d_imatrix); return 0; }
    }
    {
        hipError_t e = hipMemset(d_dst, 0, dst_bytes);
        if (e != hipSuccess) { fprintf(stderr, "hip_quantize: hipMemset failed: %s\n", hipGetErrorString(e)); hipFree(d_src); hipFree(d_dst); if (d_imatrix) hipFree(d_imatrix); return 0; }
    }

    dim3 gridDim((unsigned int)nrows, (unsigned int)n_blocks_per_row);

    switch (type) {
        case 2: {
            hipLaunchKernelGGL(quantize_q4_0_kernel, gridDim, 32, 0, 0,
                d_src, d_dst, d_imatrix, (int)nrows, (int)n_per_row);
            break;
        }
        case 3: {
            hipLaunchKernelGGL(quantize_q4_1_kernel, gridDim, 32, 0, 0,
                d_src, d_dst, d_imatrix, (int)nrows, (int)n_per_row);
            break;
        }
        case 6: {
            hipLaunchKernelGGL(quantize_q5_0_kernel, gridDim, 32, 0, 0,
                d_src, d_dst, d_imatrix, (int)nrows, (int)n_per_row);
            break;
        }
        case 7: {
            hipLaunchKernelGGL(quantize_q5_1_kernel, gridDim, 32, 0, 0,
                d_src, d_dst, d_imatrix, (int)nrows, (int)n_per_row);
            break;
        }
        case 8: {
            hipLaunchKernelGGL(quantize_q8_0_kernel, gridDim, 32, 0, 0,
                d_src, d_dst, d_imatrix, (int)nrows, (int)n_per_row);
            break;
        }
        case 9: {
            hipLaunchKernelGGL(quantize_q8_1_kernel, gridDim, 32, 0, 0,
                d_src, d_dst, d_imatrix, (int)nrows, (int)n_per_row);
            break;
        }
        case 10: {
            hipLaunchKernelGGL(quantize_q2_K_kernel, gridDim, 256, 0, 0,
                d_src, d_dst, d_imatrix, (int)nrows, (int)n_per_row);
            break;
        }
        case 11: {
            hipLaunchKernelGGL(quantize_q3_K_kernel, gridDim, 256, 0, 0,
                d_src, d_dst, d_imatrix, (int)nrows, (int)n_per_row);
            break;
        }
        case 12: {
            hipLaunchKernelGGL(quantize_q4_K_kernel, gridDim, 256, 0, 0,
                d_src, d_dst, d_imatrix, (int)nrows, (int)n_per_row);
            break;
        }
        case 13: {
            hipLaunchKernelGGL(quantize_q5_K_kernel, gridDim, 256, 0, 0,
                d_src, d_dst, d_imatrix, (int)nrows, (int)n_per_row);
            break;
        }
        case 14: {
            hipLaunchKernelGGL(quantize_q6_K_kernel, gridDim, 256, 0, 0,
                d_src, d_dst, d_imatrix, (int)nrows, (int)n_per_row);
            break;
        }
        case 16: {
            hipLaunchKernelGGL(quantize_iq2_xxs_kernel, gridDim, 1, 0, 0,
                d_src, d_dst, d_imatrix, d_iq2xxs_map_data, d_iq2xxs_neighbours_data, (int)nrows, (int)n_per_row);
            break;
        }
        case 17: {
            hipLaunchKernelGGL(quantize_iq2_xs_kernel, gridDim, 1, 0, 0,
                d_src, d_dst, d_imatrix, d_iq2xs_grid_data, d_iq2xs_map_data, d_iq2xs_neighbours_data, (int)nrows, (int)n_per_row);
            break;
        }
        case 18: {
            hipLaunchKernelGGL(quantize_iq3_xxs_kernel, gridDim, 1, 0, 0,
                d_src, d_dst, d_imatrix, (int)nrows, (int)n_per_row);
            break;
        }
        case 19: {
            hipLaunchKernelGGL(quantize_iq1_s_kernel, gridDim, 1, 0, 0,
                d_src, d_dst, d_imatrix, d_iq1s_grid_data, d_iq1s_map_data, d_iq1s_neighbours_data, (int)nrows, (int)n_per_row);
            break;
        }
        case 20: {
            hipLaunchKernelGGL(quantize_iq4_nl_kernel, gridDim, QK4_NL, 0, 0,
                d_src, d_dst, d_imatrix, (int)nrows, (int)n_per_row);
            break;
        }
        case 21: {
            hipLaunchKernelGGL(quantize_iq3_s_kernel, gridDim, 1, 0, 0,
                d_src, d_dst, d_imatrix, d_iq3s_grid_data, d_iq3s_map_data, d_iq3s_neighbours_data, (int)nrows, (int)n_per_row);
            break;
        }
        case 23: {
            hipLaunchKernelGGL(quantize_iq4_xs_kernel, gridDim, 256, 0, 0,
                d_src, d_dst, d_imatrix, (int)nrows, (int)n_per_row);
            break;
        }
        default:
            hipFree(d_src);
            hipFree(d_dst);
            if (d_imatrix) hipFree(d_imatrix);
            return 0;
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "hip_quantize: kernel launch error: %s (code %d)\n", hipGetErrorString(err), (int)err);
        hipFree(d_src);
        hipFree(d_dst);
        if (d_imatrix) hipFree(d_imatrix);
        return 0;
    }

    {
        hipError_t e = hipDeviceSynchronize();
        if (e != hipSuccess) { fprintf(stderr, "hip_quantize: hipDeviceSynchronize failed: %s\n", hipGetErrorString(e)); hipFree(d_src); hipFree(d_dst); if (d_imatrix) hipFree(d_imatrix); return 0; }
    }

    {
        hipError_t e = hipMemcpy(dst, d_dst, dst_bytes, hipMemcpyDeviceToHost);
        if (e != hipSuccess) { fprintf(stderr, "hip_quantize: hipMemcpy(dst) failed: %s\n", hipGetErrorString(e)); hipFree(d_src); hipFree(d_dst); if (d_imatrix) hipFree(d_imatrix); return 0; }
    }

    hipFree(d_src);
    hipFree(d_dst);
    if (d_imatrix) hipFree(d_imatrix);

    return total_size;
}

__declspec(dllexport) int get_device_count() {
    int count = 0;
    hipGetDeviceCount(&count);
    return count;
}

#ifdef __cplusplus
}
#endif
