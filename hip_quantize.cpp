#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#if defined(_MSC_VER)
#define HIP_QUANT_EXPORT __declspec(dllexport)
#else
#define HIP_QUANT_EXPORT __attribute__((visibility("default")))
#endif

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
#include "kernels/quant_tq1_0.cu"
#include "kernels/quant_tq2_0.cu"
#include "kernels/quant_f8_e4m3.cu"
#include "kernels/quant_f8_e5m2.cu"
#include "kernels/fp8_expand.cu"
#include "kernels/fp8_gemm_test.cu"
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
#define QK_F8 32

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

// Per-thread cached GPU buffers for quantize_tensor.
// File-scope so quantize_reset() can free them from any thread.
static thread_local float   *g_d_src       = NULL;
static thread_local uint8_t *g_d_src_fp8   = NULL;
static thread_local uint8_t *g_d_dst       = NULL;
static thread_local float   *g_d_imatrix   = NULL;
static thread_local size_t   g_d_src_cap   = 0;
static thread_local size_t   g_d_src_fp8_cap = 0;
static thread_local size_t   g_d_dst_cap   = 0;
static thread_local size_t   g_d_imatrix_cap = 0;

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================
// Initialization (HIP device + I-Quant lookup tables)
// ============================================================

static void ensure_initialized() {
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
}

// ============================================================
// Size / block helpers
// ============================================================

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
        case 34: return sizeof(block_tq1_0) * (n_per_row / QK_K);
        case 35: return sizeof(block_tq2_0) * (n_per_row / QK_K);
        case 36: return sizeof(block_f8_e4m3) * (n_per_row / QK_F8);
        case 37: return sizeof(block_f8_e5m2) * (n_per_row / QK_F8);
        default: return 0;
    }
}

static int get_blocks_per_row(int type, int64_t n_per_row) {
    switch (type) {
        case 2:  case 3:  case 6:  case 7:  case 8:  case 9:
            return (int)(n_per_row / 32);
        case 10: case 11: case 12: case 13: case 14: case 16: case 17: case 18: case 19: case 21: case 34: case 35:
            return (int)(n_per_row / QK_K);
        case 20: return (int)(n_per_row / QK4_NL);
        case 23: return (int)(n_per_row / QK_K);
        case 36: case 37: return (int)(n_per_row / QK_F8);
        default: return 0;
    }
}

HIP_QUANT_EXPORT size_t ggml_type_size_for(int type) {
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
        case 34: return sizeof(block_tq1_0);
        case 35: return sizeof(block_tq2_0);
        case 36: return sizeof(block_f8_e4m3);
        case 37: return sizeof(block_f8_e5m2);
        default: return 0;
    }
}

HIP_QUANT_EXPORT int ggml_blck_size_for(int type) {
    switch (type) {
        case 2:  case 3:  case 6:  case 7:  case 8:  case 9:  return 32;
        case 10: case 11: case 12: case 13: case 14: case 16: case 17: case 18: case 19: case 21: case 34: case 35: return 256;
        case 20: return QK4_NL;
        case 23: return QK_K;
        case 36: case 37: return QK_F8;
        default: return 0;
    }
}

HIP_QUANT_EXPORT size_t ggml_row_size_for(int type, int64_t n_per_row) {
    int blck = ggml_blck_size_for(type);
    if (blck == 0) return 0;
    return (n_per_row / blck) * ggml_type_size_for(type);
}

HIP_QUANT_EXPORT const char* get_device_name() {
    static char name[256];
    ensure_initialized();
    strncpy(name, props.name, 255);
    return name;
}

// ============================================================
// Kernel dispatch (shared by quantize_tensor and quantize_tensor_fp8_input)
// ============================================================

static int dispatch_quantize_kernel(
    int type, float *d_src, uint8_t *d_dst, float *d_imatrix,
    int nrows, int n_per_row, int n_blocks_per_row
) {
    dim3 gridDim((unsigned int)nrows, (unsigned int)n_blocks_per_row);

    switch (type) {
        case 2: {
            hipLaunchKernelGGL(quantize_q4_0_kernel, gridDim, 32, 0, 0,
                d_src, d_dst, d_imatrix, nrows, n_per_row);
            break;
        }
        case 3: {
            hipLaunchKernelGGL(quantize_q4_1_kernel, gridDim, 32, 0, 0,
                d_src, d_dst, d_imatrix, nrows, n_per_row);
            break;
        }
        case 6: {
            hipLaunchKernelGGL(quantize_q5_0_kernel, gridDim, 32, 0, 0,
                d_src, d_dst, d_imatrix, nrows, n_per_row);
            break;
        }
        case 7: {
            hipLaunchKernelGGL(quantize_q5_1_kernel, gridDim, 32, 0, 0,
                d_src, d_dst, d_imatrix, nrows, n_per_row);
            break;
        }
        case 8: {
            hipLaunchKernelGGL(quantize_q8_0_kernel, gridDim, 32, 0, 0,
                d_src, d_dst, d_imatrix, nrows, n_per_row);
            break;
        }
        case 9: {
            hipLaunchKernelGGL(quantize_q8_1_kernel, gridDim, 32, 0, 0,
                d_src, d_dst, d_imatrix, nrows, n_per_row);
            break;
        }
        case 10: {
            hipLaunchKernelGGL(quantize_q2_K_kernel, gridDim, 256, 0, 0,
                d_src, d_dst, d_imatrix, nrows, n_per_row);
            break;
        }
        case 11: {
            hipLaunchKernelGGL(quantize_q3_K_kernel, gridDim, 256, 0, 0,
                d_src, d_dst, d_imatrix, nrows, n_per_row);
            break;
        }
        case 12: {
            hipLaunchKernelGGL(quantize_q4_K_kernel, gridDim, 256, 0, 0,
                d_src, d_dst, d_imatrix, nrows, n_per_row);
            break;
        }
        case 13: {
            hipLaunchKernelGGL(quantize_q5_K_kernel, gridDim, 256, 0, 0,
                d_src, d_dst, d_imatrix, nrows, n_per_row);
            break;
        }
        case 14: {
            hipLaunchKernelGGL(quantize_q6_K_kernel, gridDim, 256, 0, 0,
                d_src, d_dst, d_imatrix, nrows, n_per_row);
            break;
        }
        case 16: {
            hipLaunchKernelGGL(quantize_iq2_xxs_kernel, gridDim, 1, 0, 0,
                d_src, d_dst, d_imatrix, d_iq2xxs_map_data, d_iq2xxs_neighbours_data, nrows, n_per_row);
            break;
        }
        case 17: {
            hipLaunchKernelGGL(quantize_iq2_xs_kernel, gridDim, 1, 0, 0,
                d_src, d_dst, d_imatrix, d_iq2xs_grid_data, d_iq2xs_map_data, d_iq2xs_neighbours_data, nrows, n_per_row);
            break;
        }
        case 18: {
            hipLaunchKernelGGL(quantize_iq3_xxs_kernel, gridDim, 1, 0, 0,
                d_src, d_dst, d_imatrix, nrows, n_per_row);
            break;
        }
        case 19: {
            hipLaunchKernelGGL(quantize_iq1_s_kernel, gridDim, 1, 0, 0,
                d_src, d_dst, d_imatrix, d_iq1s_grid_data, d_iq1s_map_data, d_iq1s_neighbours_data, nrows, n_per_row);
            break;
        }
        case 20: {
            hipLaunchKernelGGL(quantize_iq4_nl_kernel, gridDim, QK4_NL, 0, 0,
                d_src, d_dst, d_imatrix, nrows, n_per_row);
            break;
        }
        case 21: {
            hipLaunchKernelGGL(quantize_iq3_s_kernel, gridDim, 1, 0, 0,
                d_src, d_dst, d_imatrix, d_iq3s_grid_data, d_iq3s_map_data, d_iq3s_neighbours_data, nrows, n_per_row);
            break;
        }
        case 23: {
            hipLaunchKernelGGL(quantize_iq4_xs_kernel, gridDim, 256, 0, 0,
                d_src, d_dst, d_imatrix, nrows, n_per_row);
            break;
        }
        case 34: {
            hipLaunchKernelGGL(quantize_tq1_0_kernel, gridDim, 256, 0, 0,
                d_src, d_dst, d_imatrix, nrows, n_per_row);
            break;
        }
        case 35: {
            hipLaunchKernelGGL(quantize_tq2_0_kernel, gridDim, 256, 0, 0,
                d_src, d_dst, d_imatrix, nrows, n_per_row);
            break;
        }
        case 36: {
            int64_t total = (int64_t)nrows * n_per_row;
            dim3 flatGrid((unsigned int)((total + 255) / 256));
            hipLaunchKernelGGL(quantize_f8_e4m3_kernel, flatGrid, 256, 0, 0,
                d_src, d_dst, d_imatrix, nrows, n_per_row);
            break;
        }
        case 37: {
            int64_t total = (int64_t)nrows * n_per_row;
            dim3 flatGrid((unsigned int)((total + 255) / 256));
            hipLaunchKernelGGL(quantize_f8_e5m2_kernel, flatGrid, 256, 0, 0,
                d_src, d_dst, d_imatrix, nrows, n_per_row);
            break;
        }
        default:
            return 0;
    }
    return 1;
}

// ============================================================
// quantize_tensor — standard F32 input path
// ============================================================

HIP_QUANT_EXPORT size_t quantize_tensor(
    int type,
    const float* src,
    uint8_t* dst,
    int64_t nrows,
    int64_t n_per_row,
    const float* imatrix
) {
    ensure_initialized();

    size_t row_size = get_row_size(type, n_per_row);
    if (row_size == 0) {
        fprintf(stderr, "hip_quantize: unsupported type %d\n", type);
        return 0;
    }
    size_t total_size = row_size * nrows;
    if (total_size == 0) return 0;

    int n_blocks_per_row = get_blocks_per_row(type, n_per_row);

    size_t src_bytes = (size_t)nrows * n_per_row * sizeof(float);
    size_t dst_bytes = total_size;
    size_t imatrix_bytes = imatrix ? src_bytes : 0;

    if (src_bytes > g_d_src_cap) {
        if (g_d_src) hipFree(g_d_src);
        hipError_t e = hipMalloc(&g_d_src, src_bytes);
        if (e != hipSuccess) { fprintf(stderr, "hip_quantize: hipMalloc(d_src, %zu) failed: %s\n", src_bytes, hipGetErrorString(e)); return 0; }
        g_d_src_cap = src_bytes;
    }
    if (dst_bytes > g_d_dst_cap) {
        if (g_d_dst) hipFree(g_d_dst);
        hipError_t e = hipMalloc(&g_d_dst, dst_bytes);
        if (e != hipSuccess) { fprintf(stderr, "hip_quantize: hipMalloc(d_dst, %zu) failed: %s\n", dst_bytes, hipGetErrorString(e)); return 0; }
        g_d_dst_cap = dst_bytes;
    }
    if (imatrix_bytes > g_d_imatrix_cap) {
        if (g_d_imatrix) hipFree(g_d_imatrix);
        hipError_t e = hipMalloc(&g_d_imatrix, imatrix_bytes);
        if (e != hipSuccess) { fprintf(stderr, "hip_quantize: hipMalloc(d_imatrix) failed: %s\n", hipGetErrorString(e)); return 0; }
        g_d_imatrix_cap = imatrix_bytes;
    }

    {
        hipError_t e = hipMemcpy(g_d_src, src, src_bytes, hipMemcpyHostToDevice);
        if (e != hipSuccess) { fprintf(stderr, "hip_quantize: hipMemcpy(d_src) failed: %s\n", hipGetErrorString(e)); return 0; }
    }
    if (imatrix) {
        hipError_t e = hipMemcpy(g_d_imatrix, imatrix, imatrix_bytes, hipMemcpyHostToDevice);
        if (e != hipSuccess) { fprintf(stderr, "hip_quantize: hipMemcpy(d_imatrix) failed: %s\n", hipGetErrorString(e)); return 0; }
    }
    {
        hipError_t e = hipMemset(g_d_dst, 0, dst_bytes);
        if (e != hipSuccess) { fprintf(stderr, "hip_quantize: hipMemset failed: %s\n", hipGetErrorString(e)); return 0; }
    }

    if (!dispatch_quantize_kernel(type, g_d_src, g_d_dst, g_d_imatrix, (int)nrows, (int)n_per_row, n_blocks_per_row)) {
        return 0;
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "hip_quantize: kernel launch error: %s (code %d)\n", hipGetErrorString(err), (int)err);
        return 0;
    }

    {
        hipError_t e = hipDeviceSynchronize();
        if (e != hipSuccess) { fprintf(stderr, "hip_quantize: hipDeviceSynchronize failed: %s\n", hipGetErrorString(e)); return 0; }
    }

    {
        hipError_t e = hipMemcpy(dst, g_d_dst, dst_bytes, hipMemcpyDeviceToHost);
        if (e != hipSuccess) { fprintf(stderr, "hip_quantize: hipMemcpy(dst) failed: %s\n", hipGetErrorString(e)); return 0; }
    }

    return total_size;
}

// ============================================================
// quantize_tensor_fp8_input_impl — FP8 input path
//
// Accepts FP8 source data (1 byte per element) instead of float32
// (4 bytes). Matching FP8 output types are copied directly; other targets
// upload to GPU and expand to float32 on-device before quantizing.
//
// Benefits:
//   - Host memory: 4x less than F32, 2x less than BF16
//   - Transfer bandwidth: 4x less than F32
//
// Quality note: FP8 E4M3 has ~3 bits of mantissa precision.
// This is fine for low-bit targets (Q4_0 through Q5_K) where
// quantization noise dominates. For Q8_0+ and I-Quants, prefer
// the standard F32 input path.
// ============================================================

static size_t quantize_tensor_fp8_input_impl(
    int type,
    const uint8_t* src_fp8,
    uint8_t* dst,
    int64_t nrows,
    int64_t n_per_row,
    const float* imatrix,
    int src_fp8_type
) {
    ensure_initialized();

    size_t row_size = get_row_size(type, n_per_row);
    if (row_size == 0) {
        fprintf(stderr, "hip_quantize: unsupported type %d\n", type);
        return 0;
    }
    size_t total_size = row_size * nrows;
    if (total_size == 0) return 0;

    int n_blocks_per_row = get_blocks_per_row(type, n_per_row);
    int64_t total_elements = nrows * n_per_row;

    if (type == src_fp8_type && (type == 36 || type == 37)) {
        memcpy(dst, src_fp8, (size_t)total_elements);
        return total_size;
    }

    // === Phase 1: Upload FP8 source (1 byte per element) ===
    size_t src_fp8_bytes = (size_t)total_elements;
    if (src_fp8_bytes > g_d_src_fp8_cap) {
        if (g_d_src_fp8) hipFree(g_d_src_fp8);
        hipError_t e = hipMalloc(&g_d_src_fp8, src_fp8_bytes);
        if (e != hipSuccess) { fprintf(stderr, "hip_quantize: hipMalloc(d_src_fp8) failed: %s\n", hipGetErrorString(e)); return 0; }
        g_d_src_fp8_cap = src_fp8_bytes;
    }
    {
        hipError_t e = hipMemcpy(g_d_src_fp8, src_fp8, src_fp8_bytes, hipMemcpyHostToDevice);
        if (e != hipSuccess) { fprintf(stderr, "hip_quantize: hipMemcpy(d_src_fp8) failed: %s\n", hipGetErrorString(e)); return 0; }
    }

    // === Phase 2: Expand FP8 -> F32 on device ===
    size_t src_f32_bytes = (size_t)total_elements * sizeof(float);
    if (src_f32_bytes > g_d_src_cap) {
        if (g_d_src) hipFree(g_d_src);
        hipError_t e = hipMalloc(&g_d_src, src_f32_bytes);
        if (e != hipSuccess) { fprintf(stderr, "hip_quantize: hipMalloc(d_src f32) failed: %s\n", hipGetErrorString(e)); return 0; }
        g_d_src_cap = src_f32_bytes;
    }
    {
        int threads = 256;
        int blocks = (int)((total_elements + threads - 1) / threads);
        if (src_fp8_type == 37) {
            hipLaunchKernelGGL(fp8_e5m2_to_f32_expand_kernel, dim3(blocks), dim3(threads), 0, 0,
                g_d_src_fp8, g_d_src, total_elements);
        } else {
            hipLaunchKernelGGL(fp8_to_f32_expand_kernel, dim3(blocks), dim3(threads), 0, 0,
                g_d_src_fp8, g_d_src, total_elements);
        }
        hipError_t e = hipGetLastError();
        if (e != hipSuccess) {
            fprintf(stderr, "hip_quantize: fp8_expand kernel error: %s\n", hipGetErrorString(e));
            return 0;
        }
    }

    // === Phase 3: Allocate output + imatrix, dispatch quantize kernel ===
    size_t dst_bytes = total_size;

    if (dst_bytes > g_d_dst_cap) {
        if (g_d_dst) hipFree(g_d_dst);
        hipError_t e = hipMalloc(&g_d_dst, dst_bytes);
        if (e != hipSuccess) { fprintf(stderr, "hip_quantize: hipMalloc(d_dst) failed: %s\n", hipGetErrorString(e)); return 0; }
        g_d_dst_cap = dst_bytes;
    }
    if (imatrix) {
        size_t imatrix_bytes = (size_t)total_elements * sizeof(float);
        if (imatrix_bytes > g_d_imatrix_cap) {
            if (g_d_imatrix) hipFree(g_d_imatrix);
            hipError_t e = hipMalloc(&g_d_imatrix, imatrix_bytes);
            if (e != hipSuccess) { fprintf(stderr, "hip_quantize: hipMalloc(d_imatrix) failed: %s\n", hipGetErrorString(e)); return 0; }
            g_d_imatrix_cap = imatrix_bytes;
        }
        hipError_t e = hipMemcpy(g_d_imatrix, imatrix, imatrix_bytes, hipMemcpyHostToDevice);
        if (e != hipSuccess) { fprintf(stderr, "hip_quantize: hipMemcpy(d_imatrix) failed: %s\n", hipGetErrorString(e)); return 0; }
    }
    {
        hipError_t e = hipMemset(g_d_dst, 0, dst_bytes);
        if (e != hipSuccess) { fprintf(stderr, "hip_quantize: hipMemset failed: %s\n", hipGetErrorString(e)); return 0; }
    }

    if (!dispatch_quantize_kernel(type, g_d_src, g_d_dst, imatrix ? g_d_imatrix : NULL, (int)nrows, (int)n_per_row, n_blocks_per_row)) {
        return 0;
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "hip_quantize: kernel launch error: %s (code %d)\n", hipGetErrorString(err), (int)err);
        return 0;
    }

    {
        hipError_t e = hipDeviceSynchronize();
        if (e != hipSuccess) { fprintf(stderr, "hip_quantize: hipDeviceSynchronize failed: %s\n", hipGetErrorString(e)); return 0; }
    }

    {
        hipError_t e = hipMemcpy(dst, g_d_dst, dst_bytes, hipMemcpyDeviceToHost);
        if (e != hipSuccess) { fprintf(stderr, "hip_quantize: hipMemcpy(dst) failed: %s\n", hipGetErrorString(e)); return 0; }
    }

    return total_size;
}

HIP_QUANT_EXPORT size_t quantize_tensor_fp8_input(
    int type,
    const uint8_t* src_fp8,
    uint8_t* dst,
    int64_t nrows,
    int64_t n_per_row,
    const float* imatrix
) {
    return quantize_tensor_fp8_input_impl(type, src_fp8, dst, nrows, n_per_row, imatrix, 36);
}

HIP_QUANT_EXPORT size_t quantize_tensor_fp8_e5m2_input(
    int type,
    const uint8_t* src_fp8,
    uint8_t* dst,
    int64_t nrows,
    int64_t n_per_row,
    const float* imatrix
) {
    return quantize_tensor_fp8_input_impl(type, src_fp8, dst, nrows, n_per_row, imatrix, 37);
}

// ============================================================
// fp8_gemm_test — Micro FP8 GEMM via rocWMMA WMMA
//
// Takes pre-quantized FP8 E4M3 matrices A (MxK) and B (KxN),
// computes C = A * B using rocWMMA WMMA fragments on gfx12.
// Returns float32 C in pre-allocated host buffer.
//
// All matrices are row-major. lda >= K, ldb >= N, ldc >= N.
// M, N must be multiples of 16; K can be any positive int.
// ============================================================

HIP_QUANT_EXPORT int fp8_gemm_test_wmma(
    const uint8_t* A_fp8,
    const uint8_t* B_fp8,
    float* C,
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc
) {
    ensure_initialized();

    const char *disable_wmma = getenv("HIP_QUANT_DISABLE_WMMA");
    if (disable_wmma != NULL && (
        strcmp(disable_wmma, "1") == 0 || strcmp(disable_wmma, "true") == 0 ||
        strcmp(disable_wmma, "yes") == 0 || strcmp(disable_wmma, "on") == 0
    )) {
        fprintf(stderr, "fp8_gemm: disabled by HIP_QUANT_DISABLE_WMMA\n");
        return 4;
    }

    const char *enable_wmma = getenv("HIP_QUANT_ENABLE_GFX12_WMMA");
    if (enable_wmma == NULL || !(
        strcmp(enable_wmma, "1") == 0 || strcmp(enable_wmma, "true") == 0 ||
        strcmp(enable_wmma, "yes") == 0 || strcmp(enable_wmma, "on") == 0
    )) {
        fprintf(stderr, "fp8_gemm: disabled by default; set HIP_QUANT_ENABLE_GFX12_WMMA=1 for controlled testing\n");
        return 4;
    }

    if (strstr(props.gcnArchName, "gfx12") == NULL) {
        fprintf(stderr, "fp8_gemm: this FP8 WMMA kernel uses gfx12/RDNA4 w32 intrinsics; current arch is %s\n", props.gcnArchName);
        return 2;
    }

    int runtime_version = 0;
    hipRuntimeGetVersion(&runtime_version);
    if (runtime_version > 0 && runtime_version < 70200000) {
        fprintf(stderr, "fp8_gemm: gfx12 FP8 WMMA requires ROCm/HIP 7.2+; current runtime is %d\n", runtime_version);
        return 3;
    }

    size_t bytes_A = (size_t)M * lda;
    size_t bytes_B = (size_t)K * ldb;
    size_t bytes_C = (size_t)M * ldc * sizeof(float);

    uint8_t *d_A = NULL, *d_B = NULL;
    float *d_C = NULL;

    hipError_t e;

    e = hipMalloc(&d_A, bytes_A);
    if (e != hipSuccess) { fprintf(stderr, "fp8_gemm: hipMalloc A failed: %s\n", hipGetErrorString(e)); return 1; }
    e = hipMalloc(&d_B, bytes_B);
    if (e != hipSuccess) { fprintf(stderr, "fp8_gemm: hipMalloc B failed: %s\n", hipGetErrorString(e)); hipFree(d_A); return 1; }
    e = hipMalloc(&d_C, bytes_C);
    if (e != hipSuccess) { fprintf(stderr, "fp8_gemm: hipMalloc C failed: %s\n", hipGetErrorString(e)); hipFree(d_A); hipFree(d_B); return 1; }

    e = hipMemcpy(d_A, A_fp8, bytes_A, hipMemcpyHostToDevice);
    if (e != hipSuccess) { fprintf(stderr, "fp8_gemm: hipMemcpy A failed: %s\n", hipGetErrorString(e)); hipFree(d_A); hipFree(d_B); hipFree(d_C); return 1; }
    e = hipMemcpy(d_B, B_fp8, bytes_B, hipMemcpyHostToDevice);
    if (e != hipSuccess) { fprintf(stderr, "fp8_gemm: hipMemcpy B failed: %s\n", hipGetErrorString(e)); hipFree(d_A); hipFree(d_B); hipFree(d_C); return 1; }

    dim3 gridDim((M + 15) / 16, (N + 15) / 16);
    hipLaunchKernelGGL(fp8_gemm_wmma_kernel, gridDim, 32, 0, 0,
        d_A, d_B, d_C, M, N, K, lda, ldb, ldc);

    e = hipGetLastError();
    if (e != hipSuccess) { fprintf(stderr, "fp8_gemm: kernel launch error: %s\n", hipGetErrorString(e)); hipFree(d_A); hipFree(d_B); hipFree(d_C); return 1; }

    e = hipDeviceSynchronize();
    if (e != hipSuccess) { fprintf(stderr, "fp8_gemm: sync error: %s\n", hipGetErrorString(e)); hipFree(d_A); hipFree(d_B); hipFree(d_C); return 1; }

    e = hipMemcpy(C, d_C, bytes_C, hipMemcpyDeviceToHost);
    if (e != hipSuccess) { fprintf(stderr, "fp8_gemm: hipMemcpy C failed: %s\n", hipGetErrorString(e)); hipFree(d_A); hipFree(d_B); hipFree(d_C); return 1; }

    hipFree(d_A); hipFree(d_B); hipFree(d_C);
    return 0;
}

HIP_QUANT_EXPORT void quantize_reset() {
    if (g_d_src)       { hipFree(g_d_src);       g_d_src = NULL; }
    if (g_d_src_fp8)   { hipFree(g_d_src_fp8);   g_d_src_fp8 = NULL; }
    if (g_d_dst)       { hipFree(g_d_dst);       g_d_dst = NULL; }
    if (g_d_imatrix)   { hipFree(g_d_imatrix);   g_d_imatrix = NULL; }
    g_d_src_cap = 0;
    g_d_src_fp8_cap = 0;
    g_d_dst_cap = 0;
    g_d_imatrix_cap = 0;
}

HIP_QUANT_EXPORT int get_device_count() {
    int count = 0;
    hipGetDeviceCount(&count);
    return count;
}

// ============================================================
// Device property queries for compatibility checker
// ============================================================

HIP_QUANT_EXPORT int get_device_prop(
    char *name_buf, int name_buf_size,
    int *major, int *minor,
    int *cu_count,
    size_t *total_mem,
    size_t *shared_mem_per_block,
    int *warp_size,
    int *max_threads_per_block
) {
    ensure_initialized();
    strncpy(name_buf, props.name, (size_t)(name_buf_size - 1));
    name_buf[name_buf_size - 1] = '\0';
    *major = props.major;
    *minor = props.minor;
    *cu_count = props.multiProcessorCount;
    *total_mem = props.totalGlobalMem;
    *shared_mem_per_block = props.sharedMemPerBlock;
    *warp_size = props.warpSize;
    *max_threads_per_block = props.maxThreadsPerBlock;
    return 0;
}

HIP_QUANT_EXPORT int get_arch_name(char *buf, int buf_size) {
    ensure_initialized();
    strncpy(buf, props.gcnArchName, (size_t)(buf_size - 1));
    buf[buf_size - 1] = '\0';
    return 0;
}

HIP_QUANT_EXPORT int get_hip_runtime_version() {
    int runtime_version = 0;
    hipRuntimeGetVersion(&runtime_version);
    return runtime_version;
}

HIP_QUANT_EXPORT int get_device_memory(size_t *free_bytes, size_t *total_bytes) {
    ensure_initialized();
    hipError_t e = hipMemGetInfo(free_bytes, total_bytes);
    if (e != hipSuccess) {
        *free_bytes = 0;
        *total_bytes = 0;
        return 1;
    }
    return 0;
}

HIP_QUANT_EXPORT int device_has_wmma() {
    ensure_initialized();
    const char *arch = props.gcnArchName;
    // This reports support for the FP8/BF8 gfx12 w32 intrinsics used by
    // fp8_gemm_wmma_kernel, not general matrix-core or rocWMMA capability.
    // CDNA has MFMA/FP8 paths, but not this RDNA4-specific builtin path.
    return (strstr(arch, "gfx12") != NULL) ? 1 : 0;
}

#ifdef __cplusplus
}
#endif
