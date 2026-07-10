// torch_ext/fp8_linear_kernels_v2.hip
//
// Cooperative LDS staging with multi-K-tile amortised synchronisation.
//
//   Row-major blockDim = 512; prepacked B selects 2/4/8/16 waves to fill CUs.
//   A_lds     =  4 KB  [32][128+4]       (4 B row-pad avoids bank conflicts)
//   B_lds     = 34 KB  [256][128+4]       (prepacked B bypasses LDS)
//   total LDS = 38 KB row-major, 4 KB prepacked
//
//   K_TILES_PER_STAGE = 8  →  128 K-elements per __syncthreads() pair
//   Each block computes two 16x16 M tiles per wave, reusing the staged B
//   slab and halving B traffic and block count for M > 16.
//
//   Savings: A loaded once per stage (16× less global traffic for A);
//            B accessed in coalesced 4-byte loads instead of 1-byte loads.
//
// Convention (identical to fp8_linear_kernels.hip):
//   A_fp8 : [M, K]  uint8_t    (activations, row-major)
//   B_fp8 : [N, K]  uint8_t    (weights,     row-major)
//   C     : [M, N]  float/half (output)

#include <hip/hip_runtime.h>
#include "../hip_quant_util.h"

#define TILE_M  16
#define TILE_N  16
#define TILE_K  16
#define MAX_WAVES_PER_BLOCK 16
#define M_TILES_PER_BLOCK   2
#define K_TILES_PER_STAGE  8
#define K_STAGE_ELEMS      (K_TILES_PER_STAGE * TILE_K)   // 128

#define HIP_QUANT_DTYPE_F32  0
#define HIP_QUANT_DTYPE_F16  1
#define HIP_QUANT_DTYPE_BF16 2

typedef float v8f __attribute__((ext_vector_type(8)));
typedef int   v2i __attribute__((ext_vector_type(2)));

static __device__ inline float load_float_input(const void* src, int dtype, int64_t idx) {
    if (dtype == HIP_QUANT_DTYPE_F16) {
        return fp16_to_fp32(((const uint16_t*)src)[idx]);
    }
    if (dtype == HIP_QUANT_DTYPE_BF16) {
        return bf16_to_fp32(((const uint16_t*)src)[idx]);
    }
    return ((const float*)src)[idx];
}

static __device__ inline void store_float_output(void* dst, int dtype, int64_t idx, float value) {
    if (dtype == HIP_QUANT_DTYPE_F16) {
        ((uint16_t*)dst)[idx] = fp32_to_fp16(value);
    } else if (dtype == HIP_QUANT_DTYPE_BF16) {
        ((uint16_t*)dst)[idx] = fp32_to_bf16(value);
    } else {
        ((float*)dst)[idx] = value;
    }
}

// LDS row stride = 128 + 4 = 132 bytes (33 words).
// The extra 4 bytes per row rotate the bank mapping so that consecutive
// rows land in consecutive LDS banks, avoiding 16-way bank conflicts
// when all 16 lanes of a wave read the same K-offset from different rows.
#define LDS_ROW_WORDS  (K_STAGE_ELEMS / 4 + 1)

static __device__ __forceinline__ uint32_t load_fp8_word(
    const uint8_t* src, int64_t offset, int remaining
) {
    if (remaining >= 4 && (offset & 3) == 0) {
        return *(const uint32_t*)(src + offset);
    }

    uint32_t value = 0;
    int count = remaining < 4 ? remaining : 4;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        if (i < count) ((uint8_t*)&value)[i] = src[offset + i];
    }
    return value;
}

static __device__ __forceinline__ v2i load_packed_b_operand(
    const uint8_t* B_packed, int n_tile, int k_tiles, int k_tile, int tid
) {
    if (k_tile >= k_tiles) return (v2i){0, 0};
    int64_t byte_offset = ((int64_t)n_tile * k_tiles + k_tile) * TILE_K * TILE_N + tid * 8;
    const uint32_t* words = (const uint32_t*)(B_packed + byte_offset);
    return (v2i){(int)words[0], (int)words[1]};
}

template<int M_TILES, int N_WAVES>
__global__
__launch_bounds__(N_WAVES * 32, 1)
void fp8_gemm_v2_input_weight_fp8_kernel(
    const uint8_t * __restrict__ A_fp8,
    const uint8_t * __restrict__ B_fp8,
    void          * __restrict__ C,
    const void    * __restrict__ bias,
    int M, int N, int K,
    float out_scale,
    int c_dtype, int bias_dtype, bool has_bias
) {
    int wave = threadIdx.x >> 5;
    int tid  = threadIdx.x & 31;
    if (wave >= N_WAVES) return;

    int m = blockIdx.y * (M_TILES * TILE_M);
    int n_tile = blockIdx.x * N_WAVES + wave;
    int n = n_tile * TILE_N;
    bool valid_wave = (m < M && n < N);

    int lane_wrapped = tid & 15;
    int lane_group   = tid >> 4;

    v8f acc[M_TILES];
    #pragma unroll
    for (int mt = 0; mt < M_TILES; ++mt) acc[mt] = (v8f){};

    __shared__ uint32_t A_lds[M_TILES * TILE_M][LDS_ROW_WORDS];
    __shared__ uint32_t B_lds[N_WAVES * TILE_N][LDS_ROW_WORDS];

    int linear_tid = threadIdx.x;
    constexpr int block_threads = N_WAVES * 32;
    constexpr int a_stage_words = M_TILES * TILE_M * (K_STAGE_ELEMS / 4);

    for (int k_start = 0; k_start < K; k_start += K_STAGE_ELEMS) {

        // ---------- cooperative load  A  ----------
        for (int a_idx = linear_tid; a_idx < a_stage_words; a_idx += block_threads) {
            int a_row = a_idx / (K_STAGE_ELEMS / 4);
            int a_word = a_idx % (K_STAGE_ELEMS / 4);
            int a_col = a_word * 4;
            int global_row = m + a_row;
            uint32_t a_val = 0;
            if (global_row < M) {
                int64_t global_off = (int64_t)global_row * K + k_start + a_col;
                int limit = K - k_start - a_col;
                a_val = load_fp8_word(A_fp8, global_off, limit);
            }
            A_lds[a_row][a_word] = a_val;
        }

        // ---------- cooperative load  B  ----------
        // Every thread loads 16 words regardless of the configured N width.
        {
            for (int pass = 0; pass < 16; pass++) {
                int b_idx = pass * block_threads + linear_tid;
                int b_row = b_idx / (K_STAGE_ELEMS / 4);
                int b_word = b_idx % (K_STAGE_ELEMS / 4);
                int b_col = b_word * 4;                         // 0,4,...,124

                int wave_idx = b_row >> 4;                     // b_row / 16
                int n_off    = b_row & 15;                     // b_row % 16
                int global_n = (blockIdx.x * N_WAVES + wave_idx) * TILE_N + n_off;
                uint32_t b_val = 0;
                if (global_n < N) {
                    int64_t global_off = (int64_t)global_n * K + k_start + b_col;
                    int limit = K - k_start - b_col;
                    b_val = load_fp8_word(B_fp8, global_off, limit);
                }
                B_lds[b_row][b_word] = b_val;
            }
        }

        __syncthreads();

        // ---------- compute  8  WMMA tiles from LDS ----------
        if (valid_wave) {
        for (int kt = 0; kt < K_TILES_PER_STAGE; kt++) {
            int word = kt * (TILE_K / 4) + lane_group * 2;
            v2i b_vec = (v2i){
                (int)B_lds[wave * TILE_N + lane_wrapped][word],
                (int)B_lds[wave * TILE_N + lane_wrapped][word + 1]
            };

            #pragma unroll
            for (int mt = 0; mt < M_TILES; ++mt) {
                if (m + mt * TILE_M < M) {
                    v2i a_vec = (v2i){
                        (int)A_lds[mt * TILE_M + lane_wrapped][word],
                        (int)A_lds[mt * TILE_M + lane_wrapped][word + 1]
                    };
                    acc[mt] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                        a_vec, b_vec, acc[mt]);
                }
            }
        }
        }

        __syncthreads();
    }

    // ---------- store  C  ----------
    if (valid_wave) {
    #pragma unroll
    for (int mt = 0; mt < M_TILES; ++mt) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int row = m + mt * TILE_M + lane_group * 8 + i;
            int col = n + lane_wrapped;
            if (row < M && col < N) {
                float value = acc[mt][i] * out_scale;
                if (has_bias) value += load_float_input(bias, bias_dtype, col);
                store_float_output(C, c_dtype, (int64_t)row * N + col, value);
            }
        }
    }
    }
}


// ---------------------------------------------------------------------------
// C-linkage launch helpers (called from pytorch_bindings_hip.cpp)
// ---------------------------------------------------------------------------
static int current_device_cu_count() {
    static thread_local int cached_device = -1;
    static thread_local int cached_cu_count = 1;

    int device = 0;
    if (hipGetDevice(&device) != hipSuccess) return 1;
    if (device != cached_device) {
        int cu_count = 0;
        if (hipDeviceGetAttribute(&cu_count, hipDeviceAttributeMultiprocessorCount, device) == hipSuccess &&
            cu_count > 0) {
            cached_cu_count = cu_count;
        }
        cached_device = device;
    }
    return cached_cu_count;
}

static int select_n_waves(int M, int N, int m_tiles) {
    int grid_y = (M + m_tiles * TILE_M - 1) / (m_tiles * TILE_M);
    int target_blocks = current_device_cu_count();

    for (int n_waves = MAX_WAVES_PER_BLOCK; n_waves >= 2; n_waves >>= 1) {
        int n_per_block = n_waves * TILE_N;
        int grid_x = (N + n_per_block - 1) / n_per_block;
        if ((int64_t)grid_x * grid_y >= target_blocks) return n_waves;
    }
    return 2;
}

template<int M_TILES, int N_WAVES>
static void launch_fp8_gemm_v2_configured(
    const uint8_t* A_fp8, const uint8_t* B_fp8, void* C, const void* bias,
    int M, int N, int K, float out_scale,
    int c_dtype, int bias_dtype, bool has_bias, hipStream_t stream
) {
    constexpr int n_per_block = TILE_N * N_WAVES;
    dim3 block(N_WAVES * 32);
    dim3 grid((N + n_per_block - 1) / n_per_block,
              (M + M_TILES * TILE_M - 1) / (M_TILES * TILE_M));
    hipLaunchKernelGGL((fp8_gemm_v2_input_weight_fp8_kernel<M_TILES, N_WAVES>),
                       grid, block, 0, stream, A_fp8, B_fp8, C, bias, M, N, K,
                       out_scale, c_dtype, bias_dtype, has_bias);
}

// Prepacked B stores the exact eight bytes consumed by each WMMA lane
// contiguously. B bypasses LDS and is reused from registers across M tiles.
template<int M_TILES, int N_WAVES>
__global__
__launch_bounds__(N_WAVES * 32, 2)
void fp8_gemm_v2_input_weight_fp8_packed_kernel(
    const uint8_t * __restrict__ A_fp8,
    const uint8_t * __restrict__ B_packed,
    void          * __restrict__ C,
    const void    * __restrict__ bias,
    int M, int N, int K, int k_tiles,
    float out_scale,
    int c_dtype, int bias_dtype, bool has_bias
) {
    int wave = threadIdx.x >> 5;
    int tid = threadIdx.x & 31;
    if (wave >= N_WAVES) return;

    int m = blockIdx.y * (M_TILES * TILE_M);
    int n_tile = blockIdx.x * N_WAVES + wave;
    int n = n_tile * TILE_N;
    bool valid_wave = (m < M && n < N);
    int lane_wrapped = tid & 15;
    int lane_group = tid >> 4;

    v8f acc[M_TILES];
    #pragma unroll
    for (int mt = 0; mt < M_TILES; ++mt) acc[mt] = (v8f){};

    __shared__ uint32_t A_lds[M_TILES * TILE_M][LDS_ROW_WORDS];
    constexpr int block_threads = N_WAVES * 32;
    constexpr int a_stage_words = M_TILES * TILE_M * (K_STAGE_ELEMS / 4);
    int linear_tid = threadIdx.x;

    for (int k_start = 0; k_start < K; k_start += K_STAGE_ELEMS) {
        for (int a_idx = linear_tid; a_idx < a_stage_words; a_idx += block_threads) {
            int a_row = a_idx / (K_STAGE_ELEMS / 4);
            int a_word = a_idx % (K_STAGE_ELEMS / 4);
            int a_col = a_word * 4;
            int global_row = m + a_row;
            uint32_t a_val = 0;
            if (global_row < M) {
                int64_t global_off = (int64_t)global_row * K + k_start + a_col;
                a_val = load_fp8_word(A_fp8, global_off, K - k_start - a_col);
            }
            A_lds[a_row][a_word] = a_val;
        }

        __syncthreads();

        if (valid_wave) {
            int first_k_tile = k_start / TILE_K;
            // Keep the current A/B operands live while the next pair is
            // fetched.  The packed path already did this for B; retaining
            // A's LDS read result here lets the scheduler overlap the next
            // LDS reads with the current WMMA as well.
            v2i a_vec[M_TILES];
            const int first_word = lane_group * 2;
            #pragma unroll
            for (int mt = 0; mt < M_TILES; ++mt) {
                if (m + mt * TILE_M < M) {
                    a_vec[mt] = (v2i){
                        (int)A_lds[mt * TILE_M + lane_wrapped][first_word],
                        (int)A_lds[mt * TILE_M + lane_wrapped][first_word + 1]
                    };
                }
            }

            v2i b_vec = load_packed_b_operand(
                B_packed, n_tile, k_tiles, first_k_tile, tid);

            #pragma unroll
            for (int kt = 0; kt < K_TILES_PER_STAGE; ++kt) {
                // Issue both next-iteration operand loads before consuming the
                // current pair.  For M_TILES=1 this is the intended four-
                // VGPR A pipeline (current + next v2i), with no extra LDS or
                // global traffic.
                v2i next_a_vec[M_TILES];
                v2i next_b_vec = (v2i){0, 0};
                if (kt + 1 < K_TILES_PER_STAGE) {
                    int next_word = first_word + (kt + 1) * (TILE_K / 4);
                    #pragma unroll
                    for (int mt = 0; mt < M_TILES; ++mt) {
                        if (m + mt * TILE_M < M) {
                            next_a_vec[mt] = (v2i){
                                (int)A_lds[mt * TILE_M + lane_wrapped][next_word],
                                (int)A_lds[mt * TILE_M + lane_wrapped][next_word + 1]
                            };
                        }
                    }
                    next_b_vec = load_packed_b_operand(
                        B_packed, n_tile, k_tiles, first_k_tile + kt + 1, tid);
                }

                #pragma unroll
                for (int mt = 0; mt < M_TILES; ++mt) {
                    if (m + mt * TILE_M < M) {
                        acc[mt] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                            a_vec[mt], b_vec, acc[mt]);
                    }
                }

                if (kt + 1 < K_TILES_PER_STAGE) {
                    #pragma unroll
                    for (int mt = 0; mt < M_TILES; ++mt) {
                        if (m + mt * TILE_M < M) a_vec[mt] = next_a_vec[mt];
                    }
                    b_vec = next_b_vec;
                }
            }
        }

        __syncthreads();
    }

    if (valid_wave) {
        #pragma unroll
        for (int mt = 0; mt < M_TILES; ++mt) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                int row = m + mt * TILE_M + lane_group * 8 + i;
                int col = n + lane_wrapped;
                if (row < M && col < N) {
                    float value = acc[mt][i] * out_scale;
                    if (has_bias) value += load_float_input(bias, bias_dtype, col);
                    store_float_output(C, c_dtype, (int64_t)row * N + col, value);
                }
            }
        }
    }
}

template<int M_TILES, int N_WAVES>
static void launch_fp8_gemm_v2_packed_configured(
    const uint8_t* A_fp8, const uint8_t* B_packed, void* C, const void* bias,
    int M, int N, int K, int k_tiles, float out_scale,
    int c_dtype, int bias_dtype, bool has_bias, hipStream_t stream
) {
    constexpr int n_per_block = TILE_N * N_WAVES;
    dim3 block(N_WAVES * 32);
    dim3 grid((N + n_per_block - 1) / n_per_block,
              (M + M_TILES * TILE_M - 1) / (M_TILES * TILE_M));
    hipLaunchKernelGGL((fp8_gemm_v2_input_weight_fp8_packed_kernel<M_TILES, N_WAVES>),
                       grid, block, 0, stream, A_fp8, B_packed, C, bias,
                       M, N, K, k_tiles, out_scale, c_dtype, bias_dtype, has_bias);
}

template<int M_TILES>
static void dispatch_fp8_gemm_v2_packed_n_waves(
    int n_waves,
    const uint8_t* A_fp8, const uint8_t* B_packed, void* C, const void* bias,
    int M, int N, int K, int k_tiles, float out_scale,
    int c_dtype, int bias_dtype, bool has_bias, hipStream_t stream
) {
    switch (n_waves) {
        case 16:
            launch_fp8_gemm_v2_packed_configured<M_TILES, 16>(
                A_fp8, B_packed, C, bias, M, N, K, k_tiles, out_scale,
                c_dtype, bias_dtype, has_bias, stream);
            break;
        case 8:
            launch_fp8_gemm_v2_packed_configured<M_TILES, 8>(
                A_fp8, B_packed, C, bias, M, N, K, k_tiles, out_scale,
                c_dtype, bias_dtype, has_bias, stream);
            break;
        case 4:
            launch_fp8_gemm_v2_packed_configured<M_TILES, 4>(
                A_fp8, B_packed, C, bias, M, N, K, k_tiles, out_scale,
                c_dtype, bias_dtype, has_bias, stream);
            break;
        default:
            launch_fp8_gemm_v2_packed_configured<M_TILES, 2>(
                A_fp8, B_packed, C, bias, M, N, K, k_tiles, out_scale,
                c_dtype, bias_dtype, has_bias, stream);
            break;
    }
}

extern "C" {

void launch_fp8_linear_forward_v2_input_weight(
    const uint8_t* A_fp8, const uint8_t* B_fp8, void* C, const void* bias,
    int M, int N, int K,
    float input_scale, float weight_inv_scale,
    int c_dtype, int bias_dtype, bool has_bias,
    hipStream_t stream
) {
    float out_scale = weight_inv_scale / input_scale;
    if (M > TILE_M) {
        launch_fp8_gemm_v2_configured<M_TILES_PER_BLOCK, MAX_WAVES_PER_BLOCK>(
            A_fp8, B_fp8, C, bias, M, N, K, out_scale,
            c_dtype, bias_dtype, has_bias, stream);
    } else {
        launch_fp8_gemm_v2_configured<1, MAX_WAVES_PER_BLOCK>(
            A_fp8, B_fp8, C, bias, M, N, K, out_scale,
            c_dtype, bias_dtype, has_bias, stream);
    }
}

void launch_fp8_linear_forward_fp8_input_weight_packed(
    const uint8_t* A_fp8, const uint8_t* B_packed, void* C, const void* bias,
    int M, int N, int K, int k_tiles,
    float input_scale, float weight_inv_scale,
    int c_dtype, int bias_dtype, bool has_bias,
    hipStream_t stream
) {
    float out_scale = weight_inv_scale / input_scale;
    // For a short M dimension and long K, expose the two 16-row tiles as
    // independent 16-wave blocks instead of serial accumulators per wave.
    if (M <= M_TILES_PER_BLOCK * TILE_M && K >= 4096) {
        int n_waves = select_n_waves(M, N, 1);
        dispatch_fp8_gemm_v2_packed_n_waves<1>(
            n_waves, A_fp8, B_packed, C, bias, M, N, K, k_tiles, out_scale,
            c_dtype, bias_dtype, has_bias, stream);
        return;
    }
    // M=64..128 benefits from lower accumulator pressure and more independently
    // schedulable blocks despite rereading B for each 16-row tile.
    if (M >= 4 * TILE_M && M <= 8 * TILE_M && K >= 4096) {
        int n_waves = select_n_waves(M, N, 1);
        dispatch_fp8_gemm_v2_packed_n_waves<1>(
            n_waves, A_fp8, B_packed, C, bias, M, N, K, k_tiles, out_scale,
            c_dtype, bias_dtype, has_bias, stream);
        return;
    }
    if (M > TILE_M) {
        int n_waves = select_n_waves(M, N, M_TILES_PER_BLOCK);
        dispatch_fp8_gemm_v2_packed_n_waves<M_TILES_PER_BLOCK>(
            n_waves, A_fp8, B_packed, C, bias, M, N, K, k_tiles, out_scale,
            c_dtype, bias_dtype, has_bias, stream);
    } else {
        int n_waves = select_n_waves(M, N, 1);
        dispatch_fp8_gemm_v2_packed_n_waves<1>(
            n_waves, A_fp8, B_packed, C, bias, M, N, K, k_tiles, out_scale,
            c_dtype, bias_dtype, has_bias, stream);
    }
}

} // extern "C"
