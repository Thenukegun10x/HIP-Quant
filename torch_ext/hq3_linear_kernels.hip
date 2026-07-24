// Direct packed-HQ3 inference kernel for the PyTorch ROCm extension.
//
// Weight layout is hq2.HQ3Tensor: [N * (K / 256), 112] uint8. Each block is
// eight FP16 centroids followed by 256 three-bit selectors packed LSB-first
// as 32 groups of eight codes in three bytes each. The kernel decodes a value
// only while multiplying it and never materialises a full weight matrix.

#include <hip/hip_runtime.h>
#include "../hip_quant_util.h"

#define HIP_QUANT_DTYPE_F32  0
#define HIP_QUANT_DTYPE_F16  1
#define HIP_QUANT_DTYPE_BF16 2

namespace {

static constexpr int HQ3_BLOCK_SIZE = 256;
static constexpr int HQ3_BLOCK_BYTES = 112;
static constexpr int HQ3_CENTROID_BYTES = 16;

__device__ __forceinline__ float hq3_load_scalar(const void* source, int64_t index, int dtype) {
    if (dtype == HIP_QUANT_DTYPE_F16) return fp16_to_fp32(((const uint16_t*)source)[index]);
    if (dtype == HIP_QUANT_DTYPE_BF16) return bf16_to_fp32(((const uint16_t*)source)[index]);
    return ((const float*)source)[index];
}

__device__ __forceinline__ void hq3_store_scalar(void* destination, int64_t index, float value, int dtype) {
    if (dtype == HIP_QUANT_DTYPE_F16) {
        ((uint16_t*)destination)[index] = fp32_to_fp16(value);
    } else if (dtype == HIP_QUANT_DTYPE_BF16) {
        ((uint16_t*)destination)[index] = fp32_to_bf16(value);
    } else {
        ((float*)destination)[index] = value;
    }
}

__global__ __launch_bounds__(256, 2)
void hq3_linear_forward_kernel(
    const void* __restrict__ input,
    const uint8_t* __restrict__ packed_weight,
    void* __restrict__ output,
    const void* __restrict__ bias,
    int M,
    int N,
    int K,
    int input_dtype,
    int bias_dtype,
    bool has_bias
) {
    const int n = blockIdx.x;
    const int m = blockIdx.y;
    const int tid = threadIdx.x;
    const int blocks_per_row = K / HQ3_BLOCK_SIZE;
    const uint8_t* row = packed_weight + (int64_t)n * blocks_per_row * HQ3_BLOCK_BYTES;
    const int64_t input_base = (int64_t)m * K;

    float partial = 0.0f;
    for (int k = tid; k < K; k += blockDim.x) {
        const int block_index = k >> 8;
        const int within_block = k & (HQ3_BLOCK_SIZE - 1);
        const uint8_t* block = row + block_index * HQ3_BLOCK_BYTES;
        const int selector_group = within_block >> 3;
        const uint8_t* selector = block + HQ3_CENTROID_BYTES + 3 * selector_group;
        const uint32_t selector_word = (uint32_t)selector[0] | ((uint32_t)selector[1] << 8) | ((uint32_t)selector[2] << 16);
        const int code = (selector_word >> (3 * (within_block & 7))) & 7;
        const uint16_t level_bits = ((const uint16_t*)block)[code];
        partial += hq3_load_scalar(input, input_base + k, input_dtype) * fp16_to_fp32(level_bits);
    }

    __shared__ float reduce[256];
    reduce[tid] = partial;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] += reduce[tid + stride];
        __syncthreads();
    }
    if (tid == 0) {
        float value = reduce[0];
        if (has_bias) value += hq3_load_scalar(bias, n, bias_dtype);
        hq3_store_scalar(output, (int64_t)m * N + n, value, input_dtype);
    }
}

} // namespace

extern "C" void launch_hq3_linear_forward(
    const void* input,
    const uint8_t* packed_weight,
    void* output,
    const void* bias,
    int M,
    int N,
    int K,
    int input_dtype,
    int bias_dtype,
    bool has_bias,
    hipStream_t stream
) {
    const dim3 grid((uint32_t)N, (uint32_t)M);
    hipLaunchKernelGGL(
        hq3_linear_forward_kernel,
        grid,
        dim3(256),
        0,
        stream,
        input,
        packed_weight,
        output,
        bias,
        M,
        N,
        K,
        input_dtype,
        bias_dtype,
        has_bias
    );
}
