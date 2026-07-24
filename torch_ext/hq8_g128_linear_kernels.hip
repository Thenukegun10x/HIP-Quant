// Direct row-major HQ8_G128 inference kernel for the PyTorch ROCm extension.
//
// The reference codec stores [fp16 scale][128 int8 values] for every output-row
// K group. This kernel keeps activations in their original floating dtype
// (W8A16), accumulates FP32, and does not materialise a decoded weight matrix.
// A later W8A8 tile format will be a distinct fast-path codec.

#include <hip/hip_runtime.h>
#include "../hip_quant_util.h"

#define HIP_QUANT_DTYPE_F32  0
#define HIP_QUANT_DTYPE_F16  1
#define HIP_QUANT_DTYPE_BF16 2

namespace {

static constexpr int HQ8_G128_GROUP_SIZE = 128;
static constexpr int HQ8_G128_GROUP_BYTES = 130;
static constexpr int HQ8_G128_SCALE_BYTES = 2;
static constexpr int HQ8_OUTPUTS_PER_BLOCK = 16;
static constexpr int HQ8_LANES_PER_OUTPUT = 16;

__device__ __forceinline__ float hq8_load_scalar(const void* source, int64_t index, int dtype) {
    if (dtype == HIP_QUANT_DTYPE_F16) {
        return fp16_to_fp32(((const uint16_t*)source)[index]);
    }
    if (dtype == HIP_QUANT_DTYPE_BF16) {
        return bf16_to_fp32(((const uint16_t*)source)[index]);
    }
    return ((const float*)source)[index];
}

__device__ __forceinline__ void hq8_store_scalar(void* destination, int64_t index, float value, int dtype) {
    if (dtype == HIP_QUANT_DTYPE_F16) {
        ((uint16_t*)destination)[index] = fp32_to_fp16(value);
    } else if (dtype == HIP_QUANT_DTYPE_BF16) {
        ((uint16_t*)destination)[index] = fp32_to_bf16(value);
    } else {
        ((float*)destination)[index] = value;
    }
}

__global__ __launch_bounds__(256, 2)
void hq8_g128_linear_forward_kernel(
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
    const int local_output = threadIdx.x / HQ8_LANES_PER_OUTPUT;
    const int lane = threadIdx.x & (HQ8_LANES_PER_OUTPUT - 1);
    const int n = blockIdx.x * HQ8_OUTPUTS_PER_BLOCK + local_output;
    const int m = blockIdx.y;
    const int groups_per_row = K / HQ8_G128_GROUP_SIZE;
    const int64_t input_base = (int64_t)m * K;

    float partial = 0.0f;
    if (n < N) {
        const uint8_t* row = packed_weight
            + (int64_t)n * groups_per_row * HQ8_G128_GROUP_BYTES;
        for (int group = 0; group < groups_per_row; ++group) {
            const uint8_t* block = row + group * HQ8_G128_GROUP_BYTES;
            const float scale = fp16_to_fp32(*(const uint16_t*)block);
            const int8_t* codes = (const int8_t*)(block + HQ8_G128_SCALE_BYTES);
            const int k = group * HQ8_G128_GROUP_SIZE + lane * 8;
            #pragma unroll
            for (int element = 0; element < 8; ++element) {
                partial += hq8_load_scalar(input, input_base + k + element, input_dtype)
                    * (scale * (float)codes[lane * 8 + element]);
            }
        }
    }

    __shared__ float reductions[HQ8_OUTPUTS_PER_BLOCK][HQ8_LANES_PER_OUTPUT];
    reductions[local_output][lane] = partial;
    __syncthreads();

    if (lane == 0 && n < N) {
        float value = 0.0f;
        #pragma unroll
        for (int source_lane = 0; source_lane < HQ8_LANES_PER_OUTPUT; ++source_lane) {
            value += reductions[local_output][source_lane];
        }
        if (has_bias) value += hq8_load_scalar(bias, n, bias_dtype);
        hq8_store_scalar(output, (int64_t)m * N + n, value, input_dtype);
    }
}

} // namespace

extern "C" void launch_hq8_g128_linear_forward(
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
    const dim3 grid(
        (uint32_t)((N + HQ8_OUTPUTS_PER_BLOCK - 1) / HQ8_OUTPUTS_PER_BLOCK),
        (uint32_t)M
    );
    hipLaunchKernelGGL(
        hq8_g128_linear_forward_kernel,
        grid,
        dim3(HQ8_OUTPUTS_PER_BLOCK * HQ8_LANES_PER_OUTPUT),
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

