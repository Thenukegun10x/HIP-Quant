#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>

namespace {

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset, 32);
    }
    return val;
}

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + __expf(-x));
}

// ---------------------------------------------------------------------------
// 1. Gated DeltaNet Single-Token Decode Recurrence Kernel
// ---------------------------------------------------------------------------
__global__ void delta_net_decode_kernel(
    const float* __restrict__ q,       // [48, 128]
    const float* __restrict__ k,       // [48, 128]
    const float* __restrict__ v,       // [48, 128]
    const float* __restrict__ decay,   // [48]
    const float* __restrict__ beta,    // [48]
    float*       __restrict__ S,       // [48, 128, 128]
    half*        __restrict__ y,       // [48, 128]
    float scale
) {
    int h = blockIdx.x;     // head 0..47
    int j = blockIdx.y;     // column 0..127
    int lane = threadIdx.x; // 0..31

    int r0 = lane;
    int r1 = lane + 32;
    int r2 = lane + 64;
    int r3 = lane + 96;

    int base_h = h * (128 * 128);
    float s0 = S[base_h + r0 * 128 + j];
    float s1 = S[base_h + r1 * 128 + j];
    float s2 = S[base_h + r2 * 128 + j];
    float s3 = S[base_h + r3 * 128 + j];

    int base_vec = h * 128;
    float k0 = k[base_vec + r0];
    float k1 = k[base_vec + r1];
    float k2 = k[base_vec + r2];
    float k3 = k[base_vec + r3];

    float kv_part = s0 * k0 + s1 * k1 + s2 * k2 + s3 * k3;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        kv_part += __shfl_down(kv_part, offset, 32);
    }
    float kv_j = __shfl(kv_part, 0, 32);

    float d_val = decay[h];
    float b_val = beta[h];
    float delta_j = (v[base_vec + j] - d_val * kv_j) * b_val;

    s0 = d_val * s0 + k0 * delta_j;
    s1 = d_val * s1 + k1 * delta_j;
    s2 = d_val * s2 + k2 * delta_j;
    s3 = d_val * s3 + k3 * delta_j;

    S[base_h + r0 * 128 + j] = s0;
    S[base_h + r1 * 128 + j] = s1;
    S[base_h + r2 * 128 + j] = s2;
    S[base_h + r3 * 128 + j] = s3;

    float q0 = q[base_vec + r0];
    float q1 = q[base_vec + r1];
    float q2 = q[base_vec + r2];
    float q3 = q[base_vec + r3];

    float attn_part = s0 * q0 + s1 * q1 + s2 * q2 + s3 * q3;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        attn_part += __shfl_down(attn_part, offset, 32);
    }

    if (lane == 0) {
        y[base_vec + j] = __float2half(attn_part * scale);
    }
}

// ---------------------------------------------------------------------------
// 2. Depthwise Conv1D Shift + SiLU Kernel
// ---------------------------------------------------------------------------
__global__ void fast_ssm_conv1d_kernel(
    const half* __restrict__ x,
    half*       __restrict__ conv_state,
    const half* __restrict__ conv_weight,
    half*       __restrict__ y,
    int num_channels
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= num_channels) return;

    int state_idx = c * 3;
    int weight_idx = c * 4;

    float s0 = __half2float(conv_state[state_idx + 0]);
    float s1 = __half2float(conv_state[state_idx + 1]);
    float s2 = __half2float(conv_state[state_idx + 2]);
    float in = __half2float(x[c]);

    float w0 = __half2float(conv_weight[weight_idx + 0]);
    float w1 = __half2float(conv_weight[weight_idx + 1]);
    float w2 = __half2float(conv_weight[weight_idx + 2]);
    float w3 = __half2float(conv_weight[weight_idx + 3]);

    float conv_out = s0 * w0 + s1 * w1 + s2 * w2 + in * w3;
    float silu_out = silu(conv_out);
    y[c] = __float2half(silu_out);

    // Shift state in-place
    conv_state[state_idx + 0] = conv_state[state_idx + 1];
    conv_state[state_idx + 1] = conv_state[state_idx + 2];
    conv_state[state_idx + 2] = x[c];
}

// ---------------------------------------------------------------------------
// 3. Fast Cooperative RMSNorm Kernel (Row Dimension D, e.g. 5120)
// ---------------------------------------------------------------------------
__global__ void fast_rms_norm_kernel(
    const half* __restrict__ x,
    const half* __restrict__ w,
    half*       __restrict__ y,
    int D, float eps
) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    const half* x_row = x + b * D;
    half* y_row = y + b * D;

    float sum_sq = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float val = __half2float(x_row[i]);
        sum_sq += val * val;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down(sum_sq, offset, 32);
    }

    __shared__ float s_warp_sums[8];
    int warp_id = tid >> 5;
    int lane = tid & 31;
    if (lane == 0) {
        s_warp_sums[warp_id] = sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < (blockDim.x >> 5)) ? s_warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            val += __shfl_down(val, offset, 32);
        }
        if (lane == 0) {
            s_warp_sums[0] = rsqrtf(val / (float)D + eps);
        }
    }
    __syncthreads();

    float rsqrt_val = s_warp_sums[0];
    for (int i = tid; i < D; i += blockDim.x) {
        float val = __half2float(x_row[i]) * rsqrt_val;
        float w_val = __half2float(w[i]);
        y_row[i] = __float2half(val * w_val);
    }
}

// ---------------------------------------------------------------------------
// 4. Fast Gated RMSNorm Kernel for SSM Heads (48 heads x 128 head_dim)
// ---------------------------------------------------------------------------
// w layout: w_stride == head_dim for per-head [num_heads, head_dim] weights,
// w_stride == 0 for a SHARED [head_dim] vector broadcast across heads
// (e.g. Qwen3.5 ssm_norm). A shared weight MUST NOT be indexed per-head:
// heads >= 1 would read out of bounds (silent garbage / NaN at decode).
__global__ void fast_rms_norm_gated_kernel(
    const half* __restrict__ x,
    const half* __restrict__ w,
    const half* __restrict__ gate,
    half*       __restrict__ y,
    int num_heads, int head_dim, float eps, int w_stride
) {
    int h = blockIdx.x; // head 0..47
    if (h >= num_heads) return;

    int lane = threadIdx.x; // 0..31
    int base = h * head_dim;
    int w_base = h * w_stride;

    int i0 = lane;
    int i1 = lane + 32;
    int i2 = lane + 64;
    int i3 = lane + 96;

    float x0 = __half2float(x[base + i0]);
    float x1 = __half2float(x[base + i1]);
    float x2 = __half2float(x[base + i2]);
    float x3 = __half2float(x[base + i3]);

    float sum_sq = x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down(sum_sq, offset, 32);
    }
    float rsqrt_val = rsqrtf(__shfl(sum_sq, 0, 32) / (float)head_dim + eps);

    float w0 = __half2float(w[w_base + i0]);
    float w1 = __half2float(w[w_base + i1]);
    float w2 = __half2float(w[w_base + i2]);
    float w3 = __half2float(w[w_base + i3]);

    float g0 = __half2float(gate[base + i0]);
    float g1 = __half2float(gate[base + i1]);
    float g2 = __half2float(gate[base + i2]);
    float g3 = __half2float(gate[base + i3]);

    y[base + i0] = __float2half(x0 * rsqrt_val * w0 * silu(g0));
    y[base + i1] = __float2half(x1 * rsqrt_val * w1 * silu(g1));
    y[base + i2] = __float2half(x2 * rsqrt_val * w2 * silu(g2));
    y[base + i3] = __float2half(x3 * rsqrt_val * w3 * silu(g3));
}

} // namespace

extern "C" {

void launch_delta_net_decode(
    const float* q,
    const float* k,
    const float* v,
    const float* decay,
    const float* beta,
    float* S,
    void* y,
    float scale,
    hipStream_t stream
) {
    dim3 grid(48, 128);
    dim3 block(32, 1);
    hipLaunchKernelGGL(delta_net_decode_kernel, grid, block, 0, stream, q, k, v, decay, beta, S, (half*)y, scale);
}

void launch_fast_ssm_conv1d(
    const void* x,
    void* conv_state,
    const void* conv_weight,
    void* y,
    int num_channels,
    hipStream_t stream
) {
    dim3 block(256);
    dim3 grid((num_channels + 255) / 256);
    hipLaunchKernelGGL(fast_ssm_conv1d_kernel, grid, block, 0, stream, (const half*)x, (half*)conv_state, (const half*)conv_weight, (half*)y, num_channels);
}

void launch_fast_rms_norm(
    const void* x,
    const void* w,
    void* y,
    int rows, int D, float eps,
    hipStream_t stream
) {
    dim3 block(256);
    dim3 grid(rows);
    hipLaunchKernelGGL(fast_rms_norm_kernel, grid, block, 0, stream, (const half*)x, (const half*)w, (half*)y, D, eps);
}

void launch_fast_rms_norm_gated(
    const void* x,
    const void* w,
    const void* gate,
    void* y,
    int num_heads, int head_dim, float eps,
    int w_stride,
    hipStream_t stream
) {
    dim3 block(32);
    dim3 grid(num_heads);
    hipLaunchKernelGGL(fast_rms_norm_gated_kernel, grid, block, 0, stream, (const half*)x, (const half*)w, (const half*)gate, (half*)y, num_heads, head_dim, eps, w_stride);
}

}
