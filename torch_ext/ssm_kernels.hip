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
    const float* __restrict__ q,       // [16 or 48, 128]
    const float* __restrict__ k,       // [16 or 48, 128]
    const float* __restrict__ v,       // [48, 128]
    const float* __restrict__ decay,   // [48]
    const float* __restrict__ beta,    // [48]
    float*       __restrict__ S,       // [48, 128, 128]
    half*        __restrict__ y,       // [48, 128]
    float scale,
    int qk_heads = 48
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
    int qk_head = (qk_heads == 16) ? (h % 16) : h;
    int base_qk = qk_head * 128;

    float k0 = k[base_qk + r0];
    float k1 = k[base_qk + r1];
    float k2 = k[base_qk + r2];
    float k3 = k[base_qk + r3];

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

    float q0 = q[base_qk + r0];
    float q1 = q[base_qk + r1];
    float q2 = q[base_qk + r2];
    float q3 = q[base_qk + r3];

    float attn_part = s0 * q0 + s1 * q1 + s2 * q2 + s3 * q3;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        attn_part += __shfl_down(attn_part, offset, 32);
    }

    if (lane == 0) {
        y[base_vec + j] = __float2half(attn_part * scale);
    }
}

// Fused multi-token Gated DeltaNet recurrence for prefill.  A block owns one
// head and eight state columns.  Q/K/V are loaded once into LDS and reused by
// all eight warps; this is important because the recurrence is sequential in
// time but embarrassingly parallel across state columns.
__global__ void delta_net_prefill_kernel(
    const float* __restrict__ q,       // [S, qk_heads, 128]
    const float* __restrict__ k,       // [S, qk_heads, 128]
    const float* __restrict__ v,       // [S, 48, 128]
    const float* __restrict__ decay,   // [S, 48]
    const float* __restrict__ beta,    // [S, 48]
    float*       __restrict__ S_state, // [48, 128, 128]
    half*        __restrict__ y,       // [S, 48, 128]
    int seq_len,
    int qk_heads,
    float scale
) {
    constexpr int COLS_PER_BLOCK = 8;
    const int tile = blockIdx.x;
    const int h = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int j = tile * COLS_PER_BLOCK + warp;

    __shared__ float k_tile[128];
    __shared__ float q_tile[128];
    __shared__ float v_tile[128];

    const int r0 = lane;
    const int r1 = lane + 32;
    const int r2 = lane + 64;
    const int r3 = lane + 96;
    const int state_base = h * (128 * 128);
    const int qk_head = (qk_heads == 16) ? (h % 16) : h;

    float s0 = S_state[state_base + r0 * 128 + j];
    float s1 = S_state[state_base + r1 * 128 + j];
    float s2 = S_state[state_base + r2 * 128 + j];
    float s3 = S_state[state_base + r3 * 128 + j];

    for (int t = 0; t < seq_len; ++t) {
        const int qk_base = (t * qk_heads + qk_head) * 128;
        const int v_base = (t * 48 + h) * 128;
        const int scalar_base = t * 48 + h;

        // The first half of the block cooperatively stages all three vectors.
        // Every warp then reuses them for its independent state column.
        if (threadIdx.x < 128) {
            k_tile[threadIdx.x] = k[qk_base + threadIdx.x];
            q_tile[threadIdx.x] = q[qk_base + threadIdx.x];
            v_tile[threadIdx.x] = v[v_base + threadIdx.x];
        }
        __syncthreads();

        const float k0 = k_tile[r0];
        const float k1 = k_tile[r1];
        const float k2 = k_tile[r2];
        const float k3 = k_tile[r3];

        float kv_part = s0 * k0 + s1 * k1 + s2 * k2 + s3 * k3;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            kv_part += __shfl_down(kv_part, offset, 32);
        }
        const float kv_j = __shfl(kv_part, 0, 32);

        const float d_val = decay[scalar_base];
        const float b_val = beta[scalar_base];
        const float delta_j = (v_tile[j] - d_val * kv_j) * b_val;

        s0 = d_val * s0 + k0 * delta_j;
        s1 = d_val * s1 + k1 * delta_j;
        s2 = d_val * s2 + k2 * delta_j;
        s3 = d_val * s3 + k3 * delta_j;

        const float q0 = q_tile[r0];
        const float q1 = q_tile[r1];
        const float q2 = q_tile[r2];
        const float q3 = q_tile[r3];
        float attn_part = s0 * q0 + s1 * q1 + s2 * q2 + s3 * q3;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            attn_part += __shfl_down(attn_part, offset, 32);
        }
        if (lane == 0) {
            y[v_base + j] = __float2half(attn_part * scale);
        }
        __syncthreads();
    }

    S_state[state_base + r0 * 128 + j] = s0;
    S_state[state_base + r1 * 128 + j] = s1;
    S_state[state_base + r2 * 128 + j] = s2;
    S_state[state_base + r3 * 128 + j] = s3;
}

// Fused DeltaNet Q/K normalization + V float conversion (supports decode seq_len=1 and prefill seq_len>1)
__global__ void fused_delta_net_prep_kernel(
    const half* __restrict__ conv_out, // [seq_len, 10240]
    float*      __restrict__ q_norm,   // [seq_len, 16, 128]
    float*      __restrict__ k_norm,   // [seq_len, 16, 128]
    float*      __restrict__ v_float,  // [seq_len, 48, 128]
    int seq_len
) {
    int head = blockIdx.x; // 0..79
    int t = blockIdx.y;    // 0..seq_len-1
    int lane = threadIdx.x; // 0..31
    if (head >= 80 || t >= seq_len) return;

    const half* in_t = conv_out + (int64_t)t * 10240;

    if (head < 16) {
        const half* in_ptr = in_t + head * 128;
        float* out_ptr = q_norm + ((int64_t)t * 16 + head) * 128;
        float v0 = __half2float(in_ptr[lane]);
        float v1 = __half2float(in_ptr[lane + 32]);
        float v2 = __half2float(in_ptr[lane + 64]);
        float v3 = __half2float(in_ptr[lane + 96]);

        float sum_sq = v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_sq += __shfl_down(sum_sq, offset, 32);
        }
        float inv_norm = rsqrtf(fmaxf(__shfl(sum_sq, 0, 32), 1e-12f));

        out_ptr[lane]      = v0 * inv_norm;
        out_ptr[lane + 32] = v1 * inv_norm;
        out_ptr[lane + 64] = v2 * inv_norm;
        out_ptr[lane + 96] = v3 * inv_norm;
    } else if (head < 32) {
        int k_head = head - 16;
        const half* in_ptr = in_t + 2048 + k_head * 128;
        float* out_ptr = k_norm + ((int64_t)t * 16 + k_head) * 128;
        float v0 = __half2float(in_ptr[lane]);
        float v1 = __half2float(in_ptr[lane + 32]);
        float v2 = __half2float(in_ptr[lane + 64]);
        float v3 = __half2float(in_ptr[lane + 96]);

        float sum_sq = v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_sq += __shfl_down(sum_sq, offset, 32);
        }
        float inv_norm = rsqrtf(fmaxf(__shfl(sum_sq, 0, 32), 1e-12f));

        out_ptr[lane]      = v0 * inv_norm;
        out_ptr[lane + 32] = v1 * inv_norm;
        out_ptr[lane + 64] = v2 * inv_norm;
        out_ptr[lane + 96] = v3 * inv_norm;
    } else {
        int v_head = head - 32;
        const half* in_ptr = in_t + 4096 + v_head * 128;
        float* out_ptr = v_float + ((int64_t)t * 48 + v_head) * 128;

        out_ptr[lane]      = __half2float(in_ptr[lane]);
        out_ptr[lane + 32] = __half2float(in_ptr[lane + 32]);
        out_ptr[lane + 64] = __half2float(in_ptr[lane + 64]);
        out_ptr[lane + 96] = __half2float(in_ptr[lane + 96]);
    }
}

// Fused SSM Gating: decay = exp(ssm_a * softplus(alpha + dt_bias)), beta = sigmoid(beta_raw)
__global__ void fused_ssm_gating_kernel(
    const half*  __restrict__ beta_raw,  // [seq_len, 48]
    const half*  __restrict__ alpha_raw, // [seq_len, 48]
    const float* __restrict__ dt_bias,   // [48]
    const float* __restrict__ ssm_a,     // [48]
    float*       __restrict__ decay_out, // [seq_len, 48]
    float*       __restrict__ beta_out,  // [seq_len, 48]
    int seq_len
) {
    int h = threadIdx.x;
    int t = blockIdx.x;
    if (h >= 48 || t >= seq_len) return;

    int64_t idx = (int64_t)t * 48 + h;
    float b = __half2float(beta_raw[idx]);
    float a = __half2float(alpha_raw[idx]);
    float a_biased = a + dt_bias[h];
    float sp = (a_biased > 20.0f) ? a_biased : log1pf(expf(a_biased));
    float gate = ssm_a[h] * sp;
    decay_out[idx] = expf(gate);
    beta_out[idx] = 1.0f / (1.0f + expf(-b));
}

// ---------------------------------------------------------------------------
// 2. Depthwise Conv1D Shift + SiLU Kernel (vectorized for any seq_len)
// ---------------------------------------------------------------------------
__global__ void fast_ssm_conv1d_kernel(
    const half* __restrict__ x,           // [seq_len, num_channels]
    half*       __restrict__ conv_state,  // [num_channels, 3]
    const half* __restrict__ conv_weight, // [num_channels, 4]
    half*       __restrict__ y,           // [seq_len, num_channels]
    int num_channels,
    int seq_len
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= num_channels) return;

    int state_idx = c * 3;
    int weight_idx = c * 4;

    float s0 = __half2float(conv_state[state_idx + 0]);
    float s1 = __half2float(conv_state[state_idx + 1]);
    float s2 = __half2float(conv_state[state_idx + 2]);

    float w0 = __half2float(conv_weight[weight_idx + 0]);
    float w1 = __half2float(conv_weight[weight_idx + 1]);
    float w2 = __half2float(conv_weight[weight_idx + 2]);
    float w3 = __half2float(conv_weight[weight_idx + 3]);

    for (int t = 0; t < seq_len; ++t) {
        float in = __half2float(x[(int64_t)t * num_channels + c]);
        float conv_out = s0 * w0 + s1 * w1 + s2 * w2 + in * w3;
        y[(int64_t)t * num_channels + c] = __float2half(silu(conv_out));

        s0 = s1;
        s1 = s2;
        s2 = in;
    }

    // Shift state in-place
    conv_state[state_idx + 0] = __float2half(s0);
    conv_state[state_idx + 1] = __float2half(s1);
    conv_state[state_idx + 2] = __float2half(s2);
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
    int num_heads, int head_dim, float eps, int w_stride,
    int total_rows
) {
    int row = blockIdx.x; // row 0 .. total_rows-1
    if (row >= total_rows) return;

    int h = row % num_heads;
    int lane = threadIdx.x; // 0..31
    int64_t base = (int64_t)row * head_dim;
    int64_t w_base = (int64_t)h * w_stride;

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
    int qk_heads,
    hipStream_t stream
) {
    dim3 grid(48, 128);
    dim3 block(32, 1);
    hipLaunchKernelGGL(delta_net_decode_kernel, grid, block, 0, stream, q, k, v, decay, beta, S, (half*)y, scale, qk_heads);
}

void launch_delta_net_prefill(
    const float* q,
    const float* k,
    const float* v,
    const float* decay,
    const float* beta,
    float* S,
    void* y,
    int seq_len,
    int qk_heads,
    float scale,
    hipStream_t stream
) {
    dim3 grid(16, 48);
    dim3 block(256, 1);
    hipLaunchKernelGGL(delta_net_prefill_kernel, grid, block, 0, stream,
                       q, k, v, decay, beta, S, (half*)y,
                       seq_len, qk_heads, scale);
}

void launch_fused_delta_net_prep(
    const void* conv_out,
    float* q_norm,
    float* k_norm,
    float* v_float,
    int seq_len,
    hipStream_t stream
) {
    dim3 grid(80, seq_len);
    dim3 block(32);
    hipLaunchKernelGGL(fused_delta_net_prep_kernel, grid, block, 0, stream,
                       (const half*)conv_out, q_norm, k_norm, v_float, seq_len);
}

void launch_fused_ssm_gating(
    const void* beta_raw,
    const void* alpha_raw,
    const float* dt_bias,
    const float* ssm_a,
    float* decay_out,
    float* beta_out,
    int seq_len,
    hipStream_t stream
) {
    dim3 grid(seq_len);
    dim3 block(64);
    hipLaunchKernelGGL(fused_ssm_gating_kernel, grid, block, 0, stream,
                       (const half*)beta_raw, (const half*)alpha_raw, dt_bias, ssm_a,
                       decay_out, beta_out, seq_len);
}

void launch_fast_ssm_conv1d(
    const void* x,
    void* conv_state,
    const void* conv_weight,
    void* y,
    int num_channels,
    int seq_len,
    hipStream_t stream
) {
    dim3 block(256);
    dim3 grid((num_channels + 255) / 256);
    hipLaunchKernelGGL(fast_ssm_conv1d_kernel, grid, block, 0, stream,
                       (const half*)x, (half*)conv_state, (const half*)conv_weight,
                       (half*)y, num_channels, seq_len);
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
    int w_stride, int total_rows,
    hipStream_t stream
) {
    dim3 block(32);
    dim3 grid(total_rows);
    hipLaunchKernelGGL(fast_rms_norm_gated_kernel, grid, block, 0, stream,
                       (const half*)x, (const half*)w, (const half*)gate,
                       (half*)y, num_heads, head_dim, eps, w_stride, total_rows);
}

}
