// torch_ext/mxfp8_kernels.hip — OCP MXFP8 true UE8M0, optimized for gfx12 w32
// Block=32, 1 warp (32 threads) per scale block, shuffle reduce, 1 load/q
// 8.25 bpw, no FP32-scale emulation. Quant: real->ue8m0+fp8, Dequant: fp8*2^(ue-127)

#include <hip/hip_runtime.h>
#include <assert.h>
#include <math.h>
#include "../hip_quant_util.h"

#define HIP_QUANT_DTYPE_F32  0
#define HIP_QUANT_DTYPE_F16  1
#define HIP_QUANT_DTYPE_BF16 2

template <typename T>
static __device__ __forceinline__ T mx_ld(const T* p) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(HIP_QUANT_ENABLE_NONTEMPORAL) && !defined(HIP_QUANT_DISABLE_NONTEMPORAL)
    return __builtin_nontemporal_load(p);
#else
    return *p;
#endif
}
template <typename T>
static __device__ __forceinline__ void mx_st(T* p, T v) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(HIP_QUANT_ENABLE_NONTEMPORAL) && !defined(HIP_QUANT_DISABLE_NONTEMPORAL)
    __builtin_nontemporal_store(v, p);
#else
    *p = v;
#endif
}
template <int DTYPE>
static __device__ inline float ld_f32(const void* s, int64_t i) {
    if constexpr (DTYPE==HIP_QUANT_DTYPE_F16) return fp16_to_fp32(mx_ld(((const uint16_t*)s)+i));
    if constexpr (DTYPE==HIP_QUANT_DTYPE_BF16) return bf16_to_fp32(mx_ld(((const uint16_t*)s)+i));
    return mx_ld(((const float*)s)+i);
}

// ---- UE8M0 helpers — branchless fast path, subnormal fallback ----
static __device__ inline uint8_t amax_to_ue8m0(float amax, float max_elem) {
    if (amax <= 0.0f || !isfinite(amax)) return 0;
    float ratio = amax / max_elem;
    uint32_t u = __float_as_uint(ratio);
    uint32_t eb = (u >> 23) & 0xFFu;
    if (eb==0u) { // tiny amax < ~5e-37, ~1/(10M) blocks
        float lg = log2f(ratio);
        int ce = (int)ceilf(lg - 1e-7f);
        int ue = ce + 127;
        if (ue<0) ue=0; if (ue>254) ue=254;
        return (uint8_t)ue;
    }
    int exp = (int)eb - 127;
    uint32_t mant = u & 0x007FFFFFu;
    int ce = (mant==0u) ? exp : exp+1;
    int ue = ce + 127;
    if (ue<0) ue=0; if (ue>254) ue=254;
    return (uint8_t)ue;
}
static __device__ __forceinline__ float ue_to_scale_fast(uint8_t ue) {
    if (ue==0xFF) return __int_as_float(0x7FC00000u);
    if (ue==0) return __int_as_float(0x00400000u); // 2^-127 subnormal
    return __int_as_float((uint32_t)ue << 23); // 2^(ue-127)
}
static __device__ __forceinline__ float ue_to_inv_fast(uint8_t ue) {
    if (ue==0xFF) return __int_as_float(0x7FC00000u);
    if (ue==254) return __int_as_float(0x00400000u); // 2^-127
    if (ue==0) return __int_as_float(0x7F000000u); // 2^127
    // 2^(127-ue) : exponent = 254-ue
    return __int_as_float((uint32_t)(254u - ue) << 23);
}

// ---------------------------------------------------------------------------
// Quant E4M3 — 1 warp (32 threads) per block, 1 load per lane, shuffle reduce
// ---------------------------------------------------------------------------
template <int DTYPE>
__global__ __launch_bounds__(32, 8)
void quant_mxfp8_e4m3_kernel_t(const void* __restrict__ src, uint8_t* __restrict__ dst,
                              uint8_t* __restrict__ scales, int64_t rows, int64_t cols, int64_t bpr) {
    constexpr int BLOCK=32;
    int64_t bid = blockIdx.x;
    int64_t total = rows * bpr;
    if (bid >= total) return;
    int64_t row = bid / bpr;
    int64_t bcol = bid - row*bpr;
    int64_t col0 = bcol * BLOCK;
    int64_t len = cols - col0; if (len>BLOCK) len=BLOCK; if (len<=0) return;
    int lane = threadIdx.x; // 0..31
    int64_t base = row*cols + col0;
    float v = 0.0f;
    if (lane < len) v = ld_f32<DTYPE>(src, base+lane);
    float av = fabsf(v);
    if (!isfinite(av)) av = 0.0f; // ignore NaN/Inf for amax
    // warp max reduce — 64-bit mask required by HIP (gfx12 w32 still 64-bit lane mask)
    #pragma unroll
    for (int off=16; off>0; off>>=1) av = fmaxf(av, __shfl_down_sync(0xffffffffffffffffULL, av, off));
    uint8_t ue = 0; float inv = 0;
    if (lane==0) {
        ue = amax_to_ue8m0(av, 448.0f);
        inv = ue_to_inv_fast(ue);
        mx_st(scales+bid, ue);
    }
    // broadcast ue/inv from lane0
    ue = (uint8_t)__shfl_sync(0xffffffffffffffffULL, (uint32_t)ue, 0);
    inv = __shfl_sync(0xffffffffffffffffULL, inv, 0);
    if (lane < len) {
        float vs = v * inv;
        mx_st(dst+base+lane, fp32_to_fp8_e4m3(vs));
    }
}
template <int DTYPE>
__global__ __launch_bounds__(32, 8)
void quant_mxfp8_e5m2_kernel_t(const void* __restrict__ src, uint8_t* __restrict__ dst,
                              uint8_t* __restrict__ scales, int64_t rows, int64_t cols, int64_t bpr) {
    constexpr int BLOCK=32;
    int64_t bid = blockIdx.x;
    int64_t total = rows * bpr;
    if (bid >= total) return;
    int64_t row = bid / bpr;
    int64_t bcol = bid - row*bpr;
    int64_t col0 = bcol * BLOCK;
    int64_t len = cols - col0; if (len>BLOCK) len=BLOCK; if (len<=0) return;
    int lane = threadIdx.x;
    int64_t base = row*cols + col0;
    float v = 0.0f;
    if (lane < len) v = ld_f32<DTYPE>(src, base+lane);
    float av = fabsf(v); if (!isfinite(av)) av=0;
    #pragma unroll
    for (int off=16; off>0; off>>=1) av = fmaxf(av, __shfl_down_sync(0xffffffffffffffffULL, av, off));
    uint8_t ue=0; float inv=0;
    if (lane==0) { ue=amax_to_ue8m0(av,57344.0f); inv=ue_to_inv_fast(ue); mx_st(scales+bid, ue); }
    ue=(uint8_t)__shfl_sync(0xffffffffffffffffULL,(uint32_t)ue,0);
    inv=__shfl_sync(0xffffffffffffffffULL,inv,0);
    if (lane < len) mx_st(dst+base+lane, fp32_to_fp8_e5m2(v*inv));
}

// ---------------------------------------------------------------------------
// Dequant — 1 warp per block, 1 load per lane
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(32, 8)
void dequant_mxfp8_e4m3_kernel(const uint8_t* __restrict__ src, const uint8_t* __restrict__ scales,
                              float* __restrict__ dst, int64_t rows, int64_t cols, int64_t bpr) {
    constexpr int BLOCK=32;
    int64_t bid = blockIdx.x;
    if (bid >= rows*bpr) return;
    int64_t row = bid / bpr;
    int64_t bcol = bid - row*bpr;
    int64_t col0 = bcol*BLOCK;
    int64_t len = cols-col0; if(len>BLOCK) len=BLOCK; if(len<=0) return;
    int lane = threadIdx.x;
    int64_t base = row*cols+col0;
    uint8_t ue = 0;
    if (lane==0) ue = mx_ld(scales+bid);
    ue = (uint8_t)__shfl_sync(0xffffffffffffffffULL, (uint32_t)ue, 0);
    float scale = ue_to_scale_fast(ue);
    bool is_nan = (ue==0xFF);
    if (lane < len) {
        float v = fp8_e4m3_to_fp32(mx_ld(src+base+lane));
        mx_st(dst+base+lane, is_nan ? scale : v*scale);
    }
}
__global__ __launch_bounds__(32, 8)
void dequant_mxfp8_e5m2_kernel(const uint8_t* __restrict__ src, const uint8_t* __restrict__ scales,
                              float* __restrict__ dst, int64_t rows, int64_t cols, int64_t bpr) {
    constexpr int BLOCK=32;
    int64_t bid = blockIdx.x;
    if (bid >= rows*bpr) return;
    int64_t row = bid / bpr;
    int64_t bcol = bid - row*bpr;
    int64_t col0 = bcol*BLOCK;
    int64_t len = cols-col0; if(len>BLOCK) len=BLOCK; if(len<=0) return;
    int lane = threadIdx.x;
    int64_t base = row*cols+col0;
    uint8_t ue = 0;
    if (lane==0) ue = mx_ld(scales+bid);
    ue = (uint8_t)__shfl_sync(0xffffffffffffffffULL, (uint32_t)ue, 0);
    float scale = ue_to_scale_fast(ue);
    bool is_nan = (ue==0xFF);
    if (lane < len) {
        float v = fp8_e5m2_to_fp32(mx_ld(src+base+lane));
        mx_st(dst+base+lane, is_nan ? scale : v*scale);
    }
}

static inline uint32_t safe_mx(int64_t r,int64_t b){int64_t b2=r*b; assert(b2<=(int64_t)0xFFFFFFFFu); return (uint32_t)b2;}
extern "C" {
void launch_quant_mxfp8_e4m3(const void* s,uint8_t* d,uint8_t* sc,int64_t r,int64_t c,int64_t b,int dtype,hipStream_t st){
    if(r==0) return; uint32_t g=safe_mx(r,b);
    if(dtype==HIP_QUANT_DTYPE_F16) hipLaunchKernelGGL(HIP_KERNEL_NAME(quant_mxfp8_e4m3_kernel_t<HIP_QUANT_DTYPE_F16>),dim3(g),dim3(32),0,st,s,d,sc,r,c,b);
    else if(dtype==HIP_QUANT_DTYPE_BF16) hipLaunchKernelGGL(HIP_KERNEL_NAME(quant_mxfp8_e4m3_kernel_t<HIP_QUANT_DTYPE_BF16>),dim3(g),dim3(32),0,st,s,d,sc,r,c,b);
    else hipLaunchKernelGGL(HIP_KERNEL_NAME(quant_mxfp8_e4m3_kernel_t<HIP_QUANT_DTYPE_F32>),dim3(g),dim3(32),0,st,s,d,sc,r,c,b);
}
void launch_quant_mxfp8_e5m2(const void* s,uint8_t* d,uint8_t* sc,int64_t r,int64_t c,int64_t b,int dtype,hipStream_t st){
    if(r==0) return; uint32_t g=safe_mx(r,b);
    if(dtype==HIP_QUANT_DTYPE_F16) hipLaunchKernelGGL(HIP_KERNEL_NAME(quant_mxfp8_e5m2_kernel_t<HIP_QUANT_DTYPE_F16>),dim3(g),dim3(32),0,st,s,d,sc,r,c,b);
    else if(dtype==HIP_QUANT_DTYPE_BF16) hipLaunchKernelGGL(HIP_KERNEL_NAME(quant_mxfp8_e5m2_kernel_t<HIP_QUANT_DTYPE_BF16>),dim3(g),dim3(32),0,st,s,d,sc,r,c,b);
    else hipLaunchKernelGGL(HIP_KERNEL_NAME(quant_mxfp8_e5m2_kernel_t<HIP_QUANT_DTYPE_F32>),dim3(g),dim3(32),0,st,s,d,sc,r,c,b);
}
void launch_dequant_mxfp8_e4m3(const uint8_t* s,const uint8_t* sc,float* d,int64_t r,int64_t c,int64_t b,hipStream_t st){
    if(r==0) return; hipLaunchKernelGGL(dequant_mxfp8_e4m3_kernel,dim3(safe_mx(r,b)),dim3(32),0,st,s,sc,d,r,c,b);
}
void launch_dequant_mxfp8_e5m2(const uint8_t* s,const uint8_t* sc,float* d,int64_t r,int64_t c,int64_t b,hipStream_t st){
    if(r==0) return; hipLaunchKernelGGL(dequant_mxfp8_e5m2_kernel,dim3(safe_mx(r,b)),dim3(32),0,st,s,sc,d,r,c,b);
}
}
