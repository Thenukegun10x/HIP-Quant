#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>

// Round to nearest int, ties to even (matches CPU `nearest_int` from ggml)
__device__ inline int nearest_int(float f) {
    float v = f + 12582912.0f;
    int i = __float_as_int(v);
    return (i & 0x007fffff) - 0x00400000;
}

// float32 → IEEE 754 half (round-to-nearest-even, matches _cvtss_sh on CPU)
__device__ inline uint16_t fp32_to_fp16(float f) {
    uint32_t u = __float_as_int(f);
    uint32_t sign = (u >> 16) & 0x8000;
    int32_t exp = ((u >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = u & 0x007FFFFF;

    if (exp > 30) {
        return (uint16_t)(sign | 0x7C00 | (mant ? 0x200 : 0));
    }

    if (exp <= 0) {
        mant = (mant | 0x800000) >> (1 - exp);
        if (mant == 0) return (uint16_t)sign;
        while (!(mant & 0x3E00000)) mant <<= 1;
        exp = 1;
        mant >>= 13;
        return (uint16_t)(sign | (exp << 10) | (mant & 0x3FF));
    }

    uint32_t rnd = mant & 0x1FFF;
    mant >>= 13;
    if (rnd > 0x1000 || (rnd == 0x1000 && (mant & 1))) {
        mant++;
        if (mant & 0x400) { mant = 0; exp++; }
    }

    if (exp >= 30) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | (exp << 10) | (mant & 0x3FF));
}

// IEEE 754 half → float32
__device__ inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    int32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0 && mant == 0) return 0.0f;
    if (exp == 0) {
        while (!(mant & 0x400)) { mant <<= 1; exp--; }
        mant &= 0x3FF;
        exp += 1;
    }
    uint32_t u = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    return __int_as_float(u);
}
