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

// ============ FP8 E4M3 (OCP standard) conversions ============
// Layout: 1 sign | 4 exponent | 3 mantissa
// Bias: 7, max finite: ±448, no infinities
// NaN: only S.1111.111 (0x7F / 0xFF)

// float32 → FP8 E4M3 (round-to-nearest-even, saturate to max finite)
__device__ inline uint8_t fp32_to_fp8_e4m3(float f) {
    uint32_t u = __float_as_int(f);
    uint32_t sign = u >> 31;
    uint32_t abs_u = u & 0x7FFFFFFF;

    // Zero
    if (abs_u == 0) return (uint8_t)(sign << 7);

    // NaN → FP8 NaN
    if (abs_u > 0x7F800000) return (uint8_t)((sign << 7) | 0x7F);

    // Inf → saturate to max finite (±448)
    if (abs_u == 0x7F800000) return (uint8_t)((sign << 7) | 0x7E);

    int32_t f32_exp = (int32_t)((abs_u >> 23) & 0xFF);
    uint32_t f32_mant = abs_u & 0x7FFFFF;

    // F32 subnormal → too small for FP8, return zero
    if (f32_exp == 0) return (uint8_t)(sign << 7);

    // Rebias: F32 bias=127, FP8 E4M3 bias=7
    int32_t exp = f32_exp - 127 + 7;

    if (exp <= 0) {
        // FP8 subnormal or underflow
        int shift = 1 - exp;
        if (shift > 4) return (uint8_t)(sign << 7); // too small → zero
        // Add implicit 1 (now 24 bits), shift to get 3-bit mantissa
        uint32_t full = 0x800000 | f32_mant;
        int total_shift = 20 + shift;
        uint32_t result = full >> total_shift;
        // Round-to-nearest-even
        uint32_t remainder = full & ((1u << total_shift) - 1);
        uint32_t midpoint = 1u << (total_shift - 1);
        if (remainder > midpoint || (remainder == midpoint && (result & 1))) {
            result++;
        }
        if (result >= 8) {
            // Rounded up to smallest normal
            return (uint8_t)((sign << 7) | (1 << 3));
        }
        return (uint8_t)((sign << 7) | (result & 0x7));
    }

    // Normal: round mantissa from 23 bits to 3 bits (round-to-nearest-even)
    uint32_t fp8_mant = (f32_mant >> 20) & 0x7;
    uint32_t rnd = f32_mant & 0xFFFFF;
    if (rnd > 0x80000 || (rnd == 0x80000 && (fp8_mant & 1))) {
        fp8_mant++;
        if (fp8_mant >= 8) { fp8_mant = 0; exp++; }
    }

    // Post-rounding overflow: saturate (also avoids NaN at exp=15 mant=7)
    if (exp >= 16 || (exp == 15 && fp8_mant == 7)) {
        return (uint8_t)((sign << 7) | 0x7E);
    }

    return (uint8_t)((sign << 7) | (exp << 3) | fp8_mant);
}

// FP8 E4M3 → float32
__device__ inline float fp8_e4m3_to_fp32(uint8_t h) {
    uint32_t sign = (h >> 7) & 1;
    uint32_t exp = (h >> 3) & 0xF;
    uint32_t mant = h & 0x7;

    // NaN (only encoding: S.1111.111)
    if (exp == 15 && mant == 7) {
        return __int_as_float((sign << 31) | 0x7FC00000);
    }

    // Zero
    if (exp == 0 && mant == 0) {
        return __int_as_float(sign << 31);
    }

    if (exp == 0) {
        // Subnormal: value = (-1)^S * 2^(-6) * (mant / 8) = mant * 2^(-9)
        float result = (float)mant * 0.001953125f; // 2^(-9)
        return sign ? -result : result;
    }

    // Normal: rebias exp from FP8 (bias=7) to F32 (bias=127): f32_exp = exp + 120
    // Mantissa: 3 bits → 23 bits (shift left by 20)
    uint32_t f32 = (sign << 31) | ((exp + 120) << 23) | (mant << 20);
    return __int_as_float(f32);
}

// ============ FP8 E5M2 (IEEE/OCP standard) conversions ============
// Layout: 1 sign | 5 exponent | 2 mantissa
// Bias: 15, max finite: +/-57344, infinities supported
// Inf: S.11111.00, NaN: S.11111.xx where xx != 00

// float32 -> FP8 E5M2 (round-to-nearest-even, overflow to infinity)
__device__ inline uint8_t fp32_to_fp8_e5m2(float f) {
    uint32_t u = __float_as_int(f);
    uint32_t sign = u >> 31;
    uint32_t abs_u = u & 0x7FFFFFFF;

    // Zero
    if (abs_u == 0) return (uint8_t)(sign << 7);

    // NaN -> canonical FP8 NaN
    if (abs_u > 0x7F800000) return (uint8_t)((sign << 7) | 0x7F);

    // Inf -> FP8 Inf
    if (abs_u == 0x7F800000) return (uint8_t)((sign << 7) | 0x7C);

    int32_t f32_exp = (int32_t)((abs_u >> 23) & 0xFF);
    uint32_t f32_mant = abs_u & 0x7FFFFF;

    // F32 subnormal -> too small for FP8, return zero
    if (f32_exp == 0) return (uint8_t)(sign << 7);

    // Rebias: F32 bias=127, FP8 E5M2 bias=15
    int32_t exp = f32_exp - 127 + 15;

    if (exp <= 0) {
        // FP8 subnormal or underflow. Subnormal unit is 2^-16.
        int shift = 1 - exp;
        if (shift > 3) return (uint8_t)(sign << 7); // less than half min subnormal

        uint32_t full = 0x800000 | f32_mant;
        int total_shift = 21 + shift;
        uint32_t result = full >> total_shift;

        uint32_t remainder = full & ((1u << total_shift) - 1);
        uint32_t midpoint = 1u << (total_shift - 1);
        if (remainder > midpoint || (remainder == midpoint && (result & 1))) {
            result++;
        }
        if (result >= 4) {
            // Rounded up to smallest normal.
            return (uint8_t)((sign << 7) | (1 << 2));
        }
        return (uint8_t)((sign << 7) | (result & 0x3));
    }

    // Normal: round mantissa from 23 bits to 2 bits (round-to-nearest-even)
    uint32_t fp8_mant = (f32_mant >> 21) & 0x3;
    uint32_t rnd = f32_mant & 0x1FFFFF;
    if (rnd > 0x100000 || (rnd == 0x100000 && (fp8_mant & 1))) {
        fp8_mant++;
        if (fp8_mant >= 4) { fp8_mant = 0; exp++; }
    }

    // Post-rounding overflow: E5M2 supports infinity.
    if (exp >= 31) {
        return (uint8_t)((sign << 7) | 0x7C);
    }

    return (uint8_t)((sign << 7) | (exp << 2) | fp8_mant);
}

// FP8 E5M2 -> float32
__device__ inline float fp8_e5m2_to_fp32(uint8_t h) {
    uint32_t sign = (h >> 7) & 1;
    uint32_t exp = (h >> 2) & 0x1F;
    uint32_t mant = h & 0x3;

    if (exp == 31) {
        if (mant == 0) {
            return __int_as_float((sign << 31) | 0x7F800000);
        }
        return __int_as_float((sign << 31) | 0x7FC00000);
    }

    // Zero
    if (exp == 0 && mant == 0) {
        return __int_as_float(sign << 31);
    }

    if (exp == 0) {
        // Subnormal: value = (-1)^S * 2^(-14) * (mant / 4) = mant * 2^(-16)
        float result = (float)mant * 0.0000152587890625f; // 2^(-16)
        return sign ? -result : result;
    }

    // Normal: rebias exp from FP8 (bias=15) to F32 (bias=127): f32_exp = exp + 112
    // Mantissa: 2 bits -> 23 bits (shift left by 21)
    uint32_t f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 21);
    return __int_as_float(f32);
}
