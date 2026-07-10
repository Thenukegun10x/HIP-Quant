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
    uint32_t f32_exp = (u >> 23) & 0xFF;
    uint32_t mant = u & 0x007FFFFF;

    if (f32_exp == 0xFF) {
        // Preserve infinities and emit a quiet half NaN for every F32 NaN.
        return (uint16_t)(sign | 0x7C00 | (mant ? 0x0200 : 0));
    }

    int32_t exp = (int32_t)f32_exp - 127 + 15;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);

    if (exp <= 0) {
        // Half subnormal. exp == -10 is exactly the range containing the
        // midpoint between zero and the smallest half subnormal.
        if (exp < -10) return (uint16_t)sign;
        uint32_t full = mant | 0x00800000;
        int shift = 14 - exp;
        uint32_t half_mant = full >> shift;
        uint32_t remainder = full & ((1u << shift) - 1u);
        uint32_t midpoint = 1u << (shift - 1);
        if (remainder > midpoint || (remainder == midpoint && (half_mant & 1u))) {
            half_mant++;
        }
        // A rounded value of 0x400 is the smallest normal half.
        return (uint16_t)(sign | half_mant);
    }

    uint32_t rnd = mant & 0x1FFF;
    uint32_t half_mant = mant >> 13;
    if (rnd > 0x1000 || (rnd == 0x1000 && (half_mant & 1u))) {
        half_mant++;
        if (half_mant == 0x400) {
            half_mant = 0;
            exp++;
        }
    }

    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | half_mant);
}

// IEEE 754 half → float32
__device__ inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0 && mant == 0) return __int_as_float(sign);
    if (exp == 31) {
        uint32_t payload = mant ? ((mant << 13) | 0x00400000u) : 0u;
        return __int_as_float(sign | 0x7F800000u | payload);
    }
    if (exp == 0) {
        int32_t unbiased_exp = -14;
        while (!(mant & 0x400)) {
            mant <<= 1;
            unbiased_exp--;
        }
        mant &= 0x3FF;
        uint32_t u = sign | ((uint32_t)(unbiased_exp + 127) << 23) | (mant << 13);
        return __int_as_float(u);
    }
    uint32_t u = sign | ((exp + 112) << 23) | (mant << 13);
    return __int_as_float(u);
}

// float32 -> bfloat16 (round-to-nearest-even)
__device__ inline uint16_t fp32_to_bf16(float f) {
    uint32_t u = __float_as_int(f);
    if ((u & 0x7FFFFFFF) > 0x7F800000) {
        return (uint16_t)((u >> 16) | 0x0040); // quiet NaN
    }
    uint32_t lsb = (u >> 16) & 1;
    return (uint16_t)((u + 0x7FFFu + lsb) >> 16);
}

// bfloat16 -> float32
__device__ inline float bf16_to_fp32(uint16_t h) {
    return __int_as_float((uint32_t)h << 16);
}

// ============ FP8 E4M3 (OCP standard) conversions ============
// Layout: 1 sign | 4 exponent | 3 mantissa
// Bias: 7, max finite: ±448, no infinities
// NaN: only S.1111.111 (0x7F / 0xFF)

// float32 bits → FP8 E4M3 (round-to-nearest-even, saturate to max finite)
__device__ inline uint8_t fp32_bits_to_fp8_e4m3(uint32_t u) {
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

// float32 → FP8 E4M3 (round-to-nearest-even, saturate to max finite)
__device__ inline uint8_t fp32_to_fp8_e4m3(float f) {
    return fp32_bits_to_fp8_e4m3(__float_as_int(f));
}

// bfloat16 → FP8 E4M3 without first materializing a float value.
__device__ inline uint8_t bf16_to_fp8_e4m3(uint16_t h) {
    return fp32_bits_to_fp8_e4m3((uint32_t)h << 16);
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

// float32 bits -> FP8 E5M2 (round-to-nearest-even, finite overflow saturates)
__device__ inline uint8_t fp32_bits_to_fp8_e5m2(uint32_t u) {
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

    // Saturate finite overflow so a large finite gradient cannot become Inf.
    // Actual F32 infinities are preserved by the special case above.
    if (exp >= 31) {
        return (uint8_t)((sign << 7) | 0x7B);
    }

    return (uint8_t)((sign << 7) | (exp << 2) | fp8_mant);
}

// float32 -> FP8 E5M2 (round-to-nearest-even, finite overflow saturates)
__device__ inline uint8_t fp32_to_fp8_e5m2(float f) {
    return fp32_bits_to_fp8_e5m2(__float_as_int(f));
}

__device__ inline uint32_t hip_quant_splitmix32(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    x = x ^ (x >> 31);
    return (uint32_t)(x >> 32);
}

__device__ inline float hip_quant_uniform01(uint64_t seed, uint64_t idx) {
    uint32_t r = hip_quant_splitmix32(seed ^ (idx * 0xD1B54A32D192ED03ull));
    return (float)(r >> 8) * (1.0f / 16777216.0f);
}

__device__ inline float fp8_e5m2_positive_code_to_fp32(uint8_t code) {
    uint32_t exp = (code >> 2) & 0x1F;
    uint32_t mant = code & 0x3;

    if (exp == 0) {
        return mant == 0 ? 0.0f : (float)mant * 0x1.0p-16f;
    }
    return ldexpf(1.0f + 0.25f * (float)mant, (int)exp - 15);
}

// float32 -> FP8 E5M2 with unbiased stochastic rounding between adjacent bins.
// NaN/Inf and finite overflow remain deterministic; finite in-range values round up
// with probability proportional to their distance from the lower E5M2 value.
__device__ inline uint8_t fp32_to_fp8_e5m2_stochastic(float f, uint64_t seed, uint64_t idx) {
    uint32_t u = __float_as_int(f);
    uint32_t sign = u >> 31;
    uint32_t abs_u = u & 0x7FFFFFFF;

    if (abs_u == 0) return (uint8_t)(sign << 7);
    if (abs_u > 0x7F800000) return (uint8_t)((sign << 7) | 0x7F);
    if (abs_u == 0x7F800000) return (uint8_t)((sign << 7) | 0x7C);

    float af = fabsf(f);
    if (af > 57344.0f) return (uint8_t)((sign << 7) | 0x7B);

    uint8_t lo = 0;
    uint8_t hi = 0x7B; // Largest positive finite E5M2 code.
    while (lo < hi) {
        uint8_t mid = (uint8_t)((lo + hi + 1) >> 1);
        if (fp8_e5m2_positive_code_to_fp32(mid) <= af) {
            lo = mid;
        } else {
            hi = (uint8_t)(mid - 1);
        }
    }

    float lower = fp8_e5m2_positive_code_to_fp32(lo);
    if (af == lower || lo == 0x7B) {
        return (uint8_t)((sign << 7) | lo);
    }

    uint8_t upper_code = (uint8_t)(lo + 1);
    float upper = fp8_e5m2_positive_code_to_fp32(upper_code);
    float p_up = (af - lower) / (upper - lower);
    uint8_t mag = hip_quant_uniform01(seed, idx) < p_up ? upper_code : lo;
    return (uint8_t)((sign << 7) | mag);
}

// bfloat16 -> FP8 E5M2 without first materializing a float value.
__device__ inline uint8_t bf16_to_fp8_e5m2(uint16_t h) {
    return fp32_bits_to_fp8_e5m2((uint32_t)h << 16);
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
