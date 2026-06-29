#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>

// kvalues_iq4nl codebook — shared by IQ4_NL, IQ4_XS, and other IQ kernels
__device__ const int8_t d_kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

// Binary search for nearest codebook entry
__device__ inline int best_index_int8_dev(int n, const int8_t *val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu - ml > 1) {
        int mav = (ml + mu) / 2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}
