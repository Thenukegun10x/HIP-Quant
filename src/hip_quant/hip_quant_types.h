#pragma once
#include <stdint.h>

#define QK_K 256
#define K_SCALE_SIZE 12

typedef uint16_t ggml_half;

typedef struct {
    ggml_half d;
    ggml_half dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K / 2];
} block_q4_K;

typedef struct {
    ggml_half d;
    ggml_half dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qh[QK_K / 8];
    uint8_t qs[QK_K / 2];
} block_q5_K;

typedef struct {
    uint8_t ql[QK_K / 2];
    uint8_t qh[QK_K / 4];
    int8_t scales[QK_K / 16];
    ggml_half d;
} block_q6_K;

typedef struct {
    uint8_t scales[QK_K / 16];
    uint8_t qs[QK_K / 4];
    ggml_half d;
    ggml_half dmin;
} block_q2_K;

typedef struct {
    uint8_t hmask[QK_K / 8];
    uint8_t qs[QK_K / 4];
    uint8_t scales[12];
    ggml_half d;
} block_q3_K;

typedef struct {
    ggml_half d;
    uint8_t qs[16];
} block_q4_0;

typedef struct {
    ggml_half d;
    ggml_half m;
    uint8_t qs[16];
} block_q4_1;

typedef struct {
    ggml_half d;
    uint8_t qh[4];
    uint8_t qs[16];
} block_q5_0;

typedef struct {
    ggml_half d;
    ggml_half m;
    uint8_t qh[4];
    uint8_t qs[16];
} block_q5_1;

typedef struct {
    ggml_half d;
    int8_t qs[32];
} block_q8_0;

typedef struct {
    ggml_half d;
    ggml_half s;
    int8_t qs[32];
} block_q8_1;

#define QK4_NL 32

typedef struct {
    ggml_half d;
    uint8_t qs[QK4_NL / 2];
} block_iq4_nl;

typedef struct {
    ggml_half d;
    uint16_t scales_h;
    uint8_t scales_l[QK_K / 64];
    uint8_t qs[QK_K / 2];
} block_iq4_xs;

typedef struct {
    ggml_half d;
    uint8_t qs[3 * QK_K / 8];
} block_iq3_xxs;
