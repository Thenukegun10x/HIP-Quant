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

// =========================================================================
// HQ2 — HipQuant-2 (TurboQuant-inspired learned-codebook weight quant)
//
// Block of 256 weights. A per-block, importance-weighted non-uniform 4-level
// codebook is fit on-device (weighted Lloyd's / k-means), so the stored levels
// adapt to the local weight distribution instead of being a fixed uniform grid.
// This is what makes 2-bit weights usable: the codebook carries the per-block
// structure that a plain 2-bit uniform grid throws away.
//
//   levels[4] : 4 absolute codebook values, FP16 (8 bytes)
//   qs[64]    : 256 indices, 2 bits each               (64 bytes)
//   total     : 72 bytes -> 2.25 bpw
// =========================================================================
#define HQ2_K 256
#define AQ2_K HQ2_K

typedef struct {
    ggml_half levels[4];
    uint8_t qs[HQ2_K / 4];
} block_hq2;

// AQ2 uses the same physical 72-byte layout as HQ2.  Keeping the type
// distinct lets calibration, model metadata, and dispatch distinguish an
// attention-calibrated tensor without adding inference-time bits.
typedef struct {
    ggml_half levels[4];
    uint8_t qs[HQ2_K / 4];
} block_aq2;

// AQ2-QK and AQ2-VO intentionally share the AQ2 wire layout.  Their type
// IDs keep role-specific calibration and model metadata explicit without
// creating a second inference decoder or adding bits to the tensor.
typedef block_aq2 block_aq2_qk;
typedef block_aq2 block_aq2_vo;

#if defined(__cplusplus)
static_assert(sizeof(block_hq2) == 72, "block_hq2 wire layout must remain 72 bytes");
static_assert(sizeof(block_aq2) == 72, "block_aq2 wire layout must remain 72 bytes");
static_assert(sizeof(block_aq2_qk) == 72, "block_aq2_qk wire layout must remain 72 bytes");
static_assert(sizeof(block_aq2_vo) == 72, "block_aq2_vo wire layout must remain 72 bytes");
#endif

typedef struct {
    ggml_half d;
    uint8_t qs[3 * QK_K / 8];
} block_iq3_xxs;

typedef struct {
    ggml_half d;
    uint16_t qs[QK_K / 8];
} block_iq2_xxs;

typedef struct {
    ggml_half d;
    uint16_t qs[QK_K / 8];
    uint8_t scales[QK_K / 32];
} block_iq2_xs;

#define IQ3S_N_SCALE (QK_K / 64)

typedef struct {
    ggml_half d;
    uint8_t qs[QK_K / 4];
    uint8_t qh[QK_K / 32];
    uint8_t signs[QK_K / 8];
    uint8_t scales[IQ3S_N_SCALE];
} block_iq3_s;

typedef struct {
    ggml_half d;
    uint8_t qs[QK_K / 8];
    uint16_t qh[QK_K / 32];
} block_iq1_s;

typedef struct {
    uint8_t qs[(QK_K - 4 * QK_K / 64) / 5]; // 5 elements per byte
    uint8_t qh[QK_K / 64];                  // 4 elements per byte
    ggml_half d;
} block_tq1_0;

typedef struct {
    uint8_t qs[QK_K / 4]; // 2 bits per element
    ggml_half d;
} block_tq2_0;

#define QK_F8 32

typedef struct {
    uint8_t qs[QK_F8]; // 32 raw FP8 E4M3 values, no scale
} block_f8_e4m3;

typedef struct {
    uint8_t qs[QK_F8]; // 32 raw FP8 E5M2 values, no scale
} block_f8_e5m2;
