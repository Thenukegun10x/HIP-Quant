# IU4 KV Cache: Architecture & Plan

Native 4-bit KV cache on gfx1201's own matrix ISA. Headline: **2.7x smaller KV**
(K-IU4 + V-FP8) with a forward path to 4x, at near-lossless accuracy.

> Scope honesty first: at 2048 ctx, KV traffic is ~0.14GB of ~9.5GB per
> decode step (~1.5%) — this feature will NOT move short-context tok/s. Its
> wins are **capacity and long context**: 32k ctx goes 2.2GB → ~0.8GB (fits
> comfortably next to 12GB of weights on 16GB), and KV traffic stops mattering
> exactly when it would start to (19% of traffic at 32k). Short-context speed
> stays on the GEMV/MTP/graph program.

## 1. Verified foundation (Sep 2026)

- **Hardware**: gfx1201 has native `wmma_i32_16x16x16_iu4` and
  `wmma_i32_16x16x32_iu4` (K=32, 2x the K=16 rate), i32 accumulators,
  per-operand signedness flags (`neg_a/neg_b`), 4096 ops/clk/CU (4x FP16).
  Lane mapping is the same column-distributed convention (`lane % 16` =
  column) — verified on gfx1201 hardware by third parties; matches the
  repo's wave_attn experience (no transpose trap).
- **Toolchain**: ROCm 7.2 / clang 21 accepts the intrinsic and compiles it
  clean (probe kernel built and removed; keep probes out of the tree).
- **FP8 path exists**: `quantize_e4m3` + fp8 WMMA are already proven in
  `wave_attn` — V-FP8 reuses them, it does not reinvent them.
- Deliberately out of scope: SWMMAC 2:4 sparsity (needs sparse codecs first;
  another 2x left for later).

## 2. Design decisions (locked)

| Decision | Choice | Why |
|---|---|---|
| Split | **K-IU4 + V-IU4-G4** (full INT4, 4x), promoted from experiment by P0 | QK→softmax tolerates K noise (ordering); P0 measured V-INT4-G4 at +0.05% PPL — no reason to stop at FP8 |
| K granularity | **per-channel symmetric INT4**, fp16 scales | K outliers are channel-wise and mild here (7x, QK-norm works); per-token dead on arrival; symmetric beats asymmetric cross-prompt (P0) |
| V granularity | **per-token asymmetric UINT4, 4 groups of 64ch**, fp16 scales + u8 zp | V spikes are token-dependent (no stable channels); G4 lifts worst layers 0.88→0.95 cosine at ~10% scale overhead |
| Accuracy stack | calibrated scales → per-layer opt-out (rotation/KIVI deferred) | P0 passes rung 1 cleanly; rungs 2-3 stay contingency, not plan |
| V-FP8 | fallback rung, not main path | validated +0.02% PPL; kept as safe fallback alongside INT8 |
| Math | dequant-via-LDS-LUT, then native WMMA | 1 LDS read vs 6-8 ALU ops (proven pattern on this GPU); iu4 for QK, iu4 for PV |
| Default | fp16 KV stays default; opt-in flag | no behavior change until proven; per-layer opt-out on accuracy failure |

Memory per token/layer (fp16 baseline 2KB): K-IU4 0.25KB + V-IU4 0.25KB +
scales ≈ **~0.55KB (3.6x)**.

## 3. P0 results (measured Sep 2026 — all P0 gates PASS)

Torch-ref fake-quant on real KV dumps (Qwen3.8-27B, 16 attn layers, GQA
n_kv=4/head_dim=256; 2 dump prompts + 5-prompt/1536-token PPL corpus).
Scripts: `iu4_p0/` scratch (`dump_kv.py`, `sim_offline.py`, `sim_e2e.py`,
`sim_ppl.py`, `sim_determinism.py`).

- **K distribution**: per-channel max|.| median 2.8, p99 9.7, max 19.8
  (outlier ratio **7x, not 100x** — QK-norm earns its keep). K-only
  per-channel-symmetric-INT4: cosine 0.95–0.999 every layer. **No rotation
  needed; rung 1 suffices.**
- **V distribution**: per-token max|.| median 4.7, max 35 — spikes are
  **token-dependent** (top-8 spiky channels overlap only 3–4/8 across
  prompts), so per-channel calibration can't save V; online groups can.
- **Offline** (single decode step, cross-prompt scales): Ksym/Vasym-G1 min
  cosine 0.864 → G4 0.949 on worst layers (11/15/23/55/59). Asymmetric-K
  calibrates 0.56 cross-prompt (min/max range fragility) — **symmetric-K
  locked for robustness**, not just simplicity.
- **PPL** (teacher-forced, deterministic baseline verified 3/3 identical):
  base 3.8449 → V-FP8 3.8458 (+0.02%) → V-INT4-G1 3.8561 (+0.29%) →
  **V-INT4-G4 3.8469 (+0.05%)**. Bar was Δppl < 0.5%; G4 clears it 10x.
- **Exact-match e2e is chaos** (single flip cascades; G1 1/3 vs G4 0/3 while
  PPL ranks G4 strictly better) — PPL is the gate, exact-match is anecdote.
- Caveats: in-distribution corpus (own greedy outputs); K scales were online
  in-sim, deployment freezes calibrated ones (strictly more stable);
  WikiText-scale calibration still open in P1.

## 4. Storage spec

Per attention layer slot (replaces the fp16 `[n_kv, max_seq, head_dim]` pair):

- **K-IU4**: packed nibbles `[n_kv, max_seq, head_dim/2]` uint8 (low nibble =
  even channel, high nibble = odd channel; channel-contiguous so a 16-channel
  WMMA fragment is a linear read) + scales `[n_kv, head_dim]` fp16 (one per
  channel, amortized: 1KB/layer regardless of seq).
- **V-FP8**: bytes `[n_kv, max_seq, head_dim]` uint8 (E4M3) + scales
  `[n_kv, max_seq]` fp16 (one per token: 8B/token/layer, negligible).
- Quantize **after RoPE** (append point): scales adapt to the rotated
  distribution; rotation/outlier work must be RoPE-aware (see §5).
- Signs: K symmetric INT4 (-8..7) via unsigned-nibble + `neg` flags, or
  asymmetric UINT4 + zero-point per channel if the harness prefers it —
  the intrinsic supports both; decide by measurement, not debate.
- Payload is EXACTLY 4.0 bpw (vs 4.5–5.0 for Q4_0/Q4_1/IQ4_NL/Q4_K): element
  `i` lives at bit-offset `i*4`, power-of-2 packing aligns to any vector
  width (compare Q4_0's 18-byte blocks straddling vector loads), and scales
  live decoupled in sidecars (K per-channel fp16 amortizes to ~0.01% at 32k;
  V per-token fp16 ≈ 3%, fp8 scales optional). Consequence: the cache is a
  plain array — memcpy/slice/prefix-share without a codec, and MTP-style
  snapshot/restore stays trivial byte copies.

## 5. Interactions (do not regress)

- **MTP draft is cacheless** (own token only) — KV format changes cannot
  affect it. Prefill/verify batches DO read KV: they are the accuracy
  harness's main path.
- **Non-square mask**: every `is_causal` backend here aligns top-left;
  `attention_forward` already routes non-square through explicit bottom-right
  masks. Any new attention kernel (iu4/fp8 WMMA) must pass the same
  asymmetric-matrix verification (identity tests cannot catch lane bugs).
- **SSM state untouched** (fp32, precision-critical). **Gated-norm**,
  **K-split**, **MTP snapshot** (seq-based KV rollback) all keep working:
  rollback only moves `seq`; stale tails are overwritten, never read.
- **Decode GEMV reality**: QK^T stays memory-bound; the win is traffic shrink
  (and capacity), not instruction rate. Prefill/verify batches ride the 4x/2x
  WMMA rates.

## 6. Kernel inventory (new file: `torch_ext/kv_iu4_kernels.hip`)

1. `kv_quant_k_i4_append` — fp16 K row(s) + running/static per-channel scales
   → packed nibbles at `[seq:seq+S]`. Vectorized loads, one wave per head.
2. `kv_quant_v_fp8_append` — fp16 V row(s) + per-token amax → E4M3 bytes +
   scale. (Thin wrapper over the existing `quantize_e4m3` pattern where possible.)
3. `kv_dequant_*` for the P1 read path (LDS-LUT nibble decode; fp8 convert).
4. P3: `attn_qk_iu4_wmma` (Q dynamically quantized per 16/32-tile, iu4 WMMA,
   fp32 rescale+softmax) and `attn_pv_fp8_wmma` (P dynamically quantized,
   fp8 WMMA). Asymmetric-matrix tests mandatory (see §7).
5. `torch_ext/pytorch_bindings.cpp`: `TORCH_CHECK` dtype/numel contracts on
   every entry (post-`ssm_norm` policy — short buffers must raise, never OOB).

## 7. Integration

- `SparseKVCache`: per-side buffers allocated from the config (default
  `"fp16"` both sides = today's path, bit-identical). Slot/drop logic
  unchanged; 64-slot policy unchanged.
- Runners take **independent knobs** (defaults = today, no behavior change):
  `Qwen35Runner(..., k_dtype="fp16", v_dtype="fp16")`, plumbed through
  `load_model`. Each side: `"fp16" | "int8" | "fp8" | "int4"` (K-int4 =
  per-channel-sym; V-int4 = per-token-G4-asym; unsupported side/dtype raises
  at init, never silently). Shorthand `kv_dtype="int4"` sets both sides;
  explicit `k_`/`v_` override it. Per-layer escape hatch
  `kv_fallback_layers=[11, ...]` keeps listed layers fp16 (or fp8) —
  ship hybrid before blocking on stragglers.
- Init print reports KV MB per side like weights (e.g. `KV cache K-int4
  12MB + V-int4 14MB`).
- Scale calibration: offline pass over a few hundred prompt tokens producing
  per-(layer, head, channel) K scales, stored alongside (GGUF-adjacent sidecar
  file, versioned). No runtime stats in v1.

## 8. Accuracy plan (gates every phase)

- Harness: cosine vs fp16-KV per layer + `debug_decode` health + e2e
  coherence on fixed prompts + (for K) a perplexity spot-check. Bars: cosine
  ≥ 0.999 vs fp16-KV, zero NaN/inf, coherence indistinguishable.
- Rung 1: calibrated per-channel K scales. Rung 2 (only if rung 1 fails):
  RoPE-aware offline rotation (QuaRot-style; fused into Q/K projs, FWHT where
  RoPE intervenes). Rung 3 (last resort): KIVI residual window (recent-R fp16
  + quantized tail).
- "Zero-compute" is a deployment property, not a method property: scales and
  rotations are only free if FUSED into adjacent weights once at load time
  (one-time GEMM, invisible per token). An unfused integration pays an
  epsilon (extra vector-scale / ~1% online FWHT where RoPE blocks fusion).
  LUT dequant is reduced-cost (1 LDS read vs 6-8 ALU), not zero-cost.
- Kernel verification: asymmetric matrices (117/120 asymmetric elements
  minimum), never identity-only; small→large sizes up to 17408-wide.
- Fallback: per-layer opt-out list (fp16 KV for failing layers); ship hybrid
  before blocking on stragglers.

## 9. Phase plan

- **P0 — harness + spec proof (no kernels)**: simulate the exact quant
  (torch reference: per-channel i4 K, per-token fp8 V) on real KV dumps;
  confirm bars are reachable. Kill criterion lives here, cheaply.
- **P1 — V-INT4 live**: append/quant + LDS-LUT dequant read path + cache
  config (`k_dtype`/`v_dtype`/`kv_dtype`/`kv_fallback_layers`) + integration;
  K stays fp16. Ships ~1.8x KV + all plumbing K reuses. e2e-gated (PPL gate:
  rerun P0 `sim_ppl` against live kernels, Δppl < 0.5%).
  - DONE Sep 2026: `kv_iu4_kernels.hip` + bindings + build green; kernels
    verify vs independent np.float32 IEEE ref (scales/zp bit-exact, all
    nibble diffs <= 1, rounding contract in file header); live PPL 3.8449
    -> 3.8511 (**+0.16%**, bar 0.5%); CPU suite 92 passed; fp16 path
    bit-identical. V side 34MB -> 9MB + P1-temp fp16 scratch (P3 fuses it).
- **P2 — K-IU4 storage**: append/quant + LDS-LUT dequant read path;
  calibration sidecar; full ~3.6x live on fp-grade math. e2e-gated.
- **P3 — native WMMA math**: iu4 QK + fp8 PV paths for batched shapes;
  asymmetric verification; headline benchmarks (long-context capacity +
  traffic). Decode GEMV path stays dequant-based unless measured otherwise.
- **P4 (optional experiment)**: V-INT4 payload swap behind the same scales;
  ship only with a measured quality delta on the table.

## 10. Verification matrix

| Check | How | Pass bar |
|---|---|---|
| P0 reachable | torch-ref quant on dumped KV, cosine + e2e | §7 bars |
| No OOB | short/odd shapes, K≠multiple-of-32 probe | clean error or correct (never silent) |
| Lane mapping | asymmetric matrices, small→17408 | bit-exact vs torch ref layout |
| e2e parity | fixed prompts, fp16-KV vs iu4-KV text | coherent; cosine-gated |
| Perf | 32k-ctx capacity + KV-traffic/ctx slope | ~2.7x capacity; traffic slope matches |
| No regressions | pipeline CPU suite + gemv A/B + debug health | green |

## 11. Risks & open questions

1. K outlier severity on *this* model family decides rung 1 vs 2 — P0 answers in days.
2. RoPE × rotation interaction (known-unknown; QuaRot literature covers it, verify locally).
3. E4M3 ±448 range on V spikes — per-token scales should cover; harness confirms.
4. AOTriton/SDPA interplay for new dtypes — new kernels bypass SDPA, but the
   explicit-mask policy (§4) applies to any fallback path touching iu4 data.
5. `max_seq` discipline matters MORE now: mis-sized max_seq wastes 4x less,
   but 32k ambitions need deliberate sizing (init print already reports MB).
