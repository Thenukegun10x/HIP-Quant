# imatrix Fix Plan

Reference: vendored `Own Quant/llama_cpp_stock` (ggml-quants.c, ggml.c, src/llama-quant.cpp).
Do NOT modify that tree (upstream contribution rules); read it as ground truth only.

## 1. Reference semantics (llama.cpp)

- The imatrix is **one float per input column, shared across all rows**.
  `quantize_iq2_xs` passes the *same* `quant_weights` pointer for every row
  (`ggml-quants.c:3669`); size check is `ne[0]*ne[2]`, i.e. one `n_per_row`
  slice per expert (`src/llama-quant.cpp:1188,1247`).
- Weighting math (IQ2_XS, `ggml-quants.c:3519`):
  `weight[i] = qw[i] * sqrt(sigma2 + x[i]^2)`.
- `NULL` is **refused** for `IQ2_XXS` / `IQ2_XS` / `IQ1_S`
  (`ggml.c:7905-7924`, `GGML_ASSERT(imatrix != NULL)`).

## 2. Current hip_quant behavior

| # | Location | Finding |
|---|---|---|
| IM-1 | `__init__.py:673-676` | `quantize_numpy` **rejects** llama-format `(n_per_row,)` vectors (`ValueError`); callers must hand-tile to full `rows x cols`. A 5120x5120 weight then carries a 100MB fp32 upload per call. |
| IM-2 | `__init__.py:852-854` | `quantize_from_fp8` has **no imatrix shape check**. A short buffer flows into `quantize_tensor_fp8_input_impl`, which uploads `src_bytes` (`hip_quantize.cpp:1258-1264`, host OOB read in `hipMemcpy`) while kernels index `row*n_per_row` (device OOB read). Same bug class as the `ssm_norm` OOB (fixed Sep 2026, see root `AGENTS.md`). |
| IM-3 | `tests/test_iq2s_iq1m_byte_exact.py`, `e2e_iq2s_iq1m.py` | Every imatrix in the suite is **all-ones**, for which any weighting scheme is the identity. The weighting path is therefore **unverified against llama.cpp with real weights** (math matches by inspection for IQ2_XS only). |
| IM-4 | all `kernels/quant_iq*.cu` | `NULL` imatrix silently falls back to unweighted (`qw ? ... : x^2`) even for the three types where llama.cpp aborts. Silent quality degradation. |
| IM-5 | `hip_quantize.cpp:1008-1022` | Full-matrix H2D copy on **every** call (no caching); folds into the IM-1 fix. |

Kernels themselves are NULL-safe (`if (imatrix)` / `if (imatrix != NULL)` in
`quant_iq{1_s,2_xxs,2_xs,2_s,3_xxs,4_nl,4_xs,1_m}.cu`) and index
`imatrix + row*n_per_row + ...`, i.e. correct **iff** the buffer really is a
full matrix. No kernel change is needed for correctness; only for speed (P2).

## 3. Fix design

### P0 - safety, no rebuild (do first)
1. Mirror the `quantize_numpy` shape check into `quantize_from_fp8` (and audit
   `quantize_numpy_to` for the same gap): reject anything that is not exactly
   `arr.shape`. Closes the host/device OOB read (IM-2).
2. `UserWarning` (loud, once per process) when `imatrix=None` for
   `IQ2_XXS` / `IQ2_XS` / `IQ1_S` (IM-4). Do not hard-fail yet: existing
   callers/tests rely on the fallback; flip to error after P1 tests land.

### P1 - llama compat, Python only (no rebuild)
3. Accept `(n_per_row,)` fp32 vectors in `quantize_numpy` /
   `quantize_numpy_to` / `quantize_from_fp8`: validate `shape == (n_per_row,)`,
   tile with `np.broadcast_to(im, arr.shape)` (zero-copy view) before the
   existing contiguous+upload path. Tiled values are row-identical, so kernel
   output becomes bit-identical to llama by construction (IM-1).
4. Add `hip_quant.imatrix.load_dat(path) -> {tensor_name: np.float32[ncols]}`
   parsing the current `.dat` format (`common_imatrix_load` field order:
   per-tensor name, counts, sums/ncall) plus a `for_tensor(name, nrows, ncols)`
   helper returning the tiled full matrix. Keeps gguf/llama naming in one place
   (reuse the remap rules from `src/llama-quant.cpp:75-90` for renamed tensors).
5. Docstrings + root README: state the contract (per-column shared, tiled
   internally; full-matrix input still accepted).

### P2 - speed, needs DLL rebuild (`build.ps1`)
6. Upload once: accept a per-column device buffer. Either
   (a) tile on-device with a tiny broadcast kernel into the existing
   `g_d_imatrix` staging (no kernel-signature changes), or
   (b) add a row-stride flag to the IQ kernels (`w_stride`, mirroring the
   `fast_rms_norm_gated` shared-weight fix) and skip staging for the
   column vector. Prefer (b) for large weights: upload drops from
   `nrows*n_per_row` to `n_per_row` floats (100MB -> 20KB at 5120^2).
7. C-side size validation: pass the staged byte count into
   `dispatch_quantize_kernel` and assert per-kernel read footprints against it
   (defense in depth for IM-2 class bugs).

### P3 - tests (prove it, GPU)
8. Non-trivial byte-exact test: extend `tests/test_iq2s_iq1m_byte_exact.py`
   with a tapered/random (seeded) imatrix driven through **both**
   `llama-quantize.exe --imatrix` (real `.dat`, not the legacy all-ones `.n`)
   and `quantize_numpy` (per-column vector after P1). Start with
   `IQ2_S` / `IQ1_M`, then the P0/P1-audited IQ types.
9. Per-type audit table (math vs `ggml-quants.c` impl, NULL behavior,
   byte-exact status with non-trivial imatrix) committed in this file's
   appendix; every IQ/K kernel that dereferences the pointer gets a row.
10. Flip the P0 warning to an error for the three llama-mandatory types once
    green, and require non-ones imatrix in at least one byte-exact test so the
    weighting path can never silently regress again.

## 4. Verification matrix

| Check | How | Pass bar |
|---|---|---|
| No shape-check bypass | pass `(ncols,)`, `(1,ncols)`, wrong-size to all three entry points | `ValueError` before any copy/launch |
| llama compat | seeded non-trivial imatrix, `IQ2_S`+`IQ1_M`, `llama-quantize.exe` reference | byte-identical |
| Upload size | 5120x5120 + per-column vector (P2) | H2D imatrix bytes == `n_per_row*4` |
| Perf neutral | `benchmark_quantize_kernel` before/after | within noise |
| Full suite | `tests/test_iq2s_iq1m_byte_exact.py` + pipeline CPU suite | green |

## 5. Rollout order

P0 (safe, immediate) -> P1 (compat, immediate) -> P3-tests (needs P1) ->
P2 (rebuild, needs a DLL rev + e2e rerun) -> warning-to-error flip.

## Appendix A - per-type audit (fill in during P3)

| Type | ggml ID | `qw` math matches llama | NULL-safe | non-trivial byte-exact |
|---|---|---|---|---|
| IQ2_XS | 17 | yes (`ggml-quants.c:3519` vs `quant_iq2_xs.cu:76`) | yes | TODO |
| IQ2_XXS | 16 | TODO | yes | TODO |
| IQ2_S | 22 | TODO | yes | TODO (`test_iq2s...` ones-only today) |
| IQ3_XXS | 18 | TODO | yes | TODO |
| IQ3_S | 21 | TODO | yes | TODO |
| IQ1_S | 20 | TODO | yes | TODO |
| IQ1_M | 29 | TODO | yes | TODO (ones-only today) |
| IQ4_NL | 24 | TODO | yes | TODO |
| IQ4_XS | 23 | TODO | yes | TODO |
| K-quants / legacy | various | take pointer; audit deref | TODO | n/a (no imatrix weighting) |
