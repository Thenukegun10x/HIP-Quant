# TODO (Sep 2026)

The old PyTorch-extension plan (`PYTORCH_EXTENSION_PLAN.md`) is implemented:
`torch_ext/` + `torch_api.py` + autograd FP8 linear (fwd/bwd) all exist. What
remains, in priority order:

## 1. imatrix fix sequence (`IMATRIX_PLAN.md`, P0 first)
- P0: shape-check `quantize_from_fp8` imatrix (host/device OOB today); warn on
  `None` for IQ2_XXS/XS/IQ1_S. No rebuild.
- P1: accept llama per-column `(n_per_row,)` vectors (tile internally) + `.dat`
  loader. No rebuild.
- P2: per-column device upload + kernel row-stride flag (DLL rebuild).
- P3: non-trivial-imatrix byte-exact tests vs `llama-quantize.exe`.

## 2. Decode throughput (Qwen3.8-27B at 5.4 tok/s, target 26)
Profile with `python -m hip_inference.debug_decode --fine` before touching code.
- IQ GEMV kernels at ~70 GB/s effective vs ~640 GB/s HBM roof (FFN is 64% of a
  token). Target 250 GB/s effective.
- LM head 9.3ms (Q4_K 248320x5120): dedicated tall-GEMV path, target <=3ms.
  Also dominates the MTP draft cost (19.7ms).
- HIP graph capture for the qwen35 decode step (~1000 launches/token;
  qwen3 already has `_generate_graph`).
- MTP accept-rate measurement (`draft_mtp` returned token 0 once — verify
  quality before relying on the ~1.9x multiplier).

## 3. Kernel hardening backlog
- Same `TORCH_CHECK` treatment (dtype + numel) for remaining `torch_ext`
  entries that reinterpret bits (`delta_net`, `fast_ssm_conv1d`, `gemv_q`).
- C-side staged-size validation for quantize dispatch (defense in depth, IM-2).

## 4. Docs discipline
- Keep `IMATRIX_PLAN.md` appendix (per-type audit table) current as P3 lands.
- `DOCUMENTATION.md` line counts are dated Sep 2026; refresh on big merges.
- `Own Quant/` experiment logs are historical — do not rewrite.
