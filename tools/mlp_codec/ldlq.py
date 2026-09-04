"""LDLQ: Hessian error feedback on top of the HQ2V joint codebook.

The thesis this tests
--------------------
Every measurement in this study says HQ2V's remaining deficit is *error shaping*,
not reconstruction accuracy:

  * `iq2_xxs` reconstructs weights *less* accurately than HQ2V (9.16 dB vs
    9.87 dB weighted SQNR) and still produces a better model.
  * The weighted scale search moved SQNR by 0.03 dB while cutting KL median 18%.
  * The sign-symmetric variant improved a defect that MSE liked and cost
    +0.096 ln PPL.

If that reading is right, the lever that should work is the one that optimises
*output* error directly rather than weight-space error. That is LDLQ.

What LDLQ does
--------------
The quantity that matters is not ||W - W_hat||^2 but the error the layer's
*output* inherits, ||(W - W_hat) X||^2_F = tr(E H E^T) with H = E[x x^T].  Plain
quantization minimises the former and merely hopes for the latter. LDLQ
minimises the latter by construction: coordinates are quantized in order, and
after each one the residual error is pushed onto the *not-yet-quantized*
coordinates along the directions H says are correlated. A weight that is about
to be quantized anyway can absorb its neighbour's mistake for free.

This costs **zero bits**. The format, the codebook, the scale, the rate are all
unchanged -- only the values handed to the quantizer differ. That is what makes
it the cleanest possible test of the shaping thesis: if shaping is really the
binding constraint, this should move KL substantially at 2.0632 bpw.

Implementation notes
--------------------
Follows GPTQ's formulation. With R the upper-triangular Cholesky factor of
H^-1 (so H^-1 = R^T R), quantizing coordinate j and compensating is

    err_j      = (w_j - q_j) / R[j, j]
    W[:, j+1:] -= err_j (x) R[j, j+1:]

and GPTQ's "lazy batch" structure applies it finely inside a column block, then
once in bulk to everything after the block. Here the block is 256 columns --
exactly HQ2V's own block -- so the two granularities coincide for free.

Two deliberate deviations, both stated rather than hidden:

* **Feedback is quad-granular, not coordinate-granular.** HQ2V's atomic unit is
  a jointly-quantized 4-D quad, so error from a quad propagates only to
  coordinates *after* the quad, never within it. Forgoing intra-quad feedback is
  inherent to joint quantization: the codebook search already optimises those 4
  coordinates together, which is a different and generally better trade than
  sequential feedback among them. This is the same choice GPTVQ makes.
* **Precision is split.** The factorisation runs in float64 because at the
  measured kappa ~= 4.7e3 float32 retains only ~3.5 decimal digits, and LDLQ
  consumes the factors over d = 9216 sequential steps where a corrupted factor
  amplifies error past plain rounding. The *sweep* then runs in float32: kappa
  amplification applies to solving, not to storing an already-solved factor, so
  rounding R once costs ~6e-8 relative against a ~10% quantization error.
  `--sweep-dtype float64` exists so that claim can be checked rather than
  trusted.

`hq2v.py` is imported, never modified: it is a measured artifact behind published
ladder rows, so its assignment and scale-refinement primitives are reused
verbatim to guarantee the only difference from the baseline row is the feedback.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import hq2v
from hq2v import BLOCK, DIM, CODEBOOK, FIT_SAMPLE, ROW_CHUNK_WEIGHTS

HESSIAN_DIR = Path(r"G:\hq2_research\hessians")

# Fraction of mean(diag H) added to the diagonal before factorising. The measured
# spectra need essentially none of this (60/64 tensors are PSD at zero, and the
# 4 exceptions are rank-deficient by exactly one dead input channel rather than
# ill-conditioned), so this is a floor for safety, not a tuned parameter.
DEFAULT_DAMPING = 1e-2


def hessian_path_for(tensor_name: str, hessian_dir: Path = HESSIAN_DIR) -> Path:
    """Map a GGUF tensor name onto its collected Hessian.

    gate and up share an input -- verified bit-identical on all 32 blocks -- so
    they share one Hessian file.
    """
    import re

    m = re.match(r"blk\.(\d+)\.ffn_(gate|up|down)\.weight$", tensor_name)
    if not m:
        raise ValueError(f"not an MLP projection this study quantizes: {tensor_name!r}")
    block = int(m.group(1))
    kind = "ffn_down" if m.group(2) == "down" else "ffn_gate_up"
    return hessian_dir / f"blk{block:02d}.{kind}.npy"


def load_hessian(tensor_name: str, expect_dim: int,
                 hessian_dir: Path = HESSIAN_DIR) -> np.ndarray:
    path = hessian_path_for(tensor_name, hessian_dir)
    if not path.exists():
        raise FileNotFoundError(f"no Hessian for {tensor_name}: {path}")
    h = np.load(path)
    if h.shape != (expect_dim, expect_dim):
        raise ValueError(f"{path.name}: shape {h.shape} != ({expect_dim}, {expect_dim})")
    return h


# ---------------------------------------------------------------------------
# Factorisation
# ---------------------------------------------------------------------------

_FACTOR_CACHE: dict[tuple[str, float], np.ndarray] = {}
_FACTOR_CACHE_MAX = 2      # gate and up share a Hessian and are adjacent in offset order

_GPU_CHOLESKY: bool | None = None


def gpu_cholesky_available() -> bool:
    """Probe once whether the GPU can factorise at all.

    This host's PyTorch is 2.9.1+rocm7.2.1, which is built without MAGMA, so
    `torch.linalg.cholesky` raises on a device tensor. Probing a 4x4 once is far
    cheaper than allocating a 680 MiB float64 Hessian per tensor only to catch
    the same exception 96 times.
    """
    global _GPU_CHOLESKY
    if _GPU_CHOLESKY is not None:
        return _GPU_CHOLESKY
    try:
        import torch
        if not torch.cuda.is_available():
            _GPU_CHOLESKY = False
        else:
            probe = torch.eye(4, dtype=torch.float64, device="cuda")
            torch.linalg.cholesky(probe)
            torch.cholesky_inverse(probe)
            del probe
            torch.cuda.empty_cache()
            _GPU_CHOLESKY = True
    except Exception:
        _GPU_CHOLESKY = False
    return _GPU_CHOLESKY


def _factor_numpy(h: np.ndarray, shift: float) -> np.ndarray:
    d = h.shape[0]
    work = np.array(h, dtype=np.float64, order="F")
    idx = np.arange(d)
    work[idx, idx] += shift
    lower = np.linalg.cholesky(work)                  # H_damped = L L^T
    del work
    gc.collect()
    l_inv = np.linalg.inv(lower)                      # once, not once per operand
    del lower
    gc.collect()
    h_inv = l_inv.T @ l_inv                           # H^-1 = L^-T L^-1
    del l_inv
    gc.collect()
    h_inv = 0.5 * (h_inv + h_inv.T)                   # exact symmetry before factoring
    factor = np.linalg.cholesky(h_inv).T              # upper triangular
    del h_inv
    gc.collect()
    return factor


def _factor_torch(h: np.ndarray, shift: float, device: str) -> np.ndarray:
    import torch

    work = torch.from_numpy(np.ascontiguousarray(h, dtype=np.float64)).to(device)
    work.diagonal().add_(shift)
    lower = torch.linalg.cholesky(work)
    del work
    h_inv = torch.cholesky_inverse(lower)
    del lower
    factor = torch.linalg.cholesky(h_inv, upper=True)
    del h_inv
    out = factor.cpu().numpy()
    del factor
    if device != "cpu":
        torch.cuda.empty_cache()
    return out


def prepare_factor(h: np.ndarray, damping: float = DEFAULT_DAMPING,
                   backend: str = "auto", device: str | None = None) -> np.ndarray:
    """Upper-triangular R with H^-1 = R^T R, computed in float64.

    GPTQ's exact sequence -- cholesky, inverse, cholesky-upper -- in float64 for
    the reason argued in the module docstring. Verified against its definition:
    `max|R^T R H_damped - I|` is 3.7e-14 at dim 9216.

    Backend defaults to numpy, which is *measured* rather than assumed: on this
    host numpy/scipy-openblas takes 12.05 s for a 9216x9216 Hessian against
    37.37 s for torch CPU (agreeing to 8.9e-14), and the GPU cannot factorise at
    all because this ROCm PyTorch build lacks MAGMA. Across the 32 `ffn_down`
    Hessians in a ladder row that is ~6 min instead of ~20. `backend="torch"`
    stays available for a host where the GPU path works.

    Dead input channels -- a channel identically zero across the whole
    calibration set, which 4 of the 64 tensors have -- make H exactly singular.
    Damping fixes that, and it is the correct fix rather than a workaround: a
    channel the model never activates places no constraint on the weights that
    read it.
    """
    mean_diag = float(np.diag(h).mean())
    if not np.isfinite(mean_diag) or mean_diag <= 0:
        raise ValueError("Hessian has a non-positive mean diagonal")
    shift = damping * mean_diag

    if backend == "auto":
        backend = "numpy"
    if backend == "numpy":
        return _factor_numpy(h, shift)
    if backend == "torch":
        if device is None:
            device = "cuda" if gpu_cholesky_available() else "cpu"
        return _factor_torch(h, shift, device)
    raise ValueError(f"unknown factor backend {backend!r}")


def cached_factor(hessian: np.ndarray, key: str, damping: float,
                  backend: str = "auto") -> np.ndarray:
    """`prepare_factor` with a tiny cache, so a shared Hessian is factored once."""
    cache_key = (key, damping)
    hit = _FACTOR_CACHE.get(cache_key)
    if hit is not None:
        return hit
    factor = prepare_factor(hessian, damping, backend=backend)
    if len(_FACTOR_CACHE) >= _FACTOR_CACHE_MAX:
        _FACTOR_CACHE.pop(next(iter(_FACTOR_CACHE)))
    _FACTOR_CACHE[cache_key] = factor
    return factor


def h_weighted_error(error: np.ndarray, h: np.ndarray) -> float:
    """tr(E H E^T) -- the objective LDLQ actually minimises.

    Computed as sum((E @ H) * E) in float32 with a float64 final reduction.
    Upcasting H to float64 instead would allocate a fresh 680 MiB temporary on
    every chunk; the float32 matmul's ~1e-5 relative error is far below the
    percent-level differences being compared.
    """
    h32 = np.ascontiguousarray(h, dtype=np.float32)
    total = 0.0
    step = max(1, 8_000_000 // max(1, h.shape[0]))
    for start in range(0, error.shape[0], step):
        chunk = np.ascontiguousarray(error[start:start + step], dtype=np.float32)
        total += float(np.sum((chunk @ h32) * chunk, dtype=np.float64))
    return total


# ---------------------------------------------------------------------------
# Codebook, fitted exactly as HQ2V fits it
# ---------------------------------------------------------------------------

def fit_codebook(values: np.ndarray, iterations: int = 30, seed: int = 0) -> np.ndarray:
    """Mirror of HQ2V's pass 1: RMS-normalised quad subsample, then Lloyd.

    Calls `hq2v._block_scales` and `hq2v._fit_codebook` so the procedure is
    literally the baseline's, and rounds through FP16 because that is how the
    format stores the codebook.
    """
    work = np.ascontiguousarray(values, dtype=np.float32)
    n_out, n_in = work.shape
    rows_per_chunk = max(1, ROW_CHUNK_WEIGHTS // n_in)
    total_quads = (n_out * n_in) // DIM
    rng = np.random.default_rng(seed)

    parts = []
    for start in range(0, n_out, rows_per_chunk):
        stop = min(start + rows_per_chunk, n_out)
        blocks = work[start:stop].reshape(-1, BLOCK)
        quads = (blocks / hq2v._block_scales(blocks)).reshape(-1, DIM)
        want = min(max(1, int(round(FIT_SAMPLE * quads.shape[0] / total_quads))),
                   quads.shape[0])
        parts.append(quads[rng.choice(quads.shape[0], size=want, replace=False)].copy())
        del blocks, quads

    codebook = hq2v._fit_codebook(np.concatenate(parts), iterations, seed, CODEBOOK)
    del parts
    return codebook.astype(np.float16).astype(np.float32)


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

@dataclass
class LdlqStats:
    mse: float
    rel_h_error: float
    rel_h_error_baseline: float | None = None
    damping: float = DEFAULT_DAMPING
    blocks: int = 0
    sweep_dtype: str = "float32"


def _quantize_block(work_block: np.ndarray, codebook: np.ndarray,
                    weight_block: np.ndarray | None, scale_rounds: int,
                    factor_block: np.ndarray, propagate: bool) -> tuple[np.ndarray, np.ndarray]:
    """Quantize one 256-column block with intra-block quad feedback.

    `work_block` is (out, 256) and is modified in place by the feedback.
    Returns (reconstruction, scaled_errors), both (out, 256).
    """
    n_out = work_block.shape[0]
    n_quads = BLOCK // DIM
    entry = work_block.copy()          # to restore between scale rounds
    diag = np.diag(factor_block)

    scale = None
    recon = np.empty_like(work_block)
    scaled_err = np.empty_like(work_block)

    for round_index in range(max(1, scale_rounds)):
        if round_index:
            work_block[:] = entry
        blk_scale = hq2v._block_scales(work_block) if scale is None else scale
        chosen_all = np.empty_like(work_block)

        for q in range(n_quads):
            lo, hi = q * DIM, (q + 1) * DIM
            x = np.ascontiguousarray(work_block[:, lo:hi] / blk_scale)
            if weight_block is None:
                labels = hq2v._assign_plain(x, codebook)
            else:
                w = np.ascontiguousarray(
                    np.broadcast_to(weight_block[lo:hi], (n_out, DIM)))
                labels = hq2v._assign_weighted(x, w, codebook)
            chosen = codebook[labels]
            chosen_all[:, lo:hi] = chosen
            q_vals = chosen * blk_scale
            recon[:, lo:hi] = q_vals

            err = work_block[:, lo:hi] - q_vals
            scaled = err / diag[lo:hi]
            scaled_err[:, lo:hi] = scaled
            # Propagate only past the quad: its own 4 coordinates were committed
            # jointly and must not be revised.
            if propagate and hi < BLOCK:
                work_block[:, hi:] -= scaled @ factor_block[lo:hi, hi:]

        if round_index + 1 >= max(1, scale_rounds):
            break
        # Re-solve the block scale against the codewords just chosen, then redo
        # the block from its entry state with the improved scale.
        scale = hq2v._refine_scale(entry, weight_block[None, :].repeat(n_out, 0)
                                   if weight_block is not None else None,
                                   chosen_all, blk_scale)

    return recon, scaled_err


def quantize_tensor(values: np.ndarray, hessian: np.ndarray,
                    importance_row: np.ndarray | None,
                    codebook: np.ndarray | None = None,
                    iterations: int = 30, seed: int = 0,
                    scale_rounds: int = 3,
                    damping: float = DEFAULT_DAMPING,
                    sweep_dtype: str = "float32",
                    measure_baseline: bool = False,
                    propagate: bool = True,
                    factor_key: str | None = None) -> tuple[np.ndarray, LdlqStats]:
    """Quantize one [out, in] matrix with LDLQ feedback; return reconstruction.

    `propagate=False` disables the feedback while keeping every other code path
    identical, which is the control that isolates the feedback's contribution
    from any incidental difference between this implementation and `hq2v`.
    """
    original_dtype = values.dtype
    n_out, n_in = values.shape
    if n_in % BLOCK:
        raise ValueError(f"LDLQ needs the input dim divisible by {BLOCK}, got {n_in}")
    if hessian.shape != (n_in, n_in):
        raise ValueError(f"Hessian {hessian.shape} does not match input dim {n_in}")

    if codebook is None:
        codebook = fit_codebook(values, iterations=iterations, seed=seed)

    weight_row = None
    if importance_row is not None:
        if importance_row.size != n_in:
            raise ValueError(f"imatrix length {importance_row.size} != input dim {n_in}")
        weight_row = importance_row.astype(np.float32)

    dtype = np.dtype(sweep_dtype)
    if factor_key is None:
        factor = prepare_factor(hessian, damping)
    else:
        factor = cached_factor(hessian, factor_key, damping)
    factor = factor.astype(dtype, copy=False)
    # `work` is written in place by the feedback sweep, so it MUST be private.
    # np.ascontiguousarray returns its input unchanged when dtype and contiguity
    # already match, which silently aliased the caller's array whenever `values`
    # arrived as float32 C-contiguous -- destroying it. The GGUF bridge was spared
    # only because its payloads are float16, forcing a conversion copy. np.array
    # copies unconditionally.
    work = np.array(values, dtype=dtype, order="C")
    original = np.array(values, dtype=np.float32, order="C")
    if np.shares_memory(work, values) or np.shares_memory(original, values):
        raise AssertionError("quantize_tensor would mutate its caller's weights")
    output = np.empty((n_out, n_in), dtype=dtype)

    n_blocks = n_in // BLOCK
    for b in range(n_blocks):
        c0, c1 = b * BLOCK, (b + 1) * BLOCK
        work_block = np.ascontiguousarray(work[:, c0:c1])
        recon, scaled_err = _quantize_block(
            work_block, codebook,
            None if weight_row is None else weight_row[c0:c1],
            scale_rounds, np.ascontiguousarray(factor[c0:c1, c0:c1]), propagate)
        output[:, c0:c1] = recon
        # One bulk update for every column after this block.
        if propagate and c1 < n_in:
            work[:, c1:] -= scaled_err @ factor[c0:c1, c1:]
        del work_block, recon, scaled_err

    del factor, work
    gc.collect()

    restored = output.astype(np.float32)
    error = restored - original
    stats = LdlqStats(
        mse=float(np.mean(error.astype(np.float64) ** 2)),
        rel_h_error=0.0,
        damping=damping,
        blocks=n_blocks,
        sweep_dtype=str(dtype),
    )
    denom = h_weighted_error(original, hessian)
    stats.rel_h_error = h_weighted_error(error, hessian) / denom

    if measure_baseline:
        base, _ = hq2v.roundtrip(values, importance_row, iterations=iterations,
                                 seed=seed, scale_rounds=scale_rounds)
        base_err = base.astype(np.float32) - original
        stats.rel_h_error_baseline = h_weighted_error(base_err, hessian) / denom
        del base, base_err

    del error, original
    gc.collect()
    return restored.astype(original_dtype), stats


def roundtrip(values: np.ndarray, importance_row: np.ndarray | None,
              tensor_name: str, iterations: int = 30, seed: int = 0,
              scale_rounds: int = 3, damping: float = DEFAULT_DAMPING,
              hessian_dir: Path = HESSIAN_DIR,
              sweep_dtype: str = "float32") -> tuple[np.ndarray, float]:
    """Bridge-facing entry point, matching `hq2v.roundtrip`'s contract."""
    path = hessian_path_for(tensor_name, hessian_dir)
    hessian = load_hessian(tensor_name, values.shape[1], hessian_dir)
    restored, stats = quantize_tensor(
        values, hessian, importance_row, iterations=iterations, seed=seed,
        scale_rounds=scale_rounds, damping=damping, sweep_dtype=sweep_dtype,
        factor_key=str(path))
    del hessian
    gc.collect()
    return restored, stats.mse
