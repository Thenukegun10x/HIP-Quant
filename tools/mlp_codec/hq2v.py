"""HQ2V: 2-bit *joint* quantization with importance-weighted assignment.

Why this format exists, from the ladder's own measurements: `iq2_xxs`
reconstructs weights no more accurately than HQ2 -- 9.16 dB versus 9.18 dB SQNR
-- yet produces a 1.59x better model while spending 0.19 fewer bits per weight.
HQ2's deficit is therefore not rate-distortion. It is *error placement*, and the
cause is structural.

With four scalar centroids, every weight independently snaps to whichever is
nearest, so the allocation of error within a block is forced. Importance
weighting can only slide the four shared centroids, which trades accuracy
globally across all 256 weights in the block; it cannot make one important
weight more accurate than an unimportant weight holding the same value.

A joint codebook removes that constraint. Quantizing `DIM` weights as one vector
lets the search pick a codeword that concentrates error on the coordinates the
model is least sensitive to. Measured on six Qwen3.5-4B MLP tensors, this is
worth +1.03 dB of importance-weighted SQNR against shipping HQ2 while spending
0.1875 fewer bits per weight.

Physical format, per 256-weight block:

    1 FP16 scale (block RMS)                        16 bits
    64 quad indices at 8 bits each                 512 bits
                                                   ---------
                                                   528 bits   = 2.0625 bpw

plus one codebook shared across the whole tensor: 256 entries x 4 dims x FP16 =
2 KiB. Against a ~23.6 M-weight MLP tensor that is 0.0007 bpw, so the true rate
is 2.0632 bpw. That is deliberately the same rate as `iq2_xxs` (2.0625), which
makes the ladder comparison head to head rather than an interpolation.

Two deliberate choices worth stating:

* The codebook is fitted **unweighted** and only the *assignment* is weighted.
  That is what was measured, and it isolates the mechanism under test: the gain
  comes from having somewhere to put the error, not from a better-placed
  codebook. Weighted fitting is a further refinement, not this experiment.
* Centroids are rounded to FP16 before assignment, so the reported error is the
  error of the format as it would actually be stored, not of an idealised
  float32 codebook.
"""

from __future__ import annotations

import numpy as np

BLOCK = 256          # weights per block, matching the HQ family
DIM = 4              # weights quantized jointly
CODEBOOK = 256       # signed codewords, i.e. 8 bits per DIM weights
SCALE_BITS = 16      # FP16 per-block scale
CENTROID_BITS = 16   # FP16 codebook entries

# Sign-symmetric variant. A signed 4-D codeword carries both a sign pattern
# (2^DIM = 16 possibilities) and a magnitude pattern, so 256 signed codewords is
# exactly 16 sign patterns x 16 magnitude patterns of capacity. Coding the signs
# explicitly and indexing a 16-entry *magnitude* codebook therefore costs the
# identical 8 bits per quad -- DIM sign bits plus 4 index bits -- while making a
# sign error structurally impossible.
#
# This is not a rate trade, it is a fix to a mis-specified codebook. Weight
# distributions are near-symmetric about zero, so the optimal codebook is
# sign-symmetric; unconstrained k-means allocates codewords by density instead
# and under-serves some orthants. Measured on six Qwen3.5-4B MLP tensors, the
# signed variant puts 9.42% of weights on the wrong side of zero, and those
# weights carry 16.4% of the squared error and 15.5% of the importance-weighted
# error. An oracle that corrects only the signs gains +0.47 dB weighted SQNR.
MAGNITUDE_CODEBOOK = 16

FIT_SAMPLE = 200_000     # quads used to fit the codebook
ROW_CHUNK_WEIGHTS = 2_000_000
ASSIGN_CHUNK = 50_000    # quads per distance evaluation; bounds peak memory


def bits_per_weight(n_weights: int, symmetric: bool = False) -> float:
    """Exact rate including the amortized per-tensor codebook.

    Both variants spend 8 bits per quad: the signed one on a single index into
    256 codewords, the symmetric one on DIM sign bits plus a 4-bit index into 16
    magnitude patterns. The symmetric variant is marginally *cheaper* overall
    because its codebook is 16 entries rather than 256.
    """
    blocks = n_weights // BLOCK
    payload = blocks * (SCALE_BITS + (BLOCK // DIM) * 8)
    entries = MAGNITUDE_CODEBOOK if symmetric else CODEBOOK
    codebook = entries * DIM * CENTROID_BITS
    return (payload + codebook) / n_weights


def _block_scales(blocks: np.ndarray) -> np.ndarray:
    """Per-block RMS, rounded through FP16 as the format stores it."""
    scale = np.sqrt((blocks.astype(np.float32) ** 2).mean(axis=1, keepdims=True))
    scale[scale == 0] = 1.0
    return scale.astype(np.float16).astype(np.float32)


def _assign_plain(vectors: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    norms = (codebook * codebook).sum(1)
    labels = np.empty(vectors.shape[0], dtype=np.int32)
    for start in range(0, vectors.shape[0], ASSIGN_CHUNK):
        chunk = vectors[start:start + ASSIGN_CHUNK]
        labels[start:start + ASSIGN_CHUNK] = np.argmin(
            norms[None, :] - 2.0 * (chunk @ codebook.T), axis=1)
    return labels


def _assign_weighted(vectors: np.ndarray, weights: np.ndarray,
                     codebook: np.ndarray) -> np.ndarray:
    """argmin_c sum_k w_k (x_k - c_k)^2, kept as two matmuls.

    The sum_k w_k x_k^2 term is constant for a given vector and so drops out of
    the argmin, leaving -2 (w*x).c + w.(c^2).
    """
    squared = (codebook ** 2).astype(np.float32)
    labels = np.empty(vectors.shape[0], dtype=np.int32)
    for start in range(0, vectors.shape[0], ASSIGN_CHUNK):
        x = vectors[start:start + ASSIGN_CHUNK]
        w = weights[start:start + ASSIGN_CHUNK]
        score = -2.0 * ((w * x) @ codebook.T) + (w @ squared.T)
        labels[start:start + ASSIGN_CHUNK] = np.argmin(score, axis=1)
    return labels


def _refine_scale(blocks: np.ndarray, weights: np.ndarray | None,
                  chosen: np.ndarray, previous: np.ndarray) -> np.ndarray:
    """Closed-form importance-weighted least-squares scale per block.

    Minimising sum_k w_k (x_k - s c_k)^2 over the single scalar s gives

        s = sum_k w_k x_k c_k / sum_k w_k c_k^2

    which is the optimum llama.cpp approaches by searching candidate scales.
    Solving it directly costs one pass instead of one pass per candidate.

    Accumulation is in float64: imatrix magnitudes span several orders across
    channels, and the products w*x*c summed over 256 terms are exactly where a
    float32 accumulator would quietly lose precision.  Rounded through FP16
    afterwards because that is how the format stores the scale.
    """
    if weights is None:
        num = (blocks.astype(np.float64) * chosen).sum(1, keepdims=True)
        den = (chosen.astype(np.float64) ** 2).sum(1, keepdims=True)
    else:
        w64 = weights.astype(np.float64)
        num = (w64 * blocks * chosen).sum(1, keepdims=True)
        den = (w64 * chosen * chosen).sum(1, keepdims=True)
    updated = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    # A block whose codewords are degenerate, or that lands on a non-positive
    # scale, keeps its previous value rather than being allowed to flip sign.
    bad = ~np.isfinite(updated) | (updated <= 0)
    updated[bad] = previous.astype(np.float64)[bad]
    return updated.astype(np.float16).astype(np.float32)


def _fit_codebook(sample: np.ndarray, iterations: int, seed: int,
                  entries: int = CODEBOOK) -> np.ndarray:
    """Lloyd's algorithm in DIM dimensions, from a random-sample initialisation."""
    rng = np.random.default_rng(seed)
    take = min(entries, sample.shape[0])
    codebook = sample[rng.choice(sample.shape[0], size=take, replace=False)].astype(np.float32)
    if take < entries:  # degenerate tensor; pad so the index space stays fixed-width
        codebook = np.vstack([codebook, np.zeros((entries - take, DIM), np.float32)])
    for _ in range(iterations):
        labels = _assign_plain(sample, codebook)
        for j in range(entries):
            members = labels == j
            if members.any():
                codebook[j] = sample[members].mean(0)
            else:
                # An empty cell wastes an index; reseed it onto a live vector.
                codebook[j] = sample[rng.integers(0, sample.shape[0])]
    return codebook


def roundtrip(values: np.ndarray, importance_row: np.ndarray | None,
              iterations: int = 30, seed: int = 0,
              scale_rounds: int = 1, symmetric: bool = False) -> tuple[np.ndarray, float]:
    """Encode and decode one [out, in] matrix; return the reconstruction and MSE.

    Work proceeds in two passes so that peak memory stays at one row-chunk
    rather than the whole tensor: the first pass collects block scales and a
    subsample of normalised vectors to fit the shared codebook, the second
    assigns and reconstructs. The codebook is fitted once per tensor, which is
    what the format's rate accounting assumes.

    ``scale_rounds`` controls scale selection. At 1 the block scale is simply its
    RMS, which is what the first HQ2V rows used. Above 1, assignment and a
    closed-form weighted least-squares scale are alternated, so the scale is
    chosen against the objective that actually matters rather than fixed a priori.
    The codebook is still fitted from RMS-normalised samples; refitting it against
    refined scales is a further level of alternation not attempted here.

    ``symmetric`` selects the sign-symmetric variant: signs are coded explicitly
    and the codebook holds 16 magnitude patterns rather than 256 signed ones. The
    rate is unchanged (see ``bits_per_weight``), but a sign error becomes
    impossible.
    """
    original_dtype = values.dtype
    work = np.ascontiguousarray(values, dtype=np.float32)
    n_out, n_in = work.shape
    if n_in % BLOCK:
        raise ValueError(f"HQ2V needs the input dim divisible by {BLOCK}, got {n_in}")
    if BLOCK % DIM:
        raise ValueError(f"BLOCK {BLOCK} must be divisible by DIM {DIM}")

    weights_row = None
    if importance_row is not None:
        if importance_row.size != n_in:
            raise ValueError(f"imatrix length {importance_row.size} != input dim {n_in}")
        weights_row = importance_row.astype(np.float32)

    rows_per_chunk = max(1, ROW_CHUNK_WEIGHTS // n_in)
    blocks_per_row = n_in // BLOCK
    total_quads = (n_out * n_in) // DIM
    rng = np.random.default_rng(seed)

    # Pass 1: block scales, plus a proportional subsample of normalised quads.
    scales = np.empty((n_out * blocks_per_row, 1), dtype=np.float32)
    sample_parts: list[np.ndarray] = []
    for start in range(0, n_out, rows_per_chunk):
        stop = min(start + rows_per_chunk, n_out)
        blocks = work[start:stop].reshape(-1, BLOCK)
        scale = _block_scales(blocks)
        scales[start * blocks_per_row:stop * blocks_per_row] = scale

        quads = (blocks / scale).reshape(-1, DIM)
        if symmetric:
            # The magnitude codebook is fitted on |x|; signs are carried exactly.
            quads = np.abs(quads)
        want = max(1, int(round(FIT_SAMPLE * quads.shape[0] / total_quads)))
        want = min(want, quads.shape[0])
        sample_parts.append(quads[rng.choice(quads.shape[0], size=want, replace=False)].copy())
        del blocks, quads

    entries = MAGNITUDE_CODEBOOK if symmetric else CODEBOOK
    codebook = _fit_codebook(np.concatenate(sample_parts), iterations, seed, entries)
    codebook = codebook.astype(np.float16).astype(np.float32)
    del sample_parts

    # Pass 2: assignment, optionally alternated with weighted scale refinement.
    rounds = max(1, int(scale_rounds))
    output = np.empty_like(work)
    squared_error = 0.0
    for start in range(0, n_out, rows_per_chunk):
        stop = min(start + rows_per_chunk, n_out)
        blocks = work[start:stop].reshape(-1, BLOCK)
        scale = scales[start * blocks_per_row:stop * blocks_per_row].copy()

        block_weights = None
        if weights_row is not None:
            block_weights = np.ascontiguousarray(
                np.broadcast_to(weights_row, (stop - start, n_in))).reshape(-1, BLOCK)

        # Signs are exact in the symmetric variant, so they are taken from the
        # data once and never quantized. Zeros are treated as positive to keep the
        # assignment objective identical to the reconstruction.
        signs = None
        if symmetric:
            signs = np.where(blocks < 0.0, np.float32(-1.0), np.float32(1.0))

        labels = None
        for index in range(rounds):
            quads = (blocks / scale).reshape(-1, DIM)
            quads = np.ascontiguousarray(np.abs(quads) if symmetric else quads)
            if block_weights is None:
                labels = _assign_plain(quads, codebook)
            else:
                labels = _assign_weighted(quads, block_weights.reshape(-1, DIM), codebook)
            del quads
            if index + 1 >= rounds:
                break
            chosen = codebook[labels].reshape(blocks.shape)
            if symmetric:
                chosen = chosen * signs
            # With chosen = sign(x) * m, the closed form below reduces to
            # s = sum w |x| m / sum w m^2, which is the right solve for the
            # symmetric variant too.
            scale = _refine_scale(blocks, block_weights, chosen, scale)

        chosen = codebook[labels].reshape(blocks.shape)
        if symmetric:
            chosen = chosen * signs
        restored = (chosen * scale).astype(np.float32)
        output[start:stop] = restored.reshape(stop - start, n_in)
        squared_error += float(np.sum((restored - blocks).astype(np.float64) ** 2))
        del blocks, labels, restored, block_weights, chosen, signs

    return output.astype(original_dtype), squared_error / work.size
