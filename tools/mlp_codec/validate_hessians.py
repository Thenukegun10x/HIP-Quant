"""Validate collected Hessians, and cross-check them against the llama.cpp imatrix.

Three independent checks, cheapest first.

1. Structure.  Every H must be square, finite, symmetric, and have a
   non-negative diagonal.  A Gram matrix that fails any of these was not
   accumulated correctly.

2. Cross-codebase agreement.  This is the check that matters.  llama.cpp's
   imatrix stores `<tensor>.in_sum2` = sum_t x_j^2 and `<tensor>.counts` = n,
   so `in_sum2 / counts` is *precisely* diag(H) -- the same quantity, measured
   by a completely separate implementation (C++/Vulkan ggml graph) from a
   completely separate forward pass (llama.cpp's, not transformers').  If the
   two agree closely, then the PyTorch hooks are attached to the right tensors,
   in the right input space, with the right semantics.  If they disagree, the
   Hessians are measuring something other than what the ladder quantizes.

   Agreement will not be exact: the imatrix ran 264 chunks in file order while
   this collector strides and shuffles, and the two forward implementations
   differ in kernel and accumulation order.  Correlation is the signal, not
   equality.

3. Conditioning.  LDLQ needs a Cholesky factor, so the practical question is
   the smallest diagonal damping lambda (as a fraction of mean(diag H)) for
   which `cholesky(H + lambda * mean(diag H) * I)` succeeds.  That number is a
   direct input to the next step, not just a diagnostic.

The Cholesky probes on the 9216x9216 `ffn_down` Hessians are the expensive part
(~2.6e11 flops each), so by default they run on a sample of blocks; pass
`--damp-blocks all` to do every one.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

RESEARCH = Path(r"G:\hq2_research")
HESS_DIR = RESEARCH / "hessians"
IMATRIX = RESEARCH / "imatrix" / "qwen35-4b-wikitrain.gguf"

# One Hessian serves gate and up, so a single collected file is compared against
# two imatrix entries.
KIND_TO_IMATRIX = {
    "ffn_gate_up": ("ffn_gate", "ffn_up"),
    "ffn_down": ("ffn_down",),
}

DAMP_CANDIDATES = (0.0, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1)


def load_imatrix_diagonals(path: Path) -> dict[str, np.ndarray]:
    """tensor name -> mean(x_j^2), i.e. what diag(H) should equal."""
    from gguf import GGUFReader

    reader = GGUFReader(str(path), "r")
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, float] = {}
    for tensor in reader.tensors:
        name = str(tensor.name)
        if name.endswith(".in_sum2"):
            sums[name[: -len(".in_sum2")]] = np.array(tensor.data, dtype=np.float64)
        elif name.endswith(".counts"):
            counts[name[: -len(".counts")]] = float(np.array(tensor.data).reshape(-1)[0])
    out = {}
    for name, s in sums.items():
        n = counts.get(name)
        if n:
            out[name] = s / n
    del reader, sums
    gc.collect()
    return out


def structure_check(h: np.ndarray) -> dict:
    d = h.shape[0]
    diag = np.diag(h)
    asym = float(np.abs(h - h.T).max())
    scale = float(np.abs(h).max()) or 1.0
    return {
        "dim": int(d),
        "square": h.ndim == 2 and h.shape[0] == h.shape[1],
        "finite": bool(np.isfinite(h).all()),
        "max_asymmetry_rel": asym / scale,
        "diag_min": float(diag.min()),
        "diag_mean": float(diag.mean()),
        "diag_max": float(diag.max()),
        "diag_nonneg": bool((diag >= 0).all()),
        "diag_zeros": int((diag == 0).sum()),
        "trace": float(np.trace(h)),
    }


def min_damping(h: np.ndarray) -> dict:
    """Smallest lambda in DAMP_CANDIDATES for which Cholesky succeeds."""
    d = h.shape[0]
    mean_diag = float(np.diag(h).mean())
    work = np.array(h, dtype=np.float64, order="F")
    idx = np.arange(d)
    base_diag = work[idx, idx].copy()
    result = {"mean_diag": mean_diag, "cholesky_lambda": None, "probes": {}}
    for lam in DAMP_CANDIDATES:
        work[idx, idx] = base_diag + lam * mean_diag
        try:
            np.linalg.cholesky(work)
            ok = True
        except np.linalg.LinAlgError:
            ok = False
        result["probes"][f"{lam:g}"] = ok
        if ok:
            result["cholesky_lambda"] = lam
            break
    del work
    gc.collect()
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hessians", type=Path, default=HESS_DIR)
    ap.add_argument("--imatrix", type=Path, default=IMATRIX)
    ap.add_argument("--damp-blocks", default="0,15,31",
                    help="'all', 'none', or a comma list of block indices")
    ap.add_argument("--min-corr", type=float, default=0.99,
                    help="fail if any diag(H) vs imatrix correlation falls below this")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    files = sorted(args.hessians.glob("blk*.npy"))
    if not files:
        print(f"no hessians found in {args.hessians}", file=sys.stderr)
        return 2

    print(f"imatrix : {args.imatrix}")
    imat = load_imatrix_diagonals(args.imatrix)
    print(f"          {len(imat)} tensors with in_sum2/counts")

    if args.damp_blocks == "all":
        damp_blocks = set(range(64))
    elif args.damp_blocks == "none":
        damp_blocks = set()
    else:
        damp_blocks = {int(x) for x in args.damp_blocks.split(",") if x.strip()}

    print(f"hessians: {len(files)} files in {args.hessians}")
    print()
    header = (f"{'key':22s} {'dim':>5s} {'asym':>9s} {'diag_mean':>11s} "
              f"{'corr':>9s} {'ratio':>8s} {'lambda':>8s}")
    print(header)
    print("-" * len(header))

    report: dict[str, dict] = {}
    worst_corr = 1.0
    failures: list[str] = []

    for path in files:
        key = path.stem                       # e.g. blk07.ffn_down
        block_s, kind = key.split(".", 1)
        block = int(block_s[3:])

        h = np.load(path).astype(np.float64, copy=False)
        entry = structure_check(h)
        entry["kind"] = kind
        entry["block"] = block

        for flag in ("square", "finite", "diag_nonneg"):
            if not entry[flag]:
                failures.append(f"{key}: {flag} failed")
        if entry["max_asymmetry_rel"] > 1e-12:
            failures.append(f"{key}: asymmetry {entry['max_asymmetry_rel']:.2e} > 1e-12")

        # --- cross-check against the imatrix ---
        diag = np.diag(h)
        corrs = {}
        ratios = {}
        for short in KIND_TO_IMATRIX[kind]:
            name = f"blk.{block}.{short}.weight"
            ref = imat.get(name)
            if ref is None:
                failures.append(f"{key}: no imatrix entry {name}")
                continue
            if ref.shape != diag.shape:
                failures.append(f"{key}: imatrix dim {ref.shape} != H dim {diag.shape}")
                continue
            # Channels the imatrix never saw activate are structurally zero in
            # both; including them would inflate the correlation.
            mask = (ref > 0) & (diag > 0)
            if mask.sum() < 16:
                failures.append(f"{key}: only {int(mask.sum())} comparable channels")
                continue
            c = float(np.corrcoef(diag[mask], ref[mask])[0, 1])
            corrs[short] = c
            ratios[short] = float(np.median(diag[mask] / ref[mask]))
            if c < args.min_corr:
                failures.append(f"{key} vs {name}: corr {c:.5f} < {args.min_corr}")
            worst_corr = min(worst_corr, c)
        entry["imatrix_corr"] = corrs
        entry["imatrix_median_ratio"] = ratios

        # --- conditioning ---
        lam_s = ""
        if block in damp_blocks:
            damp = min_damping(h)
            entry["damping"] = damp
            lam = damp["cholesky_lambda"]
            lam_s = "PSD" if lam == 0.0 else (f"{lam:g}" if lam is not None else "FAIL")
            if lam is None:
                failures.append(f"{key}: no damping up to {DAMP_CANDIDATES[-1]} gave a Cholesky")

        c_s = "/".join(f"{v:.5f}" for v in corrs.values()) or "-"
        r_s = "/".join(f"{v:.3f}" for v in ratios.values()) or "-"
        print(f"{key:22s} {entry['dim']:5d} {entry['max_asymmetry_rel']:9.2e} "
              f"{entry['diag_mean']:11.4g} {c_s:>9s} {r_s:>8s} {lam_s:>8s}", flush=True)

        report[key] = entry
        del h, diag
        gc.collect()

    print()
    print(f"worst diag(H) vs imatrix correlation: {worst_corr:.6f}")
    lams = [e["damping"]["cholesky_lambda"] for e in report.values() if "damping" in e]
    if lams:
        finite = [x for x in lams if x is not None]
        print(f"damping probed on {len(lams)} tensors; "
              f"max lambda needed: {max(finite) if finite else 'FAIL'}")
    if failures:
        print()
        print(f"FAILURES ({len(failures)}):")
        for f in failures:
            print("  -", f)
    else:
        print("all checks passed")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(
            {"worst_corr": worst_corr, "failures": failures, "tensors": report},
            indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
