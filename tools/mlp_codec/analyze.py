"""Paired-bootstrap analysis of the codec ladder.

Every row scores the identical token stream against the identical base logit
file, which makes this a *paired* design.  Exploiting that matters: the chunk
to chunk variance of perplexity is large compared with the differences between
codecs, so unpaired confidence intervals would swamp real effects.  Resampling
chunks and comparing rows within each resample cancels the shared per-chunk
difficulty and gives intervals that reflect the codec difference alone.

llama.cpp's KL mode makes the pairing free.  Its running table prints, per
chunk, the cumulative mean of ln(PPL(Q)/PPL(base)) -- a quantity already
differenced against the base model token by token.  Inverting the cumulative
mean recovers a per-chunk paired difference series, so no separate F16
perplexity run is needed and no base/row chunk misalignment is possible.

Two reporting choices are deliberate:

* KL is summarised by median and p99, not only the mean.  Mean KL is dominated
  by a small number of catastrophic tokens, so two codecs with very different
  typical behaviour can report similar means.
* Perplexity differences are reported in log space.  PPL ratios are
  multiplicative, so ln PPL is the scale on which they are symmetric and on
  which a bootstrap interval is meaningful.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def per_chunk_from_cumulative(cumulative: list[float]) -> np.ndarray:
    """Recover per-chunk mean negative log-likelihood from llama.cpp's running PPL.

    llama-perplexity prints a cumulative estimate after each chunk: entry k is
    exp(mean NLL over chunks 1..k).  Undoing that gives the per-chunk mean NLL,
    which is the quantity that is actually independent across chunks and can be
    resampled.
    """
    values = np.asarray(cumulative, dtype=np.float64)
    if values.size == 0:
        return values
    cumulative_nll = np.log(values) * np.arange(1, values.size + 1)
    return np.diff(np.concatenate(([0.0], cumulative_nll)))


def per_chunk_from_cumulative_mean(cumulative: list[float]) -> np.ndarray:
    """Recover a per-chunk series from a cumulative *mean* column.

    The ln-ratio and KL columns of the KL-divergence table are already means in
    their own units, so unlike the PPL column they must not be log-transformed
    first: entry k times k is the running sum, and differencing that gives the
    per-chunk value.  Each chunk contributes the same number of scored tokens,
    so a mean over tokens and a mean over chunk-means coincide.
    """
    values = np.asarray(cumulative, dtype=np.float64)
    if values.size == 0:
        return values
    running_sum = values * np.arange(1, values.size + 1)
    return np.diff(np.concatenate(([0.0], running_sum)))


def inversion_error(per_chunk: np.ndarray, cumulative: list[float]) -> float:
    """Largest absolute error from re-deriving the cumulative column.

    The inversions above are the one place a silent indexing slip would corrupt
    every interval downstream, so each series is reconstructed and checked.
    """
    values = np.asarray(cumulative, dtype=np.float64)
    if values.size == 0:
        return 0.0
    rebuilt = np.cumsum(per_chunk) / np.arange(1, values.size + 1)
    return float(np.max(np.abs(rebuilt - values)))


def bootstrap_mean(delta: np.ndarray, iterations: int = 20000, seed: int = 0) -> dict:
    """Bootstrap the mean of an already-paired per-chunk series."""
    rng = np.random.default_rng(seed)
    n = int(delta.size)
    if n == 0:
        return {}
    idx = rng.integers(0, n, size=(iterations, n))
    means = delta[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "mean": float(delta.mean()),
        "ci95": [float(lo), float(hi)],
        "n_chunks": n,
        "bootstrap_iterations": iterations,
    }


def paired_bootstrap(row_nll: np.ndarray, base_nll: np.ndarray,
                     iterations: int = 20000, seed: int = 0) -> dict:
    """Bootstrap the mean per-chunk NLL difference against a separate base run."""
    n = min(row_nll.size, base_nll.size)
    if n == 0:
        return {}
    return bootstrap_mean(row_nll[:n] - base_nll[:n], iterations, seed)


def summarise_delta(delta: np.ndarray, iterations: int, seed: int = 0) -> dict:
    """Express a paired log-space difference as both a log delta and a PPL ratio."""
    boot = bootstrap_mean(delta, iterations, seed)
    if not boot:
        return {}
    lo, hi = boot["ci95"]
    return {
        "delta_ln_ppl": boot["mean"],
        "delta_ln_ppl_ci95": [lo, hi],
        "ppl_ratio": math.exp(boot["mean"]),
        "ppl_ratio_ci95": [math.exp(lo), math.exp(hi)],
        "n_chunks": boot["n_chunks"],
        "bootstrap_iterations": boot["bootstrap_iterations"],
    }


def compare_rows(a_delta: np.ndarray, b_delta: np.ndarray,
                 iterations: int = 20000, seed: int = 0) -> dict:
    """Is row A better than row B, accounting for shared chunk difficulty?

    Both inputs are differences against the same base, so subtracting them
    cancels the base and leaves a directly paired A-vs-B series.
    """
    rng = np.random.default_rng(seed)
    n = min(a_delta.size, b_delta.size)
    if n == 0:
        return {}
    delta = a_delta[:n] - b_delta[:n]
    idx = rng.integers(0, n, size=(iterations, n))
    means = delta[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "delta_ln_ppl": float(delta.mean()),
        "ci95": [float(lo), float(hi)],
        "ppl_ratio": float(math.exp(delta.mean())),
        # Fraction of resamples in which A is worse; a two-sided screen for
        # whether the sign of the difference is stable.
        "p_a_worse": float((means > 0).mean()),
        "significant_at_95": bool(lo > 0 or hi < 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="ladder_results.json")
    parser.add_argument("--base-ppl-log", default=None,
                        help="fallback F16 perplexity log, only needed for rows "
                             "scored outside KL mode")
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--bpw", default=None,
                        help="optional JSON mapping row name -> bits per weight")
    args = parser.parse_args()

    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    bpw = json.loads(Path(args.bpw).read_text(encoding="utf-8")) if args.bpw else {}

    base_nll = None
    if args.base_ppl_log:
        import re
        text = Path(args.base_ppl_log).read_text(encoding="utf-8", errors="replace").replace("\r", "")
        chunks = [float(v) for _, v in re.findall(r"\[(\d+)\]([\d.]+),", text)]
        if chunks:
            base_nll = per_chunk_from_cumulative(chunks)

    table: dict[str, dict] = {}
    delta_series: dict[str, np.ndarray] = {}
    checks: dict[str, dict] = {}

    for row, data in results.items():
        entry = {
            "kind": data.get("kind"),
            "bpw": bpw.get(row),
            # KL mode never prints "Final estimate: PPL ="; it reports the same
            # quantity as "Mean PPL(Q)".  Fall back so both modes populate one column.
            "ppl": data.get("ppl") or data.get("ppl_q"),
            "ppl_stderr": data.get("ppl_stderr") or data.get("ppl_q_stderr"),
            "ppl_base": data.get("ppl_base"),
            "kl_mean": data.get("kl_mean"),
            "kl_median": data.get("kl_median"),
            "kl_p90": data.get("kl_p90"),
            "kl_p99": data.get("kl_p99"),
            "top1_agreement": data.get("top1_agreement"),
            "delta_p_rms": data.get("delta_p_rms"),
            "gguf_gib": round(data.get("gguf_bytes", 0) / 1024**3, 3),
        }

        # Preferred path: the KL table's ln-ratio column is already paired.
        if cumulative := data.get("cumulative_ln_ratio_by_chunk"):
            delta = per_chunk_from_cumulative_mean(cumulative)
            checks[row] = {"ln_ratio_inversion_max_error": inversion_error(delta, cumulative)}
            delta_series[row] = delta
            entry["paired_via"] = "kl_ln_ratio_column"
            entry.update(summarise_delta(delta, args.iterations))
        elif base_nll is not None and (cumulative := data.get("cumulative_ppl_by_chunk")):
            nll = per_chunk_from_cumulative(cumulative)
            n = min(nll.size, base_nll.size)
            delta = nll[:n] - base_nll[:n]
            delta_series[row] = delta
            entry["paired_via"] = "separate_base_ppl_log"
            entry.update(summarise_delta(delta, args.iterations))

        # A bootstrap interval on mean KL, which the log reports without one.
        if cumulative := data.get("cumulative_kl_by_chunk"):
            kl_per_chunk = per_chunk_from_cumulative_mean(cumulative)
            checks.setdefault(row, {})["kl_inversion_max_error"] = inversion_error(
                kl_per_chunk, cumulative)
            boot = bootstrap_mean(kl_per_chunk, args.iterations)
            entry["kl_mean_bootstrap"] = boot.get("mean")
            entry["kl_mean_ci95"] = boot.get("ci95")

        entry["n_chunks"] = data.get("n_chunks")
        table[row] = entry

    # Pairwise comparison against the best row by mean paired delta, which is
    # the question that actually matters: is any apparent ordering real?
    pairwise: dict[str, dict] = {}
    if delta_series:
        best = min(delta_series, key=lambda r: delta_series[r].mean())
        for row, delta in delta_series.items():
            if row != best:
                pairwise[f"{row}_vs_{best}"] = compare_rows(delta, delta_series[best],
                                                            args.iterations)
        pairwise["_reference_row"] = best

    # Neighbour comparisons ordered by bit rate: the adjacent-row question is
    # where a rate-distortion claim actually lives.
    neighbours: dict[str, dict] = {}
    rated = sorted((r for r in delta_series if table[r].get("bpw")),
                   key=lambda r: table[r]["bpw"])
    for lower, higher in zip(rated, rated[1:]):
        neighbours[f"{lower}_vs_{higher}"] = {
            "bpw": [table[lower]["bpw"], table[higher]["bpw"]],
            **compare_rows(delta_series[lower], delta_series[higher], args.iterations),
        }

    worst_check = max((v for row in checks.values() for v in row.values()), default=0.0)
    report = {
        "rows": table,
        "pairwise_vs_best": pairwise,
        "adjacent_by_bpw": neighbours,
        "inversion_checks": checks,
        "worst_inversion_error": worst_check,
        "notes": [
            "Paired series taken from the KL table's ln(PPL(Q)/PPL(base)) column, "
            "which is differenced against the base per token.",
            "Cumulative-mean columns are inverted by running-sum differencing, not "
            "by the log inversion used for the PPL column.",
            "Every inverted series is reconstructed and checked; see inversion_checks.",
            "delta_ln_ppl is in log space; ppl_ratio = exp(delta_ln_ppl).",
            "KL median and p99 are reported because mean KL is outlier-dominated.",
        ],
    }
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")

    width = max((len(r) for r in table), default=4)
    print(f"{'row':<{width}}  {'bpw':>5} {'PPL':>9} {'ratio':>7} {'ratio CI95':>18} "
          f"{'KLmed':>8} {'KLp99':>8} {'top1%':>7}")
    for row, entry in sorted(table.items(),
                             key=lambda kv: kv[1].get("delta_ln_ppl") if
                             kv[1].get("delta_ln_ppl") is not None else 1e9):
        ci = entry.get("ppl_ratio_ci95")
        ci_text = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "-"
        bpw_text = f"{entry['bpw']:.2f}" if entry.get("bpw") else "-"
        print(f"{row:<{width}}  {bpw_text:>5} {entry.get('ppl') or 0:>9.4f} "
              f"{entry.get('ppl_ratio') or 0:>7.4f} {ci_text:>18} "
              f"{entry.get('kl_median') or 0:>8.4f} {entry.get('kl_p99') or 0:>8.4f} "
              f"{entry.get('top1_agreement') or 0:>7.2f}")
    print(f"\nworst inversion error: {worst_check:.3e}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
