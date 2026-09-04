"""Run the MLP-only codec ladder.

One variable changes across rows: the codec applied to the MLP projections.
Everything else -- the tied token embedding that doubles as the output head,
attention, norms, the token stream, the base logit file, and the importance
matrix -- is identical in every row.

GGML codecs are produced by llama-quantize.  Note the construction: a base type
of F16 plus --tensor-type overrides silently does nothing, because the override
block in src/llama-quant.cpp is guarded by `ggml_is_quantized(default_type)`.
So the base type is the codec under test and two ordered regex overrides do the
work -- the first claims the MLP tensors, the second sweeps everything else back
to F16.  Matching is regex_search and breaks on first match.

HQ rows cannot be produced this way because llama.cpp has no HQ codec; they go
through hq2_bridge.py, which decodes the codec back to F16 in place.

Every row is scored against one shared base logit file, so PPL ratio, KL, and
top-1 agreement are all paired comparisons on identical targets.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "mlp_codec"))
# gguf-py ships inside the vendored llama.cpp checkout rather than being installed.
sys.path.insert(0, str(REPO / "Own Quant" / "llama_cpp_stock" / "gguf-py"))

import ledger  # noqa: E402
from safe_run import launch  # noqa: E402

BIN = REPO / "Own Quant" / "llama_cpp_stock_vulkan_build" / "bin"
QUANTIZE = BIN / "llama-quantize.exe"
PERPLEXITY = BIN / "llama-perplexity.exe"

# The catch-all that protects every tensor not under test.  Order matters:
# the first matching pattern wins, so the MLP pattern is always passed first.
CATCH_ALL = "."


def mlp_pattern_for(base_gguf: Path) -> tuple[str, list[str]]:
    """Build the MLP regex from the model itself, excluding non-forward-path blocks.

    Qwen3.5 ships a Multi-Token Prediction head as one extra block past the
    transformer stack (it carries `nextn.*` tensors alongside ordinary ffn
    ones).  That block is not on the normal forward path, so llama-imatrix
    never collects activation statistics for it, and a very-low-bit quantize
    aborts with "Missing importance matrix".  Quantizing it would also distort
    the bits-per-weight accounting while having no effect on perplexity.

    Deriving the block set from the file rather than hardcoding a layer count
    keeps this correct for models with or without such a head.
    """
    from gguf import GGUFReader

    reader = GGUFReader(str(base_gguf), "r")
    names = [str(t.name) for t in reader.tensors]
    del reader

    excluded = {m.group(1) for n in names if (m := re.match(r"blk\.(\d+)\.nextn\.", n))}
    targets = sorted(
        n for n in names
        if (m := re.match(r"blk\.(\d+)\.ffn_(gate|up|down)\.weight$", n)) and m.group(1) not in excluded
    )
    if not targets:
        raise SystemExit(f"no MLP tensors found in {base_gguf}")

    keep = sorted({int(re.match(r"blk\.(\d+)\.", n).group(1)) for n in targets})
    alternation = "|".join(str(i) for i in keep)
    pattern = rf"^blk\.({alternation})\.ffn_(gate|up|down)\.weight$"
    if excluded:
        print(f"[ladder] excluding non-forward-path blocks {sorted(excluded)} "
              f"(MTP/nextn head); {len(targets)} MLP tensors under test", flush=True)
    return pattern, targets

# GGML rows.  `imatrix` marks codecs that require or strongly benefit from one;
# the IQ2 family refuses to build without it.
# `type` must be a real ggml_type, not an ftype.  IQ2_M and Q3_K_S are ftype
# *mixtures* with no ggml_type of their own, so naming them here makes
# llama-quantize print its usage banner and exit 1.  `ftype` only has to be a
# quantized type: it sets default_type, and the override block that assigns the
# MLP tensors is reached only when ggml_is_quantized(default_type) holds.
# bpw values are the codec's own, for reference when reading the ladder.
GGML_ROWS = {
    "iq2_xxs":  {"ftype": "IQ2_XXS", "type": "iq2_xxs", "imatrix": True,  "bpw": 2.0625},
    "iq2_xs":   {"ftype": "IQ2_XS",  "type": "iq2_xs",  "imatrix": True,  "bpw": 2.3125},
    "iq2_s":    {"ftype": "IQ2_S",   "type": "iq2_s",   "imatrix": True,  "bpw": 2.5000},
    "q2_k":     {"ftype": "Q2_K",    "type": "q2_k",    "imatrix": True,  "bpw": 2.5625},
    "iq3_xxs":  {"ftype": "IQ3_XXS", "type": "iq3_xxs", "imatrix": True,  "bpw": 3.0625},
    "iq3_s":    {"ftype": "IQ3_S",   "type": "iq3_s",   "imatrix": True,  "bpw": 3.4375},
    "q3_k":     {"ftype": "Q3_K_S",  "type": "q3_k",    "imatrix": True,  "bpw": 3.4375},
    "q4_0":     {"ftype": "Q4_0",    "type": "q4_0",    "imatrix": False, "bpw": 4.5000},
}

# HQ rows, produced by the bridge rather than llama-quantize.
# HQ rows.  bpw is exact rather than nominal: an HQ2 block stores 4 FP16
# centroids plus 256 two-bit selectors in 72 bytes for 256 weights (2.25 bpw),
# and an HQ3 block stores 8 centroids plus 256 three-bit selectors in 112 bytes
# (3.5 bpw).  There is no per-block scale or min to account for.
HQ_ROWS = {
    "hq2_l8_imat":   {"format": "hq2", "iterations": 8, "imatrix": True,  "bpw": 2.25},
    "hq2_l8_noimat": {"format": "hq2", "iterations": 8, "imatrix": False, "bpw": 2.25},
    "hq3_l8_imat":   {"format": "hq3", "iterations": 8, "imatrix": True,  "bpw": 3.50},
    # HQ2V: joint 4-D quantization with importance-weighted assignment. Its rate
    # is deliberately iq2_xxs's 2.0625 bpw (2.0632 counting the shared codebook),
    # so the comparison against the codec that beat HQ2 is head to head.
    # 2.0632 not 2.0625: the payload is 2.0625 bpw and the shared per-tensor
    # codebook adds 0.0007. Counted honestly it is a hair *above* iq2_xxs.
    "hq2v_l30_imat":   {"format": "hq2v", "iterations": 30, "imatrix": True,  "bpw": 2.0632},
    "hq2v_l30_noimat": {"format": "hq2v", "iterations": 30, "imatrix": False, "bpw": 2.0632},
    # Same format and rate; the only change is that the per-block scale is solved
    # against the weighted objective instead of being fixed at the block RMS.
    "hq2v_l30s3_imat": {"format": "hq2v", "iterations": 30, "imatrix": True,
                        "bpw": 2.0632, "scale_rounds": 3},
    # Sign-symmetric: explicit sign bits plus a 16-entry magnitude codebook. Same
    # 8 bits per quad, and a hair cheaper overall than hq2v because the codebook
    # is 16 entries rather than 256. Sign errors become impossible.
    "hq2vs_l30s3_imat": {"format": "hq2vs", "iterations": 30, "imatrix": True,
                         "bpw": 2.0625, "scale_rounds": 3},
    # LDLQ: identical format, identical codebook, identical scale objective,
    # identical rate -- the only addition is Hessian error feedback, which costs
    # no bits, so this shares hq2v_l30s3_imat's 2.0632 exactly. Measured on two
    # tensors before committing: tr(EHE') falls 47.9% on average while MSE rises
    # 35.9%, which is the shaping signature the ladder has been pointing at.
    # Running it with feedback disabled reproduces hq2v_l30s3_imat's MSE to six
    # significant figures, so any ladder difference is attributable to feedback.
    "hq2vl_l30s3_imat": {"format": "hq2vl", "iterations": 30, "imatrix": True,
                         "bpw": 2.0632, "scale_rounds": 3},
}


def parse_perplexity(log_path: Path) -> dict:
    """Pull the scalar results and the per-chunk PPL series out of a run log."""
    text = log_path.read_text(encoding="utf-8", errors="replace").replace("\r", "")
    out: dict = {}

    if m := re.search(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)", text):
        out["ppl"] = float(m.group(1))
        out["ppl_stderr"] = float(m.group(2))

    # Plain perplexity mode emits "[1]12.34,[2]12.56,".
    chunks = re.findall(r"\[(\d+)\]([\d.]+),", text)
    if chunks:
        out["cumulative_ppl_by_chunk"] = [float(v) for _, v in chunks]
        out["n_chunks"] = len(chunks)

    # KL mode instead emits a running table whose columns are
    #   chunk | PPL ± e | ln(PPL(Q)/PPL(base)) ± e | KL ± e | Δp RMS ± e | Same top p ± e
    # Every value is cumulative over chunks 1..k.  The ln-ratio column is
    # already differenced against the base, so recovering its per-chunk values
    # yields a paired series directly -- no separate base run is needed to
    # bootstrap it.
    table = re.findall(
        r"^\s*(\d+)\s+([\d.]+)\s*±\s*([\d.]+)"
        r"\s+([-\d.]+)\s*±\s*([\d.]+)"
        r"\s+([-\d.]+)\s*±\s*([\d.]+)",
        text, flags=re.M)
    if table:
        out["cumulative_ppl_by_chunk"] = [float(r[1]) for r in table]
        out["cumulative_ln_ratio_by_chunk"] = [float(r[3]) for r in table]
        out["cumulative_kl_by_chunk"] = [float(r[5]) for r in table]
        out["n_chunks"] = len(table)

    # llama.cpp pads these labels to fixed columns, so every pattern has to
    # tolerate runs of whitespace.  Where a value carries its own standard
    # error ("x ± y") both halves are captured: the errors are what make the
    # comparison between rows defensible rather than decorative.
    scalar_patterns = {
        "kl_mean": r"Mean\s+KLD:\s*([-\d.eE+]+)\s*±\s*([\d.eE+-]+)",
        "ppl_q": r"Mean PPL\(Q\)\s*:\s*([\d.eE+-]+)\s*±\s*([\d.eE+-]+)",
        "ppl_base": r"Mean PPL\(base\)\s*:\s*([\d.eE+-]+)\s*±\s*([\d.eE+-]+)",
        "ln_ppl_ratio": r"Mean ln\(PPL\(Q\)/PPL\(base\)\)\s*:\s*([-\d.eE+]+)\s*±\s*([\d.eE+-]+)",
        "ppl_ratio": r"Mean PPL\(Q\)/PPL\(base\)\s*:\s*([\d.eE+-]+)\s*±\s*([\d.eE+-]+)",
        "top1_agreement": r"Same top p:\s*([\d.]+)\s*±\s*([\d.]+)\s*%",
        "delta_p_mean": r"Mean\s+Δp:\s*([-\d.]+)\s*±\s*([\d.]+)\s*%",
        "delta_p_rms": r"RMS Δp\s*:\s*([\d.]+)\s*±\s*([\d.]+)\s*%",
    }
    for key, pattern in scalar_patterns.items():
        if m := re.search(pattern, text):
            out[key] = float(m.group(1))
            out[f"{key}_stderr"] = float(m.group(2))

    # Quantiles are reported without an error term.
    quantile_patterns = {
        "kl_median": r"Median\s+KLD:\s*([-\d.eE+]+)",
        "kl_p90": r"90\.0%\s+KLD:\s*([-\d.eE+]+)",
        "kl_p99": r"99\.0%\s+KLD:\s*([-\d.eE+]+)",
        "kl_p999": r"99\.9%\s+KLD:\s*([-\d.eE+]+)",
        "kl_max": r"Maximum\s+KLD:\s*([-\d.eE+]+)",
    }
    for key, pattern in quantile_patterns.items():
        if m := re.search(pattern, text):
            out[key] = float(m.group(1))
    return out


def build_ggml_row(row: str, spec: dict, base_gguf: Path, out_gguf: Path,
                   imatrix: Path | None, log_dir: Path, mlp_pattern: str) -> int:
    cmd = [str(QUANTIZE), "--pure",
           "--token-embedding-type", "f16", "--output-tensor-type", "f16",
           "--tensor-type", f"{mlp_pattern}={spec['type']}",
           "--tensor-type", f"{CATCH_ALL}=f16"]
    if spec["imatrix"]:
        if imatrix is None:
            raise SystemExit(f"row {row} requires an imatrix")
        cmd += ["--imatrix", str(imatrix)]
    cmd += [str(base_gguf), str(out_gguf), spec["ftype"]]
    return launch(cmd, log_dir / f"quantize_{row}.log", timeout_s=7200, require_vram=False)


def build_hq_row(row: str, spec: dict, base_gguf: Path, out_gguf: Path,
                 imatrix: Path | None, log_dir: Path, backend: str, report: Path) -> int:
    python = REPO / ".venv" / "Scripts" / "python.exe"
    cmd = [str(python), str(REPO / "tools" / "mlp_codec" / "hq2_bridge.py"),
           "--source", str(base_gguf), "--output", str(out_gguf),
           "--format", spec["format"], "--iterations", str(spec["iterations"]),
           "--backend", backend, "--report", str(report)]
    if spec.get("scale_rounds"):
        cmd += ["--scale-rounds", str(spec["scale_rounds"])]
    if spec["imatrix"]:
        if imatrix is None:
            raise SystemExit(f"row {row} requires an imatrix")
        cmd += ["--imatrix", str(imatrix)]
    return launch(cmd, log_dir / f"bridge_{row}.log", timeout_s=14400, require_vram=False)


def score_row(row: str, gguf: Path, corpus: Path, base_logits: Path,
              log_dir: Path, chunks: int, n_ctx: int, ngl: int) -> dict:
    log_path = log_dir / f"ppl_{row}.log"
    cmd = [str(PERPLEXITY), "-m", str(gguf), "-f", str(corpus),
           "--ctx-size", str(n_ctx), "-b", "512", "-ub", "512",
           "--no-warmup", "-ngl", str(ngl),
           "--kl-divergence-base", str(base_logits), "--kl-divergence"]
    if chunks:
        cmd += ["--chunks", str(chunks)]
    code = launch(cmd, log_path, timeout_s=14400)
    result = parse_perplexity(log_path)
    result["exit_code"] = code
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-gguf", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--base-logits", required=True)
    parser.add_argument("--imatrix", default=None)
    parser.add_argument("--workdir", default="G:/hq2_research")
    parser.add_argument("--rows", default="all", help="comma-separated row names, or 'all'")
    parser.add_argument("--chunks", type=int, default=0, help="0 = whole corpus")
    parser.add_argument("--n-ctx", type=int, default=512)
    parser.add_argument("--ngl", type=int, default=99)
    parser.add_argument("--hq-backend", default="torch")
    parser.add_argument("--keep-gguf", action="store_true",
                        help="retain each row's GGUF instead of deleting after scoring")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    quant_dir, log_dir = workdir / "quant", workdir / "logs"
    analysis_dir = workdir / "analysis"
    for directory in (quant_dir, log_dir, analysis_dir):
        directory.mkdir(parents=True, exist_ok=True)

    base_gguf = Path(args.base_gguf)
    imatrix = Path(args.imatrix) if args.imatrix else None
    mlp_pattern, mlp_tensors = mlp_pattern_for(base_gguf)
    (analysis_dir / "mlp_tensors_under_test.json").write_text(
        json.dumps({"pattern": mlp_pattern, "count": len(mlp_tensors), "tensors": mlp_tensors}, indent=2),
        encoding="utf-8")
    all_rows = {**{k: ("ggml", v) for k, v in GGML_ROWS.items()},
                **{k: ("hq", v) for k, v in HQ_ROWS.items()}}
    selected = list(all_rows) if args.rows == "all" else [r.strip() for r in args.rows.split(",")]

    for row in selected:
        if row not in all_rows:
            raise SystemExit(f"unknown row {row!r}; known: {', '.join(all_rows)}")

    results_path = analysis_dir / "ladder_results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    for row in selected:
        kind, spec = all_rows[row]
        out_gguf = quant_dir / f"{row}.gguf"
        report = analysis_dir / f"bridge_{row}.json"
        print(f"\n=== row: {row} ({kind}) ===", flush=True)

        with ledger.Stage(f"ROW-{row}", f"MLP-only codec row: {row}",
                          f"Only blk.*.ffn_{{gate,up,down}}.weight differ from the F16 base. "
                          f"Tied embedding and output head pinned to F16. Scored against the "
                          f"shared base logit file on identical targets.") as stage:
            if kind == "ggml":
                code = build_ggml_row(row, spec, base_gguf, out_gguf, imatrix, log_dir, mlp_pattern)
            else:
                code = build_hq_row(row, spec, base_gguf, out_gguf, imatrix, log_dir,
                                    args.hq_backend, report)
            if code != 0 or not out_gguf.exists():
                stage.result(build_exit_code=code, built=False)
                print(f"[ladder] row {row} FAILED to build (exit {code})", flush=True)
                continue

            stage.command(f"build {row} -> {out_gguf.name}")
            stage.output(out_gguf)

            scored = score_row(row, out_gguf, Path(args.corpus), Path(args.base_logits),
                               log_dir, args.chunks, args.n_ctx, args.ngl)
            stage.result(**scored, gguf_bytes=out_gguf.stat().st_size)
            if report.exists():
                stage.result(codec_report=json.loads(report.read_text()).get("mean_relative_mse"))

            results[row] = {**scored, "gguf_bytes": out_gguf.stat().st_size, "kind": kind}
            results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

        if not args.keep_gguf and out_gguf.exists():
            out_gguf.unlink()  # rows are large; the scored numbers are what matter
            print(f"[ladder] removed {out_gguf.name} to reclaim disk", flush=True)

    print(f"\n[ladder] wrote {results_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
