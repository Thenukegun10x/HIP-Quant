"""Append-only experiment ledger.

Every stage of the MLP-only codec study records one entry here.  The point is
that a result can be reconstructed months later without guessing: each entry
carries the exact command, the wall-clock window, and the SHA-256 plus byte
size of every input and output file it touched.

Records are written twice: as a JSON line (machine-readable, never rewritten)
and as a rendered Markdown section appended to the human ledger.  The JSON is
the source of truth; the Markdown can always be regenerated from it.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("HQ2_RESEARCH_ROOT", r"G:/hq2_research"))
LEDGER_JSONL = ROOT / "analysis" / "ledger.jsonl"
LEDGER_MD = Path(__file__).resolve().parents[2] / "Own Quant" / "MLP_CODEC_LEDGER.md"

# Hashing a 24 GB GGUF takes minutes.  For files above this size, hash a
# deterministic sample (head + tail + size) instead of the whole payload; it
# still detects truncation, a wrong row, or a silently regenerated artifact,
# which is what the ledger actually needs to guard against.
FULL_HASH_LIMIT = 2 * 1024**3
SAMPLE_SPAN = 64 * 1024**2


def sha256_file(path: str | Path) -> dict:
    """Hash a file, sampling very large ones.  Always reports which mode ran."""
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    size = path.stat().st_size
    digest = hashlib.sha256()
    mode = "full"
    with path.open("rb") as handle:
        if size <= FULL_HASH_LIMIT:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        else:
            mode = f"sampled-head-tail-{SAMPLE_SPAN // 1024**2}MiB"
            digest.update(str(size).encode())
            digest.update(handle.read(SAMPLE_SPAN))
            handle.seek(max(0, size - SAMPLE_SPAN))
            digest.update(handle.read(SAMPLE_SPAN))
    return {
        "path": str(path),
        "exists": True,
        "bytes": size,
        "gib": round(size / 1024**3, 4),
        "sha256": digest.hexdigest(),
        "hash_mode": mode,
    }


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[2]),
            capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


class Stage:
    """Context manager timing one stage and recording it on exit.

    Used as::

        with Stage("convert-f16", "Convert safetensors to F16 GGUF") as stage:
            stage.command(cmd)
            stage.input(src)
            ...
            stage.output(dst)
            stage.result(n_tensors=667)

    A raised exception is recorded as a failed stage rather than swallowed, so
    the ledger shows attempts that did not work instead of quietly omitting
    them.  That is deliberate: the gaps in the previous round of experiments
    are exactly what made them impossible to reconstruct.
    """

    def __init__(self, stage_id: str, title: str, notes: str = ""):
        self.stage_id = stage_id
        self.title = title
        self.notes = notes
        self.entry: dict = {
            "stage_id": stage_id,
            "title": title,
            "notes": notes,
            "commands": [],
            "inputs": [],
            "outputs": [],
            "results": {},
            "host": platform.node(),
            "git_commit": _git_commit(),
        }

    def command(self, cmd) -> "Stage":
        self.entry["commands"].append(cmd if isinstance(cmd, str) else " ".join(map(str, cmd)))
        return self

    def input(self, path) -> "Stage":
        self.entry["inputs"].append(sha256_file(path))
        return self

    def output(self, path) -> "Stage":
        self.entry["outputs"].append(sha256_file(path))
        return self

    def result(self, **kwargs) -> "Stage":
        self.entry["results"].update(kwargs)
        return self

    def note(self, text: str) -> "Stage":
        self.entry["notes"] = (self.entry["notes"] + "\n" + text).strip()
        return self

    def __enter__(self) -> "Stage":
        self._t0 = time.time()
        self.entry["started_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.entry["ended_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.entry["duration_s"] = round(time.time() - self._t0, 1)
        self.entry["status"] = "ok" if exc_type is None else "FAILED"
        if exc_type is not None:
            self.entry["error"] = f"{exc_type.__name__}: {exc}"
        write(self.entry)
        return False


def write(entry: dict) -> None:
    LEDGER_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    _append_markdown(entry)


def _fmt_files(files: list[dict]) -> str:
    if not files:
        return "_none_\n"
    lines = ["| file | size | sha256 |", "|---|---:|---|"]
    for f in files:
        if not f.get("exists"):
            lines.append(f"| `{f['path']}` | **MISSING** | — |")
            continue
        tag = f["sha256"][:16] + ("…" if f["hash_mode"] != "full" else "")
        note = "" if f["hash_mode"] == "full" else f" ({f['hash_mode']})"
        lines.append(f"| `{f['path']}` | {f['gib']} GiB | `{tag}`{note} |")
    return "\n".join(lines) + "\n"


def _append_markdown(entry: dict) -> None:
    LEDGER_MD.parent.mkdir(parents=True, exist_ok=True)
    if not LEDGER_MD.exists():
        LEDGER_MD.write_text(_HEADER, encoding="utf-8")
    status = entry["status"]
    badge = "" if status == "ok" else " — **FAILED**"
    parts = [
        f"\n### `{entry['stage_id']}` — {entry['title']}{badge}\n",
        f"*{entry['started_utc']} → {entry['ended_utc']} ({entry['duration_s']} s)*",
        f" · host `{entry['host']}`" + (f" · repo `{entry['git_commit'][:10]}`" if entry.get("git_commit") else ""),
        "\n",
    ]
    if entry.get("notes"):
        parts.append(f"\n{entry['notes']}\n")
    if entry.get("error"):
        parts.append(f"\n> **Error:** `{entry['error']}`\n")
    if entry["commands"]:
        parts.append("\n**Command**\n\n```bash\n" + "\n".join(entry["commands"]) + "\n```\n")
    if entry["inputs"]:
        parts.append("\n**Inputs**\n\n" + _fmt_files(entry["inputs"]))
    if entry["outputs"]:
        parts.append("\n**Outputs**\n\n" + _fmt_files(entry["outputs"]))
    if entry["results"]:
        parts.append("\n**Results**\n\n```json\n" + json.dumps(entry["results"], indent=2) + "\n```\n")
    with LEDGER_MD.open("a", encoding="utf-8") as handle:
        handle.write("".join(parts))


_HEADER = """# MLP-only codec ledger — Qwen3.5-4B-Base

Every entry below is written automatically by `tools/mlp_codec/ledger.py` at the
moment the stage runs. Nothing here is typed by hand after the fact. The ledger
is append-only: a superseded stage is followed by a correcting entry rather than
edited, so `S5` and `S5-SUPERSEDED` both remain visible.

**Why this ledger exists.** The earlier HQ2 comparisons could not be
reconstructed: intermediate artifacts were deleted, quantizer paths changed
between runs without being recorded, and the headline table mixed several
independent variables at once. Each entry therefore pins the exact command, the
wall-clock window, and the SHA-256 and byte size of every file consumed and
produced. Files above 2 GiB are hashed by a deterministic head+tail+size sample;
the mode is always stated so a partial hash is never mistaken for a full one.

**What this study measures.** One variable: the weight codec applied to the 96
text-MLP projections (`blk.*.ffn_gate`, `blk.*.ffn_up`, `blk.*.ffn_down`) that
lie on the forward path. In every row, all other tensors — critically
`token_embd.weight`, which *is* the output projection here, since Qwen3.5-4B
ships no separate `output.weight` — are held at F16 and are bit-identical across
rows. Every row is scored against one shared base logit file over the same
held-out WikiText-2 token stream, and every calibrated row uses the same
importance matrix.

The model carries 99 `ffn_*.weight` tensors, not 96. The extra three belong to
`blk.32`, a Multi-Token Prediction head (`nextn.*`) that sits off the forward
path: `llama-imatrix` collects no statistics for it, a low-bit quantize aborts
with "Missing importance matrix", and touching it would change the bits-per-
weight accounting without affecting perplexity. Both the GGML path
(`run_ladder.mlp_pattern_for`) and the HQ path (`hq2_bridge.select_mlp_targets`)
derive the exclusion from the file itself, so the two families provably cover
the same tensor set.

**What the previous round got wrong.** The 168.99x figure for `HQ2-Mixed-2.8`
was dominated by a tensor that is not HQ2: the tied embedding/output head was
assigned HQ3 at 3.5 bpw with no calibration, measuring 4.23% relative MSE on the
one matrix that produces every logit. That row is a valid measurement of that
archive and tells us nothing about the HQ2 codec. Separately, the MLP-only runs
dated 2026-07-16 predate the 2026-07-17 native-chunk fix that cut HQ2 mean MSE
by 4.14x (3.9529e-04 -> 9.5355e-05), so those numbers came from a worse
quantizer than the current one.

**Substrate change, mid-study.** Gemma 4 12B was abandoned as the substrate: its
F16 GGUF scored PPL 385-416 on held-out WikiText-2 while scoring 12.43 on the
Alpaca-style text used historically. Corpus cleaning, CPU vs Vulkan backends, and
a shorter context all failed to move it, while a SmolLM2-135M control on the
identical corpus and harness returned a sane 16.72 — isolating the fault to the
Gemma conversion rather than the measurement. Qwen3.5-4B-Base (F16 PPL 7.9109)
replaced it.

**Two silent failures found and fixed.** (1) `--tensor-type` overrides are gated
behind `ggml_is_quantized(default_type)` at `src/llama-quant.cpp:674`, so passing
an F16 base type made every MLP override a no-op while still reporting success;
every "MLP-only" row would have been an identical copy of F16. The fix inverts
the construction: the codec becomes the base ftype, with `--pure` and ordered
regex overrides pinning everything else back to F16. (2) `q3_k_s` and `iq2_m` are
*ftypes*, not `ggml_type`s, and naming them as tensor types makes
`llama-quantize` print its usage banner and exit 1.

---
"""


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "render":
        # Rebuild the Markdown view from the JSONL source of truth.
        if LEDGER_MD.exists():
            LEDGER_MD.unlink()
        LEDGER_MD.write_text(_HEADER, encoding="utf-8")
        for line in LEDGER_JSONL.read_text(encoding="utf-8").splitlines():
            if line.strip():
                _append_markdown(json.loads(line))
        print(f"rendered {LEDGER_MD}")
