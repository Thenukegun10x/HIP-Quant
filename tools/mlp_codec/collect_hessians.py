"""Collect full input Hessians H = E[x x^T] for every MLP projection.

Why this exists
---------------
The ladder's imatrix is *diagonal*: llama.cpp accumulates mean(x_j^2) per input
channel.  That is enough to weight a per-coordinate MSE, which is all HQ2/HQ2V
used.  It is not enough for the two techniques that the measured 2-bit gap
actually points at:

  * LDLQ / GPTQ-style error feedback needs the LDL factors of the full H, so it
    can push each weight's rounding error onto the *not-yet-quantized* weights
    along correlated input directions.
  * Hadamard incoherence processing rotates the input space, which turns a
    diagonal H into a dense one.  A diagonal imatrix cannot be rotated -- the
    off-diagonal mass was never recorded.  Collecting H is the prerequisite.

Geometry of this model (Qwen3.5-4B-Base, verified on the meta device)
---------------------------------------------------------------------
32 blocks, `model.layers.N.mlp.{gate,up,down}_proj`.  `AutoModelForCausalLM`
resolves to `Qwen3_5ForCausalLM`, which instantiates neither the vision tower
nor the MTP/`nextn` head -- so the tensor set here is exactly the 96 the ladder
quantizes, with no extra exclusion logic needed.

  gate_proj, up_proj : (9216, 2560)  ->  d_in = 2560, and they share an input
  down_proj          : (2560, 9216)  ->  d_in = 9216

gate and up both read the post-attention RMSNorm output, so one 2560x2560
Hessian serves both.  That halves the count from 96 to 64.  It is an assumption
about the architecture, so `--verify-only` tests it rather than trusting it.

Memory discipline
-----------------
This box has hard-frozen twice under memory pressure, once badly enough to need
a volume repair, so the bounds here are deliberate rather than incidental:

  * Weights never enter CPU RAM.  `device_map={"": 0}` streams the safetensors
    shards straight to VRAM.
  * The forward runs `model.model`, not the `ForCausalLM` wrapper, so `lm_head`
    is never applied.  With a 248320-entry vocab those logits would be ~1 GiB
    per batch of 4x512 and are pure waste -- we only want hidden states.
  * All 64 Hessians in fp64 would be 23.4 GiB of host RAM against ~21 GiB
    available.  So blocks are processed in groups (`--blocks-per-pass`), which
    bounds the fp64 master accumulators to group_size * 732 MiB.
  * Nothing is memmapped writable.  A writable whole-file memmap is what caused
    the second freeze; every write here is a bounded `np.save`.
  * Accumulation is fp32 on the GPU, reduced into fp64 on the host every
    `--reduce-every` batches.  Keeping the fp32 partial sums short-lived is
    what stops precision loss over ~65k tokens.

Because a group only needs layers 0..k_last, the forward is cut short right
after the group's last block.  Over 8 groups that averages ~4.5 full passes
instead of 8.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RESEARCH = Path(r"G:\hq2_research")
MODEL_DIR = RESEARCH / "models" / "Qwen3.5-4B-Base"
# The same corpus the ladder's imatrix was built from -- its GGUF records
# `imatrix.datasets = ['G:/hq2_research/corpus/calib.train.txt']` at chunk_size
# 512 over 264 chunks.  LDLQ will be applied alongside that imatrix, so both
# must describe the same activation distribution, and neither may touch
# wiki.test.clean.txt, which is the held-out eval set.
CORPUS = RESEARCH / "corpus" / "calib.train.txt"
OUT_DIR = RESEARCH / "hessians"

# 264 windows of 512 = the whole calibration set, matching the imatrix exactly.
# Against d_in = 9216 that is a 14.7x oversampling, which is comfortable for a
# well-conditioned H.
DEFAULT_TOKENS = 135_168

HIDDEN = 2560
INTERMEDIATE = 9216
N_BLOCKS = 32

# Host RAM floor.  Deliberately higher than safe_run's 6.0 GiB: that floor was
# tuned for llama.cpp processes, which hold their weights in the page cache and
# can be evicted.  The fp64 accumulators here are anonymous committed memory
# and cannot be reclaimed under pressure.
MIN_AVAIL_RAM_GIB = 8.0


# ---------------------------------------------------------------------------
# Host memory, measured correctly
# ---------------------------------------------------------------------------

class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def avail_ram_gib() -> float:
    """Available physical RAM, including reclaimable standby cache.

    Not `FreePhysicalMemory`.  That counter excludes the standby list, so on a
    box with a warm page cache it reads near zero while gigabytes are in fact
    reclaimable -- it produced a false alarm earlier in this study.  `ullAvailPhys`
    is the figure that actually tracks pressure.
    """
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return -1.0
    return status.ullAvailPhys / 1024 ** 3


def require_ram(stage: str) -> float:
    avail = avail_ram_gib()
    if 0 <= avail < MIN_AVAIL_RAM_GIB:
        raise RuntimeError(
            f"{stage}: only {avail:.2f} GiB RAM available (floor {MIN_AVAIL_RAM_GIB}); aborting"
        )
    return avail


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HessianSpec:
    """One Hessian to collect: which block, which input space, which tensors."""
    block: int
    kind: str          # "ffn_gate_up" or "ffn_down"
    dim: int
    modules: tuple[str, ...]   # module suffixes that share this input

    @property
    def key(self) -> str:
        return f"blk{self.block:02d}.{self.kind}"

    def path(self, out_dir: Path) -> Path:
        return out_dir / f"{self.key}.npy"


def build_specs(blocks: range | list[int]) -> list[HessianSpec]:
    specs = []
    for b in blocks:
        specs.append(HessianSpec(b, "ffn_gate_up", HIDDEN, ("gate_proj", "up_proj")))
        specs.append(HessianSpec(b, "ffn_down", INTERMEDIATE, ("down_proj",)))
    return specs


def group_bytes(group_size: int, store_dtype: str) -> tuple[float, float, float]:
    """(host fp64 accumulator GiB, device fp32 accumulator GiB, saved GiB per group)."""
    per_block_elems = HIDDEN ** 2 + INTERMEDIATE ** 2
    host = group_size * per_block_elems * 8 / 1024 ** 3
    dev = group_size * per_block_elems * 4 / 1024 ** 3
    saved = group_size * per_block_elems * np.dtype(store_dtype).itemsize / 1024 ** 3
    return host, dev, saved


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

def build_batches(tokenizer, corpus: Path, n_ctx: int, n_tokens: int, batch: int, seed: int):
    """Evenly-strided, non-overlapping ctx-length windows across the whole corpus.

    Strided rather than head-of-file: the imatrix this must stay consistent with
    was built from all of wiki.train, so sampling only the first few hundred KB
    would shift the activation distribution.
    """
    text = corpus.read_text(encoding="utf-8", errors="replace")
    ids = np.asarray(tokenizer(text, add_special_tokens=False)["input_ids"], dtype=np.int64)
    del text
    gc.collect()

    n_windows_avail = len(ids) // n_ctx
    n_windows_want = max(1, n_tokens // n_ctx)
    if n_windows_want > n_windows_avail:
        raise RuntimeError(
            f"corpus has {n_windows_avail} windows of {n_ctx} tokens, need {n_windows_want}"
        )

    stride = n_windows_avail // n_windows_want
    starts = (np.arange(n_windows_want) * stride) * n_ctx
    windows = np.stack([ids[s:s + n_ctx] for s in starts])

    # Shuffle so any residual positional structure is not aligned with the
    # fp32 -> fp64 reduction boundaries.
    rng = np.random.default_rng(seed)
    rng.shuffle(windows)

    batches = [windows[i:i + batch] for i in range(0, len(windows), batch)]
    return batches, int(windows.size), n_windows_avail


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

class StopForward(Exception):
    """Raised from a hook to cut the forward short once the group is done."""


def load_model(dtype_name: str, verbose: bool = True):
    import torch
    from transformers import AutoModelForCausalLM

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]
    before = avail_ram_gib()
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        dtype=dtype,
        device_map={"": 0},        # straight to VRAM; never materialised on the host
        low_cpu_mem_usage=True,
    )
    model.eval()
    after = require_ram("after model load")
    if verbose:
        free, total = torch.cuda.mem_get_info()
        print(f"[load] {time.time() - t0:.0f}s  host avail {before:.2f} -> {after:.2f} GiB  "
              f"VRAM free {free / 1024 ** 3:.2f}/{total / 1024 ** 3:.2f} GiB", flush=True)
    return model


def collect_group(model, specs: list[HessianSpec], batches, reduce_every: int,
                  out_dir: Path, store_dtype: str, verbose: bool = True) -> dict:
    """Accumulate every Hessian in `specs` in one pass over `batches`."""
    import torch

    base = model.model                      # skips lm_head and its ~1 GiB logits
    layers = base.layers
    blocks = sorted({s.block for s in specs})
    last_block = max(blocks)

    device = next(model.parameters()).device
    dev_acc = {s.key: torch.zeros((s.dim, s.dim), dtype=torch.float32, device=device)
               for s in specs}
    host_acc = {s.key: np.zeros((s.dim, s.dim), dtype=np.float64) for s in specs}
    counts = {s.key: 0 for s in specs}

    # module suffix -> hessian key, for the blocks in this group only
    route: dict[str, str] = {}
    for s in specs:
        for suffix in s.modules:
            route[f"layers.{s.block}.mlp.{suffix}"] = s.key

    def make_hook(key: str):
        def hook(_module, args):
            x = args[0]
            x = x.reshape(-1, x.shape[-1]).to(torch.float32)
            dev_acc[key].addmm_(x.t(), x)
            counts[key] += x.shape[0]
        return hook

    handles = []
    # One accumulator is shared by gate and up, so both get a hook and the
    # token count doubles; it is divided out at save time.
    for name, module in base.named_modules():
        if name in route:
            handles.append(module.register_forward_pre_hook(make_hook(route[name])))

    def stop_hook(*_args, **_kwargs):
        raise StopForward
    handles.append(layers[last_block].register_forward_hook(stop_hook))

    def reduce_to_host():
        for key, acc in dev_acc.items():
            np.add(host_acc[key], acc.cpu().numpy(), out=host_acc[key])
            acc.zero_()

    t0 = time.time()
    peak_dev = 0.0
    min_host = avail_ram_gib()
    try:
        with torch.inference_mode():
            for i, batch in enumerate(batches, start=1):
                ids = torch.from_numpy(np.ascontiguousarray(batch)).to(device)
                try:
                    base(input_ids=ids, use_cache=False)
                except StopForward:
                    pass
                if i % reduce_every == 0:
                    reduce_to_host()
                    min_host = min(min_host, require_ram(f"batch {i}"))
                    peak_dev = max(peak_dev, torch.cuda.max_memory_allocated() / 1024 ** 3)
                if verbose and i % 25 == 0:
                    print(f"  [{i}/{len(batches)}] {time.time() - t0:.0f}s  "
                          f"host avail {avail_ram_gib():.2f} GiB", flush=True)
        reduce_to_host()
    finally:
        for h in handles:
            h.remove()
        for acc in dev_acc.values():
            del acc
        dev_acc.clear()
        gc.collect()
        torch.cuda.empty_cache()

    min_host = min(min_host, require_ram("group end"))
    peak_dev = max(peak_dev, torch.cuda.max_memory_allocated() / 1024 ** 3)

    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for s in specs:
        n = counts[s.key]
        if n == 0:
            raise RuntimeError(f"{s.key}: no activations captured; module routing is wrong")
        h = host_acc[s.key] / n                       # normalised: H = E[x x^T]
        h = 0.5 * (h + h.T)                           # kill fp asymmetry from the reduction
        np.save(s.path(out_dir), h.astype(store_dtype, copy=False))
        written[s.key] = {
            "path": str(s.path(out_dir)),
            "dim": s.dim,
            "rows_accumulated": n,
            "modules": list(s.modules),
            "diag_mean": float(np.mean(np.diag(h))),
            "diag_min": float(np.min(np.diag(h))),
            "diag_max": float(np.max(np.diag(h))),
            "trace": float(np.trace(h)),
        }
        del h
        host_acc[s.key] = None
    host_acc.clear()
    gc.collect()

    return {
        "tensors": written,
        "seconds": time.time() - t0,
        "min_host_avail_gib": min_host,
        "peak_device_gib": peak_dev,
    }


# ---------------------------------------------------------------------------
# Verification: do gate and up really share an input?
# ---------------------------------------------------------------------------

def verify_shared_input(model, batches, verbose: bool = True) -> dict:
    """Test the 96 -> 64 assumption instead of trusting it.

    If gate_proj and up_proj ever see different inputs, one Hessian cannot serve
    both and the plan needs 96 Hessians.
    """
    import torch

    base = model.model
    seen: dict[tuple[int, str], torch.Tensor] = {}

    def make_hook(block: int, suffix: str):
        def hook(_module, args):
            seen[(block, suffix)] = args[0].detach().reshape(-1, args[0].shape[-1])
        return hook

    handles = []
    for b in range(N_BLOCKS):
        for suffix in ("gate_proj", "up_proj", "down_proj"):
            module = base.get_submodule(f"layers.{b}.mlp.{suffix}")
            handles.append(module.register_forward_pre_hook(make_hook(b, suffix)))

    try:
        with torch.inference_mode():
            ids = torch.from_numpy(np.ascontiguousarray(batches[0])).to(
                next(model.parameters()).device)
            base(input_ids=ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    report = {"blocks": {}, "all_identical": True, "dims_ok": True}
    for b in range(N_BLOCKS):
        g, u, d = seen[(b, "gate_proj")], seen[(b, "up_proj")], seen[(b, "down_proj")]
        identical = bool(torch.equal(g, u))
        max_abs = float((g.float() - u.float()).abs().max())
        dims_ok = g.shape[-1] == HIDDEN and d.shape[-1] == INTERMEDIATE
        report["blocks"][b] = {
            "gate_up_bit_identical": identical,
            "gate_up_max_abs_diff": max_abs,
            "gate_in_dim": int(g.shape[-1]),
            "down_in_dim": int(d.shape[-1]),
        }
        report["all_identical"] &= identical
        report["dims_ok"] &= bool(dims_ok)
        if verbose:
            print(f"  blk{b:02d}  gate/up identical={identical}  max|diff|={max_abs:.3e}  "
                  f"gate_in={g.shape[-1]}  down_in={d.shape[-1]}", flush=True)

    seen.clear()
    gc.collect()
    torch.cuda.empty_cache()
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_blocks(text: str) -> list[int]:
    out: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if m := re.fullmatch(r"(\d+)-(\d+)", part):
            lo, hi = int(m.group(1)), int(m.group(2))
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    bad = [b for b in out if not 0 <= b < N_BLOCKS]
    if bad:
        raise ValueError(f"block(s) out of range 0..{N_BLOCKS - 1}: {sorted(bad)}")
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--blocks", default=f"0-{N_BLOCKS - 1}",
                    help="which blocks to collect, e.g. '0-7' or '0,5,31'")
    ap.add_argument("--blocks-per-pass", type=int, default=4,
                    help="bounds host RAM: group_size * 732 MiB of fp64 accumulators")
    ap.add_argument("--tokens", type=int, default=DEFAULT_TOKENS,
                    help="calibration tokens; needs to comfortably exceed d_in=9216")
    ap.add_argument("--n-ctx", type=int, default=512,
                    help="matches the ladder's eval context")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--reduce-every", type=int, default=8,
                    help="fp32 -> fp64 host reduction cadence, in batches")
    ap.add_argument("--store-dtype", default="float32", choices=("float32", "float64"))
    ap.add_argument("--model-dtype", default="bfloat16", choices=("bfloat16", "float16"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and projected memory, load nothing")
    ap.add_argument("--verify-only", action="store_true",
                    help="test the gate/up shared-input assumption and exit")
    ap.add_argument("--skip-ram-check", action="store_true")
    args = ap.parse_args()

    global MIN_AVAIL_RAM_GIB
    if args.skip_ram_check:
        MIN_AVAIL_RAM_GIB = 0.0

    blocks = parse_blocks(args.blocks)
    groups = [blocks[i:i + args.blocks_per_pass]
              for i in range(0, len(blocks), args.blocks_per_pass)]
    host, dev, saved = group_bytes(args.blocks_per_pass, args.store_dtype)
    total_saved = len(blocks) / max(1, args.blocks_per_pass) * saved

    print(f"model    : {MODEL_DIR}")
    print(f"corpus   : {args.corpus}")
    print(f"out      : {args.out}")
    print(f"blocks   : {len(blocks)} -> {len(groups)} pass(es) of <= {args.blocks_per_pass}")
    print(f"tokens   : {args.tokens} at ctx {args.n_ctx}, batch {args.batch}")
    print(f"projected: host fp64 acc {host:.2f} GiB | device fp32 acc {dev:.2f} GiB "
          f"| on disk {total_saved:.2f} GiB ({args.store_dtype})")
    print(f"host RAM : {avail_ram_gib():.2f} GiB available (floor {MIN_AVAIL_RAM_GIB})")

    if args.dry_run:
        for i, g in enumerate(groups, start=1):
            print(f"  pass {i}: blocks {g[0]}..{g[-1]}  "
                  f"({2 * len(g)} hessians, forward cut after layer {max(g)})")
        return 0

    require_ram("startup")

    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    batches, n_tok, n_avail = build_batches(
        tokenizer, args.corpus, args.n_ctx, args.tokens, args.batch, args.seed)
    print(f"corpus   : {n_avail} windows available, using {n_tok} tokens "
          f"in {len(batches)} batch(es)", flush=True)

    model = load_model(args.model_dtype)

    if args.verify_only:
        print("\n[verify] gate/up shared-input assumption, all 32 blocks:", flush=True)
        report = verify_shared_input(model, batches)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "verify_shared_input.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nall blocks gate/up bit-identical: {report['all_identical']}")
        print(f"all input dims as expected      : {report['dims_ok']}")
        return 0 if (report["all_identical"] and report["dims_ok"]) else 1

    manifest = {
        "model": str(MODEL_DIR),
        "corpus": str(args.corpus),
        "n_ctx": args.n_ctx,
        "batch": args.batch,
        "tokens_requested": args.tokens,
        "tokens_used": n_tok,
        "windows_available": n_avail,
        "seed": args.seed,
        "model_dtype": args.model_dtype,
        "store_dtype": args.store_dtype,
        "reduce_every": args.reduce_every,
        "normalised": "H = (1/n) sum_t x_t x_t^T",
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "tensors": {},
        "passes": [],
    }

    overall = time.time()
    for i, group in enumerate(groups, start=1):
        specs = build_specs(group)
        print(f"\n[pass {i}/{len(groups)}] blocks {group[0]}..{group[-1]}  "
              f"{len(specs)} hessians  (forward cut after layer {max(group)})", flush=True)
        stats = collect_group(model, specs, batches, args.reduce_every,
                              args.out, args.store_dtype)
        manifest["tensors"].update(stats.pop("tensors"))
        manifest["passes"].append({"blocks": group, **stats})
        print(f"  done in {stats['seconds']:.0f}s  "
              f"min host avail {stats['min_host_avail_gib']:.2f} GiB  "
              f"peak device {stats['peak_device_gib']:.2f} GiB", flush=True)

    manifest["total_seconds"] = time.time() - overall
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nwrote {len(manifest['tensors'])} hessians in "
          f"{manifest['total_seconds']:.0f}s -> {args.out}")
    print(f"min host avail across all passes: "
          f"{min(p['min_host_avail_gib'] for p in manifest['passes']):.2f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
