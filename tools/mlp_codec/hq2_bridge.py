"""Apply the HQ-family codec to MLP tensors of a GGUF, by streaming rewrite.

llama.cpp cannot execute HQ2/HQ3 blocks, so an HQ row is evaluated the same way
the earlier study did it: the codec is applied, the result is decoded back to
F16, and stock llama.cpp scores the reconstructed weights.  That measures the
codec's reconstruction quality, which is exactly what the ladder compares; it
says nothing about the speed of a packed HQ kernel.

Only the MLP tensor payloads change.  Every other byte -- metadata, the tied
token embedding, attention, norms, alignment padding -- is copied verbatim, so
a row differs from the F16 baseline in precisely the tensors under test.

Memory discipline is the reason this file looks the way it does.  An earlier
revision copied the file with ``shutil.copy2`` and then reopened it as a
read-write ``GGUFReader`` memmap, writing reconstructions back through that
mapping.  A writable mapping over the whole 8.66 GB output means every touched
page becomes dirty, and dirty pages cannot be evicted until they are flushed;
on a 32 GB machine that drove the page cache into the pagefile and hard-froze
the host.  This version therefore:

  * uses the GGUF reader for tensor *metadata only*, then releases it before
    any bulk I/O, so no whole-file mapping is ever live during the rewrite;
  * performs a single sequential source -> output pass with an 8 MiB copy
    buffer, holding at most one tensor payload in memory at a time;
  * fsyncs the output on a byte budget, bounding how much dirty state the OS
    is asked to hold;
  * refuses to start if free RAM is under the same floor ``safe_run`` enforces.

Work is chunked to roughly two million weights per quantizer call.  That bound
is not arbitrary: the repository's own findings record that full-layer HQ2
encoding hit a Windows access violation, and that bounded native chunks both
avoided it and cut mean HQ2 MSE 4.14x (3.9529e-04 -> 9.5355e-05).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "Own Quant" / "llama_cpp_stock" / "gguf-py"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

MLP_SUFFIXES = ("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight")
CHUNK_WEIGHTS = 2_000_000

GGML_TYPE_F16 = 1

COPY_BUF = 8 * 1024**2            # bulk copy granularity
SYNC_EVERY_BYTES = 256 * 1024**2  # flush cadence; caps the dirty-page backlog


def is_mlp(name: str) -> bool:
    return name.startswith("blk.") and name.endswith(MLP_SUFFIXES)


def select_mlp_targets(specs: list["TensorSpec"]) -> list["TensorSpec"]:
    """Pick the MLP tensors under test, in ascending file order.

    This must agree exactly with ``run_ladder.mlp_pattern_for``, or an HQ row
    would cover a different tensor set than the GGML rows it is compared
    against.  Qwen3.5 ships a Multi-Token Prediction head as one extra block
    past the transformer stack, carrying ``nextn.*`` tensors alongside ordinary
    ffn ones.  It is off the forward path, so llama-imatrix collects no
    statistics for it and the GGML rows leave it at F16; encoding it here would
    both break comparability and quietly produce uncalibrated tensors in an
    otherwise calibrated row.
    """
    excluded = {m.group(1) for s in specs if (m := re.match(r"blk\.(\d+)\.nextn\.", s.name))}
    targets = [
        s for s in specs
        if (m := re.match(r"blk\.(\d+)\.ffn_(gate|up|down)\.weight$", s.name))
        and m.group(1) not in excluded
    ]
    if excluded:
        print(f"[bridge] excluding non-forward-path blocks {sorted(excluded)} (MTP/nextn head)",
              flush=True)
    return sorted(targets, key=lambda s: s.offset)


# --------------------------------------------------------------------------
# metadata
# --------------------------------------------------------------------------

@dataclass
class TensorSpec:
    name: str
    ggml_type: int
    offset: int            # absolute byte offset in the file
    n_bytes: int
    np_shape: tuple[int, ...]   # numpy order, i.e. reversed GGUF dims


def read_manifest(path: Path) -> list[TensorSpec]:
    """Harvest tensor offsets and shapes, then drop the reader's mapping.

    ``GGUFReader`` builds numpy *views* over a memmap rather than reading tensor
    bytes, so constructing it only faults in header and metadata pages.  Even
    so, the reader is deleted before any bulk I/O begins: nothing in this
    process should hold a mapping over a multi-gigabyte file while it is being
    rewritten.
    """
    from gguf import GGUFReader

    reader = GGUFReader(str(path), "r")
    specs = [
        TensorSpec(
            name=str(t.name),
            ggml_type=int(t.tensor_type),
            offset=int(t.data_offset),
            n_bytes=int(t.n_bytes),
            # gguf-py exposes tensor.data reshaped to reversed(dims), so a
            # [n_in, n_out] GGUF tensor presents to numpy as [out, in].
            np_shape=tuple(int(d) for d in reversed(t.shape.tolist())),
        )
        for t in reader.tensors
    ]
    del reader
    gc.collect()
    return specs


# --------------------------------------------------------------------------
# bounded sequential writer
# --------------------------------------------------------------------------

class BoundedWriter:
    """Sequential writer that fsyncs on a byte budget.

    Writing ~8.7 GB without periodic flushes lets the OS accumulate that much
    unevictable dirty state.  Flushing every ``sync_every`` bytes keeps the
    backlog small at negligible cost, since the access pattern is sequential.
    """

    def __init__(self, handle, sync_every: int = SYNC_EVERY_BYTES):
        self.handle = handle
        self.sync_every = sync_every
        self.total = 0
        self._since_sync = 0

    def _account(self, n: int) -> None:
        self.total += n
        self._since_sync += n
        if self._since_sync >= self.sync_every:
            self.sync()

    def sync(self) -> None:
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self._since_sync = 0

    def write(self, data: bytes) -> None:
        self.handle.write(data)
        self._account(len(data))

    def copy_from(self, src, n_bytes: int) -> None:
        """Copy exactly ``n_bytes`` from ``src`` at its current position."""
        if n_bytes < 0:
            raise ValueError(f"negative copy length {n_bytes}: tensor offsets are out of order")
        remaining = n_bytes
        while remaining:
            chunk = src.read(min(COPY_BUF, remaining))
            if not chunk:
                raise EOFError(f"source ended with {remaining} bytes still to copy")
            self.handle.write(chunk)
            self._account(len(chunk))
            remaining -= len(chunk)

    def copy_rest(self, src) -> None:
        while True:
            chunk = src.read(COPY_BUF)
            if not chunk:
                break
            self.handle.write(chunk)
            self._account(len(chunk))


# --------------------------------------------------------------------------
# imatrix
# --------------------------------------------------------------------------

def load_imatrix(path: str | Path) -> dict[str, np.ndarray]:
    """Read a llama.cpp GGUF imatrix into {tensor_name: mean squared activation}.

    llama.cpp stores, per tensor, the summed squared input activations
    (`<name>.in_sum2`) and the number of contributing chunks (`<name>.counts`).
    The per-input-channel mean is the ratio, which is the same quantity
    llama-quantize itself consumes.
    """
    from gguf import GGUFReader

    reader = GGUFReader(str(path), "r")
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    for tensor in reader.tensors:
        name = str(tensor.name)
        if name.endswith(".in_sum2"):
            sums[name[: -len(".in_sum2")]] = np.array(tensor.data, dtype=np.float32)
        elif name.endswith(".counts"):
            counts[name[: -len(".counts")]] = np.array(tensor.data, dtype=np.float32)

    importance: dict[str, np.ndarray] = {}
    for name, total in sums.items():
        divisor = counts.get(name)
        n = float(divisor.sum()) if divisor is not None and divisor.size else 1.0
        importance[name] = (total.astype(np.float64) / max(n, 1.0)).astype(np.float32).ravel()
    del reader
    gc.collect()
    return importance


# --------------------------------------------------------------------------
# codec
# --------------------------------------------------------------------------

def roundtrip_tensor(values: np.ndarray, importance_row: np.ndarray | None,
                     fmt: str, iterations: int, backend: str,
                     device: str | None = None) -> tuple[np.ndarray, float]:
    """HQ-encode and decode one [out, in] matrix, returning the reconstruction and its MSE."""
    import hq2

    original_dtype = values.dtype
    work = np.ascontiguousarray(values, dtype=np.float32)
    n_out, n_in = work.shape

    weights = None
    if importance_row is not None:
        if importance_row.size != n_in:
            raise ValueError(f"imatrix length {importance_row.size} != input dim {n_in}")
        # llama.cpp importance is per input channel; broadcasting it across output
        # rows is the same treatment llama-quantize applies.
        weights = np.broadcast_to(importance_row.astype(np.float32), (n_out, n_in))

    rows_per_chunk = max(1, CHUNK_WEIGHTS // n_in)
    output = np.empty_like(work)
    squared_error = 0.0

    for start in range(0, n_out, rows_per_chunk):
        stop = min(start + rows_per_chunk, n_out)
        block = np.ascontiguousarray(work[start:stop])
        block_importance = None
        if weights is not None:
            block_importance = np.ascontiguousarray(weights[start:stop])

        if backend == "torch":
            import torch
            tensor = torch.from_numpy(block).to(device)
            imp = None if block_importance is None else torch.from_numpy(block_importance).to(device)
            packed = hq2.quantize(tensor, importance=imp, backend="torch",
                                  iterations=iterations, format=fmt)
            restored = packed.dequantize().detach().to("cpu").numpy()
            del tensor, imp, packed
            torch.cuda.empty_cache()
        else:
            packed = hq2.quantize(block, importance=block_importance, backend=backend,
                                  iterations=iterations, format=fmt)
            restored = np.asarray(packed.dequantize(), dtype=np.float32)

        output[start:stop] = restored
        squared_error += float(np.sum((restored - block).astype(np.float64) ** 2))
        del block, block_importance, restored

    return output.astype(original_dtype), squared_error / work.size


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

@dataclass
class Encoder:
    fmt: str
    iterations: int
    backend: str
    importance: dict[str, np.ndarray]
    device: str | None = None
    scale_rounds: int = 1
    damping: float = 1e-2
    hessian_dir: Path = Path(r"G:\hq2_research\hessians")
    per_tensor: dict[str, dict] = field(default_factory=dict)
    matched: int = 0

    def __call__(self, spec: TensorSpec, values: np.ndarray) -> np.ndarray:
        row = self.importance.get(spec.name)
        if row is not None:
            self.matched += 1
        if self.fmt == "hq2vl":
            # Same format, same rate, same codebook as hq2v -- the only addition
            # is Hessian error feedback, which costs no bits. It needs the tensor
            # name to find its Hessian, which is why this is dispatched here
            # rather than inside the codec.
            import ldlq
            restored, mse = ldlq.roundtrip(values, row, spec.name,
                                           iterations=self.iterations,
                                           scale_rounds=self.scale_rounds,
                                           damping=self.damping,
                                           hessian_dir=self.hessian_dir)
        elif self.fmt in ("hq2v", "hq2vs"):
            # HQ2V fits one codebook per tensor, so it handles its own chunking
            # rather than going through the per-chunk HQ kernel path.
            import hq2v
            restored, mse = hq2v.roundtrip(values, row, iterations=self.iterations,
                                           scale_rounds=self.scale_rounds,
                                           symmetric=self.fmt == "hq2vs")
        else:
            restored, mse = roundtrip_tensor(values, row, self.fmt, self.iterations,
                                             self.backend, self.device)
        variance = float(np.var(values.astype(np.float64)))
        self.per_tensor[spec.name] = {
            "shape": list(spec.np_shape),
            "mse": mse,
            "relative_mse": mse / variance if variance > 0 else None,
            "calibrated": row is not None,
        }
        return restored


def _read_payload(handle, spec: TensorSpec) -> np.ndarray:
    raw = handle.read(spec.n_bytes)
    if len(raw) != spec.n_bytes:
        raise EOFError(f"{spec.name}: wanted {spec.n_bytes} bytes, got {len(raw)}")
    return np.frombuffer(raw, dtype=np.float16).reshape(spec.np_shape)


def dry_run(source: Path, targets: list[TensorSpec], encode: Encoder) -> None:
    """Encode the target tensors and report error, writing nothing.

    This exercises manifest parsing, offset arithmetic, dtype and shape
    handling, and the GPU codec path at a few tens of megabytes of resident
    memory and zero output I/O -- the cheapest way to prove the pipeline before
    committing to a multi-gigabyte rewrite.
    """
    started = time.time()
    with source.open("rb") as fin:
        for index, spec in enumerate(targets, 1):
            fin.seek(spec.offset)
            values = _read_payload(fin, spec)
            encode(spec, values)
            stats = encode.per_tensor[spec.name]
            print(f"[bridge]   {index}/{len(targets)}  {spec.name}  {tuple(spec.np_shape)}  "
                  f"mse={stats['mse']:.6e}  rel={stats['relative_mse']:.4%}  "
                  f"imat={stats['calibrated']}  ({time.time() - started:.0f}s)", flush=True)


def stream_build(source: Path, output: Path, targets: list[TensorSpec],
                 encode: Encoder) -> int:
    """Copy source -> output in one sequential pass, replacing target payloads."""
    started = time.time()
    output.parent.mkdir(parents=True, exist_ok=True)

    with source.open("rb") as fin, output.open("wb") as raw_out:
        out = BoundedWriter(raw_out)
        position = 0
        for index, spec in enumerate(targets, 1):
            out.copy_from(fin, spec.offset - position)
            position = spec.offset

            values = _read_payload(fin, spec)
            restored = encode(spec, values)
            payload = np.ascontiguousarray(restored, dtype=np.float16).tobytes()
            if len(payload) != spec.n_bytes:
                raise RuntimeError(
                    f"{spec.name}: reconstruction is {len(payload)} bytes but the "
                    f"slot is {spec.n_bytes}; refusing to shift the file")
            out.write(payload)
            position += spec.n_bytes

            encode.per_tensor[spec.name]["md5"] = hashlib.md5(payload).hexdigest()
            stats = encode.per_tensor[spec.name]
            del values, restored, payload

            if index % 12 == 0 or index == len(targets):
                print(f"[bridge]   {index}/{len(targets)}  {spec.name}  "
                      f"mse={stats['mse']:.6e}  ({time.time() - started:.0f}s)", flush=True)

        out.copy_rest(fin)
        out.sync()

    written = output.stat().st_size
    expected = source.stat().st_size
    if written != expected:
        raise RuntimeError(f"output is {written} bytes but source is {expected}; offsets are wrong")
    print(f"[bridge] wrote {written} bytes in {time.time() - started:.0f}s (size matches source)", flush=True)
    return written


def verify_output(output: Path, targets: list[TensorSpec], per_tensor: dict[str, dict]) -> None:
    """Re-read each rewritten payload and confirm it landed where intended.

    An offset error is the one mistake here that produces a plausible-looking
    file, so the written bytes are hashed on the way out and checked on the way
    back in.
    """
    with output.open("rb") as fin:
        for spec in targets:
            fin.seek(spec.offset)
            raw = fin.read(spec.n_bytes)
            digest = hashlib.md5(raw).hexdigest()
            if digest != per_tensor[spec.name]["md5"]:
                raise RuntimeError(f"{spec.name}: payload at offset {spec.offset} does not "
                                   f"match what was written")
    print(f"[bridge] verified {len(targets)} payloads at their recorded offsets", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="F16 GGUF to read from")
    parser.add_argument("--output", default=None, help="row GGUF to create (omit with --dry-run)")
    parser.add_argument("--format", default="hq2",
                        choices=["hq2", "hq3", "hq2v", "hq2vs", "hq2vl"])
    parser.add_argument("--damping", type=float, default=1e-2,
                        help="hq2vl only: diagonal damping as a fraction of mean(diag H)")
    parser.add_argument("--hessian-dir", type=Path, default=Path(r"G:\hq2_research\hessians"),
                        help="hq2vl only: directory of collected Hessians")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--scale-rounds", type=int, default=1,
                        help="HQ2V only: alternate assignment with a weighted "
                             "least-squares scale this many times (1 = plain RMS scale)")
    parser.add_argument("--backend", default="torch", choices=["torch", "rocm", "cpu"])
    parser.add_argument("--imatrix", default=None, help="GGUF imatrix; omit for the uncalibrated row")
    parser.add_argument("--report", default=None, help="write per-tensor MSE JSON here")
    parser.add_argument("--limit", type=int, default=0, help="only process the first N MLP tensors")
    parser.add_argument("--dry-run", action="store_true",
                        help="encode and report error without writing an output file")
    parser.add_argument("--no-verify", action="store_true", help="skip the post-write payload check")
    parser.add_argument("--skip-ram-check", action="store_true")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise SystemExit(f"missing source {source}")
    if not args.dry_run and not args.output:
        raise SystemExit("--output is required unless --dry-run is given")

    # Defence in depth: the guard belongs in the component that can hurt the
    # host, not only in the wrapper that is supposed to launch it.
    if not args.skip_ram_check:
        from safe_run import MIN_FREE_RAM_GIB, free_ram_gib
        ram = free_ram_gib()
        print(f"[bridge] free RAM: {ram:.1f} GiB (floor {MIN_FREE_RAM_GIB})", flush=True)
        if 0 <= ram < MIN_FREE_RAM_GIB:
            raise SystemExit(f"only {ram:.1f} GiB RAM free; refusing to start")

    importance = load_imatrix(args.imatrix) if args.imatrix else {}
    if args.imatrix:
        print(f"[bridge] imatrix entries: {len(importance)}", flush=True)

    specs = read_manifest(source)
    targets = select_mlp_targets(specs)
    if args.limit:
        targets = targets[: args.limit]

    bad = [s.name for s in targets if s.ggml_type != GGML_TYPE_F16]
    if bad:
        raise SystemExit(f"expected F16 MLP tensors, got other types: {bad[:4]}")
    for a, b in zip(targets, targets[1:]):
        if a.offset + a.n_bytes > b.offset:
            raise SystemExit(f"tensors {a.name} and {b.name} overlap; manifest is unusable")

    total_mb = sum(s.n_bytes for s in targets) / 1024**2
    print(f"[bridge] {len(targets)} MLP tensors ({total_mb:.0f} MiB) to encode as "
          f"{args.format.upper()}", flush=True)

    device = None
    if args.backend == "torch" and args.format not in ("hq2v", "hq2vs"):
        # HQ2V is pure NumPy; creating a HIP context for it would cost VRAM and
        # buy nothing.
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[bridge] torch device: {device}", flush=True)
    if args.format in ("hq2v", "hq2vs"):
        import hq2v
        symmetric = args.format == "hq2vs"
        example = next((s for s in targets), None)
        if example is not None:
            n = int(np.prod(example.np_shape))
            entries = hq2v.MAGNITUDE_CODEBOOK if symmetric else hq2v.CODEBOOK
            print(f"[bridge] {args.format.upper()} rate on {example.name}: "
                  f"{hq2v.bits_per_weight(n, symmetric):.5f} bpw "
                  f"(dim={hq2v.DIM}, codebook={entries}, "
                  f"signs={'explicit' if symmetric else 'in-codebook'})", flush=True)

    encode = Encoder(fmt=args.format, iterations=args.iterations, backend=args.backend,
                     importance=importance, device=device, scale_rounds=args.scale_rounds,
                     damping=args.damping, hessian_dir=args.hessian_dir)

    started = time.time()
    if args.dry_run:
        print("[bridge] DRY RUN: no output file will be written", flush=True)
        dry_run(source, targets, encode)
        written = 0
    else:
        written = stream_build(source, Path(args.output), targets, encode)
        if not args.no_verify:
            verify_output(Path(args.output), targets, encode.per_tensor)

    per_tensor = encode.per_tensor
    relative = [v["relative_mse"] for v in per_tensor.values() if v["relative_mse"] is not None]
    summary = {
        "format": args.format,
        "iterations": args.iterations,
        "backend": args.backend,
        "device": device,
        "imatrix": args.imatrix,
        "dry_run": bool(args.dry_run),
        "source": str(source),
        "output": None if args.dry_run else str(args.output),
        "bytes_written": written,
        "tensors_encoded": len(per_tensor),
        "tensors_with_imatrix": encode.matched,
        "mean_mse": float(np.mean([v["mse"] for v in per_tensor.values()])) if per_tensor else None,
        "mean_relative_mse": float(np.mean(relative)) if relative else None,
        "elapsed_s": round(time.time() - started, 1),
        "per_tensor": per_tensor,
    }
    print(f"[bridge] done: {summary['tensors_encoded']} tensors, {encode.matched} calibrated, "
          f"mean MSE {summary['mean_mse']:.6e}, mean rel {summary['mean_relative_mse']:.4%}, "
          f"{summary['elapsed_s']}s", flush=True)

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[bridge] wrote {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
