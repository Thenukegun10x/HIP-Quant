"""Paired, cache-fair IQ2_S GEMV variant benchmark.

This is deliberately a tuning tool rather than a unit test.  It alternates
the launch order of production variant 0 and one candidate while using two
different, equally shaped GGUF weights, which makes both packed matrices cold
to L2 and makes clock drift affect each variant symmetrically.
"""

from __future__ import annotations

import argparse
import statistics

import torch

from hip_quant import gguf
from hip_quant import torch_api as T


IQ2_S = 22
BYTES_PER_BLOCK = 82
BLOCK_SIZE = 256


def _time_ms(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="GGUF containing at least two same-shape IQ2_S tensors")
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--matrices", type=int, default=4)
    parser.add_argument("--variant", type=int, default=2,
                        help="benchmark-only candidate: 1 = old 4 rows; 2 = scalar 8 rows; 3 = forced 32K LDS; 4 = f16 dot2")
    parser.add_argument("--baseline-variant", type=int, default=0,
                        help="variant used as the paired control (default: production variant 0)")
    parser.add_argument("--shape", metavar="N,K",
                        help="select a particular IQ2_S matrix shape, e.g. 5120,17408")
    args = parser.parse_args()
    if args.reps < 6:
        parser.error("--reps must be at least 6")
    if not torch.cuda.is_available():
        raise RuntimeError("a ROCm/CUDA GPU is required")

    ext = T._load_extension()
    if not hasattr(ext, "gemv_q_forward_variant"):
        raise RuntimeError("extension lacks gemv_q_forward_variant; rebuild it first")

    gf = gguf.load(args.model)
    with gf.open():
        groups: dict[tuple[int, int], list[gguf.GGUFTensor]] = {}
        for tensor in gf.tensors:
            if tensor.ggml_type != IQ2_S or len(tensor.shape) != 2:
                continue
            n, k = tensor.shape
            if k % BLOCK_SIZE or tensor.n_bytes != n * (k // BLOCK_SIZE) * BYTES_PER_BLOCK:
                continue
            groups.setdefault((n, k), []).append(tensor)
        if args.shape:
            try:
                wanted = tuple(int(v) for v in args.shape.split(","))
            except ValueError as exc:
                raise SystemExit("--shape must be N,K") from exc
            if len(wanted) != 2:
                raise SystemExit("--shape must be N,K")
            shape, tensors = wanted, groups.get(wanted, [])
        else:
            shape, tensors = max(groups.items(), key=lambda pair: len(pair[1]), default=(None, []))
        if shape is None or len(tensors) < 2:
            raise RuntimeError("need at least two same-shape IQ2_S GGUF tensors")
        n, k = shape
        tensors = tensors[: max(2, args.matrices)]
        weights = [torch.frombuffer(gf.raw_bytes(t), dtype=torch.uint8).to("cuda") for t in tensors]

    torch.manual_seed(0)
    x = torch.randn((1, k), dtype=torch.float16, device="cuda")
    baseline = ext.gemv_q_forward_variant(x, weights[0], IQ2_S, n, args.baseline_variant, None)
    candidate = ext.gemv_q_forward_variant(x, weights[0], IQ2_S, n, args.variant, None)
    max_diff = (baseline.float() - candidate.float()).abs().max().item()
    if max_diff > 1e-3:
        raise RuntimeError(f"candidate differs from baseline: max_abs={max_diff}")

    for weight in weights:
        ext.gemv_q_forward_variant(x, weight, IQ2_S, n, args.baseline_variant, None)
        ext.gemv_q_forward_variant(x, weight, IQ2_S, n, 1, None)
    torch.cuda.synchronize()

    base_times: list[float] = []
    alt_times: list[float] = []
    for rep in range(args.reps):
        base_weight = weights[rep % len(weights)]
        alt_weight = weights[(rep + 1) % len(weights)]
        jobs = [(args.baseline_variant, base_weight), (args.variant, alt_weight)]
        if rep & 1:
            jobs.reverse()
        for variant, weight in jobs:
            elapsed = _time_ms(lambda v=variant, w=weight: ext.gemv_q_forward_variant(x, w, IQ2_S, n, v, None))
            (base_times if variant == args.baseline_variant else alt_times).append(elapsed)

    # Discard the first two rounds while the display GPU settles its clocks.
    base_times = base_times[2:]
    alt_times = alt_times[2:]
    base_med = statistics.median(base_times)
    alt_med = statistics.median(alt_times)
    paired = statistics.median((alt - base) / base * 100.0 for base, alt in zip(base_times, alt_times))
    packed_mb = weights[0].numel() / 1e6
    print(f"shape={n}x{k}; matrices={len(weights)}; packed={packed_mb:.3f} MB")
    print(f"correctness: max_abs={max_diff:.6g}")
    print(f"variant{args.baseline_variant}:  {base_med:.4f} ms  {packed_mb / base_med:.1f} GB/s")
    print(f"variant{args.variant}:  {alt_med:.4f} ms  {packed_mb / alt_med:.1f} GB/s")
    print(f"paired delta (variant{args.variant} vs variant{args.baseline_variant}): {paired:+.2f}%")


if __name__ == "__main__":
    main()
