"""Measure real packed-HQ2 inference on representative Gemma 4 MLP layers.

The script deliberately reports two separate scopes:

* packed ``HQ2Linear`` versus BF16 ``torch.nn.functional.linear`` for the
  exact MLP shapes stored in an HQ archive; and
* optional end-to-end decode/prefill timing for the executable standalone
  Gemma package (HQ2 MLPs plus BF16 RAW tensors).

It does *not* call the decoded-F16 GGUF bridge.  The two routes are therefore
real ROCm PyTorch measurements, but the full mixed 2.8-BPW archive cannot be
timed end-to-end until Q4_0/Q8_0 packed dispatch is implemented.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hq2


DEFAULT_ARCHIVE = ROOT / "Own Quant" / "Gemma4-12B-HQ2-Mixed-2p8.hq"
DEFAULT_PACKAGE = ROOT / "Own Quant" / "Gemma4-12B-HQ2-Standalone"
DEFAULT_OUTPUT = ROOT / "Own Quant" / "experimental" / "inference_speed" / "hq2_gemma4_rocm.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE, help="HQ archive used for representative packed MLP weights")
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE, help="standalone runnable HQ2 Gemma package")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="write JSON evidence here")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--kernel-batches", type=int, nargs="+", default=(1, 8, 32), help="token rows (M) per linear call")
    parser.add_argument("--kernel-warmup", type=int, default=8)
    parser.add_argument("--kernel-iterations", type=int, default=24)
    parser.add_argument("--prefill-tokens", type=int, nargs="+", default=(32, 128))
    parser.add_argument("--prefill-warmup", type=int, default=1)
    parser.add_argument("--prefill-iterations", type=int, default=3)
    parser.add_argument("--decode-warmup", type=int, default=2)
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--skip-model", action="store_true", help="run only representative packed-layer benchmarks")
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bfloat16" else torch.float16


def _percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    if not ordered:
        raise ValueError("no timings")
    return ordered[round((len(ordered) - 1) * fraction)]


def _timing_summary(samples_ms: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(samples_ms),
        "mean_ms": statistics.fmean(samples_ms),
        "p10_ms": _percentile(samples_ms, 0.10),
        "p90_ms": _percentile(samples_ms, 0.90),
        "samples": len(samples_ms),
    }


def _event_time(call: Callable[[], Any], *, warmup: int, iterations: int) -> tuple[dict[str, float], Any]:
    """Time GPU work with ROCm events, retaining the final output."""

    with torch.inference_mode():
        result = None
        for _ in range(warmup):
            result = call()
        torch.cuda.synchronize()
        samples_ms: list[float] = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = call()
            end.record()
            end.synchronize()
            samples_ms.append(float(start.elapsed_time(end)))
    return _timing_summary(samples_ms), result


def _close_packed_mapping(value: Any) -> None:
    mapping = getattr(getattr(value, "packed", None), "_mmap", None)
    if mapping is not None:
        mapping.close()


def _mlp_names(archive: hq2.HQModel) -> list[str]:
    preferred = [
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "model.language_model.layers.0.mlp.up_proj.weight",
        "model.language_model.layers.0.mlp.down_proj.weight",
    ]
    actual = []
    for name in preferred:
        descriptor = archive.descriptor(name)
        if descriptor.format.name != "HQ2":
            raise ValueError(f"{name} is {descriptor.format.name}, expected HQ2 for packed benchmark")
        actual.append(name)
    return actual


def _error_summary(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    delta = (actual.float() - expected.float()).abs()
    return {
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
        "output_finite": bool(torch.isfinite(actual).all().item()),
    }


def benchmark_hq2_layers(args: argparse.Namespace, dtype: torch.dtype) -> dict[str, Any]:
    archive = hq2.load_model(args.archive)
    result: dict[str, Any] = {
        "archive": str(args.archive),
        "archive_file_bytes": args.archive.stat().st_size,
        "archive_payload_bpw": archive.metadata.get("planned_payload_bpw"),
        "fused_hq2_available": hq2.rocm_fused_available(),
        "dtype": str(dtype).removeprefix("torch."),
        "layers": [],
    }
    for name in _mlp_names(archive):
        descriptor = archive.descriptor(name)
        host_packed = archive.tensor(name)
        packed = host_packed.to("cuda")
        layer = hq2.HQ2Linear.from_archive(packed).to("cuda").eval()
        reference_weight = packed.dequantize(dtype=dtype).contiguous()
        out_features, in_features = descriptor.shape
        layer_result: dict[str, Any] = {
            "tensor": name,
            "shape": list(descriptor.shape),
            "packed_bytes": descriptor.nbytes,
            "measurements": [],
        }
        for batch in args.kernel_batches:
            if batch <= 0:
                raise ValueError("--kernel-batches values must be positive")
            input = torch.randn((batch, in_features), device="cuda", dtype=dtype)
            with torch.inference_mode():
                packed_result = layer(input)
                reference_result = F.linear(input, reference_weight)
            correctness = _error_summary(packed_result, reference_result)
            packed_time, _ = _event_time(
                lambda: layer(input), warmup=args.kernel_warmup, iterations=args.kernel_iterations
            )
            reference_time, _ = _event_time(
                lambda: F.linear(input, reference_weight), warmup=args.kernel_warmup, iterations=args.kernel_iterations
            )
            packed_ms = packed_time["median_ms"]
            reference_ms = reference_time["median_ms"]
            flops = 2.0 * batch * out_features * in_features
            packed_time.update(
                {
                    "effective_tflop_s": flops / (packed_ms * 1e9),
                    "packed_weight_read_gib_s": (descriptor.nbytes * batch * 1000.0) / packed_ms / (1024**3),
                }
            )
            reference_time.update({"effective_tflop_s": flops / (reference_ms * 1e9)})
            layer_result["measurements"].append(
                {
                    "token_rows": batch,
                    "packed_hq2": packed_time,
                    "decoded_bf16_f_linear": reference_time,
                    "packed_vs_decoded_speedup": reference_ms / packed_ms,
                    "correctness": correctness,
                }
            )
            del input, packed_result, reference_result
        result["layers"].append(layer_result)
        del reference_weight, layer, packed
        _close_packed_mapping(host_packed)
        gc.collect()
        torch.cuda.empty_cache()
    return result


def _input_ids(tokenizer: Any, tokens: int) -> torch.Tensor:
    encoded = tokenizer("HQ2 benchmark prompt. " * (tokens + 16), add_special_tokens=True, return_tensors="pt")["input_ids"]
    if encoded.shape[1] < tokens:
        repeats = (tokens + encoded.shape[1] - 1) // encoded.shape[1]
        encoded = encoded.repeat(1, repeats)
    return encoded[:, :tokens].contiguous().to("cuda")


def benchmark_hybrid_model(args: argparse.Namespace, dtype: torch.dtype) -> dict[str, Any]:
    if not args.package.is_dir():
        raise FileNotFoundError(f"Missing standalone HQ2 package: {args.package}")
    tokenizer = AutoTokenizer.from_pretrained(args.package, local_files_only=True)
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    model = hq2.load_gemma4_hq2_package(args.package, dtype=dtype, progress=lambda message: print(f"[load] {message}", flush=True))
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    result: dict[str, Any] = {
        "package": str(args.package),
        "package_archive_bytes": (args.package / "model.hq2").stat().st_size,
        "scope": "hybrid Gemma: 144 packed HQ2 MLP projections plus 533 BF16 RAW tensors",
        "load_seconds": load_seconds,
        "memory_after_load_gib": torch.cuda.memory_allocated() / (1024**3),
        "peak_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "prefill": [],
    }
    with torch.inference_mode():
        for tokens in args.prefill_tokens:
            if tokens <= 0:
                raise ValueError("--prefill-tokens values must be positive")
            inputs = {"input_ids": _input_ids(tokenizer, tokens)}
            timing, output = _event_time(
                lambda: model(**inputs, use_cache=True),
                warmup=args.prefill_warmup,
                iterations=args.prefill_iterations,
            )
            timing["tokens_per_second"] = tokens * 1000.0 / timing["median_ms"]
            result["prefill"].append({"tokens": tokens, **timing})
            del output, inputs
        prompt = _input_ids(tokenizer, max(args.prefill_tokens))
        initial = model(input_ids=prompt, use_cache=True)
        next_token = initial.logits[:, -1:].argmax(dim=-1)
        cache = initial.past_key_values
        for _ in range(args.decode_warmup):
            warm = model(input_ids=next_token, past_key_values=cache, use_cache=True)
            cache = warm.past_key_values
            next_token = warm.logits[:, -1:].argmax(dim=-1)
            del warm
        torch.cuda.synchronize()
        decode_ms: list[float] = []
        for _ in range(args.decode_tokens):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            decoded = model(input_ids=next_token, past_key_values=cache, use_cache=True)
            end.record()
            end.synchronize()
            decode_ms.append(float(start.elapsed_time(end)))
            cache = decoded.past_key_values
            next_token = decoded.logits[:, -1:].argmax(dim=-1)
            del decoded
    result["decode"] = _timing_summary(decode_ms)
    result["decode"]["tokens_per_second"] = 1000.0 / result["decode"]["median_ms"]
    result["memory_after_benchmark_gib"] = torch.cuda.memory_allocated() / (1024**3)
    result["peak_memory_gib"] = torch.cuda.max_memory_allocated() / (1024**3)
    del model, initial, prompt, cache, next_token
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError("This benchmark requires a visible ROCm Torch GPU")
    if not hq2.rocm_fused_available():
        raise RuntimeError("The packed HQ2 extension is unavailable; rebuild with setup_torch.py before measuring")
    dtype = _dtype(args.dtype)
    device = torch.cuda.get_device_properties(0)
    report: dict[str, Any] = {
        "benchmark": "hq2_packed_rocm_gemma4_v1",
        "device": torch.cuda.get_device_name(0),
        "device_arch": getattr(device, "gcnArchName", "unknown"),
        "torch": torch.__version__,
        "rocm": torch.version.hip,
        "generated_at_unix": time.time(),
        "kernel": benchmark_hq2_layers(args, dtype),
        "limitations": [
            "The complete 2.828-BPW mixed archive has Q4_0/Q8_0 payloads without packed PyTorch dispatch, so it cannot yet be an end-to-end packed model.",
            "The executable standalone package is a hybrid: HQ2 MLPs are packed, while its non-MLP tensors remain BF16 RAW.",
            "The BF16 F.linear rows compare the same HQ-decoded weights, not raw-model quality or GGUF inference.",
        ],
    }
    if not args.skip_model:
        report["hybrid_model"] = benchmark_hybrid_model(args, dtype)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
