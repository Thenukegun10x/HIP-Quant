"""Inspect the reproducible Gemma HQ2-Mixed-2.8 placement before conversion.

This reads Safetensors metadata only: no weights are materialized and no GPU is
initialized.  It reports the exact packed payload BPW implied by HQ2, HQ3,
Q4_0, Q8_0, and FP32 tiers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from safetensors import safe_open


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hq2.mixed_policy import plan_gemma4_hq2_2p8


def _human_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    current = float(value)
    for unit in units:
        if current < 1024.0 or unit == units[-1]:
            return f"{current:.2f} {unit}"
        current /= 1024.0
    raise AssertionError("unreachable")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Gemma 4 BF16 Safetensors checkpoint")
    parser.add_argument("--json-out", type=Path, help="optional complete machine-readable plan output")
    parser.add_argument("--tensors", action="store_true", help="print every tensor placement")
    parser.add_argument(
        "--tier-overrides",
        type=Path,
        help="JSON object mapping exact source tensor names to hq2/hq3/q4_0/q8_0/f32 tiers",
    )
    parser.add_argument("--policy-name", default="gemma4-hq2-mixed-2p8")
    parser.add_argument("--target-payload-bpw", type=float, default=2.8)
    return parser.parse_args(argv)


def _load_tier_overrides(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or not all(isinstance(name, str) and isinstance(tier, str) for name, tier in loaded.items()):
        raise ValueError("--tier-overrides must contain a JSON object of tensor-name to tier strings")
    return dict(loaded)


def plan_checkpoint(
    path: Path,
    *,
    tier_overrides: dict[str, str],
    policy_name: str,
    target_payload_bpw: float,
):
    if not path.is_file():
        raise FileNotFoundError(path)
    with safe_open(path, framework="pt", device="cpu") as source:
        tensors = ((name, tuple(source.get_slice(name).get_shape())) for name in source.keys())
        return plan_gemma4_hq2_2p8(
            tensors,
            tier_overrides=tier_overrides,
            policy_name=policy_name,
            target_payload_bpw=target_payload_bpw,
        )


def render(plan, *, tensors: bool) -> str:
    lines = [
        f"Policy: {plan.name}",
        f"Payload: {_human_bytes(plan.payload_bytes)} for {plan.logical_value_count:,} values "
        f"({plan.payload_bits_per_weight:.6f} BPW; target {plan.target_payload_bpw:.2f})",
        "Tiers:",
    ]
    for tier, item in plan.tier_summary.items():
        lines.append(
            f"  {tier:5s} {int(item['tensors']):3d} tensors | {int(item['values']):,} values | "
            f"{_human_bytes(int(item['payload_bytes']))} | {float(item['bpw']):.4f} BPW"
        )
    if tensors:
        lines.append("Tensors:")
        for item in plan.tensors:
            lines.append(
                f"  {item.tier:5s} {item.name} | {item.shape} | "
                f"{_human_bytes(item.payload_bytes)}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = plan_checkpoint(
        args.input,
        tier_overrides=_load_tier_overrides(args.tier_overrides),
        policy_name=args.policy_name,
        target_payload_bpw=args.target_payload_bpw,
    )
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(plan.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(render(plan, tensors=args.tensors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
