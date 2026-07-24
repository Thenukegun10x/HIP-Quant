"""Reproducible mixed-precision HQ-family policies.

The named policies here decide *where* a precision tier is used; codecs remain
independent implementation details.  This keeps whole-model BPW accounting
honest before a large conversion starts and makes a policy portable across the
archive writer, analyser, and benchmark harness.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import prod
from typing import Iterable, Mapping, Sequence

from .archive import HQ2_FORMAT, HQ3_FORMAT, HQFormatDescriptor
from .raw import raw_format


Q4_0_FORMAT = HQFormatDescriptor(
    name="Q4_0",
    version=1,
    layout="linear_out_in_row_major_blocks32",
    block_size=32,
    block_bytes=18,
    bits_per_weight=4.5,
    packing="fp16-scale-le+u4-signed-offset8-low16-high16",
)

Q8_0_FORMAT = HQFormatDescriptor(
    name="Q8_0",
    version=1,
    layout="linear_out_in_row_major_blocks32",
    block_size=32,
    block_bytes=34,
    bits_per_weight=8.5,
    packing="fp16-scale-le+i8-values-row-major",
)

F32_FORMAT = raw_format("float32")

_FORMATS: Mapping[str, HQFormatDescriptor] = {
    "hq2": HQ2_FORMAT,
    "hq3": HQ3_FORMAT,
    "q4_0": Q4_0_FORMAT,
    "q8_0": Q8_0_FORMAT,
    "f32": F32_FORMAT,
}


def _shape(value: Sequence[int]) -> tuple[int, ...]:
    result = tuple(int(dimension) for dimension in value)
    if not result or any(dimension <= 0 for dimension in result):
        raise ValueError(f"Tensor shape must be non-empty and positive, got {result}")
    return result


def _packed_bytes(shape: tuple[int, ...], format: HQFormatDescriptor) -> int:
    values = int(prod(shape))
    if values % format.block_size:
        raise ValueError(
            f"{format.name} requires a value count divisible by {format.block_size}, "
            f"got {values} for {shape}"
        )
    return values // format.block_size * format.block_bytes


@dataclass(frozen=True)
class MixedTensorPlan:
    """One tensor's assigned codec and exact packed-byte budget."""

    name: str
    shape: tuple[int, ...]
    tier: str
    format: HQFormatDescriptor

    @property
    def values(self) -> int:
        return int(prod(self.shape))

    @property
    def payload_bytes(self) -> int:
        return _packed_bytes(self.shape, self.format)

    @property
    def payload_bits_per_weight(self) -> float:
        return self.payload_bytes * 8 / self.values

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "tier": self.tier,
            "format": self.format.as_dict(),
            "values": self.values,
            "payload_bytes": self.payload_bytes,
            "payload_bits_per_weight": self.payload_bits_per_weight,
        }


@dataclass(frozen=True)
class MixedPolicyPlan:
    """A complete, auditable assignment for a checkpoint."""

    name: str
    target_payload_bpw: float
    tensors: tuple[MixedTensorPlan, ...]

    @property
    def logical_value_count(self) -> int:
        return sum(tensor.values for tensor in self.tensors)

    @property
    def payload_bytes(self) -> int:
        return sum(tensor.payload_bytes for tensor in self.tensors)

    @property
    def payload_bits_per_weight(self) -> float:
        values = self.logical_value_count
        return 0.0 if values == 0 else self.payload_bytes * 8 / values

    @property
    def tier_summary(self) -> dict[str, dict[str, int | float]]:
        grouped: dict[str, dict[str, int | float]] = defaultdict(
            lambda: {"tensors": 0, "values": 0, "payload_bytes": 0, "bpw": 0.0}
        )
        for tensor in self.tensors:
            item = grouped[tensor.tier]
            item["tensors"] = int(item["tensors"]) + 1
            item["values"] = int(item["values"]) + tensor.values
            item["payload_bytes"] = int(item["payload_bytes"]) + tensor.payload_bytes
            item["bpw"] = tensor.payload_bits_per_weight
        return dict(sorted(grouped.items()))

    def tensor(self, name: str) -> MixedTensorPlan:
        for tensor in self.tensors:
            if tensor.name == name:
                return tensor
        raise KeyError(name)

    def as_dict(self, *, include_tensors: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "policy": self.name,
            "target_payload_bpw": self.target_payload_bpw,
            "logical_value_count": self.logical_value_count,
            "payload_bytes": self.payload_bytes,
            "payload_bits_per_weight": self.payload_bits_per_weight,
            "tier_summary": self.tier_summary,
        }
        if include_tensors:
            result["tensors"] = [tensor.as_dict() for tensor in self.tensors]
        return result


def gemma4_hq2_2p8_tier(name: str, shape: Sequence[int]) -> str:
    """Assign the initial 2.8-BPW Gemma 4 policy.

    The routing is deliberately conservative where the existing quality tests
    were sensitive: Q/K/output attention stays Q4_0, while V and the tied
    embedding use HQ3.  The large text MLP remains the fast HQ2 path.  The
    policy is 2.8279 payload BPW on the supplied Gemma 4 12B checkpoint.
    """

    dimensions = _shape(shape)
    rank_two = len(dimensions) == 2
    width = dimensions[-1]
    text_layer = name.startswith("model.language_model.layers.")
    if rank_two and text_layer and ".mlp." in name and name.endswith(".weight"):
        if width % HQ2_FORMAT.block_size:
            raise ValueError(f"HQ2 MLP tensor {name!r} has unaligned width {width}")
        return "hq2"
    if rank_two and name == "model.language_model.embed_tokens.weight":
        if width % HQ3_FORMAT.block_size:
            raise ValueError(f"HQ3 embedding {name!r} has unaligned width {width}")
        return "hq3"
    if rank_two and text_layer and name.endswith(".self_attn.v_proj.weight"):
        if width % HQ3_FORMAT.block_size:
            raise ValueError(f"HQ3 V projection {name!r} has unaligned width {width}")
        return "hq3"
    if rank_two and text_layer and name.endswith((
        ".self_attn.q_proj.weight",
        ".self_attn.k_proj.weight",
        ".self_attn.o_proj.weight",
    )):
        if width % Q4_0_FORMAT.block_size:
            raise ValueError(f"Q4_0 attention tensor {name!r} has unaligned width {width}")
        return "q4_0"
    if rank_two and width % Q8_0_FORMAT.block_size == 0:
        return "q8_0"
    return "f32"


def plan_gemma4_hq2_2p8(
    tensors: Iterable[tuple[str, Sequence[int]]],
    *,
    tier_overrides: Mapping[str, str] | None = None,
    policy_name: str = "gemma4-hq2-mixed-2p8",
    target_payload_bpw: float = 2.8,
) -> MixedPolicyPlan:
    """Plan the complete Gemma policy, optionally overriding named tensor tiers.

    Overrides are deliberately exact-name only.  This prevents a mixed-policy
    experiment from silently applying a broad pattern to unintended tensors,
    and makes a one- or few-tensor upcast fully reproducible in the plan JSON.
    """

    if not policy_name:
        raise ValueError("policy_name must be non-empty")
    if target_payload_bpw <= 0.0:
        raise ValueError("target_payload_bpw must be positive")
    overrides = dict(tier_overrides or {})
    invalid_tiers = sorted({tier for tier in overrides.values() if tier not in _FORMATS})
    if invalid_tiers:
        raise ValueError(f"Unknown override tiers: {invalid_tiers}")

    selected: list[MixedTensorPlan] = []
    seen: set[str] = set()
    for name, supplied_shape in tensors:
        if not isinstance(name, str) or not name:
            raise ValueError("Tensor names must be non-empty strings")
        if name in seen:
            raise ValueError(f"Duplicate tensor name in mixed policy plan: {name!r}")
        seen.add(name)
        shape = _shape(supplied_shape)
        tier = overrides.pop(name, gemma4_hq2_2p8_tier(name, shape))
        # Validate the replacement tier's physical contract early, before a
        # large native conversion begins.
        _packed_bytes(shape, _FORMATS[tier])
        selected.append(MixedTensorPlan(name=name, shape=shape, tier=tier, format=_FORMATS[tier]))
    if not selected:
        raise ValueError("Cannot plan an empty checkpoint")
    if overrides:
        raise KeyError(f"Tier overrides name absent from checkpoint: {sorted(overrides)}")
    return MixedPolicyPlan(
        name=policy_name,
        target_payload_bpw=target_payload_bpw,
        tensors=tuple(selected),
    )


def format_for_tier(tier: str) -> HQFormatDescriptor:
    """Return the immutable physical-format contract for one policy tier."""

    try:
        return _FORMATS[tier]
    except KeyError as exc:
        raise ValueError(f"Unknown mixed HQ tier {tier!r}") from exc


__all__ = [
    "F32_FORMAT",
    "MixedPolicyPlan",
    "MixedTensorPlan",
    "Q4_0_FORMAT",
    "Q8_0_FORMAT",
    "format_for_tier",
    "gemma4_hq2_2p8_tier",
    "plan_gemma4_hq2_2p8",
]
