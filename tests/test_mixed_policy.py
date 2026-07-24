"""CPU-only coverage for the reproducible Gemma HQ2-Mixed-2.8 router."""

from __future__ import annotations

import os
import sys

import pytest


_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_root = os.path.dirname(_src)
for _path in (_root, _src):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from hq2.mixed_policy import (
    Q4_0_FORMAT,
    Q8_0_FORMAT,
    gemma4_hq2_2p8_tier,
    plan_gemma4_hq2_2p8,
)


def test_gemma_mixed_router_uses_the_intended_tiers():
    assert gemma4_hq2_2p8_tier("model.language_model.layers.0.mlp.up_proj.weight", (15360, 3840)) == "hq2"
    assert gemma4_hq2_2p8_tier("model.language_model.layers.0.self_attn.v_proj.weight", (2048, 3840)) == "hq3"
    assert gemma4_hq2_2p8_tier("model.language_model.layers.0.self_attn.q_proj.weight", (4096, 3840)) == "q4_0"
    assert gemma4_hq2_2p8_tier("model.language_model.embed_tokens.weight", (262144, 3840)) == "hq3"
    assert gemma4_hq2_2p8_tier("model.embed_vision.embedding_projection.weight", (3840, 3840)) == "q8_0"
    assert gemma4_hq2_2p8_tier("model.language_model.norm.weight", (3840,)) == "f32"


def test_mixed_plan_accounts_physical_block_sizes_exactly():
    plan = plan_gemma4_hq2_2p8(
        (
            ("model.language_model.layers.0.mlp.up_proj.weight", (1, 256)),
            ("model.language_model.layers.0.self_attn.v_proj.weight", (1, 256)),
            ("model.language_model.layers.0.self_attn.q_proj.weight", (1, 32)),
            ("model.language_model.embed_tokens.weight", (1, 256)),
            ("model.embed_audio.embedding_projection.weight", (1, 32)),
            ("model.language_model.norm.weight", (1,)),
        )
    )
    assert plan.tensor("model.language_model.layers.0.mlp.up_proj.weight").payload_bytes == 72
    assert plan.tensor("model.language_model.layers.0.self_attn.v_proj.weight").payload_bytes == 112
    assert plan.tensor("model.language_model.layers.0.self_attn.q_proj.weight").payload_bytes == Q4_0_FORMAT.block_bytes
    assert plan.tensor("model.embed_audio.embedding_projection.weight").payload_bytes == Q8_0_FORMAT.block_bytes
    assert plan.tensor("model.language_model.norm.weight").payload_bytes == 4
    assert set(plan.tier_summary) == {"f32", "hq2", "hq3", "q4_0", "q8_0"}


def test_mixed_router_rejects_unaligned_required_hq_tensor():
    with pytest.raises(ValueError, match="unaligned"):
        gemma4_hq2_2p8_tier("model.language_model.layers.0.mlp.up_proj.weight", (4, 3839))


def test_mixed_plan_supports_exact_name_upcasts_and_rejects_typos():
    name = "model.language_model.layers.0.mlp.down_proj.weight"
    plan = plan_gemma4_hq2_2p8(
        ((name, (1, 256)),),
        tier_overrides={name: "hq3"},
        policy_name="gemma4-hq2-layer0-hq3",
        target_payload_bpw=2.85,
    )
    assert plan.name == "gemma4-hq2-layer0-hq3"
    assert plan.target_payload_bpw == 2.85
    assert plan.tensor(name).tier == "hq3"
    assert plan.tensor(name).payload_bytes == 112

    with pytest.raises(KeyError, match="absent"):
        plan_gemma4_hq2_2p8(((name, (1, 256)),), tier_overrides={"missing.weight": "hq3"})
