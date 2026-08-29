"""essence 保守边界（MEMORY-LIFECYCLE §3.2）单元测试。

覆盖三处实现缺口：
1. essence 不再由单一事件直接产生（prior 表移除 essence）。
2. essence 豁免 forget/decay（不淡化、不抽象掉）。
3. essence 修订门槛更高（默认只 reinforce，仅高门槛才改写）。
"""

import pytest

from soul_elevation.engine import InternalizingEngine
from soul_elevation.llm import StubElevationLLM
from soul_elevation.models import ElevationInput
from soul_elevation.prior import CATEGORY_PRIOR_TABLE, PRIOR_TABLE, resolve_prior


def _inp(**overrides):
    defaults = dict(
        event_type="diary:night",
        content="我重视自由",
        source_id="evt-1",
        source_type="inner_life_event",
        timestamp="2026-08-29T00:00:00Z",
    )
    defaults.update(overrides)
    return ElevationInput(**defaults)


def _essence_engine(confidence=0.3):
    """LLM 后验在极高置信度下归类 essence（设计文档允许的「高置信 LLM 后验」路径）。"""
    return InternalizingEngine(
        StubElevationLLM(confidence=confidence, keyword_map={"温柔": "essence"})
    )


def _essence_node(eng, valence="positive"):
    return eng.consume(_inp(content="我温柔而疏离", provenance={"valence": valence}))[0]


# —— 缺口 1：essence 不再由单一事件直接产生 ——


def test_essence_never_primary_prior():
    for prior in list(PRIOR_TABLE.values()) + list(CATEGORY_PRIOR_TABLE.values()):
        assert prior[0] != "essence"


def test_essence_absent_from_prior_tuples():
    # 工作单决策：从 prior 元组里「移除 essence」，而非降为备选维度。
    for prior in list(PRIOR_TABLE.values()) + list(CATEGORY_PRIOR_TABLE.values()):
        assert "essence" not in prior


def test_single_calendar_event_produces_trait_not_essence():
    eng = InternalizingEngine(StubElevationLLM())
    node = eng.consume(_inp(event_type="world:calendar_event", content="今天去爬山"))[0]
    assert node.node_type == "trait"


def test_single_dream_produces_trait_not_essence():
    eng = InternalizingEngine(StubElevationLLM())
    node = eng.consume(_inp(event_type="dream:dream", content="梦见一片海"))[0]
    assert node.node_type == "trait"


def test_milestone_produces_value_not_essence():
    assert resolve_prior(
        "conversation:user_message", {"llm_judge": {"category": "milestone"}}
    ) == ("value",)
    eng = InternalizingEngine(StubElevationLLM())
    node = eng.consume(
        _inp(
            event_type="conversation:user_message",
            content="我毕业了",
            provenance={"llm_judge": {"category": "milestone"}},
        )
    )[0]
    assert node.node_type == "value"


# —— 缺口 2：essence 豁免 forget/decay ——


def test_decay_essence_is_noop():
    eng = _essence_engine()
    essence = _essence_node(eng)
    faded = eng.decay(essence.node_id)
    assert faded == []  # 不淡化
    # 证据边仍有效（未被淡化）。
    edges = [e for e in eng.evidence_edges if e.node_id == essence.node_id]
    assert edges
    assert all(e.valid_until_ts is None for e in edges)


def test_forget_essence_raises():
    eng = _essence_engine()
    essence = _essence_node(eng)
    with pytest.raises(ValueError):
        eng.forget([essence.node_id], "抽象掉内涵")


def test_forget_mixed_with_essence_raises_without_partial_fade():
    eng = _essence_engine()  # 带 keyword_map，可产 essence
    belief = eng.consume(_inp(event_type="world:news_event", content="世界很危险"))[0]
    essence = _essence_node(eng)  # 同一引擎内再产一个 essence
    with pytest.raises(ValueError):
        eng.forget([belief.node_id, essence.node_id], "抽象")
    # 整体拒绝：belief 的证据边未被淡化（无部分副作用）。
    belief_edges = [e for e in eng.evidence_edges if e.node_id == belief.node_id]
    assert all(e.valid_until_ts is None for e in belief_edges)


# —— 缺口 3：essence 修订门槛更高 ——


def test_essence_revise_default_reinforces_only():
    eng = _essence_engine(confidence=0.3)
    essence = _essence_node(eng)
    reinforced = eng.revise(essence.node_id, "我温柔而疏离（强化）")
    # 只 reinforce：不换 node_id、不改 lineage、不产生新因果节点。
    assert reinforced.node_id == essence.node_id
    assert reinforced.lineage_path == essence.lineage_path
    assert reinforced.parent_node_id is None
    assert reinforced.confidence > essence.confidence
    assert reinforced.stability > essence.stability
    # 注册表里仍是同一个节点（原地替换，无新节点）。
    assert eng.get_node(essence.node_id).node_id == essence.node_id


def test_essence_revise_high_threshold_rewrites():
    eng = _essence_engine(confidence=0.3)
    essence = _essence_node(eng, valence="positive")
    new = eng.revise(
        essence.node_id,
        "我变得冷漠了",
        new_confidence=0.8,  # delta = 0.5 > 0.3
        valence="negative",  # valence 反转
        source_ids=["evt-a", "evt-b"],  # ≥2 条新独立证据
    )
    # 高门槛满足 → 完整 reconsolidation 改写。
    assert new.node_id != essence.node_id
    assert new.parent_node_id == essence.node_id
    assert new.lineage_depth == essence.lineage_depth + 1
    assert new.valence == "negative"


def test_essence_revise_partial_threshold_reinforces():
    eng = _essence_engine(confidence=0.3)
    essence = _essence_node(eng, valence="positive")
    # 只有 valence 反转，缺 ≥2 证据 + confidence delta → 只 reinforce。
    reinforced = eng.revise(essence.node_id, "改写", valence="negative")
    assert reinforced.node_id == essence.node_id
    assert reinforced.confidence > essence.confidence


def test_belief_revise_unaffected_by_essence_threshold():
    # 非 essence 节点不受高门槛约束，仍走普通 reconsolidation 改写。
    eng = InternalizingEngine(StubElevationLLM())
    belief = eng.consume(_inp(event_type="world:news_event", content="世界很危险"))[0]
    new = eng.revise(belief.node_id, "世界没那么危险")
    assert new.node_id != belief.node_id
    assert new.parent_node_id == belief.node_id
