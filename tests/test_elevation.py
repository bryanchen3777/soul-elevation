"""elevate() 升华（Consolidation ≠ Elevation 边界）单元测试。

覆盖：单一事件 → pattern 候选；证据累积达阈值 → elevate 成 belief/value/trait/
essence；LLM = interpretation（候选解释，非 truth）；pattern 保留；阈值可配。
"""

import pytest

from soul_elevation.engine import (
    DEFAULT_ELEVATE_MIN_EVIDENCE,
    InternalizingEngine,
)
from soul_elevation.llm import StubElevationLLM
from soul_elevation.models import ElevationInput


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


def _engine(**llm_kwargs):
    return InternalizingEngine(StubElevationLLM(**llm_kwargs))


# —— 单一事件 → pattern（consolidation 输出，候选）——


def test_single_event_produces_pattern_not_soul_node():
    eng = _engine()
    nodes = eng.consume(_inp())
    assert len(nodes) == 1
    assert nodes[0].node_type == "pattern"
    assert nodes[0].candidate_node_type == "value"  # LLM 后验候选解释
    # 注册表里只有 pattern，没有灵魂结构。
    assert all(n.node_type == "pattern" for n in eng.nodes)


def test_single_event_does_not_elevate():
    # 单一事件证据不足（1 < 2）：LLM 说「这是价值」不算数，证据说了算。
    eng = _engine()
    p = eng.consume(_inp())[0]
    with pytest.raises(ValueError):
        eng.elevate(p.node_id)


# —— 证据累积 → elevate 成灵魂结构 ——


def test_two_events_elevate_to_value():
    eng = _engine()
    p1 = eng.consume(_inp(content="我重视自由", source_id="evt-1"))[0]
    eng.consume(_inp(content="我重视自由", source_id="evt-2"))
    value = eng.elevate(p1.node_id)
    assert value.node_type == "value"
    assert value.content == "我重视自由"
    assert value.parent_node_id == p1.node_id  # 因果树 parent 关系


def test_two_news_events_elevate_to_belief():
    eng = _engine()
    p1 = eng.consume(
        _inp(event_type="world:news_event", content="世界很危险", source_id="evt-1")
    )[0]
    eng.consume(
        _inp(event_type="world:news_event", content="世界很危险", source_id="evt-2")
    )
    belief = eng.elevate(p1.node_id)
    assert belief.node_type == "belief"


def test_two_events_elevate_to_trait():
    eng = _engine()
    p1 = eng.consume(
        _inp(event_type="user_going_outside", content="今天去爬山", source_id="evt-1")
    )[0]
    eng.consume(
        _inp(event_type="user_going_outside", content="今天去爬山", source_id="evt-2")
    )
    trait = eng.elevate(p1.node_id)
    assert trait.node_type == "trait"


def test_two_essence_candidate_events_elevate_to_essence():
    eng = _engine(keyword_map={"温柔": "essence"})
    p1 = eng.consume(_inp(content="我温柔而疏离", source_id="evt-1"))[0]
    eng.consume(_inp(content="我温柔而疏离", source_id="evt-2"))
    essence = eng.elevate(p1.node_id)
    assert essence.node_type == "essence"


# —— LLM = interpretation（候选解释，非 truth）——


def test_llm_candidate_is_interpretation_not_truth():
    # LLM 后验判 value 只是候选：单一事件不升华；证据累积后才成为灵魂事实。
    eng = _engine(keyword_map={"自由": "value"})
    p = eng.consume(_inp(event_type="world:news_event", content="新闻里我感悟到自由"))[0]
    assert p.node_type == "pattern"
    assert p.candidate_node_type == "value"  # 候选解释
    with pytest.raises(ValueError):
        eng.elevate(p.node_id)  # 1 条证据，不升华


def test_elevate_defaults_to_llm_candidate_dimension():
    # 不显式传 node_type → 升华维度 = LLM 后验候选（interpretation）。
    eng = _engine(keyword_map={"自由": "value"})
    p1 = eng.consume(_inp(event_type="world:news_event", content="新闻里我感悟到自由", source_id="evt-1"))[0]
    eng.consume(_inp(event_type="world:news_event", content="新闻里我感悟到自由", source_id="evt-2"))
    soul = eng.elevate(p1.node_id)
    assert soul.node_type == "value"  # 候选维度被采纳


def test_elevate_explicit_node_type_overrides_candidate():
    # 显式 node_type（如由 prior 表决定）覆盖 LLM 候选。
    eng = _engine(keyword_map={"自由": "value"})
    p1 = eng.consume(_inp(event_type="world:news_event", content="新闻里我感悟到自由", source_id="evt-1"))[0]
    eng.consume(_inp(event_type="world:news_event", content="新闻里我感悟到自由", source_id="evt-2"))
    soul = eng.elevate(p1.node_id, node_type="belief")
    assert soul.node_type == "belief"


# —— pattern 保留 + 证据留痕 ——


def test_pattern_preserved_after_elevate():
    eng = _engine()
    p1 = eng.consume(_inp(source_id="evt-1"))[0]
    p2 = eng.consume(_inp(source_id="evt-2"))[0]
    eng.elevate(p1.node_id)
    # pattern 仍在注册表（不覆盖、不删除）。
    assert eng.get_node(p1.node_id).node_type == "pattern"
    assert eng.get_node(p2.node_id).node_type == "pattern"


def test_elevate_supersedes_pattern_edges_and_creates_soul_edges():
    eng = _engine()
    p1 = eng.consume(_inp(source_id="evt-1"))[0]
    p2 = eng.consume(_inp(source_id="evt-2"))[0]
    soul = eng.elevate(p1.node_id)
    # pattern 的有效证据边被 superseded（留痕，不删 source_id）。
    pattern_edges = [e for e in eng.evidence_edges if e.node_id in (p1.node_id, p2.node_id)]
    assert pattern_edges
    assert all(e.valid_until_ts is not None for e in pattern_edges)
    # 灵魂节点新建证据边回指同一批 source_id（原文可回查）。
    soul_edges = [e for e in eng.evidence_edges if e.node_id == soul.node_id]
    assert sorted(e.source_id for e in soul_edges) == ["evt-1", "evt-2"]


def test_elevate_boosts_stability():
    eng = _engine()
    p1 = eng.consume(_inp(source_id="evt-1"))[0]
    eng.consume(_inp(source_id="evt-2"))  # 独立事件（SE-1：同事件重送只计 1）
    soul = eng.elevate(p1.node_id)
    assert soul.stability > p1.stability  # 证据累积 → 更稳定（slower change）


# —— 阈值可配 ——


def test_min_evidence_configurable():
    eng = _engine()
    p1 = eng.consume(_inp(source_id="evt-1"))[0]
    p2 = eng.consume(_inp(source_id="evt-2"))[0]
    # 默认阈值 2 满足；阈值 3 不满足。
    assert DEFAULT_ELEVATE_MIN_EVIDENCE == 2
    with pytest.raises(ValueError):
        eng.elevate(p1.node_id, min_evidence=3)
    eng.consume(_inp(source_id="evt-3"))
    soul = eng.elevate(p1.node_id, min_evidence=3)
    assert soul.node_type == "value"


def test_min_evidence_must_be_positive_int():
    eng = _engine()
    p = eng.consume(_inp())[0]
    for bad in (0, -1, 1.5):
        with pytest.raises(ValueError):
            eng.elevate(p.node_id, min_evidence=bad)


# —— 边界 ——


def test_elevate_requires_pattern_node():
    eng = _engine()
    p1 = eng.consume(_inp(source_id="evt-1"))[0]
    eng.consume(_inp(source_id="evt-2"))
    soul = eng.elevate(p1.node_id)
    with pytest.raises(ValueError):
        eng.elevate(soul.node_id)  # 灵魂结构不可再被 elevate


def test_elevate_unknown_node_raises():
    eng = _engine()
    with pytest.raises(KeyError):
        eng.elevate("no-such-pattern")


def test_elevate_extra_source_ids_add_edges():
    eng = _engine()
    p1 = eng.consume(_inp(source_id="evt-1"))[0]
    eng.consume(_inp(source_id="evt-2"))
    soul = eng.elevate(p1.node_id, source_ids=["evt-9"])
    soul_edges = [e for e in eng.evidence_edges if e.node_id == soul.node_id]
    assert sorted(e.source_id for e in soul_edges) == ["evt-1", "evt-2", "evt-9"]
