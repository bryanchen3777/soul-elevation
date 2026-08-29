"""SE-2 Pattern Terminal Semantics：Pattern 是合法终态，不是升华候车室。

Contract：``consume()`` → candidate Pattern；``elevate()`` → accepted
Soul-level node（belief/value/trait/essence）。Pattern 可长期存在，即使永远
不 elevate；未 elevate ≠ 失败。``consume()`` 不直接写 Soul state；WorldEvent
→ Pattern 允许，WorldEvent → trait/essence 直接路径 v1 禁止；knowledge /
capability 只留位不实作；Soul destination 只有 meaning/identity。
"""

import pytest

from soul_elevation.engine import InternalizingEngine
from soul_elevation.llm import StubElevationLLM
from soul_elevation.models import (
    SOUL_NODE_TYPES,
    VALID_NODE_TYPES,
    ElevationInput,
)


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


def test_consume_yields_only_pattern():
    # 单次 consume() 只得到 Pattern（candidate），不直接写 Soul state。
    eng = _engine()
    nodes = eng.consume(_inp())
    assert len(nodes) == 1
    assert nodes[0].node_type == "pattern"
    # 注册表里只有 pattern，无任何灵魂结构（consume 不写 Soul state）。
    assert all(n.node_type == "pattern" for n in eng.nodes)


def test_pattern_queryable_without_elevate():
    # Pattern 可 query / persist：不 elevate 也不消失、不报错。
    eng = _engine()
    p = eng.consume(_inp())[0]
    assert eng.get_node(p.node_id).node_id == p.node_id
    assert eng.get_node(p.node_id).node_type == "pattern"
    assert p.node_id in [n.node_id for n in eng.nodes]


def test_never_elevated_pattern_is_legal_terminal_state():
    # 永远不 elevate 的 Pattern 是合法终态：引擎状态有效、pattern 长期存在。
    eng = _engine()
    p1 = eng.consume(_inp(source_id="evt-1"))[0]
    p2 = eng.consume(_inp(source_id="evt-2"))[0]
    # 不调用 elevate，直接做其他生命周期操作（revise pattern 仍是 pattern）。
    revised = eng.revise(p1.node_id, "我更重视自由了")
    assert revised.node_type == "pattern"
    # 所有 pattern 都在注册表，无灵魂结构被意外产生。
    assert all(n.node_type == "pattern" for n in eng.nodes)
    assert eng.get_node(p2.node_id).node_type == "pattern"


def test_elevate_reject_below_threshold_pattern_remains():
    # 证据未达独立门槛时 elevate() reject，且 Pattern 仍在（不销毁、不消失）。
    eng = _engine()
    p = eng.consume(_inp())[0]
    with pytest.raises(ValueError):
        eng.elevate(p.node_id)
    assert eng.get_node(p.node_id).node_type == "pattern"
    assert eng.get_node(p.node_id).node_id == p.node_id


def test_world_event_produces_pattern_not_trait_essence():
    # WorldEvent → Pattern 允许；WorldEvent → trait/essence 直接路径 v1 禁止。
    eng = _engine()
    for event_type, content in (
        ("world:news_event", "世界很危险"),
        ("world:calendar_event", "今天去爬山"),
        ("world:weather_event", "今天有雨"),
    ):
        node = eng.consume(_inp(event_type=event_type, content=content))[0]
        assert node.node_type == "pattern"
        assert node.node_type not in ("trait", "essence")


def test_knowledge_capability_not_implemented():
    # knowledge / capability 只留位不实作：不在节点类型词汇表里。
    assert "knowledge" not in VALID_NODE_TYPES
    assert "capability" not in VALID_NODE_TYPES
    assert "competence" not in VALID_NODE_TYPES
    # Soul destination 只有 meaning/identity 维度（competence reserved/out of scope）。
    assert SOUL_NODE_TYPES == frozenset({"belief", "value", "trait", "essence"})


def test_pattern_persists_across_engine_operations():
    # Pattern 可长期存在：后续 consume / elevate 其他 pattern 不影响它。
    eng = _engine()
    p_never = eng.consume(_inp(source_id="evt-1"))[0]
    p_a = eng.consume(_inp(source_id="evt-2"))[0]
    eng.consume(_inp(source_id="evt-3"))
    eng.elevate(p_a.node_id)  # 别的 pattern 升华
    # 未升华的 pattern 仍在、仍可 query。
    assert eng.get_node(p_never.node_id).node_type == "pattern"
    assert eng.get_node(p_never.node_id).node_id == p_never.node_id
