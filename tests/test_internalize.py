"""consume() 内化映射（先验 + 后验 → node + edge）单元测试。"""

import pytest

from soul_elevation.engine import InternalizingEngine
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


# —— 各 trigger_type → node_type 先验（经默认 stub 沿用基调）——


def test_news_maps_to_belief_node():
    eng = InternalizingEngine(StubElevationLLM())
    nodes = eng.consume(_inp(event_type="world:news_event", content="世界很危险"))
    assert len(nodes) == 1
    assert nodes[0].node_type == "belief"


def test_leisure_maps_to_essence_trait():
    eng = InternalizingEngine(StubElevationLLM())
    nodes = eng.consume(_inp(event_type="user_going_outside", content="今天去爬山"))
    assert nodes[0].node_type in ("essence", "trait")
    assert nodes[0].node_type == "essence"  # 默认 stub 沿用先验基调


def test_diary_maps_to_value_trait_belief():
    eng = InternalizingEngine(StubElevationLLM())
    nodes = eng.consume(_inp(event_type="diary:night", content="我重视自由"))
    assert nodes[0].node_type in ("value", "trait", "belief")
    assert nodes[0].node_type == "value"


def test_conversation_by_category():
    eng = InternalizingEngine(StubElevationLLM())
    pref = eng.consume(
        _inp(
            event_type="conversation:user_message",
            content="我喜欢喝咖啡",
            provenance={"llm_judge": {"category": "preference_plan_event_fact"}},
        )
    )[0]
    assert pref.node_type == "belief"

    milestone = eng.consume(
        _inp(
            event_type="conversation:user_message",
            content="我毕业了",
            provenance={"llm_judge": {"category": "milestone"}},
        )
    )[0]
    assert milestone.node_type == "essence"


# —— LLM 后验覆盖先验 ——


def test_llm_posterior_overrides_prior():
    # 新闻先验 = belief，但 LLM 依 content 后验判为 value。
    llm = StubElevationLLM(keyword_map={"自由": "value"})
    eng = InternalizingEngine(llm)
    nodes = eng.consume(_inp(event_type="world:news_event", content="新闻里我感悟到自由"))
    assert nodes[0].node_type == "value"


def test_invalid_llm_node_type_raises():
    class _BadLLM:
        def classify(self, content, provenance, prior_node_type):
            from soul_elevation.llm import Classification
            return Classification("memory", content, 0.5)  # memory 不是合法维度

    eng = InternalizingEngine(_BadLLM())
    with pytest.raises(ValueError):
        eng.consume(_inp())


# —— 产出 node + edge ——


def test_produced_edge_references_input_source():
    eng = InternalizingEngine(StubElevationLLM())
    nodes = eng.consume(
        _inp(event_type="world:news_event", source_type="v1_memory", source_id="mem-42")
    )
    node = nodes[0]
    edges = eng.evidence_edges
    assert len(edges) == 1
    edge = edges[0]
    assert edge.node_id == node.node_id
    assert edge.source_type == "v1_memory"
    assert edge.source_id == "mem-42"
    assert edge.trigger_type == "world:news_event"


def test_edge_dual_temporal_validity():
    eng = InternalizingEngine(StubElevationLLM())
    nodes = eng.consume(_inp())
    node = nodes[0]
    edge = eng.evidence_edges[0]
    assert edge.valid_from_ts == node.created_ts  # valid_from_ts = now
    assert edge.valid_until_ts is None            # 仍有效
    assert edge.valid_from_ts                     # 非空


def test_node_causal_tree_root():
    eng = InternalizingEngine(StubElevationLLM())
    node = eng.consume(_inp())[0]
    assert node.parent_node_id is None
    assert node.lineage_depth == 0
    assert node.lineage_path == node.node_id


def test_provenance_ref_for_inner_life_event_source():
    eng = InternalizingEngine(StubElevationLLM())
    node = eng.consume(_inp(source_type="inner_life_event", source_id="evt-9"))[0]
    assert node.provenance_ref == "evt-9"
    assert eng.evidence_edges[0].inner_life_event_id == "evt-9"


def test_agent_id_from_provenance_then_default():
    eng = InternalizingEngine(StubElevationLLM(), agent_id="default-agent")
    node = eng.consume(_inp(provenance={"agent_id": "agent-77"}))[0]
    assert node.agent_id == "agent-77"

    node2 = eng.consume(_inp())[0]
    assert node2.agent_id == "default-agent"


def test_default_valence_neutral():
    eng = InternalizingEngine(StubElevationLLM())
    node = eng.consume(_inp())[0]
    assert node.valence == "neutral"


def test_consume_accumulates_edges():
    eng = InternalizingEngine(StubElevationLLM())
    eng.consume(_inp())
    eng.consume(_inp())
    assert len(eng.evidence_edges) == 2
    assert len(eng.evidence_edges[0].edge_id) == 32  # 32-hex
