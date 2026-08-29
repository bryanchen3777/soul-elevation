"""数据模型（ElevationNode / EvidenceEdge）单元测试。"""

from dataclasses import FrozenInstanceError

import pytest

from soul_elevation.models import (
    SOUL_NODE_TYPES,
    VALID_NODE_TYPES,
    VALID_SOURCE_TYPES,
    VALID_VALENCES,
    ElevationNode,
    EvidenceEdge,
    new_id,
)


def _node(**overrides):
    defaults = dict(
        node_id="n1",
        node_type="belief",
        content="世界是危险的",
        confidence=0.7,
        stability=0.5,
        valence="negative",
        agent_id="agent-a",
        parent_node_id=None,
        lineage_depth=0,
        lineage_path="n1",
        created_ts="2026-08-29T00:00:00Z",
        provenance_ref=None,
    )
    defaults.update(overrides)
    return ElevationNode(**defaults)


def _edge(**overrides):
    defaults = dict(
        edge_id="e1",
        node_id="n1",
        source_type="v1_memory",
        source_id="mem-1",
        agent_id="agent-a",
        weight=0.5,
        valid_from_ts="2026-08-29T00:00:00Z",
        valid_until_ts=None,
        inner_life_event_id=None,
        trigger_type="diary:night",
    )
    defaults.update(overrides)
    return EvidenceEdge(**defaults)


# —— ElevationNode ——


def test_new_id_is_32_hex():
    nid = new_id()
    assert len(nid) == 32
    int(nid, 16)  # 是合法十六进制


def test_node_is_frozen():
    n = _node()
    with pytest.raises(FrozenInstanceError):
        n.content = "改写"


def test_node_type_vocabulary():
    # pattern 是第 5 类节点（consolidation 输出，非灵魂结构）。
    assert sorted(VALID_NODE_TYPES) == ["belief", "essence", "pattern", "trait", "value"]
    assert sorted(SOUL_NODE_TYPES) == ["belief", "essence", "trait", "value"]
    with pytest.raises(ValueError):
        _node(node_type="memory")


def test_pattern_node_accepts_candidate_node_type():
    # pattern 节点可带 LLM 后验候选维度（interpretation，非 truth）。
    p = _node(node_type="pattern", candidate_node_type="belief")
    assert p.node_type == "pattern"
    assert p.candidate_node_type == "belief"


def test_candidate_node_type_must_be_soul_dimension():
    # 候选维度必须是灵魂结构（belief/value/trait/essence），pattern 自身不可作候选。
    with pytest.raises(ValueError):
        _node(node_type="pattern", candidate_node_type="pattern")
    with pytest.raises(ValueError):
        _node(node_type="pattern", candidate_node_type="memory")


def test_soul_node_candidate_defaults_none():
    # 灵魂结构节点不携带候选维度（默认 None）。
    n = _node(node_type="belief")
    assert n.candidate_node_type is None


def test_confidence_stability_bounds():
    for bad in (1.5, -0.1):
        with pytest.raises(ValueError):
            _node(confidence=bad)
    with pytest.raises(ValueError):
        _node(stability=2.0)


def test_valence_vocabulary():
    assert sorted(VALID_VALENCES) == ["negative", "neutral", "positive"]
    with pytest.raises(ValueError):
        _node(valence="mixed")


def test_lineage_depth_nonnegative():
    with pytest.raises(ValueError):
        _node(lineage_depth=-1)


def test_causal_tree_fields():
    n = _node(parent_node_id="root", lineage_depth=1, lineage_path="root/n1")
    assert n.parent_node_id == "root"
    assert n.lineage_depth == 1
    assert n.lineage_path == "root/n1"


def test_node_does_not_store_evidence_body():
    # 关键保真约束：节点不含证据正文字段，只带证据边索引。
    n = _node()
    assert not hasattr(n, "evidence_body")
    assert not hasattr(n, "evidence_text")
    assert "evidence" not in n.__dict__


# —— EvidenceEdge ——


def test_edge_source_type_vocabulary():
    assert sorted(VALID_SOURCE_TYPES) == [
        "inner_life_event",
        "sage_fact",
        "v1_memory",
        "world_event",
    ]
    with pytest.raises(ValueError):
        _edge(source_type="s3_file")


def test_edge_weight_bounds():
    with pytest.raises(ValueError):
        _edge(weight=1.1)
    with pytest.raises(ValueError):
        _edge(weight=-0.5)


def test_edge_dual_temporal_validity():
    active = _edge()
    assert active.valid_until_ts is None  # 仍有效
    superseded = _edge(valid_until_ts="2026-08-30T00:00:00Z")
    assert superseded.valid_until_ts == "2026-08-30T00:00:00Z"  # 已被取代（留痕）


def test_edge_source_id_retained_for_lookup():
    e = _edge(source_type="sage_fact", source_id="fact-42")
    assert e.source_id == "fact-42"  # 回指原文，可回查
