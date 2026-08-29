"""reconsolidation 式信念修订（revise）单元测试。"""

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


def _engine():
    return InternalizingEngine(StubElevationLLM())


def test_revise_creates_new_node_with_parent():
    eng = _engine()
    old = eng.consume(_inp())[0]
    new = eng.revise(old.node_id, "我更重视自由了")
    assert new.node_id != old.node_id
    assert new.parent_node_id == old.node_id
    assert new.lineage_depth == old.lineage_depth + 1
    assert new.lineage_path == f"{old.lineage_path}/{new.node_id}"


def test_revise_does_not_overwrite_old_node():
    eng = _engine()
    old = eng.consume(_inp())[0]
    eng.revise(old.node_id, "改写后的内容")
    kept = eng.get_node(old.node_id)
    assert kept.node_id == old.node_id
    assert kept.content == old.content  # 旧节点原文未改


def test_revise_marks_old_edges_valid_until():
    eng = _engine()
    old = eng.consume(_inp())[0]
    eng.revise(old.node_id, "改写后的内容")
    old_edges = [e for e in eng.evidence_edges if e.node_id == old.node_id]
    assert old_edges
    assert all(e.valid_until_ts is not None for e in old_edges)  # 留痕
    assert all(e.source_id == "evt-1" for e in old_edges)        # source_id 保留


def test_revise_new_node_gets_new_edge_when_source_provided():
    eng = _engine()
    old = eng.consume(_inp())[0]
    new = eng.revise(old.node_id, "改写", source_id="evt-2", inner_life_event_id="evt-2")
    new_edges = [e for e in eng.evidence_edges if e.node_id == new.node_id]
    assert len(new_edges) == 1
    assert new_edges[0].source_id == "evt-2"
    assert new.provenance_ref == "evt-2"


def test_revise_uses_llm_reassessment():
    llm = StubElevationLLM(keyword_map={"自由": "value"})
    eng = InternalizingEngine(llm)
    old = eng.consume(_inp(event_type="world:news_event", content="世界很危险"))[0]
    assert old.node_type == "pattern"
    assert old.candidate_node_type == "belief"
    new = eng.revise(old.node_id, "我感悟到自由")  # 命中 keyword → 候选维度变 value
    assert new.node_type == "pattern"  # 修订产出仍是候选 pattern
    assert new.candidate_node_type == "value"


def test_revise_new_confidence_overrides_llm():
    eng = _engine()  # 默认 stub confidence=0.5
    old = eng.consume(_inp())[0]
    new = eng.revise(old.node_id, "确定性内容", new_confidence=0.9)
    assert new.confidence == 0.9
    assert new.node_type == old.node_type
    assert new.content == "确定性内容"


def test_revise_reinforces_stability():
    eng = _engine()
    old = eng.consume(_inp())[0]
    assert old.stability == 0.0
    new = eng.revise(old.node_id, "改写")
    assert new.stability > old.stability


def test_revise_unknown_node_raises():
    eng = _engine()
    with pytest.raises(KeyError):
        eng.revise("no-such-node", "x")


def test_multiple_revisions_lineage_path_correct():
    eng = _engine()
    n0 = eng.consume(_inp())[0]
    n1 = eng.revise(n0.node_id, "第一次修订")
    n2 = eng.revise(n1.node_id, "第二次修订")
    n3 = eng.revise(n2.node_id, "第三次修订")

    assert n1.lineage_depth == 1
    assert n2.lineage_depth == 2
    assert n3.lineage_depth == 3
    assert n1.lineage_path == f"{n0.node_id}/{n1.node_id}"
    assert n2.lineage_path == f"{n0.node_id}/{n1.node_id}/{n2.node_id}"
    assert n3.lineage_path == f"{n0.node_id}/{n1.node_id}/{n2.node_id}/{n3.node_id}"
    # 每代父指针正确
    assert n2.parent_node_id == n1.node_id
    assert n3.parent_node_id == n2.node_id
    # 所有代数都在注册表内（不覆盖旧节点）
    assert eng.get_node(n0.node_id).node_id == n0.node_id
    assert eng.get_node(n1.node_id).node_id == n1.node_id
    assert eng.get_node(n2.node_id).node_id == n2.node_id
