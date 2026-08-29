"""SE-3 Lineage vs Evidence：两图职责锁死（REGRESSION / INVARIANT）。

Invariant：Lineage = How did this node evolve?（``parent_node_id`` /
``lineage_depth`` / ``lineage_path``）；Evidence = What supports this node?
（``EvidenceEdge`` 回指原始 source）。N1→N2（reconsolidation）= lineage ≠
evidence。即使 N2 吸收 N1 的 evidence，N1→N2 仍只是 lineage，N2 的 supporting
evidence 仍是原始独立 evidence keys，不得因吸收就把 N1 本身算成一份新 evidence。
"""

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


def _active_edges(eng, node_id):
    return [
        e
        for e in eng.evidence_edges
        if e.node_id == node_id and e.valid_until_ts is None
    ]


def test_reconsolidation_lineage_not_evidence():
    # N1 → N2（revise）：lineage 含 N1→N2；N1 不是 N2 的 evidence neighbor。
    eng = _engine()
    n1 = eng.consume(_inp(source_id="evt-1"))[0]
    n2 = eng.revise(n1.node_id, "我更重视自由了", source_id="evt-2")
    # lineage：N2 的父是 N1（演化边）。
    assert n2.parent_node_id == n1.node_id
    assert n2.lineage_path == f"{n1.lineage_path}/{n2.node_id}"
    # evidence：N2 的 supporting evidence 是原始 source（evt-2），N1 不是 evidence。
    n2_sources = [e.source_id for e in _active_edges(eng, n2.node_id)]
    assert n2_sources == ["evt-2"]
    assert n1.node_id not in n2_sources


def test_reconsolidation_absorbs_old_evidence_without_counting_lineage():
    # N2 吸收 N1 的 evidence（elevate 聚合 pattern 证据）后，
    # 独立 evidence 计数不因 lineage 边 +1；N1 本身不算新 evidence。
    eng = _engine()
    p1 = eng.consume(_inp(source_id="evt-1"))[0]
    p2 = eng.consume(_inp(source_id="evt-2"))[0]
    soul = eng.elevate(p1.node_id)
    # lineage：soul 的父是 p1（pattern）。
    assert soul.parent_node_id == p1.node_id
    # evidence：soul 的 supporting evidence 是原始独立 evidence keys
    # （evt-1, evt-2），不含 p1 / p2 节点 id。
    soul_sources = [e.source_id for e in _active_edges(eng, soul.node_id)]
    assert sorted(soul_sources) == ["evt-1", "evt-2"]
    assert len(set(soul_sources)) == 2  # 独立计数 = 2，lineage 边不 +1
    assert p1.node_id not in soul_sources
    assert p2.node_id not in soul_sources


def test_evidence_never_references_elevation_nodes():
    # 程式内 invariant：任何 EvidenceEdge.source_id 不得是注册表里的节点 id。
    eng = _engine()
    p1 = eng.consume(_inp(source_id="evt-1"))[0]
    p2 = eng.consume(_inp(source_id="evt-2"))[0]
    eng.elevate(p1.node_id)
    n3 = eng.revise(p2.node_id, "改写", source_id="evt-3")
    eng.forget([n3.node_id], "抽象命题")
    node_ids = {n.node_id for n in eng.nodes}
    for e in eng.evidence_edges:
        assert e.source_id not in node_ids


def test_check_invariants_passes_after_full_lifecycle():
    # check_invariants() 在完整生命周期后不抛（两图职责未被破坏）。
    eng = _engine()
    p1 = eng.consume(_inp(source_id="evt-1"))[0]
    p2 = eng.consume(_inp(source_id="evt-2"))[0]
    eng.elevate(p1.node_id)
    eng.revise(p2.node_id, "改写", source_id="evt-3")
    eng.check_invariants()  # 不抛即通过


def test_lineage_edge_never_written_into_evidence_graph():
    # 演化边（parent_node_id）绝不进入 evidence graph：
    # 没有任何 EvidenceEdge 以 lineage 节点 id 为 source_id。
    eng = _engine()
    n1 = eng.consume(_inp(source_id="evt-1"))[0]
    n2 = eng.revise(n1.node_id, "改写", source_id="evt-2")
    n3 = eng.revise(n2.node_id, "再改写", source_id="evt-3")
    lineage_ids = {n1.node_id, n2.node_id, n3.node_id}
    for e in eng.evidence_edges:
        assert e.source_id not in lineage_ids


def test_old_edges_superseded_not_copied_as_evidence():
    # revise 后旧节点证据边被 superseded（留痕），不复制成新节点的 evidence。
    eng = _engine()
    n1 = eng.consume(_inp(source_id="evt-1"))[0]
    n2 = eng.revise(n1.node_id, "改写", source_id="evt-2")
    # 旧节点 evt-1 边：superseded（valid_until_ts 非 None），source_id 保留。
    old_edges = [e for e in eng.evidence_edges if e.node_id == n1.node_id]
    assert old_edges
    assert all(e.valid_until_ts is not None for e in old_edges)
    assert all(e.source_id == "evt-1" for e in old_edges)
    # 新节点 evidence 只有 evt-2（原始 source），不含 evt-1 也不含 n1。
    n2_sources = [e.source_id for e in _active_edges(eng, n2.node_id)]
    assert n2_sources == ["evt-2"]
