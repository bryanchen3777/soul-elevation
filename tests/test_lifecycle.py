"""SE-5 Durable Soul Structure Lifecycle 单元测试。

覆盖（照 docs/ELEVATION-LIFECYCLE.md）：
- 四态（ACTIVE → WEAKENING → DORMANT → SUPERSEDED，显式持久字段，状态机非动作）。
- 两转换（REINFORCE / SUPERSEDE）+ 默认「证据不足什么都不做」。
- Contradiction ≠ Revision（矛盾进压力累积器，达阈值才改变；一次反例不推翻）。
- Forgetting = lifecycle transition（节点不删，lineage / 证据链保留）。
- essence 保守（豁免自动衰减；SUPERSEDE 门槛全系统最高；reconsideration-candidate）。
- Decay 锚点复用 M5.13 语义（last_support 优先，created + grace 兜底；old ≠ outdated）。
- Trace additive 扩展（node_state_changed / node_superseded / essence_reconsideration_candidate）。
"""

from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from soul_elevation.engine import InternalizingEngine
from soul_elevation.llm import StubElevationLLM
from soul_elevation.models import (
    VALID_LIFECYCLE_STATES,
    ContradictionRecord,
    ElevationInput,
    ElevationNode,
)
from soul_elevation.trace import EVENT_TYPES, ElevationTraceWriter, read_trace


def _inp(**overrides):
    defaults = dict(
        event_type="world:news_event",
        content="世界很危险",
        source_id="evt-1",
        source_type="inner_life_event",
        timestamp="2026-08-29T00:00:00Z",
    )
    defaults.update(overrides)
    return ElevationInput(**defaults)


def _engine():
    return InternalizingEngine(StubElevationLLM())


def _essence_engine(confidence=0.3):
    return InternalizingEngine(
        StubElevationLLM(confidence=confidence, keyword_map={"温柔": "essence"})
    )


def _belief(eng, valence="negative"):
    """两次独立事件 → 2 pattern（candidate=belief）→ elevate 出 belief 灵魂节点。"""
    p1 = eng.consume(
        _inp(content="世界很危险", provenance={"valence": valence}, source_id="evt-1")
    )[0]
    eng.consume(
        _inp(content="世界很危险", provenance={"valence": valence}, source_id="evt-2")
    )
    return eng.elevate(p1.node_id)


def _essence(eng, valence="positive"):
    p1 = eng.consume(
        _inp(content="我温柔而疏离", provenance={"valence": valence}, source_id="evt-1")
    )[0]
    eng.consume(
        _inp(content="我温柔而疏离", provenance={"valence": valence}, source_id="evt-2")
    )
    return eng.elevate(p1.node_id)


def _days_later(ts, days):
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return (dt + timedelta(days=days)).isoformat()


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


# —— 字段与词汇表（additive schema）——


def test_lifecycle_state_defaults_active():
    n = _node()
    assert n.lifecycle_state == "active"  # 节点创建即 ACTIVE
    assert n.last_support_ts is None
    assert n.contradiction_pressure == ()
    assert n.superseded_by is None
    assert n.reconsideration_candidate is False


def test_valid_lifecycle_states():
    assert sorted(VALID_LIFECYCLE_STATES) == ["active", "dormant", "superseded", "weakening"]


def test_invalid_lifecycle_state_rejected():
    with pytest.raises(ValueError):
        _node(lifecycle_state="archived")


def test_contradiction_record_fields():
    r = ContradictionRecord(
        source_id="s1", ts="2026-09-01T00:00:00Z", event_identity="e1"
    )
    assert r.source_id == "s1"
    assert r.ts == "2026-09-01T00:00:00Z"
    assert r.event_identity == "e1"
    assert r.provenance_ref is None


def test_consume_sets_last_support_ts():
    eng = _engine()
    node = eng.consume(_inp())[0]
    assert node.last_support_ts is not None  # 创建即有支持证据（decay 锚点）


# —— 默认「证据不足什么都不做」——


def test_default_noop_without_evidence():
    eng = _engine()
    belief = _belief(eng)
    node = eng.get_node(belief.node_id)
    assert node.lifecycle_state == "active"
    assert node.contradiction_pressure == ()
    # T_weaken 内评估 → 状态不变
    now = _days_later(belief.created_ts, 3)
    assert eng.evaluate_lifecycle(now_ts=now) == []
    assert eng.get_node(belief.node_id).lifecycle_state == "active"


# —— REINFORCE 转换 ——


def test_reinforce_active_stays_active_in_place():
    eng = _engine()
    belief = _belief(eng)
    before = eng.get_node(belief.node_id)
    reinforced = eng.reinforce(belief.node_id, source_id="evt-3")
    # 不换 node_id、不改 lineage、不产生新因果节点
    assert reinforced.node_id == belief.node_id
    assert reinforced.lineage_path == before.lineage_path
    assert reinforced.parent_node_id == before.parent_node_id
    assert reinforced.lifecycle_state == "active"
    assert reinforced.confidence > before.confidence
    assert reinforced.stability > before.stability
    # 新支持证据边 + last_support_ts 刷新
    new_edges = [
        e
        for e in eng.evidence_edges
        if e.node_id == belief.node_id and e.source_id == "evt-3"
    ]
    assert len(new_edges) == 1
    assert reinforced.last_support_ts is not None
    # 注册表里仍是同一个节点（无新节点）
    assert len(eng.nodes) == 3  # 2 pattern + 1 belief


def test_reinforce_reactivates_weakening():
    eng = _engine()
    belief = _belief(eng)
    eng._nodes[belief.node_id] = replace(
        eng.get_node(belief.node_id), lifecycle_state="weakening"
    )
    reinforced = eng.reinforce(belief.node_id, source_id="evt-3")
    assert reinforced.node_id == belief.node_id
    assert reinforced.lifecycle_state == "active"  # 重新激活


def test_reinforce_reactivates_dormant():
    eng = _engine()
    belief = _belief(eng)
    eng._nodes[belief.node_id] = replace(
        eng.get_node(belief.node_id), lifecycle_state="dormant"
    )
    reinforced = eng.reinforce(belief.node_id, source_id="evt-3")
    assert reinforced.node_id == belief.node_id
    assert reinforced.lifecycle_state == "active"  # 被想起、被重新支持


def test_reinforce_superseded_rejected():
    eng = _engine()
    belief = _belief(eng)
    eng._nodes[belief.node_id] = replace(
        eng.get_node(belief.node_id), lifecycle_state="superseded"
    )
    with pytest.raises(ValueError):
        eng.reinforce(belief.node_id, source_id="evt-3")  # 终态不自动复活


# —— Contradiction ≠ Revision ——


def test_single_contradiction_does_not_change_state():
    eng = _engine()
    belief = _belief(eng)
    eng.record_contradiction(
        belief.node_id, source_id="contra-1", ts="2026-09-10T00:00:00Z"
    )
    node = eng.get_node(belief.node_id)
    # 只累积压力，不改变任何状态
    assert node.lifecycle_state == "active"
    assert node.content == belief.content
    assert node.confidence == belief.confidence
    assert len(node.contradiction_pressure) == 1
    # 一次反例不推翻：supersede 拒绝（1 < 3）
    with pytest.raises(ValueError):
        eng.supersede(
            belief.node_id,
            new_content="世界不危险",
            new_confidence=0.8,
            valence="positive",
            source_ids=[],
        )
    assert eng.get_node(belief.node_id).lifecycle_state == "active"


def test_contradiction_dedup_by_source_and_identity():
    eng = _engine()
    belief = _belief(eng)
    eng.record_contradiction(
        belief.node_id, source_id="c1", ts="2026-09-10T00:00:00Z"
    )
    eng.record_contradiction(
        belief.node_id, source_id="c1", ts="2026-09-11T00:00:00Z"
    )  # 同 source → 幂等
    node = eng.get_node(belief.node_id)
    assert len(node.contradiction_pressure) == 1


def test_contradiction_pressure_keeps_references_not_body():
    eng = _engine()
    belief = _belief(eng)
    eng.record_contradiction(
        belief.node_id, source_id="contra-1", ts="2026-09-10T00:00:00Z"
    )
    rec = eng.get_node(belief.node_id).contradiction_pressure[0]
    assert rec.source_id == "contra-1"  # 只留来源引用
    assert not hasattr(rec, "content")  # 不复制正文


# —— SUPERSEDE 转换 ——


def _accumulate_contradictions(eng, node_id, n, start_day=1):
    for i in range(n):
        day = start_day + i
        eng.record_contradiction(
            node_id,
            source_id=f"contra-{day}",  # 用 day 编号保证全局唯一（跨调用不撞车）
            ts=f"2026-09-{day:02d}T00:00:00Z",
        )


def test_supersede_creates_new_node_and_freezes_old():
    eng = _engine()
    belief = _belief(eng, valence="negative")
    _accumulate_contradictions(eng, belief.node_id, 3)  # 3 条独立矛盾，跨 3 天

    new = eng.supersede(
        belief.node_id,
        new_content="世界没那么危险",
        new_confidence=0.8,
        valence="positive",
        source_ids=[],
    )
    # 新节点：同层 node_type + lineage 复用既有字段族
    assert new.node_type == "belief"
    assert new.parent_node_id == belief.node_id
    assert new.lineage_depth == belief.lineage_depth + 1
    assert new.lineage_path == f"{belief.lineage_path}/{new.node_id}"
    assert new.lifecycle_state == "active"
    # 旧节点 → SUPERSEDED（冻结，永久保留不删）
    old = eng.get_node(belief.node_id)
    assert old.lifecycle_state == "superseded"
    assert old.superseded_by == new.node_id
    assert old.content == belief.content  # 本体保留
    assert old.lineage_path == belief.lineage_path  # lineage 保留
    # 证据留痕：旧节点有效边 valid_until_ts 标记
    old_edges = [e for e in eng.evidence_edges if e.node_id == belief.node_id]
    assert old_edges
    assert all(e.valid_until_ts is not None for e in old_edges)
    # 新节点证据边回指旧支持证据 + 触发矛盾的新证据源
    new_sources = {e.source_id for e in eng.evidence_edges if e.node_id == new.node_id}
    assert {"evt-1", "evt-2"} <= new_sources  # 回指同一批原始 source_id


def test_supersede_insufficient_evidence_rejected():
    eng = _engine()
    belief = _belief(eng)
    _accumulate_contradictions(eng, belief.node_id, 2)  # 2 < 3
    with pytest.raises(ValueError):
        eng.supersede(
            belief.node_id,
            new_content="x",
            new_confidence=0.8,
            valence="positive",
            source_ids=[],
        )
    assert eng.get_node(belief.node_id).lifecycle_state == "active"


def test_supersede_rejects_single_day_noise():
    eng = _engine()
    belief = _belief(eng)
    # 同一天 3 条矛盾 → 跨时间不一致（防单日噪声）→ 拒绝
    with pytest.raises(ValueError):
        eng.supersede(
            belief.node_id,
            new_content="x",
            new_confidence=0.8,
            valence="positive",
            source_ids=["c1", "c2", "c3"],
            ts="2026-09-10T00:00:00Z",
        )
    assert eng.get_node(belief.node_id).lifecycle_state == "active"


def test_supersede_superseded_node_rejected():
    eng = _engine()
    belief = _belief(eng)
    _accumulate_contradictions(eng, belief.node_id, 3)
    eng.supersede(
        belief.node_id,
        new_content="x",
        new_confidence=0.8,
        valence="positive",
        source_ids=[],
    )
    with pytest.raises(ValueError):
        eng.supersede(
            belief.node_id,
            new_content="y",
            new_confidence=0.9,
            valence="positive",
            source_ids=[],
        )  # 终态不可再超驰


# —— Forgetting = lifecycle transition（不是 delete）——


def test_forgetting_is_lifecycle_transition_not_delete():
    eng = _engine()
    belief = _belief(eng)
    now = _days_later(belief.created_ts, 40)
    eng.evaluate_lifecycle(now_ts=now)  # → WEAKENING
    eng.evaluate_lifecycle(now_ts=now)  # → DORMANT
    node = eng.get_node(belief.node_id)
    assert node.lifecycle_state == "dormant"
    # 节点本体 + lineage + 证据链永不物理删除
    assert node.content == belief.content
    assert node.lineage_path == belief.lineage_path
    assert node.parent_node_id == belief.parent_node_id
    edges = [e for e in eng.evidence_edges if e.node_id == belief.node_id]
    assert edges  # 证据链仍在
    assert all(e.source_id in ("evt-1", "evt-2") for e in edges)


# —— evaluate_lifecycle：衰减链 + 锚点（M5.13 语义）——


def test_evaluate_weakening_after_t_weaken():
    eng = _engine()
    belief = _belief(eng)
    now = _days_later(belief.created_ts, 10)  # 10 天 ≥ 7
    changed = eng.evaluate_lifecycle(now_ts=now)
    # belief（及其 pattern 前身）都失去支持 → 转移；断言 belief 已 WEAKENING
    assert belief.node_id in [n.node_id for n in changed]
    assert eng.get_node(belief.node_id).lifecycle_state == "weakening"


def test_evaluate_dormant_after_t_dormant():
    eng = _engine()
    belief = _belief(eng)
    now = _days_later(belief.created_ts, 10)
    eng.evaluate_lifecycle(now_ts=now)  # → WEAKENING
    now2 = _days_later(belief.created_ts, 40)  # 40 天 ≥ 30
    eng.evaluate_lifecycle(now_ts=now2)  # → DORMANT
    assert eng.get_node(belief.node_id).lifecycle_state == "dormant"


def test_evaluate_transitions_one_step_at_a_time():
    eng = _engine()
    belief = _belief(eng)
    now = _days_later(belief.created_ts, 40)  # 同时超 T_weaken 与 T_dormant
    eng.evaluate_lifecycle(now_ts=now)
    assert eng.get_node(belief.node_id).lifecycle_state == "weakening"  # 只一级
    eng.evaluate_lifecycle(now_ts=now)
    assert eng.get_node(belief.node_id).lifecycle_state == "dormant"  # 第二级


def test_grace_period_for_never_supported_node():
    eng = _engine()
    node = _node(
        node_id="n-grace",
        content="从未被支持",
        created_ts="2026-09-01T00:00:00Z",
        last_support_ts=None,  # 无支持证据历史 → created + grace 兜底
    )
    eng._nodes[node.node_id] = node
    # grace 期内（created + 2 天 < 3 天）→ 不衰减
    assert eng.evaluate_lifecycle(now_ts="2026-09-03T00:00:00Z") == []
    # grace 后（created + 10 天 → 锚点 created+3 → 7 天 ≥ T_weaken）→ WEAKENING
    eng.evaluate_lifecycle(now_ts="2026-09-11T00:00:00Z")
    assert eng.get_node("n-grace").lifecycle_state == "weakening"


def test_old_node_with_recent_support_stays_active():
    # old ≠ outdated：创建很久前，但最近仍有新支持证据 → 保持 ACTIVE
    eng = _engine()
    belief = _belief(eng)
    now = _days_later(belief.created_ts, 100)
    eng.reinforce(belief.node_id, source_id="evt-recent", ts=now)
    eng.evaluate_lifecycle(now_ts=now)
    # belief 最近被支持 → 保持 ACTIVE（pattern 前身无新支持，可衰减，不影响本断言）
    assert eng.get_node(belief.node_id).lifecycle_state == "active"


def test_bad_timestamp_skipped_no_crash():
    eng = _engine()
    node = _node(
        node_id="n-bad",
        content="坏时间戳",
        created_ts="not-a-timestamp",
        last_support_ts="also-bad",
    )
    eng._nodes[node.node_id] = node
    assert eng.evaluate_lifecycle(now_ts="2026-09-30T00:00:00Z") == []  # 跳过，不 crash


# —— essence 保守（锁死层）——


def test_essence_exempt_from_auto_decay():
    eng = _essence_engine()
    essence = _essence(eng)
    now = _days_later(essence.created_ts, 100)
    eng.evaluate_lifecycle(now_ts=now)
    # essence 节点豁免自动衰减（pattern 前身非 essence，可衰减，不影响本断言）
    assert eng.get_node(essence.node_id).lifecycle_state == "active"


def test_essence_supersede_threshold_highest():
    eng = _essence_engine()
    essence = _essence(eng, valence="positive")
    # 3 条矛盾（其他层已够，essence 不够：3 < 5）
    _accumulate_contradictions(eng, essence.node_id, 3)
    with pytest.raises(ValueError):
        eng.supersede(
            essence.node_id,
            new_content="x",
            new_confidence=0.8,
            valence="negative",
            source_ids=[],
        )
    # 补到 5 条 → 才允许
    _accumulate_contradictions(eng, essence.node_id, 2, start_day=4)
    new = eng.supersede(
        essence.node_id,
        new_content="我变得冷漠了",
        new_confidence=0.8,
        valence="negative",
        source_ids=[],
    )
    assert new.node_type == "essence"
    assert eng.get_node(essence.node_id).lifecycle_state == "superseded"


def test_essence_supersede_requires_valence_reversal():
    eng = _essence_engine()
    essence = _essence(eng, valence="positive")
    _accumulate_contradictions(eng, essence.node_id, 5)
    # 无 valence 反转（positive → positive）→ 拒绝
    with pytest.raises(ValueError):
        eng.supersede(
            essence.node_id,
            new_content="x",
            new_confidence=0.8,
            valence="positive",
            source_ids=[],
        )


def test_essence_supersede_requires_confidence_delta():
    eng = _essence_engine(confidence=0.3)
    essence = _essence(eng, valence="positive")
    _accumulate_contradictions(eng, essence.node_id, 5)
    # delta = 0.5 - 0.3 = 0.2 ≤ 0.3 → 拒绝
    with pytest.raises(ValueError):
        eng.supersede(
            essence.node_id,
            new_content="x",
            new_confidence=0.5,
            valence="negative",
            source_ids=[],
        )


def test_essence_reconsideration_candidate_channel():
    eng = _essence_engine()
    essence = _essence(eng, valence="positive")
    # 3 条矛盾累积 → 标记 reconsideration candidate（待复核，非自动改写）
    _accumulate_contradictions(eng, essence.node_id, 3)
    node = eng.get_node(essence.node_id)
    assert node.reconsideration_candidate is True
    assert node.lifecycle_state == "active"  # 未自动 SUPERSEDE
    # 未达 5 条 → supersede 仍拒绝
    with pytest.raises(ValueError):
        eng.supersede(
            essence.node_id,
            new_content="x",
            new_confidence=0.8,
            valence="negative",
            source_ids=[],
        )


def test_non_essence_never_gets_reconsideration_flag():
    eng = _engine()
    belief = _belief(eng)
    _accumulate_contradictions(eng, belief.node_id, 3)
    assert eng.get_node(belief.node_id).reconsideration_candidate is False


# —— Trace additive 扩展 ——


def test_event_types_include_lifecycle_events():
    assert {
        "node_state_changed",
        "node_superseded",
        "essence_reconsideration_candidate",
    } <= EVENT_TYPES
    # 既有 5 事件语义 0 变更（仍在词汇表内）
    assert {
        "node_created",
        "node_elevated",
        "node_revised",
        "edge_decayed",
        "node_forgotten",
    } <= EVENT_TYPES


def test_trace_node_state_changed(tmp_path):
    w = ElevationTraceWriter(tmp_path / "trace.jsonl")
    eng = InternalizingEngine(StubElevationLLM(), trace_writer=w)
    belief = _belief(eng)
    now = _days_later(belief.created_ts, 10)
    eng.evaluate_lifecycle(now_ts=now)
    records = read_trace(w.path)
    sc = [r for r in records if r["event_type"] == "node_state_changed"]
    # 只断言 belief 的转移事件（pattern 前身也会转移，不影响本断言）
    sc_belief = [r for r in sc if r["node_id"] == belief.node_id]
    assert len(sc_belief) == 1
    assert sc_belief[0]["lifecycle_state_before"] == "active"
    assert sc_belief[0]["lifecycle_state_after"] == "weakening"
    assert sc_belief[0]["anchor_ts"] is not None
    assert sc_belief[0]["reason"] == "no_new_support_evidence"


def test_trace_node_superseded(tmp_path):
    w = ElevationTraceWriter(tmp_path / "trace.jsonl")
    eng = InternalizingEngine(StubElevationLLM(), trace_writer=w)
    belief = _belief(eng)
    _accumulate_contradictions(eng, belief.node_id, 3)
    new = eng.supersede(
        belief.node_id,
        new_content="x",
        new_confidence=0.8,
        valence="positive",
        source_ids=[],
    )
    records = read_trace(w.path)
    sup = [r for r in records if r["event_type"] == "node_superseded"]
    assert len(sup) == 1
    assert sup[0]["old_node_id"] == belief.node_id
    assert sup[0]["new_node_id"] == new.node_id
    assert sup[0]["lineage_path"] == new.lineage_path
    assert set(sup[0]["contradiction_evidence_ids"]) == {
        "contra-1",
        "contra-2",
        "contra-3",
    }


def test_trace_essence_reconsideration_candidate(tmp_path):
    w = ElevationTraceWriter(tmp_path / "trace.jsonl")
    eng = InternalizingEngine(
        StubElevationLLM(keyword_map={"温柔": "essence"}), trace_writer=w
    )
    essence = _essence(eng)
    _accumulate_contradictions(eng, essence.node_id, 3)
    records = read_trace(w.path)
    rc = [r for r in records if r["event_type"] == "essence_reconsideration_candidate"]
    assert len(rc) == 1
    assert rc[0]["node_id"] == essence.node_id
    assert rc[0]["reason"] == "contradiction_pressure_accumulated"


def test_trace_reinforce_reactivation_emits_state_changed(tmp_path):
    w = ElevationTraceWriter(tmp_path / "trace.jsonl")
    eng = InternalizingEngine(StubElevationLLM(), trace_writer=w)
    belief = _belief(eng)
    eng._nodes[belief.node_id] = replace(
        eng.get_node(belief.node_id), lifecycle_state="dormant"
    )
    eng.reinforce(belief.node_id, source_id="evt-3")
    records = read_trace(w.path)
    sc = [r for r in records if r["event_type"] == "node_state_changed"]
    assert len(sc) == 1
    assert sc[0]["lifecycle_state_before"] == "dormant"
    assert sc[0]["lifecycle_state_after"] == "active"
    assert sc[0]["reason"] == "reinforced_by_new_support_evidence"
