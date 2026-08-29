"""可审计闭环（第四阶段）单元测试：trace writer / reader / 引擎审计事件。"""

import pytest

from soul_elevation.engine import InternalizingEngine
from soul_elevation.llm import StubElevationLLM
from soul_elevation.models import ElevationInput
from soul_elevation.trace import (
    DEFAULT_TRACE_PATH,
    EVENT_TYPES,
    ElevationTraceWriter,
    build_event,
    read_trace,
    rebuild_lineage,
    records_by_node,
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


def _writer(tmp_path, name="trace.jsonl"):
    return ElevationTraceWriter(tmp_path / name)


class _NullLogger:
    """静默 logger，供失败隔离测试保持输出干净。"""

    def warning(self, *args, **kwargs):
        pass


# —— 事件类型 + 字段契约 ——


def test_event_types_cover_required_set():
    assert {"node_created", "node_revised", "edge_decayed", "node_forgotten"} <= EVENT_TYPES


def test_build_event_has_required_fields():
    rec = build_event("node_created", "n1")
    for field in ("ts", "event_type", "node_id", "parent_node_id", "source_id", "provenance_ref"):
        assert field in rec


def test_build_event_rejects_unknown_type():
    with pytest.raises(ValueError):
        build_event("no_such_event", "n1")


# —— writer：append-only + 读回 ——


def test_writer_appends_and_reads_back(tmp_path):
    w = _writer(tmp_path)
    assert w.write(build_event("node_created", "n1")) is True
    assert w.write(build_event("node_revised", "n2", parent_node_id="n1")) is True

    records = read_trace(w.path)
    assert [r["event_type"] for r in records] == ["node_created", "node_revised"]
    assert [r["node_id"] for r in records] == ["n1", "n2"]


def test_writer_is_append_only(tmp_path):
    w = _writer(tmp_path)
    w.write(build_event("node_created", "n1"))
    w.write(build_event("node_created", "n2"))
    records = read_trace(w.path)
    assert [r["node_id"] for r in records] == ["n1", "n2"]  # 两行都在，未被覆盖


def test_writer_creates_parent_directory(tmp_path):
    w = ElevationTraceWriter(tmp_path / "nested" / "dir" / "trace.jsonl")
    assert w.write(build_event("node_created", "n1")) is True
    assert len(read_trace(w.path)) == 1


def test_writer_default_path():
    assert ElevationTraceWriter().path == DEFAULT_TRACE_PATH


# —— 失败隔离 ——


def test_writer_failure_returns_false_no_raise(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("this is a file, not a directory")
    w = ElevationTraceWriter(blocker / "sub" / "trace.jsonl", logger=_NullLogger())
    # 父目录是文件 → 无法建目录 → 写失败，但绝不 raise。
    assert w.write(build_event("node_created", "n1")) is False


def test_engine_main_path_not_blocked_by_trace_failure(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("this is a file, not a directory")
    writer = ElevationTraceWriter(blocker / "sub" / "trace.jsonl", logger=_NullLogger())
    eng = InternalizingEngine(StubElevationLLM(), trace_writer=writer)
    # 主路径照常：consume 不 raise、节点照常产出并可检索。
    node = eng.consume(_inp())[0]
    assert eng.get_node(node.node_id).node_id == node.node_id


# —— 三要素可溯 ——


def test_source_id_traceable(tmp_path):
    w = _writer(tmp_path)
    eng = InternalizingEngine(StubElevationLLM(), trace_writer=w)
    eng.consume(_inp(source_type="v1_memory", source_id="mem-42"))
    rec = read_trace(w.path)[0]
    assert rec["event_type"] == "node_created"
    assert rec["source_id"] == "mem-42"
    assert rec["evidence_source_ids"] == ["mem-42"]


def test_provenance_ref_traceable(tmp_path):
    w = _writer(tmp_path)
    eng = InternalizingEngine(StubElevationLLM(), trace_writer=w)
    eng.consume(
        _inp(
            source_type="v1_memory",
            source_id="mem-42",
            provenance={"inner_life_event_id": "evt-9"},
        )
    )
    rec = read_trace(w.path)[0]
    # provenance_ref 回指触发事件，可 join 上游 InnerLifeEvent。
    assert rec["provenance_ref"] == "evt-9"


def test_parent_node_id_causal_chain_rebuild(tmp_path):
    w = _writer(tmp_path)
    eng = InternalizingEngine(StubElevationLLM(), trace_writer=w)
    n0 = eng.consume(_inp())[0]
    n1 = eng.revise(n0.node_id, "第一次修订", source_id="evt-2", inner_life_event_id="evt-2")
    n2 = eng.revise(n1.node_id, "第二次修订")

    records = read_trace(w.path)
    # 修订事件带 parent_node_id + 置信度快照。
    rev = records_by_node(records, n2.node_id)[0]
    assert rev["event_type"] == "node_revised"
    assert rev["parent_node_id"] == n1.node_id
    assert rev["confidence_before"] == n1.confidence
    assert rev["confidence_after"] == n2.confidence

    # 按 parent_node_id 反向重建因果链：根 → … → 目标。
    chain = rebuild_lineage(records, n2.node_id)
    assert [r["node_id"] for r in chain] == [n0.node_id, n1.node_id, n2.node_id]


# —— 生命周期审计事件全覆盖 ——


def test_decay_emits_edge_decayed(tmp_path):
    w = _writer(tmp_path)
    eng = InternalizingEngine(StubElevationLLM(), trace_writer=w)
    node = eng.consume(_inp(source_type="v1_memory", source_id="mem-7"))[0]
    eng.decay(node.node_id, decay_rate=0.5)
    decayed = [r for r in read_trace(w.path) if r["event_type"] == "edge_decayed"]
    assert len(decayed) == 1
    assert decayed[0]["source_id"] == "mem-7"
    assert decayed[0]["weight_after"] < decayed[0]["weight_before"]


def test_forget_emits_node_forgotten_and_edge_decayed(tmp_path):
    w = _writer(tmp_path)
    eng = InternalizingEngine(StubElevationLLM(), trace_writer=w)
    n1 = eng.consume(_inp(content="今天被雨淋", source_id="mem-rain"))[0]
    n2 = eng.consume(_inp(content="今天忘带伞", source_id="mem-umbrella"))[0]
    sem = eng.forget([n1.node_id, n2.node_id], "我容易忽略天气细节")

    records = read_trace(w.path)
    types = [r["event_type"] for r in records]
    assert types.count("node_forgotten") == 1

    forgotten = [r for r in records if r["event_type"] == "node_forgotten"][0]
    assert forgotten["node_id"] == sem.node_id
    assert set(forgotten["forgotten_node_ids"]) == {n1.node_id, n2.node_id}
    assert set(forgotten["evidence_source_ids"]) == {"mem-rain", "mem-umbrella"}
    # forget 内部对每个情景节点做 decay → 2 条 edge_decayed。
    assert types.count("edge_decayed") == 2


# —— reader 边界 ——


def test_read_trace_missing_path_returns_empty(tmp_path):
    assert read_trace(tmp_path / "does-not-exist.jsonl") == []


def test_engine_without_writer_still_works():
    # 不注入 writer 时保持向后兼容：主路径照常，不产生 trace 副作用。
    eng = InternalizingEngine(StubElevationLLM())
    node = eng.consume(_inp())[0]
    assert eng.get_node(node.node_id).node_id == node.node_id
