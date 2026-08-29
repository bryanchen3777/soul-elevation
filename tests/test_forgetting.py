"""升华式遗忘（decay / forget）单元测试。"""

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


# —— decay：情景细节淡化 ——


def test_decay_fades_edge_without_deleting_source():
    eng = _engine()
    node = eng.consume(_inp(source_type="v1_memory", source_id="mem-1"))[0]
    faded = eng.decay(node.node_id)
    assert len(faded) == 1
    e = faded[0]
    assert e.valid_until_ts is not None  # 已淡化
    assert e.source_id == "mem-1"        # 不删源，仍可回查
    assert e.source_type == "v1_memory"
    assert e.weight < 1.0                # 降级


def test_decay_reduces_weight_by_rate():
    eng = _engine()
    node = eng.consume(_inp())[0]
    faded = eng.decay(node.node_id, decay_rate=0.5)[0]
    assert faded.weight == 0.5  # 1.0 * 0.5


def test_decay_is_visible_in_engine_edges():
    eng = _engine()
    node = eng.consume(_inp())[0]
    eng.decay(node.node_id)
    e = [x for x in eng.evidence_edges if x.node_id == node.node_id][0]
    assert e.valid_until_ts is not None
    assert e.source_id == "evt-1"  # source_id 仍保留


def test_decay_only_affects_still_valid_edges():
    eng = _engine()
    node = eng.consume(_inp())[0]
    eng.decay(node.node_id)               # 第一次淡化
    faded_again = eng.decay(node.node_id)  # 已淡化，不应再动
    assert faded_again == []


def test_decay_rejects_invalid_rate():
    eng = _engine()
    node = eng.consume(_inp())[0]
    for bad in (0.0, 1.0, 1.5, -0.5):
        with pytest.raises(ValueError):
            eng.decay(node.node_id, decay_rate=bad)


# —— forget：语义核心强化 ——


def test_forget_produces_semantic_core_node():
    eng = _engine()
    n1 = eng.consume(_inp(content="今天被雨淋", source_id="mem-rain"))[0]
    n2 = eng.consume(_inp(content="今天忘带伞", source_id="mem-umbrella"))[0]
    n3 = eng.consume(_inp(content="今天错过车", source_id="mem-bus"))[0]
    sem = eng.forget([n1.node_id, n2.node_id, n3.node_id], "我容易忽略天气细节")
    assert sem.content == "我容易忽略天气细节"
    assert sem.parent_node_id is None  # 新抽象 = 根，非因果改写
    assert sem.lineage_depth == 0
    assert sem.lineage_path == sem.node_id


def test_forget_fades_episodic_edges():
    eng = _engine()
    n1 = eng.consume(_inp(source_id="mem-a"))[0]
    n2 = eng.consume(_inp(source_id="mem-b"))[0]
    eng.forget([n1.node_id, n2.node_id], "抽象命题")
    episodic_edges = [e for e in eng.evidence_edges if e.node_id in (n1.node_id, n2.node_id)]
    assert episodic_edges
    assert all(e.valid_until_ts is not None for e in episodic_edges)


def test_forget_boosts_confidence_and_stability():
    eng = _engine()
    n1 = eng.consume(_inp())[0]
    n2 = eng.consume(_inp())[0]
    sem = eng.forget([n1.node_id, n2.node_id], "抽象")
    assert sem.confidence > n1.confidence
    assert sem.stability > n1.stability
    assert sem.stability > n2.stability


def test_forget_semantic_edges_point_back_to_sources():
    eng = _engine()
    n1 = eng.consume(_inp(source_id="mem-a"))[0]
    n2 = eng.consume(_inp(source_id="mem-b"))[0]
    sem = eng.forget([n1.node_id, n2.node_id], "抽象")
    sem_edges = [e for e in eng.evidence_edges if e.node_id == sem.node_id]
    assert sorted(e.source_id for e in sem_edges) == ["mem-a", "mem-b"]


def test_forget_defaults_node_type_to_most_common():
    eng = _engine()
    n1 = eng.consume(_inp(event_type="world:news_event", content="世界很危险"))[0]  # belief
    n2 = eng.consume(_inp(event_type="world:news_event", content="世界很复杂"))[0]  # belief
    sem = eng.forget([n1.node_id, n2.node_id], "抽象")
    assert sem.node_type == "belief"


def test_forget_requires_nonempty():
    eng = _engine()
    with pytest.raises(ValueError):
        eng.forget([], "抽象")
