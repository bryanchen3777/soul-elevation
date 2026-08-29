"""SE-1 Evidence Independence：min_evidence 必须是 2 个**独立** evidence。

Independence contract：``evidence_key = (source_id, event_identity)``，
``event_identity = novelty_id | inner_life_event_id | explicit event_id``。
同一 source_id **或** 同一 event identity → 同一份独立证据（计 1）。

锁死的案例：重复 ingest / 同一 source 重送 / 同一事件两笔 memory record /
一次 InnerLifeEvent 抽多条 fact / 同一场 weather·news 轮询连打 → 都计 1；
不同 source 且不同 event identity → 才独立。0 embedding / similarity 代码。
"""

import pytest

from soul_elevation.engine import InternalizingEngine
from soul_elevation.llm import StubElevationLLM
from soul_elevation.models import ElevationInput, evidence_key


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


# —— evidence_key 结构 ——


def test_evidence_key_structure():
    assert evidence_key("src-1", "evt-9") == ("src-1", "evt-9")
    assert evidence_key("src-1", None) == ("src-1", None)
    # 同一 source_id + 同一 event identity → 同一 key（计 1）。
    assert evidence_key("src-1", "evt-9") == evidence_key("src-1", "evt-9")
    # 不同 source 或不同 event identity → 不同 key。
    assert evidence_key("src-1", "evt-9") != evidence_key("src-2", "evt-9")
    assert evidence_key("src-1", "evt-9") != evidence_key("src-1", "evt-10")


# —— 同一事件重送 / 重复 ingest / 同一 source 重送 → 计 1 ——


def test_same_event_resent_twice_does_not_elevate():
    # 同一事件重送 2 次（同 source_id + 同 event identity）→ 1 份独立证据 → 不过门槛。
    eng = _engine()
    p1 = eng.consume(_inp(source_id="evt-1"))[0]
    eng.consume(_inp(source_id="evt-1"))  # 重送同一事件
    with pytest.raises(ValueError):
        eng.elevate(p1.node_id)


def test_duplicate_ingest_does_not_elevate():
    # 重复 ingest 同一输入 → 2 笔 record 但 1 份独立证据。
    eng = _engine()
    p1 = eng.consume(_inp())[0]
    eng.consume(_inp())
    with pytest.raises(ValueError):
        eng.elevate(p1.node_id)


def test_same_source_resent_does_not_elevate():
    # 同一 source 重送（同 source_id，event identity 缺失）→ 计 1。
    eng = _engine()
    p1 = eng.consume(
        _inp(event_type="world:news_event", content="世界很危险", source_id="news-1")
    )[0]
    eng.consume(
        _inp(event_type="world:news_event", content="世界很危险", source_id="news-1")
    )
    with pytest.raises(ValueError):
        eng.elevate(p1.node_id)


# —— 同一事件多笔 record / 一次事件抽多条 fact → 计 1 ——


def test_one_inner_life_event_multiple_facts_does_not_elevate():
    # 一次 InnerLifeEvent 抽多条 fact：不同 source_id、同一 event identity → 计 1。
    eng = _engine()
    p1 = eng.consume(
        _inp(
            event_type="memory_fact",
            content="我重视自由",
            source_id="fact-1",
            source_type="sage_fact",
            provenance={"inner_life_event_id": "evt-9"},
        )
    )[0]
    eng.consume(
        _inp(
            event_type="memory_fact",
            content="我重视自由",
            source_id="fact-2",
            source_type="sage_fact",
            provenance={"inner_life_event_id": "evt-9"},
        )
    )
    with pytest.raises(ValueError):
        eng.elevate(p1.node_id)


def test_same_event_two_memory_records_does_not_elevate():
    # 同一事件两笔 memory record：不同 source_id、同一 event identity → 计 1。
    eng = _engine()
    p1 = eng.consume(
        _inp(
            event_type="memory_fact",
            content="我重视自由",
            source_id="mem-1",
            source_type="v1_memory",
            provenance={"inner_life_event_id": "evt-9"},
        )
    )[0]
    eng.consume(
        _inp(
            event_type="memory_fact",
            content="我重视自由",
            source_id="mem-2",
            source_type="v1_memory",
            provenance={"inner_life_event_id": "evt-9"},
        )
    )
    with pytest.raises(ValueError):
        eng.elevate(p1.node_id)


def test_weather_polling_repeated_does_not_elevate():
    # 同一场 weather 轮询连打：同 source_id + 同 event identity → 计 1。
    eng = _engine()
    p1 = eng.consume(
        _inp(
            event_type="world:weather_event",
            content="今天有雨",
            source_id="weather-0800",
            source_type="world_event",
            provenance={"event_id": "weather-0800"},
        )
    )[0]
    eng.consume(
        _inp(
            event_type="world:weather_event",
            content="今天有雨",
            source_id="weather-0800",
            source_type="world_event",
            provenance={"event_id": "weather-0800"},
        )
    )
    with pytest.raises(ValueError):
        eng.elevate(p1.node_id)


# —— 独立事件 → 才可 elevate ——


def test_two_independent_events_elevate():
    eng = _engine()
    p1 = eng.consume(_inp(source_id="evt-1"))[0]
    eng.consume(_inp(source_id="evt-2"))
    soul = eng.elevate(p1.node_id)
    assert soul.node_type == "value"


def test_different_sources_same_event_identity_do_not_elevate():
    # 不同 source 但同一 event identity → 仍计 1（同一事件）。
    eng = _engine()
    p1 = eng.consume(
        _inp(
            source_id="mem-1",
            source_type="v1_memory",
            provenance={"inner_life_event_id": "evt-9"},
        )
    )[0]
    eng.consume(
        _inp(
            source_id="mem-2",
            source_type="v1_memory",
            provenance={"inner_life_event_id": "evt-9"},
        )
    )
    with pytest.raises(ValueError):
        eng.elevate(p1.node_id)


def test_different_sources_different_event_identity_elevate():
    # 不同 source + 不同 event identity → 独立 → 可 elevate。
    eng = _engine()
    p1 = eng.consume(
        _inp(
            source_id="mem-1",
            source_type="v1_memory",
            provenance={"inner_life_event_id": "evt-9"},
        )
    )[0]
    eng.consume(
        _inp(
            source_id="mem-2",
            source_type="v1_memory",
            provenance={"inner_life_event_id": "evt-10"},
        )
    )
    soul = eng.elevate(p1.node_id)
    assert soul.node_type == "value"


# —— event_identity 三种来源：novelty_id | inner_life_event_id | explicit event_id ——


def test_novelty_id_used_as_event_identity():
    # event_identity 可来自 novelty_id（provenance）。
    eng = _engine()
    p1 = eng.consume(
        _inp(
            source_id="nov-1",
            source_type="v1_memory",
            provenance={"novelty_id": "novelty-77"},
        )
    )[0]
    eng.consume(
        _inp(
            source_id="nov-2",
            source_type="v1_memory",
            provenance={"novelty_id": "novelty-77"},
        )
    )
    with pytest.raises(ValueError):
        eng.elevate(p1.node_id)


def test_explicit_event_id_used_as_event_identity():
    # event_identity 可来自显式 event_id（provenance）。
    eng = _engine()
    p1 = eng.consume(
        _inp(
            source_id="w-1",
            source_type="world_event",
            provenance={"event_id": "weather-0800"},
        )
    )[0]
    eng.consume(
        _inp(
            source_id="w-2",
            source_type="world_event",
            provenance={"event_id": "weather-0800"},
        )
    )
    with pytest.raises(ValueError):
        eng.elevate(p1.node_id)


def test_inner_life_event_source_identity_is_source_id():
    # inner_life_event 源的事件身份即其 source_id（canonical event）。
    eng = _engine()
    p1 = eng.consume(_inp(source_id="evt-1"))[0]
    eng.consume(_inp(source_id="evt-1"))
    with pytest.raises(ValueError):
        eng.elevate(p1.node_id)
