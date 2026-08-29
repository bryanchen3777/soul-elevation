"""输入契约（ElevationInput）单元测试。"""

from dataclasses import FrozenInstanceError

import pytest

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


def test_input_required_fields():
    i = _inp()
    assert i.event_type == "diary:night"
    assert i.content == "我重视自由"
    assert i.source_id == "evt-1"
    assert i.source_type == "inner_life_event"
    assert i.timestamp == "2026-08-29T00:00:00Z"


def test_input_provenance_defaults_to_empty_dict():
    assert _inp().provenance == {}


def test_input_provenance_carries_metadata():
    i = _inp(provenance={"trigger_type": "diary:night", "source_system": "inner_life"})
    assert i.provenance["trigger_type"] == "diary:night"


def test_input_source_type_validation():
    with pytest.raises(ValueError):
        _inp(source_type="unknown")


def test_input_supports_all_source_types():
    for st in ("v1_memory", "sage_fact", "inner_life_event"):
        assert _inp(source_type=st).source_type == st


def test_input_is_frozen():
    i = _inp()
    with pytest.raises(FrozenInstanceError):
        i.content = "改写"
