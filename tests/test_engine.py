"""引擎接口（ElevationEngine）单元测试。"""

import inspect
from typing import get_args, get_origin, get_type_hints

import pytest

from soul_elevation.engine import ElevationEngine
from soul_elevation.models import ElevationInput, ElevationNode


def test_engine_is_abstract():
    assert inspect.isabstract(ElevationEngine)


def test_engine_cannot_be_instantiated():
    with pytest.raises(TypeError):
        ElevationEngine()  # 抽象方法 consume 未实现，不可实例化


def test_consume_signature_fixed():
    # 接口签名定死：consume(input: ElevationInput) -> list[ElevationNode]
    params = list(inspect.signature(ElevationEngine.consume).parameters)
    assert params == ["self", "input"]

    hints = get_type_hints(ElevationEngine.consume)
    assert hints["input"] is ElevationInput
    ret = hints["return"]
    assert get_origin(ret) is list
    assert get_args(ret) == (ElevationNode,)


class _StubEngine(ElevationEngine):
    def consume(self, input):
        return []


def test_stub_engine_returns_list_of_nodes():
    eng = _StubEngine()
    inp = ElevationInput(
        event_type="diary:night",
        content="c",
        source_id="s",
        source_type="v1_memory",
        timestamp="2026-08-29T00:00:00Z",
    )
    assert eng.consume(inp) == []
    assert isinstance(eng.consume(inp), list)
