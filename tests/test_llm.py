"""ElevationLLM 接口 + 桩实现单元测试。"""

import pytest

from soul_elevation.llm import Classification, ElevationLLM, StubElevationLLM


def test_classification_is_tuple_like():
    c = Classification(node_type="belief", content="c", confidence=0.7)
    node_type, content, confidence = c
    assert (node_type, content, confidence) == ("belief", "c", 0.7)


def test_elevation_llm_exposes_classify_signature():
    # 结构类型接口：方法名 + 参数序 = (content, provenance, prior_node_type)。
    assert hasattr(ElevationLLM, "classify")
    params = list(ElevationLLM.classify.__annotations__)
    assert params == ["content", "provenance", "prior_node_type", "return"]


def test_stub_defaults_to_prior_node_type():
    stub = StubElevationLLM(confidence=0.6)
    result = stub.classify("c", {}, "belief")
    assert result == Classification("belief", "c", 0.6)


def test_stub_keyword_map_overrides_prior():
    stub = StubElevationLLM(keyword_map={"自由": "value"})
    result = stub.classify("我重视自由", {}, "belief")
    assert result.node_type == "value"
    assert result.content == "我重视自由"
    assert result.confidence == 0.5


def test_stub_rejects_invalid_confidence():
    with pytest.raises(ValueError):
        StubElevationLLM(confidence=2.0)
    with pytest.raises(ValueError):
        StubElevationLLM(confidence=-0.1)
