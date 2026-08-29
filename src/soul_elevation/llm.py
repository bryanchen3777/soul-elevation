"""LLM 后验接口 + 桩实现（第二阶段）。

「源类型给先验 + LLM 依 content/stance 做后验」的**后验**部分。
``ElevationLLM`` 是 ``typing.Protocol``，LLM 通过构造注入（**不硬编码任何 provider**），
保持零依赖。本阶段只提供确定性桩 ``StubElevationLLM`` 供测试。
"""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple, Optional, Protocol

from .models import NodeType


class Classification(NamedTuple):
    """LLM 后验结果：最终维度 + 归一化内容 + 置信度（0-1）。"""

    node_type: NodeType
    content: str
    confidence: float


class ElevationLLM(Protocol):
    """LLM 后验接口（结构类型）。

    任何实现 ``classify(content, provenance, prior_node_type)`` 的对象都满足此接口；
    不绑定任何 provider。实现应依 content / provenance 的语义在先验基调上做后验归类。
    """

    def classify(
        self,
        content: str,
        provenance: Mapping[str, Any],
        prior_node_type: NodeType,
    ) -> Classification:
        """依 content/provenance 在先验基调上做后验归类。

        返回最终 ``node_type``（可覆盖先验）、归一化 ``content``、``confidence``（0-1）。
        """
        ...


class StubElevationLLM:
    """确定性桩：默认沿用先验 node_type；``keyword_map`` 命中 content 关键词时覆盖。

    用于测试「先验 → 后验」链路，并能演示「语义内容覆盖源类型基调」的后验能力。
    """

    def __init__(
        self,
        *,
        confidence: float = 0.5,
        keyword_map: Optional[Mapping[str, NodeType]] = None,
    ) -> None:
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"confidence must be in [0.0, 1.0], got {confidence!r}")
        self.confidence = confidence
        self.keyword_map = dict(keyword_map or {})

    def classify(
        self,
        content: str,
        provenance: Mapping[str, Any],
        prior_node_type: NodeType,
    ) -> Classification:
        node_type = prior_node_type
        for keyword, mapped in self.keyword_map.items():
            if keyword in content:
                node_type = mapped
                break
        return Classification(node_type=node_type, content=content, confidence=self.confidence)
