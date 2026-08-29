"""soul_elevation 引擎接口（第一阶段）。

第一阶段**只定义接口 + 数据模型**；``consume`` 的完整升华逻辑
（活动→灵魂维度内化映射、reconsolidation 式信念修订、升华式遗忘、可审计闭环）
留后续阶段，但接口签名已定死。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from .models import ElevationInput, ElevationNode


class ElevationEngine(ABC):
    """升华引擎接口。

    只读消费 ``ElevationInput``，产出 ``ElevationNode``（+ 证据链，后续阶段）。
    零 Soul OS 依赖：Soul OS 通过 adapter 把 InnerLifeEvent + Memory 映射成
    ``ElevationInput`` 后喂进来，本接口不感知上游任何类型。
    """

    @abstractmethod
    def consume(self, input: ElevationInput) -> List[ElevationNode]:
        """消费一条归一化输入，产出（可能的）升华节点列表。

        接口签名定死：``consume(input: ElevationInput) -> list[ElevationNode]``。
        第一阶段不实现完整逻辑。
        """
        raise NotImplementedError
