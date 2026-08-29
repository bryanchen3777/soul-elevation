"""soul_elevation — 记忆升华层（memory elevation layer）。

独立 repo，零 Soul OS 依赖。第一阶段：数据模型（ElevationNode / EvidenceEdge）
+ 输入契约（ElevationInput）+ 引擎接口（ElevationEngine）。
"""

from .engine import ElevationEngine
from .models import (
    VALID_NODE_TYPES,
    VALID_SOURCE_TYPES,
    VALID_VALENCES,
    ElevationInput,
    ElevationNode,
    EvidenceEdge,
    new_id,
)

__all__ = [
    "ElevationInput",
    "ElevationNode",
    "EvidenceEdge",
    "ElevationEngine",
    "VALID_NODE_TYPES",
    "VALID_SOURCE_TYPES",
    "VALID_VALENCES",
    "new_id",
]

__version__ = "0.1.0"
