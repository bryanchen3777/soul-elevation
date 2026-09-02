"""soul_elevation — 记忆升华层（memory elevation layer）。

独立 repo，零 Soul OS 依赖。第一阶段：数据模型（ElevationNode / EvidenceEdge）
+ 输入契约（ElevationInput）+ 引擎接口（ElevationEngine）。
第二阶段：内化映射（先验映射表 + LLM 后验接口/桩 + InternalizingEngine.consume）。
第三阶段：reconsolidation 修订（InternalizingEngine.revise）+ 升华式遗忘
（InternalizingEngine.decay / forget）。
第四阶段：可审计闭环（自有 elevation_trace.jsonl：ElevationTraceWriter +
read_trace / rebuild_lineage）。
"""

from .engine import (
    DEFAULT_ELEVATE_MIN_EVIDENCE,
    ElevationEngine,
    InternalizingEngine,
)
from .llm import Classification, ElevationLLM, StubElevationLLM
from .models import (
    SOUL_NODE_TYPES,
    VALID_LIFECYCLE_STATES,
    VALID_NODE_TYPES,
    VALID_SOURCE_TYPES,
    VALID_VALENCES,
    ContradictionRecord,
    ElevationInput,
    ElevationNode,
    EvidenceEdge,
    evidence_key,
    new_id,
)
from .prior import (
    CATEGORY_PRIOR_TABLE,
    CATEGORY_TRIGGER_TYPES,
    DEFAULT_PRIOR,
    LIFECYCLE_ESSENCE_CONFIDENCE_DELTA,
    LIFECYCLE_ESSENCE_N_SUPERSEDE,
    LIFECYCLE_ESSENCE_RECONSIDERATION_MIN_EVIDENCE,
    LIFECYCLE_GRACE_DAYS,
    LIFECYCLE_N_SUPERSEDE,
    LIFECYCLE_SUPERSEDE_MIN_DAYS_SPREAD,
    LIFECYCLE_T_DORMANT_DAYS,
    LIFECYCLE_T_WEAKEN_DAYS,
    PRIOR_TABLE,
    resolve_prior,
)
from .trace import (
    DEFAULT_TRACE_PATH,
    EVENT_TYPES,
    ElevationTraceWriter,
    build_event,
    read_trace,
    rebuild_lineage,
    records_by_node,
)

__all__ = [
    "ElevationInput",
    "ElevationNode",
    "EvidenceEdge",
    "ContradictionRecord",
    "ElevationEngine",
    "InternalizingEngine",
    "ElevationLLM",
    "Classification",
    "StubElevationLLM",
    "VALID_NODE_TYPES",
    "SOUL_NODE_TYPES",
    "VALID_SOURCE_TYPES",
    "VALID_VALENCES",
    "VALID_LIFECYCLE_STATES",
    "DEFAULT_ELEVATE_MIN_EVIDENCE",
    "evidence_key",
    "new_id",
    "PRIOR_TABLE",
    "CATEGORY_PRIOR_TABLE",
    "CATEGORY_TRIGGER_TYPES",
    "DEFAULT_PRIOR",
    "resolve_prior",
    "LIFECYCLE_T_WEAKEN_DAYS",
    "LIFECYCLE_T_DORMANT_DAYS",
    "LIFECYCLE_GRACE_DAYS",
    "LIFECYCLE_N_SUPERSEDE",
    "LIFECYCLE_SUPERSEDE_MIN_DAYS_SPREAD",
    "LIFECYCLE_ESSENCE_N_SUPERSEDE",
    "LIFECYCLE_ESSENCE_CONFIDENCE_DELTA",
    "LIFECYCLE_ESSENCE_RECONSIDERATION_MIN_EVIDENCE",
    "ElevationTraceWriter",
    "DEFAULT_TRACE_PATH",
    "EVENT_TYPES",
    "build_event",
    "read_trace",
    "rebuild_lineage",
    "records_by_node",
]

__version__ = "0.1.0"
