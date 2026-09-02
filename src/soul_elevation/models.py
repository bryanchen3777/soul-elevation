"""soul_elevation 数据模型 + 输入契约（第一阶段）。

严格照设计文档（MEMORY-ELEVATION-DESIGN）§4 落地。数据模型为**自研**：
因为要携带「证据链 + 置信度 + 版本因果树」，现成系统（MemGPT 扁平 persona、
Mem0 fact 级、GA lossy 摘要）都丢证据链 / 置信度，会失真。

零 Soul OS 依赖：本模块不 import 任何 Soul OS 类型；证据正文不复制进节点，
只保留证据边索引（source_id）回查原文。

**Soul destination（v1 边界，SE-2）**：当前升华目标只有 meaning/identity 维度
（belief / value / trait / essence）；competence（knowledge / capability）
reserved / out of scope，本票不实作。**Pattern 是合法终态**：可长期存在，
即使永远不 elevate；未 elevate ≠ 失败（不是升华候车室）。

**Evidence Independence（SE-1）**：``evidence_key = (source_id, event_identity)``，
``event_identity = novelty_id | inner_life_event_id | explicit event_id``。
同一 source_id 或同一 event identity → 同一份独立证据（计 1）。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

# —— 词汇表（frozen，作为运行时校验依据）——
# pattern 是第 5 类节点：consolidation 输出（候选），**非灵魂结构**。
# 灵魂结构 = belief / value / trait / essence（SOUL_NODE_TYPES），只能经 elevate()
# 由 pattern 升华产生，单一事件不直接产灵魂结构（Consolidation ≠ Elevation 边界）。
NodeType = Literal["belief", "value", "trait", "essence", "pattern"]
Valence = Literal["positive", "negative", "neutral"]
# world_event 是 additive 扩展：Soul OS 的 world→elevation 直通 adapter 把
# WorldEvent（news/weather/calendar）直接映射成 ElevationInput，不经 InnerLifeEvent。
# 既有 3 值（v1_memory / sage_fact / inner_life_event）语义 0 变更。
SourceType = Literal["v1_memory", "sage_fact", "inner_life_event", "world_event"]
# SE-5 Lifecycle：durable soul structure 四态（状态机，不是动作）。
# ACTIVE → WEAKENING → DORMANT → SUPERSEDED；显式持久状态字段，additive schema。
LifecycleState = Literal["active", "weakening", "dormant", "superseded"]

VALID_NODE_TYPES: frozenset = frozenset(
    {"belief", "value", "trait", "essence", "pattern"}
)
# 灵魂结构子集：elevate() 的合法升华目标维度（pattern 自身不可作为升华目标）。
SOUL_NODE_TYPES: frozenset = frozenset({"belief", "value", "trait", "essence"})
VALID_VALENCES: frozenset = frozenset({"positive", "negative", "neutral"})
VALID_SOURCE_TYPES: frozenset = frozenset(
    {"v1_memory", "sage_fact", "inner_life_event", "world_event"}
)
VALID_LIFECYCLE_STATES: frozenset = frozenset(
    {"active", "weakening", "dormant", "superseded"}
)


def new_id() -> str:
    """生成 32-hex 节点/边 id。

    采用 uuid4().hex 语义，但独立命名空间（不复用上游的 generate_event_id）。
    """
    return uuid.uuid4().hex


def evidence_key(source_id: str, event_identity: Optional[str]) -> tuple:
    """独立证据键（SE-1 Evidence Independence contract）。

    ``evidence_key = (source_id, event_identity)``；``event_identity`` 取
    ``novelty_id | inner_life_event_id | explicit event_id`` 之一（可为 None）。

    同一 source_id + 同一 event identity → 同一份独立证据（计 1）：
    重复 ingest / 同一 source 重送 / 同一事件两笔 memory record /
    一次 InnerLifeEvent 抽多条 fact / 同一场 weather·news 轮询连打 → 都计 1。
    不同 source 且不同 event identity（或 event identity 缺失）→ 才独立。
    """
    return (source_id, event_identity)


def _ensure_unit_interval(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or not (0.0 <= float(value) <= 1.0):
        raise ValueError(f"{name} must be in [0.0, 1.0], got {value!r}")


@dataclass(frozen=True)
class ContradictionRecord:
    """矛盾证据压力记录（SE-5，Contradiction ≠ Revision）。

    只留**来源引用 + 时间**（source_id / ts / event_identity / provenance_ref），
    **不复制证据正文**——原文永远留在上游 Memory / Fact / Event 里可回查。
    写入 ``ElevationNode.contradiction_pressure`` 累积器；累积本身**不改变任何
    状态**（压力 ≠ 改变），达 SUPERSEDE 阈值才触发转换。
    """

    source_id: str                    # 回指上游 memory_id / fact_id / event_id
    ts: str                           # 矛盾证据出现时刻（ISO 8601 UTC）
    event_identity: Optional[str] = None  # SE-1 event identity（独立性判定）
    provenance_ref: Optional[str] = None  # 触发事件 id（join 回 trace）


@dataclass(frozen=True)
class ElevationNode:
    """信念/价值/性格/内涵/模式节点（五类统一 schema）。

    **关键保真设计**：节点本身**不含**摘要化的「证据正文」——证据正文永远留在
    证据边指向的 Memory / Fact / Event 里。节点只带 ``content`` + ``confidence`` +
    ``stability`` + 证据边索引，避免「摘要化导致证据丢失」这一现有系统的共性失真。

    **Consolidation ≠ Elevation 边界**：``node_type="pattern"`` 是 consolidation
    输出（候选，非灵魂结构）；``candidate_node_type`` 只对 pattern 节点有意义，
    承载 LLM 后验的**候选解释**维度（interpretation，非 truth）——证据累积达阈值
    后由 ``elevate()`` 升华成 belief/value/trait/essence 时才成为灵魂事实。
    """

    node_id: str                      # 32-hex（独立命名空间）
    node_type: NodeType               # belief / value / trait / essence / pattern
    content: str                      # 自然语言命题
    confidence: float                 # 0.0-1.0，由证据链聚合而来
    stability: float                  # 0.0-1.0，被 reconsolidation 次数 + 一致证据数决定
    valence: Valence                  # 情感极性
    agent_id: str                     # 归属 agent（灵魂本体）
    # —— 因果树（复用 InnerLifeEvent.parent_event_id 语义：一父、改写不覆盖）——
    parent_node_id: Optional[str]     # 修订因果父节点（改写 = 新节点引用旧节点）
    lineage_depth: int                # 根 = 0，父 + 1
    lineage_path: str                 # "parent_path/own_id"，反范式化
    # —— 审计 ——
    created_ts: str                   # ISO 8601 UTC
    provenance_ref: Optional[str]     # 触发本节点的上游事件 id（join 回 trace）
    # —— Consolidation ≠ Elevation：LLM 后验候选解释（仅 pattern 节点使用）——
    candidate_node_type: Optional[NodeType] = None  # 候选灵魂维度（interpretation，非 truth）
    # —— SE-5 Lifecycle（additive 扩展，既有字段语义 0 变更）——
    lifecycle_state: LifecycleState = "active"  # 四态之一，默认 active（创建即 ACTIVE）
    last_support_ts: Optional[str] = None       # 最后一条仍有效支持证据的 ts（decay 锚点，§8）
    contradiction_pressure: tuple = ()          # 矛盾证据压力累积器（ContradictionRecord 元组）
    superseded_by: Optional[str] = None        # 反向 lineage：本节点被哪个新节点取代（仅 SUPERSEDED）
    reconsideration_candidate: bool = False    # 仅 essence 使用：新证据长期累积但未达转换门槛

    def __post_init__(self) -> None:
        if self.node_type not in VALID_NODE_TYPES:
            raise ValueError(
                f"invalid node_type {self.node_type!r}; expected one of {sorted(VALID_NODE_TYPES)}"
            )
        if self.candidate_node_type is not None and self.candidate_node_type not in SOUL_NODE_TYPES:
            raise ValueError(
                f"invalid candidate_node_type {self.candidate_node_type!r}; "
                f"expected one of {sorted(SOUL_NODE_TYPES)}"
            )
        if self.valence not in VALID_VALENCES:
            raise ValueError(
                f"invalid valence {self.valence!r}; expected one of {sorted(VALID_VALENCES)}"
            )
        if self.lifecycle_state not in VALID_LIFECYCLE_STATES:
            raise ValueError(
                f"invalid lifecycle_state {self.lifecycle_state!r}; "
                f"expected one of {sorted(VALID_LIFECYCLE_STATES)}"
            )
        _ensure_unit_interval("confidence", self.confidence)
        _ensure_unit_interval("stability", self.stability)
        if self.lineage_depth < 0:
            raise ValueError(f"lineage_depth must be >= 0, got {self.lineage_depth}")


@dataclass(frozen=True)
class EvidenceEdge:
    """证据边：回指原始记忆 / 事件，带双时序有效区间。

    双时序（valid_from_ts / valid_until_ts）采用 Zep/Graphiti 语义：
    ``valid_until_ts=None`` 表示仍有效；非 None 表示已被后续证据取代
    （reconsolidation 留痕）。``source_id`` 永远保留，原文仍可回查。
    """

    edge_id: str
    node_id: str                      # → ElevationNode
    source_type: SourceType           # v1_memory / sage_fact / inner_life_event
    source_id: str                    # memory_id / fact_id / event_id（回查原文）
    agent_id: str
    weight: float                     # 本证据对节点的贡献权重（0-1）
    # —— 时序有效区间（双时序）——
    valid_from_ts: str                # 证据生效时刻
    valid_until_ts: Optional[str]     # None = 仍有效；非 None = 已被取代（留痕）
    # —— 因果回溯 ——
    inner_life_event_id: Optional[str]  # 触发本证据的 canonical event
    trigger_type: str                 # 证据来源的活动类型（diary:night / world:news …）

    def __post_init__(self) -> None:
        if self.source_type not in VALID_SOURCE_TYPES:
            raise ValueError(
                f"invalid source_type {self.source_type!r}; "
                f"expected one of {sorted(VALID_SOURCE_TYPES)}"
            )
        _ensure_unit_interval("weight", self.weight)


@dataclass(frozen=True)
class ElevationInput:
    """输入契约（归一化输入记录）。

    这是 **adapter seam** 的接口：Soul OS 未来提供 adapter 把 InnerLifeEvent +
    Memory 映射成 ElevationInput，本 repo 不依赖 Soul OS 任何类型。``provenance``
    用 ``dict`` 承载上游 provenance 元数据（trigger_type / source_system / extras 等）。
    """

    event_type: str                   # 活动类型（如 diary:night / world:news_event）
    content: str                      # 归一化后的原文内容
    source_id: str                    # 回指上游 memory_id / fact_id / event_id
    source_type: SourceType           # v1_memory / sage_fact / inner_life_event
    timestamp: str                    # ISO 8601 UTC
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source_type not in VALID_SOURCE_TYPES:
            raise ValueError(
                f"invalid source_type {self.source_type!r}; "
                f"expected one of {sorted(VALID_SOURCE_TYPES)}"
            )
