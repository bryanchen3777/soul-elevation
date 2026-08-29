"""soul_elevation 引擎：抽象接口 + 第二阶段「内化映射」+ Pattern 中间层（Consolidation ≠ Elevation）+ 第三阶段「reconsolidation 修订 / 升华式遗忘」+ 第四阶段「可审计闭环」。

第一阶段定义抽象接口 ``ElevationEngine``（``consume`` 签名定死）。
第二阶段在其上实现 ``InternalizingEngine``：先验映射（prior）→ LLM 后验（posterior）
→ 产出 ``ElevationNode`` + 对应 ``EvidenceEdge``。

**Consolidation ≠ Elevation 边界（Pattern 中间层）**：``consume`` 对单一事件只产
``pattern`` 节点（候选，consolidation 输出），LLM 后验结果作为 ``candidate_node_type``
候选解释（interpretation，非 truth）；证据累积达阈值后由 ``elevate`` 升华成
belief/value/trait/essence（灵魂结构）。单一事件不直接产生灵魂结构。

第三阶段新增三个节点生命周期方法：

- ``revise``：reconsolidation 式信念修订（检索激活→重估→改写 N'→留痕），
  复用注入的 ``ElevationLLM`` 做重估，不硬编码 provider。
- ``decay``：情景细节淡化（证据边按 Ebbinghaus 式衰减降级 + ``valid_until_ts``
  标记，不删源）。
- ``forget``：升华式遗忘（情景淡化 + 语义核心强化，非删除）。

第四阶段：注入 ``ElevationTraceWriter``（可选）时，上述生命周期动作各自向
**自有** ``elevation_trace.jsonl`` 追加审计事件（``node_created`` /
``node_elevated`` / ``node_revised`` / ``edge_decayed`` / ``node_forgotten``）。
trace 写失败只告警、不阻断主路径（失败隔离）。

零 Soul OS 依赖：Soul OS 通过 adapter 把 InnerLifeEvent + Memory 映射成
``ElevationInput`` 后喂进来，本模块不感知上游任何类型。

**SE-1 Evidence Independence**：``elevate`` 的 ``min_evidence`` 数的是**独立**
evidence（``evidence_key = (source_id, event_identity)``），不是 record 数。
同一 source_id 或同一 event identity → 计 1；不同 source 且不同 event identity
→ 才独立。0 embedding / similarity 代码。

**SE-2 Pattern Terminal Semantics**：Pattern 是**合法终态**，不是升华候车室。
``consume()`` → candidate Pattern；``elevate()`` → accepted Soul-level node。
Pattern 可长期存在，即使永远不 elevate；未 elevate ≠ 失败。``consume()``
不直接写 Soul state（只写本引擎自有注册表）。WorldEvent → Pattern 允许，
WorldEvent → trait/essence 直接路径 v1 禁止。Soul destination 只有
meaning/identity（belief/value/trait/essence），competence reserved/out of scope。

**SE-3 Lineage vs Evidence**：两图职责锁死。Lineage = How did this node evolve?
（``parent_node_id`` / ``lineage_depth`` / ``lineage_path``）；Evidence = What
supports this node?（``EvidenceEdge`` 回指原始 source）。N1→N2（reconsolidation）
= lineage ≠ evidence；即使 N2 吸收 N1 的 evidence，N1 本身也**不是** N2 的
evidence。``check_invariants()`` 程式内断言：证据边 source_id 绝不指向
ElevationNode（lineage 节点不是证据）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from typing import Dict, List, Mapping, Optional, Sequence

from .llm import ElevationLLM
from .models import (
    SOUL_NODE_TYPES,
    VALID_NODE_TYPES,
    VALID_VALENCES,
    ElevationInput,
    ElevationNode,
    EvidenceEdge,
    NodeType,
    SourceType,
    Valence,
    new_id,
)
from .prior import resolve_prior
from .trace import ElevationTraceWriter, build_event

DEFAULT_AGENT_ID = "default"
DEFAULT_DECAY_RATE = 0.5
DEFAULT_CONFIDENCE_BOOST = 0.2
DEFAULT_STABILITY_BOOST = 0.3

# —— Consolidation ≠ Elevation 边界（Pattern 中间层）——
# 单一事件 → pattern（候选，consolidation 输出）；证据累积达阈值后才 elevate 成
# 灵魂结构（belief/value/trait/essence）。默认阈值 N=2，可配。
DEFAULT_ELEVATE_MIN_EVIDENCE = 2

# —— essence 保守边界（MEMORY-LIFECYCLE §3.2）——
# essence 修订高门槛：仅当 valence 反转 + ≥2 条新独立证据 + confidence delta 超阈值
# 才允许 reconsolidation 改写；否则只 reinforce（stability/confidence 微升）。
ESSENCE_REVISE_MIN_EVIDENCE = 2
ESSENCE_REVISE_CONFIDENCE_DELTA = 0.3
DEFAULT_REINFORCE_CONFIDENCE_BOOST = 0.1
DEFAULT_REINFORCE_STABILITY_BOOST = 0.1


def _utcnow_iso() -> str:
    """当前 UTC 时刻的 ISO 8601 字符串（用于 created_ts / valid_from_ts）。"""
    return datetime.now(timezone.utc).isoformat()


def _clamp_unit(value: float) -> float:
    """把数值夹到 [0.0, 1.0]。"""
    return max(0.0, min(1.0, value))


def _bump(value: float, delta: float) -> float:
    """在 [0.0, 1.0] 内提升 value（+delta 后夹紧）。"""
    return _clamp_unit(value + delta)


def _most_common_node_type(nodes: Sequence[ElevationNode]) -> NodeType:
    """多数派 node_type（并列取出现顺序里先到的那个）。"""
    counts = Counter(n.node_type for n in nodes)
    return counts.most_common(1)[0][0]


def _valence_reversed(old: Valence, new: Valence) -> bool:
    """valence 反转：极性翻转（positive ↔ negative）；neutral 不算反转。"""
    return (old == "positive" and new == "negative") or (
        old == "negative" and new == "positive"
    )


def _resolve_event_identity(input: ElevationInput) -> Optional[str]:
    """解析输入的事件身份（event identity，SE-1）。

    ``event_identity = novelty_id | inner_life_event_id | explicit event_id``
    （按此优先级取 provenance 中第一个非空值）；``inner_life_event`` 源的事件
    身份即其 ``source_id``（canonical event）。返回 None 表示无事件身份
    （此时独立性只由 source_id 决定）。
    """
    if input.source_type == "inner_life_event":
        return input.source_id
    for key in ("novelty_id", "inner_life_event_id", "event_id"):
        value = input.provenance.get(key)
        if value:
            return value
    return None


def _count_independent_evidence(edges: Sequence[EvidenceEdge]) -> int:
    """数**独立** evidence（SE-1 Evidence Independence）。

    Independence contract：``evidence_key = (source_id, event_identity)``。
    同一 source_id **或** 同一 event identity → 同一份独立证据（计 1）：

    - 重复 ingest / 同一 source 重送 → 同 source_id → 1
    - 同一事件两笔 memory record / 一次 InnerLifeEvent 抽多条 fact /
      同一场 weather·news 轮询连打 → 同 event identity → 1

    不同 source **且** 不同 event identity（或 event identity 缺失）→ 才独立。
    0 embedding / similarity：只做集合成员判断。
    """
    seen_sources: set = set()
    seen_events: set = set()
    count = 0
    for e in edges:
        if e.source_id in seen_sources:
            continue
        if e.inner_life_event_id is not None and e.inner_life_event_id in seen_events:
            continue
        seen_sources.add(e.source_id)
        if e.inner_life_event_id is not None:
            seen_events.add(e.inner_life_event_id)
        count += 1
    return count


class ElevationEngine(ABC):
    """升华引擎接口。

    只读消费 ``ElevationInput``，产出 ``ElevationNode``（+ 证据链，后续阶段）。
    零 Soul OS 依赖：Soul OS 通过 adapter 把 InnerLifeEvent + Memory 映射成
    ``ElevationInput`` 后喂进来，本接口不感知上游任何类型。
    """

    @abstractmethod
    def consume(self, input: ElevationInput) -> List[ElevationNode]:
        """消费一条归一化输入，产出（可能的）pattern 候选节点列表。

        接口签名定死：``consume(input: ElevationInput) -> list[ElevationNode]``。
        单一事件只产 ``pattern`` 节点（consolidation 输出，候选）；灵魂结构
        （belief/value/trait/essence）须经 ``elevate`` 证据累积后升华。
        """
        raise NotImplementedError


class InternalizingEngine(ElevationEngine):
    """第二阶段：内化映射引擎 + Pattern 中间层 + 第三阶段：节点生命周期（修订 / 遗忘）。

    消费一条 ``ElevationInput``：确定性先验映射 → 注入的 LLM 后验（候选解释）→
    产出 ``pattern`` 节点（``candidate_node_type`` 承载 LLM 后验候选维度）；
    对应 ``EvidenceEdge`` 由本引擎持有（``evidence_edges``），供上层审计回查。
    source_id 回指 input.source_id，双时序 valid_from_ts=now、valid_until_ts=None。

    **Consolidation ≠ Elevation**：``elevate`` 在证据累积达阈值后把 pattern 升华成
    灵魂结构；pattern 保留（因果树 parent 关系 + 证据边 superseded 留痕）。

    第三阶段在此之上维护**节点注册表**（``nodes`` / ``get_node``）并新增：

    - ``revise``：reconsolidation 式修订（改写 = 新节点引用旧节点，旧节点不覆盖）。
    - ``decay`` / ``forget``：升华式遗忘（淡化不删源，语义核心强化）。
    """

    def __init__(
        self,
        llm: ElevationLLM,
        *,
        agent_id: str = DEFAULT_AGENT_ID,
        default_stability: float = 0.0,
        default_valence: Valence = "neutral",
        trace_writer: Optional[ElevationTraceWriter] = None,
    ) -> None:
        self._llm = llm
        self._agent_id = agent_id
        self._default_stability = default_stability
        self._default_valence = default_valence
        self._trace_writer = trace_writer
        self._edges: List[EvidenceEdge] = []
        self._nodes: Dict[str, ElevationNode] = {}

    @property
    def evidence_edges(self) -> List[EvidenceEdge]:
        """本引擎已产出的证据边（consume / revise / forget 累积）。返回副本。"""
        return list(self._edges)

    @property
    def nodes(self) -> List[ElevationNode]:
        """本引擎已产出的节点（consume / revise / forget 累积）。返回副本。"""
        return list(self._nodes.values())

    def get_node(self, node_id: str) -> ElevationNode:
        """按 id 检索节点（reconsolidation「检索激活」）。找不到抛 ``KeyError``。"""
        try:
            return self._nodes[node_id]
        except KeyError:
            raise KeyError(f"node not found: {node_id!r}") from None

    def check_invariants(self) -> None:
        """程式内 invariant（SE-3 Lineage vs Evidence）：两图职责锁死。

        - **Evidence 绝不指向 ElevationNode**：任何 ``EvidenceEdge.source_id``
          不得是注册表里的节点 id——lineage 节点（N1）不是证据，N2 的 supporting
          evidence 永远是原始独立 evidence keys。
        - **Lineage 完整性**：任何节点的 ``parent_node_id``（演化边）必须指向
          注册表里真实存在的节点。

        违反任一 → 抛 ``AssertionError``（fail fast）。consume / elevate /
        revise / forget 每次变更后自动调用。
        """
        node_ids = set(self._nodes)
        for e in self._edges:
            if e.source_id in node_ids:
                raise AssertionError(
                    f"evidence edge {e.edge_id} references elevation node "
                    f"{e.source_id!r} as source — lineage nodes are not evidence"
                )
        for n in self._nodes.values():
            if n.parent_node_id is not None and n.parent_node_id not in node_ids:
                raise AssertionError(
                    f"lineage parent {n.parent_node_id!r} of node {n.node_id!r} "
                    f"not in registry"
                )

    def _emit(
        self,
        event_type: str,
        node_id: str,
        *,
        ts: Optional[str] = None,
        parent_node_id: Optional[str] = None,
        source_id: Optional[str] = None,
        provenance_ref: Optional[str] = None,
        **snapshot: object,
    ) -> None:
        """向注入的 trace writer 追加一条审计事件（未注入 writer 则为 no-op）。

        trace 写失败由 ``ElevationTraceWriter.write`` 内部隔离（不 raise），此处
        不再二次包裹，保证主路径照常。
        """
        if self._trace_writer is None:
            return
        record = build_event(
            event_type,
            node_id,
            ts=ts,
            parent_node_id=parent_node_id,
            source_id=source_id,
            provenance_ref=provenance_ref,
            **snapshot,
        )
        self._trace_writer.write(record)

    def consume(self, input: ElevationInput) -> List[ElevationNode]:
        # 1) 先验映射（deterministic，无 LLM）
        prior = resolve_prior(input.event_type, input.provenance)
        prior_node_type = prior[0]  # 基调（primary prior）

        # 2) LLM 后验 = **候选解释**（interpretation，非 truth）：依 content /
        #    provenance 可覆盖先验，但结果只作为 pattern 的候选维度，不直接成为
        #    灵魂结构（Consolidation ≠ Elevation 边界）。
        classification = self._llm.classify(input.content, input.provenance, prior_node_type)
        candidate_node_type = classification.node_type
        content = classification.content
        confidence = classification.confidence

        if candidate_node_type not in SOUL_NODE_TYPES:
            raise ValueError(
                f"LLM returned invalid candidate node_type {candidate_node_type!r}; "
                f"expected one of {sorted(SOUL_NODE_TYPES)}"
            )

        # 3) 产出 pattern 节点（consolidation 输出，候选）+ edge
        #    （双时序：valid_from_ts=now，valid_until_ts=None）
        #    consume() 只写本引擎自有注册表（_nodes / _edges），**不直接写
        #    Soul state**（SE-2：consume → candidate Pattern，非灵魂结构）。
        now = _utcnow_iso()
        node_id = new_id()

        agent_id = input.provenance.get("agent_id", self._agent_id)
        valence = input.provenance.get("valence", self._default_valence)
        if valence not in VALID_VALENCES:
            raise ValueError(
                f"invalid valence {valence!r}; expected one of {sorted(VALID_VALENCES)}"
            )

        # 触发本节点的 canonical event（event identity，SE-1）：
        # inner_life_event 源即其 source_id；其余按 novelty_id | inner_life_event_id
        # | explicit event_id 优先级取 provenance。
        canonical_event_id = _resolve_event_identity(input)

        node = ElevationNode(
            node_id=node_id,
            node_type="pattern",  # 单一事件只产候选 pattern，不直接产灵魂结构
            candidate_node_type=candidate_node_type,  # LLM 后验候选解释
            content=content,
            confidence=confidence,
            stability=self._default_stability,
            valence=valence,
            agent_id=agent_id,
            parent_node_id=None,
            lineage_depth=0,
            lineage_path=node_id,
            created_ts=now,
            provenance_ref=canonical_event_id,
        )

        edge = EvidenceEdge(
            edge_id=new_id(),
            node_id=node_id,
            source_type=input.source_type,
            source_id=input.source_id,
            agent_id=agent_id,
            weight=1.0,  # 单条证据全额贡献（多证据聚合留后续阶段）
            valid_from_ts=now,
            valid_until_ts=None,
            inner_life_event_id=canonical_event_id,
            trigger_type=input.event_type,
        )
        self._edges.append(edge)
        self._nodes[node_id] = node

        self._emit(
            "node_created",
            node.node_id,
            ts=now,
            parent_node_id=None,
            source_id=input.source_id,
            provenance_ref=node.provenance_ref,
            node_type=node.node_type,
            lineage_depth=node.lineage_depth,
            lineage_path=node.lineage_path,
            confidence_after=node.confidence,
            evidence_source_ids=[input.source_id],
            reason="internalized",
        )

        self.check_invariants()  # SE-3：变更后锁两图职责
        return [node]

    # —— Consolidation ≠ Elevation：pattern → 灵魂结构升华 ——

    def _active_edges(self, node_id: str) -> List[EvidenceEdge]:
        """某节点仍有效（valid_until_ts is None）的证据边。"""
        return [e for e in self._edges if e.node_id == node_id and e.valid_until_ts is None]

    def elevate(
        self,
        pattern_node_id: str,
        *,
        node_type: Optional[NodeType] = None,
        content: Optional[str] = None,
        confidence: Optional[float] = None,
        valence: Optional[Valence] = None,
        min_evidence: int = DEFAULT_ELEVATE_MIN_EVIDENCE,
        source_ids: Optional[Sequence[str]] = None,
        source_type: SourceType = "inner_life_event",
        inner_life_event_id: Optional[str] = None,
        trigger_type: str = "elevation",
    ) -> ElevationNode:
        """把 pattern 升华成灵魂结构（belief/value/trait/essence）。

        **Consolidation ≠ Elevation 边界**：单一事件只产 pattern（候选）；证据累积
        达阈值（默认 ``min_evidence=2``，可配）后，``elevate`` 才把 pattern 升华成
        灵魂结构。升华维度默认取 pattern 的 LLM 后验候选（``candidate_node_type``，
        interpretation），也可显式传入 ``node_type``（如由 prior 表决定）。

        - **独立证据累积（SE-1）**：聚合所有与目标 pattern 同候选维度
          （``candidate_node_type`` 相同）的 pattern 的**有效证据边**
          （``valid_until_ts is None``），按 ``evidence_key = (source_id,
          event_identity)`` 数**独立** evidence（同一 source_id 或同一 event
          identity → 计 1）；独立数 < ``min_evidence`` 时抛 ``ValueError``
          （证据不足，不升华）。
        - **pattern 保留**：升华后 pattern 节点仍在注册表；灵魂节点以
          ``parent_node_id=pattern`` 挂到因果树（改写不覆盖），pattern 的有效证据边
          标记 ``valid_until_ts=now``（superseded 留痕），灵魂节点新建证据边回指
          同一批 ``source_id``（原文仍可回查）。
        - **LLM = interpretation**：LLM 后验只决定候选维度/内容/置信度，是否成为
          灵魂事实由「证据累积 + 阈值」决定，不由单次 LLM 输出直接决定。
        """
        pattern = self.get_node(pattern_node_id)
        if pattern.node_type != "pattern":
            raise ValueError(
                f"elevate requires a pattern node, got node_type {pattern.node_type!r}"
            )
        if not isinstance(min_evidence, int) or min_evidence < 1:
            raise ValueError(f"min_evidence must be a positive int, got {min_evidence!r}")

        # 证据累积：同候选维度的 pattern 组（candidate 为 None 时只算目标自身）。
        candidate = pattern.candidate_node_type
        group = [
            n
            for n in self._nodes.values()
            if n.node_type == "pattern" and n.candidate_node_type == candidate
        ]
        active_edges: List[EvidenceEdge] = []
        for n in group:
            active_edges.extend(self._active_edges(n.node_id))
        # SE-1 Evidence Independence：min_evidence 数的是**独立** evidence
        # （evidence_key = (source_id, event_identity)），不是 record 数。
        independent = _count_independent_evidence(active_edges)
        if independent < min_evidence:
            raise ValueError(
                f"insufficient independent evidence to elevate: "
                f"{independent} < {min_evidence}"
            )

        # 升华维度：显式 node_type（prior 表决定）或 LLM 后验候选（interpretation）。
        resolved_type = node_type or candidate
        if resolved_type not in SOUL_NODE_TYPES:
            raise ValueError(
                f"invalid elevated node_type {resolved_type!r}; "
                f"expected one of {sorted(SOUL_NODE_TYPES)}"
            )
        resolved_content = content if content is not None else pattern.content
        resolved_confidence = (
            confidence if confidence is not None else pattern.confidence
        )
        resolved_valence = valence if valence is not None else pattern.valence
        if resolved_valence not in VALID_VALENCES:
            raise ValueError(
                f"invalid valence {resolved_valence!r}; expected one of {sorted(VALID_VALENCES)}"
            )

        now = _utcnow_iso()
        soul_node_id = new_id()
        soul_node = ElevationNode(
            node_id=soul_node_id,
            node_type=resolved_type,
            content=resolved_content,
            confidence=resolved_confidence,
            stability=_bump(pattern.stability, DEFAULT_STABILITY_BOOST),  # 证据累积 → 更稳定
            valence=resolved_valence,
            agent_id=pattern.agent_id,
            parent_node_id=pattern.node_id,  # 因果树 parent 关系，pattern 保留
            lineage_depth=pattern.lineage_depth + 1,
            lineage_path=f"{pattern.lineage_path}/{soul_node_id}",
            created_ts=now,
            provenance_ref=inner_life_event_id or pattern.provenance_ref,
        )
        self._nodes[soul_node_id] = soul_node

        # 留痕：被聚合 pattern 的有效证据边 superseded（不删 source_id）。
        for n in group:
            self._supersede_edges(n.node_id, now)

        # 灵魂节点证据边：回指被聚合 pattern 的 source_id（原文可回查）。
        for edge in active_edges:
            self._edges.append(
                EvidenceEdge(
                    edge_id=new_id(),
                    node_id=soul_node_id,
                    source_type=edge.source_type,
                    source_id=edge.source_id,
                    agent_id=soul_node.agent_id,
                    weight=edge.weight,
                    valid_from_ts=now,
                    valid_until_ts=None,
                    inner_life_event_id=inner_life_event_id or edge.inner_life_event_id,
                    trigger_type=edge.trigger_type,
                )
            )
        # 额外新证据源（可选，如调用方补交的独立证据）。
        for sid in source_ids or ():
            self._edges.append(
                EvidenceEdge(
                    edge_id=new_id(),
                    node_id=soul_node_id,
                    source_type=source_type,
                    source_id=sid,
                    agent_id=soul_node.agent_id,
                    weight=1.0,
                    valid_from_ts=now,
                    valid_until_ts=None,
                    inner_life_event_id=inner_life_event_id,
                    trigger_type=trigger_type,
                )
            )

        self._emit(
            "node_elevated",
            soul_node.node_id,
            ts=now,
            parent_node_id=pattern.node_id,
            source_id=None,
            provenance_ref=soul_node.provenance_ref,
            node_type=soul_node.node_type,
            lineage_depth=soul_node.lineage_depth,
            lineage_path=soul_node.lineage_path,
            confidence_after=soul_node.confidence,
            evidence_source_ids=[e.source_id for e in active_edges],
            pattern_node_ids=[n.node_id for n in group],
            reason="evidence_threshold_met",
        )

        self.check_invariants()  # SE-3：变更后锁两图职责
        return soul_node

    # —— 第三阶段：reconsolidation 式信念修订 ——

    def _reinforce(self, node: ElevationNode) -> ElevationNode:
        """只 reinforce（不改写）：原地提升 stability/confidence，不产生新因果节点。

        essence 保守边界（MEMORY-LIFECYCLE §3.2）：essence 几乎不 revise，只 reinforce。
        与 ``revise`` 的「改写 = 新节点引用旧节点」不同，reinforce **不换 node_id、
        不改 lineage**，仅把 stability/confidence 微升后原地替换注册表里的节点。
        """
        reinforced = replace(
            node,
            confidence=_bump(node.confidence, DEFAULT_REINFORCE_CONFIDENCE_BOOST),
            stability=_bump(node.stability, DEFAULT_REINFORCE_STABILITY_BOOST),
        )
        self._nodes[node.node_id] = reinforced
        return reinforced

    def _essence_revise_allowed(
        self,
        old: ElevationNode,
        *,
        valence: Optional[Valence],
        evidence_ids: Sequence[str],
        new_confidence: Optional[float],
    ) -> bool:
        """essence 修订高门槛（MEMORY-LIFECYCLE §3.2）：三条件**同时**满足才允许改写。

        (a) valence 反转（极性翻转）；(b) ≥2 条新独立证据；(c) confidence delta 超阈值。
        任一不满足 → 只 reinforce，不 reconsolidation 改写。
        """
        if valence is None or not _valence_reversed(old.valence, valence):
            return False
        if len(evidence_ids) < ESSENCE_REVISE_MIN_EVIDENCE:
            return False
        if new_confidence is None or (
            new_confidence - old.confidence
        ) <= ESSENCE_REVISE_CONFIDENCE_DELTA:
            return False
        return True

    def revise(
        self,
        node_id: str,
        new_content: str,
        new_confidence: Optional[float] = None,
        *,
        valence: Optional[Valence] = None,
        provenance: Optional[Mapping[str, object]] = None,
        source_id: Optional[str] = None,
        source_ids: Optional[Sequence[str]] = None,
        source_type: SourceType = "inner_life_event",
        inner_life_event_id: Optional[str] = None,
        trigger_type: str = "reconsolidation",
    ) -> ElevationNode:
        """reconsolidation 式信念修订：检索激活 → 重估 → 改写 N' → 留痕。

        - **检索激活**：按 ``node_id`` 取旧节点（``get_node``）。
        - **重估**：把新旧证据一起喂给注入的 ``ElevationLLM``（``classify``；
          旧节点上下文经 ``provenance`` 注入），不硬编码任何 provider；
          ``new_confidence`` 提供时覆盖 LLM 返回的置信度（供确定性测试）。
        - **改写**：生成新节点 ``N'``：``parent_node_id=旧节点``、
          ``lineage_depth=旧+1``、``lineage_path=旧路径/新 id``；**不覆盖旧节点**。
        - **留痕**：旧节点仍有效的证据边标记 ``valid_until_ts=now``（不删 source_id）；
          若提供 ``source_id``/``source_ids``，为 ``N'`` 追加新证据边（``provenance_ref``
          指向触发事件）。

        **essence 保守边界**：``old.node_type == "essence"`` 时，仅当 valence 反转 +
        ≥2 条新独立证据 + confidence delta 超阈值才改写；否则只 ``_reinforce``
        （stability/confidence 微升，不换 node_id、不改 lineage）。
        """
        old = self.get_node(node_id)

        # 收集新证据源（source_id 单条 + source_ids 多条，向后兼容）。
        evidence_ids: List[str] = []
        if source_id is not None:
            evidence_ids.append(source_id)
        if source_ids is not None:
            evidence_ids.extend(source_ids)

        # essence 保守边界：默认只 reinforce，仅高门槛才改写。
        if old.node_type == "essence" and not self._essence_revise_allowed(
            old,
            valence=valence,
            evidence_ids=evidence_ids,
            new_confidence=new_confidence,
        ):
            reinforced = self._reinforce(old)
            self._emit(
                "node_revised",
                reinforced.node_id,
                ts=_utcnow_iso(),
                parent_node_id=None,
                source_id=source_id,
                provenance_ref=inner_life_event_id,
                node_type=reinforced.node_type,
                lineage_depth=reinforced.lineage_depth,
                lineage_path=reinforced.lineage_path,
                confidence_before=old.confidence,
                confidence_after=reinforced.confidence,
                evidence_source_ids=evidence_ids,
                reason="reinforced_only",
            )
            self.check_invariants()  # SE-3：变更后锁两图职责
            return reinforced

        # 重估：旧节点上下文经 provenance 注入，供 LLM 做新旧对比。
        # 先验基调取旧节点的**候选维度**（pattern 的 candidate_node_type 或灵魂结构
        # 自身的 node_type），保证 LLM 后验落在灵魂维度上（Consolidation ≠ Elevation）。
        merged: Dict[str, object] = dict(provenance or {})
        merged.setdefault("old_node_id", old.node_id)
        merged.setdefault("old_content", old.content)
        merged.setdefault("old_confidence", old.confidence)
        prior_for_llm = (
            old.candidate_node_type if old.node_type == "pattern" else old.node_type
        )
        classification = self._llm.classify(new_content, merged, prior_for_llm)
        candidate_node_type = classification.node_type
        content = classification.content
        confidence = new_confidence if new_confidence is not None else classification.confidence

        if candidate_node_type not in SOUL_NODE_TYPES:
            raise ValueError(
                f"LLM returned invalid candidate node_type {candidate_node_type!r}; "
                f"expected one of {sorted(SOUL_NODE_TYPES)}"
            )
        resolved_valence = old.valence if valence is None else valence
        if resolved_valence not in VALID_VALENCES:
            raise ValueError(
                f"invalid valence {resolved_valence!r}; "
                f"expected one of {sorted(VALID_VALENCES)}"
            )

        now = _utcnow_iso()
        new_node_id = new_id()
        # Consolidation ≠ Elevation：修订 pattern 产出仍是候选 pattern（candidate 更新）；
        # 修订既有灵魂结构（belief/value/trait/essence）才产出同层灵魂结构（reconsolidation）。
        is_pattern = old.node_type == "pattern"
        new_node = ElevationNode(
            node_id=new_node_id,
            node_type="pattern" if is_pattern else candidate_node_type,
            candidate_node_type=candidate_node_type if is_pattern else None,
            content=content,
            confidence=confidence,
            stability=_bump(old.stability, DEFAULT_STABILITY_BOOST),  # 被想起 → 强化
            valence=resolved_valence,
            agent_id=old.agent_id,
            parent_node_id=old.node_id,  # 改写 = 新节点引用旧节点
            lineage_depth=old.lineage_depth + 1,
            lineage_path=f"{old.lineage_path}/{new_node_id}",
            created_ts=now,
            provenance_ref=inner_life_event_id,
        )
        self._nodes[new_node_id] = new_node

        # 留痕：旧节点仍有效的证据边 valid_until_ts = now（不删 source_id）。
        self._supersede_edges(old.node_id, now)

        # 触发修订的新证据边（每条新证据源一条边）。
        for eid in evidence_ids:
            self._edges.append(
                EvidenceEdge(
                    edge_id=new_id(),
                    node_id=new_node_id,
                    source_type=source_type,
                    source_id=eid,
                    agent_id=new_node.agent_id,
                    weight=1.0,
                    valid_from_ts=now,
                    valid_until_ts=None,
                    inner_life_event_id=inner_life_event_id,
                    trigger_type=trigger_type,
                )
            )

        self._emit(
            "node_revised",
            new_node.node_id,
            ts=now,
            parent_node_id=old.node_id,
            source_id=source_id,
            provenance_ref=inner_life_event_id,
            node_type=new_node.node_type,
            lineage_depth=new_node.lineage_depth,
            lineage_path=new_node.lineage_path,
            confidence_before=old.confidence,
            confidence_after=new_node.confidence,
            evidence_source_ids=evidence_ids,
            reason="reconsolidated_by_new_evidence",
        )

        self.check_invariants()  # SE-3：变更后锁两图职责
        return new_node

    def _supersede_edges(self, node_id: str, ts: str) -> List[EvidenceEdge]:
        """把某节点仍有效的证据边标记 ``valid_until_ts=ts``（reconsolidation 留痕）。

        只标记，不降权重、不删边、不删 source_id。
        """
        superseded: List[EvidenceEdge] = []
        for i, edge in enumerate(self._edges):
            if edge.node_id == node_id and edge.valid_until_ts is None:
                updated = replace(edge, valid_until_ts=ts)
                self._edges[i] = updated
                superseded.append(updated)
        return superseded

    # —— 第三阶段：升华式遗忘 ——

    def decay(
        self,
        node_id: str,
        *,
        decay_rate: float = DEFAULT_DECAY_RATE,
        fade_ts: Optional[str] = None,
    ) -> List[EvidenceEdge]:
        """情景细节淡化：按 Ebbinghaus 式衰减降级证据权重 + 标记 ``valid_until_ts``。

        衰减采用「每次淡化乘一个保留因子 ``decay_rate``」的离散指数衰减（Ebbinghaus
        曲线的最简可测形式）。**不删除任何证据边、不删除 source**——``source_id``
        始终保留，原文仍可回查。仅作用于仍有效（``valid_until_ts is None``）的边。
        """
        if not isinstance(decay_rate, (int, float)) or not (0.0 < float(decay_rate) < 1.0):
            raise ValueError(f"decay_rate must be in (0.0, 1.0), got {decay_rate!r}")
        now = fade_ts or _utcnow_iso()
        node = self._nodes.get(node_id)
        # essence 豁免 decay（MEMORY-LIFECYCLE §3.2）：essence 是语义核心，不淡化。
        if node is not None and node.node_type == "essence":
            return []
        faded: List[EvidenceEdge] = []
        for i, edge in enumerate(self._edges):
            if edge.node_id == node_id and edge.valid_until_ts is None:
                weight_before = edge.weight
                updated = replace(
                    edge,
                    weight=edge.weight * float(decay_rate),
                    valid_until_ts=now,
                )
                self._edges[i] = updated
                faded.append(updated)
                self._emit(
                    "edge_decayed",
                    edge.node_id,
                    ts=now,
                    parent_node_id=None,
                    source_id=edge.source_id,
                    provenance_ref=None,
                    node_type=node.node_type if node is not None else None,
                    lineage_path=node.lineage_path if node is not None else None,
                    weight_before=weight_before,
                    weight_after=updated.weight,
                    evidence_source_ids=[edge.source_id],
                    reason="episodic_detail_faded",
                )
        return faded

    def forget(
        self,
        node_ids: Sequence[str],
        abstract_content: str,
        *,
        node_type: Optional[NodeType] = None,
        valence: Valence = "neutral",
        confidence_boost: float = DEFAULT_CONFIDENCE_BOOST,
        stability_boost: float = DEFAULT_STABILITY_BOOST,
        decay_rate: float = DEFAULT_DECAY_RATE,
        agent_id: Optional[str] = None,
        inner_life_event_id: Optional[str] = None,
    ) -> ElevationNode:
        """升华式遗忘：情景细节淡化 + 语义核心强化（**非删除**）。

        **Architecture note（SE-2）**：本 API 目前同时承担 decay（情景淡化）与
        consolidate（语义核心聚合）两个职责；长期语义应拆分为两个独立 API
        （decay 与 consolidate 分离），但本票**不改动 API**，仅记录架构意图。

        ① **情景细节淡化**：对每个输入节点做 ``decay``（降权重 + ``valid_until_ts``
           标记），原文 source_id 永远保留。
        ② **语义核心强化**：聚合淡化来源为一个抽象语义节点（新抽象 = 根节点，
           ``parent_node_id=None``，非因果改写），``confidence``/``stability`` 相对
           情景节点**提升**（对应 systems consolidation「海马痕迹重置、新皮层语义留存」）。
        语义节点的证据边回指淡化来源的 ``source_id``，保证原文可回查。
        """
        if not node_ids:
            raise ValueError("forget requires at least one node_id")

        # essence 豁免 forget（MEMORY-LIFECYCLE §3.2）：essence 是语义核心，不可被
        # 「抽象掉」。先整体校验，避免对前面节点做了淡化后才撞到 essence（部分副作用）。
        for nid in node_ids:
            if self.get_node(nid).node_type == "essence":
                raise ValueError(f"essence node {nid!r} is exempt from forget")

        # ① 情景细节淡化
        faded_nodes: List[ElevationNode] = []
        faded_sources: List[SourceType] = []
        faded_source_ids: List[str] = []
        faded_trigger_types: List[str] = []
        for nid in node_ids:
            node = self.get_node(nid)
            faded_nodes.append(node)
            for edge in self.decay(nid, decay_rate=decay_rate):
                faded_sources.append(edge.source_type)
                faded_source_ids.append(edge.source_id)
                faded_trigger_types.append(edge.trigger_type)

        # ② 语义核心强化
        base_confidence = sum(n.confidence for n in faded_nodes) / len(faded_nodes)
        base_stability = max(n.stability for n in faded_nodes)
        confidence = _bump(base_confidence, confidence_boost)
        stability = _bump(base_stability, stability_boost)

        resolved_node_type = node_type or _most_common_node_type(faded_nodes)
        if resolved_node_type not in VALID_NODE_TYPES:
            raise ValueError(
                f"invalid node_type {resolved_node_type!r}; "
                f"expected one of {sorted(VALID_NODE_TYPES)}"
            )
        if valence not in VALID_VALENCES:
            raise ValueError(
                f"invalid valence {valence!r}; expected one of {sorted(VALID_VALENCES)}"
            )

        resolved_agent = agent_id or faded_nodes[0].agent_id
        now = _utcnow_iso()
        new_node_id = new_id()
        semantic_node = ElevationNode(
            node_id=new_node_id,
            node_type=resolved_node_type,
            content=abstract_content,
            confidence=confidence,
            stability=stability,
            valence=valence,
            agent_id=resolved_agent,
            parent_node_id=None,  # 新抽象 = 根节点，非因果改写
            lineage_depth=0,
            lineage_path=new_node_id,
            created_ts=now,
            provenance_ref=inner_life_event_id,
        )
        self._nodes[new_node_id] = semantic_node

        # 语义节点证据边：回指淡化来源（source_id 保留 → 原文可回查）。
        n_sources = len(faded_source_ids)
        per_weight = 1.0 / n_sources if n_sources else 1.0
        for source_type, source_id, trigger_type in zip(
            faded_sources, faded_source_ids, faded_trigger_types
        ):
            self._edges.append(
                EvidenceEdge(
                    edge_id=new_id(),
                    node_id=new_node_id,
                    source_type=source_type,
                    source_id=source_id,
                    agent_id=resolved_agent,
                    weight=per_weight,
                    valid_from_ts=now,
                    valid_until_ts=None,
                    inner_life_event_id=inner_life_event_id,
                    trigger_type=trigger_type,
                )
            )

        self._emit(
            "node_forgotten",
            semantic_node.node_id,
            ts=now,
            parent_node_id=None,
            source_id=None,
            provenance_ref=inner_life_event_id,
            node_type=semantic_node.node_type,
            lineage_depth=semantic_node.lineage_depth,
            lineage_path=semantic_node.lineage_path,
            confidence_after=semantic_node.confidence,
            evidence_source_ids=list(faded_source_ids),
            forgotten_node_ids=[n.node_id for n in faded_nodes],
            reason="semantic_core_reinforced",
        )

        self.check_invariants()  # SE-3：变更后锁两图职责
        return semantic_node
