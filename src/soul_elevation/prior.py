"""活动 → 灵魂维度内化映射的**先验映射表**（deterministic，无 LLM）。

照设计文档（MEMORY-ELEVATION-DESIGN）§5「活动 → 灵魂维度内化映射」落地。
机制是「源类型给先验（prior）+ LLM 依 content/stance 做后验（posterior）」——
本模块只负责**确定性先验**部分，不调用任何 LLM。

映射表为**显式 dict（可扩展）**，不做散落的硬编码 if-else。零 Soul OS 依赖。
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

from .models import NodeType

# 先验 = 候选 node_type 的**有序元组**；``[0]`` 是「基调」（primary prior），
# 其余为备选维度。LLM 后验可从其中（或任意合法维度）依 content 定最终维度。
Prior = Tuple[NodeType, ...]

# —— 固定先验（源类型 → 维度）——
_PRIOR_BELIEF: Prior = ("belief",)
_PRIOR_TRAIT: Prior = ("trait",)
_PRIOR_VALUE_TRAIT_BELIEF: Prior = ("value", "trait", "belief")

# —— 类别相关先验（llm_judge.category → 维度）——
# 用于「对话 / 记忆 fact」这类依既有记忆分类的活动：复用 llm_judge.category 语义，
# 不另起一套（保真：分类已有、语义已验证）。
CATEGORY_PRIOR_TABLE: Mapping[str, Prior] = {
    "preference_plan_event_fact": _PRIOR_BELIEF,   # 偏好/计划/事件/事实 → 认知信念
    "milestone": ("value",),                        # 里程碑 → 价值（单条不产 essence）
    "diary": ("value", "trait"),                    # 自指内省 → 价值/性格
}

# —— 主映射表：trigger_type（活动类型）→ 先验 ——
# essence 保守边界（MEMORY-LIFECYCLE §3.2）：essence 永不作为任何单一事件的 primary
# prior，只能经 consolidation（多条一致证据聚合，即 forget 的抽象语义节点）产生。
PRIOR_TABLE: Mapping[str, Prior] = {
    # 看新闻：关于外部世界的断言 → 世界观信念（不进性格/内涵，避免「看一篇新闻就改性格」失真）
    "world:news_event": _PRIOR_BELIEF,
    # 现实活动：塑造「这个人如何生活」的性格倾向（单条不产 essence）
    "world:calendar_event": _PRIOR_TRAIT,
    "user_going_outside": _PRIOR_TRAIT,
    # 日记/自我内省：自指方向最强 → 价值/性格/信念
    "diary:morning": _PRIOR_VALUE_TRAIT_BELIEF,
    "diary:night": _PRIOR_VALUE_TRAIT_BELIEF,
    # 梦：潜意识投影，单条低置信度、累积升格 → 性格倾向（单条不产 essence）
    "dream:dream": _PRIOR_TRAIT,
    "dream:event": _PRIOR_TRAIT,
}

# 类别相关 trigger_type：其先验不固定，依 provenance 里的 llm_judge.category 决定。
CATEGORY_TRIGGER_TYPES: frozenset = frozenset({"conversation:user_message", "memory_fact"})

# 未知 trigger_type / 未知 category 的保守回退（可扩展性：不崩溃，落入最通用的信念维度）。
DEFAULT_PRIOR: Prior = _PRIOR_BELIEF


def _category_from_provenance(provenance: Mapping[str, Any]) -> Optional[str]:
    """从 provenance 提取 llm_judge.category（兼容嵌套与扁平两种承载方式）。"""
    llm_judge = provenance.get("llm_judge")
    if isinstance(llm_judge, dict):
        category = llm_judge.get("category")
        if category is not None:
            return category
    return provenance.get("category")


def resolve_prior(
    event_type: str,
    provenance: Optional[Mapping[str, Any]] = None,
) -> Prior:
    """确定性地解析某活动类型的先验维度元组（无 LLM）。

    - 类别相关类型（conversation / memory_fact）：依 provenance 的 llm_judge.category 查表。
    - 其余类型：直接查 ``PRIOR_TABLE``。
    - 未知：回退 ``DEFAULT_PRIOR``（不崩溃，保持可扩展）。
    """
    prov = provenance or {}
    if event_type in CATEGORY_TRIGGER_TYPES:
        category = _category_from_provenance(prov)
        if category is not None and category in CATEGORY_PRIOR_TABLE:
            return CATEGORY_PRIOR_TABLE[category]
        return DEFAULT_PRIOR
    return PRIOR_TABLE.get(event_type, DEFAULT_PRIOR)
