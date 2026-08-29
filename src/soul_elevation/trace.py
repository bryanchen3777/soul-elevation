"""升华层自有审计 trace（第四阶段）：append-only JSONL sidecar。

照设计文档（MEMORY-ELEVATION-DESIGN）§8「可审计闭环」落地：

- 升华层写**自有** ``elevation_trace.jsonl``（**不写** Soul OS 的 trace.jsonl）。
- 每条记录是 identity + lineage 为主的快照，**不复制证据正文**（只带 ``source_id``
  回指原文）。
- 失败隔离：trace 写失败只 ``logger.warning`` + 返回 ``False``，**不 raise、不阻断
  升华主路径**（对应 M5.4-5.6「Trace failure cannot invalidate canonical event」同构）。

零 Soul OS 依赖：本模块只用标准库，不 import 任何 Soul OS 类型。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

# 默认 trace 路径（相对 CWD 的 data/elevation/ 目录）。
DEFAULT_TRACE_PATH = "data/elevation/elevation_trace.jsonl"

# 事件类型词汇表（frozen，作为运行时校验依据）。
EVENT_TYPES: frozenset = frozenset(
    {
        "node_created",
        "node_elevated",  # pattern → 灵魂结构升华（Consolidation ≠ Elevation）
        "node_revised",
        "edge_decayed",
        "node_forgotten",
    }
)

_LOGGER = logging.getLogger("soul_elevation.trace")


def _utcnow_iso() -> str:
    """当前 UTC 时刻的 ISO 8601 字符串（用于 trace 的 ``ts``）。"""
    return datetime.now(timezone.utc).isoformat()


def build_event(
    event_type: str,
    node_id: str,
    *,
    ts: Optional[str] = None,
    parent_node_id: Optional[str] = None,
    source_id: Optional[str] = None,
    provenance_ref: Optional[str] = None,
    **snapshot: Any,
) -> Dict[str, Any]:
    """构造一条 trace 事件记录：六个必带字段 + 关键字段快照。

    必带字段：``ts`` / ``event_type`` / ``node_id`` / ``parent_node_id`` /
    ``source_id`` / ``provenance_ref``（后三者可空）。其余为事件相关的关键字段
    快照（node_type / lineage / confidence / evidence_source_ids / reason 等）。
    """
    if event_type not in EVENT_TYPES:
        raise ValueError(
            f"invalid event_type {event_type!r}; expected one of {sorted(EVENT_TYPES)}"
        )
    record: Dict[str, Any] = {
        "ts": ts or _utcnow_iso(),
        "event_type": event_type,
        "node_id": node_id,
        "parent_node_id": parent_node_id,
        "source_id": source_id,
        "provenance_ref": provenance_ref,
    }
    record.update(snapshot)
    return record


class ElevationTraceWriter:
    """append-only JSONL trace writer（失败隔离）。

    每次 ``write`` 向文件末尾追加一行 JSON；**不覆盖、不重写**历史（append-only）。
    路径在构造时注入，默认 ``data/elevation/elevation_trace.jsonl``。
    """

    def __init__(
        self,
        path: Union[str, "Path", None] = None,
        *,
        logger: Optional[Any] = None,
    ) -> None:
        self.path = str(path) if path is not None else DEFAULT_TRACE_PATH
        self._logger = logger if logger is not None else _LOGGER

    def write(self, record: Mapping[str, Any]) -> bool:
        """追加一条 JSON 行；写失败只告警 + 返回 ``False``，**绝不 raise**。"""
        try:
            self._ensure_parent()
            line = json.dumps(dict(record), ensure_ascii=False)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            return True
        except Exception as exc:  # noqa: BLE001 — 失败隔离：绝不阻断升华主路径
            self._logger.warning("elevation_trace write failed: %s", exc)
            return False

    def _ensure_parent(self) -> None:
        parent = Path(self.path).parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)


def read_trace(path: Union[str, "Path"]) -> List[Dict[str, Any]]:
    """读回 JSONL trace → ``list[dict]``。

    跳过空行；坏行（非法 JSON）抛 ``ValueError`` 并带行号——审计日志完整性优先，
    静默吞坏行会掩盖审计缺口。
    """
    p = Path(path)
    if not p.exists():
        return []
    records: List[Dict[str, Any]] = []
    for lineno, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            records.append(json.loads(raw))
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed trace line {lineno} in {p}: {exc}") from exc
    return records


def records_by_node(records: List[Dict[str, Any]], node_id: str) -> List[Dict[str, Any]]:
    """筛选某节点的全部 trace 事件（按出现顺序）。"""
    return [r for r in records if r.get("node_id") == node_id]


def rebuild_lineage(records: List[Dict[str, Any]], node_id: str) -> List[Dict[str, Any]]:
    """按 ``parent_node_id`` 反向重建某节点的因果链（根 → 目标，含目标）。

    每个节点的「身份记录」取其最早出现的一条（创建/修订时刻的快照最完整）。
    返回从根到目标的有序记录列表，供「如何一步步被修订」审计。
    """
    by_id: Dict[str, Dict[str, Any]] = {}
    for r in records:
        nid = r.get("node_id")
        if nid is not None and nid not in by_id:
            by_id[nid] = r

    parent: Dict[str, str] = {}
    for r in records:
        nid = r.get("node_id")
        pid = r.get("parent_node_id")
        if nid is not None and pid:
            parent[nid] = pid

    chain: List[Dict[str, Any]] = []
    cur: Optional[str] = node_id
    seen = set()
    while cur is not None and cur not in seen:
        seen.add(cur)
        chain.append(by_id.get(cur, {"node_id": cur}))
        cur = parent.get(cur)
    chain.reverse()
    return chain
