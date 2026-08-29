# soul-elevation

记忆升华层（Memory Elevation）—— 独立 repo。**第一阶段：数据模型 + 证据链 + 输入契约 + 引擎接口**。

## 为什么独立开 repo

记忆升华是 Soul OS 的真实空白与差异化机会：现有 AI 记忆系统止步于「分层 + 检索 + 摘要 + 自编辑 persona 块」，无人做 first-class 的「记忆 → 信念/性格/内涵」升华。独立 repo 从物理上强制「只读消费、不碰 frozen contract」的边界。

## 设计（简述）

- 升华层是**只读消费**层：读上游 `InnerLifeEvent` / `Memory` / SAGE `Fact`，写自有 store；不改任何 frozen contract。
- 数据模型**自研**：带完整证据链 + 置信度 + 版本因果树的信念/价值/性格/内涵节点 + **Pattern 中间层**。
- 三个核心机制自研（证据链节点 / reconsolidation 修订 / 升华式遗忘），四个机制采用现成（CLS 双系统、SAGE 强化原则、时序有效区间、sidecar 审计）。

## Consolidation ≠ Elevation 边界（Pattern 中间层）

**原则原文**：

> Memory preserves experience. Consolidation distills durable patterns. Elevation internalizes durable meaning into Soul structure. Higher abstraction = stronger evidence + slower change.

**采纳原则**：

1. **Consolidation ≠ Elevation**：记忆巩固（Consolidation）与灵魂升华（Elevation）是两件事，分开处理。
2. **单一事件不直接产生灵魂结构**：essence 保守边界扩展到**所有** node type——单一事件只产 `pattern`（候选），不直接产 belief/value/trait/essence。
3. **LLM = interpretation 不是 truth**：LLM 后验结果只是「候选解释」（`candidate_node_type`），不直接当灵魂事实；证据累积 + 阈值决定是否升华。
4. **两个图分开**：lineage（因果树）vs evidence（证据边，`EvidenceEdge` 已做）。
5. **不要过度工程**：不加 embedding / vector DB / graph DB / scoring framework / 新 persistence layer。

**机制**：

- `consume(input)`：单一事件 → 产 **`pattern` 节点**（consolidation 输出，候选）。LLM 后验的维度/内容/置信度作为 `candidate_node_type` / `content` / `confidence` 存在 pattern 上（interpretation，非 truth）。签名不变：`consume(input: ElevationInput) -> list[ElevationNode]`。
- `elevate(pattern_node_id, ...)`：证据累积达阈值（默认 `min_evidence=2`，可配）后，把 pattern 升华成 belief/value/trait/essence。升华维度默认取 LLM 后验候选，也可显式传入（如由 prior 表决定）。升华后 **pattern 保留**（灵魂节点以 `parent_node_id=pattern` 挂因果树；pattern 的有效证据边标记 `valid_until_ts` superseded 留痕，灵魂节点新建证据边回指同一批 `source_id`）。
- 证据不足（< 阈值）时 `elevate` 抛 `ValueError`——LLM 说「这是信念」不算数，证据说了算。

## 数据模型（第一阶段）

### `ElevationNode` — 信念/价值/性格/内涵/模式节点

五类节点统一 schema（`pattern` 是第 5 类：consolidation 输出，**非灵魂结构**）。**关键保真约束**：节点**不存证据正文**，只带证据边索引——证据正文永远留在证据边指向的 `Memory` / `Fact` / `Event` 里。

| 字段 | 类型 | 说明 |
|------|------|------|
| `node_id` | `str` | 32-hex（独立命名空间） |
| `node_type` | `belief/value/trait/essence/pattern` | 五类节点统一 schema |
| `content` | `str` | 自然语言命题 |
| `confidence` | `float` | 0.0–1.0，证据链聚合 |
| `stability` | `float` | 0.0–1.0，reconsolidation 次数 + 一致证据数 |
| `valence` | `positive/negative/neutral` | 情感极性 |
| `agent_id` | `str` | 归属 agent（灵魂本体） |
| `parent_node_id` | `Optional[str]` | 因果父节点（改写=新节点引用旧节点，一父） |
| `lineage_depth` | `int` | 根=0，父+1 |
| `lineage_path` | `str` | 反范式化路径 |
| `created_ts` | `str` | ISO 8601 UTC |
| `provenance_ref` | `Optional[str]` | 触发节点的上游事件 id |
| `candidate_node_type` | `Optional[belief/value/trait/essence]` | 仅 pattern 节点：LLM 后验候选维度（interpretation，非 truth） |

### `EvidenceEdge` — 证据边

| 字段 | 类型 | 说明 |
|------|------|------|
| `edge_id` | `str` | — |
| `node_id` | `str` | → ElevationNode |
| `source_type` | `v1_memory/sage_fact/inner_life_event` | 证据来源类型 |
| `source_id` | `str` | 回指原始记忆/事件（回查原文） |
| `agent_id` | `str` | — |
| `weight` | `float` | 0–1 贡献权重 |
| `valid_from_ts` | `str` | 证据生效时刻 |
| `valid_until_ts` | `Optional[str]` | `None`=仍有效；非 `None`=已被取代（reconsolidation 留痕） |
| `inner_life_event_id` | `Optional[str]` | canonical event |
| `trigger_type` | `str` | 证据来源的活动类型 |

## 输入契约

### `ElevationInput` — adapter seam

**整合方式**：Soul OS 未来提供 **adapter** 把 `InnerLifeEvent` + `Memory` 映射成 `ElevationInput`，本 repo 不依赖 Soul OS 任何类型。`provenance` 用 `dict` 承载上游 provenance 元数据（trigger_type / source_system / extras 等）。

| 字段 | 类型 | 说明 |
|------|------|------|
| `event_type` | `str` | 活动类型（如 `diary:night` / `world:news_event`） |
| `content` | `str` | 归一化后的内容 |
| `source_id` | `str` | 回指上游 memory_id / fact_id / event_id |
| `source_type` | `v1_memory/sage_fact/inner_life_event` | 来源类型 |
| `timestamp` | `str` | ISO 8601 UTC |
| `provenance` | `dict` | 上游 provenance 元数据 |

## 引擎接口

```python
class ElevationEngine(ABC):
    @abstractmethod
    def consume(self, input: ElevationInput) -> list[ElevationNode]:
        ...
```

`consume(input: ElevationInput) -> list[ElevationNode]` —— **接口签名已定死**。单一事件只产 `pattern` 候选节点（consolidation 输出）；灵魂结构（belief/value/trait/essence）须经 `elevate(pattern_node_id, ...)` 证据累积达阈值后升华（Consolidation ≠ Elevation 边界）。完整升华逻辑（活动→灵魂维度内化映射、reconsolidation 式信念修订、升华式遗忘、可审计闭环）见各阶段实现。

## 零 Soul OS 依赖

本 repo 不 import 任何 Soul OS 模块；`tests/test_zero_soulos_dependency.py` 用测试强制执行（所有非相对 import 必须来自标准库）。

## 开发 / 测试

```powershell
# python 不在 PATH 时用完整路径
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m pytest -v
```

## 不做（第一阶段 Out of Scope）

- 不实现 `consume` 的完整升华逻辑（后续阶段）。
- 不 import Soul OS 任何模块。
- 不改 frozen contract（InnerLifeEvent / TriggerEnvelope / Agency 4 stages / SAGE 写入逻辑）。
- 不 push GitHub（本地 repo 即可）。
