# TopoGuard 代码修改计划

## 背景

当前代码存在两套并行实现：
- `src/` — 正式框架代码（`ProfileManager`、`TemplateLibrary`、`WorkflowExecutor` 等）
- `experiment_water_qa_topo.py` / `experiment_overall.py` — 论文实际跑的顶层脚本，内联重写了 Pareto、Q-score、修复逻辑，几乎不调用 `src/`

两套实现存在 7 处实质性分歧，导致论文代码与框架代码不一致，部分 `src/` 实现存在 bug。

---

## 修改项清单

### P1：修复 src/ 中的 bug（影响论文正确性，必须做）

#### P1-1：`profile_manager.pareto_frontier` 加延迟维度

**文件：** `src/primitives/profile_manager.py`，约第 531 行

**问题：** 当前 Pareto 计算只有 2D（质量 vs 成本），完全丢弃了延迟 L。
论文 §3.2 明确定义 Q(G,X) 包含三个目标，Pareto 剪枝也应在 3D 空间进行。

**修改：**
```python
# 修改前（2D）
data = np.column_stack([qualities, costs])
mask = compute_pareto(data, sense=["max", "min"])

# 修改后（3D）
latencies = np.array([c.get("pred_latency", 0.0) for c in all_candidates])
data = np.column_stack([qualities, costs, latencies])
mask = compute_pareto(data, sense=["max", "min", "min"])
```

同时在 `select_from_frontier` 的签名里补上 `gamma` 参数（目前只有 alpha/beta）。

---

#### P1-2：`select_from_frontier` 加 S_SCALE 归一化，统一权重

**文件：** `src/primitives/profile_manager.py`，约第 559 行；
         `src/primitives/topology_template.py`，约第 440 行

**问题：** 两处 Q-score 公式都缺少 `S_SCALE=1.5` 归一化。
不加 S_SCALE 时，`pred_acc ∈ [0,1]` 而 `pred_cost` 可能是任意正数，
公式实际退化为 `argmax S`，成本/延迟权重形同虚设。

**修改（profile_manager.py）：**
```python
# 修改前
def _q(c: dict) -> float:
    return a * c.get("pred_acc", 0.0) - b * c.get("pred_cost", 0.0)

# 修改后
S_SCALE = 1.5
def _q(c: dict) -> float:
    S_norm = c.get("pred_acc", 0.0) / S_SCALE
    C_norm = c.get("pred_cost_norm", c.get("pred_cost", 0.0))
    L_norm = c.get("pred_latency_norm", c.get("pred_latency", 0.0))
    return a * S_norm - b * C_norm - g * L_norm
```

同时将默认权重从 `alpha=0.7, beta=0.3` 改为 `alpha=0.65, beta=0.25, gamma=0.10`，
与论文 §3.2 和顶层脚本保持一致。

**修改（topology_template.py）：**
同样加入 S_SCALE 归一化，默认权重改为 `alpha=0.65, beta=0.25, gamma=0.10`。

---

#### P1-3：确认 `run_train_test_experiment` 调用 `template_library.add_feedback()`

**文件：** `src/experiments/mvp_experiment.py`，约第 3309 行

**状态：** 已存在，无需修改。代码已在每个 episode 后调用：
```python
template_library.add_feedback(
    template_id=template_id,
    difficulty_bucket=bucket,
    observed_quality=1.0 if eval_result.eval_pass else 0.0,
    observed_cost=...,
    observed_latency=...,
)
```

---

### P2：补充缺失实现（弥补论文弱点）

#### P2-1：修复策略顺序——已确认一致，无需修改

**状态：** 经过仔细核查，两套实现在效果上等价，无需修改。

- `repair_topo()`（顶层脚本）：收集 A/B/C 三种策略的所有成功候选，
  然后按 `(ΔV, -Q)` 排序，ΔV=0 的 B/C 策略自然排在 ΔV>0 的 A 策略前面。
  这与"最小编辑优先"原则一致。

- `_repair_subgraph()`（src/）：显式按 B→C→A 顺序尝试，找到第一个成功的就返回。

两者都优先最小编辑（B/C），最后才做拓扑升级（A），逻辑等价。

---

#### P2-2：`topology_template.py` 加入 `bad_direct` 模板

**文件：** `src/primitives/topology_template.py`，约第 168 行

**问题：** `DEFAULT_TEMPLATES` 缺少 `bad_direct` 模板（质量受限的单节点执行）。
顶层脚本中 `bad_direct` 的参数为 `quality_mult=0.70, cost_mult=1.30`，
它的存在使 Pareto 剪枝有被支配候选可以过滤，验证了剪枝机制的有效性。

**修改：** 在 `DEFAULT_TEMPLATES` 中添加：
```python
TopologyTemplate(
    template_id="bad_direct",
    description="Single executor with degraded quality (dominated candidate for Pareto validation)",
    supported_node_types=["forecast", "state_parse", "data_analysis"],
    supported_task_types=["time_series", "text_analysis", "tabular_analysis", "multimodal"],
    nodes=[
        TemplateNode(node_id="exec", node_type="executor", primitive_name="", depends_on=[]),
    ],
    estimated_cost=0.65,    # cost_mult=1.30 × direct baseline
    estimated_latency=1.0,
    estimated_quality=0.56, # quality_mult=0.70 × direct baseline 0.8
),
```

---

### P3：重构顶层脚本调用 src/（提升代码可信度）

**文件：** `experiment_water_qa_topo.py`，`experiment_overall.py`

**目标：** 将顶层脚本中内联重写的逻辑替换为对 `src/` 的调用：

| 顶层脚本内联函数 | 替换为 src/ 调用 |
|---|---|
| `pareto_frontier()` | `ProfileManager.pareto_frontier()` |
| `q_score()` | `ProfileManager.select_from_frontier()` |
| `filter_by_constraints()` | `select_from_frontier()` 的硬过滤参数 |
| `repair_topo()` | `mvp_experiment._repair_subgraph()` |
| 拓扑模板定义 | `TemplateLibrary` + `TopologyTemplate` |

**前提：** P1-1 和 P1-2 必须先完成，否则 `src/` 的实现仍有 bug，重构后结果会不一致。

**注意：** 顶层脚本的 log-scale 归一化（`C_norm`/`L_norm`）在 `src/` 中没有对应实现，
重构时需要在 `ProfileManager` 或调用层加入归一化步骤，或将归一化后的值作为参数传入。

---

## 执行顺序

```
P1-1 → P1-2 → P2-2 → P2-1 → P3
```

P1-3 已存在，跳过。P3 依赖 P1 全部完成。

---

## 各修改项的文件定位

| 修改项 | 文件 | 行号（约） |
|---|---|---|
| P1-1 pareto 3D | `src/primitives/profile_manager.py` | 531–557 |
| P1-2 Q-score S_SCALE (profile_manager) | `src/primitives/profile_manager.py` | 559–633 |
| P1-2 Q-score S_SCALE (topology_template) | `src/primitives/topology_template.py` | 440–490 |
| P2-1 修复顺序统一 | `experiment_water_qa_topo.py` | 707–820 |
| P2-2 bad_direct 模板 | `src/primitives/topology_template.py` | 168–250 |
| P3 重构顶层脚本 | `experiment_water_qa_topo.py` | 全文 |
