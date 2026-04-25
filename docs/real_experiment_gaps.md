# 真实实验缺口分析

## 核心问题

论文声称 TopoGuard 是一个"闭环执行系统"，但当前所有实验数字来自一个**纯仿真框架**，没有任何真实 LLM 调用参与实验主循环。

---

## 问题清单

### 问题 1：评估器是硬编码 stub，不是真实 LLM judge

**位置：** `src/evaluation/mock_evaluator.py:950-999`，`experiment_overall.py` 的 `simulate_training_rounds()`

**现状：**
```python
# mock_evaluator.py 第 955 行注释
# 在真实场景中，这里会调用真实 LLM judge API。

# experiment_overall.py 里 Strategy C 的"评估器升级"
boosted["S"] = min(1.0, candidate["S"] + EVAL_UPGRADE_QUALITY_BOOST)  # += 0.02
boosted["C"] = candidate["C"] + EVAL_UPGRADE_COST_PENALTY              # += 0.005
```

**论文声称：** 评估器是 LLM-as-judge，Strategy C 是"升级到更强的评估器"。

**实际：** 质量分数加 0.02，成本加 0.005，纯算术 stub。

**缺失文件：** `src/evaluation/claude_evaluator.py` — 被 `__init__.py` 和 `mvp_experiment.py` import，但文件不存在，会导致 `ImportError`。

---

### 问题 2：任务分解器是关键词规则，不是 LLM

**位置：** `src/decomposer/task_decomposer.py:1-80`

**现状：** 用 `re` 正则匹配关键词判断任务类型，注释明确写"MVP 用关键词规则系统"。

**论文声称：** 任务分解由 LLM 驱动，能理解自然语言任务描述。

**缺失文件：** `src/decomposer/llm_decomposer.py` — 被 `decomposer/__init__.py` import，文件不存在，会导致 `ImportError`。

---

### 问题 3：拓扑质量来自公式，不是真实执行

**位置：** `experiment_overall.py:_build_workflow_scl_wqa()`，`TOPO_BONUS` 常量

**现状：**
```python
# 拓扑质量公式（非测量值）
headroom = 1.0 - base_quality
topo_mult = 1.0 + topo_bonus * headroom
sq = min(0.99, base_quality * topo_mult * DIFF_COEFF.get(diff, 1.0))

# TOPO_BONUS 是手工校准的常数
"executor_plus_verifier": {"easy": 0.16, "medium": -0.34, ...}
"executor_verifier_agg":  {"easy": 0.20, "medium": 0.20, ...}
```

**论文声称：** 多节点拓扑的质量来自真实执行测量。

**实际：** 基于 14 次 Kimi-K2.5 人工校准的乘数，不是系统测量值。

---

### 问题 4：所有实验数据加了 Gaussian 噪声，不是真实观测

**位置：** `experiment_overall.py:generate_dataset()` 约第 530 行

**现状：**
```python
nq = wq + rng.gauss(0, 0.015)   # 质量加噪声
nc = wc + abs(rng.gauss(0, 0.001))  # 成本加噪声
nl = wl + abs(rng.gauss(0, 0.5))    # 延迟加噪声
```

**论文声称：** 实验在真实任务执行记录上进行。

**实际：** 从公式值加噪声生成，不是真实 LLM 调用的观测值。

---

### 问题 5：ProfileManager 从未被实验调用

**位置：** `experiment_overall.py` 全文，`experiment_water_qa_topo.py` 全文

**现状：** `TemplateLibrary` 在 `experiment_water_qa_topo.py` 第 43 行被 import，但从未实例化或调用。`ProfileManager` 在两个顶层脚本中完全不存在。

**论文声称：** TopoGuard 使用 ProfileManager 进行在线画像学习和 Pareto 选择。

**实际：** 实验用内联的 `estimate_profiles()` + `pareto_frontier()` 函数，与 `ProfileManager` 无关。

---

### 问题 6：没有 requirements.txt，依赖未声明

**位置：** 项目根目录

**现状：** 无 `requirements.txt`，无 `pyproject.toml`。`anthropic` SDK 被引用但未声明。

---

## 各问题的影响等级

| 问题 | 对论文可信度的影响 | 修复难度 |
|------|------------------|---------|
| 1. 评估器是 stub | 高 — Strategy C 的贡献数字不可信 | 中（需实现 ClaudeEvaluator） |
| 2. 分解器是规则 | 中 — 影响任务分解的泛化性声明 | 中（需实现 LLMTaskDecomposer） |
| 3. 拓扑质量是公式 | 高 — 多节点拓扑的核心优势数字不可信 | 大（需真实多节点执行） |
| 4. 数据加噪声 | 中 — 离线重放协议本身可接受，但需在论文中明确说明 | 小（只需在论文中说明） |
| 5. ProfileManager 未调用 | 中 — 代码与论文描述不一致 | 中（需接入实验主循环） |
| 6. 无 requirements.txt | 低 — 可复现性问题 | 小 |
