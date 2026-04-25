# 真实实验修改计划

## 目标

用最小工作量让论文最关键的声明有真实代码支撑：
1. 评估器是真实 LLM judge（Claude API）
2. 任务分解器能处理自然语言
3. 实验主循环能切换到真实模式运行

不重写整个实验框架，只补缺失的两个文件 + 接入开关。

---

## 执行顺序

```
Step 1 → Step 2 → Step 3 → Step 4 → Step 5
```

---

## Step 1：实现 `claude_evaluator.py`

**文件：** `src/evaluation/claude_evaluator.py`（新建）

**接口：** 实现 `BaseEvaluator`（定义在 `evaluator_types.py`）

**核心逻辑：**
- 接收 `EvaluatorInput`（含 `candidate_output`、`reference_output`、`node_type`、`difficulty_bucket`）
- 按 `NODE_TYPE_RUBRIC` 构造 prompt，调用 `claude-sonnet-4-6` 打分
- 返回 `EvaluatorOutput`（`quality_score`、`passed`、`confidence`、`latency`、`api_cost`）

**Prompt 结构：**
```
你是一个专业评估员。请对以下输出打分（0-10分）。
任务类型：{node_type}，难度：{difficulty_bucket}
评估维度：{rubric_dimensions}
候选输出：{candidate_output}
参考输出：{reference_output}（如有）
请返回 JSON：{"score": 0-10, "passed": true/false, "confidence": 0-1, "reason": "..."}
```

**成本控制：**
- 默认用 `claude-haiku-4-5-20251001`（最便宜）
- `use_strong_evaluator=True` 时升级到 `claude-sonnet-4-6`（Strategy C 触发时）

**关键参数：**
```python
class ClaudeEvaluator(BaseEvaluator):
    def __init__(self, api_key, model="claude-haiku-4-5-20251001",
                 pass_threshold=0.60, strong_model="claude-sonnet-4-6"):
```

---

## Step 2：实现 `llm_decomposer.py`

**文件：** `src/decomposer/llm_decomposer.py`（新建）

**接口：** 实现 `LLMTaskDecomposer`，与现有 `TaskDecomposer` 接口兼容

**核心逻辑：**
- 接收自然语言任务描述
- 调用 `claude-haiku-4-5-20251001` 识别 `node_type`、`difficulty`、`task_type`
- 返回与 `TaskDecomposer` 相同格式的 `SubTaskSpec` 列表

**降级策略：** LLM 调用失败时自动 fallback 到关键词规则分解器

**关键参数：**
```python
class LLMTaskDecomposer:
    def __init__(self, api_key, model="claude-haiku-4-5-20251001",
                 fallback_to_rules=True):
```

---

## Step 3：写 `requirements.txt`

**文件：** `requirements.txt`（新建，项目根目录）

**内容：**
```
anthropic>=0.40.0
openai>=1.0.0
numpy>=1.24.0
matplotlib>=3.7.0
paretoset>=1.0.0
scipy>=1.10.0
```

---

## Step 4：在 `experiment_overall.py` 接入真实评估器开关

**文件：** `experiment_overall.py`

**修改点：**

1. 加 `--real-eval` CLI 参数（默认 False，不影响现有结果）
2. 在 `simulate_training_rounds()` 里，当 `--real-eval` 开启时：
   - Strategy C 的"评估器升级"改为调用 `ClaudeEvaluator(strong_model=True)`
   - 用真实返回的 `quality_score` 替换 `S += 0.02` stub
3. 结果写入单独的 `outputs/overall_water_qa_real/` 目录，不覆盖现有结果

**这样做的好处：**
- 现有仿真结果完全不受影响
- 可以对比 `--real-eval` 和仿真结果，验证 stub 的误差范围
- Strategy C 的贡献数字有真实 LLM 支撑

---

## Step 5：在 `experiment_overall.py` 接入 ProfileManager

**文件：** `experiment_overall.py`

**修改点：**

在 `strategy_comparison()` 的 `Pareto+Q` 策略里，用 `ProfileManager` 替换内联的 `estimate_profiles()` + `pareto_frontier()`：

```python
# 修改前（内联实现）
profiles = estimate_profiles(records)
frontier = pareto_frontier(profiles)
best = max(filter_by_constraints(frontier), key=q_score)

# 修改后（调用 ProfileManager）
from src.primitives.profile_manager import PrimitivePerformanceProfileManager
manager = PrimitivePerformanceProfileManager(...)
# 用 train records 初始化 manager
for r in train_records:
    manager.add_feedback(r.primitive_name, r.model, r.difficulty, r.quality, r.cost, r.latency)
# 用 manager 做 Pareto 选择
frontier = manager.pareto_frontier(primitive_name, difficulty)
best = manager.select_from_frontier(frontier, alpha=Q_ALPHA, beta=Q_BETA, gamma=Q_GAMMA)
```

**注意：** 只替换 `Pareto+Q` 策略，其他 6 个 baseline 策略保持不变（它们本来就是对照组）。

---

## 不做的事

- 不重写 `generate_dataset()`（离线重放协议在论文中明确说明即可）
- 不替换 `TOPO_BONUS` 公式（需要大量真实多节点执行，成本高）
- 不重写 `mvp_experiment.py` 的实验主循环（工作量过大）
- 不改变现有实验结果（所有修改都是新增，不覆盖）

---

## 完成后能声称什么

| 声明 | 修改前 | 修改后 |
|------|--------|--------|
| 评估器是 LLM judge | ❌ stub | ✅ ClaudeEvaluator |
| Strategy C 有真实 LLM 支撑 | ❌ S+=0.02 | ✅ 真实打分 |
| 任务分解器理解自然语言 | ❌ 关键词规则 | ✅ LLMTaskDecomposer |
| ProfileManager 参与实验 | ❌ 从未调用 | ✅ Pareto+Q 策略使用 |
| 代码可复现 | ❌ 无依赖声明 | ✅ requirements.txt |
