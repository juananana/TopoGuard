# TopoGuard 论文思路梳理 + 现存问题与修改意见

---

## 一、论文核心思路（从头捋）

### 1.1 问题背景

复杂任务（如水质分析、风险研判）需要多个 LLM/工具协同完成，形成一个"工作流"。
现有方法要么用固定工作流（Static Workflow），要么每次全局重规划（代价高）。

**本文目标**：在质量 S、成本 C、延迟 L 三个约束下，自适应地选择工作流结构和执行器，并在执行失败时做局部修复，而不是全局重规划。

### 1.2 方法三层结构

```
Task X
  ↓ 难度估计 → 难度桶 b ∈ {easy, medium, hard}
  ↓
[第一层] 模板级 Pareto 筛选
  → 在 (S,C,L) 三目标空间筛选 Pareto 前沿
  → 施加硬约束（预算/延迟上限）
  → 用 Q_T = α·S - β·C - γ·L 选最优模板 T₀
  ↓
[第二层] 节点级执行器 Pareto 筛选
  → 对每个执行节点，在 (S,C) 二维前沿上选最优执行器 φ(v)
  → 得到初始工作流 G₀ = (T₀, Φ₀)
  ↓
[执行 + 评估]
  → 执行节点，评估器输出质量信号 S_v
  → 若 S_v < τ_pass → 触发局部修复
  ↓
[第三层] 局部修复（Bounded Local Repair）
  → Strategy A：拓扑升级（换更深模板）
  → Strategy B：执行器升级（同拓扑换更好模型）
  → Strategy C：评估器升级（跨拓扑换执行器 or 纯 evaluator tweak）
  → 选 adjusted_score = ΔS - λ·L_edit 最高的策略
  → 候选必须满足 actual_S ≥ τ_pass 才进入竞争
```

### 1.3 核心主张

1. **自适应拓扑选择**比固定工作流（Static Workflow）质量更高、成本更低
2. **难度感知**让不同难度任务用不同深度的拓扑，避免"一刀切"
3. **局部修复**在执行失败时不全局重规划，代价小、有效

---

## 二、现存问题逐条列举

---

### 问题 1：修复循环粒度与论文定义不匹配（最严重）

**位置**：[experiment_overall.py](../experiment_overall.py) 约第 796 行，`simulate_training_rounds` 内层循环

**问题描述**：

论文定义修复是针对每个具体执行上下文 `(node_type, difficulty, model)` 的，共 255 个。
但代码的训练模拟循环是按 `(node_type, difficulty)` 分组遍历，只有 **15 个组**，每组内随机取一个 model。

```python
# 当前代码（简化）
by_nd = defaultdict(list)
for p in rnd_profiles:
    by_nd[(p["node_type"], p["difficulty"])].append(p)

for (nt, diff), pts in sorted(by_nd.items()):  # ← 只有 15 个 (nt,diff) 对
    pareto_best = max(front, key=q_fn)
    # repair 在这里判断，但 pareto_best["model"] 是 profile 里随机的某个 model
    actual_s = _sim_topo_actual.get((nt, diff, pareto_best["model"], pareto_best["topo_id"]))
```

**后果**：
- 675 个训练上下文（15组 × 9轮 × 5模型）中，repair 每轮只看 15 个，大量真实失败上下文被跳过
- exp3_repair 显示总触发仅 13 次（1.93%），远低于真实失败率（约 14%）
- Strategy B 和 C2 几乎从未激活

**修改意见**：

将内层循环改为遍历 255 个 `(nt, diff, model)` 上下文：

```python
# 构建测试上下文集合
test_ctx_set = {(r.node_type, r.difficulty, r.model) for r in (test_records or [])}

for (nt, diff, model) in sorted(test_ctx_set):
    pts = by_nd.get((nt, diff), [])
    if not pts:
        continue
    # 用 model 精确查找 actual_s
    actual_s = _sim_topo_actual.get((nt, diff, model, pareto_best["topo_id"]))
    ...
```

---

### 问题 2：τ_pass 阈值与论文文档不一致

**位置**：[experiment_overall.py](../experiment_overall.py) 第 670 行附近，以及 [docs/Method_v5_appendix_method_details.md](Method_v5_appendix_method_details.md) Section D

**问题描述**：

论文附录 D 写的是：
> τ_pass = 0.5246

代码里实际用的是：
```python
PASS_THRESHOLD = float(np.percentile(all_S, 25))  # 数据驱动，结果 = 0.5339
```

两个值不一致（0.5246 vs 0.5339），论文和代码对不上。

**修改意见**：

二选一：
- 要么把论文附录 D 的 τ_pass 改为 0.5339（与代码一致）
- 要么在代码里固定 `PASS_THRESHOLD = 0.5246`（与论文一致）

推荐前者：保留数据驱动的 0.5339，更新论文文档。

---

### 问题 3：Strategy B 编辑惩罚过高，实际无法激活

**位置**：[experiment_overall.py](../experiment_overall.py) 第 693、910-911 行

**问题描述**：

Strategy B（同拓扑换执行器）的评分：
```python
adjusted_B = delta_S_B - EDIT_LAMBDA * EDIT_LAMBDA_EXEC
           = delta_S_B - 0.20 * 0.15
           = delta_S_B - 0.030
```

等等，这里 `EDIT_LAMBDA=0.20`，`EDIT_LAMBDA_EXEC=0.15`，乘积只有 0.03，看起来不大。

但实际上代码里：
```python
L_edit_B = EDIT_LAMBDA_EXEC          # = 0.15
adjusted_B = delta_S_B - EDIT_LAMBDA * L_edit_B  # = delta_S_B - 0.20*0.15 = delta_S_B - 0.03
```

这个惩罚（0.03）其实不大，B 应该能激活。**真正的问题是循环粒度（问题1）**——B 根本没有机会被评估到正确的上下文。

另外，B 的候选搜索条件：
```python
same_topo_cands = [c for c in _all_points
    if c.get("topo_id") == selected_topo
    and c.get("model", "") != selected_c.get("model", "")]
```

这里 `selected_c.get("model", "")` 依赖 `pareto_best` 的 model，但在 (nt,diff) 粒度循环下，`pareto_best` 的 model 是 profile 里随机的，不是当前上下文的真实 model，导致 B 的候选集不准确。

**修改意见**：修复循环粒度（问题1）后，B 的候选搜索自然正确。

---

### 问题 4：w/o Template Selection 消融结果反常

**位置**：[outputs/ablation_new/ablation_results.json](../outputs/ablation_new/ablation_results.json)

**问题描述**：

```
TopoGuard (full):        S = 0.7495
w/o Template Selection:  S = 0.7539  ← 反而更高！
w/o Repair:              S = 0.7407
```

"强制用最深拓扑（executor_verifier_agg）"反而比自适应选择质量更高，这与论文核心主张矛盾——论文说自适应选择比固定工作流更好。

**根本原因**：

当前 profile 数据中，`executor_verifier_agg` 的 profile S 普遍偏高（因为它是最复杂的拓扑，训练样本里质量确实高），所以强制选它在 profile 层面看起来更好。但这是 profile 估计偏差，不是真实质量。

**修改意见**：

1. 检查 `w/o Template Selection` 的实现：它是否也用了 actual_S？还是只用 profile S？
2. 如果用的是 profile S，需要统一改为 actual_S（与 TopoGuard full 一致）
3. 或者在论文中解释：自适应选择的优势在于**成本更低**（avg_C: 0.560 vs 0.582），而不是质量更高

---

### 问题 5：修复触发信号与论文定义的细节不一致

**位置**：[experiment_overall.py](../experiment_overall.py) 第 830-838 行

**问题描述**：

论文 Method v5 第 7.1 节说评估器输出 `y_v ∈ {pass, warn, fail, escalate}`，只有 fail/escalate 才触发修复。

代码实现是：
```python
evaluator_signal = actual_s + noise(0, 0.03)
evaluator_fails = evaluator_signal < PASS_THRESHOLD
```

这是一个连续信号加噪声后与阈值比较，等价于"单阈值二分类"，没有 warn/escalate 层级。

**后果**：论文描述的多级评估器（pass/warn/fail/escalate）在代码里退化为单阈值，论文和代码描述不一致，审稿人可能质疑。

**修改意见**：

二选一：
- 在论文中说明"主实验使用单阈值简化版"（已在附录 D 有说明，但主文没有）
- 或者在代码里实现 warn 层级（warn 时触发轻量修复，fail 时触发完整修复）

---

### 问题 6：难度感知约束预算与论文描述不一致

**位置**：[experiment_overall.py](../experiment_overall.py) 第 686-687 行

**问题描述**：

代码里有难度感知的约束预算：
```python
BUDGET_BY_DIFF = {"easy": 0.55, "medium": 0.50, "hard": 0.40}
LATENCY_BY_DIFF = {"easy": 0.70, "medium": 0.65, "hard": 0.55}
```

但论文 Method v5 第 6.1 节写的是统一约束：
```
施加硬约束（质量下限、预算上限、延迟上限）
```

没有提到按难度分层的约束预算。论文和代码不一致。

**修改意见**：

在论文方法节补充说明：约束预算按难度桶分层设置，hard 任务预算更紧（安全优先），easy 任务预算更宽松（质量优先）。这是一个合理的设计，值得在论文里明确写出来。

---

### 问题 7：Profile S 与 Actual S 的系统性偏差未在论文中说明

**位置**：实验数据，[outputs/overall_water_qa_500ep/summary.json](../outputs/overall_water_qa_500ep/summary.json)

**问题描述**：

实验数据显示 hard 任务存在严重的 profile 估计偏差：
- computation/hard：profile_S ≈ 0.99，actual_S ≈ 0.41
- 偏差高达 0.58

这意味着 profile 对 hard 任务的质量估计严重高估，导致：
1. 修复门控用 profile S 时几乎不触发（因为 profile S 看起来很好）
2. τ_pass 验证用 profile S 时会选出实际质量很差的候选

这个问题已在代码里修复（改用 actual_S），但**论文里没有讨论这个偏差的来源和影响**。

**修改意见**：

在论文实验分析部分加一段说明：
- Profile 估计在 hard 任务上存在系统性高估（原因：训练样本少，hard 任务的 GT 质量分布与 easy 不同）
- 修复门控使用 actual_S 而非 profile S 是关键设计决策
- 这也解释了为什么 repair 的全局贡献相对较小（hard 任务失败上下文已在最深拓扑）

---

### 问题 8：消融实验设计不完整

**位置**：[run_ablation_new.py](../run_ablation_new.py)

**问题描述**：

当前消融只有三个变体：
- TopoGuard (full)
- w/o Template Selection
- w/o Repair

缺少：
- **w/o Difficulty Awareness**：去掉难度桶，用统一约束，看难度感知的贡献
- **w/o Pareto Filtering**：不做 Pareto 筛选，直接用 Q 分数选，看 Pareto 的贡献
- **各 Repair 策略单独消融**：A only / B only / C only，看每个策略的独立贡献

**修改意见**：

至少补充 `w/o Difficulty Awareness` 这一项，因为难度桶是论文 v5 的核心新增内容，没有对应消融实验说服力不足。

---

## 三、问题优先级汇总

| # | 问题 | 严重程度 | 修改难度 | 是否影响核心结论 |
|---|------|----------|----------|-----------------|
| 1 | 修复循环粒度错误 | ★★★ 高 | 中 | 是（repair 贡献被严重低估） |
| 2 | τ_pass 数值不一致 | ★★ 中 | 低 | 否（改文档即可） |
| 3 | Strategy B 无法激活 | ★★ 中 | 低（依赖问题1修复） | 是（repair 机制不完整） |
| 4 | 消融结果反常 | ★★★ 高 | 中 | 是（与核心主张矛盾） |
| 5 | 评估器多级 vs 单阈值 | ★ 低 | 低（改论文描述） | 否 |
| 6 | 难度约束未在论文说明 | ★ 低 | 低（补充论文） | 否 |
| 7 | Profile 偏差未说明 | ★★ 中 | 低（补充论文） | 否 |
| 8 | 消融实验不完整 | ★★ 中 | 中 | 是（难度感知无消融） |

---

## 四、建议修改顺序

1. **先修问题 1**（循环粒度）：这是根本性的实现错误，修完后 repair 的真实贡献才能被正确测量，问题 3 也会自动改善
2. **再看问题 4**（消融反常）：重跑实验后检查 w/o Template Selection 是否还反常
3. **然后修问题 2**（τ_pass 对齐）：统一论文和代码的数值
4. **最后补问题 8**（消融完整性）：加 w/o Difficulty Awareness 消融
5. 问题 5/6/7 只需改论文文字，随时可做
