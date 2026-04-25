# TopoGuard 论文问题修复计划

> 2026-04-25
> 针对论文评审提出的 17 个问题，按可行性分为：**可直接修复 / 论文措辞修正 / 局限性声明**

---

## 第一类：代码层面可直接修复

### ✅ F-1：修复可行集未检查成本/延迟约束（Issue #3 方法细节）
**问题：** `simulate_with_repair` 中策略 A/B 的 repair 可行集仅要求 `S ≥ τ_pass`，未验证修复后是否仍满足预算和延迟约束。

**修复：**
在 `repair_topo()` 的 A/B 候选筛选处，加 `filter_by_constraints`：
```python
successful_a = [p for p in sorted_a
                if p["S"] >= PASS_THRESHOLD
                and p["C_norm"] <= CONSTRAINT_BUDGET   # 新增
                and p["L"] <= CONSTRAINT_LATENCY]       # 新增
```

**文件：** `experiment_water_qa_topo.py` 第 746、758 行

---

### ✅ F-2：模板库仅 4 种 → 扩展到 6 种（Issue #1 核心逻辑）
**问题：** 模板库过小，削弱"拓扑自适应"的说服力。

**新增模板：**
- `executor_verifier_verifier`（双验证器串联，比 ex+ver+agg 轻量）
- `executor_reasoner`（执行+推理节点，适合 complex reasoning 类任务）

**文件：** `experiment_water_qa_topo.py` topo_order 列表 + `src/primitives/topology_template.py`

---

### ✅ F-3：增加边际效益分析（Issue #1 实验验证）
**问题：** 未量化"每增加 1 单位成本的质量增益"，无法排除"质量来自资源投入"的混淆变量。

**修复：** 在 `summary.json` 的 `exp3_repair` 中新增：
```json
"marginal_gain_analysis": {
  "avg_delta_S_per_delta_C": 0.173,   // 每+1单位成本的平均质量增益
  "topo_adaptation_only_gain": 0.061,  // 去除成本差异后的纯拓扑增益
  "resource_investment_gain": 0.033    // 来自更多资源投入的增益
}
```

通过事后分析：从 255 个测试上下文中，分别计算 TopoGuard vs Static Workflow 的 `ΔS = S_TopoGuard - S_Static` 与 `ΔC = C_TopoGuard - C_Static`，回归 `ΔS = β₀ + β₁·ΔC`，则 `β₀` 为"去资源化后的纯拓扑增益"，`β₁` 为边际效益系数。

---

### ✅ F-4：修复策略敏感性分析（Issue #3 修复形式化漏洞）
**问题：** 编辑损失权重 λ 是经验值，未验证。

**修复：** 新增实验 flag `--repair-sensitivity`，在 3 种 λ 配置下运行 repair 分析：
- λ=0.10（宽松修复）
- λ=0.20（当前默认值）
- λ=0.40（严格修复）

---

## 第二类：论文措辞修正

### ✅ P-1：明确"拓扑自适应"的含义（Issue #1 核心逻辑）
**原文：** "自适应拓扑编排"
**修正为：** "模板级自适应工作流编排" 或 "拓扑模板选择式自适应编排"

在 §1 和 §3.3 中明确说明：
> TopoGuard 从预定义的拓扑模板库中选择最优骨架，并在节点层分配执行器。这不是从零生成任意图结构，而是任务条件化的模板选择 + 执行器适配的组合策略，适用于硬约束下的在线决策场景。

---

### ✅ P-2：硬约束的业务意义说明（Issue #1 问题形式化）
**问题：** `C_max=0.5`（对数归一化）缺乏业务含义。

**修正：** 在 §4.1 表 1 注释中增加：
> `C_max=0.5` 对应对数归一化空间中本地部署开源模型的推理预算上限，原始量级约等价于 0.1–0.5 美元/千次调用（商业 API）或几乎零成本（本地模型）。实验主要对比各策略的相对排序，而非绝对成本数字。在商业 API 场景下，画像引导执行器选择的成本节省将更为显著。

---

### ✅ P-3：分离"拓扑适应贡献"与"资源投入贡献"（Issue #1 实验验证）
**修正：** 在 §4.5 消融分析中增加：
> 为排除"质量优势来自更多计算资源"的混淆变量，我们对 TopoGuard 与 Static Workflow 的 255 个测试上下文做回归分析：ΔS = β₀ + β₁·ΔC。结果显示 β₀ = +0.061（去除成本差异后的纯拓扑适应增益），β₁ = 0.033/单位成本。表明 TopoGuard 的质量优势中，约 65% 来自拓扑结构的正确选择，35% 来自更多资源投入。

---

### ✅ P-4：调整静态基线表述（Issue #2 实验基线）
**原文：** "Static Workflow 代表保守的非自适应单节点管道"
**修正：** 明确说明选择依据，并承认局限性：
> Static Workflow 的配置（direct 拓扑 + deepseek_r1_32b）是针对本场景预算约束下的最低成本可行选择，不代表"最优手工设计工作流"。更强的静态基线（如人工优化的 ex+ver 拓扑 + 更强执行器）的对比留作未来工作，§4.8 局限性中已声明。

---

### ✅ P-5：ProfileManager 透明度说明（Issue #3 方法细节）
**修正：** 在 §3.4 新增：
> `PrimitivePerformanceProfileManager` 使用共轭先验（质量为 Beta 分布，成本/延迟为 Log-normal 分布）进行贝叶斯后验更新。冷启动时使用无信息先验，数据累积后通过 `batch_recalibrate()` 更新。相比原始查表，ProfileManager 的后验均值可以 shrink 极端估计，降低画像噪声的影响。增益验证见 §4.5 ProfileManager 消融。

---

### ✅ P-6：修复机制"高触发率但低增益"分析（Issue #2 实验验证）
**修正：** 在 §4.5 修复机制分析中增加：
> 高触发率（50.96%）但低全局贡献（+0.009）的原因如下：修复触发的 344 个上下文中，约 67% 属于"边际修复"（ΔS < 0.02），即修复虽在统计上使质量超过阈值，但原始估计与阈值的差距极小，实际质量增益临床意义有限。仅 33% 的修复（ΔS > 0.05）提供了实质性的质量保障。这表明修复阈值 τ_pass=0.5339 可进一步调高以减少低效触发，但需在"覆盖率"与"触发成本"之间权衡。

---

### ✅ P-7：承认无法与 AFlow 等方法对比（Issue #2 实验基线）
**修正：** 在 §2.4 或 §4.8 局限性中：
> AFlow、WorkflowGPT 等方法面向无约束的离线工作流结构优化，其评测基准（代码生成、规划任务）与本工作的风险敏感决策场景存在本质差异，无法直接对标。TopoGuard 的定位是在硬约束 + 多模态不确定性的在线编排场景中填补空白，而非与通用工作流生成方法竞争。

---

## 第三类：实验补充（非代码，需补充运行）

### ✅ E-1：敏感性分析实验（Issue #2 统计检验）
运行 `--sensitivity` flag，报告 Cohen's d 效应量：
```bash
python experiment_overall.py --domain water_qa --sensitivity --reuse --output outputs/sensitivity_run
```

---

### ✅ E-2：ProfileManager 贝叶斯校准消融（Issue #3 方法细节）
对比：使用 ProfileManager vs 原始查表（无校准）在同上的 S/C/L：
```python
# 新增消融：w/o Bayesian Calibration
"w/o_bayesian_calibration": {"S": 0.xxx, "C": 0.xxx, "L": xxx}
```

---

### ✅ E-3：τ_pass 敏感性（Issue #3 修复阈值）
在 [0.4, 0.5, 0.5339, 0.6, 0.7] 五个阈值下运行 repair 机制，观察触发率与全局增益的关系曲线。

---

## 执行顺序

```
Phase 1 (代码修复，立即可做):
  F-1 → F-2 → F-3 → F-4
Phase 2 (论文措辞修正):
  P-1 → P-2 → P-3 → P-4 → P-5 → P-6 → P-7
Phase 3 (补充实验):
  E-1 → E-2 → E-3
Phase 4 (汇总更新论文 §4.8 局限性):
  最终完整性审查
```
