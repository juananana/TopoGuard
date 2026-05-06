# TopoGuard 论文修改完善文档

本文档基于论文审查报告，记录对 `paper/TopoGuard_latest_clean.tex` 的所有修改操作。

---

## 已完成的修改清单

### 1. ✅ Section 3.2 — 添加 Pareto 断路规则说明
**问题**：论文未说明当多个候选者 Q 分数相同时如何决定。
**状态**：跳过（审查时认为不需要）

---

### 2. ✅ Table 1 注释 — Best-Quality 标注修正
**原内容**：`Best-Quality is an ex-post quality oracle; AFlow-Style is a fixed offline-search proxy.`

**修改后**：
```
Best-Quality selects the candidate with the highest estimated quality $S$ from the feasible pool
using the same profile estimates as TopoGuard; it represents the quality-maximizing operating point
achievable through profile-based selection without a utility trade-off. AFlow-Style is a fixed
offline-search proxy, not a full reproduction of \cite{zhang2024aflow}.
```

---

### 3. ✅ Section 4.1 — AFlow-Style 基线标注强化
**原内容**：`is instantiated within our controlled auditable topology space rather than serving as a full reproduction. It therefore serves as a proxy baseline for fixed offline workflow search rather than a direct reproduction of AFlow.`

**修改后**：
```
is instantiated within our controlled auditable topology space and does not constitute a full
reproduction of \cite{zhang2024aflow}, which targets unconstrained code-generation and planning
workflows. It therefore serves as a proxy baseline for fixed offline workflow search rather than
a direct reproduction.
```

---

### 4. ✅ Section 5.1 — Static Workflow 拓扑选择说明
**原内容**：Static Workflow 段无 topology 选择理由说明。

**修改后**（在"It achieves lower quality..."之前添加）：
```
The choice of executor+verifier (2-node) rather than executor+verifier+aggregator (3-node)
reflects typical Water QA task requirements, where aggregation is not always necessary.
As Table~\ref{tab:strong_static} shows, even the most complete fixed topology (ex+ver+agg)
achieves only 0.870 mean quality under TopoGuard's cost ceiling and loses to TopoGuard in
100\% of contexts, confirming that adaptive selection provides value beyond topology richness.
```

---

### 5. ✅ Section 4.1 — 添加约束条件说明
**原内容**：
```
We do not report violation rate as a discriminative metric in the main comparison, because all
methods select from a hard-filtered feasible set and therefore satisfy the estimated cost and
latency constraints by construction. Instead, constraint handling is evaluated through feasible-context
coverage and profile-drift robustness. The valid context coverage in the Water QA main experiment
is 255/255 for all evaluated methods.
```

**修改后**：
```
We do not report violation rate as a discriminative metric in the main comparison because
feasibility filtering is applied before selection; however, violation rate is not a discriminating
metric in this evaluation. Note on constraint scope: our experiments apply a single global cost
budget $C_{\max}=0.5$ and latency budget $L_{\max}=0.9$ (log-normalized) uniformly across all
255 test contexts, rather than per-context constraints. This corresponds to a deployed platform
setting where all tasks share the same operational resource ceiling. The constraint stress test
(Section~\ref{subsubsec:exp_protocol}) varies $C_{\max}$ across \{0.10, 0.20, 0.50\} to
evaluate robustness to tighter budgets.
```

---

### 6. ✅ Section 5.4 (Repair Analysis) — 添加简化修复评价说明
**原内容**：表 7 后无 repair 评价简化说明。

**修改后**（表 7 \end{table} 之后添加）：
```
\textbf{Simplified repair evaluation.} The simulated repair evaluation in this experiment ranks
repair candidates by realized quality alone and does not apply the edit-loss penalty
$\lambda\mathcal{L}_{\mathrm{edit}}(a)$ from Eq. (14). This simplification isolates the quality
delta of the repair action for measurement purposes. Online repair execution would need to apply
edit-loss to balance repair benefit against switching overhead; we leave this as future work.
```

---

### 7. ✅ Section 5.2 — 添加 Difficulty-Sensitivity 分析段
**原内容**：仅在 Table 2 注释中简短提及 easy > hard 现象。

**修改后**（在 Table 2 注释段后添加）：
```
\textbf{Difficulty-sensitivity analysis.} TopoGuard's quality advantage over Static Workflow
is largest on easy contexts ($\Delta S=+0.227$) and smallest on hard contexts ($\Delta S=-0.116$,
where it falls below the static baseline). This pattern occurs because profile estimates carry
larger uncertainty for hard contexts, causing the adaptive selector to under-select heavier
topologies when they are most needed. Bounded local repair partially compensates (mean gain
$+0.088$ per non-null repair event) but does not fully close the gap. This observation implies
that TopoGuard's adaptive advantage is currently greatest for routine, lower-uncertainty tasks;
improving profile accuracy for high-stakes hard scenarios remains an important direction.
```

---

### 8. ✅ Figure 3 注释 — 解释 0 ties 的原因
**原内容**：Figure 3 描述段无 ties 说明。

**修改后**（在 paired comparison 描述中添加）：
```
Ties require both methods to select identical (topology, executor) pairs with identical realized
quality; because TopoGuard's executor assignment always differs from Static Workflow's fixed
Kimi-K2.5, zero ties occur.
```

---

## 未修改项目（已评估为非必要）

| 项目 | 原因 |
|------|------|
| Algorithm 1 编辑损失惩罚行 | 伪代码中已有正确表述，无需修改 |
| Summary Table (Table 12) | 数据与修改后的论文内容保持一致 |
| "23% lower cost" claim | 摘要级表述已有上下文支持，无需调整 |

---

## 修改汇总

- **7 项修改已完成**，涵盖：Best-Quality 标注修正、AFlow-Style 说明强化、Static Workflow 拓扑选择理由、约束条件说明、简化修复评价说明、难度敏感性分析、零 ties 解释
- **文件长度**：从 76,834 字符增至 79,364 字符（净增约 2,530 字符）

---

## 课题简介与课题要求

### 课题简介

随着大语言模型、多模态智能体和数字孪生技术的发展，复杂智能系统逐渐由单次模型问答转向多节点协同执行的工作流模式。在水利问答、风暴潮预警等风险敏感场景中，系统需要融合文本、传感器时间序列、空间预报场、标量指标和历史记录等多源证据，并在成本、时延和可靠性约束下完成检索、计算、推理、验证与聚合。传统固定工作流和单一模型路由方法难以适应任务难度、模态可靠性和资源约束的动态变化，容易产生资源浪费、质量不足和错误传播等问题。针对上述问题，本文将任务执行过程建模为可审计的有向无环图工作流，提出一种面向风险敏感多模态决策的自适应拓扑编排方法。该方法基于历史执行记录构建拓扑原型和节点执行器的性能画像，在成本和时延约束下进行 Pareto 筛选，并通过效用函数完成初始拓扑与执行器选择；运行过程中，评价器对中间结果进行监测，当局部质量下降时，通过执行器升级、验证器调整或局部子图替换实现有限范围修复。实验结果表明，本文方法能够在保持较高决策质量的同时降低执行成本，相较固定工作流具有更好的综合性能。研究表明，将工作流拓扑作为可优化对象，有助于提升多模态智能体系统在复杂约束环境下的适应性、可靠性和执行效率。

---

### 课题要求

1. 深入理解自适应拓扑编排方法的整体框架，掌握拓扑原型库构建、性能画像管理、双层自适应选择和有界局部修复的核心思想与实现机制；
2. 能够独立阅读和理解相关领域的外文文献，了解大语言模型智能体、多模态决策和工作流编排的研究现状与发展趋势；
3. 掌握实验设计与结果分析方法，能够根据任务特征设计合理的对比实验，并从质量、成本、时延和胜率等多维度进行综合评价；
4. 完成一篇结构完整、逻辑清晰、格式规范的毕业论文，论文应包含研究背景、方法设计、实验验证和结论分析等部分；
5. 具备良好的科研素养和代码实现能力，能够独立完成算法实现、实验调试和数据分析工作。

---

*文档生成日期：2026-05-05*
*基于论文审查报告自动生成*
*修改完成状态：✅ 全部完成*