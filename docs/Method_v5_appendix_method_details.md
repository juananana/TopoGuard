# Appendix: Supplementary Methodological Details

> This appendix complements the main Method section and is intended for direct Overleaf integration.

## A. Difficulty Bucket Design and Scope

### A.1 Definition

We estimate a continuous difficulty score $d\in[0,1]$ and map it to a discrete bucket:

$$
b=\text{bucket}(d),\quad b\in\{\text{easy},\text{medium},\text{hard}\}.
$$

### A.2 Practical Effect in Current Implementation

In the current main experimental setting, difficulty bucket $b$ is used primarily for:

1. template profile lookup and template-level candidate filtering,
2. executor profile lookup and node-level candidate filtering,
3. context-aligned retrieval during repair candidate search.

Notably, in the current main scripts, we do **not** apply difficulty-dependent rubric threshold shifts in the core evaluation gate.

### A.3 Motivation

Difficulty buckets are treated as forward control variables (not post-hoc analysis labels), enabling stratified quality-cost-latency estimation for the same primitive under different difficulty regimes.

## B. Performance Modeling Details

### B.1 Node-Level Observation

For an executor candidate $a$ on task $X$ under bucket $b$, we model observations as:

$$
(a,X,b)\rightarrow (S(a,X,b),C(a,X,b),L(a,X,b)).
$$

### B.2 Workflow-Level Aggregation

For executor-node subset $V_{exec}$:

$$
S(G;X)=\frac{1}{|V_{exec}|}\sum_{v\in V_{exec}}S(\phi(v),X,b),
$$

$$
C(G;X)=\sum_{v\in V_{exec}}C(\phi(v),X,b),
$$

$$
L(G;X)=\sum_{v\in V_{exec}}L(\phi(v),X,b).
$$

In our reporting protocol, decomposer/evaluator costs can be tracked as additional cost channels and excluded from the primary executor-cost metric when needed.

## C. Local Repair Action Parameterization

To align with implementation, we instantiate three repair operators:

1. **Topology/template upgrade (A)**:

$$
C' = C + \kappa_{\text{topo}}\Delta n,\quad \Delta n\ge 1.
$$

2. **Executor upgrade (B)**:

$$
(m,t,b,n)\rightarrow(m',t,b,n),\quad \Delta n=0.
$$

3. **Evaluator upgrade (C)**:

$$
S'=\min(1,S+\delta_{\text{eval}}),\quad C'=C+\kappa_{\text{eval}},\quad \Delta n=0.
$$

This parameterization yields a compact and reproducible implementation-level abstraction while remaining methodologically interpretable.

## D. Evaluation/Repair Boundary Clarification

The framework-level evaluator can support multi-level outputs (pass/warn/fail/escalate). However, in the main experiment protocol used for primary plots and quantitative comparison, repair triggering is governed by a unified threshold gate:

$$
S_v<\tau_{\text{pass}}\Rightarrow \text{trigger repair},\quad \tau_{\text{pass}}=0.5246.
$$

This distinction prevents mismatch between conceptual framework capability and the concrete experimental setting.

## E. Extended Symbol Table (Appendix Scope)

| Symbol | Meaning |
|---|---|
| $T_0,\Phi_0,G_0$ | Initial template, node assignment, and instantiated workflow |
| $Q_N$ | Node-level utility function |
| $V_{exec}$ | Executor-node subset used for aggregation |
| $s_t$ | Repair-time state |
| $p$ | Candidate state tuple $(S,C,L,m,t,b,n)$ |
| $m,t,n$ | Model ID, topology ID, node-type/context ID |
| $\Delta n$ | Number of additional nodes introduced by repair |
| $\kappa_{\text{topo}}$ | Topology-repair cost coefficient |
| $\delta_{\text{eval}}$ | Evaluator-upgrade quality gain |
| $\kappa_{\text{eval}}$ | Evaluator-upgrade cost increment |
| $\mathcal{F}(p)$ | Feasible repaired-candidate set |
| $p^*$ | Selected repaired candidate |
| $\eta$ | Penalty when no feasible repair exists |
| $\alpha_N,\beta_N$ | Node-level utility weights |

## F. Suggested Appendix Positioning in Paper

Recommended placement in an ACM MM manuscript:

1. Put Sections A and D immediately after the experimental protocol appendix to clarify implementation fidelity.
2. Put Sections B and C near reproducibility details for metric and repair parameter definitions.
3. Put Section E as the final appendix table to support notation lookup during rebuttal/review.

---

# 中文审核版（Appendix 对应）

## A. 难度桶设计与作用范围

### A.1 定义

我们先估计连续难度分数 $d\in[0,1]$，再映射为离散难度桶：

$$
b=\text{bucket}(d),\quad b\in\{\text{easy},\text{medium},\text{hard}\}.
$$

### A.2 当前实现中的实际作用

在当前主实验设定中，难度桶 $b$ 主要用于：

1. 模板画像查询与模板级候选过滤；
2. 执行器画像查询与节点级候选过滤；
3. 修复阶段按上下文对齐的候选检索。

需要强调的是：在当前主实验脚本中，核心评估门控并不采用按难度动态变化的 rubric 阈值。

### A.3 设计动机

难度桶作为前向控制变量（而非后验分析标签），使同一 primitive 在不同难度条件下的质量-成本-延迟分布可分层建模。

## B. 性能建模细节

### B.1 节点级观测

对于难度桶 $b$ 下任务 $X$ 上的执行器候选 $a$，观测建模为：

$$
(a,X,b)\rightarrow (S(a,X,b),C(a,X,b),L(a,X,b)).
$$

### B.2 工作流级聚合

对执行节点子集 $V_{exec}$：

$$
S(G;X)=\frac{1}{|V_{exec}|}\sum_{v\in V_{exec}}S(\phi(v),X,b),
$$

$$
C(G;X)=\sum_{v\in V_{exec}}C(\phi(v),X,b),
$$

$$
L(G;X)=\sum_{v\in V_{exec}}L(\phi(v),X,b).
$$

在实验报告中，decomposer/evaluator 成本可作为附加成本通道单独统计，并在需要时不计入主 executor 成本口径。

## C. 局部修复动作参数化

为与实现对齐，我们采用三类修复算子：

1. **模板/拓扑升级（A）**：

$$
C' = C + \kappa_{\text{topo}}\Delta n,\quad \Delta n\ge 1.
$$

2. **执行器升级（B）**：

$$
(m,t,b,n)\rightarrow(m',t,b,n),\quad \Delta n=0.
$$

3. **评估器升级（C）**：

$$
S'=\min(1,S+\delta_{\text{eval}}),\quad C'=C+\kappa_{\text{eval}},\quad \Delta n=0.
$$

该参数化在保证可解释性的同时，与实现行为保持紧密一致，便于复现实验。

## D. 评估与修复边界说明

框架层评估器可支持多级输出（pass/warn/fail/escalate）。但在主实验协议（用于主要图表与核心量化结论）中，修复触发由统一阈值门控决定：

$$
S_v<\tau_{\text{pass}}\Rightarrow \text{trigger repair},\quad \tau_{\text{pass}}=0.5246.
$$

这一边界说明可避免“框架能力”与“具体实验设定”之间的口径不一致。

## E. 扩展符号表（附录范围）

| 符号 | 含义 |
|---|---|
| $T_0,\Phi_0,G_0$ | 初始模板、节点分配与实例化工作流 |
| $Q_N$ | 节点级效用函数 |
| $V_{exec}$ | 用于聚合的执行节点子集 |
| $s_t$ | 修复时刻状态 |
| $p$ | 候选状态元组 $(S,C,L,m,t,b,n)$ |
| $m,t,n$ | 模型 ID、拓扑 ID、节点类型/上下文 ID |
| $\Delta n$ | 修复引入的新增节点数 |
| $\kappa_{\text{topo}}$ | 拓扑修复成本系数 |
| $\delta_{\text{eval}}$ | 评估器升级质量增益 |
| $\kappa_{\text{eval}}$ | 评估器升级成本增量 |
| $\mathcal{F}(p)$ | 可行修复候选集合 |
| $p^*$ | 被选中的修复候选 |
| $\eta$ | 无可行修复时的惩罚项 |
| $\alpha_N,\beta_N$ | 节点级效用权重 |

## F. 论文附录放置建议

在 ACM MM 论文中，建议如下放置：

1. 将 A 与 D 放在实验协议附录之后，用于澄清实现口径；
2. 将 B 与 C 放在可复现性说明附近，用于给出指标与修复参数定义；
3. 将 E 作为附录最后的符号检索表，便于审稿与答辩阶段快速查阅。
