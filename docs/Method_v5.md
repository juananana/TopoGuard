# Method v5

## 1. Problem Definition（问题定义）

在复杂风险研判与多阶段决策任务中，系统通常需要调用多个模型/工具协同完成任务。与固定工作流不同，本文目标是在预算、延迟与可靠性约束下，联合优化：

1. 工作流结构（模板/拓扑）；
2. 节点执行器配置；
3. 执行失败后的局部修复动作。

本文方法可概括为：

> Difficulty-Aware Hierarchical Pareto Orchestration with Local Graph Repair

相较于 v4，本版明确引入“难度桶（difficulty bucket）”作为统一控制变量，并将初始决策写成两层帕累托筛选。

---

## 2. Workflow Representation（工作流表示）

工作流表示为有向图：

$$
G=(V,E,\tau,\phi)
$$

其中：

1. $V$：节点集合；
2. $E$：边集合；
3. $\tau(v)$：节点类型（executor/verifier/aggregator/human gate 等）；
4. $\phi(v)$：节点执行器（candidate）或评估器配置。

---

## 3. Decision Variables（决策变量）

给定任务 $X$，本文优化对象不只是单一图，而是策略：

$$
\pi=(\pi_{\text{init}}^{T},\pi_{\text{init}}^{N},\pi_{\text{repair}})
$$

1. $\pi_{\text{init}}^{T}$：模板级初始拓扑策略；
2. $\pi_{\text{init}}^{N}$：节点级执行器选择策略；
3. $\pi_{\text{repair}}$：局部修复策略。

其中：

$$
T_0=\pi_{\text{init}}^{T}(X,b),\quad
\Phi_0=\pi_{\text{init}}^{N}(X,b,T_0),\quad
G_0=(T_0,\Phi_0)
$$

$b$ 为难度桶。

---

## 4. Difficulty Bucket as a Control Variable（难度桶控制变量）

### 4.1 定义

任务先估计连续难度 $d\in[0,1]$，再离散映射为难度桶：

$$
b=\text{bucket}(d),\quad b\in\{\text{easy},\text{medium},\text{hard},\text{extreme}\}
$$

### 4.2 作用位置

难度桶在当前实现中同时作用于：

1. 模板级画像查询与模板前沿筛选；
2. 节点级执行器画像查询与候选前沿筛选；
3. 评估器初始档位选择；
4. 评估阶段的 GT 查表、执行观测噪声倍率与 rubric 阈值修正。

### 4.3 设计动机

难度桶不是后验统计标签，而是前向决策条件变量。其目标是让“同一 primitive 在不同难度下”的质量-成本-延迟分布可分层建模。

---

## 5. Workflow Performance Modeling（性能建模）

### 5.1 节点级观测

对执行器 $a$ 与任务 $X$，记录：

$$
(a,X,b)\rightarrow (S(a,X,b),C(a,X,b),L(a,X,b))
$$

### 5.2 工作流级聚合

设执行节点集合为 $V_{exec}$：

$$
S(G;X)=\frac{1}{|V_{exec}|}\sum_{v\in V_{exec}}S(\phi(v),X,b)
$$

$$
C(G;X)=\sum_{v\in V_{exec}}C(\phi(v),X,b)
$$

$$
L(G;X)=\sum_{v\in V_{exec}}L(\phi(v),X,b)
$$

其中主实验默认把 LLM decomposer/evaluator 调用成本单独统计为附加项，不并入主成本口径。

### 5.3 工作流效用

模板级效用函数为：

$$
Q_T(G;X)=\alpha S(G;X)-\beta C(G;X)-\gamma L(G;X)
$$

总体期望：

$$
Q_T(G)=\mathbb{E}_{X\sim D}[Q_T(G;X)]
$$

---

## 6. Hierarchical Pareto Initialization（分层帕累托初始化）

本节是 v5 的核心修改：将“初始优化”明确写成两层。

### 6.1 第一层：模板级三目标帕累托

给定任务类型、节点类型与难度桶，构建模板候选集 $\mathcal{T}(X,b)$，在三目标空间筛选前沿：

$$
\mathcal{P}_T=\text{Pareto}_{\max S,\min C,\min L}(\mathcal{T}(X,b))
$$

然后施加硬约束（质量下限、预算上限、延迟上限），在可行前沿上最大化 $Q_T$：

$$
T_0=\arg\max_{T\in\mathcal{P}_T^{\text{feasible}}}Q_T(T;X)
$$

### 6.2 第二层：节点级执行器帕累托

对每个执行节点 $v$，在其 primitive 对应候选集 $\mathcal{A}_v(b)$ 上筛选二维前沿：

$$
\mathcal{P}_N(v)=\text{Pareto}_{\max S,\min C}(\mathcal{A}_v(b))
$$

当前实现中，latency 作为硬约束过滤项（而非进入节点级效用）；在可行前沿上使用：

$$
Q_N(a;v,X)=\alpha_N S(a;v,X)-\beta_N C(a;v,X)
$$

并选择：

$$
\phi(v)=\arg\max_{a\in\mathcal{P}_N^{\text{feasible}}(v)}Q_N(a;v,X)
$$

### 6.3 与 v4 的关系

v4 将初始化写为单层候选工作流选择；v5 将其展开为“模板层 + 节点层”两级决策，更贴合当前实验代码结构。

---

## 7. Execution and Evaluation（执行与评估）

### 7.1 节点执行后的评估

当前主实验中，评估是每个执行节点后的必经步骤（逻辑必经），并输出：

$$
y_v\in\{\text{pass},\text{warn},\text{fail},\text{escalate}\}
$$

仅当 $y_v\in\{\text{fail},\text{escalate}\}$ 时进入局部修复门控。

### 7.2 评估中的难度桶影响

难度桶在评估阶段通过三条路径生效：

1. GT 查表按 $(primitive,candidate,b)$ 取基线；
2. 执行观测噪声按 primitive 与 $b$ 的倍率注入；
3. rubric 判定阈值按 $b$ 修正。

此外，评估器初始档位由难度桶驱动；低置信度/关键错误可触发评估器升档。

---

## 8. Local Repair Optimization（局部修复优化）

当节点失败时不全局重规划，而采用局部动作：

$$
a_t=\pi_{\text{repair}}(s_t),\quad G_{t+1}=T(G_t,a_t)
$$

当前主实验已实现的核心 repair 子集包括：

1. 执行器升级（candidate upgrade）；
2. 评估器升级（evaluator upgrade）；
3. 模板/结构升级（template/topology upgrade）；
4. 相关重试路径。

可写为代价-风险折中：

$$
\min_{\Delta G}\;\lambda_1|\Delta G|+\lambda_2 C_{extra}+\lambda_3 R(G+\Delta G;X)
$$

---

## 9. Overall Objective（整体目标）

综合初始化与修复，最终策略优化目标为：

$$
\max_{\pi}\;\mathbb{E}_{X\sim D}\big[Q_T(G_{\pi}(X);X)\big]
$$

其中随机性来自任务分布、执行噪声、评估不确定性与修复路径。

---

## 10. Method Overview（整体流程）

```text
Task X
  -> Difficulty/Constraint Analysis
  -> Template-level Pareto (S,C,L) + Q_T selection
  -> Node-level Pareto (S,C) + latency constraint + Q_N selection
  -> Execute node
  -> Evaluate node
  -> if fail/escalate: Local Repair
  -> Log EpisodeRecord / FeedbackRecord
  -> Update candidate profiles + template profiles
  -> Next episode
```

该流程构成完整闭环：

```text
结构选择 -> 节点选择 -> 执行评估 -> 局部修复 -> 画像更新 -> 再决策
```

---

## 11. One-Sentence Summary（一句话方法论）

We formulate adaptive workflow orchestration as a difficulty-aware hierarchical policy optimization problem, where template-level and node-level Pareto screening initialize workflows, and local repair policies adapt execution online under quality-cost-latency constraints.

中文：

我们将自适应工作流编排建模为“难度感知的分层策略优化”问题：先进行模板级与节点级两层帕累托筛选完成初始化，再通过局部修复在执行中动态调整，以在质量、成本与延迟约束下最大化整体效用。

---
# Appendix: Supplementary Methodological Details

> This appendix complements the main Method section and is intended for direct Overleaf integration.

## A. Difficulty Bucket Design and Scope

### A.1 Definition

We estimate a continuous difficulty score $d\in[0,1]$ and map it to a discrete bucket:

$$
b=\text{bucket}(d),\quad b\in\{\text{easy},\text{medium},\text{hard},\text{extreme}\}.
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
b=\text{bucket}(d),\quad b\in\{\text{easy},\text{medium},\text{hard},\text{extreme}\}.
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
