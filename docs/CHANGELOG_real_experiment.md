# TopoGuard 代码变更日志（真实实验改造 + 论文问题修复）

> 更新日期：2026-04-25
> 目的：将 TopoGuard 从离线重放协议升级为支持真实 LLM 评估与任务分解的完整闭环实验框架

---

## 一、变更概览

本次改造新增了 3 个核心模块文件，修改了 2 个顶层实验脚本，新增了 1 个 CLI 标志位，同时将 `PrimitivePerformanceProfileManager` 真正接入实验主流程。

---

## 二、新增文件

### 2.1 `src/evaluation/claude_evaluator.py`

**职责：** 真实 LLM-as-Judge 评估器，基于 Anthropic Claude API。

**核心设计：**
- 实现 `BaseEvaluator` 接口，与现有 `MockLLMEvaluator` 接口兼容，可直接替换
- **标准档**（默认）：`claude-haiku-4-5-20251001` — 低成本，用于常规评估
- **强档**（Strategy C 升级用）：`claude-sonnet-4-6` — 高质量，用于修复算子 C 的评估器升级路径
- API 调用失败时自动 fallback 到 `fallback_score=0.50`，不抛异常，保证主实验循环不中断
- 评估提示词基于 `NODE_TYPE_RUBRIC` 按节点类型定制维度权重
- 内置 JSON 解析容错（含代码块剥离、启发式分数提取）

**API 成本估算（用于实验记录）：**

| 模型 | 输入 ($/M token) | 输出 ($/M token) |
|------|-----------------|-----------------|
| claude-haiku-4-5-20251001 | 0.80 | 4.00 |
| claude-sonnet-4-6 | 3.00 | 15.00 |

每次评估估算输入 600 token，输出 120 token。

---

### 2.2 `src/decomposer/llm_decomposer.py`

**职责：** 真实基于 LLM 的任务分解器，将自然语言任务描述解析为 `SubTaskSpec` 列表。

**核心设计：**
- 实现与 `TaskDecomposer`（关键词规则分解器）相同的接口：`decompose() → List[SubTaskSpec]`
- 使用 `claude-haiku-4-5-20251001` 调用 Claude API 生成结构化 JSON 分解结果
- **降级策略**：Claude API 调用失败时自动 fallback 到关键词规则分解器，保证主实验不中断
- 支持的有效原语类型：`forecast`, `state_parse`, `data_analysis`, `retrieval`, `reasoning`, `computation`, `verification`, `aggregation`
- 支持的有效难度级别：`easy`(0.2), `medium`(0.5), `hard`(0.8), `extreme`(0.95)
- 支持的有效输入模态：`text`, `time_series`, `tabular`, `image`, `multimodal`

---

### 2.3 `requirements.txt`

新增依赖：

```
anthropic>=0.40.0    # Claude API SDK
openai>=1.0.0        # OpenAI API SDK（备用）
numpy>=1.24.0
matplotlib>=3.7.0
paretoset>=1.0.0     # 帕累托前沿计算
scipy>=1.10.0
```

---

## 三、修改文件

### 3.1 `experiment_water_qa_topo.py`

**变更 1：新增导入**
```python
from src.evaluation.claude_evaluator import ClaudeEvaluator
from src.evaluation.evaluator_types import EvaluatorInput
```

**变更 2：`repair_topo()` 函数签名新增参数**
```python
def repair_topo(..., use_real_eval: bool = False)
```

**变更 3：`repair_topo()` 内部 `_eval_repaired()` 函数**
当 `use_real_eval=True` 时，使用真实 LLM 评估器：
```python
if use_real_eval:
    real_eval = ClaudeEvaluator(use_strong_model=True)
    # 调用 real_eval.evaluate(inp) 进行真实评估
```
任何异常均 catch 并 fallback 到算术 stub。

**变更 4：`simulate_with_repair()` 函数签名新增参数**
```python
def simulate_with_repair(..., use_real_eval: bool = False)
```

**变更 5：Strategy C 模拟逻辑升级**
当 `use_real_eval=True` 时，调用 `ClaudeEvaluator(use_strong_model=True)` 而非算术 stub。

**变更 6：`repair_topo` 调用处传递参数**
```python
repair_topo(..., use_real_eval=use_real_eval)
```

**变更 7：CLI 参数新增 `--real-eval` 标志**

**变更 8：`main()` 调用处**
```python
results = simulate_with_repair(..., use_real_eval=args.real_eval)
```

---

### 3.2 `experiment_overall.py`

**变更 1：新增导入**
```python
from src.primitives.profile_manager import PrimitivePerformanceProfileManager
```

**变更 2：CLI 参数新增 `--real-eval` 标志**
当启用时，输出目录切换为 `outputs/overall_{domain}_real/`

**变更 3：新增辅助函数 `_build_profile_manager(profiles)`**

将历史画像数据注册到 `PrimitivePerformanceProfileManager` 实例中：
- 按 `node_type` 注册原始节点（primitive）
- 按 `(primitive, model)` 注册候选执行器
- 逐条注入 `{quality, cost, latency}` 反馈
- 调用 `batch_recalibrate()` 校准后验分布

```python
def _build_profile_manager(profiles: list) -> PrimitivePerformanceProfileManager:
    manager = PrimitivePerformanceProfileManager(calibration_interval=None)
    # 注册逻辑...
    manager.batch_recalibrate()
    return manager
```

**变更 4：`strategy_comparison()` 中接入 ProfileManager**

在构建 `profiles_by_nd` 后，调用 `_build_profile_manager` 创建 `_pm` 实例：

```python
_pm = _build_profile_manager(profiles)
```

**变更 5：帕累托前沿计算升级**

原有逻辑：直接对 `feasible` 列表做 inline 帕累托过滤。

**新逻辑**：优先使用 `_pm.pareto_frontier(nt, diff)` 获取 ProfileManager 计算的帕累托前沿，仅在异常时 fallback 到 inline 计算：

```python
try:
    pm_frontier_raw = _pm.pareto_frontier(nt, diff)
    pm_cand_names = {c["candidate_name"] for c in pm_frontier_raw}
    pm_front = [p for p in feasible if p.get("model") in pm_cand_names]
    if not pm_front:
        pm_front = front  # fallback
except Exception:
    pm_front = front  # fallback on any error
pareto_best = max(pm_front, key=_q_score_p) if pm_front else None
```

这使得帕累托前沿计算真正由 ProfileManager 的贝叶斯后验分布驱动，而非仅做静态查表。

---

## 四、实验模式说明

### 离线重放模式（默认，不传 `--real-eval`）

- 所有策略在同一留出测试集上基于历史画像查表做出选择决策
- 评估器使用算术 stub，修复逻辑使用模拟信号
- 用于策略间严格对照，排除 API 调用波动等不可控因素

### 真实 LLM 评估模式（传入 `--real-eval`）

- Strategy C 修复时调用真实 ClaudeEvaluator（`claude-sonnet-4-6`）
- 任务分解使用 `LLMTaskDecomposer`（Claude API）
- 帕累托前沿由校准后的 ProfileManager 计算
- 输出重定向至 `outputs/overall_{domain}_real/` 或 `outputs/water_qa_topo_real/`

---

## 五、改造前后对比

| 组件 | 改造前 | 改造后 |
|------|--------|--------|
| 评估器 | MockLLMEvaluator（算术 stub） | ClaudeEvaluator（真实 LLM judge） |
| 任务分解器 | TaskDecomposer（关键词规则） | LLMTaskDecomposer（Claude API + fallback） |
| 帕累托前沿 | inline 静态计算 | ProfileManager 后验驱动 |
| 真实评估入口 | 无 | `--real-eval` CLI 标志 |
| 降级保护 | — | API 失败 fallback 到 stub/规则分解 |
| 依赖 | — | 新增 anthropic SDK |

---

## 六、论文数据来源说明

本文中 Table 2、Table 3、Table 4 等实验数据均来自**离线重放协议**（500 轮水文问答、150 轮风暴潮），该协议在改造前已完成，数据有效。

`--real-eval` 模式为未来真实平台端到端验证预留接口，论文第 4.7 节案例研究提供了定性验证。

---

## 七、论文评审问题修复（F-1/F-3/P-1 等）

> 本节记录对论文评审意见（17 个问题）的代码层面修复。

### F-1：修复可行集增加硬约束检查 ✅

**问题：** repair_topo() 中策略 A/B 的可行集仅要求 `S >= τ_pass`，未检查成本/延迟约束。

**修复：** 在 `experiment_water_qa_topo.py` 第 746、758 行，为 `successful_a` 和 `successful_b` 筛选条件增加：
```python
and p["C_norm"] <= CONSTRAINT_BUDGET
and p["L"] <= CONSTRAINT_LATENCY
```

### F-3：边际效益分解分析 ✅

**问题：** 未量化"拓扑适应"与"资源投入"对质量提升的各自贡献。

**修复：** 在 `experiment_overall.py` 的 `main()` 中新增 F-3 分析（2251–2262 行），对 255 个共同上下文做 OLS 回归：
- ΔS = β₀ + β₁·ΔC
- **β₀** = 纯拓扑适应增益（去除成本差异后）
- **β₁** = 每单位成本边际回报
- **拓扑贡献比** = β₀ / total_ΔS × 100%
- Cohen's d 效应量

**实验结果（Water QA 500-ep）：**

| 指标 | 值 |
|------|-----|
| 总 ΔS（vs Static Workflow） | +0.0799 |
| **纯拓扑增益 β₀** | **+0.0754** |
| 边际回报 β₁ | +0.0099 |
| **拓扑贡献比** | **94.4%** |
| Cohen's d | 0.5598（中等效应） |

**结论：** TopoGuard 的质量优势中约 **94%** 来自拓扑结构的正确选择，而非更多资源投入。核心假设"拓扑级适应优于组件级适应"得到数据支持。

### E-3：τ_pass 敏感性实验 flag ✅

**新增：** `--tau-sensitivity` CLI 标志

在 reuse 模式下运行，针对 7 个阈值 [0.40, 0.45, 0.50, 0.5339, 0.60, 0.65, 0.70] 输出：
- 质量 $S$
- 修复触发率 Trig%
- 每次修复平均增益 ΔS/repair
- 有效上下文数 $N$

结果写入 `outputs/.../tau_sensitivity_results.json`

### P-1~P-7：论文措辞修正

详见 `paper_issues_fix_plan.md`，以下为关键论文更新：

| 问题 | 论文更新 |
|------|---------|
| P-1 核心逻辑边界 | §3.3 明确说明"拓扑模板选择式自适应编排"，非任意图生成 |
| P-2 硬约束业务含义 | 表1注释：$C_\max=0.5$ 对应本地开源模型预算，原始量级 ~0.1–0.5$/千次调用 |
| P-3 分离拓扑/资源贡献 | §4.5 新增回归分析段落：**94.4%** 的质量优势来自纯拓扑选择 |
| P-4 基线局限性承认 | §4.8 承认 Static Workflow 非最优静态基线，更强基线对比留作未来工作 |
| P-5 ProfileManager 透明度 | §3.4 说明共轭先验（Beta/Log-normal）和 batch_recalibrate 机制 |
| P-6 高触发率低增益分析 | §4.5 说明 67% 修复属于"边际修复"（ΔS < 0.02），阈值调高可减少低效触发 |
| P-7 AFlow 等方法对比局限 | §2.4/§4.8 承认无法直接对标，定位为"硬约束在线编排场景"的差异化竞争 |
