"""
evaluator_types.py
==================
Evaluator 统一抽象：BaseEvaluator 接口 + EvaluatorInput/Output 结构 + 节点级 rubric。

设计原则：
- 与 executor 一样是可插拔组件（BaseEvaluator 抽象接口）
- EvaluatorInput/Output 为结构化 I/O，所有字段可被 repair 和 ProfileManager 直接消费
- NODE_TYPE_RUBRIC 将 rubric 配置外部化，新增节点类型无需改代码

使用方式：
- 新增 evaluator = 实现 BaseEvaluator + 注册到 EvaluatorRegistry
- 替换 mock → 真实 LLM evaluator：只需实现 BaseEvaluator，不改主流程
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any


# ---------------------------------------------------------------------------
# Evaluator Output Level (4-level, per method definition Section 7)
# ---------------------------------------------------------------------------

class EvalLevel(Enum):
    """
    Four-level evaluator output, per method definition Section 7:

        y_v ∈ {pass, warn, fail, escalate}

    Each level has a distinct semantic meaning and triggers different actions:

    | Level     | Meaning                                      | Action               |
    |-----------|----------------------------------------------|-----------------------|
    | PASS      | Output quality meets all criteria             | Continue normally     |
    | WARN      | Quality borderline; may degrade downstream    | Continue + log warning|
    | FAIL      | Quality below threshold; repair needed        | Trigger local repair  |
    | ESCALATE  | Critical issue; needs immediate intervention  | Upgrade evaluator     |

    Design rationale:
    - PASS/WARN are separated because borderline quality can cascade through
      downstream nodes; a warn allows early intervention without full repair.
    - ESCALATE is distinct from FAIL because format errors / unsafe decisions
      require evaluator upgrade, not candidate upgrade (different repair path).
    """
    PASS     = "pass"
    WARN     = "warn"
    FAIL     = "fail"
    ESCALATE = "escalate"

    def needs_repair(self) -> bool:
        """True if this level should trigger repair."""
        return self in (EvalLevel.FAIL, EvalLevel.ESCALATE)

    def needs_evaluator_upgrade(self) -> bool:
        """True if this level should trigger evaluator upgrade (not candidate upgrade)."""
        return self == EvalLevel.ESCALATE

    def is_acceptable(self) -> bool:
        """True if output is passable (pass or warn)."""
        return self in (EvalLevel.PASS, EvalLevel.WARN)


# ---------------------------------------------------------------------------
# Evaluator Input / Output
# ---------------------------------------------------------------------------

@dataclass
class EvaluatorInput:
    """
    Evaluator 的结构化输入。

    Attributes
    ----------
    task_type : str
        任务域，如 "time_series"、"text_analysis"、"tabular_analysis"、"multimodal"。
    node_type : str
        节点类型，如 "forecast"、"state_parse"、"data_analysis"、"aggregator"。
    node_id : str
        拓扑中节点 ID。
    template_id : str
        所用拓扑模板 ID（来自 TopologyTemplate.template_id）。
    primitive_name : str
        执行器所属 primitive（如 "forecast"、"state_parse"）。
    candidate_name : str
        被评估的执行器名称（如 "fast_nn"、"llm_large"）。
    difficulty : float
        归一化难度值 [0, 1]。
    difficulty_bucket : str
        难度桶：easy / medium / hard / extreme。
    input_payload : Any
        任务原始输入（evaluator 参考用）。
    candidate_output : Any
        executor 的执行输出（evaluator 评判对象）。
    reference_output : Any, optional
        参考输出（如果有客观指标，用于 ground-truth 对照）。
    context : dict, optional
        前序节点输出（用于端到端质量判断）。
    rubric_name : str, optional
        指定使用的 rubric 名称（默认按 node_type 自动选择）。
    budget_remaining : float, optional
        剩余预算（供 evaluator 决策参考，如选择是否升级）。
    metadata : dict
        额外上下文。
    """

    task_type: str
    node_type: str
    node_id: str
    template_id: str
    primitive_name: str
    candidate_name: str
    difficulty: float
    difficulty_bucket: str
    input_payload: Any
    candidate_output: Any
    reference_output: Any = None
    context: Optional[dict] = None
    rubric_name: Optional[str] = None
    budget_remaining: Optional[float] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class EvaluatorOutput:
    """
    Evaluator 的结构化输出。

    所有字段均可被 repair 模块和 ProfileManager 直接消费。

    Attributes
    ----------
    evaluator_id : str
        执行此评估的 evaluator 标识。
    quality_score : float
        质量分数 [0, 1]。含义取决于 rubric：
        - 有客观指标时：与 ground truth 对照的 ACC/F1/BLEU 等
        - 无客观指标时：judge-based 质量评分（不代表绝对 ACC）
    pass : bool
        是否通过评估（pass = True）。
    error_type : str | None
        具体错误类型，供 repair 策略使用：
        - "low_quality"         : 质量低于阈值
        - "format_error"        : 输出格式不合规
        - "inconsistent_output"  : 与 context 或 reference 不一致
        - "unsafe_decision"    : 决策违反安全约束
        - "insufficient_evidence": 证据不足
        - "timeout"            : 执行超时
        - "crash"             : 执行崩溃
        - "unknown"           : 其他错误
        - None                : 无错误
    confidence : float
        评估置信度 [0, 1]。用于判断是否需要升级到更强的 evaluator 或 HCI。
        - confidence < 0.5：evaluator 本身判断不确定，建议升级
        - confidence >= 0.5：判断可信，可用于 profile 更新
    latency : float
        评估耗时（秒）。
    api_cost : float
        API 调用成本。
    human_cost : float
        人工介入成本（如有人工审批步骤则为 > 0）。
    constraint_violations : List[dict]
        约束违例列表 [{"constraint_id", "constraint_type", "reason"}, ...]。
    evaluator_profile_snapshot : dict, optional
        调用时的 evaluator profile 快照（precision/recall 等），用于 EMA 更新。
    metadata : dict
        额外输出字段。
    """

    evaluator_id: str
    quality_score: float
    passed: bool  # renamed from "pass" to avoid Python reserved keyword
    error_type: Optional[str] = None
    confidence: float = 1.0
    latency: float = 0.0
    api_cost: float = 0.0
    human_cost: float = 0.0
    constraint_violations: List[dict] = field(default_factory=list)
    evaluator_profile_snapshot: Optional[dict] = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Base Evaluator Interface
# ---------------------------------------------------------------------------

class BaseEvaluator(ABC):
    """
    Evaluator 的抽象基类。

    所有 evaluator（mock / rule-based / LLM-based / human-based）
    必须实现此接口，实现后可注册即插即用。

    新增真实 evaluator 步骤：
    1. 实现 BaseEvaluator（实现 evaluate() 方法）
    2. 注册到 EvaluatorRegistry（或直接注入 MockEvaluator.profile_store）
    3. 无需修改主实验流程代码
    """

    @property
    @abstractmethod
    def evaluator_id(self) -> str:
        """Evaluator 全局唯一标识（如 "rule_eval"、"llm_judge"、"large_eval"）"""
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """人类可读名称"""
        raise NotImplementedError

    @property
    @abstractmethod
    def supported_node_types(self) -> List[str]:
        """此 evaluator 支持的节点类型列表"""
        raise NotImplementedError

    @property
    @abstractmethod
    def supported_task_types(self) -> List[str]:
        """此 evaluator 支持的任务域列表"""
        raise NotImplementedError

    @property
    def latency_mean(self) -> float:
        """平均评估耗时（秒），来自 evaluator profile"""
        return 0.0

    @property
    def api_cost_mean(self) -> float:
        """平均 API 成本，来自 evaluator profile"""
        return 0.0

    @abstractmethod
    def evaluate(self, inp: EvaluatorInput) -> EvaluatorOutput:
        """
        执行评估。

        Parameters
        ----------
        inp : EvaluatorInput
            结构化输入。

        Returns
        -------
        EvaluatorOutput
            结构化输出，包含 quality_score / pass / error_type / confidence 等。
        """
        raise NotImplementedError

    def get_profile_snapshot(self, difficulty: str) -> dict:
        """
        返回此 evaluator 在给定难度下的 profile 快照。
        用于 ProfileStore.update_evaluator_profile() 的 EMA 更新。
        """
        return {
            "evaluator_id": self.evaluator_id,
            "difficulty": difficulty,
            "latency_mean": self.latency_mean,
            "api_cost_mean": self.api_cost_mean,
        }


# ---------------------------------------------------------------------------
# Node-type-aware Rubric Registry
# ---------------------------------------------------------------------------

# Rubric 定义：每个条目描述该节点类型的评估维度及其权重。
#
# 结构：
#   rubric_name: {
#     "dimensions": [
#       (dimension_name, weight, evaluation_method),  # weight 和为 1.0
#     ],
#     "pass_threshold_score": float,  # quality_score >= 此值则 pass
#     "error_types": [str, ...],     # 此节点类型可能产生的 error_type 列表
#   }
#
# evaluation_method:
#   "objective" : 有客观指标，直接用指标值（如 ACC/F1/BLEU/NSE）
#   "judge"     : 无客观指标，用 judge 模型评分（precision 决定噪声大小）
#   "rule"      : 规则判定，二元结果（0 或 1）

NODE_TYPE_RUBRIC: Dict[str, dict] = {
    # forecast / time_series 类节点：支持客观指标 + judge 辅助
    "forecast": {
        "dimensions": [
            ("accuracy",          0.50, "objective"),  # NSE/RMSE/MAE 等客观指标
            ("consistency",       0.25, "judge"),       # 时间序列一致性
            ("usability",         0.25, "judge"),       # 下游可用性
        ],
        "pass_threshold_score": 0.60,
        "error_types": ["low_quality", "format_error", "inconsistent_output"],
    },
    # state_parse 类节点：结构化解析
    "state_parse": {
        "dimensions": [
            ("correctness",    0.45, "objective"),  # token match / exact match rate
            ("consistency",    0.30, "judge"),       # 与参考结构一致性
            ("format",         0.25, "rule"),        # 是否符合目标 schema
        ],
        "pass_threshold_score": 0.65,
        "error_types": ["low_quality", "format_error", "inconsistent_output"],
    },
    # text_analysis 类节点：文本理解与抽取
    "text_analysis": {
        "dimensions": [
            ("relevance",              0.40, "judge"),
            ("evidence_support",       0.35, "judge"),
            ("hallucination_penalty", 0.25, "judge"),
        ],
        "pass_threshold_score": 0.60,
        "error_types": ["insufficient_evidence", "inconsistent_output", "unsafe_decision"],
    },
    # data_analysis / tabular 类节点
    "data_analysis": {
        "dimensions": [
            ("correctness",            0.50, "objective"),  # 分类 accuracy / 指标达标
            ("interpretability",       0.30, "judge"),
            ("constraint_compliance",  0.20, "rule"),
        ],
        "pass_threshold_score": 0.65,
        "error_types": ["low_quality", "unsafe_decision", "format_error"],
    },
    # reasoning 类节点：推理链质量
    "reasoning": {
        "dimensions": [
            ("correctness",       0.40, "judge"),
            ("logical_coherence", 0.35, "judge"),
            ("evidence_support",  0.25, "judge"),
        ],
        "pass_threshold_score": 0.60,
        "error_types": ["inconsistent_output", "insufficient_evidence", "unsafe_decision"],
    },
    # aggregator 类节点：汇聚并行分支输出
    "aggregator": {
        "dimensions": [
            ("merge_correctness",    0.60, "judge"),
            ("divergence_handling",  0.40, "judge"),
        ],
        "pass_threshold_score": 0.65,
        "error_types": ["inconsistent_output", "format_error"],
    },
    # decision 类节点（预留）：决策类任务
    "decision": {
        "dimensions": [
            ("correctness",             0.45, "judge"),
            ("constraint_compliance",    0.35, "rule"),
            ("rationale_consistency",    0.20, "judge"),
        ],
        "pass_threshold_score": 0.70,
        "error_types": ["unsafe_decision", "low_quality", "insufficient_evidence"],
    },
    # default rubric（兜底）
    "__default__": {
        "dimensions": [
            ("quality", 1.0, "judge"),
        ],
        "pass_threshold_score": 0.60,
        "error_types": ["low_quality"],
    },
}


def get_rubric(node_type: str) -> dict:
    """
    根据 node_type 获取对应 rubric，无则返回 __default__。

    Parameters
    ----------
    node_type : str

    Returns
    -------
    dict
        rubric 定义字典，包含 dimensions / pass_threshold_score / error_types。
    """
    return NODE_TYPE_RUBRIC.get(node_type, NODE_TYPE_RUBRIC["__default__"])
