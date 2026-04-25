"""
feedback_record.py
==================
FeedbackRecord 数据类定义。

描述一条运行时反馈记录的 schema，
由 evaluator 执行后产生，交给 PrimitivePerformanceProfileManager 缓存。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
import numpy as np


@dataclass
class FeedbackRecord:
    """
    单条执行反馈记录。

    Attributes
    ----------
    task_id : str
        所属任务的全局 ID。
    node_id : str
        拓扑中具体节点的 ID（对应某个 SubTask 实例）。
    primitive_name : str
        该节点所属的原语模块名（如 "Q_forecast"）。
    candidate_name : str
        该节点实际选用的 candidate（agent 或 agent 组合）名称。
    difficulty : float
        该次执行时任务难度的原始值（归一化到 [0, 1]）。
    difficulty_bucket : str
        难度桶名称，可选值由 DifficultyBucket 定义（easy / medium / hard / extreme）。
    predicted_quality : float
        执行前 profile manager 给出的质量预测值。
    predicted_cost : float
        执行前 profile manager 给出的成本预测值。
    observed_quality : float
        evaluator 观测到的真实质量分数（0 ~ 1）。
    observed_cost : float
        evaluator 统计的真实执行成本。
    eval_pass : bool
        是否通过评估。
    failure_type : str | None
        失败类型，可能值：
        - "crash"        : 执行异常崩溃
        - "low_quality" : 质量低于阈值
        - "timeout"     : 超时
        - "error"       : 其他运行时错误
        - None          : 通过评估
    task_features : np.ndarray | None
        任务特征向量 φ（可选），用于后续模型化。
    episode : int
        所属的 episode 编号。
    timestamp : datetime
        记录生成时间。
    metadata : dict
        额外扩展字段。

    constraint_violations : List[dict]
        本次执行中检测到的违约束列表。
        每条: {"constraint_id": str, "constraint_type": str, "reason": str}
    execution_duration : float | None
        实际执行时长（秒）。
    human_approved : bool
        人工审批结果（True = 批准，False = 拒绝）。
    input_modality : str
        任务输入模态（"text" | "image" | "time_series" | "tabular" | "multimodal"）。
    intermediate_modality : str | None
        中间结果模态（可选）。
    violation_count : int
        违约束总数（简化为计数，供分析使用）。
    calibration_event_counter : int
        标记该记录是在第几次校准事件中产生的。
    """

    task_id: str
    node_id: str
    primitive_name: str
    candidate_name: str
    difficulty: float

    # 难度桶（由 DifficultyBucket 定义）
    difficulty_bucket: str

    # 执行前的预测值
    predicted_quality: float
    predicted_cost: float

    # 执行后的真实观测值
    observed_quality: float
    observed_cost: float

    # 评估结果
    eval_pass: bool
    failure_type: Optional[str] = None

    # 任务特征向量（可选，用于后续模型化）
    task_features: Optional[np.ndarray] = None

    # 元信息
    episode: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)
    calibration_event_counter: int = -1

    # === 约束与多模态字段 ===
    constraint_violations: List[Dict[str, Any]] = field(default_factory=list)
    execution_duration: Optional[float] = None
    human_approved: bool = True
    input_modality: str = "text"
    intermediate_modality: Optional[str] = None
    violation_count: int = 0
    evaluator_name: str = "rule_eval"   # evaluator type used for this execution
    # === 新增：结构化 evaluator 字段 ===
    evaluator_id: str = "rule_eval"         # 同 evaluator_name，显式标识
    error_type: Optional[str] = None     # "low_quality" | "format_error" | "inconsistent_output" | ...
    confidence: float = 1.0              # evaluator 置信度 [0, 1]
    evaluator_latency: float = 0.0        # 评估耗时（秒）
    evaluator_cost: float = 0.0          # 评估 API 成本
    # === 新增：上下文信息（便于按 task_type / node_type 查询 profile）===
    task_type: str = "unknown"           # "time_series" | "text_analysis" | "tabular_analysis" | ...
    node_type: str = "unknown"           # "forecast" | "state_parse" | "data_analysis" | ...
    template_id: str = "unknown"         # 使用的拓扑模板 ID

    def to_dict(self) -> dict:
        """序列化为字典（task_features 转为 list）"""
        d = {
            "task_id": self.task_id,
            "node_id": self.node_id,
            "primitive_name": self.primitive_name,
            "candidate_name": self.candidate_name,
            "difficulty": self.difficulty,
            "difficulty_bucket": self.difficulty_bucket,
            "predicted_quality": self.predicted_quality,
            "predicted_cost": self.predicted_cost,
            "observed_quality": self.observed_quality,
            "observed_cost": self.observed_cost,
            "eval_pass": self.eval_pass,
            "failure_type": self.failure_type,
            "task_features": (
                self.task_features.tolist()
                if self.task_features is not None else None
            ),
            "episode": self.episode,
            "calibration_event_counter": self.calibration_event_counter,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            # === 约束与多模态字段 ===
            "constraint_violations": self.constraint_violations,
            "execution_duration": self.execution_duration,
            "human_approved": self.human_approved,
            "input_modality": self.input_modality,
            "intermediate_modality": self.intermediate_modality,
            "violation_count": self.violation_count,
            "evaluator_name": self.evaluator_name,
            # === 新增 evaluator 字段 ===
            "evaluator_id": self.evaluator_id,
            "error_type": self.error_type,
            "confidence": self.confidence,
            "evaluator_latency": self.evaluator_latency,
            "evaluator_cost": self.evaluator_cost,
            "task_type": self.task_type,
            "node_type": self.node_type,
            "template_id": self.template_id,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FeedbackRecord":
        """从字典反序列化"""
        d = dict(d)  # 复制，避免修改原字典
        if d.get("task_features") is not None:
            d["task_features"] = np.array(d["task_features"])
        if isinstance(d.get("timestamp"), str):
            d["timestamp"] = datetime.fromisoformat(d["timestamp"])
        # 移除不在 dataclass 字段中的键
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        extra = {k: v for k, v in d.items() if k not in field_names}
        if extra:
            existing_meta = d.get("metadata", {})
            d["metadata"] = {**existing_meta, **extra}
        return cls(**{k: v for k, v in d.items() if k in field_names})
