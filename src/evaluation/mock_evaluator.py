"""
mock_evaluator.py
=================
Learning-based Evaluator for TopoGuard.

Evaluator 职责：
    Given (candidate, sub_task, execution_result) -> (observed_quality, observed_cost, eval_pass, error_type)

评估器是基于学习方法（不是 LLM 或人工标注）：
- 初始 acc-cost 基准来自 Ground Truth 曲线（DEFAULT_GROUND_TRUTH）
- 随着执行历史累积，通过 EMA 逐步更新 acc-cost 估计
- acc-cost 曲线通过大量任务执行后求均值得到：
    * 4/5 任务用于训练（更新 profile 的 EMA 均值）
    * 1/5 任务用于测试（验证泛化性能）
- 帕累托前沿完全无监督：只用支配关系（dominance），无需人工标注"方案 A 比方案 B 好"

Evaluator 输出四级结果（per 方法定义 Section 7）：
    pass / warn / fail / escalate

Constraint validation:
    The evaluator also receives SubTaskSpec (which may carry constraints) and
    validates observed results against them, producing constraint_violations.
    This is the "evaluator semantic" layer that checks hard constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Any, List, TYPE_CHECKING
import numpy as np
from pathlib import Path
import json
import random

if TYPE_CHECKING:
    from src.decomposer.task_decomposer import SubTaskSpec

from src.evaluation.evaluator_types import BaseEvaluator, EvaluatorOutput


# ---------------------------------------------------------------------------
# Ground Truth Curves for MVP experiments
# ---------------------------------------------------------------------------

# Format: {primitive: {candidate: {bucket: (true_acc, true_cost)}}}
DEFAULT_GROUND_TRUTH: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]] = {
    "forecast": {
        "fast_nn": {
            "easy":    (0.72, 0.3),
            "medium":  (0.65, 0.3),
            "hard":    (0.55, 0.3),
            "extreme": (0.40, 0.3),
        },
        "ensemble_nn": {
            "easy":    (0.85, 0.8),
            "medium":  (0.78, 0.8),
            "hard":    (0.68, 0.8),
            "extreme": (0.52, 0.8),
        },
        "strong_nn": {
            "easy":    (0.88, 1.2),
            "medium":  (0.83, 1.2),
            "hard":    (0.78, 1.2),
            "extreme": (0.65, 1.2),
        },
        "fvcom": {
            "easy":    (0.92, 2.0),
            "medium":  (0.89, 2.0),
            "hard":    (0.84, 2.0),
            "extreme": (0.78, 2.0),
        },
        "physics_hybrid": {
            "easy":    (0.95, 3.5),
            "medium":  (0.93, 3.5),
            "hard":    (0.90, 3.5),
            "extreme": (0.85, 3.5),
        },
    },
    "state_parse": {
        "rule_parser": {
            "easy":    (0.85, 0.2),
            "medium":  (0.70, 0.2),
            "hard":    (0.50, 0.2),
            "extreme": (0.30, 0.2),
        },
        "llm_small": {
            "easy":    (0.88, 1.0),
            "medium":  (0.82, 1.0),
            "hard":    (0.72, 1.0),
            "extreme": (0.55, 1.0),
        },
        "rag_parser": {
            "easy":    (0.90, 1.5),
            "medium":  (0.86, 1.5),
            "hard":    (0.80, 1.5),
            "extreme": (0.68, 1.5),
        },
        "llm_large": {
            "easy":    (0.95, 3.0),
            "medium":  (0.92, 3.0),
            "hard":    (0.88, 3.0),
            "extreme": (0.80, 3.0),
        },
    },
    "data_analysis": {
        "rule_based": {
            "easy":    (0.65, 0.1),
            "medium":  (0.55, 0.1),
            "hard":    (0.40, 0.1),
            "extreme": (0.25, 0.1),
        },
        "ensemble": {
            "easy":    (0.75, 0.5),
            "medium":  (0.70, 0.5),
            "hard":    (0.60, 0.5),
            "extreme": (0.45, 0.5),
        },
        "ml_pipeline": {
            "easy":    (0.80, 0.8),
            "medium":  (0.78, 0.8),
            "hard":    (0.73, 0.8),
            "extreme": (0.60, 0.8),
        },
        "deep_learning": {
            "easy":    (0.88, 2.5),
            "medium":  (0.85, 2.5),
            "hard":    (0.80, 2.5),
            "extreme": (0.72, 2.5),
        },
    },
    # Aggregator primitive: merges parallel branch outputs — low cost, high quality
    "aggregator": {
        "weighted_avg": {
            "easy":    (0.90, 0.1),
            "medium":  (0.88, 0.1),
            "hard":    (0.85, 0.1),
            "extreme": (0.80, 0.1),
        },
    },
}


# ---------------------------------------------------------------------------
# Evaluation Result
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """
    Output of the Evaluator after executing a candidate on a sub-task.

    Attributes
    ----------
    observed_quality : float
        Measured ACC on this execution (0~1). If ground truth is known
        (MVP), this is sampled from the GT curve with noise.
    observed_cost : float
        Actual execution cost observed.
    eval_pass : bool
        Legacy field. Kept for backward compatibility.
        True if eval_level in (PASS, WARN).
    eval_level : str
        Four-level evaluator output (per method definition Section 7):
        "pass" | "warn" | "fail" | "escalate"
        - pass: quality meets all criteria, continue normally
        - warn: quality borderline, continue + log warning
        - fail: quality below threshold, trigger repair
        - escalate: critical issue, upgrade evaluator immediately
    failure_type : str | None
        - "low_quality" : quality below threshold
        - "crash"        : execution crashed
        - "timeout"       : exceeded time limit
        - "constraint_violation" : hard constraint was breached
        - None            : passed
    true_quality : float | None
        Ground truth quality (MVP only, for analysis).
    true_cost : float | None
        Ground truth cost (MVP only, for analysis).
    metadata : dict
        Additional context (task_id, node_id, candidate_name, etc.).
    constraint_violations : list[dict]
        List of constraint violations detected during evaluation.
        Each entry: {"constraint_id", "constraint_type", "reason"}
    execution_duration : float | None
        Simulated wall-clock execution duration in seconds.
    human_approved : bool
        Simulated human approval result (True=approved, False=rejected).
    """

    observed_quality: float
    observed_cost: float
    eval_pass: bool  # Legacy: True if eval_level in (PASS, WARN)
    failure_type: str | None
    true_quality: float | None = None
    true_cost: float | None = None
    metadata: dict | None = None
    # === 约束验证与多模态字段 ===
    constraint_violations: list[dict] = field(default_factory=list)
    execution_duration: float | None = None
    human_approved: bool = True
    evaluator_name: str = "rule_eval"
    evaluator_cost: float = 0.0
    # === 四级输出 (per method definition Section 7) ===
    eval_level: str = "pass"          # "pass" | "warn" | "fail" | "escalate"
    quality_score: float = 0.0         # judge-based rubric 加权得分（不同于 raw observed_quality）
    error_type: Optional[str] = None   # "low_quality" | "format_error" | "inconsistent_output" | ...
    confidence: float = 1.0            # [0,1]，evaluator 判断置信度
    evaluator_latency: float = 0.0     # 评估耗时（秒）
    # === Cost breakdown (method §4.3) ===
    c_main: float = 0.0              # executor cost in real USD (linear model or scalar mean)
    c_llm: float = 0.0               # judge/evaluator cost in real USD
    c_total: float = 0.0             # c_main + c_llm (real USD)
    c_usd_raw: float = 0.0          # alias for c_total (real USD)
    latency: float = 0.0             # raw latency in seconds


# ---------------------------------------------------------------------------
# Evaluator Profiles — executor x evaluator joint optimization
# ---------------------------------------------------------------------------

# Evaluator precision: probability of making a correct pass/fail judgment.
# Low precision (rule_eval) → high false-pass and false-fail rates.
# High precision (large_eval) → accurate detection.
EVALUATOR_PROFILES: Dict[str, Dict[str, float]] = {
    "rule_eval": {
        "precision": 0.65,   # low accuracy — cheap but unreliable
        "cost": 0.1,
        "latency": 0.01,
    },
    "small_eval": {
        "precision": 0.80,   # medium accuracy
        "cost": 0.5,
        "latency": 0.1,
    },
    "large_eval": {
        "precision": 0.95,   # high accuracy — expensive but reliable
        "cost": 2.0,
        "latency": 0.5,
    },
}

def _judge_by_rubric(
    true_acc: float,
    obs_quality: float,
    node_type: str,
    evaluator_name: str,
    rng: random.Random,
) -> Tuple[float, Optional[str], float]:
    """
    Judge-based evaluation using node-type-specific rubric.

    Parameters
    ----------
    true_acc : float
        Ground truth quality (MVP only).
    obs_quality : float
        Noisy observed quality from executor run.
    node_type : str
        Node type determines which rubric to apply.
    evaluator_name : str
        Evaluator ID determines precision (affects judge noise).
    rng : random.Random
        Random source for judge noise simulation.

    Returns
    -------
    Tuple[float, Optional[str], float]
        (quality_score, error_type, confidence):
        - quality_score: rubric 加权得分 [0, 1]
        - error_type: detected error type or None
        - confidence: evaluator confidence [0, 1]
        NOTE: eval_level is determined in evaluate() using quality_score + error_type + confidence
    """
    from src.evaluation.evaluator_types import get_rubric

    rubric = get_rubric(node_type)
    dimensions = rubric["dimensions"]
    threshold = rubric["pass_threshold_score"]

    # Evaluator precision determines judge noise magnitude
    prof = EVALUATOR_PROFILES.get(evaluator_name, EVALUATOR_PROFILES["rule_eval"])
    precision = prof["precision"]
    # High precision (large_eval) → small noise; low precision (rule_eval) → large noise
    noise_std = (1.0 - precision) * 0.20  # max noise ~0.14 for rule_eval

    # Compute weighted rubric score
    weighted_score = 0.0
    rule_dim_failed = False  # track if any rule dimension binary-failed
    for dim_name, weight, method in dimensions:
        if method == "objective":
            # 有客观指标：直接用 obs_quality
            dim_score = obs_quality
        elif method == "judge":
            # judge 维度：precision 高则 noise 小
            judge_noise = rng.gauss(0, noise_std)
            dim_score = float(np.clip(obs_quality + judge_noise, 0.0, 1.0))
        else:  # rule: 二元判定
            dim_score = 1.0 if obs_quality >= threshold else 0.0
            if dim_score == 0.0:
                rule_dim_failed = True
        weighted_score += dim_score * weight

    weighted_score = round(float(np.clip(weighted_score, 0.0, 1.0)), 4)

    # Confidence: derived from evaluator precision, with small noise
    # rule_eval gets wider noise band (less reliable confidence)
    conf_noise_range = 0.05 if precision < 0.70 else 0.03
    confidence = float(np.clip(precision + rng.uniform(-conf_noise_range, conf_noise_range), 0.0, 1.0))
    confidence = round(confidence, 4)

    # Detect error_type based on score vs threshold and true_acc
    error_type: Optional[str] = None
    if weighted_score < threshold:
        # Failed: determine why (priority order)
        if weighted_score < 0.35:
            error_type = "unsafe_decision"
        elif rule_dim_failed:
            error_type = "format_error"
        elif obs_quality < 0.40:
            error_type = "low_quality"
        elif obs_quality >= threshold:
            # Inconsistent: raw quality was OK but rubric weighted down
            error_type = "inconsistent_output"
        elif true_acc >= 0.70 and precision < 0.75:
            # Likely false rejection: good executor, weak evaluator
            error_type = "low_quality"
        else:
            error_type = "low_quality"

    return weighted_score, error_type, confidence


def _compute_eval_level(
    quality_score: float,
    error_type: Optional[str],
    confidence: float,
    pass_threshold: float,
) -> str:
    """
    Determine 4-level evaluator output from rubric results.

    Level definitions (per method definition Section 7):

    | Level     | Condition                                                           |
    |-----------|---------------------------------------------------------------------|
    | PASS      | quality_score >= pass_threshold AND error_type is None               |
    | WARN      | quality_score >= pass_threshold but borderline (near threshold)       |
    | FAIL      | quality_score < pass_threshold AND NOT escalate conditions             |
    | ESCALATE  | quality_score << pass_threshold (gap > 0.15)                        |
    |           | OR error_type in (unsafe_decision, format_error)                    |
    |           | OR confidence < 0.4 (evaluator very uncertain)                       |

    Rationale for WARN:
    - Near-threshold quality (within 0.05 of pass_threshold) may degrade downstream
      even if current evaluation technically passes.
    - Low confidence (0.4-0.7) even with passing score means evaluator is uncertain.
    - WARN allows early intervention (log + continue) without full repair cost.

    Rationale for ESCALATE:
    - unsafe_decision / format_error: candidate cannot self-correct, needs stronger evaluator
    - Very low confidence: evaluator itself is unreliable, upgrade needed
    - Severe quality gap: problem is likely in the executor, but evaluator must confirm first
    """
    ESCALATE_ERRORS = {"unsafe_decision", "format_error"}
    ESCALATE_CONFIDENCE = 0.40
    ESCALATE_GAP = 0.15  # quality_score below threshold by more than this → escalate

    # ESCALATE: critical issues requiring immediate evaluator upgrade
    if error_type in ESCALATE_ERRORS:
        return "escalate"
    if confidence < ESCALATE_CONFIDENCE:
        return "escalate"
    if quality_score < pass_threshold - ESCALATE_GAP:
        return "escalate"

    # PASS: quality meets all criteria
    if error_type is None and quality_score >= pass_threshold:
        return "pass"

    # WARN: borderline quality (passing but near threshold or low confidence)
    if quality_score >= pass_threshold:
        near_threshold = (pass_threshold - quality_score) < 0.05
        low_confidence = confidence < 0.70
        if near_threshold or low_confidence:
            return "warn"

    # Default: FAIL (quality below threshold, but not escalate-level severity)
    return "fail"


# ---------------------------------------------------------------------------
# Mock Evaluator
# ---------------------------------------------------------------------------

class MockEvaluator:
    """
    MVP Evaluator: simulates real evaluation using Ground Truth curves.

    Real Evaluator responsibilities (to be implemented later):
    - Execute the candidate tool on the sub-task
    - Compute observed_quality via metrics (accuracy, F1, BLEU, etc.)
    - Record observed_cost (time, token usage, monetary cost)
    - Determine eval_pass and failure_type
    - Return EvaluationResult

    MockEvaluator behavior (MVP):
    - Looks up ground truth (true_acc, true_cost) from predefined curves
    - Adds Gaussian noise to true_acc (configurable std)
    - Adds multiplicative noise to true_cost (configurable range)
    - eval_pass = observed_quality >= pass_threshold

    ProfileStore integration (exp.md interface 2):
    - If profile_store is provided, uses it for GT lookups
    - This means GT data comes from data/executor_profiles.jsonl,
      NOT from hardcoded constants
    """

    def __init__(
        self,
        ground_truth: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]]
        | None = None,
        noise_std: float = 0.05,
        cost_noise_range: Tuple[float, float] = (0.95, 1.05),
        pass_threshold: float = 0.5,
        seed: int | None = None,
        profile_store=None,  # Optional ProfileStore for external data
        pareto_mode: bool = False,  # True: skip rubric judge, direct quality→pass/fail
    ):
        """
        Parameters
        ----------
        ground_truth : dict | None
            Predefined GT curves. Defaults to DEFAULT_GROUND_TRUTH.
            Ignored if profile_store is provided.
        noise_std : float
            Std dev of Gaussian noise added to true_acc.
        cost_noise_range : tuple
            Multiplicative range for cost noise: cost * uniform(range[0], range[1]).
        pass_threshold : float
            Minimum quality for eval_pass.
        seed : int | None
            Random seed for reproducibility.
        profile_store : ProfileStore | None
            If provided, GT data is read from ProfileStore
            (from data/executor_profiles.jsonl).
            Adding new tools = adding rows to the JSONL file,
            NO code changes needed.
        """
        self._profile_store = profile_store
        # Fallback to hardcoded GT only if no profile_store
        self.gt = ground_truth if ground_truth is not None else DEFAULT_GROUND_TRUTH
        self.noise_std = noise_std
        self.cost_range = cost_noise_range
        self.pass_threshold = pass_threshold if pass_threshold != 0.5 else 0.60

        self.rng = np.random.default_rng(seed)
        self._py_rng = random.Random(seed)  # for non-numpy randomness

        # Track all evaluation results for analysis
        self.history: list[EvaluationResult] = []
        # last_call_cost: API cost of last evaluation (MockEvaluator=0.0, ClaudeEvaluator=actual USD)
        self.last_call_cost: float = 0.0
        self.pareto_mode = pareto_mode

    def _get_gt(self, primitive_name: str, candidate_name: str, difficulty_bucket: str) -> Tuple[float, float]:
        """
        Look up ground truth (true_acc, true_cost) for an executor.

        Priority:
        1. ProfileStore (if injected) → uses data/executor_profiles.jsonl
        2. self.gt (hardcoded DEFAULT_GROUND_TRUTH, for backward compatibility)
        """
        if self._profile_store is not None:
            result = self._profile_store.get_executor_quality_cost(
                primitive_name, candidate_name, difficulty_bucket
            )
            if result is not None:
                return result
            # ProfileStore doesn't have this (primitive/candidate/difficulty) → fallback

        # Fallback: hardcoded GT
        try:
            return self.gt[primitive_name][candidate_name][difficulty_bucket]
        except KeyError:
            # No GT data for this (primitive, candidate, difficulty).
            # Return fallback: quality=0.5, cost=0.001
            # This prevents crashes for models without collected profile data.
            return (0.5, 0.001)

    def evaluate(
        self,
        candidate_name: str,
        primitive_name: str,
        difficulty_bucket: str,
        task_id: str = "",
        node_id: str = "",
        metadata: dict | None = None,
        task_spec: SubTaskSpec | None = None,  # optional constraint context
        evaluator_name: str = "rule_eval",      # evaluator type for joint optimization
        node_type: str = "unknown",             # node type for rubric-based evaluation
    ) -> EvaluationResult:
        """
        Simulate evaluation of a candidate on a given primitive + difficulty bucket.

        Parameters
        ----------
        candidate_name : str
        primitive_name : str
        difficulty_bucket : str
        task_id, node_id : str
            Identifiers for tracing.
        metadata : dict, optional
            Extra fields to store.
        evaluator_name : str
            Which evaluator to use: "rule_eval" (cheap/low precision),
            "small_eval" (medium), or "large_eval" (expensive/high precision).
            Affects both the eval_pass outcome and the observed_quality signal.
        node_type : str
            Node type for rubric-based evaluation. Determines which
            NODE_TYPE_RUBRIC is used to compute quality_score and error_type.
            Defaults to "unknown" (uses __default__ rubric).

        Returns
        -------
        EvaluationResult

        Raises
        ------
        KeyError
            If (primitive_name, candidate_name, difficulty_bucket) not in ground truth.
        """
        # Lookup ground truth (ProfileStore priority over hardcoded GT)
        true_acc, true_cost = self._get_gt(primitive_name, candidate_name, difficulty_bucket)

        # Add noise to simulate executor variance
        obs_acc = float(
            np.clip(true_acc + self.rng.normal(0, self.noise_std), 0.0, 1.0)
        )
        cost_mult = self.rng.uniform(self.cost_range[0], self.cost_range[1])
        obs_cost = round(true_cost * cost_mult, 4)

        # Cost breakdown (method §4.3):
        # c_main = executor cost, computed via linear model:
        #   c_main = (cost_a1/1e6) * typ_input + (cost_a2/1e6) * typ_output  [real USD]
        #          × COST_SCALE  [→ COST_SCALE-scaled USD]
        # Falls back to true_cost (scalar) if linear coefficients unavailable.
        # c_llm = evaluator_cost * COST_SCALE (judge runs separately)
        # c_total = c_main + c_llm
        _prof = None
        if self._profile_store is not None:
            tool_id = f"{primitive_name}/{candidate_name}"
            _prof = self._profile_store.get_executor_profile(tool_id, difficulty_bucket)

        if _prof is not None and (_prof.cost_a1_per_mtok > 0 or _prof.cost_a2_per_mtok > 0):
            # Linear model: c_main = (a1/1e6)*in + (a2/1e6)*out  [real USD]
            typ_in  = _prof.typical_input_tokens  or 32
            typ_out = _prof.typical_output_tokens or 100
            c_main_real_usd = (
                _prof.cost_a1_per_mtok / 1_000_000 * typ_in +
                _prof.cost_a2_per_mtok / 1_000_000 * typ_out
            )
            c_main = c_main_real_usd  # real USD
        else:
            # Fallback: scalar api_cost_mean
            c_main = true_cost

        c_llm  = 0.0  # placeholder, updated below after eval_profile lookup
        c_total = c_main + c_llm
        c_usd_raw = c_total

        # === Judge-based scoring ===
        if self.pareto_mode:
            # Pareto 模式：跳过 rubric judge，直接用 observed_quality 判定
            quality_score = obs_acc
            error_type = None
            confidence = 1.0
            evaluator_cost = 0.0
            evaluator_latency = 0.0
        else:
            # 原有 rubric judge 逻辑
            # Uses node-type-specific rubric, evaluator precision determines judge noise
            quality_score, error_type, confidence = _judge_by_rubric(
                true_acc=true_acc,
                obs_quality=obs_acc,
                node_type=node_type,
                evaluator_name=evaluator_name,
                rng=self._py_rng,
            )

            # Evaluator cost and latency
            eval_profile = EVALUATOR_PROFILES.get(evaluator_name, {})
            evaluator_cost = eval_profile.get("cost", 0.0)
            evaluator_latency = eval_profile.get("latency", 0.0)

        # Finalize cost breakdown — evaluator cost is already in real USD
        c_llm    = evaluator_cost   # judge cost (real USD, per EVALUATOR_PROFILES)
        c_total  = c_main + c_llm  # total real USD
        c_usd_raw = c_total

        # Simulate latency for non-constrained tasks
        _sim_latency = self._simulate_duration(candidate_name, primitive_name, difficulty_bucket)

        # === Four-level judgment (per method definition Section 7) ===
        if self.pareto_mode:
            eval_level = "pass" if quality_score >= self.pass_threshold else "fail"
        else:
            eval_level = _compute_eval_level(
                quality_score=quality_score,
                error_type=error_type,
                confidence=confidence,
                pass_threshold=self.pass_threshold,
            )

        # Legacy eval_pass: True if (pass or warn)
        eval_pass = eval_level in ("pass", "warn")
        failure_type: str | None = error_type if not eval_pass else None

        result = EvaluationResult(
            observed_quality=round(obs_acc, 4),   # raw noisy executor output
            quality_score=quality_score,           # rubric-weighted judge score
            observed_cost=obs_cost,
            eval_pass=eval_pass,
            eval_level=eval_level,               # four-level output
            failure_type=failure_type,
            true_quality=true_acc,
            true_cost=true_cost,
            evaluator_name=evaluator_name,
            evaluator_cost=evaluator_cost,
            evaluator_latency=evaluator_latency,
            error_type=error_type,
            confidence=confidence,
            c_main=c_main,                       # executor cost (scaled)
            c_llm=c_llm,                        # judge cost (scaled)
            c_total=c_total,                    # total (scaled)
            c_usd_raw=c_usd_raw,               # raw USD
            latency=_sim_latency,              # raw latency seconds
            metadata={
                "task_id": task_id,
                "node_id": node_id,
                "primitive_name": primitive_name,
                "candidate_name": candidate_name,
                "difficulty_bucket": difficulty_bucket,
                "node_type": node_type,
                **(metadata or {}),
            },
        )

        # === Constraint validation layer ===
        if task_spec is not None and task_spec.constraints:
            violations = self._check_constraints(result, task_spec)
            result.constraint_violations = violations

            # Simulate execution duration
            result.execution_duration = self._simulate_duration(
                candidate_name, primitive_name, difficulty_bucket
            )

            # Constraint violations can upgrade eval_level:
            # - timeout → escalate (critical time constraint breach)
            # - risk boundary violation (quality below floor) → escalate
            # - human rejection → escalate (human disapproves critical decision)
            # - non-critical violations → fail or warn
            time_violations = [v for v in violations if v["constraint_type"] == "time_window"]
            risk_violations = [v for v in violations if v["constraint_type"] == "risk_boundary"]

            if time_violations and result.eval_level not in ("escalate",):
                result.eval_level = "escalate"
                result.eval_pass = False
                result.failure_type = "timeout"
            elif risk_violations and result.eval_level == "pass":
                # Quality floor breach is serious even if rubric passed
                result.eval_level = "escalate"
                result.eval_pass = False
                result.failure_type = result.failure_type or "risk_violation"

            # Simulate human approval if HITL constraint exists
            hitl_violations = [v for v in violations if v["constraint_type"] == "human_in_the_loop"]
            if hitl_violations:
                result.human_approved = self._simulate_human_approval(
                    result, task_spec
                )
                # If human rejects, escalate (needs human intervention)
                if not result.human_approved:
                    result.eval_level = "escalate"
                    result.eval_pass = False
                    result.failure_type = result.failure_type or "human_rejected"

            # Update legacy eval_pass to match final eval_level
            result.eval_pass = result.eval_level in ("pass", "warn")

        self.history.append(result)
        return result

    # -------------------------------------------------------------------------
    # Constraint Validation Helpers
    # -------------------------------------------------------------------------

    def _check_constraints(
        self,
        result: EvaluationResult,
        task_spec: SubTaskSpec,
    ) -> list[dict]:
        """
        Check all enabled constraints in task_spec against the evaluation result.
        Returns a list of violation dicts.

        Supported constraint types:
        - TimeWindowConstraint: observed_duration > max_duration
        - RiskBoundaryConstraint: quality < min_quality OR cost > max_cost
        - HumanInTheLoopConstraint: presence is noted (actual approval is simulated)
        - MandatoryNodeConstraint: checked by orchestrator, not here
        """
        from src.decomposer.task_decomposer import (
            TimeWindowConstraint,
            RiskBoundaryConstraint,
            HumanInTheLoopConstraint,
            MandatoryNodeConstraint,
        )

        violations: list[dict] = []

        for c in task_spec.constraints:
            if not c.enabled:
                continue

            # MandatoryNodeConstraint: enforced by orchestrator, not evaluator
            if isinstance(c, MandatoryNodeConstraint):
                continue

            # HumanInTheLoopConstraint: presence is a signal, approval is simulated
            if isinstance(c, HumanInTheLoopConstraint):
                violations.append({
                    "constraint_id": c.constraint_id,
                    "constraint_type": "human_in_the_loop",
                    "reason": f"Human approval required at '{c.approval_point}' "
                              f"(approval_point='{c.approval_point}', "
                              f"role='{c.required_role}'). "
                              f"Approval outcome: {'approved' if result.human_approved else 'rejected'}.",
                })
                continue

            if isinstance(c, TimeWindowConstraint):
                # Duration is simulated before this check, so read from result
                if result.execution_duration is not None:
                    if result.execution_duration > c.max_duration:
                        violations.append({
                            "constraint_id": c.constraint_id,
                            "constraint_type": "time_window",
                            "reason": (
                                f"execution_duration={result.execution_duration:.2f}s "
                                f"exceeds max_duration={c.max_duration}s "
                                f"(constraint: {c.description})"
                            ),
                        })

            elif isinstance(c, RiskBoundaryConstraint):
                if c.is_quality_violation(result.observed_quality):
                    violations.append({
                        "constraint_id": c.constraint_id,
                        "constraint_type": "risk_boundary",
                        "reason": (
                            f"observed_quality={result.observed_quality:.4f} "
                            f"is below min_quality={c.min_quality} "
                            f"(constraint: {c.description})"
                        ),
                    })
                if c.is_cost_violation(result.observed_cost, task_spec.difficulty_bucket):
                    limit = c.get_max_cost_for_bucket(task_spec.difficulty_bucket)
                    violations.append({
                        "constraint_id": c.constraint_id,
                        "constraint_type": "risk_boundary",
                        "reason": (
                            f"observed_cost={result.observed_cost:.4f} "
                            f"exceeds max_cost={limit} "
                            f"for bucket '{task_spec.difficulty_bucket}' "
                            f"(constraint: {c.description})"
                        ),
                    })

        return violations

    def _simulate_duration(
        self,
        candidate_name: str,
        primitive_name: str,
        difficulty_bucket: str,
    ) -> float:
        """
        Simulate wall-clock execution duration from ground-truth cost.

        Uses a cost-to-duration conversion:
            duration = cost * DURATION_COST_RATIO + uniform(-0.5, 0.5) seconds

        DURATION_COST_RATIO = 10 means a cost of 1.0 -> ~10 seconds.
        This is a simplified model; real systems would measure actual wall time.
        """
        DURATION_COST_RATIO = 10.0
        try:
            _, true_cost = self._get_gt(primitive_name, candidate_name, difficulty_bucket)
        except KeyError:
            true_cost = 1.0

        jitter = self._py_rng.uniform(-0.5, 0.5)
        return round(true_cost * DURATION_COST_RATIO + jitter, 2)

    def _simulate_human_approval(
        self,
        result: EvaluationResult,
        task_spec: SubTaskSpec,
    ) -> bool:
        """
        Simulate human approval outcome based on observed quality.

        Model:
        - observed_quality >= 0.80: high approval probability (95%)
        - observed_quality >= 0.60: medium approval probability (70%)
        - observed_quality >= 0.50: low approval probability (40%)
        - otherwise: very low (10%)
        """
        q = result.observed_quality
        if q >= 0.80:
            approval_prob = 0.95
        elif q >= 0.60:
            approval_prob = 0.70
        elif q >= 0.50:
            approval_prob = 0.40
        else:
            approval_prob = 0.10

        roll = self._py_rng.random()
        approved = roll < approval_prob
        return approved

    # -------------------------------------------------------------------------
    # History Management
    # -------------------------------------------------------------------------

    def reset_history(self) -> None:
        """Clear evaluation history."""
        self.history.clear()

    def get_history_dataframe(self) -> list[dict]:
        """
        Return evaluation history as a list of dicts (for pandas DataFrame).
        """
        rows = []
        for r in self.history:
            row = {
                "observed_quality": r.observed_quality,
                "observed_cost": r.observed_cost,
                "eval_pass": r.eval_pass,
                "failure_type": r.failure_type,
                "true_quality": r.true_quality,
                "true_cost": r.true_cost,
                "constraint_violations": r.constraint_violations,
                "violation_count": len(r.constraint_violations),
                "execution_duration": r.execution_duration,
                "human_approved": r.human_approved,
            }
            if r.metadata:
                row.update(r.metadata)
            rows.append(row)
        return rows

    @property
    def pass_rate(self) -> float:
        """Overall pass rate across all evaluations so far."""
        if not self.history:
            return 0.0
        return sum(1 for r in self.history if r.eval_pass) / len(self.history)


# ---------------------------------------------------------------------------
# MockLLMEvaluator — BaseEvaluator implementation (mock, swappable to real LLM judge)
# ---------------------------------------------------------------------------

class MockLLMEvaluator(BaseEvaluator):
    """
    基于 rubric 的 mock LLM judge evaluator。

    实现 BaseEvaluator 接口，可注册即插即用。
    后续替换为真实 LLM judge（如 GPT-4o judge）时，只需：
    1. 实现 BaseEvaluator 接口，替换此类
    2. 在 __init__ 中传入真实 API client
    无需修改任何主流程代码。

    行为：
    - 读取 NODE_TYPE_RUBRIC 中的 rubric 定义
    - 根据 evaluator_id 决定 precision（对应 rule_eval/small_eval/large_eval）
    - 返回包含 error_type 和 confidence 的 EvaluatorOutput
    """

    def __init__(
        self,
        evaluator_id: str = "mock_llm_judge",
        seed: int | None = None,
        profile_store=None,
    ):
        """
        Parameters
        ----------
        evaluator_id : str
            对应 EVALUATOR_PROFILES 中的某一档：
            "rule_eval" → 低精度（便宜）
            "small_eval" → 中精度（中等）
            "large_eval" → 高精度（昂贵）
            或自定义字符串（使用 __default__ profile 值）。
        seed : int | None
            随机种子。
        profile_store : ProfileStore | None
            可选，注入外部 profile 数据。
        """
        self._evaluator_id = evaluator_id
        self._seed = seed
        self._profile_store = profile_store
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        # 确定使用的 evaluator profile
        self._eval_profile = EVALUATOR_PROFILES.get(
            evaluator_id,
            {"precision": 0.75, "cost": 0.5, "latency": 0.1}
        )

    @property
    def evaluator_id(self) -> str:
        return self._evaluator_id

    @property
    def name(self) -> str:
        return f"Mock LLM Judge ({self._evaluator_id})"

    @property
    def supported_node_types(self) -> List[str]:
        from src.evaluation.evaluator_types import NODE_TYPE_RUBRIC
        return [k for k in NODE_TYPE_RUBRIC if not k.startswith("__")]

    @property
    def supported_task_types(self) -> List[str]:
        return ["time_series", "text_analysis", "tabular_analysis", "multimodal"]

    @property
    def latency_mean(self) -> float:
        return self._eval_profile["latency"]

    @property
    def api_cost_mean(self) -> float:
        return self._eval_profile["cost"]

    def evaluate(self, inp: EvaluatorInput) -> EvaluatorOutput:
        """
        实现 BaseEvaluator 接口。

        使用 MockEvaluator 的 GT 模拟 + rubric 打分逻辑。
        在真实场景中，这里会调用真实 LLM judge API。
        """
        from src.evaluation.evaluator_types import get_rubric

        # 构造内部 mock 调用
        mock = MockEvaluator(
            ground_truth=None,  # 使用 profile_store
            noise_std=0.05,
            pass_threshold=0.60,
            seed=self._seed,
            profile_store=self._profile_store,
        )

        # 调用 mock evaluator
        # 注意：这里用 primitive_name 作为 node_type 的 fallback
        node_type = inp.node_type or inp.primitive_name or "unknown"
        result = mock.evaluate(
            candidate_name=inp.candidate_name,
            primitive_name=inp.primitive_name,
            difficulty_bucket=inp.difficulty_bucket,
            task_id=inp.node_id,
            node_id=inp.node_id,
            evaluator_name=self._evaluator_id,
            node_type=node_type,
        )

        return EvaluatorOutput(
            evaluator_id=self._evaluator_id,
            quality_score=result.quality_score,
            passed=result.eval_pass,
            error_type=result.error_type,
            confidence=result.confidence,
            latency=result.evaluator_latency,
            api_cost=result.evaluator_cost,
            human_cost=0.0,
            constraint_violations=result.constraint_violations,
            evaluator_profile_snapshot=self._eval_profile,
            metadata={
                "task_type": inp.task_type,
                "node_type": node_type,
                "node_id": inp.node_id,
                "template_id": inp.template_id,
                "primitive_name": inp.primitive_name,
                "candidate_name": inp.candidate_name,
                "difficulty_bucket": inp.difficulty_bucket,
                **inp.metadata,
            },
        )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("MockEvaluator + MockLLMEvaluator Demo")
    print("=" * 60)

    ev = MockEvaluator(noise_std=0.05, pass_threshold=0.60, seed=42)

    print("\n-- Evaluate forecast/fast_nn @ medium (node_type=forecast) --")
    r = ev.evaluate("fast_nn", "forecast", "medium",
                     task_id="task_001", node_id="node_001",
                     node_type="forecast")
    print(f"  true:         acc={r.true_quality}, cost={r.true_cost}")
    print(f"  raw_obs:      acc={r.observed_quality:.4f}")
    print(f"  quality_score={r.quality_score:.3f} (rubric-weighted)")
    print(f"  error_type={r.error_type}, confidence={r.confidence:.3f}")
    print(f"  pass={r.eval_pass}, eval_cost={r.evaluator_cost}, latency={r.evaluator_latency}s")

    print("\n-- Evaluate forecast/fvcom @ hard (node_type=forecast) --")
    r2 = ev.evaluate("fvcom", "forecast", "hard",
                      task_id="task_001", node_id="node_002",
                      node_type="forecast")
    print(f"  quality_score={r2.quality_score:.3f}, error_type={r2.error_type}, "
          f"confidence={r2.confidence:.3f}, pass={r2.eval_pass}")

    print("\n-- Evaluate state_parse @ hard (node_type=state_parse) --")
    r3 = ev.evaluate("llm_large", "state_parse", "hard",
                      task_id="task_002", node_id="node_003",
                      node_type="state_parse")
    print(f"  quality_score={r3.quality_score:.3f}, error_type={r3.error_type}, "
          f"confidence={r3.confidence:.3f}, pass={r3.eval_pass}")

    print(f"\n  Overall pass rate: {ev.pass_rate:.1%}")

    print("\n-- MockLLMEvaluator (BaseEvaluator interface) --")
    from src.evaluation.evaluator_types import EvaluatorInput

    mock_judge = MockLLMEvaluator(evaluator_id="small_eval", seed=42)
    inp = EvaluatorInput(
        task_type="time_series",
        node_type="forecast",
        node_id="st_0",
        template_id="direct",
        primitive_name="forecast",
        candidate_name="strong_nn",
        difficulty=0.5,
        difficulty_bucket="medium",
        input_payload="Predict next 7-day sales",
        candidate_output={"forecast": [1.2, 1.5, 1.8]},
    )
    out = mock_judge.evaluate(inp)
    print(f"  evaluator_id={out.evaluator_id}")
    print(f"  quality_score={out.quality_score:.3f}, passed={out.passed}")
    print(f"  error_type={out.error_type}, confidence={out.confidence:.3f}")
    print(f"  latency={out.latency}s, api_cost={out.api_cost}")

    print("\n-- Unknown candidate (expect KeyError) --")
    try:
        ev.evaluate("unknown_agent", "forecast", "medium")
    except KeyError as e:
        print(f"  [Expected] KeyError: {str(e)[:80]}")

    print("\nDemo complete.")
