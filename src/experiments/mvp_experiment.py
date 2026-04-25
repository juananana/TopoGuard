"""
mvp_experiment.py
=================
MVP End-to-End Experiment: training loop for the profile manager.

This script validates the full feedback loop:

    Task -> TaskDecomposer -> SubTaskSpec list (with constraints)
        -> ProfileManager.predict_all -> candidate ranking
        -> _apply_constraints() -> constraint-filtered ranking
        -> MockEvaluator.evaluate (with constraint validation) -> EvaluationResult
        -> ProfileManager.add_feedback -> buffer
        -> [every K episodes] batch_recalibrate -> update curves
        -> ProfileManager.export_curve_table -> analyze convergence

Key experimental variables:
- calibration_interval: how many feedback records before recalibration
- acc_target / cost_budget / latency_budget: orchestration hard constraint strategy
- enable_constraints: inject hard constraints (time window, human HITL, mandatory nodes, risk boundary)
- constraint_violation_penalty: score penalty applied when a candidate violates constraints
- noise_std: evaluator noise level (robustness test)

Outputs:
- Episode execution log (CSV)
- Before/after curve comparison per recalibration event
- Final export_curve_table
- Per-episode metrics CSV
- Constraint violation summary
"""

from __future__ import annotations

import sys
import csv
import random
import argparse
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict

# Add src/ to path for imports
_SRC_DIR = Path(__file__).parent.parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from src.primitives.profile_manager import PrimitivePerformanceProfileManager
from src.primitives.primitive_profile import InitPoint
from src.primitives.feedback_record import FeedbackRecord
from src.primitives.profile_store import ProfileStore
from src.primitives.topology_template import TemplateLibrary
from src.decomposer.task_decomposer import (
    TaskDecomposer,
    SubTaskSpec,
    ConstraintSpec,
    TimeWindowConstraint,
    HumanInTheLoopConstraint,
    MandatoryNodeConstraint,
    RiskBoundaryConstraint,
    ModalityType,
)
from src.evaluation.mock_evaluator import MockEvaluator, DEFAULT_GROUND_TRUTH, EVALUATOR_PROFILES


# ---------------------------------------------------------------------------
# Experiment Configuration
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """
    Configuration for a single MVP experiment run.
    """

    name: str = "mvp_default"
    n_episodes: int = 20
    calibration_interval: int = 5
    acc_target: Optional[float] = None
    cost_budget: Optional[float] = None
    latency_budget: Optional[float] = None  # 硬约束：候选 pred_latency 必须 ≤ 此值
    noise_std: float = 0.05
    pass_threshold: float = 0.60
    seed: int = 42
    output_dir: Path = field(default_factory=lambda: Path("./outputs"))

    # === 约束与多模态配置 ===
    enable_constraints: bool = True     # 是否注入硬约束（时间窗、人工审批、必须节点、风险边界）
    constraint_violation_penalty: float = -0.5  # 违约束时的 score 惩罚
    enable_multimodal: bool = True     # 是否启用多模态约束
    enable_human_hitl: bool = True     # 是否模拟 human-in-the-loop
    constrained_task_ratio: float = 0.4  # 约束任务占总任务的比例
    task_override: str | None = None    # 强制使用单个任务描述（测试拓扑专用）

    # === A/B 对比开关 ===
    fixed_template: bool = False       # True: 跳过 score_templates，全部用 "direct"
    enable_repair: bool = True          # False: 跳过 repair 循环

    # === 训练输入模式 ===
    input_mode: str = "sample_tasks"        # "sample_tasks" | "direct" | "calibration_jsonl"
    calibration_file: Path | None = None    # input_mode="calibration_jsonl" 时的 JSONL 路径
    difficulty_policy_train: str = "external_first"   # train stage: external_first | infer_first
    difficulty_policy_online: str = "infer_first"     # online stage: infer_first | external_first
    use_pareto: bool = False                # True: predict_all 结果先过帕累托前沿过滤
    pareto_mode: bool = False               # True: 评估器跳过 rubric，直接 quality+noise→pass/fail

    # === LLM 组件配置 ===
    use_llm_decomposer: bool = False   # 使用 Claude Opus 4.6 任务分解器（替代 keyword-based）
    use_llm_evaluator: bool = False    # 使用 Claude Opus 4.6 评估器（替代 MockEvaluator）
    llm_api_key: Optional[str] = None  # Anthropic API key（优先级高于环境变量）

    # === WorkflowGraph 执行层 ===
    use_workflow_graph: bool = False   # 使用显式 WorkflowGraph 层（替代 SubTaskSpec 循环）

    # === Q(G;X) utility weights (per method definition Section 4.4) ===
    # Q(G;X) = αS(G;X) − βC(G;X) − γL(G;X)
    # Weights are normalized internally so α+β+γ=1
    q_alpha: float = 0.6   # weight for quality S
    q_beta: float = 0.2    # weight for cost C
    q_gamma: float = 0.2   # weight for latency L

    def __post_init__(self):
        # Ensure output_dir is a Path
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Episode Log Record
# ---------------------------------------------------------------------------

@dataclass
class EpisodeRecord:
    """
    One record per sub-task execution within an episode.

    Aligned with Method §4 & §8:
      - Cost: c_main (executor), c_llm (judge/evaluator), c_total = c_main + c_llm
              c_usd_raw: raw USD before COST_SCALE; c_norm: normalized [0,1]
      - Q weights: q_alpha, q_beta, q_gamma, q_formula_version
      - Repair scale: delta_v, delta_e, delta_phi, delta_tau, delta_g
      - metric_version for reproducibility
    """

    episode: int
    task_id: str
    sub_task_id: str
    primitive_name: str
    difficulty_bucket: str
    difficulty: float
    selected_candidate: str
    predicted_acc: float
    predicted_cost: float      # predicted c_main (COST_SCALE-scaled USD)
    true_acc: float
    true_cost: float
    observed_acc: float
    observed_cost: float       # observed c_total (COST_SCALE-scaled USD) — deprecated alias
    eval_pass: bool
    failure_type: str | None
    recalibrated: bool
    source: str               # "init_profile" | "bucket" | "fallback"
    difficulty_source: str = "inferred"     # inferred | external
    difficulty_conflict: bool = False
    difficulty_external_bucket: Optional[str] = None
    difficulty_inferred_bucket: Optional[str] = None
    # === 约束与多模态 ===
    constraint_violations: List[Dict[str, Any]] = field(default_factory=list)
    violation_count: int = 0
    human_approved: bool = True
    execution_duration: Optional[float] = None
    input_modality: str = "text"
    intermediate_modality: Optional[str] = None
    # === Evaluator 结构化字段 ===
    evaluator_name: str = "rule_eval"
    evaluator_id: str = "rule_eval"
    error_type: Optional[str] = None
    confidence: float = 1.0
    evaluator_latency: float = 0.0
    evaluator_cost: float = 0.0
    quality_score: float = 0.0
    node_type: str = "unknown"
    # === 实验元信息 ===
    task_type: str = "unknown"
    template_id: str = "unknown"
    repair_action: str = "none"
    template_upgraded_from: str = "none"
    template_upgraded_to: str = "none"
    # === Repair DeltaG (method §8) ===
    repair_deltaG: Optional[dict] = None
    repair_delta_nodes: int = 0    # = delta_v
    repair_delta_edges: int = 0   # = delta_e
    delta_phi: float = 0.0        # executor-tier change magnitude
    delta_tau: float = 0.0       # topology-tier change magnitude
    delta_g: float = 0.0          # combined repair scale = α₀·ΔV + αₑ·ΔE + α_φ·Δφ + α_τ·Δτ
    # === LLM 附加成本 (不计入主成本 C(G)) ===
    llm_decomposer_cost: float = 0.0
    llm_evaluator_cost: float = 0.0
    # === 成本口径 (method §4.3 & §8 主实验口径协议) ===
    c_main: float = 0.0           # executor cost (COST_SCALE-scaled USD)
    c_llm: float = 0.0           # judge/evaluator cost (COST_SCALE-scaled USD)
    c_total: float = 0.0          # c_main + c_llm = main budget item
    c_usd_raw: float = 0.0       # raw USD (before COST_SCALE)
    c_norm: float = 0.0         # normalized cost [0, 1]
    l_raw: float = 0.0          # raw latency in seconds
    l_norm: float = 0.0         # normalized latency [0, 1]
    # === 四级评判 (method §7) ===
    eval_level: str = "pass"    # "pass" | "warn" | "fail" | "escalate"
    initial_eval_level: str = "pass"  # eval_level before repair (for repair_tier tracking)
    repair_tried: List[str] = field(default_factory=list)  # ["candidate", "evaluator", "template"]
    # === Q 公式参数 ===
    q_alpha: float = 0.6
    q_beta: float = 0.2
    q_gamma: float = 0.2
    q_formula_version: str = "v1"
    # === 元数据 ===
    metric_version: str = "v5.0"   # bumped: added cost/utility columns


# ---------------------------------------------------------------------------
# Structured Bias Definition
# ---------------------------------------------------------------------------

BIAS_TABLE: Dict[str, Dict[str, Dict[str, float]]] = {
    "forecast": {
        "fast_nn":        {"easy": +0.08, "medium": +0.18, "hard": +0.22, "extreme": +0.20},
        "ensemble_nn":    {"easy": +0.03, "medium": +0.08, "hard": +0.10, "extreme": +0.10},
        "strong_nn":      {"easy": +0.00, "medium": -0.03, "hard": -0.05, "extreme": -0.05},
        "fvcom":          {"easy": -0.05, "medium": -0.10, "hard": -0.12, "extreme": -0.10},
        "physics_hybrid": {"easy": -0.03, "medium": -0.06, "hard": -0.08, "extreme": -0.08},
    },
    "state_parse": {
        "rule_parser":  {"easy": +0.05, "medium": +0.15, "hard": +0.20, "extreme": +0.18},
        "llm_small":    {"easy": +0.00, "medium": +0.00, "hard": -0.03, "extreme": -0.05},
        "rag_parser":   {"easy": +0.00, "medium": -0.02, "hard": -0.05, "extreme": -0.05},
        "llm_large":    {"easy": -0.03, "medium": -0.08, "hard": -0.10, "extreme": -0.08},
    },
    "data_analysis": {
        "rule_based":     {"easy": +0.08, "medium": +0.20, "hard": +0.25, "extreme": +0.20},
        "ensemble":       {"easy": +0.03, "medium": +0.08, "hard": +0.12, "extreme": +0.10},
        "ml_pipeline":    {"easy": +0.00, "medium": +0.00, "hard": -0.03, "extreme": -0.05},
        "deep_learning":  {"easy": -0.03, "medium": -0.05, "hard": -0.08, "extreme": -0.08},
    },
}


# ---------------------------------------------------------------------------
# Initialization Helper
# ---------------------------------------------------------------------------

def initialize_profile_manager(
    gt_curves: Dict[str, Dict[str, Dict[str, tuple]]],
    fallback_quality: float = 0.5,
    fallback_cost: float = 1.0,
    inject_bias: bool = True,
    # Legacy parameter kept for API compatibility (ignored)
    default_ema_alpha: float = 0.3,
) -> PrimitivePerformanceProfileManager:
    """
    Create a ProfileManager and register all primitives + candidates.
    """
    manager = PrimitivePerformanceProfileManager(
        calibration_interval=10,
        fallback_quality=fallback_quality,
        fallback_cost=fallback_cost,
    )

    for primitive_name, candidates in gt_curves.items():
        manager.register_primitive(primitive_name, primitive_type=primitive_name)

        for candidate_name, buckets in candidates.items():
            init_curve: Dict[str, Dict[str, float]] = {}
            bias_row = BIAS_TABLE.get(primitive_name, {}).get(candidate_name, {})

            for bucket, (acc, cost) in buckets.items():
                bias = bias_row.get(bucket, 0.0) if inject_bias else 0.0
                biased_acc = round(min(max(acc + bias, 0.0), 1.0), 4)
                init_curve[bucket] = {
                    "acc_mean": biased_acc,
                    "cost_mean": round(cost, 4),
                }

            manager.register_candidate(
                primitive_name=primitive_name,
                candidate_name=candidate_name,
                init_curve=init_curve,
            )

    return manager


# ---------------------------------------------------------------------------
# Synthetic Task Generator
# ---------------------------------------------------------------------------

SAMPLE_TASKS = [
    # ---- forecast (easy) ----
    "Simple time series forecasting for next 7 days of daily sales data.",
    "Easy stock price prediction using historical closing prices.",
    # ---- forecast (medium) ----
    "Forecast next 30 days temperature using time series analysis on sensor data.",
    "Time series forecasting for stock price prediction with weekly patterns.",
    "Medium difficulty energy consumption forecasting for building HVAC system.",
    # ---- forecast (hard) ----
    "Complex multi-step long-term weather prediction with extreme weather events.",
    "Hard climate forecasting for seasonal rainfall with noisy satellite data.",
    "Long-term drought prediction using complex time series of soil moisture.",
    # ---- forecast (extreme) ----
    "Extreme weather event prediction across 10-year climate reconstruction.",
    "Catastrophic flood forecasting using chaotic rainfall-runoff time series.",
    # ---- state_parse (easy) ----
    "Easy rule-based parsing of well-formed CSV with clean delimiters.",
    "Simple text parsing for extracting dates and numbers from receipts.",
    # ---- state_parse (medium) ----
    "Parse structured text and extract named entities from news articles.",
    "State parsing of semi-structured JSON logs to extract error messages.",
    "Medium complexity parsing of invoice PDFs to extract line items.",
    # ---- state_parse (hard) ----
    "Complex parsing of messy OCR output from handwritten forms.",
    "Hard parsing of nested JSON with inconsistent schema from API responses.",
    # ---- state_parse (extreme) ----
    "Extreme parsing of multi-column scanned legal documents with table structures.",
    "Complex parsing of unstructured PDF text with mixed languages and noise.",
    # ---- data_analysis (easy) ----
    "Simple data analysis of clean CSV with basic statistics.",
    "Basic classification of iris dataset using simple threshold rules.",
    # ---- data_analysis (medium) ----
    "Medium difficulty data analysis with some missing values in customer data.",
    "Analyze financial transaction data for basic fraud pattern detection.",
    "Medium complexity customer segmentation using behavioral features.",
    # ---- data_analysis (hard) ----
    "Complex multi-step classification with noisy labels on large-scale dataset.",
    "Hard fraud detection in imbalanced credit card transaction data.",
    # ---- data_analysis (extreme) ----
    "Complex anomaly detection in high-dimensional sensor stream data.",
    "Extreme risk scoring using millions of financial transaction features.",
]

# ---------------------------------------------------------------------------
# Constrained Task Bank (hard constraint demonstrations)
# ---------------------------------------------------------------------------

CONSTRAINED_TASK_BANK = [
    # ---- 时间窗约束：必须在 N 秒内完成 ----
    "Analyze fraud in transaction data under 5 second time window.",
    "Detect anomalies in sensor readings with realtime latency requirement.",
    "Classify financial transactions with strict 5s latency.",
    "Forecast temperature with strict 10 second time budget.",
    # ---- 人工审批约束：必须人工确认 ----
    "Forecast quarterly revenue and escalate to human for final approval.",
    "Assess loan default risk and require human review for high-stakes decisions.",
    "Predict stock prices with domain expert audit trail.",
    "Make investment recommendations with regulatory audit compliance.",
    # ---- 必须节点约束：特定 primitive/candidate 必须出现 ----
    "First forecast market trends, then analyze sentiment from news articles.",
    "Parse structured data, then run ML classification with mandatory rule-checker.",
    "Detect fraud, validator is mandatory for hard difficulty tasks.",
    # ---- 风险边界约束：质量地板/成本上限 ----
    "Predict patient readmission with safety-critical quality floor of 0.95.",
    "Classify financial transactions with high-stakes min quality 0.90 and max cost 5.0.",
    "Detect critical infrastructure failures with safety-critical quality floor.",
    "Assess insurance risk with high-stakes min quality 0.88.",
    # ---- 多模态约束：多模态输入 ----
    "Classify image and text pairs for sentiment analysis (multimodal input).",
    "Forecast time series from sensor data with tabular metadata overlay.",
    "Detect anomalies in image and text pairs from surveillance reports.",
    "Analyze multimodal customer feedback combining text reviews and product images.",
]


# ---------------------------------------------------------------------------
# Topology Helpers
# ---------------------------------------------------------------------------

def _topo_sort(tasks: List[SubTaskSpec]) -> List[SubTaskSpec]:
    """
    Topological sort for a DAG of SubTaskSpec nodes (by predecessor_ids).

    Returns tasks ordered so all predecessors appear before their dependents.
    Falls back to insertion order if the graph contains cycles or isolated nodes.
    """
    sorted_tasks: List[SubTaskSpec] = []
    remaining = {st.sub_task_id: st for st in tasks}
    done_ids: set = set()

    while remaining:
        # Find nodes whose all predecessors are already done
        ready = [
            sid for sid, st in remaining.items()
            if all(pid in done_ids for pid in st.predecessor_ids)
        ]
        if not ready:
            ready = list(remaining.keys())  # degenerate: cycle or orphan
        # Pop the first ready node
        sid = ready[0]
        sorted_tasks.append(remaining.pop(sid))
        done_ids.add(sid)

    return sorted_tasks


# ---------------------------------------------------------------------------
# Evaluator Selection Helpers
# ---------------------------------------------------------------------------

def _select_evaluator(
    st: SubTaskSpec,
    executor_name: str,
    config: ExperimentConfig,
) -> str:
    """
    Select evaluator based on task difficulty and executor capability.

    Strategy (minimal implementation):
    - hard/extreme + strong executor (GT >= 0.75): large_eval (high precision)
    - hard/extreme + weak executor: small_eval (medium precision)
    - easy/medium: rule_eval (low precision, low cost)

    The evaluator choice affects both eval_pass accuracy and executor learning signal.
    """
    if st.difficulty_bucket in ("hard", "extreme"):
        # Try to use high-precision evaluator for hard tasks
        return "large_eval"
    elif st.difficulty_bucket == "medium":
        return "small_eval"
    else:
        return "rule_eval"


# ---------------------------------------------------------------------------
# Local Repair Mechanism
# ---------------------------------------------------------------------------

# Upgrade chains per primitive: ordered by quality/cost (low to high)
# Each entry: (candidate_name, description)
UPGRADE_CHAIN: Dict[str, List[Tuple[str, str]]] = {
    "state_parse": [
        ("rule_parser",   "规则解析器"),
        ("llm_small",    "小模型"),
        ("rag_parser",   "RAG 解析器"),
        ("llm_large",   "大模型"),
    ],
    "forecast": [
        ("fast_nn",         "快速神经网络"),
        ("ensemble_nn",     "集成神经网络"),
        ("strong_nn",       "强神经网络"),
        ("fvcom",           "FVCOM 机理模型"),
        ("physics_hybrid",  "物理混合模型"),
    ],
    "data_analysis": [
        ("rule_based",     "规则分析"),
        ("ensemble",       "集成方法"),
        ("ml_pipeline",    "ML 管道"),
        ("deep_learning",  "深度学习"),
    ],
}

REPAIR_THRESHOLD = 0.60     # eval_pass threshold (same as evaluator pass_threshold)
MAX_REPAIR_ATTEMPTS = 2     # max upgrade retries per node per episode

# Template structural upgrade chain for repair Strategy C
TEMPLATE_UPGRADE_CHAIN: Dict[str, str] = {
    "direct": "exec_verify",
    "exec_verify": "dual_exec_aggregate",
    # exec_verify_hci is constraint-driven, not in the quality escalation chain
}


def _upgrade_evaluator(
    st: SubTaskSpec,
    candidate_name: str,
    prev_eval_result: EvaluationResult,
    evaluator: MockEvaluator,
    config: ExperimentConfig,
    rng: random.Random,
) -> Tuple[EvaluationResult | None, dict | None]:
    """
    Attempt repair by upgrading the evaluator (for format/safety errors or low confidence).

    Returns (new_EvaluationResult, deltaG_dict) on success, (None, None) on giving up.

    Strategy:
    - If current evaluator is "rule_eval" → try "small_eval"
    - If current evaluator is "small_eval" → try "large_eval"
    - If current is already "large_eval" → give up

    DeltaG: |ΔG| = 0 (no topology change, only evaluator tier changes)
    """
    current_eval = prev_eval_result.evaluator_name
    upgrade_map = {
        "rule_eval": "small_eval",
        "small_eval": "large_eval",
    }
    next_eval = upgrade_map.get(current_eval)
    if next_eval is None:
        return None, None  # Already at highest evaluator tier

    new_result = evaluator.evaluate(
        candidate_name=candidate_name,
        primitive_name=st.primitive_name,
        difficulty_bucket=st.difficulty_bucket,
        task_id=st.metadata.get("task_id", ""),
        node_id=st.sub_task_id,
        task_spec=st,
        evaluator_name=next_eval,
        node_type=st.primitive_name,
    )
    deltaG = {
        "action": "evaluator_upgrade",
        "from_evaluator": current_eval,
        "to_evaluator": next_eval,
        "candidate_changed": False,
        "template_changed": False,
        "delta_nodes": 0,
        "delta_edges": 0,
        "delta_phi": 1,  # executor tier unchanged (same candidate), but evaluator tier changed
        "delta_tau": 0,
    }
    return new_result, deltaG


def _should_repair(
    eval_result: EvaluationResult,
    repair_attempt_count: int,
) -> bool:
    """
    Decide whether to attempt local repair after a node failure.

    Uses 4-level eval output (per method definition Section 7):
    - FAIL / ESCALATE: trigger repair (escalate = evaluator upgrade first)
    - WARN: continue without repair (borderline quality, log only)
    - PASS: should not reach here

    Conditions:
    1. eval_level in (fail, escalate)
    2. We haven't exhausted repair attempts
    """
    eval_level = getattr(eval_result, "eval_level", "fail")
    if eval_level not in ("fail", "escalate"):
        return False  # PASS or WARN: no repair needed
    if repair_attempt_count >= MAX_REPAIR_ATTEMPTS:
        return False
    return True


def _repair_subgraph(
    st: SubTaskSpec,
    failed_candidate: str,
    eval_result: EvaluationResult,
    manager: PrimitivePerformanceProfileManager,
    evaluator: MockEvaluator,
    config: ExperimentConfig,
    rng: random.Random,
    profile_store: ProfileStore,
    template_library: "TemplateLibrary | None" = None,
    current_template_id: str = "direct",
    use_pareto: bool = False,
    remaining_budget: float | None = None,
    repair_chains: Dict[str, List[Tuple[str, str]]] | None = None,
) -> Tuple[EvaluationResult | None, dict | None, List[str]]:
    """
    Execute local repair for a failed node.

    Returns (new_eval_result, deltaG_dict, strategies_tried) on success,
    (None, None, []) on giving up.

    Repair strategy order (per method: minimize |ΔG| first):
    1. Strategy B (candidate_upgrade) — upgrade executor: smallest Δφ = 1, no topology change
    2. Strategy C (evaluator_upgrade) — upgrade evaluator: for format/safety errors or low confidence
    3. Strategy A (template_upgrade) — change topology: largest |ΔG|, last resort

    Strategy routing by eval_level:
    - FAIL:     B first → C second → A last
    - ESCALATE: C first → B second → A last (evaluator may be unreliable)
    """
    error_type = eval_result.error_type
    confidence = eval_result.confidence
    eval_level = getattr(eval_result, "eval_level", "fail")
    prefer_evaluator_first = (eval_level == "escalate")

    # ---- Helper: get best-ranked candidate that passes hard constraints ----
    def _get_best_candidate(
        candidate_name: str,
        within_budget: float | None,
    ) -> dict | None:
        if use_pareto:
            frontier = manager.pareto_frontier(st.primitive_name, st.difficulty_bucket)
            if not frontier:
                return None
            try:
                return manager.select_from_frontier(
                    frontier,
                    acc_target=config.acc_target,
                    cost_budget=within_budget,
                    latency_budget=config.latency_budget,
                    alpha=config.q_alpha,
                    beta=config.q_beta,
                )
            except ValueError:
                return None
        else:
            all_cands = manager.predict_all(
                st.primitive_name,
                st.difficulty_bucket,
                acc_target=config.acc_target,
                cost_budget=within_budget,
                latency_budget=config.latency_budget,
            )
            return all_cands[0] if all_cands else None

    # ---- Strategy B: upgrade candidate along repair_chains ----
    # repair_chains overrides the global UPGRADE_CHAIN (allows per-task chains)
    _chains = repair_chains if repair_chains is not None else UPGRADE_CHAIN
    chain = _chains.get(st.primitive_name, [])
    chain_names = [c for c, _ in chain] if chain else []

    def _try_candidate_upgrade(from_cand: str) -> Tuple[EvaluationResult | None, dict | None]:
        try:
            current_idx = chain_names.index(from_cand)
        except ValueError:
            current_idx = -1

        for next_cand, _ in chain[current_idx + 1:]:
            # Check remaining budget before attempting upgrade
            if remaining_budget is not None:
                pred = manager.predict(st.primitive_name, next_cand, st.difficulty_bucket)
                if pred.get("pred_cost", 0) > remaining_budget:
                    continue  # Would exceed remaining budget, skip

            cand_info = _get_best_candidate(next_cand, remaining_budget)
            if cand_info is None:
                continue

            # Pre-check HITL constraint
            if st.has_human_approval_required() and config.enable_human_hitl:
                if not _simulate_human_approval(st, config):
                    continue

            eval_name = _select_evaluator(st, next_cand, config)
            new_result = evaluator.evaluate(
                candidate_name=next_cand,
                primitive_name=st.primitive_name,
                difficulty_bucket=st.difficulty_bucket,
                task_id=st.metadata.get("task_id", ""),
                node_id=st.sub_task_id,
                task_spec=st,
                evaluator_name=eval_name,
                node_type=st.primitive_name,
            )

            if profile_store is not None:
                profile_store.update_evaluator_profile(
                    evaluator_id=new_result.evaluator_name,
                    difficulty=st.difficulty_bucket,
                    observed_pass=new_result.eval_pass,
                    true_pass=(new_result.true_quality or 0) >= evaluator.pass_threshold,
                    evaluator_latency=new_result.evaluator_latency,
                    evaluator_cost=new_result.evaluator_cost,
                )

            if new_result.eval_pass:
                deltaG = {
                    "action": "candidate_upgrade",
                    "from_candidate": from_cand,
                    "to_candidate": next_cand,
                    "candidate_changed": True,
                    "template_changed": False,
                    "delta_nodes": 0,
                    "delta_edges": 0,
                    "delta_phi": 1,   # one executor tier changed
                    "delta_tau": 0,
                }
                return new_result, deltaG
        return None, None

    # ---- Strategy C: upgrade evaluator (for format/safety errors or low confidence) ----
    def _try_evaluator_upgrade(from_cand: str) -> Tuple[EvaluationResult | None, dict | None]:
        if error_type in ("format_error", "unsafe_decision") or confidence < 0.5:
            new_result, deltaG = _upgrade_evaluator(
                st, from_cand, eval_result, evaluator, config, rng
            )
            if new_result is not None and new_result.eval_pass:
                return new_result, deltaG
        return None, None

    # ---- Strategy A: template/topology upgrade (largest |ΔG|, last resort) ----
    def _try_template_upgrade() -> Tuple[EvaluationResult | None, dict | None]:
        if template_library is None:
            return None, None

        next_template_id = TEMPLATE_UPGRADE_CHAIN.get(current_template_id)
        if next_template_id is None:
            return None, None

        next_template = template_library.get_template(next_template_id)
        if next_template is None:
            return None, None

        base_id = st.sub_task_id.rsplit("_", 1)[0] if "_" in st.sub_task_id else st.sub_task_id
        instanced = next_template.instantiate(
            base_sub_task_id=base_id,
            base_primitive=st.primitive_name,
            base_difficulty=st.difficulty,
            difficulty_bucket=st.difficulty_bucket,
            constraints=st.constraints,
        )

        exec_node = next(
            (n for n in instanced if "exec" in n.sub_task_id),
            instanced[0] if instanced else None,
        )
        if exec_node is None:
            return None, None

        # Select best candidate under the new topology (with hard constraints)
        best = _get_best_candidate(failed_candidate, remaining_budget)
        if best is None:
            return None, None

        eval_name = _select_evaluator(exec_node, best["candidate_name"], config)
        new_result = evaluator.evaluate(
            candidate_name=best["candidate_name"],
            primitive_name=st.primitive_name,
            difficulty_bucket=st.difficulty_bucket,
            task_id=st.metadata.get("task_id", ""),
            node_id=exec_node.sub_task_id,
            task_spec=exec_node,
            evaluator_name=eval_name,
            node_type=st.primitive_name,
        )

        if profile_store is not None:
            profile_store.update_evaluator_profile(
                evaluator_id=new_result.evaluator_name,
                difficulty=st.difficulty_bucket,
                observed_pass=new_result.eval_pass,
                true_pass=(new_result.true_quality or 0) >= evaluator.pass_threshold,
                evaluator_latency=new_result.evaluator_latency,
                evaluator_cost=new_result.evaluator_cost,
            )

        if new_result.eval_pass:
            # Compute topology change scale
            from_nodes = 1
            to_nodes = len(instanced)
            delta_nodes = max(0, to_nodes - from_nodes)
            delta_edges = 1  # topology upgrade adds at least one edge (verifier or aggregator)
            deltaG = {
                "action": "template_upgrade",
                "from_template": current_template_id,
                "to_template": next_template_id,
                "candidate_changed": False,
                "template_changed": True,
                "delta_nodes": delta_nodes,
                "delta_edges": delta_edges,
                "delta_phi": 0,
                "delta_tau": 1,   # topology change introduces new node types
            }
            return new_result, deltaG
        return None, None

    # ---- Execute repair strategy order ----
    # Per method §8:
    #   FAIL:     B → C → A (minimize |ΔG| first, topology last)
    #   ESCALATE: C → B → A (evaluator may be unreliable → upgrade evaluator first)
    if prefer_evaluator_first:
        strategy_order: list = ["evaluator", "candidate", "template"]
    else:
        strategy_order = ["candidate", "evaluator", "template"]

    for strategy in strategy_order:
        if strategy == "candidate":
            result, deltaG = _try_candidate_upgrade(failed_candidate)
        elif strategy == "evaluator":
            result, deltaG = _try_evaluator_upgrade(failed_candidate)
        else:
            result, deltaG = _try_template_upgrade()

        if result is not None and result.eval_pass:
            return result, deltaG, strategy_order[:strategy_order.index(strategy) + 1]

    return None, None, list(strategy_order)  # All strategies exhausted


# ---------------------------------------------------------------------------
# Constraint Orchestration Helpers
# ---------------------------------------------------------------------------

def _check_candidate_violations(
    candidate: dict,
    task_spec: SubTaskSpec,
) -> List[dict]:
    """
    Check whether a candidate violates any hard constraints from the task spec.

    Returns a list of violation dicts (empty = no violations).

    Checks:
    - RiskBoundaryConstraint: candidate's predicted cost > max_cost for this bucket
    - MandatoryNodeConstraint: checked at topology level (see _inject_mandatory_nodes)
    """
    violations = []

    for c in task_spec.get_active_constraints():
        if isinstance(c, RiskBoundaryConstraint):
            # Check predicted cost against difficulty-specific max_cost
            if c.max_cost is not None:
                limit = c.get_max_cost_for_bucket(task_spec.difficulty_bucket)
                if limit is not None and candidate["pred_cost"] > limit:
                    violations.append({
                        "constraint_id": c.constraint_id,
                        "constraint_type": "risk_boundary",
                        "reason": (
                            f"candidate '{candidate['candidate_name']}' pred_cost="
                            f"{candidate['pred_cost']:.4f} exceeds "
                            f"max_cost={limit} for bucket '{task_spec.difficulty_bucket}'"
                        ),
                    })

            # Check predicted quality against min_quality
            if c.min_quality is not None and candidate["pred_acc"] < c.min_quality:
                violations.append({
                    "constraint_id": c.constraint_id,
                    "constraint_type": "risk_boundary",
                    "reason": (
                        f"candidate '{candidate['candidate_name']}' pred_acc="
                        f"{candidate['pred_acc']:.4f} is below "
                        f"min_quality={c.min_quality}"
                    ),
                })

            # Check predicted latency against max_latency
            if c.max_latency is not None and "pred_latency" in candidate:
                if candidate["pred_latency"] > c.max_latency:
                    violations.append({
                        "constraint_id": c.constraint_id,
                        "constraint_type": "risk_boundary",
                        "reason": (
                            f"candidate '{candidate['candidate_name']}' pred_latency="
                            f"{candidate['pred_latency']:.3f}s exceeds "
                            f"max_latency={c.max_latency}s"
                        ),
                    })

        elif isinstance(c, TimeWindowConstraint):
            # The actual duration check is done by the evaluator.
            # Here we do a quick estimate: if pred_cost is very high, it's likely to timeout.
            # A simple proxy: duration ~ cost * 10, so timeout if cost > max_duration/10
            max_cost_proxy = c.max_duration / 10.0
            if candidate["pred_cost"] > max_cost_proxy * 1.5:
                # Flag as potential violation (evaluator makes the final call)
                pass  # Duration is evaluated post-execution, not pre-selection

    return violations


def _apply_constraints(
    candidates: List[dict],
    task_spec: SubTaskSpec,
    violation_penalty: float,
) -> List[dict]:
    """
    Apply hard constraints to candidate ranking.

    Strategy:
    1. Filter out candidates with hard violations (risk boundary quality floor).
       Risk boundary cost violations are penalized but NOT filtered.
    2. Demote violating candidates by reducing pred_acc (penalty proxy).
    3. Return filtered list with non-violating candidates first.

    MandatoryNodeConstraint is handled separately by _inject_mandatory_nodes().
    HumanInTheLoopConstraint is handled in the main loop (human approval step).
    """
    if not task_spec.constraints:
        return candidates

    clean: List[dict] = []
    penalized: List[dict] = []

    for c in candidates:
        viols = _check_candidate_violations(c, task_spec)
        if not viols:
            clean.append(c)
        else:
            c_copy = dict(c)
            c_copy["violations"] = viols
            # Penalize quality for constraint-violating candidates
            c_copy["pred_acc"] = max(0.0, round(c_copy["pred_acc"] + violation_penalty, 4))
            if c_copy["pred_acc"] > 0:
                penalized.append(c_copy)

    # Non-violating candidates first (sorted by quality desc), then penalized
    clean.sort(key=lambda x: x["pred_acc"], reverse=True)
    penalized.sort(key=lambda x: x["pred_acc"], reverse=True)
    filtered = clean + penalized

    if not filtered:
        # All candidates violated — return originals sorted by quality
        candidates.sort(key=lambda x: x["pred_acc"], reverse=True)
        return candidates

    return filtered


def _check_mandatory_node_violations(
    topology: List[SubTaskSpec],
    manager: "PrimitivePerformanceProfileManager | None" = None,
) -> List[dict]:
    """
    Check if mandatory node constraints are satisfied by the current topology.

    Returns a list of violations (empty = all mandatory constraints satisfied).

    Parameters
    ----------
    topology : List[SubTaskSpec]
        Current sub-task specification list.
    manager : PrimitivePerformanceProfileManager, optional
        Profile manager for verifying required_candidate availability.
        If not provided, required_candidate checks are skipped.
    """
    violations = []

    for st in topology:
        for c in st.get_active_constraints():
            if not isinstance(c, MandatoryNodeConstraint):
                continue

            # Check primitive requirement
            if c.required_primitive is not None:
                primitives_in_topology = [s.primitive_name for s in topology]
                if c.required_primitive not in primitives_in_topology:
                    violations.append({
                        "constraint_id": c.constraint_id,
                        "constraint_type": "mandatory_node",
                        "reason": (
                            f"required_primitive '{c.required_primitive}' not found "
                            f"in topology {primitives_in_topology} "
                            f"(constraint: {c.description})"
                        ),
                    })

            # Check candidate requirement: the required candidate must be
            # registered under the primitive AND selectable from the profile pool.
            if c.required_candidate is not None and manager is not None:
                required_cand = c.required_candidate
                prim = st.primitive_name
                if prim in manager._primitives:
                    available = manager._primitives[prim].list_candidates()
                    if required_cand not in available:
                        violations.append({
                            "constraint_id": c.constraint_id,
                            "constraint_type": "mandatory_node",
                            "reason": (
                                f"required_candidate '{required_cand}' is not registered "
                                f"under primitive '{prim}'. Available: {available}"
                            ),
                        })
                else:
                    violations.append({
                        "constraint_id": c.constraint_id,
                        "constraint_type": "mandatory_node",
                        "reason": (
                            f"required_candidate '{required_cand}' specified but "
                            f"primitive '{prim}' has no registered candidates."
                        ),
                    })

    return violations


def _simulate_human_approval(
    task_spec: SubTaskSpec,
    config: ExperimentConfig,
) -> bool:
    """
    Simulate human approval outcome for a HITL constraint.
    In the real system, this would pause execution and wait for human input.
    In the MVP, we use a simple model:
    - If enable_human_hitl=False: always approve
    - Otherwise: approve with probability based on task risk profile
    """
    if not config.enable_human_hitl:
        return True

    rng = random.Random(config.seed)
    # Conservative: 80% approval rate when HITL is enabled
    return rng.random() < 0.80


def _report_constraint_metrics(
    records: List[EpisodeRecord],
    config: ExperimentConfig,
) -> Dict[str, Any]:
    """
    Compute and print constraint satisfaction metrics.
    """
    if not records:
        return {}

    total = len(records)
    n_violations = sum(1 for r in records if r.violation_count > 0)
    n_human_rejected = sum(1 for r in records if not r.human_approved)
    n_timeout = sum(1 for r in records if r.failure_type == "timeout")

    violation_rate = round(n_violations / total, 4)

    # Per-constraint-type breakdown
    type_counter: Dict[str, int] = Counter()
    for r in records:
        for v in r.constraint_violations:
            type_counter[v["constraint_type"]] += 1

    metrics = {
        "total_executions": total,
        "violation_count": n_violations,
        "violation_rate": violation_rate,
        "human_rejected_count": n_human_rejected,
        "timeout_count": n_timeout,
        "violations_by_type": dict(type_counter),
    }

    print(f"\n  [Constraint Metrics]")
    print(f"    Total executions:     {total}")
    print(f"    Violations:          {n_violations} ({violation_rate:.1%})")
    print(f"    Human rejected:      {n_human_rejected}")
    print(f"    Timeouts:            {n_timeout}")
    print(f"    Violations by type:  {dict(type_counter)}")

    return metrics


# ---------------------------------------------------------------------------
# 4 Main Metrics Summary
# ---------------------------------------------------------------------------

def _build_four_metrics(
    records: List[EpisodeRecord],
    constraint_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the 4 main metrics dict (used in return value and by analyze_results.py)."""
    if not records:
        return {}

    total = len(records)
    n_pass = sum(1 for r in records if r.eval_pass)
    mae = round(
        sum(abs(r.predicted_acc - r.true_acc) for r in records) / total, 4
    )
    durations = [r.execution_duration for r in records if r.execution_duration is not None]
    eval_latencies = [r.evaluator_latency for r in records if r.evaluator_latency > 0]
    eval_costs = [r.evaluator_cost for r in records if r.evaluator_cost > 0]
    exec_costs = [r.observed_cost for r in records]

    by_bucket: Dict[str, List[EpisodeRecord]] = {}
    for r in records:
        by_bucket.setdefault(r.difficulty_bucket, []).append(r)

    return {
        "quality": {
            "pass_rate": round(n_pass / total, 4),
            "n_pass": n_pass,
            "total": total,
            "mae": mae,
            "per_bucket_pass_rate": {
                b: round(sum(1 for r in rs if r.eval_pass) / len(rs), 4)
                for b, rs in by_bucket.items()
            },
        },
        "latency": {
            "mean_execution_duration": (
                round(sum(durations) / len(durations), 4) if durations else None
            ),
            "mean_evaluator_latency": (
                round(sum(eval_latencies) / len(eval_latencies), 4) if eval_latencies else None
            ),
            "mean_total_latency": (
                round((sum(durations) + sum(eval_latencies)) / len(records), 4)
                if durations or eval_latencies else None
            ),
        },
        "tokens": {
            "mean_evaluator_cost": (
                round(sum(eval_costs) / len(eval_costs), 4) if eval_costs else None
            ),
            "mean_executor_cost": (
                round(sum(exec_costs) / len(exec_costs), 4) if exec_costs else None
            ),
            "mean_total_cost": (
                round((sum(eval_costs) + sum(exec_costs)) / len(records), 4)
                if eval_costs or exec_costs else None
            ),
        },
        "violations": {
            "violation_rate": constraint_metrics.get("violation_rate", 0.0),
            "violation_count": constraint_metrics.get("violation_count", 0),
            "violations_by_type": constraint_metrics.get("violations_by_type", {}),
        },
    }


def _print_four_metrics(
    records: List[EpisodeRecord],
    constraint_metrics: Dict[str, Any],
) -> None:
    """
    Print the 4 main metrics in a structured, paper-ready format.

    Metric 1 — Quality:   eval_pass rate + prediction MAE
    Metric 2 — Latency:  mean execution_duration + evaluator_latency
    Metric 3 — Tokens:   mean evaluator_cost (API cost as proxy)
    Metric 4 — Violation: constraint violation rate + breakdown by type
    """
    if not records:
        print("\n  [4 Main Metrics] No records — cannot compute.")
        return

    total = len(records)

    # --- Metric 1: Quality ---
    n_pass = sum(1 for r in records if r.eval_pass)
    pass_rate = n_pass / total

    mae_list = [abs(r.predicted_acc - r.true_acc) for r in records]
    mae = round(sum(mae_list) / len(mae_list), 4) if mae_list else None

    # --- Quick summary for Metric 2-4 ---
    durations = [r.execution_duration for r in records if r.execution_duration is not None]
    eval_latencies = [r.evaluator_latency for r in records if r.evaluator_latency > 0]
    eval_costs = [r.evaluator_cost for r in records if r.evaluator_cost > 0]
    exec_costs = [r.observed_cost for r in records]
    violation_rate = constraint_metrics.get("violation_rate", 0.0)

    mean_total_latency = (
        round((sum(durations) + sum(eval_latencies)) / len(records), 4)
        if durations or eval_latencies else 0.0
    )
    mean_total_cost = (
        round((sum(eval_costs) + sum(exec_costs)) / len(records), 4)
        if eval_costs or exec_costs else 0.0
    )

    # === 4-line SUMMARY banner ===
    print(f"\n{'='*35} SUMMARY {'='*35}")
    print(f"  Final Quality (pass_rate) : {pass_rate:.1%}")
    print(f"  Total Latency (mean)      : {mean_total_latency:.3f}s")
    print(f"  Total Cost (mean)         : {mean_total_cost:.3f}")
    print(f"  Violation Rate            : {violation_rate:.1%}")
    print(f"{'='*79}")

    # Per-bucket pass rate
    by_bucket: Dict[str, List[EpisodeRecord]] = {}
    for r in records:
        by_bucket.setdefault(r.difficulty_bucket, []).append(r)
    bucket_pass_rates = {
        b: round(sum(1 for r in rs if r.eval_pass) / len(rs), 4)
        for b, rs in by_bucket.items()
    }

    # --- Metric 2: Latency ---
    durations = [r.execution_duration for r in records if r.execution_duration is not None]
    eval_latencies = [r.evaluator_latency for r in records if r.evaluator_latency > 0]
    mean_duration = round(sum(durations) / len(durations), 4) if durations else None
    mean_eval_latency = round(sum(eval_latencies) / len(eval_latencies), 4) if eval_latencies else None
    mean_total_latency = (
        round((sum(durations) + sum(eval_latencies)) / len(records), 4)
        if durations or eval_latencies else None
    )

    # --- Metric 3: Tokens (API cost proxy) ---
    eval_costs = [r.evaluator_cost for r in records if r.evaluator_cost > 0]
    exec_costs = [r.observed_cost for r in records]
    mean_eval_cost = round(sum(eval_costs) / len(eval_costs), 4) if eval_costs else None
    mean_exec_cost = round(sum(exec_costs) / len(exec_costs), 4) if exec_costs else None
    mean_total_cost = (
        round((sum(eval_costs) + sum(exec_costs)) / len(records), 4)
        if eval_costs or exec_costs else None
    )

    # --- Metric 4: Violation ---
    violation_rate = constraint_metrics.get("violation_rate", 0.0)
    violations_by_type = constraint_metrics.get("violations_by_type", {})
    n_violations = constraint_metrics.get("violation_count", 0)

    print(f"\n{'='*60}")
    print("  4 MAIN METRICS (Paper-Ready)")
    print(f"{'='*60}")

    print(f"\n  [M1] Quality")
    print(f"    Overall pass rate : {pass_rate:.1%}  ({n_pass}/{total})")
    print(f"    Prediction MAE    : {mae:.4f}" if mae is not None else "    Prediction MAE    : N/A")
    print(f"    Per-bucket pass   : " + "  ".join(
        f"{b}={r:.0%}" for b, r in sorted(bucket_pass_rates.items())
    ))

    lat_str = (
        f"execution={mean_duration:.3f}s" if mean_duration is not None else "execution=N/A"
    )
    eval_lat_str = (
        f"+ eval={mean_eval_latency:.3f}s" if mean_eval_latency is not None else ""
    )
    total_lat_str = f"= {mean_total_latency:.3f}s" if mean_total_latency is not None else ""
    print(f"\n  [M2] Latency")
    print(f"    Mean per task     : {lat_str}{eval_lat_str} {total_lat_str}")
    if durations:
        print(f"    Min/Max           : {min(durations):.3f}s / {max(durations):.3f}s")

    cost_str = (
        f"execution={mean_exec_cost:.3f}" if mean_exec_cost is not None else "execution=N/A"
    )
    eval_cost_str = (
        f"+ eval={mean_eval_cost:.3f}" if mean_eval_cost is not None else ""
    )
    total_cost_str = f"= {mean_total_cost:.3f}" if mean_total_cost is not None else ""
    print(f"\n  [M3] Tokens (API cost proxy)")
    print(f"    Mean per task     : {cost_str}{eval_cost_str} {total_cost_str}")

    print(f"\n  [M4] Constraint Violations")
    print(f"    Violation rate    : {violation_rate:.1%}  ({n_violations}/{total})")
    if violations_by_type:
        print(f"    By type           : " + "  ".join(
            f"{k}={v}" for k, v in sorted(violations_by_type.items(), key=lambda x: -x[1])
        ))

    # --- Repair action breakdown ---
    action_counter: Dict[str, int] = {}
    deltaG_total: Dict[str, int] = {"delta_nodes": 0, "delta_edges": 0, "delta_phi": 0, "delta_tau": 0}
    for r in records:
        action_counter[r.repair_action] = action_counter.get(r.repair_action, 0) + 1
        if r.repair_deltaG:
            deltaG_total["delta_nodes"] += r.repair_deltaG.get("delta_nodes", 0)
            deltaG_total["delta_edges"] += r.repair_deltaG.get("delta_edges", 0)
            deltaG_total["delta_phi"]   += r.repair_deltaG.get("delta_phi", 0)
            deltaG_total["delta_tau"]   += r.repair_deltaG.get("delta_tau", 0)

    print(f"\n  [M5] Repair Actions (B→C→A = candidate→evaluator→template)")
    for action, count in sorted(action_counter.items(), key=lambda x: -x[1]):
        share = count / total
        bar = "█" * int(share * 20)
        print(f"    {action:25s}: {count:>4}  ({share:5.1%})  {bar}")
    if deltaG_total["delta_nodes"] or deltaG_total["delta_edges"] or deltaG_total["delta_phi"] or deltaG_total["delta_tau"]:
        print(f"    DeltaG totals: nodes={deltaG_total['delta_nodes']}  edges={deltaG_total['delta_edges']}  "
              f"phi={deltaG_total['delta_phi']}  tau={deltaG_total['delta_tau']}")

    print(f"\n{'='*60}")


# ---------------------------------------------------------------------------
# Structural Metrics: Template Selection & Repair Distribution
# ---------------------------------------------------------------------------

def _build_structural_metrics(
    records: List[EpisodeRecord],
) -> Dict[str, Any]:
    """
    Build structural metrics showing template selection and repair distributions.

    Used to demonstrate structure–configuration joint optimization:
    - Template selection varies with difficulty and constraint context
    - Repair can escalate template structure (Strategy A — largest |ΔG|, last resort)
    """
    if not records:
        return {}

    # Template selection distribution (overall)
    template_counts: Dict[str, int] = {}
    for r in records:
        template_counts[r.template_id] = template_counts.get(r.template_id, 0) + 1

    # Template selection distribution by difficulty bucket
    template_by_bucket: Dict[str, Dict[str, int]] = {}
    for r in records:
        if r.difficulty_bucket not in template_by_bucket:
            template_by_bucket[r.difficulty_bucket] = {}
        bucket_dict = template_by_bucket[r.difficulty_bucket]
        bucket_dict[r.template_id] = bucket_dict.get(r.template_id, 0) + 1

    # Template selection by constraint presence
    template_by_constraint: Dict[str, Dict[str, int]] = {}
    for r in records:
        constraint_key = "unconstrained"
        if r.constraint_violations:
            types = set()
            for v in r.constraint_violations:
                if isinstance(v, dict):
                    types.add(v.get("constraint_type", "unknown"))
            if types:
                constraint_key = "+".join(sorted(types))
        elif not r.human_approved:
            constraint_key = "human_in_the_loop"
        if constraint_key not in template_by_constraint:
            template_by_constraint[constraint_key] = {}
        cdict = template_by_constraint[constraint_key]
        cdict[r.template_id] = cdict.get(r.template_id, 0) + 1

    # Template upgrade count (repair Strategy A — largest |ΔG|, last resort)
    n_template_upgrades = sum(
        1 for r in records
        if r.template_upgraded_from != "none"
    )

    # Template upgrade transitions
    upgrade_transitions: Dict[str, int] = {}
    for r in records:
        if r.template_upgraded_from != "none":
            key = f"{r.template_upgraded_from} -> {r.template_id}"
            upgrade_transitions[key] = upgrade_transitions.get(key, 0) + 1

    return {
        "template_counts": template_counts,
        "template_by_bucket": template_by_bucket,
        "template_by_constraint": template_by_constraint,
        "n_template_upgrades": n_template_upgrades,
        "upgrade_transitions": upgrade_transitions,
        "total_records": len(records),
    }


def _print_structural_metrics(records: List[EpisodeRecord]) -> None:
    """Print structural metrics: template distribution by difficulty and constraint."""
    metrics = _build_structural_metrics(records)
    if not metrics:
        return

    print(f"\n{'='*60}")
    print("  STRUCTURAL METRICS (Template Selection Distribution)")
    print(f"{'='*60}")

    # Overall template distribution
    template_counts = metrics["template_counts"]
    total = metrics["total_records"]
    print(f"\n  [Overall Template Selection]")
    print(f"  {'Template':25s}  {'Count':>6}  {'Share':>7}")
    print(f"  {'-'*42}")
    for tid in sorted(template_counts.keys()):
        cnt = template_counts[tid]
        share = cnt / total if total > 0 else 0
        print(f"  {tid:25s}  {cnt:>6}  {share:>7.1%}")

    # Template by difficulty bucket
    tbb = metrics["template_by_bucket"]
    all_templates = sorted(set(
        tid for d in tbb.values() for tid in d.keys()
    ))
    if all_templates:
        print(f"\n  [Template Selection by Difficulty]")
        header = f"  {'Bucket':10s}" + "".join(f"  {t:>20s}" for t in all_templates)
        print(header)
        print(f"  {'-' * (10 + 22 * len(all_templates))}")
        for bucket in ["easy", "medium", "hard", "extreme"]:
            if bucket in tbb:
                row = f"  {bucket:10s}" + "".join(
                    f"  {tbb[bucket].get(t, 0):>20d}" for t in all_templates
                )
                print(row)

    # Template upgrades
    print(f"\n  [Template Upgrades During Repair (Strategy A — |ΔG| = largest)]")
    print(f"    Total template upgrades: {metrics['n_template_upgrades']}")
    if metrics["upgrade_transitions"]:
        for transition, count in metrics["upgrade_transitions"].items():
            print(f"    {transition}: {count}")
    else:
        print(f"    (no template upgrades occurred)")

    print(f"{'='*60}")


def _analyze_constraint_convergence(
    episode_records: List[EpisodeRecord],
    curve_snapshots: list,
    window_size: int = 10,
) -> Dict[str, Any]:
    """
    Analyze how constraint violation rates evolve over episodes.

    Returns a dict with:
        - violation_rate_by_window: list of (window_start_ep, rate)
        - violation_rate_by_type: dict[constraint_type -> list of (window_start, rate)]
        - recal_vs_no_recal: dict with rates before/after first recalibration
        - candidate_violation_distribution: dict[candidate -> count]
        - bucket_violation_distribution: dict[bucket -> count]
        - hitl_rejection_by_window: list of (window_start, rejection_rate)
        - summary: str
    """
    if not episode_records:
        return {"summary": "No records"}

    episodes = sorted(set(r.episode for r in episode_records))
    type_rates: Dict[str, list] = defaultdict(list)
    candidate_viol_dist: Dict[str, int] = defaultdict(int)
    bucket_viol_dist: Dict[str, int] = defaultdict(int)

    # Sliding window violation rate
    violation_rate_by_window = []
    for i in range(0, len(episodes), window_size):
        window_eps = episodes[i:i + window_size]
        window_recs = [r for r in episode_records if r.episode in window_eps]
        n_viol = sum(1 for r in window_recs if r.violation_count > 0)
        rate = round(n_viol / len(window_recs), 4) if window_recs else 0.0
        violation_rate_by_window.append((window_eps[0], rate))

        # Per-type in this window
        type_counter: Dict[str, int] = defaultdict(int)
        for r in window_recs:
            for v in r.constraint_violations:
                type_counter[v["constraint_type"]] += 1
        for ctype, cnt in type_counter.items():
            rate_t = cnt / len(window_recs) if window_recs else 0.0
            type_rates[ctype].append((window_eps[0], round(rate_t, 4)))

        # Distributions
        for r in window_recs:
            if r.violation_count > 0:
                candidate_viol_dist[r.selected_candidate] += 1
                bucket_viol_dist[r.difficulty_bucket] += 1

    # HITL rejection by window
    hitl_rejection_by_window = []
    for i in range(0, len(episodes), window_size):
        window_eps = episodes[i:i + window_size]
        window_recs = [r for r in episode_records if r.episode in window_eps]
        hitl_recs = [
            r for r in window_recs
            if "human_in_the_loop" in str(r.constraint_violations)
        ]
        n_rejected = sum(1 for r in hitl_recs if r.human_approved == False)
        rate = round(n_rejected / len(hitl_recs), 4) if hitl_recs else 0.0
        hitl_rejection_by_window.append((window_eps[0], rate))

    # Recal vs no-recal
    if curve_snapshots:
        first_recal_ep = curve_snapshots[0].get("episode_at_calibration", episodes[0])
        before_recal = [r for r in episode_records if r.episode < first_recal_ep]
        after_recal = [r for r in episode_records if r.episode >= first_recal_ep]
    else:
        before_recal = episode_records
        after_recal = []

    n_viol_before = sum(1 for r in before_recal if r.violation_count > 0)
    n_viol_after = sum(1 for r in after_recal if r.violation_count > 0)
    rate_before = round(n_viol_before / len(before_recal), 4) if before_recal else 0.0
    rate_after = round(n_viol_after / len(after_recal), 4) if after_recal else 0.0

    if rate_after < rate_before:
        trend = f"IMPROVING (before={rate_before:.1%}, after={rate_after:.1%})"
    elif rate_after > rate_before:
        trend = f"DEGRADING (before={rate_before:.1%}, after={rate_after:.1%})"
    else:
        trend = f"STABLE ({rate_before:.1%})"

    top_violators = ", ".join(
        f"{k}({v})" for k, v in sorted(candidate_viol_dist.items(), key=lambda x: -x[1])[:3]
    )
    summary = (
        f"Violation rate {trend}. "
        f"Top violators: {top_violators or 'none'}. "
        f"Total: {sum(candidate_viol_dist.values())} violations / {len(episode_records)} records."
    )

    return {
        "violation_rate_by_window": violation_rate_by_window,
        "violation_rate_by_type": dict(type_rates),
        "recal_vs_no_recal": {
            "rate_before": rate_before,
            "rate_after": rate_after,
            "improvement": rate_before - rate_after,
        },
        "candidate_violation_distribution": dict(candidate_viol_dist),
        "bucket_violation_distribution": dict(bucket_viol_dist),
        "hitl_rejection_by_window": hitl_rejection_by_window,
        "summary": summary,
    }


def _print_constraint_analysis(analysis: Dict[str, Any]) -> None:
    """Print constraint convergence analysis in a human-readable format."""
    print(f"\n{'='*60}")
    print("  CONSTRAINT CONVERGENCE ANALYSIS")
    print(f"{'='*60}")

    # Sliding window violation rates (ASCII bar)
    vr = analysis.get("violation_rate_by_window", [])
    if vr:
        print(f"\n  [Violation Rate by Episode Window]")
        print(f"  {'Ep':>5}  {'Rate':>7}  {'Bar'}")
        print("  " + "-" * 50)
        for start, rate in vr:
            filled = int(rate * 20)
            bar = "#" * filled + "-" * max(0, 20 - filled)
            print(f"  {start:>5}:  {rate:>7.1%}  [{bar}]")

    # Recal vs no-recal
    recal = analysis.get("recal_vs_no_recal", {})
    if recal:
        print(f"\n  [Recalibration Impact]")
        print(f"    Before first recal: {recal['rate_before']:.1%} violation rate")
        print(f"    After  first recal: {recal['rate_after']:.1%} violation rate")
        imp = recal.get("improvement", 0)
        arrow = "IMPROVED" if imp > 0 else ("WORSENED" if imp < 0 else "STABLE")
        print(f"    -> {arrow} by {abs(imp):.1%}")

    # Candidate distribution
    cand_dist = analysis.get("candidate_violation_distribution", {})
    if cand_dist:
        print(f"\n  [Violation by Candidate]")
        print(f"  {'Candidate':15s}  {'Violations':>10}")
        print("  " + "-" * 28)
        for cand, cnt in sorted(cand_dist.items(), key=lambda x: -x[1]):
            bar = "#" * min(cnt, 20)
            print(f"  {cand:15s}  {cnt:>10}  {bar}")

    # Bucket distribution
    bucket_dist = analysis.get("bucket_violation_distribution", {})
    if bucket_dist:
        print(f"\n  [Violation by Difficulty Bucket]")
        for bucket, cnt in sorted(bucket_dist.items()):
            bar = "#" * min(cnt, 20)
            print(f"  {bucket:10s}  {cnt:>10}  {bar}")

    # HITL
    hitl = analysis.get("hitl_rejection_by_window", [])
    if hitl:
        print(f"\n  [HITL Rejection Rate by Window]")
        print(f"  {'Ep':>5}  {'RejectRate':>12}  {'Bar'}")
        print("  " + "-" * 35)
        for start, rate in hitl:
            filled = int(rate * 20)
            bar = "#" * filled + "-" * max(0, 20 - filled)
            print(f"  {start:>5}:  {rate:>11.1%}  [{bar}]")

    print(f"\n  [Summary]")
    print(f"    {analysis.get('summary', 'N/A')}")
    print(f"\n{'='*60}")


# ---------------------------------------------------------------------------
# Difficulty bucket ↔ float mapping (shared by direct mode & calibration JSONL)
# ---------------------------------------------------------------------------
BUCKET_TO_FLOAT: Dict[str, float] = {
    "easy": 0.85, "medium": 0.60, "hard": 0.35, "extreme": 0.15,
}


def _normalize_bucket_name(bucket: str | None) -> str | None:
    if bucket is None:
        return None
    b = str(bucket).strip().lower()
    if b in ("easy", "medium", "hard", "extreme"):
        return b
    return None


def _resolve_difficulty_bucket(
    stage: str,
    inferred_bucket: str | None,
    external_bucket: str | None,
    config: ExperimentConfig,
) -> tuple[str, str, bool, str | None, str | None]:
    """
    Resolve difficulty bucket via dual-channel policy.

    Returns:
        (chosen_bucket, source, conflict, external_norm, inferred_norm)
    """
    ext = _normalize_bucket_name(external_bucket)
    inf = _normalize_bucket_name(inferred_bucket)

    policy = (
        config.difficulty_policy_train
        if stage == "train"
        else config.difficulty_policy_online
    )
    if policy not in ("external_first", "infer_first"):
        policy = "external_first" if stage == "train" else "infer_first"

    conflict = bool(ext and inf and ext != inf)

    if policy == "external_first":
        if ext is not None:
            return ext, "external", conflict, ext, inf
        if inf is not None:
            return inf, "inferred", conflict, ext, inf
    else:  # infer_first
        if inf is not None:
            return inf, "inferred", conflict, ext, inf
        if ext is not None:
            return ext, "external", conflict, ext, inf

    # Last-resort fallback keeps pipeline robust when both channels are absent.
    return "medium", "fallback", conflict, ext, inf


def _run_calibration_jsonl(
    config: ExperimentConfig,
    manager: PrimitivePerformanceProfileManager,
) -> List[FeedbackRecord]:
    """
    从 calibration JSONL 批量训练 profile，跳过 decomposer 和 evaluator。

    JSONL 每行格式（对齐 profile_data_guide.md）：
    {"tool_id":"forecast/fast_nn","task_type":"time_series","node_type":"forecast",
     "difficulty":"hard","observed_quality":0.52,"observed_cost":0.31}
    """
    import json as _json

    if config.calibration_file is None:
        raise ValueError("calibration_file must be set when input_mode='calibration_jsonl'")

    records: List[FeedbackRecord] = []
    with open(config.calibration_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = _json.loads(line)
            prim, cand = row["tool_id"].split("/", 1)

            # Auto-register unseen primitive/candidate so new task categories
            # can be trained directly from calibration JSONL without manual edits.
            if prim not in manager.list_primitives():
                manager.register_primitive(
                    primitive_name=prim,
                    primitive_type=prim,
                    metadata={"source": "calibration_jsonl_auto_register"},
                )
            prim_profile = manager.get_primitive(prim)
            if cand not in prim_profile.candidates:
                manager.register_candidate(
                    primitive_name=prim,
                    candidate_name=cand,
                    metadata={"source": "calibration_jsonl_auto_register"},
                )

            bucket = row["difficulty"]
            inferred_bucket = row.get("inferred_difficulty_bucket")
            chosen_bucket, diff_source, diff_conflict, ext_b, inf_b = _resolve_difficulty_bucket(
                stage="train",
                inferred_bucket=inferred_bucket,
                external_bucket=bucket,
                config=config,
            )
            obs_q = row["observed_quality"]
            obs_c = row.get("observed_cost", 0.0)
            fb = FeedbackRecord(
                task_id=f"calib_{i:04d}",
                node_id=f"calib_node_{i:04d}",
                primitive_name=prim,
                candidate_name=cand,
                difficulty=BUCKET_TO_FLOAT.get(chosen_bucket, 0.5),
                difficulty_bucket=chosen_bucket,
                predicted_quality=0.0,
                predicted_cost=0.0,
                observed_quality=obs_q,
                observed_cost=obs_c,
                eval_pass=obs_q >= config.pass_threshold,
                episode=i + 1,
                metadata={
                    "difficulty_source": diff_source,
                    "difficulty_conflict": diff_conflict,
                    "difficulty_external_bucket": ext_b,
                    "difficulty_inferred_bucket": inf_b,
                },
            )
            manager.add_feedback(fb)
            records.append(fb)

    print(f"  [calibration_jsonl] Loaded {len(records)} records from {config.calibration_file}")
    return records


# ---------------------------------------------------------------------------
# Main Experiment Runner
# ---------------------------------------------------------------------------

def run_mvp_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Run a single MVP experiment with the given configuration.

    Returns
    -------
    dict with keys:
        - config: ExperimentConfig
        - episode_records: List[EpisodeRecord]
        - curve_snapshots: List[dict]  -- snapshots before each recalibration
        - final_table: list[dict]       -- export_curve_table at end
        - summary: dict                  -- aggregate metrics
        - constraint_metrics: dict        -- constraint satisfaction metrics
    """
    rng = random.Random(config.seed)

    # --- Initialize components ---
    # ProfileStore: external GT data from data/*.jsonl (exp.md Interface 2)
    # Adding a new tool = adding one line to the JSONL, NO code changes
    profile_store = ProfileStore()

    manager = initialize_profile_manager(DEFAULT_GROUND_TRUTH)
    manager._calibration_interval = config.calibration_interval

    # --- Decomposer: LLM-based or keyword-based ---
    if config.use_llm_decomposer:
        from src.decomposer.llm_decomposer import LLMTaskDecomposer
        decomposer = LLMTaskDecomposer(
            api_key=config.llm_api_key,
            fallback=True,
            seed=config.seed,
        )
        print(f"[config] Using LLMTaskDecomposer (Claude Opus 4.6)")
    else:
        decomposer = TaskDecomposer(random_seed=config.seed)
        print(f"[config] Using keyword-based TaskDecomposer")

    # TemplateLibrary: pluggable topology templates (exp.md Interface 3)
    # Adding a new template = one entry in topology_template.py, NO main code changes
    template_library = TemplateLibrary()
    # Load template profiles from JSONL if available
    _template_profile_path = Path(__file__).parent.parent.parent / "data" / "template_profiles.jsonl"
    if _template_profile_path.exists():
        _n_loaded = template_library.load_profiles_from_jsonl(_template_profile_path)
        print(f"[config] Loaded {_n_loaded} template profile records")

    # --- Evaluator: LLM-based or mock ---
    if config.use_llm_evaluator:
        from src.evaluation.claude_evaluator import ClaudeEvaluator
        evaluator = ClaudeEvaluator(
            api_key=config.llm_api_key,
            fallback_to_mock=True,
            pass_threshold=config.pass_threshold,
            seed=config.seed,
        )
        print(f"[config] Using ClaudeEvaluator (Claude Opus 4.6)")
    else:
        evaluator = MockEvaluator(
            ground_truth=DEFAULT_GROUND_TRUTH,  # fallback; ProfileStore takes priority
            noise_std=config.noise_std,
            pass_threshold=config.pass_threshold,
            seed=config.seed,
            profile_store=profile_store,  # inject external GT data
            pareto_mode=config.pareto_mode or config.use_pareto,
        )
        mode_tag = " [pareto_mode]" if (config.pareto_mode or config.use_pareto) else ""
        print(f"[config] Using MockEvaluator{mode_tag}")

    # LLM cost accumulators (separate from executor/evaluator costs)
    total_llm_decomposer_cost = 0.0
    total_llm_evaluator_cost = 0.0

    episode_records: List[EpisodeRecord] = []
    curve_snapshots: List[dict] = []

    # Register a post-calibration hook to record snapshots
    def _snapshot_hook(summary: dict):
        snapshot = {
            "episode_at_calibration": manager.episode_counter,
            "table": manager.export_curve_table(),
            "details": summary["details"],
        }
        curve_snapshots.append(snapshot)

    manager.register_post_calibration_hook(_snapshot_hook)

    # --- Training loop ---
    print(f"\n{'='*60}")
    print(f"  Experiment: {config.name}")
    print(f"  Episodes: {config.n_episodes}  "
          f"Calibration Interval: {config.calibration_interval}")
    print(f"  Acc Target: {config.acc_target}  "
          f"Cost Budget: {config.cost_budget}  "
          f"Latency Budget: {config.latency_budget}")
    if config.enable_constraints:
        print(f"  Constraints: ENABLED  violation_penalty={config.constraint_violation_penalty}  "
              f"constrained_ratio={config.constrained_task_ratio}")
    else:
        print(f"  Constraints: DISABLED")
    print(f"  Input Mode: {config.input_mode}"
          + (f"  Pareto: ON" if config.use_pareto else ""))
    print(f"{'='*60}")

    # === calibration_jsonl 模式：跳过 episode 循环，直接从 JSONL 训练 ===
    if config.input_mode == "calibration_jsonl":
        calib_records = _run_calibration_jsonl(config, manager)
        # 构建最小 summary 并返回
        final_table = manager.export_curve_table()
        summary = {
            "total_executions": len(calib_records),
            "overall_pass_rate": (
                sum(1 for r in calib_records if r.eval_pass) / max(len(calib_records), 1)
            ),
            "unique_tasks": len(calib_records),
            "final_buffer_size": manager.feedback_buffer_size,
            "n_recalibrations": len(curve_snapshots),
            "input_mode": "calibration_jsonl",
        }
        print(f"\n  [calibration_jsonl] Training complete. Final curve table:")
        for row in final_table:
            print(f"    {row}")
        return {
            "config": config,
            "episode_records": [],
            "curve_snapshots": curve_snapshots,
            "final_table": final_table,
            "summary": summary,
            "constraint_metrics": {},
            "four_metrics": {},
            "structural_metrics": {},
            "llm_cost_summary": {},
        }

    for episode in range(1, config.n_episodes + 1):
        task_id = f"task_{episode:03d}"
        llm_decomposer_cost = 0.0

        # === WorkflowGraph execution path ===
        if config.use_workflow_graph:
            # Use the explicit WorkflowGraph layer
            task_desc = (
                config.task_override
                if config.task_override is not None
                else (rng.choice(CONSTRAINED_TASK_BANK)
                      if rng.random() < config.constrained_task_ratio
                      else rng.choice(SAMPLE_TASKS))
            )
            wg_record = run_episode_with_workflow_graph(
                task_description=task_desc,
                config=config,
                manager=manager,
                evaluator=evaluator,
                profile_store=profile_store,
                template_library=template_library,
                decomposer=decomposer,
                rng=rng,
                episode=episode,
            )
            episode_records.append(wg_record)
            # export_curve_table() returns list of snapshot dicts; store first entry
            snapshot_list = manager.export_curve_table()
            curve_snapshots.append(snapshot_list[0] if snapshot_list else {})
            # Skip to next episode
            continue

        # === input_mode 分支 ===
        if config.input_mode == "direct":
            # 直接构造 SubTaskSpec，跳过 decomposer + template selection
            prim_name = rng.choice(list(manager.list_primitives()))
            bucket = rng.choice(["easy", "medium", "hard", "extreme"])
            chosen_bucket, diff_source, diff_conflict, ext_b, inf_b = _resolve_difficulty_bucket(
                stage="train",
                inferred_bucket=None,
                external_bucket=bucket,
                config=config,
            )
            base_stages = [SubTaskSpec(
                sub_task_id="st_0",
                primitive_name=prim_name,
                difficulty=BUCKET_TO_FLOAT.get(chosen_bucket, 0.5),
                difficulty_bucket=chosen_bucket,
                description=f"[direct] {prim_name}/{chosen_bucket}",
                predecessor_ids=[],
                constraints=[],
                input_modality=ModalityType.TEXT,
                metadata={
                    "difficulty_source": diff_source,
                    "difficulty_conflict": diff_conflict,
                    "difficulty_external_bucket": ext_b,
                    "difficulty_inferred_bucket": inf_b,
                },
            )]
            task_type = "direct_training"
            sub_tasks = base_stages
            used_template_id = "direct"
            pattern_name = "linear"
        else:
            # === sample_tasks 模式（原有逻辑）===
            # Select a task: use override if provided, otherwise pick from task bank
            if config.task_override is not None:
                task_item = config.task_override
            elif config.enable_constraints and rng.random() < config.constrained_task_ratio:
                task_item = rng.choice(CONSTRAINED_TASK_BANK)
            else:
                task_item = rng.choice(SAMPLE_TASKS)

            external_bucket = None
            if isinstance(task_item, dict):
                task_desc = task_item.get("task_description") or task_item.get("task") or ""
                external_bucket = task_item.get("difficulty_bucket") or task_item.get("difficulty")
            else:
                task_desc = str(task_item)

            # Decompose into sub-tasks (with constraints and modality info)
            # LLMTaskDecomposer.decompose() returns 3 values (extra: llm_cost_usd)
            decompose_result = decomposer.decompose(
                task_desc,
                extract_constraints=config.enable_constraints,
            )
            if len(decompose_result) == 3:
                base_stages, task_type, llm_decomp_cost = decompose_result
                llm_decomposer_cost = llm_decomp_cost
                total_llm_decomposer_cost += llm_decomposer_cost
            else:
                base_stages, task_type = decompose_result

            inferred_bucket = base_stages[0].difficulty_bucket if base_stages else None
            chosen_bucket, diff_source, diff_conflict, ext_b, inf_b = _resolve_difficulty_bucket(
                stage="online",
                inferred_bucket=inferred_bucket,
                external_bucket=external_bucket,
                config=config,
            )
            for st in base_stages:
                st.difficulty_bucket = chosen_bucket
                st.difficulty = BUCKET_TO_FLOAT.get(chosen_bucket, st.difficulty)
                st.metadata["difficulty_source"] = diff_source
                st.metadata["difficulty_conflict"] = diff_conflict
                st.metadata["difficulty_external_bucket"] = ext_b
                st.metadata["difficulty_inferred_bucket"] = inf_b
            # Map decomposer pattern hint (used as tiebreaker for parallel tasks)
            pattern_name = decomposer._suggest_topology_pattern(task_desc.lower()).value

            # === TemplateLibrary integration (exp.md Interface 3)
            # Structure–configuration joint optimization: template is selected dynamically
            # based on difficulty, constraints, and budget — not a fixed pattern mapping.
            node_type = base_stages[0].primitive_name if base_stages else "unknown"
            candidates_templates = template_library.get_templates_for(node_type, task_type)

            # Collect task-level constraints and difficulty for structural scoring
            task_constraints = base_stages[0].constraints if base_stages else []
            difficulty_bucket_for_template = base_stages[0].difficulty_bucket if base_stages else "medium"

            if config.fixed_template:
                # A/B baseline: always use "direct" template (no structural optimization)
                selected_template = template_library.get_template("direct")
                used_template_id = "direct" if selected_template else "direct"
                scored_templates = [(selected_template, 1.0)] if selected_template else []
            else:
                # Stage 1: Template Pareto frontier selection
                template_frontier = template_library.pareto_frontier(
                    node_type,
                    task_type,
                    difficulty_bucket_for_template,
                )
                if template_frontier:
                    template_info = template_library.select_from_frontier(
                        template_frontier,
                        acc_target=config.acc_target,
                        cost_budget=config.cost_budget,
                        latency_budget=config.latency_budget,
                        alpha=config.q_alpha,
                        beta=config.q_beta,
                        gamma=config.q_gamma,
                    )
                    selected_template = template_info["template"]
                    used_template_id = selected_template.template_id
                else:
                    # Fallback to legacy scoring if no frontier available
                    scored_templates = template_library.score_templates(
                        candidates_templates,
                        difficulty=difficulty_bucket_for_template,
                        remaining_budget=config.cost_budget,
                        constraints=task_constraints,
                    )

                    # Parallel keyword override
                    if pattern_name == "parallel_merge" and scored_templates:
                        for i, (t, s) in enumerate(scored_templates):
                            if t.template_id == "dual_exec_aggregate":
                                scored_templates.insert(0, scored_templates.pop(i))
                                break

                    selected_template = None
                    if scored_templates:
                        selected_template = scored_templates[0][0]
                        used_template_id = selected_template.template_id
                    else:
                        used_template_id = "direct"

            # Instantiate template for each base stage, preserving all decomposer metadata.
            # Template instantiation adds structural nodes (verifier/aggregator) per stage.
            sub_tasks: List[SubTaskSpec] = []
            if selected_template is not None:
                for base_stage in base_stages:
                    instanced = selected_template.instantiate(
                        base_sub_task_id=base_stage.sub_task_id,
                        base_primitive=base_stage.primitive_name,
                        base_difficulty=base_stage.difficulty,
                        difficulty_bucket=base_stage.difficulty_bucket,
                        constraints=base_stage.constraints,
                    )
                    # Preserve decomposer metadata on every instanced node
                    for st in instanced:
                        st.metadata["task_description"] = base_stage.metadata.get(
                            "task_description", ""
                        )
                        st.metadata["matched_keywords"] = base_stage.metadata.get(
                            "matched_keywords", []
                        )
                        st.metadata["constraint_ids"] = [
                            c.constraint_id for c in base_stage.constraints
                        ]
                        st.metadata["task_id"] = task_id
                        st.metadata["task_type"] = task_type
                        st.metadata["input_modality"] = base_stage.input_modality.value
                        st.metadata["intermediate_modality"] = (
                            base_stage.intermediate_modality.value
                            if base_stage.intermediate_modality else None
                        )
                        # Promotes template-based structure over decomposer's built-in topology
                        if st.metadata.get("template_id") is None:
                            st.metadata["template_id"] = selected_template.template_id
                    sub_tasks.extend(instanced)
            else:
                # No template matched — fall back to decomposer's built-in topology
                for base_stage in base_stages:
                    base_stage.metadata["task_id"] = task_id
                    base_stage.metadata["task_type"] = task_type
                sub_tasks = base_stages

        # === Mandatory node enforcement ===
        # If any task has a mandatory_node constraint, verify the topology satisfies it.
        # If violated, inject the required primitive as a separate sub-task.
        mandatory_violations = _check_mandatory_node_violations(sub_tasks, manager)
        if mandatory_violations:
            # Inject missing primitives into the topology
            for st in sub_tasks:
                for c in st.get_active_constraints():
                    if isinstance(c, MandatoryNodeConstraint) and c.required_primitive:
                        prim_needed = c.required_primitive
                        existing_prims = {s.primitive_name for s in sub_tasks}
                        if prim_needed not in existing_prims:
                            # Insert as first task in the chain
                            new_st = SubTaskSpec(
                                sub_task_id=f"st_inject_{prim_needed}",
                                primitive_name=prim_needed,
                                difficulty=st.difficulty,
                                difficulty_bucket=st.difficulty_bucket,
                                description=f"[INJECTED mandatory] {prim_needed}",
                                predecessor_ids=[],
                                constraints=[],  # injected nodes don't inherit constraints
                                input_modality=st.input_modality,
                            )
                            sub_tasks.insert(0, new_st)
                            break

        # === Topological sort for DAG execution ===
        # Handles both linear chains (most common) and parallel-merge graphs
        execution_order = _topo_sort(sub_tasks)

        # Track accumulated cost within episode for remaining_budget in repair
        accumulated_cost: float = 0.0

        recalibrated_this_episode = False

        for st in execution_order:
            recalibrated_this_subtask = False  # initialized here; set True after add_feedback
            # Orchestration: select candidate via Pareto frontier
            frontier = manager.pareto_frontier(st.primitive_name, st.difficulty_bucket)
            if frontier:
                # Use constraint-based selection from Pareto frontier
                pareto_selected = manager.select_from_frontier(
                    frontier,
                    acc_target=config.acc_target,
                    cost_budget=config.cost_budget,
                    latency_budget=config.latency_budget,
                    alpha=config.q_alpha,
                    beta=config.q_beta,
                )
                # Put selected first, then rest of frontier for fallback
                ranks = [pareto_selected] + [c for c in frontier if c is not pareto_selected]
            else:
                ranks = manager.predict_all(
                    primitive_name=st.primitive_name,
                    difficulty=st.difficulty_bucket,
                )

            if not ranks:
                ranks = manager.predict_all(
                    st.primitive_name, st.difficulty_bucket
                )
            if not ranks:
                fallback_prim = manager.list_primitives()[0]
                ranks = manager.predict_all(fallback_prim, st.difficulty_bucket)
            if not ranks:
                print(
                    f"  [WARN] No candidates found for "
                    f"{st.primitive_name} [{st.difficulty_bucket}], skipping."
                )
                continue

            # === Apply hard constraints ===
            if config.enable_constraints and st.constraints:
                ranks = _apply_constraints(ranks, st, config.constraint_violation_penalty)

            # === Human-in-the-loop pause ===
            human_approved = True
            if config.enable_constraints and st.has_human_approval_required():
                if config.enable_human_hitl:
                    human_approved = _simulate_human_approval(st, config)
                    if not human_approved:
                        # Downgrade to the safest candidate (lowest cost, highest quality)
                        # by finding the candidate with best quality among lowest-cost options
                        safe_candidates = sorted(
                            ranks,
                            key=lambda x: (x["pred_cost"], -x["pred_acc"])
                        )
                        if safe_candidates:
                            safe_cand = safe_candidates[0]["candidate_name"]
                        else:
                            # Should not happen (ranks should never be empty), but guard
                            safe_cand = ranks[0]["candidate_name"] if ranks else None
                        if safe_cand is None:
                            continue
                        # Log but still execute with fallback
                        print(
                            f"  [HITL] Human rejected top candidate for "
                            f"{st.primitive_name}, downgrading to {safe_cand}"
                        )
                        # Update ranks to reflect fallback
                        ranks = [safe_candidates[0]] if safe_candidates else ranks

            selected = ranks[0] if ranks else None
            if selected is None:
                print(f"  [WARN] No valid candidate for {st.primitive_name}, skipping.")
                continue
            cand_name = selected["candidate_name"]

            # === Executor × Evaluator joint selection ===
            # Evaluator is selected based on task difficulty (hard=large_eval, easy=rule_eval)
            # The evaluator's precision affects eval_pass accuracy and learning signal
            evaluator_name = st.evaluator_name or _select_evaluator(st, cand_name, config)

            # Evaluation (with constraint context + evaluator selection)
            eval_result = evaluator.evaluate(
                candidate_name=cand_name,
                primitive_name=st.primitive_name,
                difficulty_bucket=st.difficulty_bucket,
                task_id=task_id,
                node_id=st.sub_task_id,
                task_spec=st,  # pass SubTaskSpec for constraint validation
                evaluator_name=evaluator_name,
                node_type=st.primitive_name,  # node type for rubric-based evaluation
            )

            # Update evaluator profile
            profile_store.update_evaluator_profile(
                evaluator_id=eval_result.evaluator_name,
                difficulty=st.difficulty_bucket,
                observed_pass=eval_result.eval_pass,
                true_pass=(eval_result.true_quality or 0) >= evaluator.pass_threshold,
                evaluator_latency=eval_result.evaluator_latency,
                evaluator_cost=eval_result.evaluator_cost,
            )

            # === Local Repair: upgrade on failure ===
            # Track remaining budget for repair decision
            if config.cost_budget is not None:
                remaining_budget = config.cost_budget - accumulated_cost
            else:
                remaining_budget = None

            repair_count = 0
            repair_action = "none"
            template_upgraded_from = "none"
            template_upgraded_to = "none"
            repaired_cand_name = cand_name
            repair_deltaG: dict | None = None
            repair_delta_nodes = 0
            repair_delta_edges = 0

            while config.enable_repair and repair_count < MAX_REPAIR_ATTEMPTS:
                if not _should_repair(eval_result, repair_count):
                    break

                new_result, deltaG, repair_tried = _repair_subgraph(
                    st,
                    repaired_cand_name,
                    eval_result,
                    manager,
                    evaluator,
                    config,
                    rng,
                    profile_store,
                    template_library=template_library,
                    current_template_id=used_template_id,
                    use_pareto=config.use_pareto,
                    remaining_budget=remaining_budget,
                )
                if new_result is not None and new_result.eval_pass:
                    # Upgrade succeeded
                    eval_result = new_result
                    repair_deltaG = deltaG
                    if deltaG is not None:
                        repair_action = deltaG.get("action", "none")
                        repair_delta_nodes = deltaG.get("delta_nodes", 0)
                        repair_delta_edges = deltaG.get("delta_edges", 0)
                        # Update candidate name if changed
                        if deltaG.get("candidate_changed"):
                            repaired_cand_name = deltaG.get("to_candidate", repaired_cand_name)
                        # Update template if changed
                        if deltaG.get("template_changed"):
                            template_upgraded_from = used_template_id
                            template_upgraded_to = deltaG.get("to_template", "unknown")
                            used_template_id = template_upgraded_to
                        # Update remaining budget after repair cost
                        if remaining_budget is not None and new_result.true_cost:
                            remaining_budget -= (new_result.true_cost or 0)
                    break
                # Upgrade didn't help — try next level
                repair_count += 1
                if deltaG and deltaG.get("candidate_changed"):
                    repaired_cand_name = deltaG.get("to_candidate", repaired_cand_name)

            # LLM evaluator cost: only ClaudeEvaluator uses real LLM API calls.
            # Cost is retrieved via getattr (both types share this field name at runtime).
            # MockEvaluator → no LLM → llm_evaluator_cost = 0.0.
            llm_eval_cost = getattr(evaluator, "last_call_cost", 0.0) or 0.0
            if config.use_llm_evaluator:
                total_llm_evaluator_cost += llm_eval_cost
            else:
                llm_eval_cost = 0.0

            # Record
            record = EpisodeRecord(
                episode=episode,
                task_id=task_id,
                sub_task_id=st.sub_task_id,
                primitive_name=st.primitive_name,
                difficulty_bucket=st.difficulty_bucket,
                difficulty=st.difficulty,
                selected_candidate=cand_name,
                predicted_acc=selected["pred_acc"],
                predicted_cost=selected["pred_cost"],
                true_acc=eval_result.true_quality or 0.0,
                true_cost=eval_result.true_cost or 0.0,
                observed_acc=eval_result.observed_quality,
                observed_cost=eval_result.observed_cost,
                eval_pass=eval_result.eval_pass,
                failure_type=eval_result.failure_type,
                recalibrated=False,  # set correctly after add_feedback below
                source=selected["source"],
                difficulty_source=st.metadata.get("difficulty_source", "inferred"),
                difficulty_conflict=bool(st.metadata.get("difficulty_conflict", False)),
                difficulty_external_bucket=st.metadata.get("difficulty_external_bucket"),
                difficulty_inferred_bucket=st.metadata.get("difficulty_inferred_bucket"),
                # === 约束与多模态字段 ===
                constraint_violations=eval_result.constraint_violations,
                violation_count=len(eval_result.constraint_violations),
                human_approved=eval_result.human_approved,
                execution_duration=eval_result.execution_duration,
                input_modality=st.input_modality.value,
                intermediate_modality=(
                    st.intermediate_modality.value
                    if st.intermediate_modality else None
                ),
                evaluator_name=eval_result.evaluator_name,
                # === 新增：结构化 evaluator 字段 ===
                evaluator_id=eval_result.evaluator_name,
                error_type=eval_result.error_type,
                confidence=eval_result.confidence,
                evaluator_latency=eval_result.evaluator_latency,
                evaluator_cost=eval_result.evaluator_cost,
                quality_score=eval_result.quality_score,
                node_type=st.primitive_name,
                # === exp.md 验收字段 ===
                task_type=task_type,
                template_id=used_template_id,
                repair_action=repair_action,
                template_upgraded_from=template_upgraded_from,
                template_upgraded_to=template_upgraded_to,
                repair_deltaG=repair_deltaG,
                repair_delta_nodes=repair_delta_nodes,
                repair_delta_edges=repair_delta_edges,
                # === LLM 组件消耗（独立统计）===
                llm_decomposer_cost=llm_decomposer_cost,
                llm_evaluator_cost=llm_eval_cost,
            )
            episode_records.append(record)

            # Track accumulated cost for remaining_budget in repair decisions
            accumulated_cost += (eval_result.true_cost or 0.0)

            # Check if recalibration will be triggered by this feedback
            # (compare snapshot count before and after add_feedback)
            snapshots_before = len(curve_snapshots)

            # Convert EpisodeRecord -> FeedbackRecord for ProfileManager
            fb_record = FeedbackRecord(
                task_id=record.task_id,
                node_id=record.sub_task_id,
                primitive_name=record.primitive_name,
                candidate_name=record.selected_candidate,
                difficulty=record.difficulty,
                difficulty_bucket=record.difficulty_bucket,
                predicted_quality=record.predicted_acc,
                predicted_cost=record.predicted_cost,
                observed_quality=record.observed_acc,
                observed_cost=record.observed_cost,
                eval_pass=record.eval_pass,
                failure_type=record.failure_type,
                episode=record.episode,
                # === 约束与多模态字段 ===
                constraint_violations=record.constraint_violations,
                execution_duration=record.execution_duration,
                human_approved=record.human_approved,
                input_modality=record.input_modality,
                intermediate_modality=record.intermediate_modality,
                violation_count=record.violation_count,
                evaluator_name=record.evaluator_name,
                # === 新增 evaluator 结构化字段 ===
                evaluator_id=eval_result.evaluator_name,
                error_type=eval_result.error_type,
                confidence=eval_result.confidence,
                evaluator_latency=eval_result.evaluator_latency,
                evaluator_cost=eval_result.evaluator_cost,
                # === 上下文信息 ===
                task_type=task_type,
                node_type=st.primitive_name,
                template_id=used_template_id,
                metadata={
                    "difficulty_source": record.difficulty_source,
                    "difficulty_conflict": record.difficulty_conflict,
                    "difficulty_external_bucket": record.difficulty_external_bucket,
                    "difficulty_inferred_bucket": record.difficulty_inferred_bucket,
                },
            )
            manager.add_feedback(fb_record)

            # Template-level feedback: aggregate (quality, cost, latency) for template profile
            # latency: use evaluator's execution_duration as proxy for node latency
            template_library.add_feedback(
                template_id=used_template_id,
                difficulty_bucket=st.difficulty_bucket,
                observed_quality=1.0 if eval_result.eval_pass else 0.0,
                observed_cost=eval_result.observed_cost + eval_result.evaluator_cost,
                observed_latency=eval_result.execution_duration or 0.0,
            )

            # Detect if a recalibration was triggered during this feedback
            snapshots_after = len(curve_snapshots)
            recalibrated_this_subtask = snapshots_after > snapshots_before
            record.recalibrated = recalibrated_this_subtask  # mutate post-creation
            if recalibrated_this_subtask:
                recalibrated_this_episode = True

        # Log episode summary
        ep_records = [r for r in episode_records if r.episode == episode]
        n_pass = sum(1 for r in ep_records if r.eval_pass)
        pass_rate = n_pass / len(ep_records) if ep_records else 0.0
        n_violations = sum(r.violation_count for r in ep_records)
        # Show HITL badge only when human_in_the_loop constraint was triggered
        hitl_violated = any(
            "human_in_the_loop" in str(r.constraint_violations)
            for r in ep_records
        )
        hitl_rejected = any(r.human_approved == False for r in ep_records)
        hitl_str = ""
        if hitl_violated:
            status = "REJ" if hitl_rejected else "HITL"
            hitl_str = f"  [{status}]"
        print(
            f"  Episode {episode:3d} | tasks={len(sub_tasks)} "
            f"| pass={pass_rate:.0%} "
            f"| viol={n_violations} "
            f"| recal={recalibrated_this_episode} "
            f"| buffer={manager.feedback_buffer_size}"
            f"{hitl_str}"
        )

    # --- Aggregate summary ---
    all_pass = sum(1 for r in episode_records if r.eval_pass)
    total = len(episode_records)
    summary = {
        "total_executions": total,
        "overall_pass_rate": round(all_pass / total, 4) if total > 0 else 0.0,
        "unique_tasks": config.n_episodes,
        "final_buffer_size": manager.feedback_buffer_size,
        "n_recalibrations": len(curve_snapshots),
        "evaluator_pass_rate": evaluator.pass_rate,
    }

    # --- Constraint metrics ---
    constraint_metrics = _report_constraint_metrics(episode_records, config)

    # --- Constraint convergence analysis ---
    convergence_analysis = _analyze_constraint_convergence(
        episode_records, curve_snapshots
    )
    _print_constraint_analysis(convergence_analysis)

    print(f"\n  [Summary]")
    print(f"    Total executions: {total}")
    print(f"    Overall pass rate: {summary['overall_pass_rate']:.1%}")
    print(f"    Recalibrations: {summary['n_recalibrations']}")
    print(f"    Evaluator pass rate: {summary['evaluator_pass_rate']:.1%}")

    # === 4 Main Metrics (supports direct paper table output) ===
    _print_four_metrics(episode_records, constraint_metrics)

    # === Structural Metrics (template selection distribution) ===
    _print_structural_metrics(episode_records)

    # === LLM 组件成本汇总（独立于 executor/evaluator profile 成本）===
    if config.use_llm_decomposer or config.use_llm_evaluator:
        print(f"\n  [LLM Component Cost Breakdown]")
        total_episodes = config.n_episodes
        avg_decomp = total_llm_decomposer_cost / max(total_episodes, 1)
        avg_eval = total_llm_evaluator_cost / max(len(episode_records), 1)
        print(f"    LLM Decomposer cost: ${total_llm_decomposer_cost:.6f} total "
              f"(${avg_decomp:.6f}/task)")
        print(f"    LLM Evaluator  cost: ${total_llm_evaluator_cost:.6f} total "
              f"(${avg_eval:.6f}/task)")
        print(f"    LLM Total       cost: ${total_llm_decomposer_cost + total_llm_evaluator_cost:.6f}")
        print(f"    ** NOTE: LLM costs are INDEPENDENT of executor/evaluator profile costs.")
        print(f"    ** Executor/evaluator costs (in profile EMA) do NOT include LLM API calls.")

        # Attach LLM cost summary to snapshots for later analysis
        snapshot_llm_summary = {
            "llm_decomposer_cost_total": round(total_llm_decomposer_cost, 6),
            "llm_evaluator_cost_total": round(total_llm_evaluator_cost, 6),
            "llm_cost_per_task_avg": round(
                (total_llm_decomposer_cost + total_llm_evaluator_cost) / max(total_episodes, 1), 6
            ),
            "use_llm_decomposer": config.use_llm_decomposer,
            "use_llm_evaluator": config.use_llm_evaluator,
        }
    else:
        snapshot_llm_summary = {}

    # --- Final export ---
    final_table = manager.export_curve_table()

    # --- Save CSV ---
    csv_path = config.output_dir / f"{config.name}.csv"
    _save_csv(episode_records, csv_path)

    # --- Save curve snapshots ---
    snapshot_path = config.output_dir / f"{config.name}_snapshots.json"
    import json as _json
    with snapshot_path.open("w", encoding="utf-8") as f:
        _json.dump({
            "config": asdict(config),
            "curve_snapshots": curve_snapshots,
            "summary": summary,
            "constraint_metrics": constraint_metrics,
            "structural_metrics": _build_structural_metrics(episode_records),
            "llm_cost_summary": snapshot_llm_summary,
        }, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n  Saved CSV: {csv_path}")
    print(f"  Saved snapshots: {snapshot_path}")

    # === 4 Main Metrics dict (for programmatic consumption) ===
    four_metrics = _build_four_metrics(episode_records, constraint_metrics)

    # === Structural metrics ===
    structural_metrics = _build_structural_metrics(episode_records)

    # === Save all 3 profile types back to JSONL ===
    profile_store.save(config.output_dir)

    return {
        "config": config,
        "episode_records": episode_records,
        "curve_snapshots": curve_snapshots,
        "final_table": final_table,
        "summary": summary,
        "constraint_metrics": constraint_metrics,
        "four_metrics": four_metrics,
        "structural_metrics": structural_metrics,
        "llm_cost_summary": snapshot_llm_summary,
    }


def _save_csv(records: List[EpisodeRecord], path: Path) -> None:
    """Save episode records to CSV."""
    if not records:
        return
    fieldnames = list(asdict(records[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            row = asdict(r)
            # Serialize list fields as JSON strings for CSV compatibility
            row["constraint_violations"] = str(row["constraint_violations"])
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Ablation Experiment: Different Calibration Intervals
# ---------------------------------------------------------------------------

def run_ablation_calibration(
    intervals: List[int | None] = None,
    n_episodes: int = 20,
    seed: int = 42,
) -> Dict[int | None, Dict[str, Any]]:
    """
    Run experiments with different calibration_interval values (including None = no calibration)
    and compare: MAE of predictions, selection changes, and pass rates.
    """
    if intervals is None:
        intervals = [None, 3, 5, 10, 20]

    results: Dict[int | None, Dict[str, Any]] = {}

    print(f"\n{'='*80}")
    print("  Ablation: Calibration Interval Comparison")
    print(f"{'='*80}")

    for interval in intervals:
        k_label = "no_cal" if interval is None else f"k{interval}"
        config = ExperimentConfig(
            name=f"ablation_{k_label}",
            n_episodes=n_episodes,
            calibration_interval=interval,
            seed=seed,
            enable_constraints=False,  # baseline: no constraints
        )
        result = run_mvp_experiment(config)
        results[interval] = result

    # Print comparison table
    print(f"\n{'='*80}")
    print("  Ablation Summary")
    print(f"{'='*80}")
    print(
        f"{'K':>8}  {'N_recal':>8}  {'MAE_before':>10}  "
        f"{'MAE_after':>10}  {'Pass_rate':>10}  {'Buffer':>8}"
    )
    print("-" * 75)
    for interval, result in sorted(results.items(), key=lambda x: (x[0] is None, x[0] or 999)):
        s = result["summary"]
        k_str = "None" if interval is None else str(interval)

        mae_final, (mae_before, mae_after) = _compute_selection_metrics(
            result["episode_records"], result["curve_snapshots"]
        )
        mae_str = f"{mae_final:.4f}" if mae_final is not None else "N/A"
        mb_str = f"{mae_before:.4f}" if mae_before is not None else "N/A"
        ma_str = f"{mae_after:.4f}" if mae_after is not None else "N/A"

        print(
            f"{k_str:>8}  "
            f"{s['n_recalibrations']:>8}  "
            f"{mb_str:>10}  "
            f"{ma_str:>10}  "
            f"{s['overall_pass_rate']:>10.1%}  "
            f"{s['final_buffer_size']:>8}"
        )


def _compute_selection_metrics(
    episode_records: List[EpisodeRecord],
    curve_snapshots: list,
) -> Tuple[float | None, Tuple[float | None, float | None]]:
    """
    Compute MAE metrics for an experiment run.

    Uses curve_snapshots to determine the first recalibration episode,
    then computes MAE before and after that point.
    """
    if not episode_records:
        return None, (None, None)

    mae = sum(abs(r.predicted_acc - r.true_acc) for r in episode_records)
    mae = round(mae / len(episode_records), 4) if episode_records else None

    by_ep: Dict[int, List[EpisodeRecord]] = {}
    for r in episode_records:
        by_ep.setdefault(r.episode, []).append(r)

    sorted_eps = sorted(by_ep.keys())

    # Determine first recalibration episode from snapshots (not from record.recalibrated)
    if curve_snapshots:
        first_recal_ep = curve_snapshots[0].get("episode_at_calibration")
        if first_recal_ep is None:
            first_recal_ep = None
    else:
        first_recal_ep = None

    if first_recal_ep is None:
        mae_before, mae_after = None, None
    else:
        err_before = [
            abs(r.predicted_acc - r.true_acc)
            for ep in sorted_eps for r in by_ep[ep]
            if ep < first_recal_ep
        ]
        err_after = [
            abs(r.predicted_acc - r.true_acc)
            for ep in sorted_eps for r in by_ep[ep]
            if ep >= first_recal_ep
        ]
        mae_before = round(sum(err_before) / len(err_before), 4) if err_before else None
        mae_after = round(sum(err_after) / len(err_after), 4) if err_after else None

    return mae, (mae_before, mae_after)


# ---------------------------------------------------------------------------
# A/B Comparison Experiments
# ---------------------------------------------------------------------------

def run_main_comparisons(
    n_episodes: int = 50,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Run 4 focused A/B comparisons for the main experiment:
    1. fixed_template (direct only) vs dynamic template selection
    2. no_repair vs repair enabled
    3. no_ema (calibration_interval=999) vs ema (calibration_interval=5)
    4. linear_score vs pareto

    Each pair shares the same seed and episodes for fair comparison.
    Results saved to output_dir (default: outputs/comparison_<timestamp>/).
    Returns dict of {tag: result}.
    """
    if output_dir is None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"./outputs/comparison_{ts}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    comparisons = [
        ("A1_fixed_template", {"fixed_template": True, "enable_repair": True}),
        ("A2_dynamic_template", {"fixed_template": False, "enable_repair": True}),
        ("B1_no_repair", {"fixed_template": False, "enable_repair": False}),
        ("B2_with_repair", {"fixed_template": False, "enable_repair": True}),
        ("C1_no_recalibration", {"fixed_template": False, "enable_repair": True, "calibration_interval": 999}),
        ("C2_frequent_recalibration", {"fixed_template": False, "enable_repair": True, "calibration_interval": 5}),
        ("D1_no_pareto", {"fixed_template": False, "enable_repair": True, "use_pareto": False}),
        ("D2_pareto", {"fixed_template": False, "enable_repair": True, "use_pareto": True}),
    ]

    results: Dict[str, Dict[str, Any]] = {}

    print(f"\n{'='*80}")
    print("  MAIN A/B COMPARISONS (4 pairs x 2 conditions)")
    print(f"  Episodes: {n_episodes}  Seed: {seed}")
    print(f"  Output: {output_dir}")
    print(f"{'='*80}")

    for tag, overrides in comparisons:
        config = ExperimentConfig(
            name=tag,
            n_episodes=n_episodes,
            calibration_interval=overrides.get("calibration_interval", 5),
            seed=seed,
            output_dir=output_dir,
            enable_constraints=True,
            fixed_template=overrides.get("fixed_template", False),
            enable_repair=overrides.get("enable_repair", True),
            use_pareto=overrides.get("use_pareto", False),
        )
        print(f"\n{'~'*60}")
        print(f"  Running: {tag}")
        print(f"  Config: {overrides}")
        print(f"{'~'*60}")
        result = run_mvp_experiment(config)
        results[tag] = result

    # Print comparison summary table
    print(f"\n{'='*80}")
    print("  A/B COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Tag':25s}  {'PassRate':>9}  {'Latency':>9}  {'Cost':>9}  {'ViolRate':>9}")
    print(f"  {'-'*67}")

    for tag, result in results.items():
        fm = result.get("four_metrics", {})
        pr = fm.get("quality", {}).get("pass_rate", 0)
        lat = fm.get("latency", {}).get("mean_total_latency", 0) or 0
        cost = fm.get("tokens", {}).get("mean_total_cost", 0) or 0
        vr = fm.get("violations", {}).get("violation_rate", 0)
        print(f"  {tag:25s}  {pr:>8.1%}  {lat:>8.3f}s  {cost:>8.3f}  {vr:>8.1%}")

    print(f"{'='*80}")
    return results


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MVP Experiment Runner with Hard Constraints & Multimodal Support"
    )
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of episodes")
    parser.add_argument("--calibration_interval", "--k", type=int, default=5,
                        help="Calibration interval (K)")
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="EMA alpha")
    parser.add_argument("--acc_target", type=float, default=None,
                        help="ACC target constraint (hard filter)")
    parser.add_argument("--cost_budget", type=float, default=None,
                        help="Cost budget constraint (hard filter)")
    parser.add_argument("--latency_budget", type=float, default=None,
                        help="Latency budget constraint (hard filter)")
    parser.add_argument("--noise_std", type=float, default=0.05,
                        help="Evaluator noise std")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory")
    parser.add_argument("--ablation", action="store_true",
                        help="Run ablation: compare K in [3,5,10,20]")
    parser.add_argument("--comparisons", action="store_true",
                        help="Run 3 A/B comparisons (fixed vs dynamic, no repair vs repair, no EMA vs EMA)")
    parser.add_argument("--single", action="store_true",
                        help="Run a single experiment (default if no --ablation/--comparisons)")
    parser.add_argument("--name", type=str, default="mvp_run",
                        help="Experiment name")
    parser.add_argument("--task", type=str, default=None,
                        help="Override task description (forces single-task mode)")
    # === 新增约束相关 CLI 参数 ===
    parser.add_argument("--enable_constraints", dest="enable_constraints",
                        action="store_true", default=True,
                        help="Enable hard constraints (default: True)")
    parser.add_argument("--no_constraints", dest="enable_constraints",
                        action="store_false",
                        help="Disable hard constraints (baseline)")
    parser.add_argument("--constraint_violation_penalty", type=float,
                        default=-0.5,
                        help="Score penalty for constraint violations (default: -0.5)")
    parser.add_argument("--constrained_task_ratio", type=float,
                        default=0.4,
                        help="Fraction of tasks that are constrained (default: 0.4)")
    parser.add_argument("--no_hitl", dest="enable_human_hitl",
                        action="store_false", default=True,
                        help="Disable human-in-the-loop simulation")
    # === 训练输入模式 ===
    parser.add_argument("--input_mode", type=str, default="sample_tasks",
                        choices=["sample_tasks", "direct", "calibration_jsonl"],
                        help="Training input mode")
    parser.add_argument("--calibration_file", type=str, default=None,
                        help="Path to calibration JSONL file (for input_mode=calibration_jsonl)")
    parser.add_argument("--difficulty_policy_train", type=str, default="external_first",
                        choices=["external_first", "infer_first"],
                        help="Difficulty policy in training stage (default: external_first)")
    parser.add_argument("--difficulty_policy_online", type=str, default="infer_first",
                        choices=["external_first", "infer_first"],
                        help="Difficulty policy in online stage (default: infer_first)")
    parser.add_argument("--use_pareto", action="store_true", default=False,
                        help="Use Pareto frontier filtering in candidate selection")
    parser.add_argument("--pareto_mode", action="store_true", default=False,
                        help="Simplified evaluator for Pareto (skip rubric, direct quality→pass/fail)")
    parser.add_argument("--use_workflow_graph", action="store_true", default=False,
                        help="Use WorkflowGraph execution layer instead of SubTaskSpec loop")
    parser.add_argument("--mode", type=str, default="single",
                        choices=["single", "train_test", "ablation", "comparisons"],
                        help="Experiment mode: single (default) | train_test (80/20 split for Q(G;X) validation) | ablation | comparisons")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Train/test split ratio (default 0.8)")
    # Q(G;X) utility weights: Q = αS − βC − γL (Section 4.4 method definition)
    parser.add_argument("--q_alpha", type=float, default=0.6,
                        help="Weight for quality S in Q(G;X) (default: 0.6)")
    parser.add_argument("--q_beta", type=float, default=0.2,
                        help="Weight for cost C in Q(G;X) (default: 0.2)")
    parser.add_argument("--q_gamma", type=float, default=0.2,
                        help="Weight for latency L in Q(G;X) (default: 0.2)")

    args = parser.parse_args()

    if args.comparisons:
        # For comparison mode, create a timestamped subfolder under output_dir
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        comp_dir = Path(args.output_dir) / f"comparison_{ts}"
        run_main_comparisons(
            n_episodes=args.episodes,
            seed=args.seed,
            output_dir=comp_dir,
        )
    elif args.ablation:
        run_ablation_calibration(
            intervals=[3, 5, 10, 20],
            n_episodes=args.episodes,
            seed=args.seed,
        )
    else:
        config = ExperimentConfig(
            name=args.name,
            n_episodes=args.episodes,
            calibration_interval=args.calibration_interval,
            acc_target=args.acc_target,
            cost_budget=args.cost_budget,
            latency_budget=args.latency_budget,
            noise_std=args.noise_std,
            seed=args.seed,
            output_dir=Path(args.output_dir),
            enable_constraints=args.enable_constraints,
            constraint_violation_penalty=args.constraint_violation_penalty,
            constrained_task_ratio=args.constrained_task_ratio,
            enable_human_hitl=args.enable_human_hitl,
            task_override=args.task,
            input_mode=args.input_mode,
            calibration_file=Path(args.calibration_file) if args.calibration_file else None,
            difficulty_policy_train=args.difficulty_policy_train,
            difficulty_policy_online=args.difficulty_policy_online,
            use_pareto=args.use_pareto,
            pareto_mode=args.pareto_mode,
            use_workflow_graph=args.use_workflow_graph,
            q_alpha=args.q_alpha,
            q_beta=args.q_beta,
            q_gamma=args.q_gamma,
        )

        if args.mode == "train_test":
            run_train_test_experiment(config, train_ratio=args.train_ratio)
        else:
            run_mvp_experiment(config)


# ---------------------------------------------------------------------------
# WorkflowGraph Integration Layer
# ---------------------------------------------------------------------------
# Provides an alternative execution path using the explicit WorkflowGraph
# representation (G = (V, E, τ, φ) from the method definition).
# Disabled by default; enable with config.use_workflow_graph = True.
# ---------------------------------------------------------------------------

def run_episode_with_workflow_graph(
    task_description: str,
    config: ExperimentConfig,
    manager: PrimitivePerformanceProfileManager,
    evaluator: MockEvaluator,
    profile_store: ProfileStore,
    template_library: TemplateLibrary,
    decomposer: TaskDecomposer,
    rng: random.Random,
    episode: int,
) -> EpisodeRecord:
    """
    Execute a single episode using the WorkflowGraph layer.

    This is an alternative to the main episode loop. It builds an explicit
    WorkflowGraph from the task description, executes it, and converts the result
    back to an EpisodeRecord for compatibility with the rest of the experiment.

    Parameters
    ----------
    task_description : str
        Raw task description text.
    config : ExperimentConfig
        Experiment configuration.
    manager : PrimitivePerformanceProfileManager
        For candidate selection within execution nodes.
    evaluator : MockEvaluator
        For node evaluation.
    profile_store : ProfileStore
        For profile updates after execution.
    template_library : TemplateLibrary
        For template-level Pareto frontier.
    decomposer : TaskDecomposer
        For task decomposition.
    rng : random.Random
        Random number generator.
    episode : int
        Episode number (for logging).

    Returns
    -------
    EpisodeRecord
        Execution result in the existing experiment record format.
    """
    from src.workflow import WorkflowBuilder, WorkflowExecutor
    from src.workflow.workflow_graph import NodeStatus
    from src.decomposer.task_decomposer import difficulty_to_bucket

    # Step 1: Decompose task
    sub_tasks, task_type = decomposer.decompose(
        task_description,
        extract_constraints=config.enable_constraints,
    )

    if not sub_tasks:
        return _make_fallback_record(episode, task_description)

    st = sub_tasks[0]  # Primary sub-task
    difficulty = st.difficulty
    bucket = st.difficulty_bucket
    node_type = st.primitive_name

    # Step 2: Select template using template library (Pareto frontier)
    if config.fixed_template or not config.use_pareto:
        template_id = "direct"
    else:
        frontier = template_library.pareto_frontier(node_type, task_type, bucket)
        if frontier:
            selected = template_library.select_from_frontier(
                frontier,
                acc_target=config.acc_target,
                cost_budget=config.cost_budget,
                latency_budget=config.latency_budget,
                alpha=config.q_alpha,
                beta=config.q_beta,
                gamma=config.q_gamma,
            )
            template_id = selected.get("template_id", "direct")
        else:
            template_id = "direct"

    # Step 3: Build WorkflowGraph
    builder = WorkflowBuilder(
        primitive_name=node_type,
        difficulty=difficulty,
        difficulty_bucket=bucket,
        task_type=task_type,
        scenario=config.name,
        profile_store=profile_store,
        graph_id_prefix=f"ep{episode}",
        constraints=st.constraints,
    )
    graph = builder.from_template_id(template_id)
    graph.metadata["task_id"] = f"ep{episode}"
    graph.metadata["task_description"] = task_description
    graph.metadata["template_id"] = template_id

    # Step 4: Execute WorkflowGraph
    executor = WorkflowExecutor(
        max_repair_attempts=MAX_REPAIR_ATTEMPTS if config.enable_repair else 0,
        debug=False,
    )
    wf_result = executor.execute(
        graph=graph,
        evaluator=evaluator,
        profile_manager=manager,
        profile_store=profile_store,
        config=config,
        rng=rng,
    )

    # Step 5: Update profile manager with execution result
    for nr in wf_result.node_results:
        if nr.status == NodeStatus.DONE and nr.executor_id and "/" in nr.executor_id:
            parts = nr.executor_id.split("/")
            prim = parts[0]
            cand = parts[1]
            # Get difficulty from the node's metadata in the graph
            wf_node = graph.nodes.get(nr.node_id)
            node_diff = wf_node.metadata.get("difficulty", 0.5) if wf_node else 0.5
            manager.add_feedback({
                "task_id": f"ep{episode}",
                "node_id": nr.node_id,
                "primitive_name": prim,
                "candidate_name": cand,
                "difficulty": node_diff,
                "difficulty_bucket": bucket,
                "predicted_quality": nr.predicted_quality or 0.5,
                "predicted_cost": nr.predicted_cost or 1.0,
                "observed_quality": nr.observed_quality,
                "observed_cost": nr.observed_cost,
                "eval_pass": nr.eval_pass,
                "failure_type": nr.error_type,
            })

    # Step 5b: Update workflow-level profile (acc/cost/latency/violation/repair)
    if profile_store is not None:
        profile_store.update_workflow_profile(
            template_id=graph.metadata.get("template_id", "unknown"),
            scenario=config.name,
            task_type=task_type,
            overall_pass=wf_result.overall_pass,
            total_cost=wf_result.total_cost,
            total_latency=wf_result.total_latency,
            violation_count=wf_result.violation_count,
            node_count=len(wf_result.node_results),
            repair_count=wf_result.repair_count,
        )

    # Step 6: Convert to EpisodeRecord (backward compatibility)
    return wf_result.to_episode_record(episode=episode, task_id=f"ep{episode}")


def _make_fallback_record(episode: int, task_description: str) -> EpisodeRecord:
    """Create a fallback record when workflow graph construction fails."""
    return EpisodeRecord(
        episode=episode,
        task_id=f"ep{episode}",
        sub_task_id="none",
        primitive_name="unknown",
        difficulty_bucket="unknown",
        difficulty=0.5,
        selected_candidate="none",
        predicted_acc=0.0,
        predicted_cost=0.0,
        true_acc=0.0,
        true_cost=0.0,
        observed_acc=0.0,
        observed_cost=0.0,
        eval_pass=False,
        failure_type="graph_construction_error",
        recalibrated=False,
        source="workflow_graph",
        constraint_violations=[],
        violation_count=0,
        human_approved=True,
        execution_duration=0.0,
        evaluator_name="unknown",
        evaluator_id="unknown",
        error_type="graph_error",
        confidence=0.0,
        evaluator_latency=0.0,
        evaluator_cost=0.0,
        quality_score=0.0,
        node_type="unknown",
        task_type="unknown",
        template_id="unknown",
        repair_action="none",
        llm_decomposer_cost=0.0,
        llm_evaluator_cost=0.0,
    )


# Monkey-patch ExperimentConfig to support use_workflow_graph
ExperimentConfig.use_workflow_graph: bool = field(default=False)


# ---------------------------------------------------------------------------
# Train / Test Experiment (Core Validation of Q(G;X) Template Selection)
# ---------------------------------------------------------------------------

def run_train_test_experiment(
    config: ExperimentConfig,
    train_ratio: float = 0.8,
) -> Dict[str, Any]:
    """
    Core experiment: train on 80% of tasks to learn template profiles,
    test on 20% using Q(G;X) = a*S - b*C - g*L for topology selection.

    训练阶段：轮转选择不同模板，累积 TemplateProfile 观测数据（quality/cost/latency）
    测试阶段：用 Q(G;X) 在帕累托前沿上选拓扑，评估选择质量

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration.
    train_ratio : float
        Fraction of episodes for training (default 0.8).

    Returns
    -------
    dict with keys:
        - train_records: List[EpisodeRecord]
        - test_records:  List[EpisodeRecord]
        - train_metrics:   dict
        - test_metrics:   dict
        - pareto_analysis: dict
    """
    rng = random.Random(config.seed)

    # ── Initialize components ───────────────────────────────────────────────
    profile_store = ProfileStore()
    manager = initialize_profile_manager(DEFAULT_GROUND_TRUTH)
    manager._calibration_interval = config.calibration_interval
    decomposer = TaskDecomposer(random_seed=config.seed)
    template_library = TemplateLibrary()
    evaluator = MockEvaluator(
        ground_truth=DEFAULT_GROUND_TRUTH,
        noise_std=config.noise_std,
        pass_threshold=config.pass_threshold,
        seed=config.seed,
        profile_store=profile_store,
    )

    n_total = config.n_episodes
    n_train = int(n_total * train_ratio)
    n_test = n_total - n_train

    print(f"\n{'='*60}")
    print(f"  Train/Test Experiment: {config.name}")
    print(f"  Total: {n_total} | Train: {n_train} ({train_ratio:.0%}) | Test: {n_test} ({(1-train_ratio):.0%})")
    print(f"  Q(G;X) = a*S - b*C - g*L  (a={config.q_alpha}, b={config.q_beta}, g={config.q_gamma})")
    print(f"  Noise: {config.noise_std}  Repair: {config.enable_repair}")
    print(f"{'='*60}")

    all_train_records: List[EpisodeRecord] = []
    all_test_records: List[EpisodeRecord] = []

    # 训练阶段轮转模板列表（确保每个模板都有观测）
    train_template_cycle = ["direct", "exec_verify", "dual_exec_aggregate", "exec_verify_hci"]
    train_templates_used: Dict[str, int] = defaultdict(int)

    for episode in range(1, n_total + 1):
        is_train = episode <= n_train

        # ── Task selection ──────────────────────────────────────────────────
        task_desc = (
            config.task_override
            if config.task_override is not None
            else (rng.choice(CONSTRAINED_TASK_BANK)
                  if rng.random() < config.constrained_task_ratio
                  else rng.choice(SAMPLE_TASKS))
        )

        sub_tasks, task_type = decomposer.decompose(
            task_desc,
            extract_constraints=config.enable_constraints,
        )
        if not sub_tasks:
            continue

        st = sub_tasks[0]
        bucket = st.difficulty_bucket
        node_type = st.primitive_name

        # ── Template selection ──────────────────────────────────────────────
        if is_train:
            # 训练阶段：轮转选择模板（保证覆盖）
            train_template = train_template_cycle[episode % len(train_template_cycle)]
            selected_template_obj = template_library.get_template(train_template)
            template_id = train_template
        else:
            # 测试阶段：用 Q(G;X) 在帕累托前沿上选
            frontier = template_library.pareto_frontier(node_type, task_type, bucket)
            if frontier:
                selected = template_library.select_from_frontier(
                    frontier,
                    acc_target=config.acc_target,
                    cost_budget=config.cost_budget,
                    latency_budget=config.latency_budget,
                    alpha=config.q_alpha,
                    beta=config.q_beta,
                    gamma=config.q_gamma,
                )
                template_id = selected.get("template_id", "direct")
            else:
                # 冷启动：没有训练数据，直接用 direct
                template_id = "direct"
            selected_template_obj = template_library.get_template(template_id)

        # ── Build & execute SubTaskSpec chain ─────────────────────────────
        if selected_template_obj is not None:
            instanced = selected_template_obj.instantiate(
                base_sub_task_id=st.sub_task_id,
                base_primitive=st.primitive_name,
                base_difficulty=st.difficulty,
                difficulty_bucket=bucket,
                constraints=st.constraints,
            )
            sub_task_list = instanced
        else:
            sub_task_list = [st]

        # 拓扑排序
        execution_order = _topo_sort(sub_task_list)

        # ── Execute chain ──────────────────────────────────────────────────
        episode_records: List[EpisodeRecord] = []
        task_passed = True
        episode_cost = 0.0
        episode_latency = 0.0

        for task_spec in execution_order:
            # 候选选择（ProfileManager）
            ranks = manager.predict_all(task_spec.primitive_name, task_spec.difficulty_bucket)
            if not ranks:
                continue
            selected = ranks[0]
            cand_name = selected["candidate_name"]

            evaluator_name = task_spec.evaluator_name or _select_evaluator(
                task_spec, cand_name, config
            )

            eval_result = evaluator.evaluate(
                candidate_name=cand_name,
                primitive_name=task_spec.primitive_name,
                difficulty_bucket=task_spec.difficulty_bucket,
                task_id=f"tt_ep{episode}",
                node_id=task_spec.sub_task_id,
                task_spec=task_spec,
                evaluator_name=evaluator_name,
                node_type=task_spec.primitive_name,
            )

            # ── Local repair (test phase 更需要修复) ──────────────────────
            # Track remaining budget for repair decision
            remaining_budget: float | None = (
                config.cost_budget - episode_cost if config.cost_budget else None
            )
            repair_count = 0
            repair_action = "none"
            repair_deltaG: dict | None = None
            repair_delta_nodes = 0
            repair_delta_edges = 0
            repaired_cand = cand_name
            while config.enable_repair and repair_count < MAX_REPAIR_ATTEMPTS:
                if not _should_repair(eval_result, repair_count):
                    break
                new_result, deltaG, repair_tried = _repair_subgraph(
                    task_spec,
                    repaired_cand,
                    eval_result,
                    manager,
                    evaluator,
                    config,
                    rng,
                    profile_store,
                    template_library=template_library,
                    current_template_id=template_id,
                    use_pareto=config.use_pareto,
                    remaining_budget=remaining_budget,
                )
                if new_result is not None and new_result.eval_pass:
                    eval_result = new_result
                    repair_deltaG = deltaG
                    if deltaG:
                        repair_action = deltaG.get("action", "unknown")
                        repair_delta_nodes = deltaG.get("delta_nodes", 0)
                        repair_delta_edges = deltaG.get("delta_edges", 0)
                        if deltaG.get("candidate_changed"):
                            repaired_cand = deltaG.get("to_candidate", repaired_cand)
                    else:
                        repair_action = "evaluator_upgrade"
                    break
                repair_count += 1
                if deltaG and deltaG.get("candidate_changed"):
                    repaired_cand = deltaG.get("to_candidate", repaired_cand)

            task_passed = task_passed and eval_result.eval_pass
            episode_cost += eval_result.observed_cost + eval_result.evaluator_cost
            episode_latency += eval_result.execution_duration or 0.0

            record = EpisodeRecord(
                episode=episode,
                task_id=f"tt_ep{episode}",
                sub_task_id=task_spec.sub_task_id,
                primitive_name=task_spec.primitive_name,
                difficulty_bucket=task_spec.difficulty_bucket,
                difficulty=task_spec.difficulty,
                selected_candidate=cand_name,
                predicted_acc=selected["pred_acc"],
                predicted_cost=selected["pred_cost"],
                true_acc=eval_result.true_quality or 0.0,
                true_cost=eval_result.true_cost or 0.0,
                observed_acc=eval_result.observed_quality,
                observed_cost=eval_result.observed_cost,
                eval_pass=eval_result.eval_pass,
                failure_type=eval_result.failure_type,
                recalibrated=False,
                source=selected["source"],
                constraint_violations=eval_result.constraint_violations,
                violation_count=len(eval_result.constraint_violations),
                human_approved=eval_result.human_approved,
                execution_duration=eval_result.execution_duration,
                input_modality=task_spec.input_modality.value,
                intermediate_modality=(
                    task_spec.intermediate_modality.value
                    if task_spec.intermediate_modality else None
                ),
                evaluator_name=eval_result.evaluator_name,
                evaluator_id=eval_result.evaluator_name,
                error_type=eval_result.error_type,
                confidence=eval_result.confidence,
                evaluator_latency=eval_result.evaluator_latency,
                evaluator_cost=eval_result.evaluator_cost,
                quality_score=eval_result.quality_score,
                node_type=task_spec.primitive_name,
                task_type=task_type,
                template_id=template_id,
                repair_action=repair_action,
                repair_deltaG=repair_deltaG,
                repair_delta_nodes=repair_delta_nodes,
                repair_delta_edges=repair_delta_edges,
            )
            episode_records.append(record)

            # 更新 ProfileManager
            fb = FeedbackRecord(
                task_id=record.task_id,
                node_id=record.sub_task_id,
                primitive_name=record.primitive_name,
                candidate_name=record.selected_candidate,
                difficulty=record.difficulty,
                difficulty_bucket=record.difficulty_bucket,
                predicted_quality=record.predicted_acc,
                predicted_cost=record.predicted_cost,
                observed_quality=record.observed_acc,
                observed_cost=record.observed_cost,
                eval_pass=record.eval_pass,
                failure_type=record.failure_type,
                episode=record.episode,
                constraint_violations=record.constraint_violations,
                execution_duration=record.execution_duration,
                human_approved=record.human_approved,
                input_modality=record.input_modality,
                intermediate_modality=record.intermediate_modality,
                violation_count=record.violation_count,
                evaluator_name=record.evaluator_name,
                evaluator_id=record.evaluator_name,
                error_type=record.error_type,
                confidence=record.confidence,
                evaluator_latency=record.evaluator_latency,
                evaluator_cost=record.evaluator_cost,
                task_type=record.task_type,
                node_type=record.node_type,
                template_id=record.template_id,
            )
            manager.add_feedback(fb)

            # ── Template-level feedback ───────────────────────────────────
            # 用 eval_result.execution_duration 作为 latency（如果没有，用 episode_latency）
            template_library.add_feedback(
                template_id=template_id,
                difficulty_bucket=bucket,
                observed_quality=1.0 if eval_result.eval_pass else 0.0,
                observed_cost=eval_result.observed_cost + eval_result.evaluator_cost,
                observed_latency=eval_result.execution_duration or episode_latency,
            )
            if is_train:
                train_templates_used[template_id] += 1

            # ProfileStore 更新
            if profile_store is not None:
                profile_store.update_workflow_profile(
                    template_id=template_id,
                    scenario=config.name,
                    task_type=task_type,
                    overall_pass=eval_result.eval_pass,
                    total_cost=eval_result.observed_cost + eval_result.evaluator_cost,
                    total_latency=eval_result.execution_duration or 0.0,
                    violation_count=len(eval_result.constraint_violations),
                    node_count=len(execution_order),
                    repair_count=repair_count,
                )

        all_records = episode_records
        if is_train:
            all_train_records.extend(all_records)
        else:
            all_test_records.extend(all_records)

        phase = "TRAIN" if is_train else "TEST"
        ep_pass = sum(1 for r in all_records if r.eval_pass)
        print(
            f"  [{phase}] ep {episode:3d} | t={template_id:<20s} "
            f"| pass={ep_pass}/{len(all_records)} "
            f"| cost={episode_cost:.2f} lat={episode_latency:.1f}s"
        )

    # ── Aggregate metrics ─────────────────────────────────────────────────
    def _bucket_groups(records):
        """Group records by difficulty_bucket into a dict of lists."""
        groups = defaultdict(list)
        for r in records:
            groups[r.difficulty_bucket].append(r)
        return dict(groups)

    def _metrics(records):
        if not records:
            return {}
        total = len(records)
        n_pass = sum(1 for r in records if r.eval_pass)
        return {
            "total": total,
            "pass_rate": round(n_pass / total, 4),
            "n_pass": n_pass,
            "mean_cost": round(sum(r.observed_cost for r in records) / total, 4),
            "mean_latency": round(sum(r.execution_duration or 0 for r in records) / total, 3),
            "mae": round(sum(abs(r.predicted_acc - r.true_acc) for r in records) / total, 4),
            "violation_rate": round(
                sum(1 for r in records if r.violation_count > 0) / total, 4
            ),
            "mean_q_score": round(sum(r.quality_score for r in records) / total, 4),
            "per_bucket_pass": {
                b: round(sum(1 for r in g if r.eval_pass) / len(g), 4)
                for b, g in _bucket_groups(records).items()
            },
            "template_counts": dict(Counter(
                r.template_id for r in records
            )),
        }

    train_metrics = _metrics(all_train_records)
    test_metrics = _metrics(all_test_records)

    # ── Pareto analysis ────────────────────────────────────────────────────
    # 在测试集上，分析帕累托前沿和 Q(G;X) 选择质量
    pareto_analysis = {}
    if all_test_records:
        ep_groups = defaultdict(list)
        for r in all_test_records:
            ep_groups[r.episode].append(r)

        dominated_count = 0    # Q(G;X) 选中的模板被其他模板支配的次数
        correct_pick = 0       # Q(G;X) 选中的模板在帕累托前沿上的次数
        frontier_sizes = []
        selected_templates = []

        for ep, recs in sorted(ep_groups.items()):
            tids = set(r.template_id for r in recs)
            frontier_sizes.append(len(tids))
            selected_templates.append(recs[0].template_id if recs else "unknown")

            # 简化：检查是否有多个模板可选（训练数据足够多时）
            if len(tids) > 1:
                # 检查选择的模板是否 cost 合理（相对其他）
                selected_cost = sum(r.observed_cost for r in recs)
                all_costs = [
                    sum(r.observed_cost for r in recs if r.template_id == t)
                    for t in tids
                ]
                if selected_cost == min(all_costs):
                    correct_pick += 1

        pareto_analysis = {
            "selected_templates": selected_templates,
            "avg_frontier_size": round(sum(frontier_sizes) / max(len(frontier_sizes), 1), 2),
            "test_template_distribution": dict(
                Counter(selected_templates)
            ),
        }

    # ── Print summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  TRAIN/TEST SUMMARY")
    print(f"{'='*60}")
    print(f"\n  [TRAIN] {len(all_train_records)} records | pass={train_metrics.get('pass_rate',0):.1%}")
    print(f"          templates used: {train_templates_used}")
    print(f"\n  [TEST]  {len(all_test_records)} records | pass={test_metrics.get('pass_rate',0):.1%}")
    print(f"          templates: {test_metrics.get('template_counts', {})}")
    print(f"          mean cost: {test_metrics.get('mean_cost',0):.3f}")
    print(f"          mean latency: {test_metrics.get('mean_cost',0):.3f}s")
    print(f"          MAE: {test_metrics.get('mae',0):.3f}")
    print(f"  [PARETO] avg frontier size: {pareto_analysis.get('avg_frontier_size','N/A')}")
    print(f"           selected: {pareto_analysis.get('test_template_distribution',{})}")

    # ── Save ───────────────────────────────────────────────────────────────
    import csv as _csv
    from dataclasses import asdict as _asdict

    for label, records in [("train", all_train_records), ("test", all_test_records)]:
        if not records:
            continue
        path = config.output_dir / f"train_test_{label}.csv"
        fields = list(_asdict(records[0]).keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = _csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in records:
                row = _asdict(r)
                row["constraint_violations"] = str(row["constraint_violations"])
                writer.writerow(row)
        print(f"  Saved {path}")

    # Save JSON summary
    summary_path = config.output_dir / f"train_test_summary.json"
    import json as _json
    summary = {
        "config": asdict(config),
        "train_ratio": train_ratio,
        "n_train": n_train,
        "n_test": n_test,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "pareto_analysis": pareto_analysis,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        _json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved {summary_path}")

    profile_store.save(config.output_dir)

    return {
        "config": config,
        "train_records": all_train_records,
        "test_records": all_test_records,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "pareto_analysis": pareto_analysis,
    }


if __name__ == "__main__":
    main()
