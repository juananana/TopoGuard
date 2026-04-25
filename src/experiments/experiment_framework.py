"""
experiment_framework.py
========================
Task-agnostic experiment framework for TopoGuard.

Usage — add a new task type in 3 steps:
  1. Create a TaskSpec instance in your experiment_*.py
  2. Write a make_subtask_fn(sample, sub_task_id, seed) -> SubTaskSpec
  3. Call run_experiment(task_spec, args) from main()

Example — water_qa:
    spec = build_water_qa_task_spec()
    run_experiment(spec, args)

Framework responsibilities (task-agnostic):
  - Load GT / annotated tasks / profile data
  - Stratified train/val split
  - Cold-start estimation from GT data
  - Initialize PrimitivePerformanceProfileManager
  - Training loop (episodes → manager → recalibration)
  - Full evaluation with oracle + baseline comparators
  - Save CSV, JSON summary, curve snapshots
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_SRC_DIR = Path(__file__).parent.parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from src.primitives.profile_manager import PrimitivePerformanceProfileManager
from src.primitives.feedback_record import FeedbackRecord
from src.primitives.profile_store import ProfileStore
from src.primitives.topology_template import TemplateLibrary
from src.decomposer.task_decomposer import SubTaskSpec
from src.evaluation.mock_evaluator import MockEvaluator
from src.experiments.mvp_experiment import (
    ExperimentConfig,
    EpisodeRecord,
    _repair_subgraph,
    _should_repair,
    _select_evaluator,
    _simulate_human_approval,
    REPAIR_THRESHOLD,
    MAX_REPAIR_ATTEMPTS,
)


# ---------------------------------------------------------------------------
# Task Specification (all task-specific configuration in one place)
# ---------------------------------------------------------------------------

@dataclass
class TaskSpec:
    """
    Complete specification for one task domain with one or more node_types.

    Add a new task type by filling in this spec — no changes to framework code needed.

    Multi-node support:
      - Single node_type: set node_type field (backward-compatible)
      - Multiple node_types: set node_type_router field
        The router function maps each question (str) → node_type (str)
        The framework then uses profiles from that specific node_type.
    """
    # ── Identity ─────────────────────────────────────────────────────────────
    name: str                         # e.g. "water_qa", "forecast", "text_classify"
    # Single node_type: backward-compatible mode
    node_type: str = "executor"       # primitive name for single-node tasks
    # Multi-node routing: function(question: str) -> node_type
    # Takes precedence over node_type if set.
    node_type_router: Optional[Callable[[str], str]] = field(default=None)
    executor_names: List[str] = field(default_factory=lambda: ["qwen_7b", "deepseek_v3"])
    difficulties: List[str] = field(default_factory=lambda: ["easy", "medium", "hard", "extreme"])
    bucket_weights: Dict[str, float] = field(default_factory=lambda: {"easy": 0.35, "medium": 0.35, "hard": 0.30, "extreme": 0.0})

    # ── File paths ────────────────────────────────────────────────────────────
    annotated_tasks_path: str = "data/tasks_annotated.json"
    raw_records_path: str = "data/raw_records.jsonl"
    executor_profiles_path: str = "data/executor_profiles.jsonl"

    # ── Upgrade chains (strategy B) — per node_type dict ─────────────────────
    # Single-node: List[Tuple[candidate, description]]
    # Multi-node:   Dict[node_type, List[Tuple[candidate, description]]]
    upgrade_chain: Any = field(default_factory=list)
    # Example (multi-node):
    #   {
    #       "SIMPLE":      [("qwen_7b", "free"), ("glm_5", "expensive")],
    #       "REASONING":    [("deepseek_v3", "fast"), ("kimi_k25", "best")],
    #   }
    # Example (single-node, backward-compatible):
    #   [("qwen_7b", "free"), ("glm_5", "expensive")]

    # ── Subtask factory ────────────────────────────────────────────────────────
    make_subtask_fn: Callable[[Dict[str, Any], str, int], SubTaskSpec] = field(default=None)
    # Signature: make_subtask_fn(sample: dict, sub_task_id: str, seed: int) -> SubTaskSpec

    # ── Cold-start overrides ───────────────────────────────────────────────────
    cold_start_quality_override: Optional[Dict[str, Dict[str, float]]] = None
    cold_start_cost_override: Optional[Dict[str, Dict[str, float]]] = None
    # Difficulty value map: {bucket: float} for SubTaskSpec.difficulty
    diff_map: Dict[str, float] = field(default_factory=lambda: {"easy": 0.15, "medium": 0.37, "hard": 0.62, "extreme": 0.82})

    # ── Multi-node helpers ──────────────────────────────────────────────────────
    def resolve_node_type(self, question: str) -> str:
        """Resolve the node_type for a given question string."""
        if self.node_type_router is not None:
            return self.node_type_router(question)
        return self.node_type

    def resolve_upgrade_chain(self, question: str) -> List[Tuple[str, str]]:
        """Resolve the upgrade chain for a given question."""
        if isinstance(self.upgrade_chain, dict):
            nt = self.resolve_node_type(question)
            return self.upgrade_chain.get(nt, [])
        return self.upgrade_chain or []


# ---------------------------------------------------------------------------
# Experiment Runner (task-agnostic)
# ---------------------------------------------------------------------------

@dataclass
class ExperimentOutput:
    train_records: List[EpisodeRecord]
    val_records: List[EpisodeRecord]
    curve_snapshots: List[dict]
    baseline_stats: Dict[str, dict]
    final_table: List[dict]


def run_experiment(
    task_spec: TaskSpec,
    args: Any,          # argparse.Namespace from experiment_*.py
    extra_episode_hook: Optional[Callable] = None,  # called each episode: hook(episode, records, stats)
) -> ExperimentOutput:
    """
    Task-agnostic training + evaluation loop.

    Parameters
    ----------
    task_spec : TaskSpec
        All task-specific configuration.
    args : argparse.Namespace
        Must have: n_episodes, calibration_interval, acc_target, cost_budget,
        pass_threshold, noise_std, seed, cold_start_k, train_ratio.
        Optional: disable_repair, use_pareto, pareto_mode, q_alpha, q_beta, q_gamma.
    extra_episode_hook : callable, optional
        Called after each episode with (episode, records, stats).
        Use for task-specific logging or metric tracking.
    """
    rng = random.Random(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or "outputs") / task_spec.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ground truth from profile store ──────────────────────────────────
    profile_store = ProfileStore(task_spec.executor_profiles_path)
    _load_gt_from_profile_store(profile_store, task_spec)

    # ── Load annotated tasks ─────────────────────────────────────────────────
    with open(task_spec.annotated_tasks_path, encoding="utf-8") as f:
        all_samples = json.load(f)
    if hasattr(args, "n_samples") and args.n_samples and args.n_samples < len(all_samples):
        all_samples = all_samples[:args.n_samples]

    # ── Load raw records → build GT lookups ──────────────────────────────────
    gt_quality: Dict[str, Dict[str, float]] = {}
    gt_cost: Dict[str, Dict[str, float]] = {}
    _load_gt_from_raw_records(task_spec.raw_records_path, gt_quality, gt_cost)

    # ── Stratified train/val split ──────────────────────────────────────────
    train_ratio = getattr(args, "train_ratio", 0.75)
    buckets = task_spec.difficulties
    gt_sids = set(gt_quality.keys())
    samples_with_gt = [s for s in all_samples
                       if s.get("sample_id", "") in gt_sids or s.get("sample_id") in {
                           _sid for _sid in gt_quality if _sid in gt_quality
                       }]

    # Safe filter: keep samples that appear in gt_quality
    gt_sample_ids = set(gt_quality.keys())
    samples_with_gt = [s for s in all_samples if s.get("sample_id", "") in gt_sample_ids]

    train_samples: List[Dict] = []
    val_samples: List[Dict] = []
    for bucket in buckets:
        bucket_samples = [s for s in samples_with_gt if s.get("difficulty") == bucket]
        rng.shuffle(bucket_samples)
        split = int(len(bucket_samples) * train_ratio)
        train_samples.extend(bucket_samples[:split])
        val_samples.extend(bucket_samples[split:])

    print(f"[Step 3] Stratified split (GT samples only): "
          f"train={len(train_samples)}, val={len(val_samples)}")
    print(f"  Train: {dict(Counter(s.get('difficulty') for s in train_samples))}")
    print(f"  Val:   {dict(Counter(s.get('difficulty') for s in val_samples))}")

    # ── Cold start ────────────────────────────────────────────────────────────
    k = getattr(args, "cold_start_k", 3)
    print(f"\n[Step 4] Cold start: k={k} items per (difficulty, executor)...")
    cold_start_estimates: Dict[str, Dict[str, float]] = {b: {} for b in buckets}
    cold_start_cost: Dict[str, Dict[str, float]] = {b: {} for b in buckets}

    for bucket in buckets:
        bucket_train = [s for s in train_samples if s.get("difficulty") == bucket]
        if not bucket_train:
            continue
        rng.shuffle(bucket_train)
        cold_samples = bucket_train[:min(k, len(bucket_train))]
        n_cold = len(cold_samples)

        for cand in task_spec.executor_names:
            qualities = []
            costs = []
            for s in cold_samples:
                sid = s.get("sample_id", "")
                if sid in gt_quality and cand in gt_quality[sid]:
                    qualities.append(gt_quality[sid][cand])
                    if sid in gt_cost and cand in gt_cost[sid]:
                        costs.append(gt_cost[sid][cand])
            if qualities:
                cold_start_estimates[bucket][cand] = sum(qualities) / len(qualities)
                cold_start_cost[bucket][cand] = (sum(costs) / len(costs)) if costs else 0.001
            else:
                prof = profile_store.get_executor_profile(f"{task_spec.node_type}/{cand}", bucket)
                cold_start_estimates[bucket][cand] = (prof.quality_mean if prof else 0.5)
                cold_start_cost[bucket][cand] = (prof.api_cost_mean if prof else 0.001)

        vals_str = ", ".join(
            f"{c}={cold_start_estimates[bucket].get(c, -1):.3f}"
            for c in task_spec.executor_names
            if cold_start_estimates[bucket].get(c, -1) >= 0
        )
        cost_str = ", ".join(
            f"{c}={cold_start_cost[bucket].get(c, -1):.4f}"
            for c in task_spec.executor_names
            if cold_start_cost[bucket].get(c, -1) >= 0
        )
        print(f"  {bucket} (n={n_cold}): quality: {vals_str}")
        print(f"           cost(c_main): {cost_str}")

    # ── Initialize manager ────────────────────────────────────────────────────
    print(f"\n[Step 5] Initializing manager with cold-start curves...")
    manager = PrimitivePerformanceProfileManager(
        calibration_interval=args.calibration_interval,
        fallback_quality=0.5,
        fallback_cost=1.0,
    )
    manager.register_primitive(task_spec.node_type, primitive_type=task_spec.node_type)

    for cand in task_spec.executor_names:
        init_curve: Dict[str, Dict[str, float]] = {}
        for diff in buckets:
            prof = profile_store.get_executor_profile(f"{task_spec.node_type}/{cand}", diff)
            if prof is not None:
                init_curve[diff] = {
                    "acc_mean": prof.quality_mean,
                    "cost_mean": max(prof.api_cost_mean, 0.001),
                }
            else:
                init_curve[diff] = {"acc_mean": 0.5, "cost_mean": 1.0}
        manager.register_candidate(
            primitive_name=task_spec.node_type,
            candidate_name=cand,
            init_curve=init_curve,
        )
    print(f"  Manager initialized with cold-start curves")

    # ── Components ────────────────────────────────────────────────────────────
    template_library = TemplateLibrary()
    evaluator = MockEvaluator(
        ground_truth={},
        noise_std=args.noise_std,
        pass_threshold=args.pass_threshold,
        seed=args.seed,
        profile_store=profile_store,
        pareto_mode=getattr(args, "pareto_mode", False) or getattr(args, "use_pareto", False),
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    n_train = args.n_episodes
    print(f"\n[Step 6] Training: {n_train} episodes on {len(train_samples)} train samples...")
    print(f"  Acc target={args.acc_target}  Cost budget={args.cost_budget}  "
          f"Repair={'ON' if not getattr(args, 'disable_repair', False) else 'OFF'}")

    train_records: List[EpisodeRecord] = []
    curve_snapshots: List[dict] = []

    def _snap(details):
        curve_snapshots.append({
            "episode": manager.episode_counter,
            "table": manager.export_curve_table(),
            "details": details,
        })
    manager.register_post_calibration_hook(_snap)

    _exp_config = _make_exp_config(args)
    enable_repair = not getattr(args, "disable_repair", False)
    use_pareto = getattr(args, "use_pareto", False) or getattr(args, "pareto_mode", False)

    for episode in range(1, n_train + 1):
        bucket = rng.choices(buckets, weights=[task_spec.bucket_weights.get(b, 0.25) for b in buckets], k=1)[0]
        candidates = [s for s in train_samples if s.get("difficulty") == bucket]
        if not candidates:
            candidates = train_samples
        sample = rng.choice(candidates)
        task_id = sample.get("sample_id", f"train_{episode:03d}")

        # Task-specific subtask factory
        st = task_spec.make_subtask_fn(sample, f"tr_{episode:03d}", args.seed + episode)
        st.metadata["task_id"] = task_id

        records, _, ep_stats = _run_episode_generic(
            episode=episode,
            tasks=[st],
            manager=manager,
            evaluator=evaluator,
            config=_exp_config,
            profile_store=profile_store,
            template_library=template_library,
            rng=rng,
            task_spec=task_spec,
            extra_hook=extra_episode_hook,
        )
        train_records.extend(records)

        if manager.episode_counter % args.calibration_interval == 0:
            manager.batch_recalibrate()

    if manager.feedback_buffer_size > 0:
        manager.batch_recalibrate()

    final_table = manager.export_curve_table()
    print(f"\n  Post-training curve table:")
    for row in final_table:
        print(f"    {row}")

    # ── Evaluation on held-out set ────────────────────────────────────────────
    print(f"\n[Step 7] Full evaluation on {len(val_samples)} test samples...")
    val_records: List[EpisodeRecord] = []

    baseline_stats = _run_evaluation_generic(
        val_samples=val_samples,
        manager=manager,
        evaluator=evaluator,
        config=_exp_config,
        profile_store=profile_store,
        template_library=template_library,
        rng=rng,
        task_spec=task_spec,
    )

    # ── Save outputs ──────────────────────────────────────────────────────────
    _save_outputs(train_records, val_records, output_dir, timestamp, final_table,
                  curve_snapshots, baseline_stats, args)

    return ExperimentOutput(
        train_records=train_records,
        val_records=val_records,
        curve_snapshots=curve_snapshots,
        baseline_stats=baseline_stats,
        final_table=final_table,
    )


# ---------------------------------------------------------------------------
# Generic episode runner (used by run_experiment)
# ---------------------------------------------------------------------------

def _make_exp_config(args):
    """Build ExperimentConfig from argparse.Namespace."""
    cfg = ExperimentConfig(
        acc_target=args.acc_target,
        cost_budget=args.cost_budget,
        latency_budget=getattr(args, "latency_budget", None),
        noise_std=args.noise_std,
        pass_threshold=args.pass_threshold,
        enable_repair=not getattr(args, "disable_repair", False),
        enable_constraints=True,
        constraint_violation_penalty=-0.5,
        fixed_template=False,
        q_alpha=getattr(args, "q_alpha", 0.6),
        q_beta=getattr(args, "q_beta", 0.2),
        q_gamma=getattr(args, "q_gamma", 0.2),
    )
    return cfg


def _run_episode_generic(
    episode: int,
    tasks: List[SubTaskSpec],
    manager: PrimitivePerformanceProfileManager,
    evaluator: MockEvaluator,
    config: ExperimentConfig,
    profile_store: ProfileStore,
    template_library: TemplateLibrary,
    rng: random.Random,
    task_spec: TaskSpec,
    extra_hook: Optional[Callable] = None,
) -> Tuple[List[EpisodeRecord], List[FeedbackRecord], Dict[str, Any]]:
    """
    Task-agnostic single-episode runner.
    Mirrors the logic previously in run_water_qa_episode().
    """
    primitive_name = task_spec.node_type
    repair_chains = {primitive_name: task_spec.upgrade_chain}

    records: List[EpisodeRecord] = []
    feedback_records: List[FeedbackRecord] = []
    episode_stats: Dict[str, Any] = {
        "repairs": 0,
        "repair_actions": Counter(),
        "deltaG_phi": 0, "deltaG_tau": 0,
        "deltaG_nodes": 0, "deltaG_edges": 0,
        "constraint_violations": 0,
        "violation_types": Counter(),
        "eval_levels": Counter(),
        "repair_tried": [],
    }

    for st in tasks:
        # predict_all returns scored candidates
        scored = manager.predict_all(
            primitive_name=primitive_name,
            difficulty_bucket=st.difficulty_bucket,
            constraints=st.constraints,
            acc_target=config.acc_target,
            cost_budget=config.cost_budget,
            latency_budget=config.latency_budget,
        )

        if not scored:
            continue

        top = scored[0]
        cand_name = top.candidate_name
        pred_acc = top.predicted_quality
        pred_cost = top.predicted_cost

        # Execute subtask with top candidate
        eval_result = evaluator.evaluate(
            candidate_name=cand_name,
            primitive_name=primitive_name,
            difficulty_bucket=st.difficulty_bucket,
            task_id=st.metadata.get("task_id", ""),
            node_id=st.sub_task_id,
            metadata=st.metadata,
            node_type=primitive_name,
        )
        eval_pass = eval_result.eval_pass

        # Repair loop
        repair_action = "none"
        repair_deltaG: Optional[dict] = None
        delta_nodes = delta_edges = delta_phi = delta_tau = 0
        repair_tried: List[str] = []
        initial_eval_level = getattr(eval_result, "eval_level", "fail")

        if config.enable_repair and not eval_pass:
            initial_lvl = getattr(eval_result, "eval_level", "fail")
            episode_stats["eval_levels"][initial_lvl] += 1

            new_result, deltaG, strategies_tried = _repair_subgraph(
                st=st,
                failed_candidate=cand_name,
                eval_result=eval_result,
                manager=manager,
                evaluator=evaluator,
                config=config,
                rng=rng,
                profile_store=profile_store,
                template_library=template_library,
                current_template_id="direct",
                use_pareto=getattr(config, "use_pareto", False),
                remaining_budget=None,
                repair_chains=repair_chains,
            )
            repair_tried = strategies_tried

            if new_result is not None and new_result.eval_pass:
                eval_pass = True
                eval_result = new_result
                repair_action = deltaG.get("action", "none") if deltaG else "none"
                repair_deltaG = deltaG
                delta_phi = deltaG.get("delta_phi", 0) if deltaG else 0
                delta_tau = deltaG.get("delta_tau", 0) if deltaG else 0
                episode_stats["repairs"] += 1
                episode_stats["repair_actions"][repair_action] += 1
                episode_stats["deltaG_phi"] += delta_phi
                episode_stats["deltaG_tau"] += delta_tau
                episode_stats["repair_tried"].append(strategies_tried)
            else:
                episode_stats["repair_tried"].append(strategies_tried)
        else:
            final_lvl = getattr(eval_result, "eval_level", "pass")
            episode_stats["eval_levels"][final_lvl] += 1

        # Cost from eval_result
        ev_c_main  = getattr(eval_result, "c_main", 0.0) or 0.0
        ev_c_llm   = getattr(eval_result, "c_llm", 0.0) or 0.0
        ev_c_total = getattr(eval_result, "c_total", 0.0) or 0.0
        ev_l_raw   = getattr(eval_result, "latency", 0.0) or 0.0

        _max_c = 3.76  # reference max for water_qa
        ev_c_norm = min(ev_c_total / _max_c, 1.0) if _max_c > 0 else 0.0
        _max_l = 60.0
        ev_l_norm = min(ev_l_raw / _max_l, 1.0) if _max_l > 0 else 0.0

        true_acc_val = getattr(eval_result, "true_quality", pred_acc) or 0.0
        obs_acc_val = getattr(eval_result, "observed_quality", pred_acc) or pred_acc
        obs_cost_val = getattr(eval_result, "observed_cost", ev_c_total) or ev_c_total

        record = EpisodeRecord(
            episode=episode,
            task_id=st.metadata.get("task_id", ""),
            sub_task_id=st.sub_task_id,
            primitive_name=primitive_name,
            difficulty_bucket=st.difficulty_bucket,
            difficulty=st.difficulty,
            selected_candidate=cand_name,
            predicted_acc=round(pred_acc, 4),
            predicted_cost=round(pred_cost, 4),
            true_acc=round(true_acc_val, 4),
            true_cost=round(getattr(eval_result, "true_cost", pred_cost) or pred_cost, 4),
            observed_acc=round(obs_acc_val, 4),
            observed_cost=round(obs_cost_val, 4),
            eval_pass=eval_pass,
            failure_type=getattr(eval_result, "error_type", None),
            recalibrated=False,
            source="bucket",
            constraint_violations=getattr(eval_result, "constraint_violations", []),
            violation_count=len(getattr(eval_result, "constraint_violations", [])),
            human_approved=True,
            execution_duration=ev_l_raw,
            evaluator_name=getattr(eval_result, "evaluator_name", "rule_eval"),
            evaluator_id=getattr(eval_result, "evaluator_name", "rule_eval"),
            error_type=getattr(eval_result, "error_type", None),
            confidence=getattr(eval_result, "confidence", 1.0),
            evaluator_latency=getattr(eval_result, "evaluator_latency", 0.0),
            evaluator_cost=getattr(eval_result, "evaluator_cost", 0.0),
            quality_score=round(getattr(eval_result, "observed_quality", obs_acc_val), 4),
            node_type=primitive_name,
            task_type=task_spec.name,
            template_id="direct",
            repair_action=repair_action,
            template_upgraded_from="none",
            template_upgraded_to="direct",
            repair_deltaG=repair_deltaG,
            repair_delta_nodes=0,
            repair_delta_edges=0,
            delta_phi=delta_phi,
            delta_tau=delta_tau,
            delta_g=0.0,
            llm_decomposer_cost=0.0,
            llm_evaluator_cost=ev_c_llm,
            c_main=ev_c_main,
            c_llm=ev_c_llm,
            c_total=ev_c_total,
            c_usd_raw=ev_c_total,
            c_norm=ev_c_norm,
            l_raw=ev_l_raw,
            l_norm=ev_l_norm,
            eval_level=getattr(eval_result, "eval_level", "pass"),
            initial_eval_level=initial_eval_level,
            repair_tried=repair_tried,
            q_alpha=getattr(config, "q_alpha", 0.6),
            q_beta=getattr(config, "q_beta", 0.2),
            q_gamma=getattr(config, "q_gamma", 0.2),
            q_formula_version="v1",
            metric_version="v5.0",
        )
        records.append(record)

        # Feedback to manager
        fb = FeedbackRecord(
            primitive_name=primitive_name,
            candidate_name=cand_name,
            difficulty_bucket=st.difficulty_bucket,
            observed_quality=getattr(eval_result, "observed_quality", pred_acc),
            observed_cost=ev_c_total,
        )
        manager.add_feedback(fb)
        feedback_records.append(fb)

    # Extra hook (task-specific per-episode logging)
    if extra_hook is not None:
        extra_hook(episode, records, episode_stats)

    return records, feedback_records, episode_stats


# ---------------------------------------------------------------------------
# Generic evaluation + baselines
# ---------------------------------------------------------------------------

def _run_evaluation_generic(
    val_samples: List[Dict],
    manager: PrimitivePerformanceProfileManager,
    evaluator: MockEvaluator,
    config: ExperimentConfig,
    profile_store: ProfileStore,
    template_library: TemplateLibrary,
    rng: random.Random,
    task_spec: TaskSpec,
) -> Dict[str, dict]:
    """Run full episodes on val set + compute baseline comparators."""
    primitive_name = task_spec.node_type
    buckets = task_spec.difficulties

    baseline_stats = {
        "Baseline-A_cheapest":      {"pass": 0, "cost": 0.0, "quality": 0.0, "count": 0},
        "Baseline-B_best_quality":  {"pass": 0, "cost": 0.0, "quality": 0.0, "count": 0},
        "Baseline-C_coldstart":     {"pass": 0, "cost": 0.0, "quality": 0.0, "count": 0},
        "Baseline-D_oracle":         {"pass": 0, "cost": 0.0, "quality": 0.0, "count": 0},
        "TopoGuard_optimizer":      {"pass": 0, "cost": 0.0, "quality": 0.0, "count": 0, "repairs": 0},
    }

    # Build quality/cost maps per executor for baselines
    executor_cost_map: Dict[str, float] = {}
    executor_quality_map: Dict[str, float] = {}
    for e in task_spec.executor_names:
        costs_e, qualities_e = [], []
        for diff in buckets:
            prof = profile_store.get_executor_profile(f"{primitive_name}/{e}", diff)
            if prof:
                costs_e.append(prof.api_cost_mean)
                qualities_e.append(prof.quality_mean)
        if costs_e:
            executor_cost_map[e] = sum(costs_e) / len(costs_e)
            executor_quality_map[e] = sum(qualities_e) / len(qualities_e)

    cheapest_cand = min(executor_cost_map, key=executor_cost_map.get) if executor_cost_map else task_spec.executor_names[0]
    best_quality_cand = max(executor_quality_map, key=executor_quality_map.get) if executor_quality_map else task_spec.executor_names[0]

    repair_chains = {primitive_name: task_spec.upgrade_chain}
    _exp_config = _make_exp_config(config) if hasattr(config, "acc_target") else config

    for idx, sample in enumerate(val_samples):
        bucket = sample.get("difficulty", "medium")
        task_id = sample.get("sample_id", f"val_{idx:03d}")
        st = task_spec.make_subtask_fn(sample, f"val_{idx:03d}", 42 + idx)
        st.metadata["task_id"] = task_id

        # ── Baseline A: cheapest ────────────────────────────────────────────
        _eval_baseline(evaluator, primitive_name, cheapest_cand, bucket, task_id, baseline_stats["Baseline-A_cheapest"], val=True)

        # ── Baseline B: best quality ────────────────────────────────────────
        _eval_baseline(evaluator, primitive_name, best_quality_cand, bucket, task_id, baseline_stats["Baseline-B_best_quality"], val=True)

        # ── Baseline C: cold-start (first executor) ────────────────────────
        cold_cand = task_spec.executor_names[0]
        _eval_baseline(evaluator, primitive_name, cold_cand, bucket, task_id, baseline_stats["Baseline-C_coldstart"], val=True)

        # ── Baseline D: oracle ──────────────────────────────────────────────
        oracle_cand = best_quality_cand
        _eval_baseline(evaluator, primitive_name, oracle_cand, bucket, task_id, baseline_stats["Baseline-D_oracle"], val=True)

        # ── TopoGuard optimizer ──────────────────────────────────────────────
        scored = manager.predict_all(
            primitive_name=primitive_name,
            difficulty_bucket=bucket,
            constraints=st.constraints,
            acc_target=config.acc_target,
            cost_budget=config.cost_budget,
            latency_budget=config.latency_budget,
        )
        top_cand = scored[0].candidate_name if scored else cheapest_cand
        top_cost = scored[0].predicted_cost if scored else 0.0

        eval_result = evaluator.evaluate(
            candidate_name=top_cand,
            primitive_name=primitive_name,
            difficulty_bucket=bucket,
            task_id=task_id,
            node_id=st.sub_task_id,
            metadata=st.metadata,
            node_type=primitive_name,
        )
        ep_pass = eval_result.eval_pass
        ep_qual = getattr(eval_result, "observed_quality", 0.0)
        ep_cost = getattr(eval_result, "c_total", top_cost)
        ep_repairs = 0

        if config.enable_repair and not ep_pass:
            new_result, _, _ = _repair_subgraph(
                st=st, failed_candidate=top_cand, eval_result=eval_result,
                manager=manager, evaluator=evaluator, config=config, rng=rng,
                profile_store=profile_store, template_library=template_library,
                current_template_id="direct", use_pareto=getattr(config, "use_pareto", False),
                remaining_budget=None, repair_chains=repair_chains,
            )
            if new_result is not None and new_result.eval_pass:
                ep_pass = True
                ep_qual = getattr(new_result, "observed_quality", ep_qual)
                ep_cost = getattr(new_result, "c_total", ep_cost)
                ep_repairs = 1
            else:
                ep_repairs = 1

        s = baseline_stats["TopoGuard_optimizer"]
        s["pass"] += int(ep_pass)
        s["cost"] += ep_cost
        s["quality"] += ep_qual
        s["count"] += 1
        s["repairs"] += ep_repairs

    # Compute vs-oracle gaps
    n = len(val_samples) or 1
    topguard_pass = baseline_stats["TopoGuard_optimizer"]["pass"] / n * 100
    oracle_pass = baseline_stats["Baseline-D_oracle"]["pass"] / n * 100

    # Print comparison table
    print("\n  [Test Set Baseline Comparison]")
    print(f"  {'Method':<30} {'PassRate':>10}   {'AvgQuality':>11}   {'AvgCost':>10}   {'vsOracle':>10}")
    print(f"  {'-'*30} {'-'*10}   {'-'*11}   {'-'*10}   {'-'*10}")
    for method, s in sorted(baseline_stats.items()):
        cnt = s["count"] or 1
        pr = s["pass"] / cnt * 100
        aq = s["quality"] / cnt
        ac = s["cost"] / cnt
        vs_o = pr - oracle_pass if method != "Baseline-D_oracle" else 0.0
        marker = " [ours]" if method == "TopoGuard_optimizer" else (" [upper]" if method == "Baseline-D_oracle" else "")
        print(f"  {method:<30} {pr:>7.1f}%   {aq:>10.3f}   {ac:>10.4f}   {vs_o:>+8.1f}%{marker}")

    return baseline_stats


def _eval_baseline(
    evaluator: MockEvaluator,
    primitive_name: str,
    candidate_name: str,
    difficulty_bucket: str,
    task_id: str,
    stats: dict,
    val: bool = False,
) -> None:
    """Evaluate one baseline candidate, update stats."""
    result = evaluator.evaluate(
        candidate_name=candidate_name,
        primitive_name=primitive_name,
        difficulty_bucket=difficulty_bucket,
        task_id=task_id,
        node_id=task_id,
        metadata={"source": "baseline", "task_id": task_id},
        node_type=primitive_name,
    )
    stats["pass"] += int(result.eval_pass)
    stats["quality"] += getattr(result, "observed_quality", 0.0)
    stats["cost"] += getattr(result, "c_total", 0.0)
    stats["count"] += 1


# ---------------------------------------------------------------------------
# GT helpers
# ---------------------------------------------------------------------------

def _load_gt_from_profile_store(ps: ProfileStore, spec: TaskSpec) -> None:
    """Verify profile store has data for this task spec."""
    # Side-effect-free — just logs summary
    counts = {}
    for cand in spec.executor_names:
        for diff in spec.difficulties:
            prof = ps.get_executor_profile(f"{spec.node_type}/{cand}", diff)
            if prof:
                counts[f"{cand}/{diff}"] = prof.sample_count
    total = sum(counts.values())
    print(f"[Step 1] Loaded GT: {len(spec.executor_names)} executors, {total} samples")
    print(f"  Cost model: c_main = (a1/1e6)*input + (a2/1e6)*output  [USD]")
    if counts:
        print(f"  Sample counts: {dict(list(counts.items())[:3])}...")


def _load_gt_from_raw_records(
    path: str,
    gt_quality: Dict[str, Dict[str, float]],
    gt_cost: Dict[str, Dict[str, float]],
) -> None:
    """Load ground truth quality/cost from raw records JSONL."""
    p = Path(path)
    if not p.exists():
        return
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            sid = r.get("sample_id", "")
            tool_id = r.get("tool_id", "")
            # Extract candidate name: "water_qa/qwen_7b" → "qwen_7b"
            cand = tool_id.split("/")[-1] if "/" in tool_id else tool_id
            quality = r.get("observed_quality", 0.0)
            cost = r.get("c_main", r.get("c_total", r.get("observed_cost", 0.0)))
            gt_quality.setdefault(sid, {})[cand] = quality
            gt_cost.setdefault(sid, {})[cand] = cost


def _save_outputs(
    train_records: List[EpisodeRecord],
    val_records: List[EpisodeRecord],
    output_dir: Path,
    timestamp: str,
    final_table: List[dict],
    curve_snapshots: List[dict],
    baseline_stats: Dict[str, dict],
    args: Any,
) -> None:
    """Save CSV, val CSV, and JSON summary."""
    prefix = f"{output_dir.parent.name}_{timestamp}"

    # Full CSV
    full_csv = output_dir / f"{prefix}_full.csv"
    _write_episode_csv(full_csv, train_records)

    # Val CSV
    val_csv = output_dir / f"{prefix}_val.csv"
    _write_episode_csv(val_csv, val_records)

    # Summary JSON
    summary = {
        "task": output_dir.parent.name,
        "timestamp": timestamp,
        "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "train": {
            "n_episodes": len(train_records),
            "pass_rate": sum(r.eval_pass for r in train_records) / max(len(train_records), 1),
        },
        "test": {
            "n_samples": len(val_records),
            "pass_rate": sum(r.eval_pass for r in val_records) / max(len(val_records), 1),
            "baseline_stats": {
                m: {
                    "pass_rate": s["pass"] / max(s["count"], 1),
                    "avg_quality": s["quality"] / max(s["count"], 1),
                    "avg_cost": s["cost"] / max(s["count"], 1),
                }
                for m, s in baseline_stats.items()
            },
        },
        "final_curve_table": final_table,
        "curve_snapshots": curve_snapshots,
    }
    summary_path = output_dir / f"{prefix}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n[Step 8] Saving results...")
    print(f"  Saved CSV: {full_csv}")
    print(f"  Saved val CSV: {val_csv}")
    print(f"  Saved summary: {summary_path}")


def _write_episode_csv(path: Path, records: List[EpisodeRecord]) -> None:
    """Write EpisodeRecord list to CSV."""
    if not records:
        return
    fields = [
        "episode", "task_id", "primitive_name", "difficulty_bucket",
        "selected_candidate", "predicted_quality", "predicted_cost",
        "observed_quality", "observed_cost", "eval_pass",
        "eval_level", "constraint_violations",
        "c_main", "c_llm", "c_total", "latency",
        "repair_action", "repair_delta_phi", "repair_delta_tau",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(asdict(r) if hasattr(r, "__dict__") else r)
