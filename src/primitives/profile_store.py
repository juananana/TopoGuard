"""
profile_store.py
==================
External profile data interface for TopoGuard.

Responsibilities:
1. Load executor / evaluator profiles from external JSONL files
2. Provide a unified lookup interface
3. Support profile update (increment sample_count, update means/stds)

Profile files:
- data/executor_profiles.jsonl   — ground truth ACC/Cost per (tool, difficulty)
- data/evaluator_profiles.jsonl  — evaluator precision/recall per (tool, difficulty)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
import os


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExecutorProfile:
    """
    Profile for a single executor tool at a given difficulty.

    Cost Model (per Method §4.3):
        c_main = cost_a1_per_mtok * input_tokens / 1_000_000
               + cost_a2_per_mtok * output_tokens / 1_000_000

    Fields:
        api_cost_mean : float
            Legacy scalar — mean of raw USD costs across collected calls.
            Kept for backward compatibility. For accurate cost estimation,
            use the linear model fields below.
        cost_a1_per_mtok : float
            Input-token price coefficient in USD per million tokens.
            E.g. 0.8 means $0.80 per 1M input tokens.
        cost_a2_per_mtok : float
            Output-token price coefficient in USD per million tokens.
            E.g. 1.6 means $1.60 per 1M output tokens.
        typical_input_tokens : int
            Typical input token count for this (task_type, difficulty).
            Used to estimate cost for new tasks of the same bucket.
        typical_output_tokens : int
            Typical output token count for this (task_type, difficulty).
        latency_a1 : float
            Latency coefficient: seconds per input token.
        latency_a2 : float
            Latency coefficient: seconds per output token.
        latency_mean : float
            Legacy scalar — mean latency in seconds. Still used as fallback.
    """
    tool_id: str
    task_type: str
    node_type: str
    difficulty: str
    quality_mean: float
    quality_std: float
    latency_mean: float
    api_cost_mean: float
    human_cost_mean: float = 0.0
    sample_count: int = 0
    # === Linear cost model fields (per §4.3) ===
    cost_a1_per_mtok: float = 0.0   # USD per million input tokens
    cost_a2_per_mtok: float = 0.0   # USD per million output tokens
    typical_input_tokens: int = 0
    typical_output_tokens: int = 0
    latency_a1: float = 0.0         # seconds per input token
    latency_a2: float = 0.0         # seconds per output token


@dataclass
class EvaluatorProfile:
    """Profile for a single evaluator tool at a given difficulty."""
    tool_id: str
    task_type: str
    node_type: str
    difficulty: str
    precision: float
    recall: float
    false_pass_rate: float
    false_reject_rate: float
    latency_mean: float
    api_cost_mean: float
    human_cost_mean: float = 0.0
    sample_count: int = 0


# ---------------------------------------------------------------------------
# ProfileStore
# ---------------------------------------------------------------------------

class ProfileStore:
    """
    Unified interface for executor and evaluator profile data.

    Can load from external JSONL files or use embedded defaults.
    Adding a new tool = adding one line to the JSONL file,
    NO changes to any other code.

    Lookup key: (tool_id, difficulty)
    """

    def __init__(
        self,
        executor_profiles_path: str | Path | None = None,
        evaluator_profiles_path: str | Path | None = None,
    ):
        """
        Parameters
        ----------
        executor_profiles_path : str | Path | None
            Path to executor_profiles.jsonl. If None, uses bundled default.
        evaluator_profiles_path : str | Path | None
            Path to evaluator_profiles.jsonl. If None, uses bundled default.
        """
        # Resolve data directory: prefer explicit path, else data/ relative to repo root
        repo_root = Path(__file__).parent.parent.parent
        default_data = repo_root / "data"

        self._executor_profiles: Dict[str, Dict[str, ExecutorProfile]] = {}
        self._evaluator_profiles: Dict[str, Dict[str, EvaluatorProfile]] = {}
        # Workflow-level profiles: keyed by (template_id, scenario, task_type)
        self._workflow_profiles: Dict[str, "WorkflowProfile"] = {}

        # Load executor profiles
        if executor_profiles_path:
            self._load_executor_profiles(Path(executor_profiles_path))
        else:
            default_path = default_data / "executor_profiles.jsonl"
            if default_path.exists():
                self._load_executor_profiles(default_path)
            # else: will use DEFAULT_GROUND_TRUTH fallback in MockEvaluator

        # Load evaluator profiles
        if evaluator_profiles_path:
            self._load_evaluator_profiles(Path(evaluator_profiles_path))
        else:
            default_path = default_data / "evaluator_profiles.jsonl"
            if default_path.exists():
                self._load_evaluator_profiles(default_path)

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    def _load_executor_profiles(self, path: Path) -> None:
        """Load executor profiles from a JSONL file."""
        if not path.exists():
            return
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                p = ExecutorProfile(**row)
                key = f"{p.tool_id}/{p.difficulty}"
                self._executor_profiles[key] = p

    def _load_evaluator_profiles(self, path: Path) -> None:
        """Load evaluator profiles from a JSONL file."""
        if not path.exists():
            return
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                p = EvaluatorProfile(**row)
                key = f"{p.tool_id}/{p.difficulty}"
                self._evaluator_profiles[key] = p

    # -------------------------------------------------------------------------
    # Query
    # -------------------------------------------------------------------------

    def get_executor_profile(
        self,
        tool_id: str,
        difficulty: str,
    ) -> ExecutorProfile | None:
        """Get executor profile for (tool_id, difficulty)."""
        return self._executor_profiles.get(f"{tool_id}/{difficulty}")

    def get_evaluator_profile(
        self,
        tool_id: str,
        difficulty: str,
    ) -> EvaluatorProfile | None:
        """Get evaluator profile for (tool_id, difficulty)."""
        return self._evaluator_profiles.get(f"{tool_id}/{difficulty}")

    def get_executor_quality_cost(
        self,
        primitive_name: str,
        candidate_name: str,
        difficulty: str,
    ) -> tuple[float, float] | None:
        """
        Get (quality_mean, api_cost_mean) for an executor.
        Returns None if not found in the store.
        """
        tool_id = f"{primitive_name}/{candidate_name}"
        prof = self.get_executor_profile(tool_id, difficulty)
        if prof is None:
            return None
        return prof.quality_mean, prof.api_cost_mean

    def list_executors_for(
        self,
        node_type: str,
        difficulty: str,
    ) -> List[ExecutorProfile]:
        """List all executor profiles for a given node_type and difficulty."""
        return [
            p for p in self._executor_profiles.values()
            if p.node_type == node_type and p.difficulty == difficulty
        ]

    def list_evaluators_for(
        self,
        node_type: str,
        difficulty: str,
    ) -> List[EvaluatorProfile]:
        """List all evaluator profiles for a given node_type and difficulty."""
        return [
            p for p in self._evaluator_profiles.values()
            if p.node_type == node_type and p.difficulty == difficulty
        ]

    # -------------------------------------------------------------------------
    # Update (from feedback records)
    # -------------------------------------------------------------------------

    def update_executor_profile(
        self,
        tool_id: str,
        difficulty: str,
        observed_quality: float,
        observed_cost: float,
        observed_latency: float | None = None,
        alpha: float = 0.3,
    ) -> None:
        """
        Update executor profile from a single observation using running mean.

        This is called by WorkflowExecutor after a real execution to refine
        the initial profile from observed data.

        Parameters
        ----------
        tool_id : str
            Format: "primitive/candidate" e.g. "forecast/fast_nn"
        difficulty : str
            Bucket name: "easy" | "medium" | "hard" | "extreme"
        observed_quality : float
            Observed output quality [0, 1].
        observed_cost : float
            Observed API cost.
        observed_latency : float | None
            Observed execution latency in seconds. If None, uses observed_cost as proxy.
        alpha : float
            Legacy parameter (kept for API compatibility, uses running mean instead).
        """
        key = f"{tool_id}/{difficulty}"
        if key not in self._executor_profiles:
            # New profile discovered at runtime — create one
            parts = tool_id.split("/")
            primitive = parts[0] if len(parts) > 0 else "unknown"
            self._executor_profiles[key] = ExecutorProfile(
                tool_id=tool_id,
                task_type="unknown",
                node_type=primitive,
                difficulty=difficulty,
                quality_mean=observed_quality,
                quality_std=0.0,
                latency_mean=observed_latency if observed_latency is not None else observed_cost,
                api_cost_mean=observed_cost,
                human_cost_mean=0.0,
                sample_count=1,
            )
            return

        p = self._executor_profiles[key]
        p.sample_count += 1
        n = p.sample_count
        # Running mean update for quality and cost
        p.quality_mean += (observed_quality - p.quality_mean) / n
        p.api_cost_mean += (observed_cost - p.api_cost_mean) / n
        # Running mean update for latency
        if observed_latency is not None:
            p.latency_mean += (observed_latency - p.latency_mean) / n

    def update_evaluator_profile(
        self,
        evaluator_id: str,
        difficulty: str,
        observed_pass: bool,
        true_pass: bool,
        evaluator_latency: float,
        evaluator_cost: float,
        alpha: float = 0.3,
    ) -> None:
        """
        Update evaluator profile from a single observation using running mean.

        Parameters
        ----------
        evaluator_id : str
        difficulty : str
        observed_pass : bool
        true_pass : bool
        evaluator_latency : float
        evaluator_cost : float
        alpha : float
            Legacy parameter (kept for API compatibility).
        """
        key = f"{evaluator_id}/{difficulty}"
        if key not in self._evaluator_profiles:
            self._evaluator_profiles[key] = EvaluatorProfile(
                tool_id=evaluator_id,
                task_type="unknown",
                node_type="unknown",
                difficulty=difficulty,
                precision=0.70,     # initial guess
                recall=0.70,
                false_pass_rate=0.30,
                false_reject_rate=0.30,
                latency_mean=evaluator_latency,
                api_cost_mean=evaluator_cost,
                sample_count=1,
            )
            return

        p = self._evaluator_profiles[key]
        p.sample_count += 1
        n = p.sample_count
        # Running mean for precision (correct judgment rate)
        correct = 1.0 if observed_pass == true_pass else 0.0
        p.precision += (correct - p.precision) / n
        # Track recall and error rates
        if observed_pass and true_pass:
            p.recall += (1.0 - p.recall) / n
        elif observed_pass and not true_pass:
            p.false_pass_rate += (1.0 - p.false_pass_rate) / n
        elif not observed_pass and true_pass:
            p.false_reject_rate += (1.0 - p.false_reject_rate) / n
        p.latency_mean += (evaluator_latency - p.latency_mean) / n
        p.api_cost_mean += (evaluator_cost - p.api_cost_mean) / n

    # -------------------------------------------------------------------------
    # Workflow Profile management
    # -------------------------------------------------------------------------

    def update_workflow_profile(
        self,
        template_id: str,
        scenario: str,
        task_type: str,
        overall_pass: bool,
        total_cost: float,
        total_latency: float,
        violation_count: int,
        node_count: int,
        repair_count: int,
    ) -> None:
        """
        Update workflow-level profile from a single execution result.

        Records aggregate statistics for a workflow template across multiple
        executions. Uses running mean EMA for all fields.

        Parameters
        ----------
        template_id : str
            e.g. "direct", "exec_verify", "dual_exec_aggregate"
        scenario : str
            e.g. "Normal", "Hard", "OOD"
        task_type : str
            e.g. "time_series", "text_analysis", "tabular_analysis"
        overall_pass : bool
            Whether the workflow produced a pass-grade output.
        total_cost : float
            Total executor + evaluator cost for this execution.
        total_latency : float
            Total executor latency (ignoring evaluator latency).
        violation_count : int
            Number of constraint violations in this execution.
        node_count : int
            Total number of nodes in the workflow.
        repair_count : int
            Number of nodes that required repair during execution.
        """
        key = f"{template_id}|{scenario}|{task_type}"
        if key not in self._workflow_profiles:
            # Import WorkflowProfile dataclass from the workflow module
            # Using full path to avoid cross-package import issues at module-load time
            try:
                from src.workflow.workflow_graph import WorkflowProfile as _WF
            except ImportError:
                from workflow.workflow_graph import WorkflowProfile as _WF
            self._workflow_profiles[key] = _WF(
                template_id=template_id,
                scenario=scenario,
                task_type=task_type,
            )

        wp = self._workflow_profiles[key]
        n = wp.support_count + 1
        quality = 1.0 if overall_pass else 0.0

        # Running mean + Welford std update (mean must be OLD mean for Welford formula)
        old_acc_mean = wp.acc_mean
        wp.acc_mean += (quality - wp.acc_mean) / n
        wp.acc_std = self._running_std(wp.acc_std, old_acc_mean, quality, n)

        old_cost_mean = wp.cost_mean
        wp.cost_mean += (total_cost - wp.cost_mean) / n
        wp.cost_std = self._running_std(wp.cost_std, old_cost_mean, total_cost, n)

        old_lat_mean = wp.latency_mean
        wp.latency_mean += (total_latency - wp.latency_mean) / n
        wp.latency_std = self._running_std(wp.latency_std, old_lat_mean, total_latency, n)

        # Running mean update for violation rate
        viol_rate = violation_count / max(node_count, 1)
        wp.violation_rate_mean += (viol_rate - wp.violation_rate_mean) / n

        # Running mean update for repair rate
        repair_rate = repair_count / max(node_count, 1)
        wp.repair_rate_mean += (repair_rate - wp.repair_rate_mean) / n

        wp.support_count = n

    @staticmethod
    def _running_std(current_std: float, current_mean: float, new_val: float, n: int) -> float:
        """
        Update running standard deviation using Welford's algorithm.
        Returns std = sqrt(variance).  We store variance internally to avoid overflow
        from squaring a large std value.  Clamp variance to [0, 1e6] as a safety net.
        """
        if n < 2:
            return 0.0
        # Welford online variance formula (uses OLD mean for the deviation term)
        new_var = ((n - 2) / (n - 1)) * (current_std ** 2) + (new_val - current_mean) ** 2 / n
        new_var = max(0.0, min(new_var, 1e6))  # clamp to prevent overflow
        import math
        return math.sqrt(new_var)

    def get_workflow_profile(
        self,
        template_id: str,
        scenario: str = "Normal",
        task_type: str = "unknown",
    ) -> "WorkflowProfile | None":
        """
        Get workflow profile by template_id + scenario + task_type.
        Falls back to fewer dimensions if exact match not found.
        """
        # Try exact first
        key = f"{template_id}|{scenario}|{task_type}"
        if key in self._workflow_profiles:
            return self._workflow_profiles[key]

        # Fallback: ignore task_type
        for k, wp in self._workflow_profiles.items():
            if k.startswith(f"{template_id}|{scenario}|"):
                return wp

        # Fallback: ignore scenario too
        for k, wp in self._workflow_profiles.items():
            if k.startswith(f"{template_id}|"):
                return wp

        return None

    def list_workflow_profiles(
        self,
        scenario: str | None = None,
        task_type: str | None = None,
    ) -> List["WorkflowProfile"]:
        """List all workflow profiles, optionally filtered by scenario / task_type."""
        results = list(self._workflow_profiles.values())
        if scenario is not None:
            results = [r for r in results if r.scenario == scenario]
        if task_type is not None:
            results = [r for r in results if r.task_type == task_type]
        return results

    # -------------------------------------------------------------------------
    # Four-key lookup (task_type, node_type, tool_id, difficulty)
    # -------------------------------------------------------------------------

    def get_executor_profile_full(
        self,
        tool_id: str,
        node_type: str | None,
        task_type: str | None,
        difficulty: str,
    ) -> ExecutorProfile | None:
        """
        Get executor profile by full four-key: (tool_id, node_type, task_type, difficulty).

        查询优先级：
        1. 精确匹配 (tool_id, node_type, task_type, difficulty)
        2. 忽略 task_type：匹配 (tool_id, node_type, difficulty)
        3. 忽略 node_type：匹配 (tool_id, task_type, difficulty)
        4. 忽略两者：匹配 (tool_id, difficulty) —— 退化为两键查询

        Parameters
        ----------
        tool_id : str
            Format: "primitive/candidate" e.g. "forecast/fast_nn"
        node_type : str | None
            Node type, e.g. "forecast". None = skip this dimension.
        task_type : str | None
            Task type, e.g. "time_series". None = skip this dimension.
        difficulty : str
            Bucket name: "easy" | "medium" | "hard" | "extreme"

        Returns
        -------
        ExecutorProfile | None
        """
        # Try exact four-key first
        for nt in [node_type, None]:
            for tt in [task_type, None]:
                key_parts = [tool_id, difficulty]
                if nt is not None:
                    key_parts.insert(0, nt)
                if tt is not None:
                    key_parts.insert(1, tt)
                key = "/".join(key_parts)
                # The stored key is always tool_id/difficulty; we need to scan
                # because we don't know how many dimensions were used at storage time.
                # Strategy: scan all entries and match on fields.
                break

        # Scan all profiles for exact/partial match
        # Fallback chain: full → no_task_type → no_node_type → two-key
        candidates: list[ExecutorProfile] = []
        for p in self._executor_profiles.values():
            if p.tool_id != tool_id or p.difficulty != difficulty:
                continue
            candidates.append(p)

        if not candidates:
            return None

        # Priority: exact match on both optional dims
        for p in candidates:
            if (node_type is None or p.node_type == node_type) and \
               (task_type is None or p.task_type == task_type):
                return p

        # Fallback: match node_type only
        if node_type is not None:
            for p in candidates:
                if p.node_type == node_type:
                    return p

        # Fallback: match task_type only
        if task_type is not None:
            for p in candidates:
                if p.task_type == task_type:
                    return p

        # Final fallback: any entry with this tool_id/difficulty
        return candidates[0]

    def get_evaluator_profile_full(
        self,
        tool_id: str,
        node_type: str | None,
        task_type: str | None,
        difficulty: str,
    ) -> EvaluatorProfile | None:
        """
        Get evaluator profile by full four-key (same fallback chain as
        get_executor_profile_full).
        """
        candidates = [
            p for p in self._evaluator_profiles.values()
            if p.tool_id == tool_id and p.difficulty == difficulty
        ]
        if not candidates:
            return None

        for p in candidates:
            if (node_type is None or p.node_type == node_type) and \
               (task_type is None or p.task_type == task_type):
                return p

        if node_type is not None:
            for p in candidates:
                if p.node_type == node_type:
                    return p

        if task_type is not None:
            for p in candidates:
                if p.task_type == task_type:
                    return p

        return candidates[0]

    # -------------------------------------------------------------------------
    # init_curve bridge: from external JSONL to ProfileManager.register_candidate
    # -------------------------------------------------------------------------

    def get_init_curve_for(
        self,
        primitive_name: str,
        candidate_name: str,
        node_type: str | None = None,
        difficulty_mapper: "DifficultyMapper | None" = None,
    ) -> dict | None:
        """
        Build an init_curve dict from external executor profile data.

        This bridges external JSONL data → ProfileManager.register_candidate(init_curve=...).
        The returned dict has the format:
            {
                "easy":    {"acc_mean": float, "cost_mean": float},
                "medium":  {"acc_mean": float, "cost_mean": float},
                "hard":    {"acc_mean": float, "cost_mean": float},
                "extreme": {"acc_mean": float, "cost_mean": float},
            }

        Per Method §4: profiles are stored per (tool_id, node_type, difficulty).
        If node_type is provided, only profiles matching that node_type are used.
        If node_type is None, profiles across all node_types are averaged.

        Parameters
        ----------
        primitive_name : str
            e.g. "water_qa"
        candidate_name : str
            e.g. "deepseek_v3"
        node_type : str | None
            Node type to filter by (e.g. "CALCULATION", "REASONING", "RETRIEVAL", "SIMPLE").
            If None, averages across all node_types for this (primitive, candidate).
        difficulty_mapper : DifficultyMapper | None
            Optional, for validating bucket names.

        Returns
        -------
        dict | None
            Init curve dict, or None if no external profile data exists for this
            (primitive, candidate) pair.
        """
        # Build the tool_id from primitive_name + candidate_name
        tool_id = f"{primitive_name}/{candidate_name}"

        # Collect all profiles matching this tool_id, grouped by difficulty bucket
        # If node_type is provided, filter by it; otherwise average across all node_types
        from collections import defaultdict
        bucket_groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for p in self._executor_profiles.values():
            if p.tool_id != tool_id:
                continue
            if node_type is not None and p.node_type != node_type:
                continue
            bucket_groups[p.difficulty].append((p.quality_mean, p.api_cost_mean))

        if not bucket_groups:
            return None

        init_curve: dict[str, dict[str, float]] = {}
        for bucket, observations in bucket_groups.items():
            n = len(observations)
            avg_quality = sum(q for q, _ in observations) / n
            avg_cost = sum(c for _, c in observations) / n
            init_curve[bucket] = {
                "acc_mean": round(avg_quality, 4),
                "cost_mean": round(avg_cost, 4),
            }

        # Validate bucket names if mapper provided
        if difficulty_mapper is not None:
            valid = set(difficulty_mapper.bucket_names_list())
            init_curve = {k: v for k, v in init_curve.items() if k in valid}

        return init_curve if init_curve else None

    def save(self, data_dir: str | Path) -> None:
        """
        Persist all three profile types back to JSONL files.
        Call this at the end of an experiment to save updated profiles.

        Saves:
        - executor_profiles.jsonl   — ExecutorProfile records
        - evaluator_profiles.jsonl  — EvaluatorProfile records
        - workflow_profiles.jsonl   — WorkflowProfile records
        """
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        exec_path = data_dir / "executor_profiles.jsonl"
        with open(exec_path, "w", encoding="utf-8") as f:
            for p in self._executor_profiles.values():
                f.write(json.dumps(p.__dict__) + "\n")

        eval_path = data_dir / "evaluator_profiles.jsonl"
        with open(eval_path, "w", encoding="utf-8") as f:
            for p in self._evaluator_profiles.values():
                f.write(json.dumps(p.__dict__) + "\n")

        wf_path = data_dir / "workflow_profiles.jsonl"
        with open(wf_path, "w", encoding="utf-8") as f:
            for wp in self._workflow_profiles.values():
                f.write(json.dumps({
                    "template_id": wp.template_id,
                    "scenario": wp.scenario,
                    "task_type": wp.task_type,
                    "acc_mean": wp.acc_mean,
                    "acc_std": wp.acc_std,
                    "cost_mean": wp.cost_mean,
                    "cost_std": wp.cost_std,
                    "latency_mean": wp.latency_mean,
                    "latency_std": wp.latency_std,
                    "violation_rate_mean": wp.violation_rate_mean,
                    "repair_rate_mean": wp.repair_rate_mean,
                    "support_count": wp.support_count,
                }) + "\n")
