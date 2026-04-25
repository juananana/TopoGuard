"""
topology_template.py
====================
TopologyTemplate & TemplateLibrary — pluggable sub-graph templates.

Purpose (per exp.md interface 3):
- TemplateLibrary.get_templates_for(node_type, task_type) -> List[TopologyTemplate]
- TopologyTemplate.instantiate(template_id, config) -> List[SubTaskSpec]
- TemplateLibrary.pareto_frontier(...) -> template-level Pareto selection
- TemplateLibrary.select_from_frontier(...) -> constraint-based pick

Adding a new template = add one entry to the library,
NO changes to main experiment code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

from .primitive_profile import BucketStats


# ---------------------------------------------------------------------------
# Template definition
# ---------------------------------------------------------------------------

@dataclass
class TemplateNode:
    """
    A single node within a topology template.

    Attributes
    ----------
    node_id : str
        Unique ID within the template (e.g. "exec", "verify", "merge").
    node_type : str
        Role: "executor", "verifier", "aggregator", "hci", "cache".
        Determines which tool pool is queried.
    primitive_name : str
        Which primitive module to use (e.g. "forecast", "state_parse").
    optional : bool
        If True, this node may be pruned if not needed.
    """
    node_id: str
    node_type: str
    primitive_name: str
    optional: bool = False
    depends_on: List[str] = field(default_factory=list)


@dataclass
class TopologyTemplate:
    """
    A reusable sub-graph template for a node type / task type.

    Attributes
    ----------
    template_id : str
        Unique identifier, e.g. "direct", "exec_verify", "dual_exec_aggregate".
    description : str
        Human-readable description.
    supported_node_types : List[str]
        Which node_type(s) this template applies to.
    supported_task_types : List[str]
        Which task_type(s) this template applies to.
    nodes : List[TemplateNode]
        Ordered list of nodes in this template.
    estimated_latency : float
        Rough latency estimate (from executor + evaluator profiles).
    estimated_quality : float
        Rough quality estimate.
    """
    template_id: str
    description: str
    supported_node_types: List[str]
    supported_task_types: List[str]
    nodes: List[TemplateNode] = field(default_factory=list)
    estimated_cost: float = 1.0      # 估算执行成本（用于冷启动）
    estimated_latency: float = 0.0  # 估算执行延迟（秒）
    estimated_quality: float = 0.0  # 估算质量

    def instantiate(
        self,
        base_sub_task_id: str,
        base_primitive: str,
        base_difficulty: float,
        difficulty_bucket: str,
        constraints: List[Any] | None = None,
    ) -> List["SubTaskSpec"]:
        """
        Instantiate this template into a concrete list of SubTaskSpec nodes.

        Parameters
        ----------
        base_sub_task_id : str
            Prefix for generated sub_task_ids (e.g. "st_0").
        base_primitive : str
            Primitive to use for executor nodes (e.g. "forecast").
        base_difficulty : float
            Difficulty value [0, 1] for all nodes.
        difficulty_bucket : str
            Difficulty bucket for all nodes.
        constraints : List[ConstraintSpec] | None
            Constraints to attach to executor nodes.

        Returns
        -------
        List[SubTaskSpec]
            Concrete sub-task list ready for execution.
        """
        from src.decomposer.task_decomposer import (
            SubTaskSpec, ModalityType
        )
        from dataclasses import replace

        sub_tasks: List[SubTaskSpec] = []
        id_map: Dict[str, str] = {}  # template node_id -> actual sub_task_id

        for template_node in self.nodes:
            # Generate actual sub_task_id
            actual_id = f"{base_sub_task_id}_{template_node.node_id}"
            id_map[template_node.node_id] = actual_id

            # Map template node_type to primitive_name
            if template_node.node_type == "executor":
                prim = base_primitive
            elif template_node.node_type == "verifier":
                # Evaluator node — handled by evaluator selection logic separately
                prim = base_primitive
            elif template_node.node_type == "aggregator":
                prim = "aggregator"
            else:
                prim = template_node.primitive_name or base_primitive

            # Determine predecessor IDs from template
            predecessor_ids = [
                id_map[dep_id]
                for dep_id in template_node.depends_on
                if dep_id in id_map
            ]

            sub_tasks.append(SubTaskSpec(
                sub_task_id=actual_id,
                primitive_name=prim,
                difficulty=base_difficulty,
                difficulty_bucket=difficulty_bucket,
                description=f"[{self.template_id}] {template_node.node_type}: {prim}",
                predecessor_ids=predecessor_ids,
                metadata={"template_id": self.template_id, "node_type": template_node.node_type},
                constraints=constraints or [],
                input_modality=ModalityType.TEXT,
                intermediate_modality=None,
            ))

        return sub_tasks


# ---------------------------------------------------------------------------
# Template library
# ---------------------------------------------------------------------------

# The minimum template set (exp.md Section 6, item 6)
DEFAULT_TEMPLATES: List[TopologyTemplate] = [
    # 1. direct: single executor
    TopologyTemplate(
        template_id="direct",
        description="Single executor, no verification",
        supported_node_types=["forecast", "state_parse", "data_analysis"],
        supported_task_types=["time_series", "text_analysis", "tabular_analysis", "multimodal"],
        nodes=[
            TemplateNode(node_id="exec", node_type="executor", primitive_name="", depends_on=[]),
        ],
        estimated_cost=0.5,
        estimated_latency=1.0,
        estimated_quality=0.8,
    ),

    # 2. exec_verify: executor + verifier
    TopologyTemplate(
        template_id="exec_verify",
        description="Executor with post-verification",
        supported_node_types=["forecast", "state_parse", "data_analysis"],
        supported_task_types=["time_series", "text_analysis", "tabular_analysis"],
        nodes=[
            TemplateNode(node_id="exec", node_type="executor", primitive_name="", depends_on=[]),
            TemplateNode(node_id="verify", node_type="verifier", primitive_name="", depends_on=["exec"]),
        ],
        estimated_cost=0.8,
        estimated_latency=1.5,
        estimated_quality=0.88,
    ),

    # 3. dual_exec_aggregate: parallel branches + aggregator
    TopologyTemplate(
        template_id="dual_exec_aggregate",
        description="Parallel dual-executor then aggregate",
        supported_node_types=["forecast", "state_parse", "data_analysis"],
        supported_task_types=["time_series", "text_analysis", "tabular_analysis", "multimodal"],
        nodes=[
            TemplateNode(node_id="exec_a", node_type="executor", primitive_name="", depends_on=[]),
            TemplateNode(node_id="exec_b", node_type="executor", primitive_name="", depends_on=[]),
            TemplateNode(node_id="merge", node_type="aggregator", primitive_name="aggregator", depends_on=["exec_a", "exec_b"]),
        ],
        estimated_cost=1.5,
        estimated_latency=2.0,
        estimated_quality=0.92,
    ),

    # 4. exec_verify_hci: executor + verifier + human escalation
    TopologyTemplate(
        template_id="exec_verify_hci",
        description="Executor + verifier + human-in-the-loop on failure",
        supported_node_types=["forecast", "state_parse", "data_analysis"],
        supported_task_types=["time_series", "text_analysis", "tabular_analysis"],
        nodes=[
            TemplateNode(node_id="exec", node_type="executor", primitive_name="", depends_on=[]),
            TemplateNode(node_id="verify", node_type="verifier", primitive_name="", depends_on=["exec"]),
        ],
        estimated_cost=2.0,
        estimated_latency=5.0,
        estimated_quality=0.95,
    ),

    # 5. bad_direct: dominated single-executor (quality_mult=0.70, cost_mult=1.30 vs direct)
    # Intentionally dominated by "direct" on both quality and cost — exists so Pareto
    # pruning has a concrete dominated candidate to filter out, validating the mechanism.
    TopologyTemplate(
        template_id="bad_direct",
        description="Degraded single executor — dominated candidate for Pareto validation",
        supported_node_types=["forecast", "state_parse", "data_analysis"],
        supported_task_types=["time_series", "text_analysis", "tabular_analysis", "multimodal"],
        nodes=[
            TemplateNode(node_id="exec", node_type="executor", primitive_name="", depends_on=[]),
        ],
        estimated_cost=0.65,    # 1.30 × direct baseline cost 0.5
        estimated_latency=1.0,
        estimated_quality=0.56, # 0.70 × direct baseline quality 0.8
    ),
]


# ---------------------------------------------------------------------------
# Template Profile (per-template performance data)
# ---------------------------------------------------------------------------

@dataclass
class TemplateProfile:
    """
    Performance profile for a single template, analogous to CandidateProfile
    for executors.

    Stores per-difficulty-bucket (quality, cost) observations.

    Attributes
    ----------
    template_id : str
        Matches TopologyTemplate.template_id.
    bucket_stats : Dict[str, BucketStats]
        Per-difficulty-bucket statistics.
    """
    template_id: str
    bucket_stats: Dict[str, BucketStats] = field(default_factory=dict)

    def get_bucket(self, bucket_name: str) -> BucketStats:
        if bucket_name not in self.bucket_stats:
            self.bucket_stats[bucket_name] = BucketStats(bucket_name=bucket_name)
        return self.bucket_stats[bucket_name]

    def total_support(self) -> int:
        return sum(s.support_count for s in self.bucket_stats.values())


class TemplateLibrary:
    """
    Repository of topology templates with performance profile management.

    Supports:
    - Template lookup by node_type / task_type
    - Template profile loading from JSONL
    - Pareto frontier computation over templates
    - Constraint-based selection from frontier
    - Feedback ingestion for online profile updates

    Usage:
        library = TemplateLibrary()
        library.load_profiles_from_jsonl("data/template_profiles.jsonl")
        frontier = library.pareto_frontier("forecast", "time_series", "hard")
        selected = library.select_from_frontier(frontier, cost_budget=2.0)
    """

    def __init__(
        self,
        templates: List[TopologyTemplate] | None = None,
        template_profiles: Dict[str, TemplateProfile] | None = None,
    ):
        self._templates = templates or DEFAULT_TEMPLATES
        self._template_profiles: Dict[str, TemplateProfile] = template_profiles or {}

    def get_templates_for(
        self,
        node_type: str,
        task_type: str | None = None,
    ) -> List[TopologyTemplate]:
        """Return all templates applicable to a node_type (and optionally task_type)."""
        results = []
        for t in self._templates:
            if node_type in t.supported_node_types:
                if task_type is None or task_type in t.supported_task_types:
                    results.append(t)
        return results

    def get_template(self, template_id: str) -> TopologyTemplate | None:
        """Get a specific template by ID."""
        for t in self._templates:
            if t.template_id == template_id:
                return t
        return None

    def register(self, template: TopologyTemplate) -> None:
        """Add a new template to the library."""
        if any(t.template_id == template.template_id for t in self._templates):
            raise ValueError(f"Template with id '{template.template_id}' already exists.")
        self._templates.append(template)

    # -------------------------------------------------------------------------
    # Template Profile Management
    # -------------------------------------------------------------------------

    def load_profiles_from_jsonl(self, path: str | Path) -> int:
        """
        Load template profiles from a JSONL file.

        Each line: {"template_id": "...", "difficulty": "...",
                     "quality_mean": float, "cost_mean": float,
                     "latency_mean": float (optional, default 0.0)}

        Returns number of records loaded.
        """
        path = Path(path)
        if not path.exists():
            return 0

        count = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                tid = row["template_id"]
                difficulty = row["difficulty"]
                quality = float(row["quality_mean"])
                cost = float(row["cost_mean"])
                latency = float(row.get("latency_mean", 0.0))

                if tid not in self._template_profiles:
                    self._template_profiles[tid] = TemplateProfile(template_id=tid)
                profile = self._template_profiles[tid]
                bucket = profile.get_bucket(difficulty)
                bucket.set_prior(quality, cost, latency)
                count += 1

        return count

    def get_profile(self, template_id: str) -> TemplateProfile | None:
        return self._template_profiles.get(template_id)

    # -------------------------------------------------------------------------
    # Pareto Frontier (Template-level)
    # -------------------------------------------------------------------------

    def pareto_frontier(
        self,
        node_type: str,
        task_type: str | None,
        difficulty_bucket: str,
    ) -> List[dict]:
        """
        Compute 3D Pareto frontier over applicable templates for a given difficulty.
        Maximizes S (quality), minimizes C (cost) and L (latency).

        Returns list of dicts with keys:
            template, template_id, pred_quality, pred_cost, pred_latency,
            uncertainty, support_count, source
        """
        templates = self.get_templates_for(node_type, task_type)
        if not templates:
            return []

        data = []
        for t in templates:
            profile = self._template_profiles.get(t.template_id)
            if profile and difficulty_bucket in profile.bucket_stats:
                stats = profile.bucket_stats[difficulty_bucket]
                data.append({
                    "template": t,
                    "template_id": t.template_id,
                    "pred_quality": stats.quality_mean,
                    "pred_cost": stats.cost_mean,
                    "pred_latency": stats.latency_mean,
                    "uncertainty": stats.uncertainty,
                    "support_count": stats.support_count,
                    "source": "profile",
                })
            else:
                # Cold-start fallback: use static estimates from template definition
                data.append({
                    "template": t,
                    "template_id": t.template_id,
                    "pred_quality": t.estimated_quality,
                    "pred_cost": t.estimated_cost,
                    "pred_latency": t.estimated_latency,
                    "uncertainty": 1.0,
                    "support_count": 0,
                    "source": "static_estimate",
                })

        if len(data) <= 1:
            return data

        from paretoset import paretoset as compute_pareto

        # 3D Pareto: maximize S (quality), minimize C (cost) and L (latency)
        qualities = np.array([d["pred_quality"] for d in data])
        costs = np.array([d["pred_cost"] for d in data])
        latencies = np.array([d["pred_latency"] for d in data])
        matrix = np.column_stack([qualities, costs, latencies])
        mask = compute_pareto(matrix, sense=["max", "min", "min"])
        return [d for d, m in zip(data, mask) if m]

    def select_from_frontier(
        self,
        frontier: List[dict],
        acc_target: float | None = None,
        cost_budget: float | None = None,
        latency_budget: float | None = None,
        alpha: float = 0.65,
        beta: float = 0.25,
        gamma: float = 0.10,
        s_scale: float = 1.5,
    ) -> dict:
        """
        Select a single template from the Pareto frontier.

        Uses Q(G;X) = α*(S/s_scale) - β*C_norm - γ*L_norm (scale-balanced 3D).
        Weights match paper §3.2: α=0.65, β=0.25, γ=0.10, s_scale=1.5.

        S is divided by s_scale so it competes on the same scale as C_norm/L_norm
        (already in [0,1] after log-normalization). Without this, the formula
        degenerates to argmax S regardless of cost/latency weights.

        Parameters
        ----------
        frontier : List[dict]
            Output from pareto_frontier(), each dict contains pred_quality/pred_cost/pred_latency.
        acc_target : float | None
            Hard filter: template must have pred_quality >= acc_target.
        cost_budget : float | None
            Hard filter: template must have pred_cost <= cost_budget.
        latency_budget : float | None
            Hard filter: template must have pred_latency <= latency_budget.
        alpha : float
            Weight for quality S (default 0.65).
        beta : float
            Weight for cost C (default 0.25).
        gamma : float
            Weight for latency L (default 0.10).
        s_scale : float
            Normalization divisor for S (default 1.5).
        """
        if not frontier:
            raise ValueError("Empty template Pareto frontier.")

        filtered = frontier

        if cost_budget is not None:
            filtered = [d for d in filtered if d["pred_cost"] <= cost_budget]

        if acc_target is not None:
            filtered = [d for d in filtered if d["pred_quality"] >= acc_target]

        if latency_budget is not None:
            filtered = [d for d in filtered if d["pred_latency"] <= latency_budget]

        if not filtered:
            raise ValueError(
                f"No templates satisfy hard constraints: "
                f"acc_target={acc_target}, cost_budget={cost_budget}, "
                f"latency_budget={latency_budget}. "
                f"Pareto frontier has {len(frontier)} templates."
            )

        total = alpha + beta + gamma
        a, b, g = alpha / total, beta / total, gamma / total

        def _q_score(d: dict) -> float:
            S_norm = d.get("pred_quality", 0.0) / s_scale
            C_norm = d.get("pred_cost_norm", d.get("pred_cost", 0.0))
            L_norm = d.get("pred_latency_norm", d.get("pred_latency", 0.0))
            return a * S_norm - b * C_norm - g * L_norm

        return max(filtered, key=_q_score)

    # -------------------------------------------------------------------------
    # Feedback
    # -------------------------------------------------------------------------

    def add_feedback(
        self,
        template_id: str,
        difficulty_bucket: str,
        observed_quality: float,
        observed_cost: float,
        observed_latency: float = 0.0,
    ) -> None:
        """
        Append a template-level observation from an episode result.
        Records (quality, cost, latency) per bucket for 3D Pareto + Q(G;X) optimization.

        Parameters
        ----------
        template_id : str
        difficulty_bucket : str
        observed_quality : float
            Episode-level quality (e.g. pass=1.0, fail=0.0).
        observed_cost : float
            Episode-level total cost.
        observed_latency : float
            Episode-level total latency (seconds).
        """
        if template_id not in self._template_profiles:
            self._template_profiles[template_id] = TemplateProfile(template_id=template_id)
        profile = self._template_profiles[template_id]
        bucket = profile.get_bucket(difficulty_bucket)
        bucket.add_observation(observed_quality, observed_cost, observed_latency)

    # -------------------------------------------------------------------------
    # Legacy scoring (kept for backward compatibility, delegates to Pareto)
    # -------------------------------------------------------------------------

    def score_templates(
        self,
        templates: List[TopologyTemplate],
        difficulty: str,
        remaining_budget: float | None = None,
        constraints: List[Any] | None = None,
    ) -> List[tuple[TopologyTemplate, float]]:
        """
        Score templates by estimated quality per unit cost, with difficulty
        and constraint awareness for structure-configuration joint optimization.

        NOTE: This is a legacy method. Prefer pareto_frontier() + select_from_frontier()
        for new code. Kept for backward compatibility with mvp_experiment.py.
        """
        # Lazy import to avoid circular dependency
        from src.decomposer.task_decomposer import (
            HumanInTheLoopConstraint,
            RiskBoundaryConstraint,
        )

        has_hitl = False
        has_risk = False
        if constraints:
            for c in constraints:
                if isinstance(c, HumanInTheLoopConstraint):
                    has_hitl = True
                elif isinstance(c, RiskBoundaryConstraint):
                    has_risk = True

        scored = []
        for t in templates:
            # Use profile data if available, otherwise static estimates
            profile = self._template_profiles.get(t.template_id)
            if profile and difficulty in profile.bucket_stats:
                stats = profile.bucket_stats[difficulty]
                est_quality = stats.quality_mean
                est_cost = stats.cost_mean
            else:
                est_quality = t.estimated_quality
                est_cost = t.estimated_cost

            if remaining_budget is not None and est_cost > remaining_budget:
                score = 0.0
            else:
                base_score = est_quality / max(est_cost, 0.01)

                # --- Difficulty multiplier ---
                diff_mult = 1.0
                if difficulty in ("hard", "extreme"):
                    if est_quality >= 0.88:
                        diff_mult = 1.10
                elif difficulty == "easy":
                    if est_cost <= 1.0:
                        diff_mult = 1.08

                # --- Constraint multiplier ---
                cons_mult = 1.0
                if has_hitl and t.template_id == "exec_verify_hci":
                    cons_mult *= 1.15
                if has_risk and est_quality >= 0.88:
                    cons_mult *= 1.10

                score = base_score * diff_mult * cons_mult

            scored.append((t, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
