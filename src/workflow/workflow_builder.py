"""
workflow_builder.py
===================
Builds WorkflowGraph from TopologyTemplate, SubTaskSpec, or raw task description.

This is the bridge between:
1. The existing TopologyTemplate / SubTaskSpec layer (backward compat)
2. The new explicit WorkflowGraph layer (method definition alignment)

Usage:
    builder = WorkflowBuilder(primitive_name="forecast", difficulty="hard", ...)
    graph = builder.from_template("exec_verify")
    graph = builder.from_subtask_specs(subtask_list)
    graph = builder.from_task_description("Forecast next 30 days temperature...")

Profile integration:
    estimate_from_profile_store() fills estimated_quality / estimated_cost / estimated_latency
    using ProfileStore.get_init_curve_for() and executor profiles.
"""

from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING
from dataclasses import dataclass, field

from .workflow_graph import (
    WorkflowGraph,
    WorkflowNode,
    WorkflowEdge,
    NodeType,
    EdgeType,
    NodeStatus,
)

if TYPE_CHECKING:
    from src.primitives.profile_store import ProfileStore
    from src.primitives.topology_template import TopologyTemplate
    from src.decomposer.task_decomposer import SubTaskSpec


# ---------------------------------------------------------------------------
# Default executor candidates per primitive (for template instantiation)
# ---------------------------------------------------------------------------

DEFAULT_CANDIDATES: Dict[str, Dict[str, str]] = {
    "forecast": {
        "easy":    "forecast/fast_nn",
        "medium":  "forecast/ensemble_nn",
        "hard":    "forecast/strong_nn",
        "extreme": "forecast/physics_hybrid",
    },
    "state_parse": {
        "easy":    "state_parse/rule_parser",
        "medium":  "state_parse/llm_small",
        "hard":    "state_parse/rag_parser",
        "extreme": "state_parse/llm_large",
    },
    "data_analysis": {
        "easy":    "data_analysis/rule_based",
        "medium":  "data_analysis/ensemble",
        "hard":    "data_analysis/ml_pipeline",
        "extreme": "data_analysis/deep_learning",
    },
    "aggregator": {
        "default": "aggregator/merge",
    },
}


# ---------------------------------------------------------------------------
# Evaluator candidates per difficulty
# ---------------------------------------------------------------------------

EVALUATOR_MAP: Dict[str, str] = {
    "easy":    "rule_eval",
    "medium":  "small_eval",
    "hard":    "large_eval",
    "extreme": "large_eval",
}


# ---------------------------------------------------------------------------
# Template -> Node/Edge blueprint
# ---------------------------------------------------------------------------

# Maps template_id -> list of (node_id, node_type, depends_on) triples
# This is the structural definition of each topology template as an explicit graph.
TEMPLATE_BLUEPRINTS: Dict[str, List[dict]] = {
    "direct": [
        {"node_id": "exec_0", "node_type": NodeType.EXECUTOR, "depends_on": []},
    ],
    "exec_verify": [
        {"node_id": "exec_0",  "node_type": NodeType.EXECUTOR,   "depends_on": []},
        {"node_id": "verify_0", "node_type": NodeType.VERIFIER, "depends_on": ["exec_0"]},
    ],
    "dual_exec_aggregate": [
        {"node_id": "exec_a",   "node_type": NodeType.EXECUTOR,   "depends_on": []},
        {"node_id": "exec_b",   "node_type": NodeType.EXECUTOR,   "depends_on": []},
        {"node_id": "merge_0",  "node_type": NodeType.AGGREGATOR, "depends_on": ["exec_a", "exec_b"]},
    ],
    "exec_verify_hci": [
        {"node_id": "exec_0",   "node_type": NodeType.EXECUTOR,   "depends_on": []},
        {"node_id": "verify_0", "node_type": NodeType.VERIFIER,   "depends_on": ["exec_0"]},
        {"node_id": "hci_0",    "node_type": NodeType.HUMAN_GATE,  "depends_on": ["verify_0"]},
    ],
}


# Edge blueprint: (src, dst, edge_type, condition)
TEMPLATE_EDGE_BLUEPRINTS: Dict[str, List[tuple]] = {
    "direct": [
        # No edges needed for single-node graph
    ],
    "exec_verify": [
        ("exec_0", "verify_0", EdgeType.VERIFY, "always"),
    ],
    "dual_exec_aggregate": [
        ("exec_a", "merge_0", EdgeType.DATA, "always"),
        ("exec_b", "merge_0", EdgeType.DATA, "always"),
    ],
    "exec_verify_hci": [
        ("exec_0", "verify_0", EdgeType.VERIFY, "always"),
        ("verify_0", "hci_0", EdgeType.CONTROL, "always"),
    ],
}


# ---------------------------------------------------------------------------
# WorkflowBuilder
# ---------------------------------------------------------------------------

@dataclass
class WorkflowBuilder:
    """
    Builder for constructing WorkflowGraph instances.

    Provides multiple construction paths:
    1. from_template()      — build from a TopologyTemplate or template_id string
    2. from_subtask_specs()  — build from existing SubTaskSpec list
    3. from_task_description() — full pipeline: task → decomposer → graph

    After construction, use estimate_from_profile_store() to fill
    quality/cost/latency estimates from ProfileStore data.

    Attributes
    ----------
    primitive_name : str
        Primary primitive, e.g. "forecast", "state_parse", "data_analysis".
    difficulty : float
        Normalized difficulty [0, 1].
    difficulty_bucket : str
        Discrete bucket: "easy", "medium", "hard", "extreme".
    task_type : str
        Task domain, e.g. "time_series", "text_analysis".
    scenario : str
        Scenario name for workflow metadata.
    candidate_override : str | None
        Force a specific executor_id (e.g. "forecast/fast_nn").
        None = auto-select based on difficulty profile.
    constraints : list
        ConstraintSpec list to attach to execution nodes.
    profile_store : ProfileStore | None
        Optional ProfileStore for quality/cost/latency estimation.
    """
    primitive_name: str = "state_parse"
    difficulty: float = 0.5
    difficulty_bucket: str = "medium"
    task_type: str = "text_analysis"
    scenario: str = "Normal"
    candidate_override: str | None = None
    constraints: list = field(default_factory=list)
    profile_store: Optional["ProfileStore"] = None
    graph_id_prefix: str = "wg"

    def _get_default_executor(self) -> str:
        """Get the default executor_id for this builder's primitive + difficulty."""
        if self.candidate_override:
            return self.candidate_override
        prim_executors = DEFAULT_CANDIDATES.get(self.primitive_name, {})
        return prim_executors.get(self.difficulty_bucket, prim_executors.get("default", f"{self.primitive_name}/default"))

    def _get_default_evaluator(self) -> str:
        return EVALUATOR_MAP.get(self.difficulty_bucket, "rule_eval")

    # -------------------------------------------------------------------------
    # Construction methods
    # -------------------------------------------------------------------------

    def from_template_id(
        self,
        template_id: str,
        graph_id: str | None = None,
    ) -> WorkflowGraph:
        """
        Build a WorkflowGraph from a template_id string.

        Uses TEMPLATE_BLUEPRINTS + TEMPLATE_EDGE_BLUEPRINTS to construct
        the explicit node + edge graph.

        Parameters
        ----------
        template_id : str
            One of: "direct", "exec_verify", "dual_exec_aggregate", "exec_verify_hci".
        graph_id : str | None
            Optional graph ID. Auto-generated if None.

        Returns
        -------
        WorkflowGraph
        """
        if template_id not in TEMPLATE_BLUEPRINTS:
            raise ValueError(
                f"Unknown template_id '{template_id}'. "
                f"Available: {list(TEMPLATE_BLUEPRINTS.keys())}"
            )

        blueprint = TEMPLATE_BLUEPRINTS[template_id]
        edge_blueprint = TEMPLATE_EDGE_BLUEPRINTS.get(template_id, [])

        wg = WorkflowGraph(
            graph_id=graph_id or f"{self.graph_id_prefix}_{template_id}",
            metadata={
                "template_id": template_id,
                "primitive_name": self.primitive_name,
                "difficulty": self.difficulty,
                "difficulty_bucket": self.difficulty_bucket,
                "task_type": self.task_type,
                "scenario": self.scenario,
            },
        )

        # Add nodes
        for node_def in blueprint:
            node = self._build_node_from_def(node_def)
            wg.add_node(node)

        # Add edges — prefix both ends to match _build_node_from_def output
        prefix = f"{self.graph_id_prefix}_"
        for src, dst, edge_type, condition in edge_blueprint:
            wg.add_edge(WorkflowEdge(
                src=prefix + src,
                dst=prefix + dst,
                edge_type=edge_type,
                condition=condition,
            ))

        # Estimate quality/cost/latency from profile store
        self._estimate_graph(wg)

        return wg

    def from_topology_template(
        self,
        template: "TopologyTemplate",
        graph_id: str | None = None,
    ) -> WorkflowGraph:
        """
        Build a WorkflowGraph from an existing TopologyTemplate.

        Maps TemplateNode.node_type -> NodeType enum and TemplateNode.depends_on -> edges.
        """
        wg = WorkflowGraph(
            graph_id=graph_id or f"{self.graph_id_prefix}_{template.template_id}",
            metadata={
                "template_id": template.template_id,
                "primitive_name": self.primitive_name,
                "difficulty": self.difficulty,
                "difficulty_bucket": self.difficulty_bucket,
                "task_type": self.task_type,
                "scenario": self.scenario,
            },
        )

        # Map template node_id -> actual node_id
        id_map: Dict[str, str] = {}

        for tnode in template.nodes:
            actual_id = f"{self.graph_id_prefix}_{tnode.node_id}"
            id_map[tnode.node_id] = actual_id

            # Map node_type string -> NodeType enum
            node_type_map = {
                "executor":   NodeType.EXECUTOR,
                "verifier":   NodeType.VERIFIER,
                "aggregator": NodeType.AGGREGATOR,
                "hci":        NodeType.HUMAN_GATE,
                "cache":      NodeType.CACHE,
            }
            node_type = node_type_map.get(tnode.node_type, NodeType.EXECUTOR)

            # Determine executor_id
            if node_type == NodeType.EXECUTOR:
                executor_id = self._get_default_executor()
            elif node_type == NodeType.VERIFIER:
                executor_id = None  # evaluator selected separately
            elif node_type == NodeType.AGGREGATOR:
                executor_id = "aggregator/merge"
            else:
                executor_id = None

            depends_on = [id_map[d] for d in tnode.depends_on if d in id_map]

            node = WorkflowNode(
                node_id=actual_id,
                node_type=node_type,
                executor_id=executor_id,
                depends_on=depends_on,
                metadata={
                    "template_node_id": tnode.node_id,
                    "difficulty": self.difficulty,
                    "difficulty_bucket": self.difficulty_bucket,
                    "constraints": self.constraints,
                },
            )
            wg.add_node(node)

        # Add edges from template depends_on
        for tnode in template.nodes:
            actual_id = id_map[tnode.node_id]
            for dep_id in tnode.depends_on:
                if dep_id in id_map:
                    wg.add_edge(WorkflowEdge(
                        src=id_map[dep_id],
                        dst=actual_id,
                        edge_type=EdgeType.DATA,
                    ))

        self._estimate_graph(wg)
        return wg

    def from_subtask_specs(
        self,
        sub_task_specs: List["SubTaskSpec"],
        graph_id: str | None = None,
    ) -> WorkflowGraph:
        """
        Build a WorkflowGraph from an existing list of SubTaskSpec.

        This enables backward compatibility: existing code generates SubTaskSpecs,
        and this method wraps them in a WorkflowGraph without changing existing logic.

        Mapping:
        - SubTaskSpec.sub_task_id  -> WorkflowNode.node_id
        - SubTaskSpec.primitive_name -> executor_id
        - SubTaskSpec.predecessor_ids -> WorkflowEdge
        - NodeType.EXECUTOR for all (SubTaskSpec doesn't distinguish verifier/aggregator)
        """
        wg = WorkflowGraph(
            graph_id=graph_id or f"{self.graph_id_prefix}_from_subtasks",
            metadata={
                "primitive_name": self.primitive_name,
                "difficulty": self.difficulty,
                "difficulty_bucket": self.difficulty_bucket,
                "task_type": self.task_type,
                "scenario": self.scenario,
                "template_id": "from_subtask_specs",
            },
        )

        for st in sub_task_specs:
            executor_id = self.candidate_override or f"{st.primitive_name}/auto"

            node = WorkflowNode(
                node_id=st.sub_task_id,
                node_type=NodeType.EXECUTOR,
                executor_id=executor_id,
                depends_on=list(st.predecessor_ids),
                metadata={
                    "primitive_name": st.primitive_name,
                    "difficulty": st.difficulty,
                    "difficulty_bucket": st.difficulty_bucket,
                    "task_description": st.metadata.get("task_description", ""),
                    "constraints": st.constraints,
                    "input_modality": st.input_modality.value,
                    "intermediate_modality": st.intermediate_modality.value if st.intermediate_modality else None,
                    "evaluator_name": st.evaluator_name,
                },
            )
            wg.add_node(node)

            # Add edges from predecessor_ids
            for pred_id in st.predecessor_ids:
                if pred_id in wg.nodes:
                    wg.add_edge(WorkflowEdge(
                        src=pred_id,
                        dst=st.sub_task_id,
                        edge_type=EdgeType.DATA,
                    ))

        self._estimate_graph(wg)
        return wg

    def from_task_description(
        self,
        task_description: str,
        decomposer: "Optional[Any]" = None,
        graph_id: str | None = None,
    ) -> WorkflowGraph:
        """
        Full pipeline: task description -> SubTaskSpec list -> WorkflowGraph.

        Uses TaskDecomposer to decompose the task, then from_subtask_specs().
        If decomposer is None, creates a default TaskDecomposer.
        """
        if decomposer is None:
            from src.decomposer.task_decomposer import TaskDecomposer
            decomposer = TaskDecomposer()

        sub_tasks, task_type = decomposer.decompose(task_description)
        self.task_type = task_type

        # Infer difficulty from decomposer output
        if sub_tasks:
            self.difficulty = sub_tasks[0].difficulty
            self.difficulty_bucket = sub_tasks[0].difficulty_bucket

        wg = self.from_subtask_specs(sub_tasks, graph_id=graph_id)
        wg.metadata["task_description"] = task_description
        wg.metadata["task_type"] = task_type
        return wg

    # -------------------------------------------------------------------------
    # Node construction helpers
    # -------------------------------------------------------------------------

    def _build_node_from_def(self, node_def: dict) -> WorkflowNode:
        """Build a WorkflowNode from a node blueprint definition."""
        node_type: NodeType = node_def["node_type"]
        node_id: str = node_def["node_id"]

        # Map actual depends_on using the graph_id_prefix
        depends_on = node_def.get("depends_on", [])

        # Determine executor_id
        if node_type == NodeType.EXECUTOR:
            executor_id = self._get_default_executor()
        elif node_type == NodeType.VERIFIER:
            executor_id = None  # evaluator is selected at runtime
            evaluator_name = self._get_default_evaluator()
        elif node_type == NodeType.AGGREGATOR:
            executor_id = "aggregator/merge"
            evaluator_name = None
        elif node_type == NodeType.HUMAN_GATE:
            executor_id = None
            evaluator_name = None
        else:
            executor_id = None
            evaluator_name = None

        return WorkflowNode(
            node_id=f"{self.graph_id_prefix}_{node_id}",
            node_type=node_type,
            executor_id=executor_id,
            depends_on=[f"{self.graph_id_prefix}_{d}" for d in depends_on],
            evaluator_name=evaluator_name if node_type == NodeType.VERIFIER else None,
            metadata={
                "difficulty": self.difficulty,
                "difficulty_bucket": self.difficulty_bucket,
                "primitive_name": self.primitive_name,
                "constraints": self.constraints,
                "task_type": self.task_type,
            },
        )

    # -------------------------------------------------------------------------
    # Profile estimation
    # -------------------------------------------------------------------------

    def _estimate_graph(self, wg: WorkflowGraph) -> None:
        """
        Fill estimated_quality / estimated_cost / estimated_latency for all nodes.

        Strategy:
        - For EXECUTOR nodes: use ProfileStore.get_init_curve_for() or DEFAULT_GROUND_TRUTH
        - For VERIFIER nodes: use evaluator profile (false_pass_rate, cost)
        - For AGGREGATOR nodes: small fixed overhead
        - For HCI nodes: fixed human_cost overhead
        """
        ps = self.profile_store
        bucket = self.difficulty_bucket
        prim = self.primitive_name

        for node in wg.nodes.values():
            if node.node_type == NodeType.EXECUTOR and node.executor_id:
                parts = node.executor_id.split("/")
                p_name = parts[0] if len(parts) > 0 else prim
                c_name = parts[1] if len(parts) > 1 else "default"

                if ps is not None:
                    init_curve = ps.get_init_curve_for(p_name, c_name)
                    if init_curve and bucket in init_curve:
                        node.estimated_quality = init_curve[bucket].get("acc_mean", 0.5)
                        node.estimated_cost = init_curve[bucket].get("cost_mean", 1.0)
                        node.estimated_latency = self._estimate_latency_from_store(ps, node.executor_id, bucket)
                    else:
                        self._apply_default_estimates(node, p_name, c_name, bucket)
                else:
                    self._apply_default_estimates(node, p_name, c_name, bucket)

            elif node.node_type == NodeType.VERIFIER:
                # Evaluator: use evaluator profile or default
                eval_name = node.evaluator_name or self._get_default_evaluator()
                node.estimated_quality = 0.0  # verifier doesn't produce quality
                node.estimated_cost = self._get_evaluator_cost(eval_name)
                node.estimated_latency = 0.0   # ignored per method definition

            elif node.node_type == NodeType.AGGREGATOR:
                node.estimated_quality = 0.02  # small quality boost from aggregation
                node.estimated_cost = 0.1      # small aggregation cost
                node.estimated_latency = 0.05  # negligible latency

            elif node.node_type == NodeType.HUMAN_GATE:
                node.estimated_quality = 0.0
                node.estimated_cost = 0.5      # human approval overhead
                node.estimated_latency = 0.0

    def _apply_default_estimates(
        self,
        node: WorkflowNode,
        primitive_name: str,
        candidate_name: str,
        bucket: str,
    ) -> None:
        """Apply hardcoded default estimates when no profile data is available."""
        # Default quality by candidate tier
        QUALITY_DEFAULTS: Dict[str, Dict[str, float]] = {
            "fast_nn":        {"easy": 0.92, "medium": 0.82, "hard": 0.70, "extreme": 0.55},
            "ensemble_nn":    {"easy": 0.88, "medium": 0.78, "hard": 0.68, "extreme": 0.52},
            "strong_nn":      {"easy": 0.85, "medium": 0.75, "hard": 0.65, "extreme": 0.50},
            "rule_parser":    {"easy": 0.90, "medium": 0.80, "hard": 0.65, "extreme": 0.50},
            "llm_small":      {"easy": 0.85, "medium": 0.75, "hard": 0.62, "extreme": 0.48},
            "rag_parser":     {"easy": 0.82, "medium": 0.72, "hard": 0.60, "extreme": 0.45},
            "llm_large":      {"easy": 0.80, "medium": 0.70, "hard": 0.58, "extreme": 0.42},
            "rule_based":     {"easy": 0.90, "medium": 0.80, "hard": 0.65, "extreme": 0.50},
            "ensemble":       {"easy": 0.86, "medium": 0.76, "hard": 0.63, "extreme": 0.48},
            "ml_pipeline":    {"easy": 0.83, "medium": 0.73, "hard": 0.60, "extreme": 0.45},
            "deep_learning":  {"easy": 0.80, "medium": 0.70, "hard": 0.58, "extreme": 0.42},
        }
        COST_DEFAULTS: Dict[str, float] = {
            "fast_nn": 0.30, "ensemble_nn": 0.60, "strong_nn": 1.20,
            "physics_hybrid": 2.00, "fvcom": 1.80,
            "rule_parser": 0.05, "llm_small": 0.30, "rag_parser": 0.80, "llm_large": 2.00,
            "rule_based": 0.05, "ensemble": 0.40, "ml_pipeline": 0.80, "deep_learning": 1.50,
        }

        row = QUALITY_DEFAULTS.get(candidate_name, {})
        node.estimated_quality = row.get(bucket, 0.65)
        node.estimated_cost = COST_DEFAULTS.get(candidate_name, 0.50)
        # Latency: use cost as proxy for now (will be updated from real execution)
        node.estimated_latency = node.estimated_cost * 1.5

    def _estimate_latency_from_store(
        self,
        ps: "ProfileStore",
        executor_id: str,
        bucket: str,
    ) -> float:
        """Get latency estimate from ProfileStore executor profile."""
        prof = ps.get_executor_profile(executor_id, bucket)
        if prof is not None:
            return prof.latency_mean
        return self._default_latency(bucket)

    def _get_evaluator_cost(self, evaluator_name: str) -> float:
        """Get evaluator API cost from EVALUATOR_PROFILES."""
        try:
            from src.evaluation.mock_evaluator import EVALUATOR_PROFILES
            return EVALUATOR_PROFILES.get(evaluator_name, {}).get("cost", 0.10)
        except Exception:
            return 0.10

    @staticmethod
    def _default_latency(bucket: str) -> float:
        """Default latency by difficulty bucket (seconds)."""
        return {"easy": 0.5, "medium": 1.0, "hard": 2.0, "extreme": 4.0}.get(bucket, 1.0)
