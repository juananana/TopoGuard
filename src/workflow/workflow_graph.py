"""
workflow_graph.py
=================
Explicit Workflow Graph data structures for TopoGuard.

This module implements the G = (V, E, τ, φ) formalism from the method definition:
- V: nodes (WorkflowNode)
- E: edges (WorkflowEdge)
- τ(v): node type (e.g. "executor", "verifier", "aggregator")
- φ(v): executor configuration (e.g. "forecast/fast_nn")

Design goals:
1. Backward compatible with existing SubTaskSpec / TopologyTemplate
2. Explicit edges allow DAG analysis (critical path, parallel branches)
3. Every node carries executor config, every edge carries data/control flow semantics
4. Latency = critical path latency (max over parallel branches, per DAG semantics)

Usage:
    # Build from template
    builder = WorkflowBuilder(primitive_name="forecast", difficulty="hard")
    graph = builder.from_template("exec_verify")

    # Execute
    result = workflow_executor.execute(graph, evaluator, profile_store)
"""

from __future__ import annotations

import uuid
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class NodeType(Enum):
    """
    Role of a node in the workflow graph.
    Determines which tool pool is queried and how latency is accounted.
    """
    EXECUTOR   = "executor"    # Primary task execution node
    VERIFIER   = "verifier"    # Post-execution quality check (evaluator)
    AGGREGATOR = "aggregator"  # Merges multiple branch outputs
    HUMAN_GATE = "hci"         # Human-in-the-loop approval gate
    CACHE      = "cache"       # Caching layer (result reuse)

    def is_execution_node(self) -> bool:
        """True if this node type runs an executor (contributes to latency)."""
        return self in (NodeType.EXECUTOR, NodeType.AGGREGATOR)


class EdgeType(Enum):
    """Semantic type of a directed edge in the workflow graph."""
    DATA     = "data"      # Data dependency: dst reads output of src
    CONTROL  = "control"  # Control flow: src must complete before dst
    FALLBACK = "fallback"  # Fallback: if src fails, try dst
    VERIFY   = "verify"    # Verification: dst verifies output of src


class NodeStatus(Enum):
    """Lifecycle status of a node within a graph execution."""
    PENDING  = "pending"
    RUNNING  = "running"
    DONE     = "done"
    FAILED   = "failed"
    SKIPPED  = "skipped"


# ---------------------------------------------------------------------------
# WorkflowNode
# ---------------------------------------------------------------------------

@dataclass
class WorkflowNode:
    """
    A single node within a WorkflowGraph.

    Corresponds to τ(v) in the method definition: the node type determines
    which tool pool is queried; the executor_id determines which specific
    candidate is used.

    Attributes
    ----------
    node_id : str
        Unique ID within the graph (e.g. "exec_0", "verify_0", "merge_0").
    node_type : NodeType
        Role: executor / verifier / aggregator / hci / cache.
    executor_id : str | None
        Specific executor to use. Format: "primitive/candidate" (e.g. "forecast/fast_nn").
        None = auto-select from profile manager.
    params : dict
        Additional parameters for this node (e.g. temperature, top_k, timeout).
    status : NodeStatus
        Execution status (updated by WorkflowExecutor).
    depends_on : List[str]
        Node IDs that must complete before this node runs.
        Note: this is the logical dependency; actual edges are in WorkflowEdge.
    metadata : dict
        Extension fields: task_type, template_id, constraint_ids, etc.
    estimated_latency : float
        Estimated execution time in seconds (from executor profile).
    estimated_quality : float
        Estimated output quality (from executor profile).
    estimated_cost : float
        Estimated API cost (from executor profile).
    evaluator_name : str | None
        Which evaluator to use for this node (verifier nodes only).
        None = auto-select.
    """
    node_id: str
    node_type: NodeType
    executor_id: str | None = None
    params: dict = field(default_factory=dict)
    status: NodeStatus = NodeStatus.PENDING
    depends_on: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    # Estimation fields (populated by WorkflowBuilder or profile lookup)
    estimated_latency: float = 0.0
    estimated_quality: float = 0.0
    estimated_cost: float = 0.0
    # Evaluator selection (only for VERIFIER nodes)
    evaluator_name: str | None = None

    @property
    def primitive_name(self) -> str:
        """Extract primitive name from executor_id (e.g. 'forecast' from 'forecast/fast_nn')."""
        if self.executor_id and "/" in self.executor_id:
            return self.executor_id.split("/")[0]
        return "unknown"

    @property
    def candidate_name(self) -> str | None:
        """Extract candidate name from executor_id (e.g. 'fast_nn' from 'forecast/fast_nn')."""
        if self.executor_id and "/" in self.executor_id:
            return self.executor_id.split("/")[1]
        return None

    def is_execution_node(self) -> bool:
        return self.node_type.is_execution_node()

    def is_verifier(self) -> bool:
        return self.node_type == NodeType.VERIFIER

    def to_subtask_spec_dict(self) -> dict:
        """
        Serialize as a dict compatible with SubTaskSpec fields.
        Used for backward compatibility with existing evaluator / profile_manager.
        """
        from src.decomposer.task_decomposer import difficulty_to_bucket
        difficulty = self.metadata.get("difficulty", 0.5)
        return {
            "sub_task_id": self.node_id,
            "primitive_name": self.primitive_name,
            "difficulty": difficulty,
            "difficulty_bucket": difficulty_to_bucket(difficulty),
            "description": f"[{self.node_type.value}] {self.primitive_name}",
            "predecessor_ids": self.depends_on,
            "metadata": {**self.metadata, "node_type": self.node_type.value},
            "constraints": self.metadata.get("constraints", []),
            "evaluator_name": self.evaluator_name,
        }


# ---------------------------------------------------------------------------
# WorkflowEdge
# ---------------------------------------------------------------------------

@dataclass
class WorkflowEdge:
    """
    A directed edge between two WorkflowNodes.

    Represents φ(v) relationships and edge semantics (data/control/fallback/verify).
    The graph is always a DAG (edges point from earlier to later in execution order).
    """
    src: str       # Source node_id
    dst: str       # Destination node_id
    edge_type: EdgeType = EdgeType.DATA
    condition: str | None = None  # e.g. "on_fail", "always"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.src == self.dst:
            raise ValueError(f"Self-loop edge detected: {self.src} -> {self.dst}")


# ---------------------------------------------------------------------------
# WorkflowGraph
# ---------------------------------------------------------------------------

@dataclass
class WorkflowGraph:
    """
    Complete workflow graph representation G = (V, E, τ, φ).

    This is the explicit graph object described in the method definition.
    It replaces implicit SubTaskSpec chains with:
    - Explicit node registry (nodes dict, keyed by node_id)
    - Explicit edge list (edges list, with semantics)
    - Metadata (scenario, task_type, budget, difficulty, etc.)

    Attributes
    ----------
    graph_id : str
        Unique identifier for this graph instance.
    nodes : Dict[str, WorkflowNode]
        Node registry: node_id -> WorkflowNode.
    edges : List[WorkflowEdge]
        Edge list: directed edges between nodes.
    metadata : dict
        Graph-level metadata:
        - scenario, task_type, difficulty, difficulty_bucket
        - budget, latency_limit
        - template_id (which TopologyTemplate was used)
        - task_description (original input)
    """
    graph_id: str = field(default_factory=lambda: f"wg_{uuid.uuid4().hex[:8]}")
    nodes: Dict[str, WorkflowNode] = field(default_factory=dict)
    edges: List[WorkflowEdge] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    # -------------------------------------------------------------------------
    # Graph queries
    # -------------------------------------------------------------------------

    def add_node(self, node: WorkflowNode) -> None:
        """Add a node to the graph. Overwrites if node_id already exists."""
        self.nodes[node.node_id] = node

    def add_edge(self, edge: WorkflowEdge) -> None:
        """Add an edge to the graph."""
        if edge.src not in self.nodes:
            raise KeyError(f"Source node '{edge.src}' not in graph")
        if edge.dst not in self.nodes:
            raise KeyError(f"Destination node '{edge.dst}' not in graph")
        self.edges.append(edge)

    def successors(self, node_id: str) -> List[str]:
        """Return node_ids that depend on this node (direct children)."""
        return [e.dst for e in self.edges if e.src == node_id]

    def predecessors(self, node_id: str) -> List[str]:
        """Return node_ids that this node depends on (direct parents)."""
        return [e.src for e in self.edges if e.dst == node_id]

    def all_successors(self, node_id: str) -> List[str]:
        """Return all descendant node_ids (transitive closure of successors)."""
        visited = set()
        queue = [node_id]
        while queue:
            nid = queue.pop()
            for succ in self.successors(nid):
                if succ not in visited:
                    visited.add(succ)
                    queue.append(succ)
        return list(visited)

    def all_predecessors(self, node_id: str) -> List[str]:
        """Return all ancestor node_ids (transitive closure of predecessors)."""
        visited = set()
        queue = [node_id]
        while queue:
            nid = queue.pop()
            for pred in self.predecessors(nid):
                if pred not in visited:
                    visited.add(pred)
                    queue.append(pred)
        return list(visited)

    def execution_order(self) -> List[str]:
        """
        Topological sort of execution nodes (executor + aggregator nodes).
        Verifier/HCI nodes are interleaved after their target nodes.

        Returns node_ids in the order they should be executed.
        """
        # Compute in-degree for all nodes
        in_degree: Dict[str, int] = {nid: 0 for nid in self.nodes}
        for edge in self.edges:
            in_degree[edge.dst] = in_degree.get(edge.dst, 0) + 1

        # Kahn's algorithm
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        sorted_nodes = []
        while queue:
            nid = queue.pop(0)
            sorted_nodes.append(nid)
            for succ in self.successors(nid):
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

        if len(sorted_nodes) != len(self.nodes):
            # Cycle detected — fall back to insertion order
            return list(self.nodes.keys())

        return sorted_nodes

    def execution_nodes(self) -> List[WorkflowNode]:
        """Return nodes that contribute to latency (executors + aggregators)."""
        return [n for n in self.nodes.values() if n.is_execution_node()]

    def verifier_nodes(self) -> List[WorkflowNode]:
        """Return all verifier (evaluator) nodes."""
        return [n for n in self.nodes.values() if n.is_verifier()]

    def total_estimated_latency(self) -> float:
        """
        Critical path latency for all execution nodes.

        For parallel branches in the DAG, uses max (critical path) instead of sum,
        per method definition: L(G) = Critical Path Length(G).
        Ignores evaluator (verifier) nodes.
        """
        exec_nodes = self.execution_nodes()
        if not exec_nodes:
            return 0.0
        if not self.edges:
            return sum(n.estimated_latency for n in exec_nodes)

        # Build in-degree for execution nodes only
        exec_ids = {n.node_id for n in exec_nodes}
        in_degree: Dict[str, int] = {nid: 0 for nid in exec_ids}
        for e in self.edges:
            if e.src in exec_ids and e.dst in exec_ids:
                in_degree[e.dst] = in_degree.get(e.dst, 0) + 1

        # DP: longest path to each execution node
        node_latency: Dict[str, float] = {}
        in_deg = dict(in_degree)
        queue = [nid for nid, d in in_deg.items() if d == 0]
        topo_order = []
        while queue:
            nid = queue.pop(0)
            topo_order.append(nid)
            for e in self.edges:
                if e.src == nid and e.dst in exec_ids:
                    in_deg[e.dst] -= 1
                    if in_deg[e.dst] == 0:
                        queue.append(e.dst)

        nid_to_node = {n.node_id: n for n in exec_nodes}
        for nid in topo_order:
            pred = [node_latency[e.src] for e in self.edges
                    if e.dst == nid and e.src in node_latency]
            self_lat = nid_to_node[nid].estimated_latency if nid in nid_to_node else 0.0
            node_latency[nid] = (max(pred) if pred else 0.0) + self_lat

        return max(node_latency.values()) if node_latency else 0.0

    def total_estimated_cost(self) -> float:
        """Sum of estimated API costs for all nodes."""
        return sum(n.estimated_cost for n in self.nodes.values())

    def node_by_id(self, node_id: str) -> WorkflowNode:
        """Get a node by ID, raising KeyError if not found."""
        if node_id not in self.nodes:
            raise KeyError(f"Node '{node_id}' not found in graph '{self.graph_id}'")
        return self.nodes[node_id]

    def summary(self) -> str:
        """Human-readable graph summary for debugging."""
        node_lines = []
        for nid, n in self.nodes.items():
            status_flag = f" [{n.status.value}]" if n.status != NodeStatus.PENDING else ""
            exec_flag = f" φ={n.executor_id}" if n.executor_id else ""
            est = f" Q={n.estimated_quality:.2f} C={n.estimated_cost:.2f} L={n.estimated_latency:.2f}"
            node_lines.append(f"  {nid}: {n.node_type.value}{exec_flag}{status_flag}{est}")
        edge_lines = [f"  {e.src} --[{e.edge_type.value}]--> {e.dst}" for e in self.edges]
        return (
            f"WorkflowGraph[{self.graph_id}]\n"
            + f"  Metadata: {self.metadata}\n"
            + "  Nodes:\n" + "\n".join(node_lines) + "\n"
            + "  Edges:\n" + "\n".join(edge_lines) + "\n"
            + f"  Total estimated latency: {self.total_estimated_latency():.3f}s"
        )


# ---------------------------------------------------------------------------
# NodeResult (per-node execution record)
# ---------------------------------------------------------------------------

@dataclass
class NodeResult:
    """
    Result of executing a single WorkflowNode.

    Analogous to EvaluationResult in the existing system, but scoped to one node.
    """
    node_id: str
    executor_id: str | None
    evaluator_name: str | None
    status: NodeStatus
    # Quality metrics
    observed_quality: float = 0.0
    quality_score: float = 0.0        # raw evaluator rubric score
    eval_pass: bool = False
    # Error diagnostics
    error_type: str | None = None     # "low_quality" | "format_error" | "unsafe_decision" | ...
    confidence: float = 1.0
    # Cost & latency
    observed_cost: float = 0.0
    observed_latency: float = 0.0
    evaluator_cost: float = 0.0
    evaluator_latency: float = 0.0
    # Profile metadata
    true_quality: float | None = None  # ground truth quality (if available)
    predicted_quality: float | None = None
    predicted_cost: float | None = None
    # Repair info
    repaired: bool = False
    repair_action: str = "none"       # "none" | "upgraded_candidate" | "upgraded_evaluator" | "template_upgrade"
    # Raw result from evaluator
    raw_result: dict | None = None


# ---------------------------------------------------------------------------
# WorkflowResult (graph-level execution record)
# ---------------------------------------------------------------------------

@dataclass
class WorkflowResult:
    """
    Result of executing a complete WorkflowGraph.

    Analogous to the per-episode aggregation in mvp_experiment.py,
    but scoped to the workflow level.

    Metrics:
    - overall_pass: whether the workflow produced a pass-grade output
    - total_latency: critical path latency (max over parallel branches, per DAG semantics)
    - total_cost: sum of all node costs
    - violation_rate: fraction of constraint violations
    - repair_count: how many nodes were repaired during execution
    """
    graph_id: str
    node_results: List[NodeResult] = field(default_factory=list)
    # Aggregate metrics
    overall_pass: bool = False
    total_latency: float = 0.0        # critical path latency (max over parallel branches, no evaluator)
    total_cost: float = 0.0
    total_executor_cost: float = 0.0
    total_evaluator_cost: float = 0.0
    # Constraint violations
    violation_count: int = 0
    violation_rate: float = 0.0
    constraint_violations: List[dict] = field(default_factory=list)
    # Repair metrics
    repair_count: int = 0
    repair_success_count: int = 0
    # Quality
    final_quality: float = 0.0        # quality of the final (output) node
    # Execution metadata
    execution_duration: float = 0.0  # wall-clock time for entire graph
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    metadata: dict = field(default_factory=dict)

    def critical_path_latency(self) -> float:
        """
        Compute critical path latency using actual observed latencies.

        For parallel branches in the DAG, uses max (critical path) instead of sum.
        This correctly models that two parallel executions run concurrently.

        Algorithm: longest path in DAG from any start node to any end node.
        A node's latency contributes only once even if multiple paths pass through it.

        Returns 0.0 if no completed execution nodes.
        """
        if not self.node_results:
            return 0.0

        # Build adjacency list from edges in metadata (set during execution)
        edges: List[Tuple[str, str]] = self.metadata.get("edges", [])
        if not edges:
            # Fallback: sum all latencies (sequential fallback)
            return sum(
                r.observed_latency
                for r in self.node_results
                if r.status == NodeStatus.DONE and r.executor_id is not None
            )

        # Build in-degree map
        node_ids = {r.node_id for r in self.node_results}
        in_degree: Dict[str, int] = {nid: 0 for nid in node_ids}
        for src, dst in edges:
            if src in node_ids and dst in node_ids:
                in_degree[dst] = in_degree.get(dst, 0) + 1

        # DP: longest path from any start node to each node
        # node_latency[node_id] = longest path ending at that node
        node_latency: Dict[str, float] = {}
        completed_ids = {
            r.node_id for r in self.node_results
            if r.status == NodeStatus.DONE and r.executor_id is not None
        }

        # Initialize: nodes with in_degree == 0 (start nodes) get their own latency
        for nid in node_ids:
            if in_degree.get(nid, 0) == 0:
                start_node = next(
                    (r for r in self.node_results if r.node_id == nid and nid in completed_ids),
                    None
                )
                node_latency[nid] = start_node.observed_latency if start_node else 0.0

        # Topological order via Kahn's algorithm (limited to relevant edges)
        in_deg = dict(in_degree)
        queue = [nid for nid, d in in_deg.items() if d == 0]
        topo_order = []
        while queue:
            nid = queue.pop(0)
            topo_order.append(nid)
            for src, dst in edges:
                if src == nid and dst in node_ids:
                    in_deg[dst] -= 1
                    if in_deg[dst] == 0:
                        queue.append(dst)

        # DP relaxation in topological order
        for nid in topo_order:
            pred_latencies = []
            for src, dst in edges:
                if dst == nid and src in node_latency:
                    pred_latencies.append(node_latency[src])
            if pred_latencies:
                node_latency[nid] = max(pred_latencies) + (
                    next((r.observed_latency for r in self.node_results
                         if r.node_id == nid and nid in completed_ids), 0.0)
                )
            elif nid not in node_latency:
                node_latency[nid] = next((r.observed_latency for r in self.node_results
                                         if r.node_id == nid and nid in completed_ids), 0.0)

        return max(node_latency.values()) if node_latency else 0.0

    def compute_totals(self) -> None:
        """Recompute aggregate metrics from node_results.

        Call this AFTER node_results is fully populated (not in __init__).
        Fields already set by caller (total_latency, execution_duration) are
        preserved if node_results is empty (graceful no-op).
        """
        if not self.node_results:
            return
        self.total_latency = self.critical_path_latency()
        self.total_executor_cost = sum(
            r.observed_cost for r in self.node_results
            if r.status == NodeStatus.DONE and r.executor_id is not None
        )
        self.total_evaluator_cost = sum(r.evaluator_cost for r in self.node_results)
        self.total_cost = self.total_executor_cost + self.total_evaluator_cost
        self.repair_count = sum(1 for r in self.node_results if r.repaired)
        self.repair_success_count = sum(
            1 for r in self.node_results if r.repaired and r.eval_pass
        )
        self.violation_count = len(self.constraint_violations)
        if self.node_results:
            self.violation_rate = self.violation_count / len(self.node_results)

    def to_episode_record(self, episode: int, task_id: str) -> "EpisodeRecord":
        """
        Convert WorkflowResult to EpisodeRecord format for backward compatibility.
        Uses the last execution node's metrics.
        """
        from dataclasses import asdict
        from src.experiments.mvp_experiment import EpisodeRecord

        last = self.node_results[-1] if self.node_results else None
        if last:
            return EpisodeRecord(
                episode=episode,
                task_id=task_id,
                sub_task_id=last.node_id,
                primitive_name=last.executor_id.split("/")[0] if last.executor_id else "unknown",
                difficulty_bucket="unknown",
                difficulty=0.5,
                selected_candidate=last.executor_id.split("/")[1] if last.executor_id and "/" in last.executor_id else last.executor_id or "unknown",
                predicted_acc=last.predicted_quality or 0.0,
                predicted_cost=last.predicted_cost or 0.0,
                true_acc=last.true_quality or 0.0,
                true_cost=last.observed_cost or 0.0,
                observed_acc=last.observed_quality or 0.0,
                observed_cost=last.observed_cost or 0.0,
                eval_pass=last.eval_pass,
                failure_type=last.error_type,
                recalibrated=False,
                source="workflow_graph",
                constraint_violations=self.constraint_violations,
                violation_count=self.violation_count,
                human_approved=True,
                execution_duration=self.execution_duration,
                input_modality="text",
                intermediate_modality=None,
                evaluator_name=last.evaluator_name or "unknown",
                evaluator_id=last.evaluator_name or "unknown",
                error_type=last.error_type,
                confidence=last.confidence,
                evaluator_latency=last.evaluator_latency,
                evaluator_cost=last.evaluator_cost,
                quality_score=last.quality_score,
                node_type=last.executor_id.split("/")[0] if last.executor_id else "unknown",
                task_type=self.metadata.get("task_type", "unknown"),
                template_id=self.metadata.get("template_id", "unknown"),
                repair_action=last.repair_action,
                template_upgraded_from="none",
                template_upgraded_to="none",
                llm_decomposer_cost=0.0,
                llm_evaluator_cost=0.0,
            )
        return EpisodeRecord(
            episode=episode,
            task_id=task_id,
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
            failure_type=None,
            recalibrated=False,
            source="workflow_graph",
            constraint_violations=self.constraint_violations,
            violation_count=self.violation_count,
            human_approved=True,
            execution_duration=self.execution_duration,
            input_modality="text",
            intermediate_modality=None,
            evaluator_name="unknown",
            evaluator_id="unknown",
            error_type=None,
            confidence=1.0,
            evaluator_latency=0.0,
            evaluator_cost=0.0,
            quality_score=0.0,
            node_type="unknown",
            task_type=self.metadata.get("task_type", "unknown"),
            template_id=self.metadata.get("template_id", "unknown"),
            repair_action="none",
            template_upgraded_from="none",
            template_upgraded_to="none",
            llm_decomposer_cost=0.0,
            llm_evaluator_cost=0.0,
        )


# ---------------------------------------------------------------------------
# WorkflowProfile (template-level performance record)
# ---------------------------------------------------------------------------

@dataclass
class WorkflowProfile:
    """
    Performance profile for a workflow template.

    Records aggregate statistics from multiple executions of the same template.
    Used for template-level Pareto frontier computation.

    Attributes
    ----------
    template_id : str
        Matches TopologyTemplate.template_id.
    scenario : str
        e.g. "Normal", "Hard", "OOD", "must_verify".
    task_type : str
        e.g. "time_series", "text_analysis", "tabular_analysis".
    acc_mean, acc_std : float
        Mean and std of pass rates across executions.
    cost_mean, cost_std : float
        Mean and std of total cost.
    latency_mean, latency_std : float
        Mean and std of total latency (excludes evaluator latency).
    violation_rate_mean : float
        Fraction of executions with constraint violations.
    repair_rate_mean : float
        Fraction of executions requiring repair.
    support_count : int
        Number of executions observed.
    """
    template_id: str
    scenario: str = "Normal"
    task_type: str = "unknown"
    acc_mean: float = 0.0
    acc_std: float = 0.0
    cost_mean: float = 0.0
    cost_std: float = 0.0
    latency_mean: float = 0.0
    latency_std: float = 0.0
    violation_rate_mean: float = 0.0
    repair_rate_mean: float = 0.0
    support_count: int = 0

    def update_from_result(self, result: WorkflowResult) -> None:
        """Update profile from a single execution result (running mean)."""
        n = self.support_count + 1
        quality = 1.0 if result.overall_pass else 0.0
        self.acc_mean += (quality - self.acc_mean) / n
        self.acc_std = self._running_std(self.acc_std, self.acc_mean, quality, n)

        self.cost_mean += (result.total_cost - self.cost_mean) / n
        self.cost_std = self._running_std(self.cost_std, self.cost_mean, result.total_cost, n)

        self.latency_mean += (result.total_latency - self.latency_mean) / n
        self.latency_std = self._running_std(self.latency_std, self.latency_mean, result.total_latency, n)

        viol_rate = result.violation_count / max(len(result.node_results), 1)
        self.violation_rate_mean += (viol_rate - self.violation_rate_mean) / n
        repair_rate = result.repair_count / max(len(result.node_results), 1)
        self.repair_rate_mean += (repair_rate - self.repair_rate_mean) / n

        self.support_count = n

    @staticmethod
    def _running_std(current_std: float, current_mean: float, new_val: float, n: int) -> float:
        if n < 2:
            return 0.0
        return ((n - 2) / (n - 1)) * current_std ** 2 + (new_val - current_mean) ** 2 / n
