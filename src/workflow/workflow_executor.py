"""
workflow_executor.py
===================
Executes a WorkflowGraph end-to-end and returns a WorkflowResult.

This module provides the execution engine for the WorkflowGraph framework.
It maps:
    WorkflowGraph → sequence of node executions → WorkflowResult

Key design decisions:
1. Backward compatible: uses existing MockEvaluator / ProfileManager internally
2. Repair integration: local repair via _repair_subgraph() strategy (same as MVP)
3. Latency accounting: only execution nodes (EXECUTOR, AGGREGATOR) count
4. Profile update: updates ProfileStore after each execution

Usage:
    from src.workflow import WorkflowBuilder, WorkflowExecutor

    builder = WorkflowBuilder(primitive_name="forecast", difficulty=0.7, ...)
    graph = builder.from_template_id("exec_verify")
    result = WorkflowExecutor().execute(graph, evaluator, profile_manager, profile_store)
"""

from __future__ import annotations

import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, TYPE_CHECKING

from .workflow_graph import (
    WorkflowGraph,
    WorkflowNode,
    WorkflowEdge,
    NodeType,
    NodeStatus,
    NodeResult,
    WorkflowResult,
)

if TYPE_CHECKING:
    from src.primitives.profile_manager import PrimitivePerformanceProfileManager
    from src.primitives.profile_store import ProfileStore
    from src.evaluation.mock_evaluator import MockEvaluator


# ---------------------------------------------------------------------------
# WorkflowExecutor
# ---------------------------------------------------------------------------

@dataclass
class WorkflowExecutor:
    """
    Executes a WorkflowGraph and returns a WorkflowResult.

    Execution algorithm (topology-first, per method definition):
        1. Topological sort of graph nodes
        2. For each node in order:
           a. Wait for all predecessors to complete
           b. Execute node (executor / verifier / aggregator)
           c. If VERIFIER: run evaluator on predecessor output
           d. If failed: attempt local repair (up to MAX_REPAIR_ATTEMPTS)
           e. If HCI: simulate human approval
           f. Update ProfileStore with observation
        3. Aggregate all node results into WorkflowResult

    Attributes
    ----------
    max_repair_attempts : int
        Max repair retries per node (default 2).
    record_latency_per_node : bool
        If True, measure actual wall-clock latency per node.
    debug : bool
        If True, print execution trace.
    """

    max_repair_attempts: int = 2
    record_latency_per_node: bool = True
    debug: bool = False

    def execute(
        self,
        graph: WorkflowGraph,
        evaluator: "MockEvaluator",
        profile_manager: "PrimitivePerformanceProfileManager | None" = None,
        profile_store: "ProfileStore | None" = None,
        config: "Optional[Any]" = None,
        rng: "random.Random | None" = None,
    ) -> WorkflowResult:
        """
        Execute a WorkflowGraph end-to-end.

        Parameters
        ----------
        graph : WorkflowGraph
            The workflow to execute.
        evaluator : MockEvaluator
            Evaluator instance (e.g. MockEvaluator).
        profile_manager : PrimitivePerformanceProfileManager | None
            Optional. Used for candidate selection within the graph.
            If None, uses the executor_id already set on each node.
        profile_store : ProfileStore | None
            ProfileStore for updating executor/evaluator profiles after execution.
        config : Any | None
            ExperimentConfig for evaluator selection, repair decisions, etc.
        rng : random.Random | None
            Random number generator for reproducibility.

        Returns
        -------
        WorkflowResult
            Complete execution result with per-node NodeResults and aggregates.
        """
        if rng is None:
            rng = random.Random()

        start_time = time.time()
        node_results: List[NodeResult] = []
        constraint_violations: List[dict] = []

        # Execution order (topological sort)
        exec_order = graph.execution_order()

        # Track which nodes are done / failed
        done_ids: set = set()
        failed_ids: set = set()
        node_outputs: Dict[str, Any] = {}  # node_id -> output data

        if self.debug:
            print(f"[WorkflowExecutor] Starting graph {graph.graph_id}")
            print(f"[WorkflowExecutor] Execution order: {exec_order}")

        for node_id in exec_order:
            node = graph.nodes[node_id]

            # Skip if already known to be permanently failed
            if node_id in failed_ids:
                node.status = NodeStatus.SKIPPED
                node_results.append(NodeResult(
                    node_id=node_id,
                    executor_id=node.executor_id,
                    evaluator_name=node.evaluator_name,
                    status=NodeStatus.SKIPPED,
                ))
                continue

            # Check predecessors
            preds = graph.predecessors(node_id)
            if not all(p in done_ids for p in preds):
                if self.debug:
                    print(f"[WorkflowExecutor] {node_id}: waiting on predecessors {preds}")
                continue

            # ---- Execute node ----
            result = self._execute_node(
                node=node,
                graph=graph,
                evaluator=evaluator,
                profile_manager=profile_manager,
                profile_store=profile_store,
                config=config,
                rng=rng,
            )
            node_results.append(result)

            # Update graph node status
            graph.nodes[node_id].status = result.status

            if result.status == NodeStatus.DONE:
                done_ids.add(node_id)
                node_outputs[node_id] = {
                    "quality": result.observed_quality,
                    "output": result.raw_result,
                }
            elif result.status == NodeStatus.FAILED:
                failed_ids.add(node_id)

            # Accumulate constraint violations
            if result.eval_pass is False:
                constraint_violations.append({
                    "node_id": node_id,
                    "error_type": result.error_type,
                    "quality": result.observed_quality,
                })

        execution_duration = time.time() - start_time

        # Build WorkflowResult
        overall_pass = (
            len(failed_ids) == 0
            and len([r for r in node_results if r.eval_pass]) == len([r for r in node_results if r.status == NodeStatus.DONE])
        )

        wf_result = WorkflowResult(
            graph_id=graph.graph_id,
            node_results=node_results,
            overall_pass=overall_pass,
            total_cost=sum(r.observed_cost for r in node_results),
            total_evaluator_cost=sum(r.evaluator_cost for r in node_results),
            constraint_violations=constraint_violations,
            execution_duration=execution_duration,
            metadata={
                **dict(graph.metadata),
                "edges": [(e.src, e.dst) for e in graph.edges],
            },
        )
        # compute_totals() must be called AFTER node_results is populated
        wf_result.compute_totals()

        if self.debug:
            print(
                f"[WorkflowExecutor] Done. pass={overall_pass} "
                f"latency={wf_result.total_latency:.3f}s cost={wf_result.total_cost:.3f} "
                f"repair={wf_result.repair_count}"
            )

        return wf_result

    # -------------------------------------------------------------------------
    # Per-node execution
    # -------------------------------------------------------------------------

    def _execute_node(
        self,
        node: WorkflowNode,
        graph: WorkflowGraph,
        evaluator: "MockEvaluator",
        profile_manager: "PrimitivePerformanceProfileManager | None",
        profile_store: "ProfileStore | None",
        config: "Optional[Any]",
        rng: "random.Random",
    ) -> NodeResult:
        """Execute a single WorkflowNode."""
        t0 = time.time()

        if self.debug:
            print(f"[WorkflowExecutor] Executing {node.node_id} ({node.node_type.value})")

        if node.node_type == NodeType.EXECUTOR:
            return self._execute_executor(node, graph, evaluator, profile_manager, profile_store, config, rng, t0)
        elif node.node_type == NodeType.VERIFIER:
            return self._execute_verifier(node, graph, evaluator, profile_store, config, rng, t0)
        elif node.node_type == NodeType.AGGREGATOR:
            preds = graph.predecessors(node.node_id)
            return self._execute_aggregator(node, len(preds), t0)
        elif node.node_type == NodeType.HUMAN_GATE:
            return self._execute_human_gate(node, config, rng, t0)
        else:
            return NodeResult(
                node_id=node.node_id,
                executor_id=node.executor_id,
                evaluator_name=None,
                status=NodeStatus.SKIPPED,
                observed_latency=time.time() - t0,
            )

    def _execute_executor(
        self,
        node: WorkflowNode,
        graph: WorkflowGraph,
        evaluator: "MockEvaluator",
        profile_manager: "PrimitivePerformanceProfileManager | None",
        profile_store: "ProfileStore | None",
        config: "Optional[Any]",
        rng: "random.Random",
        t0: float,
    ) -> NodeResult:
        """Execute an executor node (primary task execution)."""
        bucket = node.metadata.get("difficulty_bucket", "medium")
        prim = node.primitive_name

        # Resolve executor candidate
        if node.executor_id is None and profile_manager is not None:
            candidate = self._select_candidate_from_manager(
                node, profile_manager, config
            )
        else:
            candidate = node.candidate_name or "default"

        executor_id = node.executor_id or f"{prim}/{candidate}"

        # Get prediction from profile manager
        pred_quality = node.estimated_quality
        pred_cost = node.estimated_cost
        if profile_manager is not None:
            info = profile_manager.predict(prim, candidate, bucket)
            if info:
                pred_quality = info.get("quality_mean", pred_quality)
                pred_cost = info.get("cost_mean", pred_cost)

        # Run evaluator
        eval_name = self._select_evaluator(node, config)

        # Build task spec for evaluator
        from src.decomposer.task_decomposer import SubTaskSpec, ModalityType, difficulty_to_bucket

        st = SubTaskSpec(
            sub_task_id=node.node_id,
            primitive_name=node.primitive_name,
            difficulty=node.metadata.get("difficulty", 0.5),
            difficulty_bucket=node.metadata.get("difficulty_bucket", difficulty_to_bucket(node.metadata.get("difficulty", 0.5))),
            description=f"[workflow] {node.node_id}",
            predecessor_ids=list(node.depends_on),
            metadata={**node.metadata, "graph_id": graph.graph_id},
            constraints=node.metadata.get("constraints", []),
            input_modality=ModalityType.TEXT,
            intermediate_modality=None,
            evaluator_name=eval_name,
        )

        eval_result = evaluator.evaluate(
            candidate_name=candidate,
            primitive_name=node.primitive_name,
            difficulty_bucket=st.difficulty_bucket,
            task_id=graph.metadata.get("task_id", ""),
            node_id=node.node_id,
            task_spec=st,
            evaluator_name=eval_name,
            node_type=node.primitive_name,
        )

        # Attempt repair on failure (4-level: fail/escalate trigger repair)
        repaired = False
        repair_action = "none"
        final_result = eval_result

        if eval_result.eval_level in ("fail", "escalate"):
            for attempt in range(self.max_repair_attempts):
                new_result, new_info = self._repair_node(
                    node=node,
                    eval_result=eval_result,
                    eval_level=eval_result.eval_level,
                    candidate=candidate,
                    graph=graph,
                    evaluator=evaluator,
                    profile_manager=profile_manager,
                    profile_store=profile_store,
                    config=config,
                    rng=rng,
                )
                if new_result is not None and new_result.eval_pass:
                    repaired = True
                    repair_action = new_info.get("action", "unknown") if new_info else "upgraded_candidate"
                    final_result = new_result
                    candidate = new_info.get("candidate_name", candidate) if new_info else candidate
                    break
                elif new_result is not None:
                    final_result = new_result
                    break

        # Compute actual wall-clock latency (this is the observed value for learning)
        observed_latency = time.time() - t0

        # Update profile store with all three metrics
        if profile_store is not None:
            profile_store.update_executor_profile(
                tool_id=executor_id,
                difficulty=node.metadata.get("difficulty_bucket", "medium"),
                observed_quality=final_result.true_quality or final_result.observed_quality,
                observed_cost=final_result.true_cost or final_result.observed_cost,
                observed_latency=observed_latency,
            )
            profile_store.update_evaluator_profile(
                evaluator_id=eval_name,
                difficulty=node.metadata.get("difficulty_bucket", "medium"),
                observed_pass=final_result.eval_pass,
                true_pass=(final_result.true_quality or 0) >= evaluator.pass_threshold,
                evaluator_latency=final_result.evaluator_latency,
                evaluator_cost=final_result.evaluator_cost,
            )
        return NodeResult(
            node_id=node.node_id,
            executor_id=executor_id,
            evaluator_name=eval_name,
            status=NodeStatus.DONE if final_result.eval_pass else NodeStatus.FAILED,
            observed_quality=final_result.observed_quality,
            quality_score=final_result.quality_score,
            eval_pass=final_result.eval_pass,
            error_type=final_result.error_type,
            confidence=final_result.confidence,
            observed_cost=final_result.true_cost or final_result.observed_cost,
            observed_latency=observed_latency,
            evaluator_cost=final_result.evaluator_cost,
            evaluator_latency=final_result.evaluator_latency,
            true_quality=final_result.true_quality,
            predicted_quality=pred_quality,
            predicted_cost=pred_cost,
            repaired=repaired,
            repair_action=repair_action,
            raw_result={"eval_result": final_result},
        )

    def _execute_verifier(
        self,
        node: WorkflowNode,
        graph: WorkflowGraph,
        evaluator: "MockEvaluator",
        profile_store: "ProfileStore | None",
        config: "Optional[Any]",
        rng: "random.Random",
        t0: float,
    ) -> NodeResult:
        """
        Execute a verifier (evaluator) node.

        Verifier nodes evaluate the output of their predecessor execution node.
        Per method definition: verifier latency is NOT counted in total latency.
        """
        preds = graph.predecessors(node.node_id)
        target_id = preds[0] if preds else None

        # Find the execution node to verify
        exec_node_id = None
        for pred in preds:
            pred_node = graph.nodes.get(pred)
            if pred_node and pred_node.node_type == NodeType.EXECUTOR:
                exec_node_id = pred
                break

        if exec_node_id is None:
            return NodeResult(
                node_id=node.node_id,
                executor_id=None,
                evaluator_name=node.evaluator_name,
                status=NodeStatus.SKIPPED,
                observed_latency=time.time() - t0,
            )

        exec_node = graph.nodes[exec_node_id]
        eval_name = node.evaluator_name or "rule_eval"

        # Evaluate the executor node's output
        from src.decomposer.task_decomposer import SubTaskSpec, ModalityType, difficulty_to_bucket

        st = SubTaskSpec(
            sub_task_id=node.node_id,
            primitive_name=exec_node.primitive_name,
            difficulty=exec_node.metadata.get("difficulty", 0.5),
            difficulty_bucket=exec_node.metadata.get("difficulty_bucket", difficulty_to_bucket(exec_node.metadata.get("difficulty", 0.5))),
            description=f"[verifier] {node.node_id}",
            predecessor_ids=[exec_node_id],
            metadata={**exec_node.metadata, "graph_id": graph.graph_id, "verifier_node": True},
            constraints=exec_node.metadata.get("constraints", []),
            evaluator_name=eval_name,
        )

        eval_result = evaluator.evaluate(
            candidate_name=exec_node.candidate_name or "default",
            primitive_name=exec_node.primitive_name,
            difficulty_bucket=st.difficulty_bucket,
            task_id=graph.metadata.get("task_id", ""),
            node_id=node.node_id,
            task_spec=st,
            evaluator_name=eval_name,
            node_type=exec_node.primitive_name,
        )

        if profile_store is not None:
            profile_store.update_evaluator_profile(
                evaluator_id=eval_name,
                difficulty=st.difficulty_bucket,
                observed_pass=eval_result.eval_pass,
                true_pass=(eval_result.true_quality or 0) >= evaluator.pass_threshold,
                evaluator_latency=eval_result.evaluator_latency,
                evaluator_cost=eval_result.evaluator_cost,
            )

        return NodeResult(
            node_id=node.node_id,
            executor_id=None,
            evaluator_name=eval_name,
            status=NodeStatus.DONE if eval_result.eval_pass else NodeStatus.FAILED,
            observed_quality=eval_result.observed_quality,
            quality_score=eval_result.quality_score,
            eval_pass=eval_result.eval_pass,
            error_type=eval_result.error_type,
            confidence=eval_result.confidence,
            observed_cost=0.0,  # evaluator cost tracked separately
            observed_latency=0.0,  # ignored per method definition
            evaluator_cost=eval_result.evaluator_cost,
            evaluator_latency=eval_result.evaluator_latency,
            true_quality=eval_result.true_quality,
            repaired=False,
            repair_action="none",
        )

    def _execute_aggregator(
        self,
        node: WorkflowNode,
        preds_count: int,
        t0: float,
    ) -> NodeResult:
        """Execute an aggregator node (merge of parallel branches)."""
        # Aggregator: combines outputs of parallel branches
        # For MVP, assume aggregation always succeeds with small quality gain
        quality_gain = 0.02 * max(preds_count, 1)
        return NodeResult(
            node_id=node.node_id,
            executor_id=node.executor_id or "aggregator/merge",
            evaluator_name=None,
            status=NodeStatus.DONE,
            observed_quality=0.02 + quality_gain,
            quality_score=0.02 + quality_gain,
            eval_pass=True,
            observed_cost=0.1,
            observed_latency=time.time() - t0,
        )

    def _execute_human_gate(
        self,
        node: WorkflowNode,
        config: "Optional[Any]",
        rng: "random.Random",
        t0: float,
    ) -> NodeResult:
        """Simulate human-in-the-loop approval gate."""
        # Simulate human approval: 80% pass rate for high-stakes, 95% for normal
        approval_rate = 0.80 if config and hasattr(config, "enable_human_hitl") else 0.95
        approved = rng.random() < approval_rate

        return NodeResult(
            node_id=node.node_id,
            executor_id=None,
            evaluator_name=None,
            status=NodeStatus.DONE if approved else NodeStatus.FAILED,
            observed_quality=0.0,
            quality_score=0.0,
            eval_pass=approved,
            observed_cost=0.5,  # human approval overhead
            observed_latency=time.time() - t0,
            repaired=False,
            repair_action="none",
        )

    # -------------------------------------------------------------------------
    # Candidate & evaluator selection
    # -------------------------------------------------------------------------

    def _select_candidate_from_manager(
        self,
        node: WorkflowNode,
        profile_manager: "PrimitivePerformanceProfileManager",
        config: "Optional[Any]",
    ) -> str:
        """Select executor candidate using profile manager."""
        bucket = node.metadata.get("difficulty_bucket", "medium")

        if config and getattr(config, "use_pareto", False):
            frontier = profile_manager.pareto_frontier(node.primitive_name, bucket)
            if frontier:
                best = profile_manager.select_from_frontier(frontier)
                return best.get("candidate_name", "default")

        ranks = profile_manager.predict_all(node.primitive_name, bucket)
        if ranks:
            return ranks[0].get("candidate_name", "default")
        return "default"

    def _select_evaluator(
        self,
        node: WorkflowNode,
        config: "Optional[Any]",
    ) -> str:
        """Select evaluator for a node."""
        if node.evaluator_name:
            return node.evaluator_name
        bucket = node.metadata.get("difficulty_bucket", "medium")
        if bucket in ("hard", "extreme"):
            return "large_eval"
        elif bucket == "medium":
            return "small_eval"
        return "rule_eval"

    # -------------------------------------------------------------------------
    # Local repair
    # -------------------------------------------------------------------------

    def _repair_node(
        self,
        node: WorkflowNode,
        eval_result: "Any",
        eval_level: str,
        candidate: str,
        graph: WorkflowGraph,
        evaluator: "MockEvaluator",
        profile_manager: "PrimitivePerformanceProfileManager | None",
        profile_store: "ProfileStore | None",
        config: "Optional[Any]",
        rng: "random.Random",
    ) -> tuple["Any | None", dict | None]:
        """
        Attempt local repair for a failed node.

        4-level repair strategy (per method definition):
        - ESCALATE: evaluator upgrade FIRST (critical issue, evaluator uncertain/unreliable)
          → Strategy C: try evaluator upgrade, then candidate upgrade
        - FAIL: candidate upgrade FIRST (quality issue, try better executor)
          → Strategy B: try candidate upgrade along chain, then evaluator upgrade
        - WARN: no repair, continue (borderline quality, log warning)
        - PASS: should not reach here

        Returns (new_eval_result, info_dict) on success, (None, None) on failure.
        """
        from src.decomposer.task_decomposer import SubTaskSpec, difficulty_to_bucket
        from src.experiments.mvp_experiment import UPGRADE_CHAIN, _upgrade_evaluator

        error_type = getattr(eval_result, "error_type", None)
        confidence = getattr(eval_result, "confidence", 1.0)
        bucket = node.metadata.get("difficulty_bucket", "medium")

        def _make_st(eval_name: str) -> SubTaskSpec:
            return SubTaskSpec(
                sub_task_id=node.node_id,
                primitive_name=node.primitive_name,
                difficulty=node.metadata.get("difficulty", 0.5),
                difficulty_bucket=bucket,
                description=f"[repair:{eval_level}] {node.node_id}",
                predecessor_ids=list(node.depends_on),
                metadata={**node.metadata, "graph_id": graph.graph_id},
                evaluator_name=eval_name,
            )

        # === ESCALATE: evaluator upgrade FIRST ===
        if eval_level == "escalate":
            # Strategy C: upgrade evaluator immediately (critical issue)
            st = _make_st(self._select_evaluator(node, config))
            new_result = _upgrade_evaluator(
                st=st,
                candidate_name=candidate,
                prev_eval_result=eval_result,
                evaluator=evaluator,
                config=config,
                rng=rng,
            )
            if new_result is not None and new_result.eval_pass:
                return new_result, {"action": "upgraded_evaluator"}

            # Fallback: try candidate upgrade if evaluator upgrade failed
            chain = UPGRADE_CHAIN.get(node.primitive_name, [])
            chain_names = [c for c, _ in chain]
            if candidate in chain_names:
                current_idx = chain_names.index(candidate)
                for next_cand, _ in chain[current_idx + 1:]:
                    eval_name = self._select_evaluator(node, config)
                    st2 = _make_st(eval_name)
                    new_result2 = evaluator.evaluate(
                        candidate_name=next_cand,
                        primitive_name=node.primitive_name,
                        difficulty_bucket=st2.difficulty_bucket,
                        task_id=graph.metadata.get("task_id", ""),
                        node_id=node.node_id,
                        task_spec=st2,
                        evaluator_name=eval_name,
                        node_type=node.primitive_name,
                    )
                    if new_result2.eval_pass:
                        return new_result2, {"action": "upgraded_candidate", "candidate_name": next_cand}

            return None, None

        # === FAIL: candidate upgrade FIRST ===
        # Strategy B: upgrade candidate along chain
        chain = UPGRADE_CHAIN.get(node.primitive_name, [])
        chain_names = [c for c, _ in chain]

        if candidate in chain_names:
            current_idx = chain_names.index(candidate)
            for next_cand, _ in chain[current_idx + 1:]:
                eval_name = self._select_evaluator(node, config)
                st = _make_st(eval_name)
                new_result = evaluator.evaluate(
                    candidate_name=next_cand,
                    primitive_name=node.primitive_name,
                    difficulty_bucket=st.difficulty_bucket,
                    task_id=graph.metadata.get("task_id", ""),
                    node_id=node.node_id,
                    task_spec=st,
                    evaluator_name=eval_name,
                    node_type=node.primitive_name,
                )
                if new_result.eval_pass:
                    return new_result, {"action": "upgraded_candidate", "candidate_name": next_cand}

        # Strategy C: upgrade evaluator for format/safety errors
        if error_type in ("format_error", "unsafe_decision") or confidence < 0.5:
            st = _make_st(self._select_evaluator(node, config))
            new_result = _upgrade_evaluator(
                st=st,
                candidate_name=candidate,
                prev_eval_result=eval_result,
                evaluator=evaluator,
                config=config,
                rng=rng,
            )
            if new_result is not None and new_result.eval_pass:
                return new_result, {"action": "upgraded_evaluator"}

        return None, None
