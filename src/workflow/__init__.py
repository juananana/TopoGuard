"""
workflow/__init__.py
===================
TopoGuard Workflow Graph Framework.

Core abstractions:
- WorkflowNode   : explicit node in the graph
- WorkflowEdge   : directed edge between nodes
- WorkflowGraph  : complete graph representation G = (V, E, τ, φ)
- WorkflowResult : execution result of a WorkflowGraph

Sub-modules:
- workflow_graph  : core data structures
- workflow_builder: build WorkflowGraph from TopologyTemplate or task spec
"""

from .workflow_graph import (
    WorkflowNode,
    WorkflowEdge,
    EdgeType,
    NodeType,
    NodeStatus,
    WorkflowGraph,
    WorkflowProfile,
    WorkflowResult,
    NodeResult,
)

from .workflow_builder import (
    WorkflowBuilder,
)

from .workflow_executor import (
    WorkflowExecutor,
)

__all__ = [
    # Core graph structures
    "WorkflowNode",
    "WorkflowEdge",
    "EdgeType",
    "NodeType",
    "NodeStatus",
    "WorkflowGraph",
    "WorkflowProfile",
    # Execution results
    "WorkflowResult",
    "NodeResult",
    # Builder
    "WorkflowBuilder",
    # Executor
    "WorkflowExecutor",
]
