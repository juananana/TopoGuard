"""
evaluation/__init__.py
=====================
"""
from .mock_evaluator import (
    MockEvaluator,
    EvaluationResult,
    MockLLMEvaluator,
    EVALUATOR_PROFILES,
)
from .evaluator_types import (
    BaseEvaluator,
    EvaluatorInput,
    EvaluatorOutput,
    NODE_TYPE_RUBRIC,
    get_rubric,
)
from .claude_evaluator import ClaudeEvaluator

__all__ = [
    # Core evaluator
    "MockEvaluator",
    "EvaluationResult",
    "MockLLMEvaluator",
    "EVALUATOR_PROFILES",
    # Evaluator abstraction (pluggable interface)
    "BaseEvaluator",
    "EvaluatorInput",
    "EvaluatorOutput",
    "NODE_TYPE_RUBRIC",
    "get_rubric",
    # Claude Opus evaluator
    "ClaudeEvaluator",
]
