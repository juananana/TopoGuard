"""
task_decomposer.py
=================
Task Decomposer for MVP experiments.

Responsibility: task_description (str) -> List[SubTaskSpec]

In the real system, this is powered by an LLM or a trained classifier
that understands the task domain and outputs the task decomposition.

In the MVP, we use a keyword-based rule system.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING
import re
import random


# ---------------------------------------------------------------------------
# Modality Type
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Topology Pattern
# ---------------------------------------------------------------------------

class TopologyPattern(Enum):
    """Execution topology pattern — determines graph structure of sub-tasks."""
    LINEAR = "linear"
    PARALLEL_MERGE = "parallel_merge"   # parallel branches then merge
    FAN_OUT = "fan_out"                   # one-to-many broadcast


# ---------------------------------------------------------------------------
# Modality Type
# ---------------------------------------------------------------------------

class ModalityType(Enum):
    """Supported input/output modalities for tasks and intermediate results."""
    TEXT = "text"
    IMAGE = "image"
    TIME_SERIES = "time_series"
    TABULAR = "tabular"
    MULTIMODAL = "multimodal"


# ---------------------------------------------------------------------------
# Constraint Specifications
# ---------------------------------------------------------------------------

@dataclass
class ConstraintSpec:
    """
    Base class for a single constraint attached to a SubTaskSpec.

    Constraints are orthogonal to the profile manager: they are enforced
    by the orchestrator during candidate selection and validated by the
    evaluator after execution.

    Attributes
    ----------
    constraint_id : str
        Unique identifier for this constraint instance.
    constraint_type : str
        Type tag: "time_window" | "human_in_the_loop" | "mandatory_node" | "risk_boundary".
    enabled : bool
        Whether this constraint is currently active.
    description : str
        Human-readable description for logging and debugging.
    metadata : dict
        Extension fields (e.g., external system references, priority levels).
    """

    constraint_id: str
    constraint_type: str
    enabled: bool = True
    description: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class TimeWindowConstraint(ConstraintSpec):
    """
    Hard time budget: execution must complete within max_duration seconds.
    Exceeding this limit is treated as a constraint violation (timeout).

    Use cases:
    - Real-time inference pipelines
    - Streaming data processing with SLA requirements
    - Interactive applications with strict response-time budgets
    """

    constraint_type: str = "time_window"
    max_duration: float = 30.0  # seconds

    def __post_init__(self):
        if self.max_duration <= 0:
            raise ValueError(
                f"TimeWindowConstraint.max_duration must be positive, "
                f"got {self.max_duration}"
            )


@dataclass
class HumanInTheLoopConstraint(ConstraintSpec):
    """
    Human approval required at a specific point in the workflow.

    Use cases:
    - High-stakes financial decisions requiring domain-expert sign-off
    - Audit-required decisions (regulatory compliance)
    - Low-confidence predictions needing human confirmation before proceeding

    approval_point:
    - "candidate_selection": pause and wait for human to approve/reject the
      top-ranked candidate before execution.
    - "before_execute": human approves the execution plan before the candidate runs.
    - "before_repair": human reviews and approves the repair suggestion.
    """

    constraint_type: str = "human_in_the_loop"
    approval_point: str = "candidate_selection"  # "candidate_selection" | "before_execute" | "before_repair"
    required_role: str = "domain_expert"

    def __post_init__(self):
        valid_points = {"candidate_selection", "before_execute", "before_repair"}
        if self.approval_point not in valid_points:
            raise ValueError(
                f"HumanInTheLoopConstraint.approval_point must be one of "
                f"{valid_points}, got '{self.approval_point}'"
            )


@dataclass
class MandatoryNodeConstraint(ConstraintSpec):
    """
    A specific primitive or candidate MUST appear in the topology.

    Use cases:
    - Compliance: a rule-checker must always follow an ML model
    - Pipeline invariants: "forecast then analyze" must not be reordered
    - Safety overlays: a validator node is mandatory after any untrusted source

    at_difficulty_above: the constraint only applies when difficulty > this threshold.
    """

    constraint_type: str = "mandatory_node"
    required_primitive: str | None = None
    required_candidate: str | None = None
    at_difficulty_above: float | None = None  # e.g., 0.7 means constraint active for hard/extreme

    def is_active_for_difficulty(self, difficulty: float) -> bool:
        """Check if this mandatory-node constraint applies at the given difficulty."""
        if self.at_difficulty_above is None:
            return True
        return difficulty > self.at_difficulty_above


@dataclass
class RiskBoundaryConstraint(ConstraintSpec):
    """
    Safety floors and ceilings on quality/cost.

    Any observed value breaching these bounds is a constraint violation.

    Use cases:
    - Safety-critical systems: min_quality prevents dangerous underperformance
    - Cost-constrained environments: max_cost prevents budget overruns
    - SLA guarantees: max_cost_per_difficulty lets cost limits vary by difficulty

    Risk boundary violations cause eval_pass=False regardless of pass_threshold.
    """

    constraint_type: str = "risk_boundary"
    min_quality: float | None = None   # Observed quality below this = violation
    max_cost: float | None = None      # Observed cost above this = violation
    max_cost_per_difficulty: dict | None = None  # {"easy": 1.0, "hard": 5.0, ...}
    max_latency: float | None = None   # Observed latency (seconds) above this = violation

    def get_max_cost_for_bucket(self, difficulty_bucket: str) -> float | None:
        """Return the difficulty-specific cost ceiling, falling back to max_cost."""
        if self.max_cost_per_difficulty:
            return self.max_cost_per_difficulty.get(difficulty_bucket, self.max_cost)
        return self.max_cost

    def is_quality_violation(self, observed_quality: float) -> bool:
        return self.min_quality is not None and observed_quality < self.min_quality

    def is_cost_violation(self, observed_cost: float, difficulty_bucket: str) -> bool:
        limit = self.get_max_cost_for_bucket(difficulty_bucket)
        return limit is not None and observed_cost > limit

    def is_latency_violation(self, observed_latency: float) -> bool:
        return self.max_latency is not None and observed_latency > self.max_latency


# ---------------------------------------------------------------------------
# SubTask Specification
# ---------------------------------------------------------------------------

@dataclass
class SubTaskSpec:
    """
    Specification of a single sub-task within a decomposed task.

    Attributes
    ----------
    sub_task_id : str
        Unique ID for this sub-task within the task.
    primitive_name : str
        Which primitive module this sub-task belongs to.
        Maps to a Toolkit in PrimitivePerformanceProfileManager.
    difficulty : float
        Normalized difficulty in [0, 1].
    difficulty_bucket : str
        Discrete bucket name: "easy", "medium", "hard", "extreme".
    description : str
        Human-readable description of this sub-task.
    predecessor_ids : List[str]
        IDs of sub-tasks that must complete before this one.
        Used to construct the DAG. Empty = no dependencies.
    metadata : dict
        Extra fields (task_id, original task text, etc.).
    constraints : List[ConstraintSpec]
        Hard constraints attached to this sub-task (time window, human approval, etc.).
    input_modality : ModalityType
        The input modality of the task (text, image, time series, tabular, multimodal).
    intermediate_modality : ModalityType | None
        The modality of intermediate results passed between sub-tasks (optional).
        E.g., forecast produces time_series, which data_analysis then consumes.
    """

    sub_task_id: str
    primitive_name: str
    difficulty: float
    difficulty_bucket: str
    description: str
    predecessor_ids: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    # === Constraint & Multimodal fields ===
    constraints: List[ConstraintSpec] = field(default_factory=list)
    input_modality: ModalityType = ModalityType.TEXT
    intermediate_modality: ModalityType | None = None
    # === Evaluator selection ===
    evaluator_name: str | None = None   # "rule_eval", "small_eval", "large_eval"; None=auto

    def __post_init__(self):
        """Validate difficulty range."""
        if not (0.0 <= self.difficulty <= 1.0):
            raise ValueError(
                f"difficulty must be in [0, 1], got {self.difficulty}"
            )
        valid_buckets = {"easy", "medium", "hard", "extreme"}
        if self.difficulty_bucket not in valid_buckets:
            raise ValueError(
                f"difficulty_bucket must be one of {valid_buckets}, "
                f"got '{self.difficulty_bucket}'"
            )

    def get_active_constraints(self) -> List[ConstraintSpec]:
        """Return constraints that are both enabled and active for this difficulty."""
        active = []
        for c in self.constraints:
            if not c.enabled:
                continue
            if isinstance(c, MandatoryNodeConstraint):
                if not c.is_active_for_difficulty(self.difficulty):
                    continue
            active.append(c)
        return active

    def has_time_constraint(self) -> bool:
        return any(isinstance(c, TimeWindowConstraint) for c in self.get_active_constraints())

    def has_human_approval_required(self) -> bool:
        return any(isinstance(c, HumanInTheLoopConstraint) for c in self.get_active_constraints())

    def has_mandatory_node(self) -> bool:
        return any(isinstance(c, MandatoryNodeConstraint) for c in self.get_active_constraints())

    def has_risk_boundary(self) -> bool:
        return any(isinstance(c, RiskBoundaryConstraint) for c in self.get_active_constraints())


# ---------------------------------------------------------------------------
# Difficulty Mapper (standalone, matches profile_manager logic)
# ---------------------------------------------------------------------------

DEFAULT_BOUNDARIES = [0.0, 0.25, 0.5, 0.75, 1.0]
DEFAULT_BUCKET_NAMES = ["easy", "medium", "hard", "extreme"]


def difficulty_to_bucket(difficulty: float) -> str:
    """Map a [0, 1] difficulty value to a bucket name."""
    difficulty = max(0.0, min(1.0, difficulty))
    for i, boundary in enumerate(DEFAULT_BOUNDARIES[:-1]):
        if difficulty < DEFAULT_BOUNDARIES[i + 1]:
            return DEFAULT_BUCKET_NAMES[i]
    return DEFAULT_BUCKET_NAMES[-1]


# ---------------------------------------------------------------------------
# Task Decomposer
# ---------------------------------------------------------------------------

class TaskDecomposer:
    """
    MVP task decomposer: rule-based keyword matching + difficulty heuristics.

    Real decomposition methods (to be implemented):
    - LLM prompt: "Decompose this task into sub-tasks..."
    - Fine-tuned classifier: classify each clause of the task description
    - Template matching: domain-specific decomposition rules

    MVP strategy:
    1. Match keywords -> candidate primitives
    2. Heuristic difficulty detection from task text cues
    3. Extract constraints from task text (time window, human review, etc.)
    4. Extract modality information from task text
    5. Return a list of SubTaskSpec with simple linear chain (no DAG for MVP)

    Extension points:
    - Support explicit dependency syntax in task text (e.g. "first do X, then Y")
    - Add more primitives and keywords
    - Integrate with a real LLM for open-domain tasks
    """

    # Keyword -> primitive_name mapping
    # NOTE: All mapped primitives MUST exist in DEFAULT_GROUND_TRUTH
    # (src/evaluation/mock_evaluator.py).
    # Currently registered: forecast, state_parse, data_analysis
    KEYWORD_MAP: Dict[str, List[str]] = {
        "forecast":   ["forecast"],
        "predict":    ["forecast"],
        "time series": ["forecast"],
        "timeseries":  ["forecast"],
        "analyze":    ["data_analysis"],
        "analysis":   ["data_analysis"],
        "analytics":  ["data_analysis"],
        "fraud":      ["data_analysis"],     # fraud detection -> data_analysis
        "anomaly":    ["data_analysis"],       # anomaly detection -> data_analysis
        "detect":     ["data_analysis"],       # generic detection -> data_analysis
        "detection":  ["data_analysis"],
        "parse":      ["state_parse"],
        "parsing":    ["state_parse"],
        "classify":   ["data_analysis"],      # classification -> data_analysis
        "classification": ["data_analysis"],
        "sentiment":  ["data_analysis"],
    }

    # Difficulty signal keywords
    DIFFICULTY_UP_KEYWORDS = [
        "complex", "hard", "difficult", "long-term",
        "multi-step", "noisy", "large-scale", "extreme",
        "challenging", "intricate",
    ]
    DIFFICULTY_DOWN_KEYWORDS = [
        "simple", "basic", "short", "easy", "toy",
        "small", "clean", "straightforward",
    ]

    # Constraint trigger keywords -> (constraint factory, description)
    # These are checked after primitive extraction; each match creates one
    # ConstraintSpec instance with a unique constraint_id.
    CONSTRAINT_TRIGGER_PATTERNS: Dict[str, Tuple[type, dict]] = {
        # Time window constraints (urgency keywords)
        "urgent": (TimeWindowConstraint, {"max_duration": 5.0, "description": "Urgent: max 5s"}),
        "realtime": (TimeWindowConstraint, {"max_duration": 10.0, "description": "Realtime: max 10s"}),
        "real-time": (TimeWindowConstraint, {"max_duration": 10.0, "description": "Real-time: max 10s"}),
        "strict": (TimeWindowConstraint, {"max_duration": 15.0, "description": "Strict latency: max 15s"}),
        "under 5 second": (TimeWindowConstraint, {"max_duration": 5.0, "description": "5s time window"}),
        "within 10 second": (TimeWindowConstraint, {"max_duration": 10.0, "description": "10s time window"}),
        "5s latency": (TimeWindowConstraint, {"max_duration": 5.0, "description": "5s latency constraint"}),
        "10s latency": (TimeWindowConstraint, {"max_duration": 10.0, "description": "10s latency constraint"}),
        "strict 5s": (TimeWindowConstraint, {"max_duration": 5.0, "description": "Strict 5s limit"}),
        # Human-in-the-loop constraints
        "human review": (HumanInTheLoopConstraint, {"approval_point": "candidate_selection", "description": "Human review required"}),
        "require human": (HumanInTheLoopConstraint, {"approval_point": "candidate_selection", "description": "Human approval required"}),
        "human approval": (HumanInTheLoopConstraint, {"approval_point": "candidate_selection", "description": "Human approval required"}),
        "escalate to human": (HumanInTheLoopConstraint, {"approval_point": "before_execute", "description": "Escalate to human"}),
        "domain expert": (HumanInTheLoopConstraint, {"approval_point": "candidate_selection", "description": "Domain expert review"}),
        "audit": (HumanInTheLoopConstraint, {"approval_point": "before_execute", "description": "Audit trail required"}),
        "regulatory": (HumanInTheLoopConstraint, {"approval_point": "before_execute", "description": "Regulatory compliance check"}),
        # Mandatory node constraints
        "forecast then": (MandatoryNodeConstraint, {"required_primitive": "forecast", "description": "Forecast must be first"}),
        "then analyze": (MandatoryNodeConstraint, {"required_primitive": "data_analysis", "description": "Analysis must follow"}),
        "rule-checker": (MandatoryNodeConstraint, {"required_candidate": "rule_parser", "at_difficulty_above": 0.5, "description": "Rule-checker mandatory for hard+"}),
        "validator": (MandatoryNodeConstraint, {"required_primitive": "state_parse", "description": "Validator mandatory"}),
        "mandatory": (MandatoryNodeConstraint, {"description": "Mandatory constraint in task"}),
        # Risk boundary constraints
        "high-stakes": (RiskBoundaryConstraint, {"min_quality": 0.85, "description": "High-stakes: quality floor 0.85"}),
        "safety-critical": (RiskBoundaryConstraint, {"min_quality": 0.95, "max_cost": 10.0, "description": "Safety-critical: quality floor 0.95, cost cap 10.0"}),
        "financial": (RiskBoundaryConstraint, {"min_quality": 0.90, "max_cost": 5.0, "description": "Financial: quality floor 0.90, cost cap 5.0"}),
        "healthcare": (RiskBoundaryConstraint, {"min_quality": 0.95, "description": "Healthcare: quality floor 0.95"}),
        "patient": (RiskBoundaryConstraint, {"min_quality": 0.95, "description": "Patient safety: quality floor 0.95"}),
        "fraud detection": (RiskBoundaryConstraint, {"min_quality": 0.90, "description": "Fraud: quality floor 0.90"}),
    }

    # Modality trigger keywords
    MODALITY_PATTERNS: Dict[str, ModalityType] = {
        "image": ModalityType.IMAGE,
        "photo": ModalityType.IMAGE,
        "picture": ModalityType.IMAGE,
        "img": ModalityType.IMAGE,
        "photo": ModalityType.IMAGE,
        "time series": ModalityType.TIME_SERIES,
        "timeseries": ModalityType.TIME_SERIES,
        "sensor": ModalityType.TIME_SERIES,
        "sensor data": ModalityType.TIME_SERIES,
        "telemetry": ModalityType.TIME_SERIES,
        "csv": ModalityType.TABULAR,
        "table": ModalityType.TABULAR,
        "tabular": ModalityType.TABULAR,
        "spreadsheet": ModalityType.TABULAR,
        "transaction data": ModalityType.TABULAR,
        "multimodal": ModalityType.MULTIMODAL,
        "image and text": ModalityType.MULTIMODAL,
        "vision and language": ModalityType.MULTIMODAL,
    }

    # Intermediate modality inference: output of one primitive -> input of next
    # (primitive_name -> ModalityType of its intermediate output)
    PRIMITIVE_INTERMEDIATE_MODALITY: Dict[str, ModalityType] = {
        "forecast": ModalityType.TIME_SERIES,        # produces forecast time series
        "state_parse": ModalityType.TEXT,              # produces parsed text/structured text
        "data_analysis": ModalityType.TABULAR,         # produces analysis report/table
        "aggregator": ModalityType.TEXT,               # produces aggregated text/summary
    }

    def __init__(
        self,
        random_seed: int | None = None,
        default_primitive: str = "state_parse",
    ):
        """
        Parameters
        ----------
        random_seed : int | None
            For reproducible difficulty assignment.
        default_primitive : str
            Primitive to use when no keywords match.
        """
        self.default_primitive = default_primitive
        self.rng = random.Random(random_seed)
        self._constraint_counter = 0

    # Mapping from primitive to task_type (per exp.md interface 2 / ProfileStore lookup)
    PRIMITIVE_TO_TASK_TYPE: Dict[str, str] = {
        "forecast":      "time_series",
        "state_parse":   "text_analysis",
        "data_analysis": "tabular_analysis",
    }

    def decompose(
        self,
        task_description: str,
        extract_constraints: bool = True,
    ) -> Tuple[List[SubTaskSpec], str]:
        """
        Decompose a task description into a list of SubTaskSpec.

        MVP algorithm:
        1. Find all matching primitives (deduplicated)
        2. Infer difficulty from keywords, default to medium (0.5)
        3. Extract constraints and modality information (if extract_constraints=True)
        4. Generate one SubTaskSpec per unique primitive found
        5. If no keywords match, generate one SubTaskSpec with default_primitive

        Parameters
        ----------
        task_description : str
        extract_constraints : bool
            If True, extract hard constraints from task text (default).
            Set to False when running baseline experiments without constraints.

        Returns
        -------
        Tuple[List[SubTaskSpec], str]
            List of sub-tasks in execution order (linear chain for MVP),
            and the detected task_type ("time_series" | "text_analysis" | "tabular_analysis" | "unknown").
        """
        task_lower = task_description.lower()

        # Step 1: collect matching primitives (dedup)
        matched_primitives: List[str] = []
        for keyword, primitives in self.KEYWORD_MAP.items():
            if keyword in task_lower:
                for p in primitives:
                    if p not in matched_primitives:
                        matched_primitives.append(p)

        # If no match, fall back to default
        if not matched_primitives:
            matched_primitives = [self.default_primitive]

        # Derive task_type from the primary (first) primitive
        primary_prim = matched_primitives[0]
        task_type = self.PRIMITIVE_TO_TASK_TYPE.get(primary_prim, "unknown")
        # Override to multimodal if multimodal modality detected
        if ModalityType.MULTIMODAL.value in task_lower or "image and text" in task_lower:
            task_type = "multimodal"

        # Step 2: infer difficulty from keyword signals
        difficulty = self._infer_difficulty(task_lower)

        # Step 3: extract constraints and modality
        constraints = self._extract_constraints(task_lower) if extract_constraints else []
        input_mod, inter_mod = self._extract_modality(task_lower, matched_primitives)

        # Step 4: build SubTaskSpec list (linear chain: each depends on previous)
        sub_tasks: List[SubTaskSpec] = []
        for i, prim in enumerate(matched_primitives):
            predecessor_ids: List[str] = (
                [sub_tasks[-1].sub_task_id] if sub_tasks else []
            )
            # Determine intermediate modality: if this is not the last task,
            # the intermediate output goes to the next primitive
            next_inter_mod = None
            if inter_mod is not None:
                next_inter_mod = inter_mod
            elif i < len(matched_primitives) - 1:
                # Infer from PRIMITIVE_INTERMEDIATE_MODALITY
                next_inter_mod = self.PRIMITIVE_INTERMEDIATE_MODALITY.get(prim)

            sub_tasks.append(SubTaskSpec(
                sub_task_id=f"st_{i}",
                primitive_name=prim,
                difficulty=difficulty,
                difficulty_bucket=difficulty_to_bucket(difficulty),
                description=f"{prim} sub-task (extracted from: '{task_description[:50]}...')",
                predecessor_ids=predecessor_ids,
                metadata={
                    "task_description": task_description,
                    "matched_keywords": [
                        kw for kw in self.KEYWORD_MAP
                        if kw in task_lower
                    ],
                    "constraint_ids": [c.constraint_id for c in constraints],
                },
                # Constraint & multimodal fields
                constraints=constraints,
                input_modality=input_mod,
                intermediate_modality=next_inter_mod,
            ))

        # === Optional topology transformation ===
        pattern = self._suggest_topology_pattern(task_lower)
        if pattern == TopologyPattern.PARALLEL_MERGE:
            sub_tasks = self._build_parallel_merge_topology(sub_tasks)

        return sub_tasks, task_type

    # -------------------------------------------------------------------------
    # Topology pattern helpers
    # -------------------------------------------------------------------------

    def _suggest_topology_pattern(self, task_lower: str) -> TopologyPattern:
        """
        Decide execution topology pattern based on task text keywords.

        Minimal implementation: parallel_merge when the task explicitly asks for
        multi-source analysis or ensemble of approaches; otherwise linear.
        """
        PARALLEL_KEYWORDS = [
            "dual", "both", "parallel", "multi-source", "multi-model",
            "ensemble", "综合", "两种", "三方", "对比分析", "compare",
            "multi-modal", "cross-analytics",
        ]
        if any(kw in task_lower for kw in PARALLEL_KEYWORDS):
            return TopologyPattern.PARALLEL_MERGE
        return TopologyPattern.LINEAR

    def _build_parallel_merge_topology(
        self, base_tasks: List[SubTaskSpec]
    ) -> List[SubTaskSpec]:
        """
        Transform a base task chain into a parallel-merge topology.

        Cases:
        - len(base) == 1:  S1 → [S1a, S1b] → aggregator
        - len(base) >= 2:  S1 → [S2a, S2b] → aggregator → [S3...]
        """
        from dataclasses import replace
        tasks: List[SubTaskSpec] = []

        if len(base_tasks) == 1:
            # Single sub-task: create two parallel branches of the same type
            st1 = base_tasks[0]
            branch_ids = []
            for suffix in ("a", "b"):
                branch_id = f"{st1.sub_task_id}_{suffix}"
                branch_ids.append(branch_id)
                tasks.append(SubTaskSpec(
                    sub_task_id=branch_id,
                    primitive_name=st1.primitive_name,
                    difficulty=st1.difficulty,
                    difficulty_bucket=st1.difficulty_bucket,
                    description=f"[{suffix.upper()}] {st1.description}",
                    predecessor_ids=[],
                    metadata={},
                    constraints=st1.constraints,
                    input_modality=st1.input_modality,
                    intermediate_modality=st1.intermediate_modality,
                ))
            # Aggregator merges the two parallel branches
            tasks.append(SubTaskSpec(
                sub_task_id="st_merge",
                primitive_name="aggregator",
                difficulty=0.5,
                difficulty_bucket="medium",
                description="Aggregated result from parallel branches",
                predecessor_ids=branch_ids,
                metadata={},
                constraints=[],
                input_modality=ModalityType.MULTIMODAL,
                intermediate_modality=ModalityType.TEXT,
            ))
            return tasks

        # First node: unchanged
        tasks.append(base_tasks[0])

        # Second node → two parallel branches (both start from tasks[0])
        st2 = base_tasks[1]
        branch_ids = []
        first_id = tasks[0].sub_task_id  # both branches start from first node
        for suffix in ("a", "b"):
            branch_id = f"{st2.sub_task_id}_{suffix}"
            branch_ids.append(branch_id)
            tasks.append(SubTaskSpec(
                sub_task_id=branch_id,
                primitive_name=st2.primitive_name,
                difficulty=st2.difficulty,
                difficulty_bucket=st2.difficulty_bucket,
                description=f"[{suffix.upper()}] {st2.description}",
                predecessor_ids=[first_id],
                metadata={},
                constraints=st2.constraints,
                input_modality=st2.input_modality,
                intermediate_modality=st2.intermediate_modality,
            ))

        # Aggregator node: merges parallel branch outputs
        merge_id = "st_merge"
        tasks.append(SubTaskSpec(
            sub_task_id=merge_id,
            primitive_name="aggregator",
            difficulty=0.5,
            difficulty_bucket="medium",
            description="Aggregated result from parallel branches",
            predecessor_ids=branch_ids,
            metadata={},
            constraints=[],
            input_modality=ModalityType.MULTIMODAL,
            intermediate_modality=ModalityType.TEXT,
        ))

        # Remaining nodes: rewire predecessor_ids to point to merge node
        for st in base_tasks[2:]:
            tasks.append(replace(
                st,
                sub_task_id=f"{st.sub_task_id}_m",
                predecessor_ids=[tasks[-1].sub_task_id],
            ))

        return tasks

    def _extract_constraints(self, text_lower: str) -> List[ConstraintSpec]:
        """
        Extract all constraint specifications from the lowercased task text.

        Each matching keyword creates one ConstraintSpec with a unique ID.
        Patterns are checked longest-first to avoid partial matches.

        Deduplication by constraint_type: if multiple patterns produce the same
        constraint_type, keep only the most restrictive one.
        For TimeWindowConstraint: smaller max_duration = more restrictive
        For RiskBoundaryConstraint: larger min_quality = more restrictive
        """
        constraints: List[ConstraintSpec] = []
        seen_types: Dict[str, ConstraintSpec] = {}

        # Sort patterns by length descending (prefer longer, more specific matches)
        sorted_patterns = sorted(
            self.CONSTRAINT_TRIGGER_PATTERNS.keys(),
            key=len,
            reverse=True,
        )

        for pattern in sorted_patterns:
            if pattern in text_lower:
                constraint_cls, kwargs = self.CONSTRAINT_TRIGGER_PATTERNS[pattern]
                self._constraint_counter += 1
                constraint_id = f"c_{self._constraint_counter:03d}_{constraint_cls.__name__}"
                try:
                    if constraint_cls is TimeWindowConstraint:
                        c = TimeWindowConstraint(constraint_id=constraint_id, **kwargs)
                    elif constraint_cls is HumanInTheLoopConstraint:
                        c = HumanInTheLoopConstraint(constraint_id=constraint_id, **kwargs)
                    elif constraint_cls is MandatoryNodeConstraint:
                        c = MandatoryNodeConstraint(constraint_id=constraint_id, **kwargs)
                    elif constraint_cls is RiskBoundaryConstraint:
                        c = RiskBoundaryConstraint(constraint_id=constraint_id, **kwargs)
                    else:
                        c = ConstraintSpec(
                            constraint_id=constraint_id,
                            constraint_type=constraint_cls.__name__,
                            **kwargs,
                        )

                    # Deduplicate by constraint_type: keep the most restrictive
                    ctype = c.constraint_type
                    if ctype not in seen_types:
                        seen_types[ctype] = c
                    else:
                        existing = seen_types[ctype]
                        if self._is_more_restrictive(c, existing):
                            seen_types[ctype] = c
                        # else keep existing (less restrictive)

                except (ValueError, TypeError):
                    pass

        return list(seen_types.values())

    def _is_more_restrictive(
        self, candidate: ConstraintSpec, existing: ConstraintSpec
    ) -> bool:
        """
        Return True if candidate is more restrictive than existing.
        Used for deduplication (keep the stricter constraint of the same type).
        """
        if isinstance(candidate, TimeWindowConstraint) and isinstance(existing, TimeWindowConstraint):
            return candidate.max_duration < existing.max_duration
        if isinstance(candidate, RiskBoundaryConstraint) and isinstance(existing, RiskBoundaryConstraint):
            # More restrictive = higher min_quality or lower max_cost
            cand_min_q = candidate.min_quality or -1.0
            exist_min_q = existing.min_quality or -1.0
            cand_max_c = candidate.max_cost or float('inf')
            exist_max_c = existing.max_cost or float('inf')
            return (cand_min_q > exist_min_q) or (cand_max_c < exist_max_c)
        if isinstance(candidate, HumanInTheLoopConstraint):
            # Keep the one with more specific approval_point
            return len(candidate.approval_point) > len(existing.approval_point)
        if isinstance(candidate, MandatoryNodeConstraint):
            # Keep if it has a required_candidate (more specific) or required_primitive
            return (candidate.required_candidate is not None) or (
                candidate.required_primitive is not None and existing.required_primitive is None
            )
        return False

    def _extract_modality(
        self,
        text_lower: str,
        matched_primitives: List[str],
    ) -> Tuple[ModalityType, ModalityType | None]:
        """
        Extract input and intermediate modalities from the task text.

        Parameters
        ----------
        text_lower : str
            Lowercased task description.
        matched_primitives : List[str]
            Primitives matched from KEYWORD_MAP.

        Returns
        -------
        (input_modality, intermediate_modality)
            input_modality: dominant modality in the task description (default TEXT).
            intermediate_modality: modality of the output if inferable from context
                (e.g., "image and text" -> MULTIMODAL).
        """
        input_mod = ModalityType.TEXT
        intermediate_mod: ModalityType | None = None

        # Find the most specific modality match (longest pattern wins)
        best_input_pattern = ""
        for pattern, mod in self.MODALITY_PATTERNS.items():
            if pattern in text_lower and len(pattern) > len(best_input_pattern):
                best_input_pattern = pattern
                input_mod = mod

        # Special inference for intermediate modality
        # "image and text pairs" or "vision and language" -> intermediate MULTIMODAL
        multimodal_pairs = [
            "image and text", "vision and language",
            "multimodal input", "multimodal",
        ]
        for phrase in multimodal_pairs:
            if phrase in text_lower:
                intermediate_mod = ModalityType.MULTIMODAL
                input_mod = ModalityType.MULTIMODAL
                break

        return input_mod, intermediate_mod

    def _infer_difficulty(self, text_lower: str) -> float:
        """
        Infer normalized difficulty [0, 1] from task description keywords.

        Rules:
        - Count difficulty-up signals: +0.15 per keyword (capped at +0.30)
        - Count difficulty-down signals: -0.15 per keyword (capped at -0.30)
        - Base: 0.5 (medium)
        - Clamp result to [0.1, 0.9] (never fully trivial or impossible)
        """
        base = 0.5
        up = sum(1 for kw in self.DIFFICULTY_UP_KEYWORDS if kw in text_lower)
        down = sum(1 for kw in self.DIFFICULTY_DOWN_KEYWORDS if kw in text_lower)

        difficulty = base + 0.15 * up - 0.15 * down
        difficulty = max(0.1, min(0.9, difficulty))

        # Add small random noise for diversity (only if seed is set)
        if self.rng:
            difficulty = difficulty + self.rng.uniform(-0.05, 0.05)
            difficulty = max(0.1, min(0.9, difficulty))

        return round(difficulty, 3)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("TaskDecomposer Demo (with Constraints & Multimodal)")
    print("=" * 60)

    decomposer = TaskDecomposer(random_seed=42)

    tasks = [
        # Basic
        "Forecast next 30 days temperature using time series analysis.",
        "Simple data analysis of clean CSV file.",
        "Complex multi-step classification with noisy labels on large-scale dataset.",
        "Parse structured text and extract entities.",
        "Easy rule-based parsing task.",
        # --- Constraint demonstrations ---
        # Time window
        "Analyze fraud in transaction data under 5 second time window.",
        "Detect anomalies in sensor readings with realtime latency requirement.",
        "Classify financial transactions with strict 5s latency.",
        # Human-in-the-loop
        "Forecast quarterly revenue and escalate to human for final approval.",
        "Assess loan default risk and require human review for high-stakes decisions.",
        "Predict stock prices with domain expert audit trail.",
        # Mandatory node
        "First forecast market trends, then analyze sentiment from news articles.",
        "Parse structured data, then run ML classification with mandatory rule-checker.",
        "Parse messy data, validator is mandatory for hard difficulty tasks.",
        # Risk boundary
        "Predict patient readmission with safety-critical quality floor of 0.95.",
        "Classify financial transactions with high-stakes min quality 0.90 and max cost 5.0.",
        # Multimodal
        "Classify image and text pairs for sentiment analysis (multimodal input).",
        "Forecast time series from sensor data with tabular metadata overlay.",
        "Detect anomalies in image and text pairs from surveillance reports.",
    ]

    for task in tasks:
        print(f"\nTask: '{task}'")
        sub_tasks = decomposer.decompose(task)
        for st in sub_tasks:
            constraint_types = [c.constraint_type for c in st.constraints]
            active_constraints = [c.constraint_type for c in st.get_active_constraints()]
            print(
                f"  [{st.sub_task_id}] primitive={st.primitive_name:15s} "
                f"difficulty={st.difficulty:.2f} [{st.difficulty_bucket:8s}] "
                f"deps={st.predecessor_ids}"
            )
            print(
                f"         modality={st.input_modality.value}  "
                f"intermediate={st.intermediate_modality.value if st.intermediate_modality else '-'}  "
                f"constraints={constraint_types or 'none'}"
            )
            if active_constraints != constraint_types:
                print(f"         active_constraints={active_constraints}")

    print("\nDemo complete.")
