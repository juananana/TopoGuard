"""
src/primitives/__init__.py
==========================
PrimitivePerformanceProfileManager sub-module.

Exports
-------
PrimitivePerformanceProfileManager : Core profile management class
PrimitiveProfile                 : Primitive module profile collection
CandidateProfile                 : Candidate (agent/combo) profile
FeedbackRecord                   : Execution feedback record
DifficultyMapper                 : Difficulty value -> bucket mapping
DifficultyBucket                : Difficulty bucket enum
BucketStats                     : Single-bucket statistics
AgentDef                        : Agent definition
InitPoint                       : Initial profile point
PredictionResult                : Internal prediction result dataclass
ProfileStore                    : External profile data interface
TopologyTemplate                : Sub-graph template definition
TemplateLibrary                  : Repository of topology templates
"""

from .feedback_record import FeedbackRecord
from .primitive_profile import (
    AgentDef,
    AgentComboProfile,
    BucketStats,
    CandidateProfile,
    DEFAULT_BUCKET_NAMES,
    DEFAULT_BOUNDARIES,
    DifficultyBucket,
    DifficultyMapper,
    InitPoint,
    PrimitiveProfile,
)
from .profile_manager import (
    PredictionResult,
    PrimitivePerformanceProfileManager,
)
from .profile_store import ExecutorProfile, EvaluatorProfile, ProfileStore
from .topology_template import (
    TopologyTemplate,
    TemplateNode,
    TemplateLibrary,
)

__all__ = [
    # Main class
    "PrimitivePerformanceProfileManager",
    # Profile components
    "PrimitiveProfile",
    "CandidateProfile",
    "AgentComboProfile",
    "BucketStats",
    "AgentDef",
    "InitPoint",
    # Difficulty mapping
    "DifficultyMapper",
    "DifficultyBucket",
    "DEFAULT_BUCKET_NAMES",
    "DEFAULT_BOUNDARIES",
    # Feedback & results
    "FeedbackRecord",
    "PredictionResult",
    # Profile store & templates (exp.md interfaces)
    "ProfileStore",
    "ExecutorProfile",
    "EvaluatorProfile",
    "TopologyTemplate",
    "TemplateNode",
    "TemplateLibrary",
]
