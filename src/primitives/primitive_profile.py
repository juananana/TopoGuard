"""
primitive_profile.py
====================
原语性能画像子模块。

包含：
- DifficultyBucket / DifficultyMapper：难度值 → 难度桶的映射
- BucketStats：单个难度桶内的统计量
- CandidateProfile：单个 agent / agent组合的性能画像
- PrimitiveProfile：一个原语模块下所有候选单元的画像集合
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import math


# ---------------------------------------------------------------------------
# 难度桶定义
# ---------------------------------------------------------------------------

# 默认难度桶名称（按难度从低到高排列）
DEFAULT_BUCKET_NAMES = ["easy", "medium", "hard", "extreme"]

# 难度桶边界（递增序列，右侧用 +∞ 截断）
# 语义：数值越大越难（extreme），数值越小越简单（easy）
#   [0.0, 0.25)  → easy     (0.0~0.25 = 简单)
#   [0.25, 0.5)  → medium
#   [0.5, 0.75)  → hard
#   [0.75, 1.0] → extreme  (0.75~1.0 = 极难)
DEFAULT_BOUNDARIES = [0.0, 0.25, 0.5, 0.75, 1.0]


class DifficultyBucket(Enum):
    """难度桶枚举，与 DEFAULT_BUCKET_NAMES 一一对应。"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXTREME = "extreme"


class DifficultyMapper:
    """
    将连续的难度值（0~1）映射到离散难度桶。

    语义：数值越大越难（extreme），数值越小越简单（easy）。
    默认边界：[0.0, 0.25, 0.5, 0.75, 1.0]
      easy:    difficulty in [0.0, 0.25)
      medium:  difficulty in [0.25, 0.5)
      hard:    difficulty in [0.5, 0.75)
      extreme: difficulty in [0.75, 1.0]
    """

    def __init__(
        self,
        bucket_names: List[str] | None = None,
        boundaries: List[float] | None = None,
    ):
        self.bucket_names = bucket_names or DEFAULT_BUCKET_NAMES
        self.boundaries = boundaries or DEFAULT_BOUNDARIES
        assert len(self.bucket_names) == len(self.boundaries) - 1, (
            "桶数量必须等于边界数量减一"
        )

    def map(self, difficulty: float) -> str:
        """
        将 difficulty 映射到对应的桶名。

        Parameters
        ----------
        difficulty : float
            难度值，应在 [0, 1] 范围内（超出范围会被截断）。

        Returns
        -------
        str
            对应的桶名。
        """
        difficulty = max(0.0, min(1.0, difficulty))
        for i, boundary in enumerate(self.boundaries[:-1]):
            if difficulty < self.boundaries[i + 1]:
                return self.bucket_names[i]
        return self.bucket_names[-1]

    @property
    def num_buckets(self) -> int:
        return len(self.bucket_names)

    def bucket_names_list(self) -> List[str]:
        return self.bucket_names

    def is_valid_bucket_name(self, name: str) -> bool:
        """检查是否为合法的桶名。"""
        return name in self.bucket_names

    def is_valid_difficulty_value(self, value) -> bool:
        """
        检查 difficulty 值是否合法。
        接受 float [0, 1] 或已注册的桶名（str）。
        """
        if isinstance(value, float) and 0.0 <= value <= 1.0:
            return True
        if isinstance(value, str) and self.is_valid_bucket_name(value):
            return True
        return False

    def normalize_difficulty(self, difficulty) -> str:
        """
        将 difficulty 统一转换为桶名返回。
        - float [0,1] → 映射到桶名
        - str（合法桶名）→ 直接返回
        - 其他 → 抛 ValueError
        """
        if isinstance(difficulty, str):
            if self.is_valid_bucket_name(difficulty):
                return difficulty
            raise ValueError(
                f"Invalid difficulty bucket name '{difficulty}'. "
                f"Valid names: {self.bucket_names}"
            )
        if isinstance(difficulty, float):
            return self.map(difficulty)
        raise ValueError(
            f"Difficulty must be a float in [0,1] or a bucket name str, "
            f"got {type(difficulty).__name__}: {difficulty}"
        )


# ---------------------------------------------------------------------------
# 桶内统计量
# ---------------------------------------------------------------------------

@dataclass
class BucketStats:
    """
    单个难度桶内的统计量。

    基于原始观测数据计算帕累托前沿，不使用 EMA。
    存储所有 (quality, cost) 观测点，在线维护均值和方差。
    """

    bucket_name: str
    quality_mean: float = 0.0
    cost_mean: float = 0.0
    latency_mean: float = 0.0     # 工作流延迟（秒），用于 Q(G;X) 三目标优化
    support_count: int = 0        # 真实反馈样本数
    n_prior: int = 0              # 先验注入样本数（来自 init_curve）
    # 原始观测存储：每条 (quality, cost, latency)
    observations: list = field(default_factory=list)

    @property
    def quality_std(self) -> float:
        if self.support_count < 2:
            return float("nan")
        mean = self.quality_mean
        variance = sum((q - mean) ** 2 for q, *_ in self.observations) / (self.support_count - 1)
        return math.sqrt(variance)

    @property
    def cost_std(self) -> float:
        if self.support_count < 2:
            return float("nan")
        mean = self.cost_mean
        variance = sum((obs[1] - mean) ** 2 for obs in self.observations) / (self.support_count - 1)
        return math.sqrt(variance)

    @property
    def latency_std(self) -> float:
        """Latency 标准差（latency 为第 3 元素，不存在时返回 0.0）。"""
        if self.support_count < 2:
            return float("nan")
        mean = self.latency_mean
        variance = sum(
            (obs[2] - mean) ** 2 for obs in self.observations if len(obs) >= 3
        )
        n = sum(1 for obs in self.observations if len(obs) >= 3)
        if n < 2:
            return float("nan")
        return math.sqrt(variance / (n - 1))

    @property
    def uncertainty(self) -> float:
        """
        返回质量的不确定度估计。
        样本少时返回较大的默认值，以反映高不确定性。
        """
        if self.support_count < 2:
            return 1.0
        std = self.quality_std
        if math.isnan(std):
            return 1.0
        return min(std, 1.0)

    def add_observation(
        self,
        observed_quality: float,
        observed_cost: float,
        observed_latency: float = 0.0,
    ) -> None:
        """
        追加一条观测，重新计算均值（quality / cost / latency 三目标）。

        Parameters
        ----------
        observed_quality : float
            本次观测到的质量。
        observed_cost : float
            本次观测到的成本。
        observed_latency : float
            本次观测到的工作流延迟（秒）。
        """
        self.observations.append((observed_quality, observed_cost, observed_latency))
        self.support_count = len(self.observations)
        self.quality_mean = sum(obs[0] for obs in self.observations) / self.support_count
        self.cost_mean = sum(obs[1] for obs in self.observations) / self.support_count
        self.latency_mean = sum(obs[2] for obs in self.observations) / self.support_count

    def set_prior(
        self,
        quality: float,
        cost: float,
        latency: float = 0.0,
    ) -> None:
        """
        设置先验值（冷启动）。如果还没有观测数据，用先验初始化均值。
        先验值也作为一条观测记录，但标记为 n_prior。
        """
        self.observations.append((quality, cost, latency))
        self.n_prior += 1
        self.support_count = len(self.observations)
        self.quality_mean = sum(obs[0] for obs in self.observations) / self.support_count
        self.cost_mean = sum(obs[1] for obs in self.observations) / self.support_count
        self.latency_mean = sum(obs[2] for obs in self.observations) / self.support_count

    def merge_from(self, other: "BucketStats") -> None:
        """
        将另一个 BucketStats 的数据合并进来（用于跨进程的聚合场景）。
        """
        if not other.observations:
            return
        self.observations.extend(other.observations)
        self.support_count = len(self.observations)
        self.n_prior += other.n_prior
        self.quality_mean = sum(obs[0] for obs in self.observations) / self.support_count
        self.cost_mean = sum(obs[1] for obs in self.observations) / self.support_count
        self.latency_mean = sum(obs[2] for obs in self.observations) / self.support_count


# ---------------------------------------------------------------------------
# 初始画像点 & Agent 组合
# ---------------------------------------------------------------------------

@dataclass
class InitPoint:
    """
    初始画像点（经验规则注入），用于冷启动。

    在 register_candidate 时注入，作用是给每个难度桶一个先验基线，
    避免从零开始学习。

    Attributes
    ----------
    difficulty : float
        难度值 [0, 1]，会映射到对应难度桶。
    quality : float
        注入的质量先验值（0~1）。
    cost : float
        注入的成本先验值。
    source : str
        来源标签，如 "heuristic"（人工经验）、"historical"（历史数据）、"rule".
    """

    difficulty: float
    quality: float
    cost: float
    source: str = "heuristic"


@dataclass
class AgentComboProfile:
    """
    agent 组合的性能画像（alias of CandidateProfile，保持向后兼容）。

    Parameters
    ----------
    combo_id : str
        组合 ID（对应 candidate_name）。
    agents : List[AgentDef]
        组成该组合的 agent 列表。
    bucket_stats : Dict[str, BucketStats]
        各难度桶统计量。
    """

    combo_id: str
    agents: List[AgentDef] = field(default_factory=list)
    bucket_stats: Dict[str, BucketStats] = field(default_factory=dict)

    def get_bucket(self, bucket_name: str) -> BucketStats:
        if bucket_name not in self.bucket_stats:
            self.bucket_stats[bucket_name] = BucketStats(bucket_name=bucket_name)
        return self.bucket_stats[bucket_name]

    def total_support(self) -> int:
        return sum(s.support_count for s in self.bucket_stats.values())


# ---------------------------------------------------------------------------
# Agent / Agent 组合画像
# ---------------------------------------------------------------------------

@dataclass
class AgentDef:
    """
    单个 agent 的定义（用于描述组合中的各个 agent）。

    Attributes
    ----------
    agent_id : str
        全局唯一的 agent ID。
    agent_type : str
        agent 类型，如 "neural_network", "symbolic_solver", "llm".
    config : dict
        agent 的配置参数（可选）。
    """

    agent_id: str
    agent_type: str
    config: dict = field(default_factory=dict)


@dataclass
class CandidateProfile:
    """
    单个候选执行单元（单个 agent 或 agent 组合）的性能画像。

    在一个 PrimitiveProfile 内部维护，是学习的核心单元。
    画像曲线 = {难度桶 → BucketStats}。

    Attributes
    ----------
    candidate_name : str
        候选单元名称（如 "fast_nn", "strong_nn + physical_checker"）。
    agents : List[AgentDef]
        组成该候选单元的 agent 定义列表。
    bucket_stats : Dict[str, BucketStats]
        每个难度桶的统计量字典。
    last_calibrated_episode : int
        上次批量校准时的 episode 编号。
    metadata : dict
        扩展元信息（如额定成本、描述等）。
    """

    candidate_name: str
    agents: List[AgentDef] = field(default_factory=list)
    bucket_stats: Dict[str, BucketStats] = field(default_factory=dict)
    last_calibrated_episode: int = 0
    metadata: dict = field(default_factory=dict)

    def get_bucket(self, bucket_name: str) -> BucketStats:
        """获取指定桶的统计量，若不存在则创建"""
        if bucket_name not in self.bucket_stats:
            self.bucket_stats[bucket_name] = BucketStats(bucket_name=bucket_name)
        return self.bucket_stats[bucket_name]

    def get_all_buckets(self) -> Dict[str, BucketStats]:
        """返回所有已有数据的桶（按名称排序）"""
        return dict(sorted(self.bucket_stats.items()))

    def total_support_count(self) -> int:
        """返回跨所有桶的总样本数"""
        return sum(s.support_count for s in self.bucket_stats.values())


# ---------------------------------------------------------------------------
# 原语模块画像
# ---------------------------------------------------------------------------

@dataclass
class PrimitiveProfile:
    """
    一个原语模块 Q_k 下所有候选执行单元的画像集合。

    Attributes
    ----------
    primitive_name : str
        原语模块名（全局唯一），如 "Q_forecast".
    primitive_type : str
        原语类型，如 "time_series_forecast", "data_analysis".
    candidates : Dict[str, CandidateProfile]
        该原语下所有候选单元的画像（key = candidate_name）。
    metadata : dict
        扩展元信息。
    """

    primitive_name: str
    primitive_type: str
    candidates: Dict[str, CandidateProfile] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def get_or_create_candidate(self, candidate_name: str) -> CandidateProfile:
        """获取或创建一个候选画像"""
        if candidate_name not in self.candidates:
            self.candidates[candidate_name] = CandidateProfile(
                candidate_name=candidate_name
            )
        return self.candidates[candidate_name]

    def list_candidates(self) -> List[str]:
        """返回所有候选名称"""
        return list(self.candidates.keys())
