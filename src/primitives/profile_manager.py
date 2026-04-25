"""
profile_manager.py
==================
PrimitivePerformanceProfileManager - Core ACC-Cost Curve Management Module.

This module is the "knowledge base" of the multi-stage task orchestration system.
It learns and maintains the mapping:

    f(primitive, candidate, difficulty) -> (pred_acc, pred_cost, uncertainty)

Uses Pareto frontier optimization (via paretoset library) to identify
non-dominated candidates in the (quality, cost) space.

Core Responsibilities
--------------------
1. Initialization:
   - Register primitives (register_primitive)
   - Register candidates: single agent or agent combination (register_candidate)
   - Inject initial curves via init_curve dict (readme-compliant format)

2. Query (for topology orchestration):
   - predict(primitive, candidate, difficulty)  -> dict (pred_acc, pred_cost, ...)
   - predict_all(primitive, difficulty)           -> List[dict]
   - pareto_frontier(primitive, difficulty)     -> List[dict] (non-dominated set)
   - select_from_frontier(frontier, ...)       -> dict (constraint-based pick)

3. Feedback buffering:
   - add_feedback(record): buffer evaluator feedback, do NOT update immediately

4. Batch recalibration:
   - batch_recalibrate(): append observations to BucketStats, recompute means
   - save_feedback_jsonl / load_feedback_jsonl

5. Analysis export:
   - export_curve_table() -> list[dict] (for pandas DataFrame conversion)
"""

from __future__ import annotations

import json
import math
import numpy as np
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Any

from .feedback_record import FeedbackRecord
from .primitive_profile import (
    AgentDef,
    AgentComboProfile,
    CandidateProfile,
    DifficultyBucket,
    DifficultyMapper,
    PrimitiveProfile,
    BucketStats,
    InitPoint,
    DEFAULT_BUCKET_NAMES,
)


# ---------------------------------------------------------------------------
# Violation Penalty Helper
# ---------------------------------------------------------------------------

# Severity weights per violation type (hard constraints get stronger penalties)
_VIOLATION_SEVERITY = {
    "time_window": 1.0,
    "human_in_the_loop": 1.5,   # rejected by human — high severity
    "mandatory_node": 2.0,       # required primitive missing — critical
    "risk_boundary": 1.0,        # quality/cost outside safe range
}


def _apply_violation_penalty(
    observed_quality: float,
    violations: list,
    penalty_factor: float = 0.4,
) -> float:
    """
    Apply a severity-weighted multiplicative penalty to observed_quality.

    Each violation multiplies quality by (1 - penalty_factor * severity_weight).
    More violations or more severe violations compound the penalty.

    Formula:  penalized_q = q * prod((1 - factor * severity_i) for i in violations)

    Examples (factor=0.4):
        q=0.8, n=1, severity=1.0: 0.8 * 0.6 = 0.48
        q=0.8, n=2, severity=1.0: 0.8 * 0.36 = 0.288
        q=0.5, n=1, severity=1.5 (HITL): 0.5 * 0.4 = 0.20
    """
    n = len(violations)
    if n == 0:
        return observed_quality

    multiplier = 1.0
    for v in violations:
        vtype = v.get("violation_type", "")
        severity = 1.0
        for key in _VIOLATION_SEVERITY:
            if key in vtype.lower():
                severity = _VIOLATION_SEVERITY[key]
                break
        multiplier *= max(0.0, 1.0 - penalty_factor * severity)

    return max(0.0, observed_quality * multiplier)


# ---------------------------------------------------------------------------
# Data class for internal prediction result
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """
    Internal prediction result object (used by predict_all / pareto_frontier).

    All fields are also exposed as dict keys by predict().
    """

    candidate_name: str
    predicted_quality: float
    predicted_cost: float
    uncertainty: float
    support_count: int
    source: str
    difficulty_bucket: str
    meets_acc_target: bool = False
    meets_cost_budget: bool = False


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PrimitivePerformanceProfileManager:
    """
    Core performance profile management module.

    Manages ACC-Cost curves for all (primitive, candidate, difficulty_bucket)
    combinations, supporting cold-start initialization, feedback buffering,
    batch recalibration via observation accumulation, and Pareto frontier
    optimization for candidate selection.

    Parameters
    ----------
    difficulty_mapper : DifficultyMapper, optional
        Maps continuous difficulty [0,1] to bucket names.
        Default: ["easy", "medium", "hard", "extreme"].
    calibration_interval : int, default 10
        Auto-trigger batch_recalibrate() after this many add_feedback() calls.
        Set to None to disable auto-calibration.
    fallback_quality : float, default 0.5
        Quality value returned when no data exists at all.
    fallback_cost : float, default 1.0
        Cost value returned when no data exists at all.
    """

    def __init__(
        self,
        difficulty_mapper: DifficultyMapper | None = None,
        calibration_interval: int = 10,
        fallback_quality: float = 0.5,
        fallback_cost: float = 1.0,
        # Legacy parameter kept for API compatibility (ignored)
        default_ema_alpha: float = 0.3,
    ):
        # Registry: primitive_name -> PrimitiveProfile
        self._primitives: Dict[str, PrimitiveProfile] = {}

        # In-memory feedback buffer
        self._feedback_buffer: List[FeedbackRecord] = []

        # Configuration
        self._mapper = difficulty_mapper or DifficultyMapper()
        self._calibration_interval = calibration_interval
        self._fallback_quality = fallback_quality
        self._fallback_cost = fallback_cost
        self._fallback_latency: float = 10.0  # seconds

        # Global fallback statistics (updated during recalibration)
        self._global_quality_mean: float = fallback_quality
        self._global_cost_mean: float = fallback_cost
        self._global_support: int = 0

        # Episode counter (incremented each add_feedback)
        self._episode_counter: int = 0
        # Calibration event counter (incremented each batch_recalibrate)
        self._calibration_event_counter: int = 0

        # Post-calibration hooks
        self._post_calibration_hooks: List[Callable] = []

    # -------------------------------------------------------------------------
    # Public API: Registration
    # -------------------------------------------------------------------------

    def register_primitive(
        self,
        primitive_name: str,
        primitive_type: str = "",
        metadata: dict | None = None,
    ) -> PrimitiveProfile:
        """
        Register a new primitive module.

        Parameters
        ----------
        primitive_name : str
            Unique identifier, e.g. "state_parse", "forecast".
        primitive_type : str, optional
            Semantic type, e.g. "state_parsing", "time_series_forecast".
        metadata : dict, optional

        Returns
        -------
        PrimitiveProfile

        Raises
        ------
        ValueError
            If primitive_name is already registered.
        """
        if primitive_name in self._primitives:
            raise ValueError(
                f"Primitive '{primitive_name}' is already registered. "
                f"Use get_primitive() to retrieve it."
            )
        profile = PrimitiveProfile(
            primitive_name=primitive_name,
            primitive_type=primitive_type,
            metadata=metadata or {},
        )
        self._primitives[primitive_name] = profile
        return profile

    def register_candidate(
        self,
        primitive_name: str,
        candidate_name: str,
        agent_defs: List[AgentDef | dict] | None = None,
        init_profile: List[InitPoint | dict] | None = None,
        init_curve: dict | None = None,
        metadata: dict | None = None,
    ) -> CandidateProfile:
        """
        Register a candidate (single agent or agent combination) under a primitive.

        Supports TWO initialization formats:

        1. init_curve (readme-compliant dict format):
           {
               "easy":    {"acc_mean": 0.90, "cost_mean": 1.0},
               "medium":  {"acc_mean": 0.82, "cost_mean": 1.2},
               "hard":    {"acc_mean": 0.65, "cost_mean": 2.0},
               "extreme": {"acc_mean": 0.45, "cost_mean": 3.5},
           }
           Accepts "acc_mean" or "quality" as the ACC key.

        2. init_profile (list of InitPoint, legacy format):
           List[{"difficulty": 0.3, "quality": 0.72, "cost": 0.3}, ...]

        Parameters
        ----------
        primitive_name : str
        candidate_name : str
        agent_defs : List[AgentDef | dict], optional
        init_profile : List[InitPoint | dict], optional
        init_curve : dict, optional   <-- readme-compliant format
        metadata : dict, optional

        Returns
        -------
        CandidateProfile

        Raises
        ------
        KeyError
            primitive_name is not registered.
        ValueError
            init_curve contains an invalid bucket name, or missing required fields.
        """
        # Error handling: primitive not found
        if primitive_name not in self._primitives:
            raise KeyError(
                f"[register_candidate] Primitive '{primitive_name}' is not registered. "
                f"Call register_primitive() first. Registered: {self.list_primitives()}"
            )

        primitive = self._primitives[primitive_name]
        candidate = primitive.get_or_create_candidate(candidate_name)

        # Fill agent definitions
        if agent_defs:
            for ad in agent_defs:
                if isinstance(ad, dict):
                    ad = AgentDef(**ad)
                candidate.agents.append(ad)

        # Fill metadata
        if metadata:
            candidate.metadata.update(metadata)

        # Injection method 1: InitPoint list (legacy format)
        if init_profile:
            for pt in init_profile:
                if isinstance(pt, dict):
                    pt = InitPoint(**pt)
                self._inject_init_point(candidate, pt)

        # Injection method 2: init_curve dict (readme format)
        if init_curve:
            for bucket_name, stats in init_curve.items():
                # Error handling: invalid difficulty bucket
                if not self._mapper.is_valid_bucket_name(bucket_name):
                    raise ValueError(
                        f"[register_candidate] Invalid bucket name '{bucket_name}' "
                        f"in init_curve. Valid names: {self._mapper.bucket_names_list()}"
                    )
                if not isinstance(stats, dict):
                    raise ValueError(
                        f"[register_candidate] init_curve['{bucket_name}'] must be a dict "
                        f"with 'acc_mean' (or 'quality') and 'cost_mean'."
                    )
                # Accept both "acc_mean" and "quality" as ACC key
                acc = (
                    stats.get("acc_mean")
                    if stats.get("acc_mean") is not None
                    else stats.get("quality")
                )
                cost = stats.get("cost_mean")
                if acc is None:
                    raise ValueError(
                        f"[register_candidate] init_curve['{bucket_name}'] is missing "
                        f"both 'acc_mean' and 'quality' fields."
                    )
                if cost is None:
                    raise ValueError(
                        f"[register_candidate] init_curve['{bucket_name}'] is missing "
                        f"'cost_mean' field."
                    )
                # Write into the bucket as prior (not real feedback)
                bucket_stats = candidate.get_bucket(bucket_name)
                bucket_stats.set_prior(float(acc), float(cost))

        return candidate

    def get_primitive(self, primitive_name: str) -> PrimitiveProfile:
        """Get a registered PrimitiveProfile by name."""
        return self._primitives[primitive_name]

    def list_primitives(self) -> List[str]:
        """Return names of all registered primitives."""
        return list(self._primitives.keys())

    # -------------------------------------------------------------------------
    # Public API: Query (for topology orchestration)
    # -------------------------------------------------------------------------

    def predict(
        self,
        primitive_name: str,
        candidate_name: str,
        difficulty: float | str,
        task_features: np.ndarray | None = None,
    ) -> dict:
        """
        Query predicted ACC and cost for a specific candidate at given difficulty.

        Parameters
        ----------
        primitive_name : str
        candidate_name : str
        difficulty : float | str
            - float: normalized value in [0, 1], mapped to bucket automatically
            - str  : bucket name directly, e.g. "easy", "medium", "hard", "extreme"
        task_features : np.ndarray, optional
            Task feature vector phi (not used in v1; interface reserved).

        Returns
        -------
        dict with keys:
            pred_acc           : predicted ACC (0~1)
            pred_cost          : predicted execution cost
            support_count      : number of supporting feedback samples
            uncertainty        : uncertainty estimate (0~1, higher = more uncertain)
            source             : "bucket" | "init_profile" | "fallback"
            difficulty_bucket  : the bucket name used for this query

        Raises
        ------
        KeyError
            primitive_name or candidate_name not registered.
        ValueError
            difficulty is an invalid bucket name or out of [0,1] range.

        Query Priority:
        1. Bucket has real feedback samples -> return EMA mean (source="bucket")
        2. Bucket has init_curve/init_profile prior -> return prior mean (source="init_profile")
        3. No data at all -> return global fallback (source="fallback")
        """
        # Error handling: difficulty validation
        try:
            bucket = self._mapper.normalize_difficulty(difficulty)
        except ValueError as e:
            raise ValueError(f"[predict] {e}") from e

        # Error handling: primitive not found
        if primitive_name not in self._primitives:
            raise KeyError(
                f"[predict] Primitive '{primitive_name}' is not registered. "
                f"Registered: {self.list_primitives()}"
            )
        # Error handling: candidate not found
        if candidate_name not in self._primitives[primitive_name].candidates:
            raise KeyError(
                f"[predict] Candidate '{candidate_name}' is not registered "
                f"under Primitive '{primitive_name}'."
            )

        candidate = self._primitives[primitive_name].candidates[candidate_name]
        bucket_stats = candidate.get_bucket(bucket)

        # Priority 1: bucket has real feedback samples
        if bucket_stats.support_count > 0:
            return {
                "pred_acc": round(bucket_stats.quality_mean, 4),
                "pred_cost": round(bucket_stats.cost_mean, 4),
                "pred_latency": round(bucket_stats.latency_mean, 3),
                "support_count": bucket_stats.support_count,
                "uncertainty": round(bucket_stats.uncertainty, 4),
                "source": "bucket",
                "difficulty_bucket": bucket,
            }

        # Priority 2: init_curve / init_profile prior
        init_q, init_c, init_l, init_n = self._get_init_stats(candidate, bucket)
        if init_n > 0:
            return {
                "pred_acc": round(init_q, 4),
                "pred_cost": round(init_c, 4),
                "pred_latency": round(init_l, 3),
                "support_count": init_n,
                "uncertainty": round(1.0 / (1 + init_n), 4),
                "source": "init_profile",
                "difficulty_bucket": bucket,
            }

        # Priority 3: global fallback
        return {
            "pred_acc": round(self._fallback_quality, 4),
            "pred_cost": round(self._fallback_cost, 4),
            "pred_latency": round(self._fallback_latency, 3),
            "support_count": 0,
            "uncertainty": 1.0,
            "source": "fallback",
            "difficulty_bucket": bucket,
        }

    def predict_all(
        self,
        primitive_name: str,
        difficulty: float | str,
        task_features: np.ndarray | None = None,
        acc_target: float | None = None,
        cost_budget: float | None = None,
        latency_budget: float | None = None,
        top_k: int | None = None,
    ) -> List[dict]:
        """
        Return predictions for ALL candidates under a primitive (unsorted).

        Sorting is intentionally omitted — use pareto_frontier() +
        select_from_frontier() for decision-making.

        Parameters
        ----------
        primitive_name : str
        difficulty : float | str
        task_features : np.ndarray, optional
        acc_target : float, optional
        cost_budget : float, optional
        latency_budget : float, optional
        top_k : int, optional

        Returns
        -------
        List[dict]
            Each dict has keys: candidate_name, pred_acc, pred_cost,
            pred_latency, support_count, uncertainty, source,
            meets_acc_target, meets_cost_budget, meets_latency_budget,
            difficulty_bucket.
        """
        # Validate difficulty
        try:
            bucket = self._mapper.normalize_difficulty(difficulty)
        except ValueError as e:
            raise ValueError(f"[predict_all] {e}") from e

        if primitive_name not in self._primitives:
            return []

        primitive = self._primitives[primitive_name]
        results: List[dict] = []

        for candidate_name in primitive.list_candidates():
            pred = self.predict(primitive_name, candidate_name, bucket)
            meets_acc = (acc_target is None) or (pred["pred_acc"] >= acc_target)
            meets_cost = (cost_budget is None) or (pred["pred_cost"] <= cost_budget)
            meets_latency = (latency_budget is None) or (pred["pred_latency"] <= latency_budget)

            results.append({
                "candidate_name": candidate_name,
                "pred_acc": pred["pred_acc"],
                "pred_cost": pred["pred_cost"],
                "pred_latency": pred["pred_latency"],
                "support_count": pred["support_count"],
                "uncertainty": pred["uncertainty"],
                "source": pred["source"],
                "meets_acc_target": meets_acc,
                "meets_cost_budget": meets_cost,
                "meets_latency_budget": meets_latency,
                "difficulty_bucket": bucket,
            })

        return results[:top_k] if top_k else results

    def pareto_frontier(
        self,
        primitive_name: str,
        difficulty: float | str,
    ) -> List[dict]:
        """
        Return Pareto frontier candidates using the paretoset library.

        3D Pareto: maximize pred_acc, minimize pred_cost, minimize pred_latency.
        Matches paper §3.2 which defines Q(G,X) over all three objectives.

        Returns
        -------
        List[dict]
            Non-dominated candidates (same dict format as predict_all).
        """
        all_candidates = self.predict_all(primitive_name, difficulty)
        if len(all_candidates) <= 1:
            return all_candidates

        from paretoset import paretoset as compute_pareto

        qualities = np.array([c["pred_acc"] for c in all_candidates])
        costs = np.array([c["pred_cost"] for c in all_candidates])
        latencies = np.array([c.get("pred_latency", 0.0) for c in all_candidates])
        data = np.column_stack([qualities, costs, latencies])
        mask = compute_pareto(data, sense=["max", "min", "min"])
        return [c for c, m in zip(all_candidates, mask) if m]

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
        Select a single candidate from the Pareto frontier.

        Uses Q(c) = α*(S/s_scale) - β*C_norm - γ*L_norm (3D, scale-balanced).
        Weights match paper §3.2: α=0.65, β=0.25, γ=0.10, s_scale=1.5.

        S is divided by s_scale so it competes on the same scale as C_norm/L_norm
        which are already in [0,1] after log-normalization. Without this, the
        formula degenerates to argmax S regardless of cost/latency weights.

        acc_target / cost_budget / latency_budget act as STRICT hard feasibility
        filters before Q-maximization. If the filter leaves zero candidates,
        a ValueError is raised immediately (NO fallback).

        Parameters
        ----------
        frontier : List[dict]
            Non-dominated candidate list from pareto_frontier().
        acc_target : float, optional
            Hard filter: candidate must have pred_acc >= acc_target.
        cost_budget : float, optional
            Hard filter: candidate must have pred_cost <= cost_budget.
        latency_budget : float, optional
            Hard filter: candidate must have pred_latency <= latency_budget.
        alpha : float
            Weight for quality S (default 0.65).
        beta : float
            Weight for cost C (default 0.25).
        gamma : float
            Weight for latency L (default 0.10).
        s_scale : float
            Normalization divisor for S so it is comparable to C_norm/L_norm (default 1.5).

        Returns
        -------
        dict
            Selected candidate prediction dict.

        Raises
        ------
        ValueError
            If frontier is empty or if a hard constraint leaves zero candidates.
        """
        if not frontier:
            raise ValueError("Empty Pareto frontier — no candidates available.")

        filtered = frontier

        if cost_budget is not None:
            filtered = [c for c in filtered if c["pred_cost"] <= cost_budget]

        if acc_target is not None:
            filtered = [c for c in filtered if c["pred_acc"] >= acc_target]

        if latency_budget is not None:
            filtered = [c for c in filtered if c.get("pred_latency", 0.0) <= latency_budget]

        if not filtered:
            raise ValueError(
                f"No candidates satisfy hard constraints: "
                f"acc_target={acc_target}, cost_budget={cost_budget}, "
                f"latency_budget={latency_budget}. "
                f"Pareto frontier has {len(frontier)} candidates."
            )

        total = alpha + beta + gamma
        a, b, g = alpha / total, beta / total, gamma / total

        def _q(c: dict) -> float:
            S_norm = c.get("pred_acc", 0.0) / s_scale
            C_norm = c.get("pred_cost_norm", c.get("pred_cost", 0.0))
            L_norm = c.get("pred_latency_norm", c.get("pred_latency", 0.0))
            return a * S_norm - b * C_norm - g * L_norm

        return max(filtered, key=_q)

    # -------------------------------------------------------------------------
    # Public API: Feedback & Recalibration
    # -------------------------------------------------------------------------

    def add_feedback(self, record: FeedbackRecord | dict) -> None:
        """
        Buffer an evaluator feedback record (DO NOT update curves immediately).

        When the internal buffer reaches calibration_interval records,
        batch_recalibrate() is automatically triggered (if calibration_interval is set).

        Parameters
        ----------
        record : FeedbackRecord | dict
            Feedback record. dict will be converted via FeedbackRecord.from_dict().

        Raises
        ------
        KeyError
            primitive_name or candidate_name in record is not registered.
        ValueError
            difficulty_bucket in record is not a valid bucket name.
        """
        if isinstance(record, dict):
            record = FeedbackRecord.from_dict(record)

        # Error handling: validate difficulty bucket
        if not self._mapper.is_valid_bucket_name(record.difficulty_bucket):
            raise ValueError(
                f"[add_feedback] Invalid difficulty_bucket '{record.difficulty_bucket}'. "
                f"Valid: {self._mapper.bucket_names_list()}"
            )

        record.episode = self._episode_counter
        record.calibration_event_counter = self._calibration_event_counter
        self._feedback_buffer.append(record)
        self._episode_counter += 1

        # Auto-calibration trigger
        if (self._calibration_interval is not None
                and len(self._feedback_buffer) >= self._calibration_interval):
            self.batch_recalibrate()
            # After batch_recalibrate, the counter has been incremented
            # Subsequent records will see the new counter value

    def batch_recalibrate(
        self,
        alpha: float | None = None,
    ) -> dict:
        """
        Batch recalibration: group feedback by (primitive, candidate, bucket),
        then append observations to each BucketStats.

        Observations are stored raw; means are recomputed from the full
        observation list (no EMA).

        Parameters
        ----------
        alpha : float, optional
            Legacy parameter (ignored). Kept for API compatibility.

        Returns
        -------
        dict
            Calibration summary.
        """
        if not self._feedback_buffer:
            return {"status": "no_feedback", "updated_groups": 0}

        # Group by (primitive_name, candidate_name, difficulty_bucket)
        groups: Dict[Tuple[str, str, str], List[FeedbackRecord]] = defaultdict(list)
        for rec in self._feedback_buffer:
            groups[(rec.primitive_name, rec.candidate_name, rec.difficulty_bucket)].append(rec)

        updated_count = 0
        details: Dict[str, Any] = {}

        for (prim_name, cand_name, bucket), records in groups.items():
            try:
                candidate = self._primitives[prim_name].candidates[cand_name]
            except KeyError:
                # Skip feedback for unregistered candidates (defensive)
                continue

            bucket_stats = candidate.get_bucket(bucket)

            for rec in records:
                # Apply violation penalty: violations push observed_quality toward 0,
                # teaching the profile that violating candidates perform worse
                penalized_quality = _apply_violation_penalty(
                    rec.observed_quality,
                    rec.constraint_violations or [],
                    penalty_factor=0.3,
                )
                # Append observation (replaces EMA update)
                bucket_stats.add_observation(
                    observed_quality=penalized_quality,
                    observed_cost=rec.observed_cost,
                )
                # Update global fallback statistics
                self._update_global_stats(penalized_quality, rec.observed_cost)

            candidate.last_calibrated_episode = self._episode_counter
            updated_count += 1
            details[f"{prim_name}/{cand_name}/{bucket}"] = {
                "n_records": len(records),
                "new_acc_mean": round(bucket_stats.quality_mean, 4),
                "new_cost_mean": round(bucket_stats.cost_mean, 4),
                "support_count": bucket_stats.support_count,
            }

        consumed = len(self._feedback_buffer)
        self._feedback_buffer.clear()

        result = {
            "status": "ok",
            "consumed": consumed,
            "updated_groups": updated_count,
            "details": details,
            "calibration_event_counter": self._calibration_event_counter,
        }

        # Increment calibration event counter AFTER producing result
        self._calibration_event_counter += 1

        # Post-calibration hooks
        for hook in self._post_calibration_hooks:
            hook(result)

        return result

    def register_post_calibration_hook(self, hook: Callable[[dict], None]) -> None:
        """Register a callback to be invoked after each batch_recalibrate()."""
        self._post_calibration_hooks.append(hook)

    # -------------------------------------------------------------------------
    # Public API: Serialization
    # -------------------------------------------------------------------------

    def save_feedback_jsonl(self, path: str | Path) -> int:
        """
        Append feedback buffer to a JSONL file.

        Parameters
        ----------
        path : str | Path

        Returns
        -------
        int
            Number of records written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with path.open("a", encoding="utf-8") as f:
            for record in self._feedback_buffer:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                count += 1
        return count

    def load_feedback_jsonl(
        self,
        path: str | Path,
        append: bool = True,
    ) -> int:
        """
        Load feedback records from a JSONL file into the buffer.

        Parameters
        ----------
        path : str | Path
        append : bool, default True
            True = append to existing buffer; False = replace buffer.

        Returns
        -------
        int
            Number of records loaded.
        """
        path = Path(path)
        if not path.exists():
            return 0

        records: List[FeedbackRecord] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(FeedbackRecord.from_dict(json.loads(line)))

        if append:
            self._feedback_buffer.extend(records)
        else:
            self._feedback_buffer = records

        return len(records)

    def save_profile_snapshot(self, path: str | Path) -> None:
        """
        Save a snapshot of all current profiles to JSON.
        Useful for checkpointing during long experiments.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "primitives": {},
        }

        for prim_name, prim in self._primitives.items():
            prim_data = {
                "primitive_type": prim.primitive_type,
                "metadata": prim.metadata,
                "candidates": {},
            }
            for cand_name, cand in prim.candidates.items():
                cand_data = {
                    "agents": [
                        {"agent_id": a.agent_id, "agent_type": a.agent_type}
                        for a in cand.agents
                    ],
                    "metadata": cand.metadata,
                    "last_calibrated_episode": cand.last_calibrated_episode,
                    "bucket_stats": {},
                }
                for bucket, stats in cand.get_all_buckets().items():
                    cand_data["bucket_stats"][bucket] = {
                        "acc_mean": round(stats.quality_mean, 4),
                        "cost_mean": round(stats.cost_mean, 4),
                        "acc_std": (
                            round(stats.quality_std, 4)
                            if not math.isnan(stats.quality_std) else None
                        ),
                        "cost_std": (
                            round(stats.cost_std, 4)
                            if not math.isnan(stats.cost_std) else None
                        ),
                        "support_count": stats.support_count,
                        "uncertainty": round(stats.uncertainty, 4),
                    }
                prim_data["candidates"][cand_name] = cand_data
            snapshot["primitives"][prim_name] = prim_data

        with path.open("w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)

    # -------------------------------------------------------------------------
    # Public API: Analysis Export (readme-required)
    # -------------------------------------------------------------------------

    def export_curve_table(self) -> List[dict]:
        """
        Export the full curve table as a list of dicts (readme-required).

        Each record corresponds to one (primitive, candidate, difficulty) triple.
        Designed for conversion to pandas DataFrame.

        Returns
        -------
        List[dict]
            Each dict contains:
                primitive_name, candidate_name, difficulty,
                acc_mean, cost_mean, support_count, uncertainty,
                last_calibrated_episode
            Sorted by (primitive_name, candidate_name, difficulty).
        """
        table: List[dict] = []

        for prim_name in sorted(self._primitives.keys()):
            prim = self._primitives[prim_name]
            for cand_name in sorted(prim.candidates.keys()):
                cand = prim.candidates[cand_name]
                for bucket_name in self._mapper.bucket_names_list():
                    stats = cand.get_bucket(bucket_name)
                    # Include all non-empty buckets
                    table.append({
                        "primitive_name": prim_name,
                        "candidate_name": cand_name,
                        "difficulty": bucket_name,
                        "acc_mean": round(stats.quality_mean, 4),
                        "cost_mean": round(stats.cost_mean, 4),
                        "support_count": stats.support_count,
                        "uncertainty": round(stats.uncertainty, 4),
                        "last_calibrated_episode": cand.last_calibrated_episode,
                    })

        return table

    # -------------------------------------------------------------------------
    # Public API: Inspection / Debug
    # -------------------------------------------------------------------------

    def inspect(
        self,
        primitive_name: str,
        candidate_name: str,
    ) -> dict:
        """
        Return the complete profile for a candidate (for debugging / analysis).
        """
        if primitive_name not in self._primitives:
            raise KeyError(f"[inspect] Primitive '{primitive_name}' not registered.")
        if candidate_name not in self._primitives[primitive_name].candidates:
            raise KeyError(
                f"[inspect] Candidate '{candidate_name}' not under "
                f"Primitive '{primitive_name}'."
            )

        cand = self._primitives[primitive_name].candidates[candidate_name]
        return {
            "candidate_name": cand.candidate_name,
            "agents": [
                {"agent_id": a.agent_id, "agent_type": a.agent_type}
                for a in cand.agents
            ],
            "metadata": cand.metadata,
            "buckets": {
                name: {
                    "acc_mean": round(s.quality_mean, 4),
                    "cost_mean": round(s.cost_mean, 4),
                    "acc_std": (
                        round(s.quality_std, 4)
                        if not math.isnan(s.quality_std) else None
                    ),
                    "cost_std": (
                        round(s.cost_std, 4)
                        if not math.isnan(s.cost_std) else None
                    ),
                    "support_count": s.support_count,
                    "uncertainty": round(s.uncertainty, 4),
                }
                for name, s in cand.get_all_buckets().items()
            },
            "total_support": cand.total_support_count(),
            "last_calibrated_episode": cand.last_calibrated_episode,
        }

    @property
    def feedback_buffer_size(self) -> int:
        """Current number of records in the feedback buffer."""
        return len(self._feedback_buffer)

    @property
    def episode_counter(self) -> int:
        return self._episode_counter

    # -------------------------------------------------------------------------
    # Internal helper methods
    # -------------------------------------------------------------------------

    def _inject_init_point(
        self,
        candidate: CandidateProfile,
        pt: InitPoint,
    ) -> None:
        """Inject an InitPoint (float difficulty) into the corresponding bucket.

        InitPoints are prior data: they set values directly and are tracked
        in n_prior (not support_count) so they do not masquerade as real feedback.
        """
        bucket = self._mapper.map(pt.difficulty)
        stats = candidate.get_bucket(bucket)
        stats.set_prior(float(pt.quality), float(pt.cost))

    def _get_init_stats(
        self,
        candidate: CandidateProfile,
        bucket: str,
    ) -> Tuple[float, float, float, int]:
        """
        Get the aggregated prior stats for a bucket (from init_curve / init_profile).
        Returns (quality_mean, cost_mean, latency_mean, n_prior) where n_prior > 0
        indicates prior samples exist from initialization.
        """
        stats = candidate.bucket_stats.get(bucket)
        if stats and stats.quality_mean != 0.0 and stats.n_prior > 0:
            return stats.quality_mean, stats.cost_mean, stats.latency_mean, stats.n_prior
        return 0.0, 0.0, 0.0, 0

    def _update_global_stats(
        self,
        quality: float,
        cost: float,
    ) -> None:
        """Update global fallback statistics using running mean."""
        self._global_support += 1
        n = self._global_support
        self._global_quality_mean += (quality - self._global_quality_mean) / n
        self._global_cost_mean += (cost - self._global_cost_mean) / n

    def __repr__(self) -> str:
        prim_summary = ", ".join(
            f"{k}({len(v.candidates)}c)" for k, v in self._primitives.items()
        )
        return (
            f"PrimitivePerformanceProfileManager("
            f"primitives=[{prim_summary}], "
            f"feedback_buffer={len(self._feedback_buffer)}, "
            f"episodes={self._episode_counter})"
        )


# ---------------------------------------------------------------------------
# Log-scale normalization utility (used by experiment scripts and src/ alike)
# ---------------------------------------------------------------------------

def log_normalize_profiles(
    profiles: List[dict],
    cost_key: str = "C",
    latency_key: str = "L",
    cost_norm_key: str = "C_norm",
    latency_norm_key: str = "L_norm",
) -> List[dict]:
    """
    Apply log-scale normalization to cost and latency fields across a profile list.

    Matches the normalization used in experiment_water_qa_topo.py so that
    filter_by_constraints and q_score operate on comparable [0,1] scales.

    Formula: x_norm = log(1 + x) / log(1 + x_max)
    If x_max == 0, all normalized values are 0.

    Parameters
    ----------
    profiles : List[dict]
        Profile dicts, each containing cost_key and latency_key fields.
    cost_key, latency_key : str
        Source field names for raw cost and latency.
    cost_norm_key, latency_norm_key : str
        Output field names for normalized values (written in-place).

    Returns
    -------
    List[dict]
        Same list with cost_norm_key and latency_norm_key added/updated in-place.
    """
    if not profiles:
        return profiles

    costs = [p.get(cost_key, 0.0) for p in profiles]
    latencies = [p.get(latency_key, 0.0) for p in profiles]

    c_max = max(costs) if costs else 0.0
    l_max = max(latencies) if latencies else 0.0

    log_c_max = math.log1p(c_max) if c_max > 0 else 1.0
    log_l_max = math.log1p(l_max) if l_max > 0 else 1.0

    for p in profiles:
        p[cost_norm_key] = math.log1p(p.get(cost_key, 0.0)) / log_c_max
        p[latency_norm_key] = math.log1p(p.get(latency_key, 0.0)) / log_l_max

    return profiles

def _demo():
    """
    Minimal runnable demo (matching readme exactly).

    Steps:
    1. Register two primitives: state_parse, forecast
    2. Register 2 candidates per primitive with init_curve
    3. Initialize ACC-Cost curves
    4. Call predict() (bucket name) and predict_all()
    5. Add feedback records
    6. batch_recalibrate(alpha=0.3)
    7. predict() again, print before/after changes
    8. export_curve_table() and print results
    """
    print("=" * 60)
    print("PrimitivePerformanceProfileManager - Readme Demo")
    print("=" * 60)

    # Step 1: Create manager
    manager = PrimitivePerformanceProfileManager(
        calibration_interval=10,
        default_ema_alpha=0.3,
        fallback_quality=0.5,
        fallback_cost=1.0,
    )

    # Step 2: Register primitives
    manager.register_primitive("state_parse", primitive_type="state_parsing")
    manager.register_primitive("forecast", primitive_type="time_series_forecast")
    print("\n[Register] Primitives:", manager.list_primitives())

    # Step 3: Register candidates with init_curve (readme format)
    # -- state_parse --
    manager.register_candidate(
        "state_parse", "rule_parser",
        init_curve={
            "easy":    {"acc_mean": 0.85, "cost_mean": 0.2},
            "medium":  {"acc_mean": 0.70, "cost_mean": 0.2},
            "hard":    {"acc_mean": 0.50, "cost_mean": 0.2},
            "extreme": {"acc_mean": 0.30, "cost_mean": 0.2},
        },
    )
    manager.register_candidate(
        "state_parse", "llm_small",
        init_curve={
            "easy":    {"acc_mean": 0.88, "cost_mean": 1.0},
            "medium":  {"acc_mean": 0.82, "cost_mean": 1.0},
            "hard":    {"acc_mean": 0.72, "cost_mean": 1.0},
            "extreme": {"acc_mean": 0.55, "cost_mean": 1.0},
        },
    )

    # -- forecast --
    manager.register_candidate(
        "forecast", "fast_nn",
        init_curve={
            "easy":    {"acc_mean": 0.72, "cost_mean": 0.3},
            "medium":  {"acc_mean": 0.68, "cost_mean": 0.3},
            "hard":    {"acc_mean": 0.60, "cost_mean": 0.3},
            "extreme": {"acc_mean": 0.50, "cost_mean": 0.3},
        },
    )
    manager.register_candidate(
        "forecast", "fvcom",
        init_curve={
            "easy":    {"acc_mean": 0.90, "cost_mean": 2.0},
            "medium":  {"acc_mean": 0.88, "cost_mean": 2.0},
            "hard":    {"acc_mean": 0.85, "cost_mean": 2.0},
            "extreme": {"acc_mean": 0.80, "cost_mean": 2.0},
        },
    )

    print("[Register] Candidates initialized with init_curve.")

    # Step 4: Initial prediction (cold-start)
    print("\n" + "-" * 60)
    print("[Phase A] Cold-Start predict() / predict_all()")
    print("-" * 60)

    # predict with bucket name str
    pred = manager.predict("forecast", "fast_nn", "medium")
    print(f"\npredict('forecast', 'fast_nn', 'medium'):")
    print(f"  pred_acc={pred['pred_acc']}  pred_cost={pred['pred_cost']}  "
          f"uncertainty={pred['uncertainty']}  source={pred['source']}")

    # predict with float (0.6 -> hard)
    pred2 = manager.predict("forecast", "fvcom", 0.6)
    print(f"\npredict('forecast', 'fvcom', 0.6)  [hard bucket]:")
    print(f"  pred_acc={pred2['pred_acc']}  pred_cost={pred2['pred_cost']}  "
          f"uncertainty={pred2['uncertainty']}  bucket={pred2['difficulty_bucket']}")

    # predict_all
    print(f"\npredict_all('forecast', 'medium'):")
    ranks = manager.predict_all("forecast", "medium")
    for r in ranks:
        print(f"  {r['candidate_name']:12s}  acc={r['pred_acc']:.3f}  "
              f"cost={r['pred_cost']:.2f}  src={r['source']}")

    # Step 5: Add feedback records
    print("\n" + "-" * 60)
    print("[Phase B] Adding 6 Feedback Records")
    print("-" * 60)

    import random
    synthetic = [
        # (primitive, candidate, bucket, obs_acc, obs_cost, eval_pass, failure_type)
        ("forecast", "fast_nn", "medium", 0.66, 0.31, True,  None),
        ("forecast", "fast_nn", "medium", 0.63, 0.29, True,  None),
        ("forecast", "fast_nn", "medium", 0.64, 0.30, True,  None),
        ("forecast", "fvcom",   "hard",   0.83, 2.10, True,  None),
        ("forecast", "fvcom",   "hard",   0.79, 2.05, True,  None),
        ("forecast", "fvcom",   "hard",   0.50, 2.20, False, "low_quality"),
    ]

    for prim, cand, bucket, obs_a, obs_c, ev_pass, fail_type in synthetic:
        record = FeedbackRecord(
            task_id=f"task_{random.randint(1000, 9999)}",
            node_id=f"node_{random.randint(100, 999)}",
            primitive_name=prim,
            candidate_name=cand,
            difficulty=0.5,
            difficulty_bucket=bucket,
            predicted_quality=0.8,
            predicted_cost=1.0,
            observed_quality=obs_a,
            observed_cost=obs_c,
            eval_pass=ev_pass,
            failure_type=fail_type,
            episode=manager.episode_counter,
        )
        manager.add_feedback(record)
        print(f"  + {prim}/{cand} [{bucket}]  obs_acc={obs_a:.2f}  "
              f"obs_cost={obs_c:.2f}  pass={ev_pass}")

    print(f"\nBuffer size: {manager.feedback_buffer_size} "
          f"(auto-calibration triggers at 10, not triggered yet)")

    # Step 6: Capture before-state BEFORE recalibrating
    before_fastnn = manager.predict("forecast", "fast_nn", "medium")
    before_fvcom = manager.predict("forecast", "fvcom", "hard")

    # Step 7: batch_recalibrate
    print("\n" + "-" * 60)
    print("[Phase C] batch_recalibrate()")
    print("-" * 60)
    summary = manager.batch_recalibrate()
    print(f"  status={summary['status']}  consumed={summary['consumed']}  "
          f"updated_groups={summary['updated_groups']}")
    for key, val in summary["details"].items():
        print(f"    {key}: acc_mean={val['new_acc_mean']:.4f}  "
              f"cost_mean={val['new_cost_mean']:.4f}  n={val['support_count']}")

    # Step 8: before vs after
    print("\n" + "-" * 60)
    print("[Phase D] Before vs After Recalibration")
    print("-" * 60)

    print(f"\n[Before recalibration - from init_curve]:")
    print(f"  fast_nn [medium]: acc={before_fastnn['pred_acc']:.3f}  "
          f"cost={before_fastnn['pred_cost']:.2f}  src={before_fastnn['source']}")
    print(f"  fvcom   [hard]:  acc={before_fvcom['pred_acc']:.3f}  "
          f"cost={before_fvcom['pred_cost']:.2f}  src={before_fvcom['source']}")

    after_fastnn = manager.predict("forecast", "fast_nn", "medium")
    after_fvcom = manager.predict("forecast", "fvcom", "hard")

    print(f"\n[After recalibration - from real feedback]:")
    print(f"  fast_nn [medium]: acc={after_fastnn['pred_acc']:.3f}  "
          f"cost={after_fastnn['pred_cost']:.2f}  "
          f"uncertainty={after_fastnn['uncertainty']}  src={after_fastnn['source']}")
    print(f"  fvcom   [hard]:  acc={after_fvcom['pred_acc']:.3f}  "
          f"cost={after_fvcom['pred_cost']:.2f}  "
          f"uncertainty={after_fvcom['uncertainty']}  src={after_fvcom['source']}")

    delta_fnn = after_fastnn['pred_acc'] - before_fastnn['pred_acc']
    delta_fvcom = after_fvcom['pred_acc'] - before_fvcom['pred_acc']
    print(f"\n  fast_nn acc delta: {before_fastnn['pred_acc']:.3f} -> {after_fastnn['pred_acc']:.3f}  "
          f"(change={delta_fnn:+.4f})")
    print(f"  fvcom   acc delta: {before_fvcom['pred_acc']:.3f} -> {after_fvcom['pred_acc']:.3f}  "
          f"(change={delta_fvcom:+.4f})")

    # Step 8: export_curve_table
    print("\n" + "-" * 60)
    print("[Phase E] export_curve_table()")
    print("-" * 60)

    table = manager.export_curve_table()
    print(f"\n{'primitive':15s}  {'candidate':12s}  {'difficulty':10s}  "
          f"{'acc_mean':9s}  {'cost_mean':9s}  {'support':7s}  {'uncertainty':11s}")
    print("-" * 85)
    for row in table:
        print(
            f"{row['primitive_name']:15s}  "
            f"{row['candidate_name']:12s}  "
            f"{row['difficulty']:10s}  "
            f"{row['acc_mean']:9.4f}  "
            f"{row['cost_mean']:9.4f}  "
            f"{row['support_count']:7d}  "
            f"{row['uncertainty']:11.4f}"
        )

    # Error handling demo
    print("\n" + "-" * 60)
    print("[Phase F] Error Handling")
    print("-" * 60)

    # Invalid bucket name in predict
    try:
        manager.predict("forecast", "fast_nn", "super_hard")
    except ValueError as e:
        print(f"  [Expected] ValueError: {e}")

    # Invalid primitive
    try:
        manager.predict("non_existent", "fast_nn", "medium")
    except KeyError as e:
        print(f"  [Expected] KeyError: {e}")

    # Invalid init_curve bucket
    try:
        manager.register_candidate(
            "forecast", "test_cand",
            init_curve={"mega_hard": {"acc_mean": 0.9, "cost_mean": 1.0}}
        )
    except ValueError as e:
        print(f"  [Expected] ValueError: {e}")

    # Missing acc_mean in init_curve
    try:
        manager.register_candidate(
            "forecast", "test_cand2",
            init_curve={"medium": {"cost_mean": 1.0}}
        )
    except ValueError as e:
        print(f"  [Expected] ValueError: {e}")

    print("\n" + "=" * 60)
    print("Demo complete. All phases passed.")
    print("=" * 60)


if __name__ == "__main__":
    _demo()
