"""
analyze_results.py
==================
Result analysis tools for MVP experiments.

Provides:
- analyze_experiment_results(): analyze a single experiment run
- plot_convergence(): visualize prediction convergence over episodes
- compare_ablation(): compare results across calibration intervals
- compute_prediction_error(): MAE of pred_acc vs true_acc over time

All outputs are text-based (ASCII tables / matplotlib-free).
matplotlib support is optional.
"""

from __future__ import annotations

import sys
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Add src/ to path
_SRC_DIR = Path(__file__).parent.parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from src.experiments.mvp_experiment import EpisodeRecord, ExperimentConfig


# ---------------------------------------------------------------------------
# Analysis Result Containers
# ---------------------------------------------------------------------------

@dataclass
class CandidateMetrics:
    """Metrics for a specific (primitive, candidate) across episodes."""
    primitive_name: str
    candidate_name: str
    n_selections: int
    n_passes: int
    pass_rate: float
    mae_before_recal: float      # MAE of pred_acc vs true_acc before first recal
    mae_after_recal: float        # MAE after last recal
    pred_error_at_first_recal: float  # Error at moment of first recalibration
    final_pred_acc: float         # Last predicted acc


# ---------------------------------------------------------------------------
# Core Analysis
# ---------------------------------------------------------------------------

def analyze_experiment_results(
    episode_records: List[EpisodeRecord],
    curve_snapshots: List[dict],
    final_table: List[dict],
    config: ExperimentConfig,
) -> Dict[str, Any]:
    """
    Analyze a completed experiment run.

    Returns
    -------
    dict with:
        - candidate_metrics: List[CandidateMetrics]
        - per_episode_stats: dict[episode -> dict]
        - convergence_summary: dict
        - calibration_events: list of dict
        - print-friendly summary string
    """
    # --- Per-candidate metrics ---
    cand_records: Dict[Tuple[str, str], List[EpisodeRecord]] = {}
    for rec in episode_records:
        key = (rec.primitive_name, rec.selected_candidate)
        cand_records.setdefault(key, []).append(rec)

    candidate_metrics: List[CandidateMetrics] = []

    for (prim, cand), recs in sorted(cand_records.items()):
        n_pass = sum(1 for r in recs if r.eval_pass)
        pass_rate = n_pass / len(recs) if recs else 0.0

        # MAE before first recalibration
        # Use record.recalibrated flag (set at record-creation time)
        before_rec = [r for r in recs if not r.recalibrated]
        after_rec = [r for r in recs if r.recalibrated]

        mae_before = _mae(
            [(r.predicted_acc, r.true_acc) for r in before_rec]
        ) if before_rec else None

        mae_after = _mae(
            [(r.predicted_acc, r.true_acc) for r in after_rec]
        ) if after_rec else None

        pred_err_first = (
            abs(before_rec[-1].predicted_acc - before_rec[-1].true_acc)
            if before_rec else None
        )

        final_pred = recs[-1].predicted_acc if recs else None

        candidate_metrics.append(CandidateMetrics(
            primitive_name=prim,
            candidate_name=cand,
            n_selections=len(recs),
            n_passes=n_pass,
            pass_rate=round(pass_rate, 4),
            mae_before_recal=round(mae_before, 4) if mae_before is not None else None,
            mae_after_recal=round(mae_after, 4) if mae_after is not None else None,
            pred_error_at_first_recal=(
                round(pred_err_first, 4) if pred_err_first is not None else None
            ),
            final_pred_acc=round(final_pred, 4) if final_pred is not None else None,
        ))

    # --- Per-episode stats ---
    episodes = sorted(set(r.episode for r in episode_records))
    per_episode_stats: Dict[int, dict] = {}
    for ep in episodes:
        ep_recs = [r for r in episode_records if r.episode == ep]
        n_pass = sum(1 for r in ep_recs if r.eval_pass)
        per_episode_stats[ep] = {
            "n_executions": len(ep_recs),
            "n_pass": n_pass,
            "pass_rate": round(n_pass / len(ep_recs), 4),
            "mean_pred_acc": round(
                sum(r.predicted_acc for r in ep_recs) / len(ep_recs), 4
            ),
            "mean_true_acc": round(
                sum(r.true_acc for r in ep_recs) / len(ep_recs), 4
            ),
            "mean_observed_acc": round(
                sum(r.observed_acc for r in ep_recs) / len(ep_recs), 4
            ),
        }

    # --- Convergence summary ---
    convergence_summary = _compute_convergence_summary(
        episode_records, curve_snapshots, final_table
    )

    return {
        "candidate_metrics": candidate_metrics,
        "per_episode_stats": per_episode_stats,
        "convergence_summary": convergence_summary,
        "curve_snapshots": curve_snapshots,
        "episode_records": episode_records,
        "config": config,
    }


def _mae(pairs: List[Tuple[float, float]]) -> float:
    """Mean Absolute Error."""
    if not pairs:
        return float("nan")
    return sum(abs(p - t) for p, t in pairs) / len(pairs)


def _compute_convergence_summary(
    episode_records: List[EpisodeRecord],
    curve_snapshots: List[dict],
    final_table: List[dict],
) -> dict:
    """
    Summarize how predictions converge to true values over episodes.
    """
    # Group by (primitive, candidate, bucket)
    groups: Dict[Tuple, List[EpisodeRecord]] = {}
    for r in episode_records:
        key = (r.primitive_name, r.selected_candidate, r.difficulty_bucket)
        groups.setdefault(key, []).append(r)

    convergence: List[dict] = []
    for key, recs in groups.items():
        prim, cand, bucket = key
        maes = []
        for ep in sorted(set(r.episode for r in recs)):
            ep_recs = [r for r in recs if r.episode == ep]
            if ep_recs:
                mae = _mae([(r.predicted_acc, r.true_acc) for r in ep_recs])
                maes.append((ep, round(mae, 4)))

        convergence.append({
            "primitive": prim,
            "candidate": cand,
            "bucket": bucket,
            "n_selections": len(recs),
            "mae_per_episode": maes,
            "final_mae": maes[-1][1] if maes else None,
        })

    return {
        "per_candidate_convergence": convergence,
        "n_snapshots": len(curve_snapshots),
    }


# ---------------------------------------------------------------------------
# Text-based Visualization
# ---------------------------------------------------------------------------

def print_analysis(analysis: Dict[str, Any]) -> None:
    """
    Print a human-readable analysis report.
    """
    config = analysis["config"]
    summary = analysis["convergence_summary"]

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT ANALYSIS REPORT: {config.name}")
    print(f"{'='*70}")

    # --- Config summary ---
    print(f"\n  [Configuration]")
    print(f"    Episodes:        {config.n_episodes}")
    print(f"    K (calibration):  {config.calibration_interval}")
    print(f"    EMA alpha:        {config.default_ema_alpha}")
    print(f"    ACC target:       {config.acc_target}")
    print(f"    Cost budget:     {config.cost_budget}")
    print(f"    Noise std:       {config.noise_std}")

    # --- Candidate metrics ---
    print(f"\n  [Per-Candidate Metrics]")
    print(f"  {'Primitive':15s}  {'Candidate':12s}  {'N_sel':>5}  "
          f"{'Pass%':>6}  {'MAE_pre':>8}  {'MAE_post':>8}  {'Final_Q':>8}")
    print("  " + "-" * 75)
    for m in analysis["candidate_metrics"]:
        mae_pre = f"{m.mae_before_recal:.4f}" if m.mae_before_recal else "N/A"
        mae_post = f"{m.mae_after_recal:.4f}" if m.mae_after_recal else "N/A"
        final_q = f"{m.final_pred_acc:.4f}" if m.final_pred_acc else "N/A"
        print(
            f"  {m.primitive_name:15s}  {m.candidate_name:12s}  "
            f"{m.n_selections:>5}  "
            f"{m.pass_rate:>6.1%}  "
            f"{mae_pre:>8}  "
            f"{mae_post:>8}  "
            f"{final_q:>8}"
        )

    # --- Convergence per episode ---
    print(f"\n  [Per-Episode Statistics]")
    print(f"  {'Ep':>3}  {'N_exec':>6}  {'Pass%':>6}  {'Mean_Q_hat':>10}  "
          f"{'Mean_Q_true':>11}  {'Mean_Q_obs':>10}")
    print("  " + "-" * 60)
    for ep, stats in analysis["per_episode_stats"].items():
        print(
            f"  {ep:>3}  "
            f"{stats['n_executions']:>6}  "
            f"{stats['pass_rate']:>6.1%}  "
            f"{stats['mean_pred_acc']:>10.4f}  "
            f"{stats['mean_true_acc']:>11.4f}  "
            f"{stats['mean_observed_acc']:>10.4f}"
        )

    # --- Recalibration events ---
    snapshots = analysis.get("curve_snapshots", [])
    print(f"\n  [Recalibration Events: {len(snapshots)}]")
    if snapshots:
        print(f"  {'#':>3}  {'Episode':>7}  {'Group':>40}  {'New_ACC':>8}")
        print("  " + "-" * 65)
        for i, snap in enumerate(snapshots):
            ep = snap.get("episode_at_calibration", "?")
            for group_key, detail in snap.get("details", {}).items():
                print(
                    f"  {i+1:>3}  {str(ep):>7}  {group_key:>40}  "
                    f"{detail['new_acc_mean']:>8.4f}"
                )

    # --- Convergence summary ---
    conv = summary.get("per_candidate_convergence", [])
    print(f"\n  [Convergence: MAE over episodes]")
    if conv:
        print(f"  {'Primitive':15s}  {'Candidate':12s}  {'Bucket':8s}  "
              f"{'N_sel':>5}  {'Final_MAE':>9}")
        print("  " + "-" * 58)
        for c in conv:
            final_mae = f"{c['final_mae']:.4f}" if c['final_mae'] is not None else "N/A"
            print(
                f"  {c['primitive']:15s}  {c['candidate']:12s}  "
                f"{c['bucket']:8s}  {c['n_selections']:>5}  {final_mae:>9}"
            )

    # --- Template selection distribution ---
    episode_records = analysis.get("episode_records", [])
    if episode_records:
        from collections import defaultdict as _defaultdict
        template_counts: Dict[str, int] = _defaultdict(int)
        template_by_bucket: Dict[str, Dict[str, int]] = _defaultdict(lambda: _defaultdict(int))
        for rec in episode_records:
            template_counts[rec.template_id] += 1
            template_by_bucket[rec.difficulty_bucket][rec.template_id] += 1

        total_recs = len(episode_records)
        print(f"\n  [Template Selection Distribution]")
        print(f"  {'Template':25s}  {'Count':>6}  {'Share':>7}")
        print(f"  {'-'*42}")
        for tid in sorted(template_counts.keys()):
            cnt = template_counts[tid]
            share = cnt / total_recs if total_recs > 0 else 0
            print(f"  {tid:25s}  {cnt:>6}  {share:>7.1%}")

        n_upgrades = sum(1 for r in episode_records if r.template_upgraded_from != "none")
        if n_upgrades > 0:
            print(f"\n    Template upgrades during repair: {n_upgrades}")

    print(f"\n{'='*70}\n")

def plot_acc_cost_curves(
    episode_records,
    curve_snapshots: List[dict],
    output_path: Path | str | None = None,
    title_suffix: str = "",
) -> None:
    """
    Plot ACC-Cost evolution curves per (primitive, candidate).

    Generates:
    - One panel per primitive, showing all candidates as separate lines
    - X-axis: episode number
    - Y-axis (left): predicted ACC
    - Y-axis (right): predicted cost
    - Vertical dashed lines at recalibration events

    Also plots ground-truth ACC/Cost as horizontal reference lines.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("[analyze_results] matplotlib not available; skipping acc_cost_curves.")
        return
    except Exception as e:
        print(f"[analyze_results] plot_acc_cost_curves skipped: {e}")
        return

    if not episode_records:
        print("[analyze_results] No records; skipping acc_cost_curves.")
        return

    # Group records by (primitive, candidate)
    from collections import defaultdict
    groups: Dict[Tuple, List] = defaultdict(list)
    for r in episode_records:
        key = (r.primitive_name, r.selected_candidate)
        groups[key].append(r)

    primitives = sorted(set(k[0] for k in groups.keys()))
    n_prim = len(primitives)
    if n_prim == 0:
        return

    fig, axes = plt.subplots(
        n_prim, 2, figsize=(14, 4 * n_prim), squeeze=False
    )
    fig.suptitle(f"ACC-Cost Evolution {title_suffix}", fontsize=14, fontweight="bold")

    # Collect recalibration episodes for vertical lines
    recal_eps = sorted(
        {snap.get("episode_at_calibration", 0) for snap in curve_snapshots}
    )

    # Color palette per candidate (consistent across panels)
    all_candidates = sorted(set(k[1] for k in groups.keys()))
    colors = plt.colormaps["tab10"]
    cand_colors = {
        c: colors(i / max(len(all_candidates) - 1, 1))
        for i, c in enumerate(all_candidates)
    }

    for row, prim in enumerate(primitives):
        ax_acc = axes[row, 0]
        ax_cost = axes[row, 1]

        for (p, cand), recs in groups.items():
            if p != prim:
                continue
            color = cand_colors[cand]
            recs_sorted = sorted(recs, key=lambda r: r.episode)
            eps = [r.episode for r in recs_sorted]
            pred_acc = [r.predicted_acc for r in recs_sorted]
            true_acc = [r.true_acc for r in recs_sorted]
            pred_cost = [r.predicted_cost for r in recs_sorted]
            true_cost = [r.true_cost for r in recs_sorted]

            ax_acc.plot(eps, pred_acc, color=color, marker="o", markersize=3,
                        linewidth=1.5, label=f"{cand} (pred)")
            ax_acc.plot(eps, true_acc, color=color, marker="x", markersize=3,
                        linewidth=1, linestyle="--", alpha=0.6, label=f"{cand} (GT)")
            ax_cost.plot(eps, pred_cost, color=color, marker="s", markersize=3,
                         linewidth=1.5, label=f"{cand} (pred)")
            ax_cost.plot(eps, true_cost, color=color, marker="^", markersize=3,
                         linewidth=1, linestyle="--", alpha=0.6)

        # Recalibration vertical lines
        for ax in (ax_acc, ax_cost):
            for ep in recal_eps:
                ax.axvline(ep, color="red", linestyle=":", linewidth=1, alpha=0.7)
            ax.set_xlabel("Episode")
            ax.grid(True, alpha=0.3)

        ax_acc.set_ylabel("ACC")
        ax_acc.set_title(f"[{prim}] Predicted ACC (solid) vs GT (dashed)")
        ax_acc.legend(fontsize=7, loc="best")
        ax_acc.set_ylim(0, 1.05)

        ax_cost.set_ylabel("Cost")
        ax_cost.set_title(f"[{prim}] Predicted Cost (solid) vs GT (dashed)")

        if recal_eps:
            ax_acc.text(0.02, 0.95, "↕ recal", transform=ax_acc.transAxes,
                        fontsize=7, color="red", va="top")
            ax_cost.text(0.02, 0.95, "↕ recal", transform=ax_cost.transAxes,
                         fontsize=7, color="red", va="top")

    plt.tight_layout()
    _save_or_show(output_path, fig, "acc_cost_curves")


# ---------------------------------------------------------------------------
# Visualization: 4D Pareto Frontier
# ---------------------------------------------------------------------------

def plot_pareto_frontier(
    episode_records,
    final_table: List[dict] | None = None,
    output_path: Path | str | None = None,
    title_suffix: str = "",
) -> None:
    """
    Plot the 4D Pareto frontier: ACC vs Cost, with bubble size = pass_rate,
    color = difficulty bucket.

    Also overlays the Pareto-optimal candidates from final_table.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[analyze_results] matplotlib not available; skipping pareto_frontier.")
        return

    if not episode_records:
        print("[analyze_results] No records; skipping pareto_frontier.")
        return

    from collections import defaultdict

    # Aggregate per (primitive, candidate, bucket) from records
    agg: Dict[Tuple, dict] = defaultdict(
        lambda: {"accs": [], "costs": [], "passes": 0, "total": 0}
    )
    for r in episode_records:
        key = (r.primitive_name, r.selected_candidate, r.difficulty_bucket)
        agg[key]["accs"].append(r.predicted_acc)
        agg[key]["costs"].append(r.predicted_cost)
        if r.eval_pass:
            agg[key]["passes"] += 1
        agg[key]["total"] += 1

    # Build scatter data
    x_vals, y_vals, sizes, colors_val, labels = [], [], [], [], []
    bucket_to_color = {"easy": "green", "medium": "orange",
                       "hard": "red", "extreme": "darkred"}
    bucket_order = ["easy", "medium", "hard", "extreme"]

    for (prim, cand, bucket), data in sorted(agg.items()):
        mean_acc = np.mean(data["accs"])
        mean_cost = np.mean(data["costs"])
        pass_rate = data["passes"] / max(data["total"], 1)
        x_vals.append(mean_cost)
        y_vals.append(mean_acc)
        sizes.append(pass_rate * 500 + 50)   # bubble size ∝ pass rate
        colors_val.append(bucket_to_color.get(bucket, "gray"))
        labels.append(f"{prim}/{cand}")

    fig, ax = plt.subplots(figsize=(10, 7))

    # Scatter
    scatter = ax.scatter(
        x_vals, y_vals,
        s=sizes,
        c=colors_val,
        alpha=0.65,
        edgecolors="black",
        linewidths=0.5,
    )

    # Annotate each bubble
    for x, y, label, s in zip(x_vals, y_vals, labels, sizes):
        ax.annotate(
            label,
            (x, y),
            fontsize=7,
            ha="center", va="bottom",
            xytext=(0, 5),
            textcoords="offset points",
        )

    # Pareto frontier (maximize acc, minimize cost)
    # Simple pareto filter
    points = sorted(zip(x_vals, y_vals), key=lambda p: p[0])  # sort by cost
    pareto = []
    max_acc_seen = -1
    for cost, acc in points:
        if acc > max_acc_seen:
            pareto.append((cost, acc))
            max_acc_seen = acc

    if pareto:
        pareto_x, pareto_y = zip(*pareto)
        ax.plot(pareto_x, pareto_y, "b-", linewidth=2,
                label="Pareto frontier", zorder=5)
        ax.scatter(pareto_x, pareto_y, c="blue", s=80, marker="*",
                   zorder=6, label="Pareto optimal")

    # Add cost=0 / acc=1 reference
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    # Legend for colors = difficulty
    from matplotlib.lines import Line2D
    color_legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=10, label=b)
        for b, c in bucket_to_color.items()
    ]
    size_legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=s / 50, alpha=0.65, label=f"pass={s/500:.0%}")
        for s in [150, 350, 550]
    ]
    ax.legend(handles=color_legend + size_legend,
              loc="lower right", fontsize=8, title="Bucket / Pass Rate")

    ax.set_xlabel("Mean Predicted Cost", fontsize=11)
    ax.set_ylabel("Mean Predicted ACC", fontsize=11)
    ax.set_title(f"4D Pareto Frontier: ACC vs Cost (bubble size=pass_rate, color=bucket) {title_suffix}",
                 fontsize=12)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(output_path, fig, "pareto_frontier")


# ---------------------------------------------------------------------------
# Visualization: EMA Profile Convergence
# ---------------------------------------------------------------------------

def plot_profile_convergence(
    curve_snapshots: List[dict],
    output_path: Path | str | None = None,
    title_suffix: str = "",
) -> None:
    """
    Plot how EMA profile curves converge over recalibration events.

    One subplot per primitive, showing ACC curves for each candidate across
    recalibration snapshots (before vs after each calibration).
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[analyze_results] matplotlib not available; skipping profile_convergence.")
        return

    if not curve_snapshots:
        print("[analyze_results] No snapshots; skipping profile_convergence.")
        return

    from collections import defaultdict

    # Snapshot format: {episode_at_calibration: int, table: [{primitive_name, candidate_name, acc_mean, ...}]}
    prim_groups: Dict[str, Dict] = defaultdict(lambda: defaultdict(list))
    for snap in curve_snapshots:
        ep = snap.get("episode_at_calibration", "?")
        table = snap.get("table", [])
        for entry in table:
            prim = entry.get("primitive_name", "unknown")
            cand = entry.get("candidate_name", "unknown")
            acc_mean = entry.get("acc_mean", 0)
            prim_groups[prim][cand].append((ep, acc_mean))

    if not prim_groups:
        print("[analyze_results] No group data in snapshots; skipping.")
        return

    primitives = sorted(prim_groups.keys())
    n = len(primitives)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    fig.suptitle(f"EMA Profile Convergence {title_suffix}", fontsize=13, fontweight="bold")

    for col, prim in enumerate(primitives):
        ax = axes[0, col]
        cand_data = prim_groups[prim]
        for cand, vals in cand_data.items():
            eps, accs = zip(*sorted(vals, key=lambda x: str(x[0])))
            ax.plot(eps, accs, marker="o", markersize=4, linewidth=1.5, label=cand)
        ax.set_title(f"[{prim}] ACC after each recalibration")
        ax.set_xlabel("Calibration event")
        ax.set_ylabel("ACC mean")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    _save_or_show(output_path, fig, "profile_convergence")


# ---------------------------------------------------------------------------
# Visualization: Constraint Violation Evolution
# ---------------------------------------------------------------------------

def plot_constraint_violations(
    episode_records,
    curve_snapshots: List[dict],
    window_size: int = 5,
    output_path: Path | str | None = None,
    title_suffix: str = "",
) -> None:
    """
    Plot constraint violation rate and HITL rejection rate over sliding windows.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[analyze_results] matplotlib not available; skipping violation_plots.")
        return

    if not episode_records:
        print("[analyze_results] No records; skipping violation_plots.")
        return

    from collections import defaultdict

    episodes = sorted(set(r.episode for r in episode_records))

    # Sliding window violation rate
    window_viol = []
    window_hitl = []
    window_labels = []
    for i in range(0, len(episodes), window_size):
        window_eps = episodes[i:i + window_size]
        wrecs = [r for r in episode_records if r.episode in window_eps]
        n = len(wrecs)
        if n == 0:
            continue
        viol_rate = sum(1 for r in wrecs if r.violation_count > 0) / n
        hitl_recs = [r for r in wrecs if "human_in_the_loop" in str(r.constraint_violations)]
        hitl_rej = (
            sum(1 for r in hitl_recs if r.human_approved == False) / len(hitl_recs)
            if hitl_recs else 0.0
        )
        window_viol.append(viol_rate)
        window_hitl.append(hitl_rej)
        window_labels.append(f"E{window_eps[0]}")

    recal_eps = sorted({snap.get("episode_at_calibration", 0) for snap in curve_snapshots})

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f"Constraint & HITL Evolution {title_suffix}", fontsize=13, fontweight="bold")

    x = range(len(window_labels))
    ax1.bar(x, window_viol, color="crimson", alpha=0.7, label="Violation rate")
    ax1.plot(x, window_viol, "r-o", markersize=5)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(window_labels)
    ax1.set_ylabel("Violation Rate")
    ax1.set_title("Constraint Violation Rate (sliding window)")
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.legend()

    for ax in [ax1, ax2]:
        for ep in recal_eps:
            label = "recal"
            ax.axvline(ep / window_size, color="blue", linestyle=":", linewidth=1.5, alpha=0.8)

    ax2.bar(x, window_hitl, color="darkorange", alpha=0.7, label="HITL rejection rate")
    ax2.plot(x, window_hitl, "orange", marker="s", markersize=5)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(window_labels)
    ax2.set_ylabel("HITL Rejection Rate")
    ax2.set_xlabel("Episode window start")
    ax2.set_title("Human-in-the-Loop Rejection Rate (sliding window)")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.legend()

    plt.tight_layout()
    _save_or_show(output_path, fig, "violation_plots")


# ---------------------------------------------------------------------------
# Visualization: Evaluator Quality Distribution
# ---------------------------------------------------------------------------

def plot_evaluator_quality(
    episode_records,
    output_path: Path | str | None = None,
    title_suffix: str = "",
) -> None:
    """
    Plot evaluator quality breakdown:
    - Box plot: quality_score distribution per evaluator tier
    - Error type distribution pie chart
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[analyze_results] matplotlib not available; skipping evaluator_quality.")
        return

    if not episode_records:
        print("[analyze_results] No records; skipping evaluator_quality.")
        return

    from collections import defaultdict

    # Group by evaluator tier
    eval_groups: Dict[str, list] = defaultdict(list)
    for r in episode_records:
        eval_groups[r.evaluator_name].append(r.observed_acc)

    error_types: Dict[str, int] = defaultdict(int)
    for r in episode_records:
        if r.error_type:
            error_types[r.error_type] += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Evaluator Quality Breakdown {title_suffix}", fontsize=13, fontweight="bold")

    if eval_groups:
        tiers = sorted(eval_groups.keys())
        data = [eval_groups[t] for t in tiers]
        bp = ax1.boxplot(data, labels=tiers, patch_artist=True)
        colors = plt.colormaps["Set2"](np.linspace(0, 1, len(tiers)))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        ax1.set_xlabel("Evaluator Tier")
        ax1.set_ylabel("Observed ACC")
        ax1.set_title("ACC Distribution by Evaluator Tier")
        ax1.grid(True, alpha=0.3, axis="y")

    if error_types:
        labels, sizes = zip(*sorted(error_types.items(), key=lambda x: -x[1]))
        colors_pie = plt.colormaps["Set3"](np.linspace(0, 1, len(labels)))
        wedges, texts, autotexts = ax2.pie(
            sizes, labels=labels, autopct="%1.1f%%",
            colors=colors_pie, startangle=90,
        )
        ax2.set_title("Error Type Distribution")
    else:
        ax2.text(0.5, 0.5, "No errors recorded", ha="center", va="center")
        ax2.set_title("Error Type Distribution")

    plt.tight_layout()
    _save_or_show(output_path, fig, "evaluator_quality")


# ---------------------------------------------------------------------------
# Visualization: Template Selection Distribution
# ---------------------------------------------------------------------------

def plot_template_selection_distribution(
    episode_records,
    output_path: Path | str | None = None,
    title_suffix: str = "",
) -> None:
    """
    Plot template selection distribution for structure–configuration analysis:
    - Left: stacked bar chart of template selection by difficulty bucket
    - Right: template upgrade transitions during repair (Strategy C)
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[analyze_results] matplotlib not available; skipping template_distribution.")
        return

    if not episode_records:
        print("[analyze_results] No records; skipping template_distribution.")
        return

    from collections import defaultdict

    # --- Left panel: template distribution by difficulty ---
    bucket_template: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in episode_records:
        bucket_template[r.difficulty_bucket][r.template_id] += 1

    all_templates = sorted(set(
        tid for d in bucket_template.values() for tid in d.keys()
    ))
    buckets = [b for b in ["easy", "medium", "hard", "extreme"] if b in bucket_template]

    TEMPLATE_COLORS = {
        "direct": "#66b3ff",
        "exec_verify": "#ff9966",
        "dual_exec_aggregate": "#66cc99",
        "exec_verify_hci": "#ff6666",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Template Selection Distribution {title_suffix}", fontsize=13, fontweight="bold")

    if buckets and all_templates:
        x = np.arange(len(buckets))
        width = 0.6
        bottoms = np.zeros(len(buckets))
        for tid in all_templates:
            vals = [bucket_template[b].get(tid, 0) for b in buckets]
            color = TEMPLATE_COLORS.get(tid, "#cccccc")
            ax1.bar(x, vals, width, bottom=bottoms, label=tid, color=color)
            bottoms += np.array(vals)
        ax1.set_xticks(x)
        ax1.set_xticklabels(buckets)
        ax1.set_xlabel("Difficulty Bucket")
        ax1.set_ylabel("Count")
        ax1.set_title("Template by Difficulty")
        ax1.legend(fontsize=8, loc="upper left")
        ax1.grid(True, alpha=0.3, axis="y")
    else:
        ax1.text(0.5, 0.5, "No data", ha="center", va="center")
        ax1.set_title("Template by Difficulty")

    # --- Right panel: repair template upgrades ---
    upgrade_transitions: Dict[str, int] = defaultdict(int)
    repair_types: Dict[str, int] = defaultdict(int)
    for r in episode_records:
        if r.template_upgraded_from != "none":
            key = f"{r.template_upgraded_from}\n->\n{r.template_id}"
            upgrade_transitions[key] += 1
        if r.repair_action != "none":
            if r.repair_action.startswith("template_upgraded_to_"):
                repair_types["template_upgrade"] += 1
            elif r.repair_action.startswith("upgraded_to_"):
                repair_types["candidate_upgrade"] += 1
            elif r.repair_action.startswith("evaluator_upgraded_to_"):
                repair_types["evaluator_upgrade"] += 1

    if repair_types:
        labels = list(repair_types.keys())
        sizes = list(repair_types.values())
        colors_repair = ["#ff6666", "#ff9966", "#66b3ff"][:len(labels)]
        ax2.bar(range(len(labels)), sizes, color=colors_repair)
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, fontsize=9, rotation=15)
        ax2.set_ylabel("Count")
        ax2.set_title("Repair Action Distribution")
        ax2.grid(True, alpha=0.3, axis="y")
        # Annotate template upgrade transitions
        if upgrade_transitions:
            text_lines = [f"{k.replace(chr(10), ' ')}: {v}" for k, v in upgrade_transitions.items()]
            ax2.text(0.95, 0.95, "\n".join(text_lines),
                     transform=ax2.transAxes, fontsize=8,
                     verticalalignment="top", horizontalalignment="right",
                     bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    else:
        ax2.text(0.5, 0.5, "No repairs occurred", ha="center", va="center")
        ax2.set_title("Repair Action Distribution")

    plt.tight_layout()
    _save_or_show(output_path, fig, "template_distribution")


# ---------------------------------------------------------------------------
# Visualization: All Plots
# ---------------------------------------------------------------------------

def plot_all(
    result: Dict[str, Any],
    output_dir: Path | str | None = None,
    title_suffix: str = "",
) -> None:
    """
    Generate all visualization plots for a completed experiment run.

    Parameters
    ----------
    result : dict
        The dict returned by run_mvp_experiment().
    output_dir : Path | str | None
        Directory to save figures. If None, displays interactively.
    title_suffix : str
        Extra suffix added to each figure title.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[analyze_results] matplotlib not available; cannot generate plots.")
        return

    episode_records = result.get("episode_records", [])
    curve_snapshots = result.get("curve_snapshots", [])
    final_table = result.get("final_table", [])

    if output_dir is None:
        output_dir = Path("./outputs")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = result.get("config", None)
    prefix = cfg.name if (cfg and hasattr(cfg, "name")) else "exp"
    suffix = f" ({title_suffix})" if title_suffix else ""

    print(f"\n[plot_all] Generating plots in {output_dir} ...")

    # Pre-build analysis dict (needed by plot_convergence)
    try:
        analysis = analyze_experiment_results(
            episode_records, curve_snapshots, final_table,
            config=cfg or _DummyConfig(),
        )
    except Exception as e:
        print(f"[plot_all] Warning: analyze_experiment_results failed: {e}")
        analysis = None

    # Each entry: (name, callable_or_None)
    plot_calls = [
        ("acc_cost_curves",     lambda: plot_acc_cost_curves(
            episode_records, curve_snapshots,
            output_path=output_dir / f"{prefix}_acc_cost_curves.png",
            title_suffix=suffix)),
        ("pareto_frontier",     lambda: plot_pareto_frontier(
            episode_records, final_table,
            output_path=output_dir / f"{prefix}_pareto_frontier.png",
            title_suffix=suffix)),
        ("profile_convergence", lambda: plot_profile_convergence(
            curve_snapshots,
            output_path=output_dir / f"{prefix}_profile_convergence.png",
            title_suffix=suffix)),
        ("violation_plots",     lambda: plot_constraint_violations(
            episode_records, curve_snapshots,
            output_path=output_dir / f"{prefix}_violations.png",
            title_suffix=suffix)),
        ("evaluator_quality",  lambda: plot_evaluator_quality(
            episode_records,
            output_path=output_dir / f"{prefix}_evaluator_quality.png",
            title_suffix=suffix)),
        ("template_distribution", lambda: plot_template_selection_distribution(
            episode_records,
            output_path=output_dir / f"{prefix}_template_distribution.png",
            title_suffix=suffix)),
        ("convergence",         (lambda a: (
            plot_convergence(a, output_path=output_dir / f"{prefix}_convergence.png")
        ) if a else None)(analysis)),
    ]

    saved_count = 0
    for name, fn in plot_calls:
        if fn is None:
            continue
        try:
            fn()
            saved_count += 1
        except Exception as e:
            print(f"  [plot_all] '{name}' failed: {e}")

    print(f"[plot_all] Done: {saved_count}/{len(plot_calls)} plots generated.")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _save_or_show(path: Path | str | None, fig, name: str) -> None:
    """Save figure to path or show interactively."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if path:
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  [saved] {path}")
    else:
        plt.show()
    plt.close(fig)


class _DummyConfig:
    """Minimal placeholder when config is not available."""
    name = "unknown"


# ---------------------------------------------------------------------------
# Plot (matplotlib, optional)
# ---------------------------------------------------------------------------

def plot_convergence(
    analysis: Dict[str, Any],
    output_path: Path | str | None = None,
) -> None:
    """
    Plot prediction convergence over episodes using matplotlib.

    Generates:
    - One subplot per (primitive, candidate) showing MAE over episodes
    - Optionally saves to file.

    Does nothing if matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[analyze_results] matplotlib not available; skipping plot.")
        return

    per_ep = analysis["per_episode_stats"]
    if not per_ep:
        return

    episodes = sorted(per_ep.keys())
    mae_values = [
        per_ep[ep]["mean_pred_acc"] - per_ep[ep]["mean_true_acc"]
        for ep in episodes
    ]
    pass_rates = [per_ep[ep]["pass_rate"] for ep in episodes]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

    # MAE over episodes
    ax1.plot(episodes, mae_values, marker="o", linewidth=2)
    ax1.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Mean Predicted - True ACC")
    ax1.set_title("Prediction Error Convergence")
    ax1.grid(True, alpha=0.3)

    # Pass rate over episodes
    ax2.plot(episodes, pass_rates, marker="s", linewidth=2, color="green")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Pass Rate")
    ax2.set_title("Pass Rate per Episode")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"[analyze_results] Plot saved to {output_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Ablation Comparison
# ---------------------------------------------------------------------------

def compare_ablation(ablation_results: Dict[int, Dict[str, Any]]) -> None:
    """
    Print a comparison table for ablation results.
    """
    print(f"\n{'='*70}")
    print("  ABLATION: Calibration Interval Comparison")
    print(f"{'='*70}")
    print(
        f"  {'K':>4}  {'Episodes':>8}  {'Recal':>6}  "
        f"{'PassRate':>10}  {'Mean_MAE':>9}  {'Final_buffer':>13}"
    )
    print("  " + "-" * 65)

    for k, result in sorted(ablation_results.items()):
        ep_stats = result["per_episode_stats"]
        overall_pass = (
            sum(s["pass_rate"] for s in ep_stats.values()) / len(ep_stats)
            if ep_stats else 0.0
        )
        overall_pass = round(overall_pass, 4)

        # Mean MAE across candidates
        mae_list = [
            m.mae_after_recal
            for m in result["candidate_metrics"]
            if m.mae_after_recal is not None
        ]
        mean_mae = round(sum(mae_list) / len(mae_list), 4) if mae_list else None
        mae_str = f"{mean_mae:.4f}" if mean_mae else "N/A"

        print(
            f"  {k:>4}  "
            f"{result['summary']['unique_tasks']:>8}  "
            f"{result['summary']['n_recalibrations']:>6}  "
            f"{overall_pass:>10.1%}  "
            f"{mae_str:>9}  "
            f"{result['summary']['final_buffer_size']:>13}"
        )

    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Constraint Convergence Analysis
# ---------------------------------------------------------------------------

def analyze_constraint_convergence(
    episode_records: List[EpisodeRecord],
    curve_snapshots: List[dict],
    window_size: int = 10,
) -> Dict[str, Any]:
    """
    Analyze how constraint violation rates evolve over episodes.

    This is the key metric for validating that the constraint feedback loop
    is working: violations should decrease as the system learns.

    Returns
    -------
    dict with:
        - violation_rate_by_window: list of (window_start, rate) tuples
        - violation_rate_by_type: dict[constraint_type -> list of (window_start, rate)]
        - recal_vs_no_recal: dict with violation rates before/after each recalibration
        - candidate_violation_distribution: dict[candidate -> violation_count]
        - bucket_violation_distribution: dict[bucket -> violation_count]
        - hitl_rejection_rate_by_window: list
        - summary: one-line human-readable summary
    """
    if not episode_records:
        return {"summary": "No records"}

    episodes = sorted(set(r.episode for r in episode_records))
    n_records = len(episode_records)

    # Sliding window violation rate
    violation_rate_by_window = []
    for i in range(0, len(episodes), window_size):
        window_eps = episodes[i:i + window_size]
        window_recs = [r for r in episode_records if r.episode in window_eps]
        n_viol = sum(1 for r in window_recs if r.violation_count > 0)
        rate = round(n_viol / len(window_recs), 4) if window_recs else 0.0
        violation_rate_by_window.append((window_eps[0], rate))

    # Per-constraint-type violation rate by window
    from collections import defaultdict
    type_rates: Dict[str, list] = defaultdict(list)
    for i in range(0, len(episodes), window_size):
        window_eps = episodes[i:i + window_size]
        window_recs = [r for r in episode_records if r.episode in window_eps]
        type_counter: Dict[str, int] = defaultdict(int)
        type_total: Dict[str, int] = defaultdict(int)
        for r in window_recs:
            import ast
            try:
                viols = ast.literal_eval(r["constraint_violations"] if isinstance(r.get("constraint_violations"), str) else "[]")
            except Exception:
                viols = []
            for v in viols:
                ctype = v.get("constraint_type", "unknown")
                type_counter[ctype] += 1
            # Count total records that had at least one HITL constraint
            if any(v.get("constraint_type") == "human_in_the_loop" for v in viols):
                type_total["human_in_the_loop"] += 1
        for ctype in type_counter:
            rate = type_counter[ctype] / len(window_recs) if window_recs else 0.0
            type_rates[ctype].append((window_eps[0], round(rate, 4)))

    # Recal vs no-recal violation rates
    if curve_snapshots:
        first_recal_ep = curve_snapshots[0].get("episode_at_calibration", episodes[0])
        before_recal = [r for r in episode_records if r.episode < first_recal_ep]
        after_recal = [r for r in episode_records if r.episode >= first_recal_ep]
    else:
        before_recal = episode_records
        after_recal = []

    n_viol_before = sum(1 for r in before_recal if r.violation_count > 0)
    n_viol_after = sum(1 for r in after_recal if r.violation_count > 0)
    rate_before = round(n_viol_before / len(before_recal), 4) if before_recal else 0.0
    rate_after = round(n_viol_after / len(after_recal), 4) if after_recal else 0.0

    # Candidate violation distribution
    candidate_viol_dist: Dict[str, int] = defaultdict(int)
    for r in episode_records:
        if r.violation_count > 0:
            candidate_viol_dist[r.selected_candidate] += 1

    # Bucket violation distribution
    bucket_viol_dist: Dict[str, int] = defaultdict(int)
    for r in episode_records:
        if r.violation_count > 0:
            bucket_viol_dist[r.difficulty_bucket] += 1

    # HITL rejection rate by window
    hitl_rejection_by_window = []
    for i in range(0, len(episodes), window_size):
        window_eps = episodes[i:i + window_size]
        window_recs = [r for r in episode_records if r.episode in window_eps]
        hitl_recs = [
            r for r in window_recs
            if "human_in_the_loop" in str(r.constraint_violations)
        ]
        n_rejected = sum(1 for r in hitl_recs if r.human_approved == False)
        rate = round(n_rejected / len(hitl_recs), 4) if hitl_recs else 0.0
        hitl_rejection_by_window.append((window_eps[0], rate))

    # Build summary string
    if rate_after < rate_before:
        trend = f"IMPROVING (before={rate_before:.1%}, after={rate_after:.1%})"
    elif rate_after > rate_before:
        trend = f"DEGRADING (before={rate_before:.1%}, after={rate_after:.1%})"
    else:
        trend = f"STABLE (before={rate_before:.1%}, after={rate_after:.1%})"

    summary = (
        f"Violation rate {trend}. "
        f"Top violators: {', '.join(f'{k}({v})' for k, v in sorted(candidate_viol_dist.items(), key=lambda x: -x[1])[:3])}. "
        f"Total records: {n_records}, Total violations: {sum(candidate_viol_dist.values())}"
    )

    return {
        "violation_rate_by_window": violation_rate_by_window,
        "violation_rate_by_type": dict(type_rates),
        "recal_vs_no_recal": {
            "before_recal_episodes": list({r.episode for r in before_recal}),
            "after_recal_episodes": list({r.episode for r in after_recal}),
            "rate_before": rate_before,
            "rate_after": rate_after,
            "improvement": rate_before - rate_after,
        },
        "candidate_violation_distribution": dict(candidate_viol_dist),
        "bucket_violation_distribution": dict(bucket_viol_dist),
        "hitl_rejection_by_window": hitl_rejection_by_window,
        "summary": summary,
    }


def print_constraint_analysis(analysis: Dict[str, Any]) -> None:
    """Print constraint convergence analysis in a human-readable format."""
    print(f"\n{'='*70}")
    print("  CONSTRAINT CONVERGENCE ANALYSIS")
    print(f"{'='*70}")

    # Sliding window violation rates
    vr = analysis.get("violation_rate_by_window", [])
    if vr:
        print(f"\n  [Violation Rate by Episode Window]")
        print(f"  {'Window':>7}  {'Rate':>8}")
        print("  " + "-" * 18)
        for start, rate in vr:
            bar = "█" * int(rate * 20) + "░" * (20 - int(rate * 20))
            print(f"  Ep {start:>3}:  {rate:>7.1%}  {bar}")

    # Recal vs no-recal
    recal = analysis.get("recal_vs_no_recal", {})
    if recal:
        print(f"\n  [Recalibration Impact on Violations]")
        print(f"    Before first recal: {recal['rate_before']:.1%} violation rate")
        print(f"    After  first recal: {recal['rate_after']:.1%} violation rate")
        improvement = recal.get("improvement", 0)
        if improvement > 0:
            print(f"    -> Violation rate {'improved' if improvement > 0 else 'worsened'} by {abs(improvement):.1%}")
        else:
            print(f"    -> No improvement")

    # Candidate distribution
    cand_dist = analysis.get("candidate_violation_distribution", {})
    if cand_dist:
        print(f"\n  [Violation Distribution by Candidate]")
        print(f"  {'Candidate':15s}  {'Violations':>10}")
        print("  " + "-" * 28)
        for cand, cnt in sorted(cand_dist.items(), key=lambda x: -x[1]):
            print(f"  {cand:15s}  {cnt:>10}")

    # Bucket distribution
    bucket_dist = analysis.get("bucket_violation_distribution", {})
    if bucket_dist:
        print(f"\n  [Violation Distribution by Difficulty Bucket]")
        print(f"  {'Bucket':10s}  {'Violations':>10}")
        print("  " + "-" * 23)
        for bucket, cnt in sorted(bucket_dist.items()):
            print(f"  {bucket:10s}  {cnt:>10}")

    # HITL
    hitl = analysis.get("hitl_rejection_by_window", [])
    if hitl:
        print(f"\n  [HITL Rejection Rate by Window]")
        print(f"  {'Window':>7}  {'RejectionRate':>15}")
        print("  " + "-" * 25)
        for start, rate in hitl:
            bar = "█" * int(rate * 20) + "░" * (20 - int(rate * 20))
            print(f"  Ep {start:>3}:  {rate:>14.1%}  {bar}")

    # Summary
    print(f"\n  [Summary]")
    print(f"    {analysis.get('summary', 'N/A')}")
    print(f"\n{'='*70}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import argparse
    from pathlib import Path
    from src.experiments.mvp_experiment import run_mvp_experiment, ExperimentConfig

    parser = argparse.ArgumentParser(description="Analyze MVP experiment results")
    parser.add_argument("--result", type=str, default=None,
                        help="Path to result JSON file (from snapshots JSON). "
                             "If not provided, runs a new quick experiment.")
    parser.add_argument("--episodes", type=int, default=15,
                        help="Number of episodes for quick experiment (default: 15)")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory for plots (default: ./outputs)")
    parser.add_argument("--name", type=str, default="analysis_demo",
                        help="Experiment name")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.result:
        # Load from existing result file
        result_path = Path(args.result)
        with open(result_path, encoding="utf-8") as f:
            raw = json.load(f)
        config_dict = raw.get("config", {})
        class LoadedConfig:
            def __init__(self, d):
                for k, v in d.items():
                    setattr(self, k, v)
        config = LoadedConfig(config_dict)
        episode_records = []
        # Reconstruct EpisodeRecord objects from CSV if available
        csv_path = result_path.parent / f"{result_path.stem.replace('_snapshots', '')}.csv"
        if not csv_path.exists():
            csv_path = result_path.parent / f"{result_path.stem}.csv"
        if csv_path.exists():
            import csv as _csv
            with open(csv_path, encoding="utf-8") as f:
                reader = _csv.DictReader(f)
                for row in reader:
                    from src.experiments.mvp_experiment import EpisodeRecord
                    row = dict(row)
                    row["eval_pass"] = row["eval_pass"] == "True"
                    row["recalibrated"] = row["recalibrated"] == "True"
                    row["human_approved"] = row["human_approved"] == "True"
                    # Safely convert optional/added fields that may not exist in older CSVs
                    for _f, _default in [
                        ("difficulty", 0.0), ("predicted_acc", 0.0), ("predicted_cost", 0.0),
                        ("true_acc", 0.0), ("true_cost", 0.0),
                        ("observed_acc", 0.0), ("observed_cost", 0.0),
                        ("confidence", 1.0), ("evaluator_latency", 0.0), ("evaluator_cost", 0.0),
                        ("violation_count", 0),
                    ]:
                        if _f in row:
                            row[_f] = type(_default)(row[_f])
                    valid_fields = {f.name for f in EpisodeRecord.__dataclass_fields__.values()}
                    episode_records.append(EpisodeRecord(**{k: v for k, v in row.items()
                        if k in valid_fields}))
        curve_snapshots = raw.get("curve_snapshots", [])
        result = {
            "config": config,
            "episode_records": episode_records,
            "curve_snapshots": curve_snapshots,
            "final_table": raw.get("final_table", []),
        }
        print(f"[analyze_results] Loaded from {result_path}: "
              f"{len(episode_records)} records, {len(curve_snapshots)} snapshots")
    else:
        # Run a new quick experiment
        print("=" * 60)
        print(f"Running quick MVP experiment ({args.episodes} episodes)...")
        print("=" * 60)
        config = ExperimentConfig(
            name=args.name,
            n_episodes=args.episodes,
            calibration_interval=5,
            seed=args.seed,
        )
        result = run_mvp_experiment(config)

    # Text analysis
    analysis = analyze_experiment_results(
        episode_records=result["episode_records"],
        curve_snapshots=result["curve_snapshots"],
        final_table=result.get("final_table", []),
        config=result.get("config"),
    )
    print_analysis(analysis)

    # Generate all plots
    plot_all(result, output_dir=args.output_dir, title_suffix=f"[{config.name}]")
