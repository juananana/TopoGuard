"""
run_ablation_aflow.py
====================
Runs AFlow-Style baseline in the same ablation framework:
  AFlow-Style: globally optimal topo_id per node_type from TRAINING data,
              then fixed for ALL test contexts (no adaptation per difficulty).

Also recomputes TopoGuard (full) in the same framework for fair comparison.

Loads profiles and test records from outputs/overall_water_qa_500ep/data/
Outputs: outputs/ablation_aflow/aflow_results.json
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "outputs" / "overall_water_qa_500ep" / "data"
OUT_DIR  = ROOT / "outputs" / "ablation_aflow"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load saved data ───────────────────────────────────────────────────────────
profiles = []
with open(DATA_DIR / "profiles.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            profiles.append(json.loads(line))

all_records = []
with open(DATA_DIR / "episode_records.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            all_records.append(json.loads(line))

# Split train/test by source field
test_recs  = [r for r in all_records if r.get("source") == "test"]
train_recs = [r for r in all_records if r.get("source") == "train"]
print(f"Loaded {len(profiles)} profiles, {len(train_recs)} train, {len(test_recs)} test records")

# ── Constants (must match experiment_overall.py) ──────────────────────────────
CONSTRAINT_BUDGET  = 0.5
CONSTRAINT_LATENCY = 0.90
Q_ALPHA, Q_BETA, Q_GAMMA = 0.65, 0.25, 0.10

# Data-driven PASS_THRESHOLD (25th pct of profile S values)
all_S = [p["S"] for p in profiles if p.get("S") is not None]
PASS_THRESHOLD = float(np.percentile(all_S, 25))
print(f"PASS_THRESHOLD = {PASS_THRESHOLD:.4f}")

# ── Build lookup tables ────────────────────────────────────────────────────────
# profiles indexed by (node_type, difficulty)
profiles_by_nd = defaultdict(list)
for p in profiles:
    profiles_by_nd[(p["node_type"], p["difficulty"])].append(p)

# topo_actual: (node_type, difficulty, model, topo_id) → actual S/C/L from TEST records
topo_actual = {}
for r in test_recs:
    key = (r["node_type"], r["difficulty"], r["model"], r["topo_id"])
    if key not in topo_actual:
        topo_actual[key] = {
            "actual_S": r.get("S") or r.get("quality"),
            "actual_C": r.get("C") or r.get("cost", 0),
            "actual_L": r.get("L") or r.get("latency", 0),
        }

# train_actual: (node_type, difficulty, model, topo_id) → list of train records
train_actual = defaultdict(list)
for r in train_recs:
    key = (r["node_type"], r["difficulty"], r["model"], r["topo_id"])
    train_actual[key].append({
        "S": r.get("S") or r.get("quality"),
        "C": r.get("C") or r.get("cost", 0),
        "L": r.get("L") or r.get("latency", 0),
    })

# Unique test contexts: (node_type, difficulty, model)
test_contexts = list({(r["node_type"], r["difficulty"], r["model"]) for r in test_recs})
print(f"Test contexts: {len(test_contexts)}")

# ── Helpers ───────────────────────────────────────────────────────────────────
def filter_by_constraints(pts, budget, latency):
    return [p for p in pts
            if p.get("C_norm", p["C"]) <= budget
            and p.get("L_norm", p["L"]) <= latency]

def _pareto_frontier(pts):
    if not pts:
        return []
    dominated = set()
    for i, a in enumerate(pts):
        for j, b in enumerate(pts):
            if i == j or j in dominated:
                continue
            if (b["S"] >= a["S"] and b["C"] <= a["C"] and b["L"] <= a["L"]
                    and (b["S"] > a["S"] or b["C"] < a["C"] or b["L"] < a["L"])):
                dominated.add(i)
                break
    return [p for i, p in enumerate(pts) if i not in dominated]

def _q_score_p(p):
    return Q_ALPHA * p["S"] - Q_BETA * p.get("C_norm", p["C"]) - Q_GAMMA * p.get("L_norm", p["L"])

def actual_for(nt, diff, model, topo_id):
    """Get actual S/C/L from test records for given combination."""
    key = (nt, diff, model, topo_id)
    return topo_actual.get(key, {})

def train_avg_for(nt, diff, model, topo_id):
    """Get average train S/C/L from train records for given combination."""
    key = (nt, diff, model, topo_id)
    recs = train_actual.get(key, [])
    if not recs:
        return None
    return {
        "S": np.mean([r["S"] for r in recs]),
        "C": np.mean([r["C"] for r in recs]),
        "L": np.mean([r["L"] for r in recs]),
    }

# ── Step 1: Find globally optimal topo_id per node_type from TRAINING data ──
# For each node_type, collect Q scores for each topo_id across all training contexts,
# then pick the topo_id with highest average Q score.

topo_train_scores = defaultdict(lambda: defaultdict(list))
for (nt, diff, model, topo_id), recs in train_actual.items():
    # Get profile entry for this combination to compute Q score
    pts = [p for p in profiles_by_nd.get((nt, diff), [])
           if p["model"] == model and p["topo_id"] == topo_id]
    if pts:
        q = _q_score_p(pts[0])
        topo_train_scores[nt][topo_id].append(q)

best_global_topo = {}
for nt, topo_scores in topo_train_scores.items():
    # Average Q across all training contexts for each topo_id
    topo_avg_q = {t: float(np.mean(qs)) for t, qs in topo_scores.items() if qs}
    if topo_avg_q:
        best_global_topo[nt] = max(topo_avg_q, key=topo_avg_q.get)
        print(f"  {nt}: best_topo={best_global_topo[nt]} (avg_Q={topo_avg_q[best_global_topo[nt]]:.4f})")

print(f"\nGlobally optimal topo_id per node_type (from training data):")
for nt in sorted(best_global_topo):
    print(f"  {nt}: {best_global_topo[nt]}")

# ── Step 2: Evaluate AFlow-Style and TopoGuard on TEST data ─────────────────
strategies = defaultdict(list)

for ctx in test_contexts:
    nt, diff, model = ctx
    pts = profiles_by_nd.get((nt, diff), [])
    if not pts:
        continue

    feasible = filter_by_constraints(pts, CONSTRAINT_BUDGET, CONSTRAINT_LATENCY)
    if not feasible:
        feasible = pts

    front = _pareto_frontier(feasible)
    if not front:
        front = feasible

    # ── TopoGuard (full): adaptive per-context selection ───────────────────
    best_tg = max(front, key=_q_score_p)
    a_tg = actual_for(nt, diff, best_tg["model"], best_tg["topo_id"])
    if a_tg.get("actual_S") is not None:
        strategies["TopoGuard (full)"].append({
            "S": a_tg["actual_S"], "C": a_tg["actual_C"], "L": a_tg["actual_L"], "diff": diff
        })

    # ── AFlow-Style: fixed topo_id globally per node_type ──────────────────
    global_topo = best_global_topo.get(nt)
    if global_topo is None:
        global_topo = best_tg["topo_id"]  # fallback if not found

    # Filter feasible candidates to those with global_topo
    aflow_cands = [p for p in feasible if p["topo_id"] == global_topo]
    if not aflow_cands:
        # Fallback: if global topo not in feasible for this context, use all feasible
        aflow_cands = feasible

    # Select best model (by Q score) among candidates with global topo
    best_aflow = max(aflow_cands, key=_q_score_p)
    a_aflow = actual_for(nt, diff, best_aflow["model"], best_aflow["topo_id"])
    if a_aflow.get("actual_S") is not None:
        strategies["AFlow-Style (fixed topo)"].append({
            "S": a_aflow["actual_S"], "C": a_aflow["actual_C"], "L": a_aflow["actual_L"], "diff": diff
        })

# ── Aggregate ─────────────────────────────────────────────────────────────────
print("\n" + "="*75)
print("  TEST SET RESULTS (same evaluation protocol)")
print("="*75)
print(f"\n  {'Method':<30} | {'Avg S':>7} | {'Avg C':>12} | {'Avg L (s)':>10} | {'N':>4}")
print(f"  {'─'*75}")

results = {}
for name, items in strategies.items():
    if items:
        avg_s = float(np.mean([x["S"] for x in items]))
        avg_c = float(np.mean([x["C"] for x in items]))
        avg_l = float(np.mean([x["L"] for x in items]))
        n = len(items)
        results[name] = {"avg_S": round(avg_s, 4), "avg_C": round(avg_c, 6),
                         "avg_L": round(avg_l, 3), "n": n}
        print(f"  {name:<30} | {avg_s:>7.4f} | {avg_c:>12.6f} | {avg_l:>10.3f} | {n:>4}")

# Delta
if "TopoGuard (full)" in results and "AFlow-Style (fixed topo)" in results:
    dS = results["TopoGuard (full)"]["avg_S"] - results["AFlow-Style (fixed topo)"]["avg_S"]
    print(f"\n  Delta S (TopoGuard - AFlow-style): {dS:+.4f}")
    print(f"  Interpretation: {'Adaptive topology selection is valuable' if dS > 0.005 else 'Fixed topology is nearly as good'}")

# Per-difficulty breakdown
print("\n  Per-difficulty breakdown:")
for name in strategies:
    by_diff = defaultdict(list)
    for x in strategies[name]:
        by_diff[x["diff"]].append(x["S"])
    print(f"  {name}:")
    for d in ["easy", "medium", "hard"]:
        vals = by_diff.get(d, [])
        if vals:
            print(f"    {d}: avg_S={np.mean(vals):.4f} N={len(vals)}")

# Save results
OUT_DIR.mkdir(parents=True, exist_ok=True)
with open(OUT_DIR / "aflow_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved → {OUT_DIR / 'aflow_results.json'}")

# Also save global candidates
global_candidates = {nt: {"topo_id": t} for nt, t in best_global_topo.items()}
with open(OUT_DIR / "global_candidates.json", "w", encoding="utf-8") as f:
    json.dump(global_candidates, f, indent=2, ensure_ascii=False)
print(f"Saved → {OUT_DIR / 'global_candidates.json'}")