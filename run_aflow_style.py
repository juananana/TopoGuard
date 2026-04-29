"""
run_aflow_style.py
=================
Implements the AFlow-style baseline:
  - Offline-search for the globally optimal topo_id per node_type using training data
  - Fix this topology for ALL test contexts (but still select best model per context)
  - Compare against TopoGuard's task-conditional adaptive selection

Key difference from TopoGuard:
  - TopoGuard: selects best (topo_id, model) per (node_type, difficulty) context
  - AFlow-style: fixes topo_id globally per node_type, only adapts model per context

This answers: is the adaptive topology selection in TopoGuard actually useful,
or does finding one good topology globally work just as well?
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "outputs" / "overall_water_qa_500ep" / "data"
OUT_DIR  = ROOT / "outputs" / "aflow_style"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load saved data ─────────────────────────────────────────────────────────
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

test_recs  = [r for r in all_records if r.get("source") == "test"]
train_recs = [r for r in all_records if r.get("source") == "train"]
print(f"Loaded {len(profiles)} profiles, {len(train_recs)} train, {len(test_recs)} test records")

# ── Constants ───────────────────────────────────────────────────────────────
CONSTRAINT_BUDGET  = 0.5
CONSTRAINT_LATENCY = 0.90
Q_ALPHA, Q_BETA, Q_GAMMA = 0.65, 0.25, 0.10

all_S = [p["S"] for p in profiles if p.get("S") is not None]
PASS_THRESHOLD = float(np.percentile(all_S, 25))
print(f"PASS_THRESHOLD = {PASS_THRESHOLD:.4f}")

# ── Build lookup tables ────────────────────────────────────────────────────────
# profiles indexed by (node_type, difficulty)
profiles_by_nd = defaultdict(list)
for p in profiles:
    profiles_by_nd[(p["node_type"], p["difficulty"])].append(p)

# actual test results indexed by (node_type, difficulty, model, topo_id)
# Each entry is a list of records
test_actual = defaultdict(list)
for r in test_recs:
    key = (r["node_type"], r["difficulty"], r["model"], r["topo_id"])
    test_actual[key].append(r)

# train_actual indexed by (node_type, difficulty, model, topo_id)
train_actual = defaultdict(list)
for r in train_recs:
    key = (r["node_type"], r["difficulty"], r["model"], r["topo_id"])
    train_actual[key].append(r)

# Unique test contexts at two levels:
# 1. (node_type, difficulty, model) - for per-context evaluation
# 2. (node_type, difficulty) - for TopoGuard-style selection
test_contexts_3 = sorted({(r["node_type"], r["difficulty"], r["model"]) for r in test_recs})
test_contexts_2 = sorted({(r["node_type"], r["difficulty"]) for r in test_recs})
print(f"Test contexts (nt,diff,model): {len(test_contexts_3)}")
print(f"Test contexts (nt,diff): {len(test_contexts_2)}")

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

def _avg(lst):
    return float(np.mean(lst)) if lst else 0.0

def get_actual_test(nt, diff, model, topo_id):
    """Get average actual S/C/L from test records for given combination."""
    recs = test_actual.get((nt, diff, model, topo_id), [])
    if recs:
        return {
            "S": _avg([r.get("quality") or r.get("true_quality", 0) for r in recs]),
            "C": _avg([r.get("cost") or r.get("c_main", 0) for r in recs]),
            "L": _avg([r.get("latency") or 0 for r in recs]),
        }
    return None

# ── Step 1: Find globally optimal topo_id per node_type using TRAINING data ──
# For each node_type, compute average Q score across all training contexts for each topo_id.
# The globally optimal topo_id for that node_type is the one with highest average Q.

topo_train_scores = defaultdict(lambda: defaultdict(list))
for r in train_recs:
    nt, diff, model, topo_id = r["node_type"], r["difficulty"], r["model"], r["topo_id"]
    # Find profile for this combination
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

print("\nGlobally optimal topo_id per node_type (from training data):")
for nt in sorted(best_global_topo):
    print(f"  {nt}: topo={best_global_topo[nt]}")

# ── Step 2: Evaluate on TEST set ──────────────────────────────────────────────
# For each test context (node_type, difficulty, model):
#   AFlow-style: use best_global_topo[node_type] (fixed), pick best model by Q
#   TopoGuard: select best (topo_id, model) by Q per context (full adaptive)

results_aflow = []
results_topoguard = []

for ctx in test_contexts_3:
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

    # ── TopoGuard (full): adaptive selection per context ───────────────────
    best_tg = max(front, key=_q_score_p)
    a_tg = get_actual_test(nt, diff, best_tg["model"], best_tg["topo_id"])
    if a_tg:
        results_topoguard.append({
            "S": a_tg["S"], "C": a_tg["C"], "L": a_tg["L"],
            "diff": diff, "nt": nt
        })

    # ── AFlow-style: fixed topo_id, but still select best model ───────────
    global_topo = best_global_topo.get(nt)
    if global_topo is None:
        global_topo = best_tg["topo_id"]  # fallback

    # Filter to candidates with the global topo_id
    global_cands = [p for p in feasible if p["topo_id"] == global_topo]
    if not global_cands:
        global_cands = feasible  # fallback to all feasible

    best_aflow = max(global_cands, key=_q_score_p)
    a_aflow = get_actual_test(nt, diff, best_aflow["model"], best_aflow["topo_id"])
    if a_aflow:
        results_aflow.append({
            "S": a_aflow["S"], "C": a_aflow["C"], "L": a_aflow["L"],
            "diff": diff, "nt": nt
        })

# ── Aggregate ─────────────────────────────────────────────────────────────────
def aggregate(results_list, name):
    if not results_list:
        return None
    avg_s = float(np.mean([x["S"] for x in results_list]))
    avg_c = float(np.mean([x["C"] for x in results_list]))
    avg_l = float(np.mean([x["L"] for x in results_list]))
    n = len(results_list)
    print(f"\n  {name}:")
    print(f"    Avg S: {avg_s:.4f}, Avg C: {avg_c:.6f}, Avg L: {avg_l:.3f}, N: {n}")
    return {"avg_S": round(avg_s, 4), "avg_C": round(avg_c, 6), "avg_L": round(avg_l, 3), "n": n}

print("\n" + "="*70)
print("  TEST SET RESULTS")
print("="*70)

res_tg = aggregate(results_topoguard, "TopoGuard (full, adaptive)")
res_af = aggregate(results_aflow, "AFlow-style (fixed topo_id)")

if res_tg and res_af:
    dS = res_tg["avg_S"] - res_af["avg_S"]
    print(f"\n  Delta S (TopoGuard - AFlow-style): {dS:+.4f}")
    print(f"  Interpretation: {'Adaptive topology selection is valuable' if dS > 0.005 else 'Fixed topology is nearly as good'}")

# Per-difficulty breakdown
print("\n  Per-difficulty breakdown:")
for name, results_list in [("TopoGuard", results_topoguard), ("AFlow-style", results_aflow)]:
    by_diff = defaultdict(list)
    for x in results_list:
        by_diff[x["diff"]].append(x["S"])
    print(f"  {name}:")
    for d in ["easy", "medium", "hard"]:
        vals = by_diff.get(d, [])
        if vals:
            print(f"    {d}: avg_S={np.mean(vals):.4f} N={len(vals)}")

# Save results
OUT_DIR.mkdir(parents=True, exist_ok=True)
results = {
    "TopoGuard (full)": res_tg,
    "AFlow-style (fixed topo_id)": res_af,
}
with open(OUT_DIR / "aflow_style_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved → {OUT_DIR / 'aflow_style_results.json'}")

global_candidates = {nt: {"topo_id": t} for nt, t in best_global_topo.items()}
with open(OUT_DIR / "global_candidates.json", "w", encoding="utf-8") as f:
    json.dump(global_candidates, f, indent=2, ensure_ascii=False)
print(f"Saved → {OUT_DIR / 'global_candidates.json'}")