"""
run_ablation_aflow_v2.py
=======================
Adds AFlow-Style as a strategy in the strategy_comparison framework,
so it runs under the SAME episode-based protocol as TopoGuard (S=0.798).

Strategy: globally optimal topo_id per node_type from TRAINING data,
          then fixed for ALL test contexts (no adaptation per difficulty).
          Still selects best model per context by Q score.

This uses the same data loading and lookup tables as experiment_overall.py
but adds AFlow-Style as an additional strategy.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "outputs" / "overall_water_qa_500ep" / "data"

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

# Use the same dataclass-style records as experiment_overall.py expects
from dataclasses import dataclass

@dataclass
class EpisodeRecord:
    node_type: str
    difficulty: str
    model: str
    topo_id: str
    quality: float
    cost: float
    latency: float

test_recs = []
train_recs = []
for r in all_records:
    rec = EpisodeRecord(
        node_type=r["node_type"],
        difficulty=r["difficulty"],
        model=r["model"],
        topo_id=r["topo_id"],
        quality=r.get("S") or r.get("quality", 0),
        cost=r.get("C") or r.get("cost", 0),
        latency=r.get("L") or r.get("latency", 0),
    )
    if r.get("source") == "test":
        test_recs.append(rec)
    else:
        train_recs.append(rec)

print(f"Loaded {len(profiles)} profiles, {len(train_recs)} train, {len(test_recs)} test records")

# ── Constants (must match experiment_overall.py) ───────────────────────────
# ── Constants (inline to avoid import chain issues) ─────────────────────────
Q_ALPHA, Q_BETA, Q_GAMMA = 0.65, 0.25, 0.10
S_SCALE = 1.5
CONSTRAINT_BUDGET = 0.5
CONSTRAINT_LATENCY = 0.90
PASS_THRESHOLD = 0.82

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

# ── Build lookup tables (same as strategy_comparison) ───────────────────────
test_by_ctx = defaultdict(list)
for r in test_recs:
    test_by_ctx[(r.node_type, r.difficulty, r.model)].append(r)

topo_actual = {}
for ctx_key, recs in test_by_ctx.items():
    by_topo = defaultdict(list)
    for r in recs:
        by_topo[r.topo_id].append(r)
    for topo_id, topo_recs in by_topo.items():
        entry = {
            "actual_S": np.mean([r.quality for r in topo_recs]),
            "actual_C": np.mean([r.cost for r in topo_recs]),
            "actual_L": np.mean([r.latency for r in topo_recs]),
        }
        topo_actual[ctx_key + (topo_id,)] = entry

profiles_by_nd = defaultdict(list)
for p in profiles:
    profiles_by_nd[(p["node_type"], p["difficulty"])].append(p)

# ── Step 1: Find globally optimal topo_id per node_type from TRAINING ────────
train_actual = defaultdict(list)
for r in train_recs:
    key = (r.node_type, r.difficulty, r.model, r.topo_id)
    train_actual[key].append(r)

topo_train_scores = defaultdict(lambda: defaultdict(list))
for (nt, diff, model, topo_id), recs in train_actual.items():
    pts = [p for p in profiles_by_nd.get((nt, diff), [])
           if p["model"] == model and p["topo_id"] == topo_id]
    if pts:
        q = Q_ALPHA * (pts[0]["S"] / S_SCALE) - Q_BETA * pts[0].get("C_norm", pts[0]["C"]) - Q_GAMMA * pts[0].get("L_norm", pts[0]["L"])
        topo_train_scores[nt][topo_id].append(q)

best_global_topo = {}
for nt, topo_scores in topo_train_scores.items():
    topo_avg_q = {t: float(np.mean(qs)) for t, qs in topo_scores.items() if qs}
    if topo_avg_q:
        best_global_topo[nt] = max(topo_avg_q, key=topo_avg_q.get)

print("\nGlobally optimal topo_id per node_type (from training data):")
for nt in sorted(best_global_topo):
    print(f"  {nt}: topo={best_global_topo[nt]}")

# ── Step 2: Compute TopoGuard (Pareto+Q) and AFlow-Style per context ────────
def _q_score_p(p):
    return Q_ALPHA * (p["S"] / S_SCALE) - Q_BETA * p.get("C_norm", p["C"]) - Q_GAMMA * p.get("L_norm", p["L"])

def actual_for_model_topo(nt, diff, model, topo_id):
    key = (nt, diff, model, topo_id)
    return topo_actual.get(key, {}).get("actual_S")

strategies = defaultdict(list)

for ctx_key, recs in sorted(test_by_ctx.items()):
    nt, diff, model = ctx_key
    pts = profiles_by_nd.get((nt, diff), [])
    if not pts:
        continue

    feasible = filter_by_constraints(pts, CONSTRAINT_BUDGET, CONSTRAINT_LATENCY)
    if not feasible:
        feasible = pts

    front = _pareto_frontier(feasible)
    if not front:
        front = feasible

    # ── TopoGuard (full adaptive) ────────────────────────────────────────────
    best_tg = max(front, key=_q_score_p) if front else None
    if best_tg:
        s_tg = actual_for_model_topo(nt, diff, best_tg.get("model"), best_tg["topo_id"])
        if s_tg is not None:
            c_tg = topo_actual.get((nt, diff, best_tg.get("model"), best_tg["topo_id"]), {}).get("actual_C", 0)
            l_tg = topo_actual.get((nt, diff, best_tg.get("model"), best_tg["topo_id"]), {}).get("actual_L", 0)
            strategies["TopoGuard"].append({
                "S": s_tg, "C": c_tg, "L": l_tg, "diff": diff
            })

    # ── AFlow-Style: fixed topo_id per node_type, best model per context ────
    global_topo = best_global_topo.get(nt)
    if global_topo is None:
        global_topo = best_tg["topo_id"] if best_tg else None

    if global_topo:
        aflow_cands = [p for p in feasible if p["topo_id"] == global_topo]
        if not aflow_cands:
            aflow_cands = feasible  # fallback
        best_aflow = max(aflow_cands, key=_q_score_p) if aflow_cands else None
        if best_aflow:
            s_af = actual_for_model_topo(nt, diff, best_aflow.get("model"), best_aflow["topo_id"])
            if s_af is not None:
                c_af = topo_actual.get((nt, diff, best_aflow.get("model"), best_aflow["topo_id"]), {}).get("actual_C", 0)
                l_af = topo_actual.get((nt, diff, best_aflow.get("model"), best_aflow["topo_id"]), {}).get("actual_L", 0)
                strategies["AFlow-Style"].append({
                    "S": s_af, "C": c_af, "L": l_af, "diff": diff
                })

# ── Aggregate (episode-based, matching main table) ─────────────────────────
print("\n" + "="*70)
print("  EPISODE-BASED RESULTS (same protocol as main table)")
print("="*70)
print(f"\n  {'Method':<30} | {'Avg S':>7} | {'Avg C':>12} | {'Avg L (s)':>10} | {'N':>4}")
print(f"  {'─'*70}")

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

if "TopoGuard" in results and "AFlow-Style" in results:
    dS = results["TopoGuard"]["avg_S"] - results["AFlow-Style"]["avg_S"]
    print(f"\n  Delta S (TopoGuard - AFlow-style): {dS:+.4f}")

# Per-difficulty
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

# Save
OUT_DIR = ROOT / "outputs" / "ablation_aflow"
OUT_DIR.mkdir(parents=True, exist_ok=True)
with open(OUT_DIR / "aflow_v2_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved → {OUT_DIR / 'aflow_v2_results.json'}")