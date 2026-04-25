"""
collect_real_qa_data.py
=======================
知识问答类任务：真实数据收集脚本

流程（mock 流程的反面）：
  Step 1: 准备 Q&A 数据集（问题 + 标准答案）
  Step 2: 对每个 (question, tool_config) 组合运行 pipeline
  Step 3: 调用 LLM 裁判评估质量 + 记录成本 + 记录延迟
  Step 4: 输出 .jsonl 记录（格式与 profile_estimator 兼容）

工具配置（对应 pareto_demo 的 GROUND_TRUTH）：
  tool_id 格式: NODE_TYPE/tool_name
  例如: EXECUTOR/llm_small, VERIFIER/llm_check, AGGREGATOR/voting

工具设计（12 个候选）:
  EXECUTOR:  4 个 = small_basic / large_basic / structured / reason
  VERIFIER:  4 个 = rule_check / ml_check / llm_check / oracle_check
  AGGREGATOR: 3 个 = voting / weighted_voting / nn_aggregate
  HUMAN_GATE: 1 个 = compliance_review

质量评估方式：
  - 自动评估：用 LLM 裁判打分（1-5 分 → 归一化到 [0,1]）
  - 成本：API 调用费用（根据模型定价表）
  - 延迟：实际 wall-clock time

Usage:
  python src/experiments/collect_real_qa_data.py --dataset hydrology_qa
"""

import json
import time
import random
import argparse
import math
import subprocess
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).parent.parent.parent
DATA_DIR = _SRC_DIR / "data" / "knowledge_qa"
OUT_DIR  = _SRC_DIR / "data" / "knowledge_qa" / "collected"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Tool Definitions（对应 pareto_demo 的 GROUND_TRUTH）
# ---------------------------------------------------------------------------
# 格式: tool_id → {"node_type": str, "model": str, "cost_per_1k": float}
TOOL_DEFS = {
    # ── EXECUTOR ──────────────────────────────────────────────────────────
    "EXECUTOR/llm_small_basic": {
        "node_type": "EXECUTOR",
        "model": "gpt-4o-mini",
        "cost_per_1k_input": 0.00015,
        "cost_per_1k_output": 0.0006,
        "avg_input_tokens": 500,
        "avg_output_tokens": 200,
    },
    "EXECUTOR/llm_large_basic": {
        "node_type": "EXECUTOR",
        "model": "gpt-4o",
        "cost_per_1k_input": 0.0025,
        "cost_per_1k_output": 0.01,
        "avg_input_tokens": 500,
        "avg_output_tokens": 300,
    },
    "EXECUTOR/llm_structured": {
        "node_type": "EXECUTOR",
        "model": "gpt-4-turbo",
        "cost_per_1k_input": 0.01,
        "cost_per_1k_output": 0.03,
        "avg_input_tokens": 500,
        "avg_output_tokens": 400,
    },
    "EXECUTOR/llm_reason": {
        "node_type": "EXECUTOR",
        "model": "gpt-4o",
        "cost_per_1k_input": 0.0025,
        "cost_per_1k_output": 0.01,
        "avg_input_tokens": 800,
        "avg_output_tokens": 600,
        "use_cot": True,  # chain-of-thought
    },
    # ── VERIFIER ───────────────────────────────────────────────────────────
    "VERIFIER/rule_check": {
        "node_type": "VERIFIER",
        "model": "rule_based",
        "cost_per_call": 0.001,
    },
    "VERIFIER/ml_check": {
        "node_type": "VERIFIER",
        "model": "gpt-4o-mini",
        "cost_per_1k_input": 0.00015,
        "cost_per_1k_output": 0.0006,
        "avg_input_tokens": 300,
        "avg_output_tokens": 50,
    },
    "VERIFIER/llm_check": {
        "node_type": "VERIFIER",
        "model": "gpt-4o",
        "cost_per_1k_input": 0.0025,
        "cost_per_1k_output": 0.01,
        "avg_input_tokens": 400,
        "avg_output_tokens": 80,
    },
    "VERIFIER/oracle_check": {
        "node_type": "VERIFIER",
        "model": "gpt-4-turbo",
        "cost_per_1k_input": 0.01,
        "cost_per_1k_output": 0.03,
        "avg_input_tokens": 600,
        "avg_output_tokens": 100,
    },
    # ── AGGREGATOR ─────────────────────────────────────────────────────────
    "AGGREGATOR/voting": {
        "node_type": "AGGREGATOR",
        "model": "rule_based",
        "cost_per_call": 0.0001,
    },
    "AGGREGATOR/weighted_voting": {
        "node_type": "AGGREGATOR",
        "model": "gpt-4o-mini",
        "cost_per_1k_input": 0.00015,
        "cost_per_1k_output": 0.0006,
        "avg_input_tokens": 600,
        "avg_output_tokens": 30,
    },
    "AGGREGATOR/nn_aggregate": {
        "node_type": "AGGREGATOR",
        "model": "gpt-4o",
        "cost_per_1k_input": 0.0025,
        "cost_per_1k_output": 0.01,
        "avg_input_tokens": 800,
        "avg_output_tokens": 50,
    },
    # ── HUMAN_GATE ─────────────────────────────────────────────────────────
    "HUMAN_GATE/compliance_review": {
        "node_type": "HUMAN_GATE",
        "model": "human",
        "cost_per_call": 5.0,  # 人工审核成本
    },
}


# ---------------------------------------------------------------------------
# 计算成本
# ---------------------------------------------------------------------------
def calc_cost(tool_id: str, n_calls: int = 1) -> float:
    """根据工具定义计算一次调用的成本（美元）。"""
    td = TOOL_DEFS.get(tool_id, {})
    model = td.get("model", "")

    if model == "rule_based":
        return td.get("cost_per_call", 0.0) * n_calls
    if model == "human":
        return td.get("cost_per_call", 5.0) * n_calls
    if "gpt" in model:
        inp = td.get("cost_per_1k_input", 0) * td.get("avg_input_tokens", 500) / 1000 * n_calls
        out = td.get("cost_per_1k_output", 0) * td.get("avg_output_tokens", 200) / 1000 * n_calls
        return inp + out
    return 0.001 * n_calls  # fallback


# ---------------------------------------------------------------------------
# Q&A 数据集加载
# ---------------------------------------------------------------------------
def load_qa_dataset(dataset_name: str) -> List[Dict[str, Any]]:
    """
    加载 Q&A 数据集。

    数据格式（JSONL 或 JSON）：
      {
        "question": "...",
        "answer": "...",         # 标准答案（用于评估）
        "difficulty": "easy",     # 可选
        "domain": "...",          # 可选
        "context": "...",         # 可选的上下文文档
        "id": "..."
      }
    """
    # 优先从项目 qa_pairs 目录加载
    qa_dir = _SRC_DIR / "qa_pairs"
    candidates = [
        qa_dir / f"{dataset_name}.json",
        qa_dir / f"{dataset_name}.jsonl",
        qa_dir / dataset_name / f"{dataset_name}.json",
        qa_dir / dataset_name / f"{dataset_name}.jsonl",
        DATA_DIR / f"{dataset_name}.json",
        DATA_DIR / f"{dataset_name}.jsonl",
    ]

    for path in candidates:
        if path.exists():
            print(f"  Found dataset at: {path}")
            if path.suffix == ".jsonl":
                records = []
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        records.append(json.loads(line))
                return records
            elif path.suffix == ".json":
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and "data" in data:
                    return data["data"]
                else:
                    return [data]

    print(f"  [WARN] Dataset not found in: {[str(c) for c in candidates]}")
    print(f"  Generating synthetic Q&A dataset (for demo)...")
    return _generate_synthetic_qa(30)


def _generate_synthetic_qa(n: int) -> List[Dict[str, Any]]:
    """生成合成 Q&A 数据（当没有真实数据时使用）。"""
    domains = ["hydrology", "earth_science", "climate", "environmental", "geology"]
    topics = [
        "storm_surge_mechanism",
        "tide_prediction",
        "flood_risk_assessment",
        "coastal_erosion",
        "sea_level_rise",
        "wetland_ecology",
        "groundwater_flow",
        "river_discharge",
    ]
    templates = [
        "What is the primary cause of {topic} in {domain} systems?",
        "How does {factor} affect {topic} according to recent research?",
        "Compare {topic} between {region1} and {region2} in {domain} context.",
        "What are the key parameters for modeling {topic}?",
        "Explain the relationship between {factor} and {topic} in {domain}.",
    ]
    factors = ["temperature", "precipitation", "wind_speed", "topography", "vegetation"]
    regions = ["tropical", "temperate", "arid", "coastal"]

    questions = []
    for i in range(n):
        tpl = random.choice(templates)
        topic = random.choice(topics)
        domain = random.choice(domains)
        factor = random.choice(factors)
        region1 = random.choice(regions)
        region2 = random.choice(regions)
        q = tpl.format(
            topic=topic, domain=domain,
            factor=factor, region1=region1, region2=region2
        )
        questions.append({
            "id": f"qa_{i:03d}",
            "question": q,
            "answer": f"[Reference answer for: {q}]",
            "domain": domain,
            "topic": topic,
            "difficulty": random.choice(["easy", "medium", "hard"]),
            "source": "synthetic",
        })
    return questions


# ---------------------------------------------------------------------------
# LLM 调用（核心数据收集步骤）
# ---------------------------------------------------------------------------
def call_llm(
    question: str,
    tool_id: str,
    context: Optional[str] = None,
    use_cot: bool = False,
    judge_answer: Optional[str] = None,
) -> Tuple[float, float, float, str]:
    """
    调用 LLM 执行问答或验证，返回 (quality, cost, latency, response).

    质量评估：
      1. 如果提供了 judge_answer（标准答案）：用 LLM 裁判打分
      2. 否则：返回 response_length / expected_length 作为代理质量

    返回：(quality[0-1], cost_usd, latency_sec, response_text)
    """
    import openai
    # or: from anthropic import Anthropic

    td = TOOL_DEFS.get(tool_id, {})
    model = td.get("model", "gpt-4o-mini")

    # ── Build prompt ────────────────────────────────────────────────────────
    if td.get("node_type") == "EXECUTOR":
        prompt = f"Question: {question}\n"
        if context:
            prompt += f"Context:\n{context}\n"
        if use_cot:
            prompt += "Think step by step, then provide your answer.\n"
        prompt += "Answer:"

    elif td.get("node_type") == "VERIFIER":
        if judge_answer:
            prompt = (
                f"Reference answer: {judge_answer}\n"
                f"Submitted answer: {question}\n"
                "Rate the submitted answer's quality from 1 to 5 (1=completely wrong, 5=perfect): "
            )
        else:
            prompt = question + "\nIs this answer correct? Rate 1-5:"

    elif td.get("node_type") == "AGGREGATOR":
        prompt = f"Combine the following answers into one coherent response:\n{question}\nCombined answer:"

    else:
        prompt = question

    # ── Call API ────────────────────────────────────────────────────────────
    start = time.time()
    try:
        # OpenAI API（需要设置 OPENAI_API_KEY 环境变量）
        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=td.get("avg_output_tokens", 300),
        )
        response = resp.choices[0].message.content or ""
        latency = time.time() - start

        # ── Quality evaluation ──────────────────────────────────────────────
        if judge_answer and td.get("node_type") == "EXECUTOR":
            # LLM 裁判评估
            judge_prompt = (
                f"Reference: {judge_answer}\n\n"
                f"Answer: {response}\n\n"
                "Score the answer 1-5 (1=wrong, 3=partial, 5=perfect). "
                "Return ONLY a number."
            )
            judge_resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": judge_prompt}],
                max_tokens=5,
                temperature=0.0,
            )
            try:
                score_text = judge_resp.choices[0].message.content.strip()
                score = float(score_text.split()[0].strip(".,;:"))
                quality = max(0.0, min(1.0, score / 5.0))
            except (ValueError, IndexError):
                quality = 0.5  # fallback
        else:
            # 代理质量：response 长度 + 关键词匹配
            quality = _proxy_quality(response, judge_answer or question)

        cost = calc_cost(tool_id, n_calls=1)

    except Exception as e:
        # API 调用失败时的 fallback
        print(f"    [WARN] LLM call failed for {tool_id}: {e}")
        response = f"[Error: {e}]"
        quality = 0.3
        cost = 0.0
        latency = time.time() - start

    return quality, cost, latency, response


def _proxy_quality(response: str, reference: str) -> float:
    """代理质量评估：当没有标准答案时使用。"""
    if not response or len(response) < 10:
        return 0.1
    # 基于 response 长度和关键词
    score = min(0.9, len(response) / 500)  # 长度代理
    keywords = ["analysis", "result", "conclusion", "therefore", "because", "however"]
    kw_hits = sum(1 for kw in keywords if kw.lower() in response.lower())
    score += kw_hits * 0.05
    return min(1.0, max(0.1, score))


# ---------------------------------------------------------------------------
# 单任务数据收集
# ---------------------------------------------------------------------------
def collect_for_question(
    qa_item: Dict[str, Any],
    tool_ids: List[str],
    n_repeats: int = 3,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    对一个 Q&A 项，跑多个工具配置，收集真实 Q/C/L 数据。

    Args:
        qa_item: {"question": ..., "answer": ..., ...}
        tool_ids: 要测试的工具 ID 列表
        n_repeats: 每个 (question, tool) 跑几次（模拟方差）
        seed: 随机种子

    Returns: list of record dicts
    """
    question = qa_item["question"]
    answer   = qa_item.get("answer", "")
    qa_id    = qa_item.get("id", "unknown")
    domain   = qa_item.get("domain", "general")
    difficulty = qa_item.get("difficulty", "medium")

    records = []
    for tool_id in tool_ids:
        td = TOOL_DEFS.get(tool_id, {})
        node_type = td.get("node_type", "unknown")
        use_cot = td.get("use_cot", False)

        for rep in range(n_repeats):
            quality, cost, latency, response = call_llm(
                question=question,
                tool_id=tool_id,
                context=qa_item.get("context"),
                use_cot=use_cot,
                judge_answer=answer,
            )

            records.append({
                "task_id": f"{qa_id}_{tool_id.replace('/','_')}_{rep}",
                "qa_id": qa_id,
                "question": question,
                "domain": domain,
                "difficulty": difficulty,
                "node_type": node_type,
                "tool_id": tool_id,
                "quality": round(quality, 4),
                "c_total": round(cost, 4),
                "observed_latency": round(latency * 1000, 1),  # ms
                "response_length": len(response),
                "source": "real_collected",
                "timestamp": datetime.now().isoformat(),
            })

    return records


# ---------------------------------------------------------------------------
# 数据收集主流程
# ---------------------------------------------------------------------------
def collect_all(
    dataset_name: str,
    tool_ids: Optional[List[str]] = None,
    n_repeats: int = 3,
    max_questions: Optional[int] = None,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """
    完整数据收集流程。

    Args:
        dataset_name: 数据集名称（对应 qa_pairs/{dataset_name}.json）
        tool_ids: 要测试的工具 ID，默认全部
        n_repeats: 每个 (Q, tool) 重复次数
        max_questions: 最多收集多少个问题（None=全部）
        dry_run: True=只打印不调用 API
    """
    if tool_ids is None:
        tool_ids = list(TOOL_DEFS.keys())

    print(f"\n[Data Collection] Knowledge Q&A — {dataset_name}")
    print(f"  Tools: {len(tool_ids)}")
    print(f"  Repeats per (Q, tool): {n_repeats}")
    print(f"  Dry run: {dry_run}")

    # 加载 Q&A 数据
    qa_items = load_qa_dataset(dataset_name)
    if max_questions:
        qa_items = qa_items[:max_questions]
    print(f"  Q&A items: {len(qa_items)}")

    all_records = []
    total_runs = len(qa_items) * len(tool_ids) * n_repeats
    print(f"  Total runs: {total_runs}  (estimated cost: see below)")

    # 预估成本
    estimated_cost = 0.0
    for tool_id in tool_ids:
        for _ in range(len(qa_items) * n_repeats):
            estimated_cost += calc_cost(tool_id)
    print(f"  Estimated total cost: ~${estimated_cost:.2f}")

    # 收集数据
    for qi, qa_item in enumerate(qa_items):
        if dry_run:
            print(f"  [{qi+1}/{len(qa_items)}] DRY RUN: {qa_item.get('question','')[:60]}...")
            continue

        records = collect_for_question(qa_item, tool_ids, n_repeats)
        all_records.extend(records)

        if (qi + 1) % 5 == 0:
            print(f"  Progress: {qi+1}/{len(qa_items)} questions done  "
                  f"(~{len(all_records)} records)")

    print(f"\n[Done] Collected {len(all_records)} records")
    return all_records


def save_records(records: List[Dict[str, Any]], dataset_name: str):
    """保存为 .jsonl 文件。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"{dataset_name}_collected_{timestamp}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# 生成 executor_profiles.jsonl（与 pareto_demo 格式一致）
# ---------------------------------------------------------------------------
def build_profiles(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    将收集的 records 聚合成 profile entries。
    格式与 outputs/pareto_demo/data/executor_profiles.jsonl 一致。
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for r in records:
        if r.get("source") == "real_collected":
            key = (r["tool_id"], r.get("difficulty", "medium"), r["node_type"])
            groups[key].append(r)

    profiles = []
    for (tool_id, diff, node_type), recs in sorted(groups.items()):
        qs = [r["quality"] for r in recs]
        cs = [r["c_total"] for r in recs]
        ls = [r["observed_latency"] for r in recs]
        n = len(qs)
        q_mean = sum(qs) / n
        c_mean = sum(cs) / n
        l_mean = sum(ls) / n
        q_std = math.sqrt(sum((q - q_mean)**2 for q in qs) / max(1, n - 1)) if n > 1 else 0.0

        profiles.append({
            "tool_id": tool_id,
            "task_type": "knowledge_qa",
            "node_type": node_type,
            "difficulty": diff,
            "quality_mean": round(q_mean, 4),
            "quality_std": round(q_std, 4),
            "latency_mean": round(l_mean / 1000, 3),  # ms → s
            "api_cost_mean": round(c_mean, 6),
            "human_cost_mean": 0.0,
            "sample_count": n,
            "cost_a1_per_mtok": 0.0,
            "cost_a2_per_mtok": 0.0,
            "typical_input_tokens": 0,
            "typical_output_tokens": 0,
            "latency_a1": 0.0,
            "latency_a2": 0.0,
        })

    return profiles


def save_profiles(profiles: List[Dict], dataset_name: str):
    """保存 profiles 到 executor_profiles.jsonl。"""
    out_path = DATA_DIR / dataset_name / "executor_profiles.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in profiles:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"  Saved profiles: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Collect real Q&A data for TopoGuard profiles")
    parser.add_argument("--dataset",  type=str, default="hydrology_qa",
                        help="Dataset name (looks in qa_pairs/{name}.json)")
    parser.add_argument("--tools",   type=str, default="",
                        help="Comma-separated tool IDs to test (default: all)")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Repeats per (question, tool)")
    parser.add_argument("--max_q",   type=int, default=None,
                        help="Max questions to collect (default: all)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print plan without calling LLM API")
    parser.add_argument("--output",  type=str, default="",
                        help="Output .jsonl path (default: auto)")
    args = parser.parse_args()

    tool_ids = None
    if args.tools:
        tool_ids = [t.strip() for t in args.tools.split(",")]

    records = collect_all(
        dataset_name=args.dataset,
        tool_ids=tool_ids,
        n_repeats=args.repeats,
        max_questions=args.max_q,
        dry_run=args.dry_run,
    )

    if not args.dry_run and records:
        save_records(records, args.dataset)
        profiles = build_profiles(records)
        save_profiles(profiles, args.dataset)
        print(f"\n[Next Step] Run TopoGuard Pareto analysis:")
        print(f"  python src/experiments/experiment_pareto_qa.py --dataset {args.dataset}")


if __name__ == "__main__":
    main()
