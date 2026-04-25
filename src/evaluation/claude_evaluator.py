"""
claude_evaluator.py
===================
真实 LLM-as-judge 评估器，基于 Anthropic Claude API。

实现 BaseEvaluator 接口，可直接替换 MockLLMEvaluator，无需修改主流程。

两档模型：
  - 标准档（默认）：claude-haiku-4-5-20251001  — 低成本，用于常规评估
  - 强档：          claude-sonnet-4-6           — 高质量，用于 Strategy C 评估器升级

用法：
    from src.evaluation.claude_evaluator import ClaudeEvaluator
    evaluator = ClaudeEvaluator(api_key="sk-ant-...")
    result = evaluator.evaluate(inp)

Strategy C 升级：
    strong_eval = ClaudeEvaluator(api_key=..., use_strong_model=True)
    result = strong_eval.evaluate(inp)
"""

from __future__ import annotations

import json
import os
import time
from typing import List, Optional

from src.evaluation.evaluator_types import (
    BaseEvaluator,
    EvaluatorInput,
    EvaluatorOutput,
    EvalLevel,
    NODE_TYPE_RUBRIC,
)

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

_HAIKU_MODEL  = "claude-haiku-4-5-20251001"
_SONNET_MODEL = "claude-sonnet-4-6"

# Approximate cost per 1M tokens (USD) — used for api_cost estimation
_COST_PER_MTOK = {
    _HAIKU_MODEL:  {"input": 0.80,  "output": 4.00},
    _SONNET_MODEL: {"input": 3.00,  "output": 15.00},
}

# Typical token counts for a judge prompt
_JUDGE_INPUT_TOKENS  = 600
_JUDGE_OUTPUT_TOKENS = 120


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_judge_prompt(inp: EvaluatorInput) -> str:
    rubric = NODE_TYPE_RUBRIC.get(inp.node_type, NODE_TYPE_RUBRIC.get("reasoning", {}))
    dims = rubric.get("dimensions", [("quality", 1.0, "judge")])
    threshold = rubric.get("pass_threshold_score", 0.60)

    dim_lines = "\n".join(
        f"  - {name} (weight={w:.0%})" for name, w, _ in dims
    )

    ref_section = ""
    if inp.reference_output is not None:
        ref_section = f"\n参考输出（ground truth）：\n{inp.reference_output}\n"

    context_section = ""
    if inp.context:
        ctx_str = json.dumps(inp.context, ensure_ascii=False, indent=2)[:800]
        context_section = f"\n前序节点上下文（摘要）：\n{ctx_str}\n"

    return f"""你是一个专业的 AI 输出质量评估员。请对以下候选输出进行评分。

## 任务信息
- 节点类型：{inp.node_type}
- 任务域：{inp.task_type}
- 难度：{inp.difficulty_bucket}
- 节点 ID：{inp.node_id}

## 评估维度
{dim_lines}

## 候选输出
{inp.candidate_output}
{ref_section}{context_section}
## 评分要求
请综合以上维度，给出一个 0-10 的整体质量分数（10 = 完美，0 = 完全错误）。
通过阈值为 {threshold * 10:.1f} 分（对应归一化分数 {threshold:.2f}）。

请严格按以下 JSON 格式返回，不要输出任何其他内容：
{{"score": <0-10的数字>, "passed": <true或false>, "confidence": <0-1的置信度>, "error_type": <"low_quality"|"format_error"|"inconsistent_output"|"insufficient_evidence"|"unsafe_decision"|null>, "reason": "<一句话说明>"}}"""


# ---------------------------------------------------------------------------
# ClaudeEvaluator
# ---------------------------------------------------------------------------

class ClaudeEvaluator(BaseEvaluator):
    """
    基于 Claude API 的真实 LLM-as-judge 评估器。

    Parameters
    ----------
    api_key : str, optional
        Anthropic API key。若为 None，从环境变量 ANTHROPIC_API_KEY 读取。
    use_strong_model : bool
        False（默认）= claude-haiku（低成本）；True = claude-sonnet（Strategy C 升级用）。
    pass_threshold : float
        质量分数通过阈值，默认 0.60。
    max_retries : int
        API 调用失败时的最大重试次数，默认 2。
    fallback_score : float
        API 完全失败时的 fallback 质量分数，默认 0.50（不通过）。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        use_strong_model: bool = False,
        pass_threshold: float = 0.60,
        max_retries: int = 2,
        fallback_score: float = 0.50,
    ):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._model = _SONNET_MODEL if use_strong_model else _HAIKU_MODEL
        self._pass_threshold = pass_threshold
        self._max_retries = max_retries
        self._fallback_score = fallback_score
        self._client = None  # lazy init

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self._api_key)
            except ImportError as e:
                raise ImportError(
                    "anthropic SDK not installed. Run: pip install anthropic"
                ) from e
        return self._client

    # ------------------------------------------------------------------
    # BaseEvaluator properties
    # ------------------------------------------------------------------

    @property
    def evaluator_id(self) -> str:
        suffix = "strong" if self._model == _SONNET_MODEL else "standard"
        return f"claude_judge_{suffix}"

    @property
    def name(self) -> str:
        return f"ClaudeEvaluator({self._model})"

    @property
    def supported_node_types(self) -> List[str]:
        return list(NODE_TYPE_RUBRIC.keys()) + [
            "retrieval", "reasoning", "computation", "verification", "aggregation"
        ]

    @property
    def supported_task_types(self) -> List[str]:
        return ["time_series", "text_analysis", "tabular_analysis", "multimodal",
                "water_qa", "storm_surge", "knowledge_qa"]

    @property
    def latency_mean(self) -> float:
        return 3.5 if self._model == _HAIKU_MODEL else 8.0

    @property
    def api_cost_mean(self) -> float:
        costs = _COST_PER_MTOK[self._model]
        return (
            _JUDGE_INPUT_TOKENS  / 1_000_000 * costs["input"] +
            _JUDGE_OUTPUT_TOKENS / 1_000_000 * costs["output"]
        )

    # ------------------------------------------------------------------
    # Core evaluate()
    # ------------------------------------------------------------------

    def evaluate(self, inp: EvaluatorInput) -> EvaluatorOutput:
        """
        调用 Claude API 对候选输出打分，返回结构化 EvaluatorOutput。

        失败时（API 错误、JSON 解析失败）返回 fallback_score，不抛异常，
        保证主实验循环不中断。
        """
        prompt = _build_judge_prompt(inp)
        t0 = time.time()

        raw = self._call_with_retry(prompt)
        latency = time.time() - t0

        if raw is None:
            # API 完全失败，返回 fallback
            return EvaluatorOutput(
                evaluator_id=self.evaluator_id,
                quality_score=self._fallback_score,
                passed=self._fallback_score >= self._pass_threshold,
                error_type="unknown",
                confidence=0.0,
                latency=latency,
                api_cost=self.api_cost_mean,
                metadata={"fallback": True, "model": self._model},
            )

        parsed = self._parse_response(raw)
        score_norm = parsed["score"] / 10.0
        rubric = NODE_TYPE_RUBRIC.get(inp.node_type, {})
        threshold = rubric.get("pass_threshold_score", self._pass_threshold)

        return EvaluatorOutput(
            evaluator_id=self.evaluator_id,
            quality_score=round(score_norm, 4),
            passed=score_norm >= threshold,
            error_type=parsed.get("error_type"),
            confidence=float(parsed.get("confidence", 0.8)),
            latency=round(latency, 3),
            api_cost=round(self.api_cost_mean, 6),
            metadata={
                "model": self._model,
                "raw_score": parsed["score"],
                "reason": parsed.get("reason", ""),
                "node_type": inp.node_type,
                "difficulty_bucket": inp.difficulty_bucket,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_with_retry(self, prompt: str) -> Optional[str]:
        client = self._get_client()
        for attempt in range(self._max_retries + 1):
            try:
                response = client.messages.create(
                    model=self._model,
                    max_tokens=256,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text
            except Exception as e:
                if attempt == self._max_retries:
                    return None
                time.sleep(1.5 * (attempt + 1))
        return None

    def _parse_response(self, raw: str) -> dict:
        """Parse JSON from Claude response; return safe defaults on failure."""
        try:
            # Claude sometimes wraps JSON in markdown code blocks
            text = raw.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(
                    l for l in lines if not l.startswith("```")
                ).strip()
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            # Try to extract score with a simple heuristic
            import re
            m = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', raw)
            score = float(m.group(1)) if m else 5.0
            return {"score": score, "passed": score >= 6.0, "confidence": 0.5}
