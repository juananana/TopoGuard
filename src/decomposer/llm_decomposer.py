"""
llm_decomposer.py
=================
基于 Claude API 的真实任务分解器。

实现与 TaskDecomposer 相同的接口（decompose() → List[SubTaskSpec]），
可直接替换关键词规则分解器，无需修改主流程。

降级策略：Claude API 调用失败时自动 fallback 到关键词规则分解器，
保证主实验循环不中断。

用法：
    from src.decomposer.llm_decomposer import LLMTaskDecomposer
    decomposer = LLMTaskDecomposer(api_key="sk-ant-...")
    subtasks = decomposer.decompose("预测未来24小时的洪水风险等级")
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import List, Optional

from src.decomposer.task_decomposer import (
    SubTaskSpec,
    ModalityType,
    TaskDecomposer,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HAIKU_MODEL = "claude-haiku-4-5-20251001"

_VALID_PRIMITIVES = ["forecast", "state_parse", "data_analysis", "retrieval",
                     "reasoning", "computation", "verification", "aggregation"]

_VALID_DIFFICULTIES = {"easy": 0.2, "medium": 0.5, "hard": 0.8, "extreme": 0.95}

_VALID_MODALITIES = {
    "text": ModalityType.TEXT,
    "time_series": ModalityType.TIME_SERIES,
    "tabular": ModalityType.TABULAR,
    "image": ModalityType.IMAGE,
    "multimodal": ModalityType.MULTIMODAL,
}

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_DECOMPOSE_PROMPT = """你是一个 AI 工作流任务分解专家。请将以下任务描述分解为一组原子子任务。

## 任务描述
{task_description}

## 可用原语类型（primitive_name）
- forecast: 时间序列预测、趋势分析
- state_parse: 结构化信息抽取、状态解析
- data_analysis: 数据分析、统计计算、表格处理
- retrieval: 知识检索、文档查询
- reasoning: 逻辑推理、因果分析
- computation: 数值计算、公式求解
- verification: 结果验证、一致性检查
- aggregation: 多路结果汇聚、投票

## 难度等级
- easy: 简单直接，单步可完成
- medium: 中等复杂，需要一定推理
- hard: 复杂任务，需要多步推理或专业知识
- extreme: 极难，需要深度专业知识

## 输入模态
text / time_series / tabular / image / multimodal

请返回严格的 JSON 数组，每个元素代表一个子任务：
[
  {{
    "sub_task_id": "t1",
    "primitive_name": "<从上面列表选>",
    "difficulty_bucket": "<easy|medium|hard|extreme>",
    "description": "<子任务描述>",
    "predecessor_ids": [],
    "input_modality": "<text|time_series|tabular|image|multimodal>"
  }}
]

规则：
1. 子任务数量 1-4 个，不要过度分解
2. predecessor_ids 用于表达依赖关系（如 t2 依赖 t1 则写 ["t1"]）
3. 只返回 JSON 数组，不要任何其他文字"""


# ---------------------------------------------------------------------------
# LLMTaskDecomposer
# ---------------------------------------------------------------------------

class LLMTaskDecomposer:
    """
    基于 Claude API 的任务分解器，与 TaskDecomposer 接口兼容。

    Parameters
    ----------
    api_key : str, optional
        Anthropic API key。若为 None，从环境变量 ANTHROPIC_API_KEY 读取。
    model : str
        使用的 Claude 模型，默认 claude-haiku（低成本）。
    fallback_to_rules : bool
        API 失败时是否 fallback 到关键词规则分解器，默认 True。
    max_retries : int
        API 调用失败时的最大重试次数，默认 1。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = _HAIKU_MODEL,
        fallback_to_rules: bool = True,
        max_retries: int = 1,
    ):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._model = model
        self._fallback_to_rules = fallback_to_rules
        self._max_retries = max_retries
        self._client = None
        self._rule_decomposer = TaskDecomposer() if fallback_to_rules else None

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

    def decompose(self, task_description: str, seed: Optional[int] = None) -> List[SubTaskSpec]:
        """
        将自然语言任务描述分解为 SubTaskSpec 列表。

        先尝试 Claude API；失败时 fallback 到关键词规则分解器。

        Parameters
        ----------
        task_description : str
            自然语言任务描述。
        seed : int, optional
            随机种子（传给 fallback 分解器）。

        Returns
        -------
        List[SubTaskSpec]
            子任务列表，至少包含一个元素。
        """
        raw = self._call_claude(task_description)
        if raw is not None:
            subtasks = self._parse_subtasks(raw, task_description)
            if subtasks:
                return subtasks

        # fallback
        if self._rule_decomposer is not None:
            return self._rule_decomposer.decompose(task_description)

        # last resort: single generic subtask
        return [SubTaskSpec(
            sub_task_id="t1",
            primitive_name="reasoning",
            difficulty=0.5,
            difficulty_bucket="medium",
            description=task_description[:200],
        )]

    def _call_claude(self, task_description: str) -> Optional[str]:
        prompt = _DECOMPOSE_PROMPT.format(task_description=task_description[:1000])
        client = self._get_client()
        for attempt in range(self._max_retries + 1):
            try:
                response = client.messages.create(
                    model=self._model,
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text
            except Exception:
                if attempt == self._max_retries:
                    return None
                time.sleep(1.0)
        return None

    def _parse_subtasks(self, raw: str, task_description: str) -> List[SubTaskSpec]:
        """Parse Claude JSON response into SubTaskSpec list."""
        try:
            text = raw.strip()
            # strip markdown code fences
            if text.startswith("```"):
                text = "\n".join(
                    l for l in text.split("\n") if not l.startswith("```")
                ).strip()
            data = json.loads(text)
            if not isinstance(data, list):
                return []

            subtasks = []
            for item in data[:4]:  # cap at 4
                primitive = item.get("primitive_name", "reasoning")
                if primitive not in _VALID_PRIMITIVES:
                    primitive = "reasoning"

                bucket = item.get("difficulty_bucket", "medium")
                if bucket not in _VALID_DIFFICULTIES:
                    bucket = "medium"
                difficulty = _VALID_DIFFICULTIES[bucket]

                modality_str = item.get("input_modality", "text")
                modality = _VALID_MODALITIES.get(modality_str, ModalityType.TEXT)

                subtasks.append(SubTaskSpec(
                    sub_task_id=item.get("sub_task_id", f"t{len(subtasks)+1}"),
                    primitive_name=primitive,
                    difficulty=difficulty,
                    difficulty_bucket=bucket,
                    description=item.get("description", task_description[:100]),
                    predecessor_ids=item.get("predecessor_ids", []),
                    input_modality=modality,
                ))
            return subtasks
        except (json.JSONDecodeError, KeyError, TypeError):
            return []


# ---------------------------------------------------------------------------
# AnthropicClient — thin wrapper kept for import compatibility
# ---------------------------------------------------------------------------

class AnthropicClient:
    """
    Thin wrapper around the Anthropic SDK client.
    Kept for import compatibility with decomposer/__init__.py.
    """

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._client = None

    def get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def complete(self, prompt: str, model: str = _HAIKU_MODEL,
                 max_tokens: int = 512) -> str:
        client = self.get_client()
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
