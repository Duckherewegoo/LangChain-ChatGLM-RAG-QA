"""检索策略层：查询改写器。"""

from __future__ import annotations

import abc
import json
import typing

from model.protocol import ChatModel, ChatRequest
from observability import get_logger

from .types import QueryRewrite

logger = get_logger(__name__)


class QueryRewriter(abc.ABC):
    """查询改写策略的抽象基类。"""
    name: str = "abstract"

    @abc.abstractmethod
    def rewrite(self, question: str, *, history: list[dict[str, str]] | None = None) -> QueryRewrite:
        ...


class IdentityRewriter(QueryRewriter):
    """原样返回。作为默认实现，确保关闭改写时行为不变。"""
    name = "identity"

    def rewrite(self, question: str, *, history: list[dict[str, str]] | None = None) -> QueryRewrite:
        return QueryRewrite(original=question, rewritten=question, queries=[], method="identity")


class ExpansionRewriter(QueryRewriter):
    """基于规则的轻量改写：拆并列、拆疑问词、提取实体式短语。"""
    name = "expansion"

    def __init__(self, max_expansions: int = 2, *, lowercase: bool = True) -> None:
        self._max = max(1, int(max_expansions))

    def rewrite(self, question: str, *, history: list[dict[str, str]] | None = None) -> QueryRewrite:
        base = question.strip()
        if not base:
            return QueryRewrite(original=base, rewritten=base, method="identity")

        expansions: list[str] = []
        for sep in ("和", "以及", "及", "与", "、", "；", ";"):
            if sep in base:
                expansions.extend(part.strip("？?。.!！ \t") for part in base.split(sep))
                break

        stripped = base
        for prefix in ("请问", "如何", "怎么", "怎样", "什么是", "请说明"):
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix):]
                break
        if stripped and stripped != base:
            expansions.append(stripped)

        deduped: list[str] = []
        seen = {base}
        for candidate in expansions:
            candidate = candidate.strip("？?。.!！ \t")
            if candidate and candidate not in seen and len(deduped) < self._max:
                seen.add(candidate)
                deduped.append(candidate)

        return QueryRewrite(
            original=base,
            rewritten=deduped[0] if deduped else base,
            queries=deduped,
            method="expansion",
        )


class LLMQueryRewriter(QueryRewriter):
    """用 LLM 做多查询改写（Multi-Query Retrieval）。"""
    name = "llm"

    def __init__(self, model: ChatModel, *, max_queries: int = 3) -> None:
        if model is None:
            raise ValueError("LLMQueryRewriter 需要一个 ChatModel 实例。")
        self._model = model
        self._max = max(1, int(max_queries))
        self._fallback = ExpansionRewriter(max_expansions=max_queries)

    def rewrite(self, question: str, *, history: list[dict[str, str]] | None = None) -> QueryRewrite:
        base = question.strip()
        if not base:
            return QueryRewrite(original=base, rewritten=base, method="identity")

        prompt = (
            "你是一个检索查询优化器。把用户问题改写成若干独立的检索查询，"
            "每个查询对应一个独立的信息需求，措辞贴近企业内部文档。\n"
            "只输出 JSON：{\"queries\": [\"查询1\", \"查询2\"]}，不要解释。\n\n"
            f"用户问题：{base}"
        )
        try:
            # 直接使用统一的 ChatRequest 协议
            response = self._model.invoke(
                ChatRequest(messages=[{"role": "user", "content": prompt}], tools=[])
            )
        except Exception as exc:
            logger.warning(f"LLM 查询改写失败，降级为规则改写 | error={exc!s}")
            return self._fallback.rewrite(base)

        parsed = _extract_json_list(response.content)
        queries = [str(item).strip() for item in parsed if str(item).strip() and item != base][: self._max]
        if not queries:
            return self._fallback.rewrite(base)
        return QueryRewrite(original=base, rewritten=queries[0], queries=queries, method="llm")


def _extract_json_list(text: str) -> list[typing.Any]:
    """从模型输出里尽量抠出一个 JSON 数组。"""
    if not isinstance(text, str):
        return []
    text = text.strip()
    if text.startswith("```"):
        text = text[3:]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return []
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return []

    return data.get("queries") if isinstance(data, dict) else data
