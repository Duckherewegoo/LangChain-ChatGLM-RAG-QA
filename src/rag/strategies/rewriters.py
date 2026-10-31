"""检索策略层：查询改写器。"""

from __future__ import annotations

import abc
import json
import typing

from model.protocol import ChatModel, ChatRequest
from base.observability import get_logger

from .types import QueryRewrite

logger = get_logger(__name__)


class QueryRewriter(abc.ABC):
    """查询改写策略的抽象基类。

    history 参数预留给未来多轮指代改写（如"它的配置呢"）。当前所有实现
    均忽略 history；LLMQueryRewriter 的 prompt 未注入历史对话。
    """
    name: str = "abstract"

    @abc.abstractmethod
    def rewrite(self, question: str, *, history: list[dict[str, str]] | None = None) -> QueryRewrite:
        ...


class IdentityRewriter(QueryRewriter):
    """原样返回。作为默认实现，确保关闭改写时行为不变。"""
    name = "identity"

    def rewrite(self, question: str, *, history: list[dict[str, str]] | None = None) -> QueryRewrite:
        return QueryRewrite(original=question, rewritten=question, queries=[], method="identity")


# 句内并列的分隔符；按顺序匹配，第一个命中即拆。
_CLAUSE_SEPARATORS = ("和", "以及", "及", "与", "、", "；", ";")
# 可剥离的疑问 / 请求前缀；循环剥离，支持复合前缀（如"请问如何"）。
_QUESTION_PREFIXES = ("请问", "如何", "怎么", "怎样", "什么是", "请说明")
# 首尾剥除的标点与空白。
_STRIP_CHARS = "？?。.!！ \t"


class ExpansionRewriter(QueryRewriter):
    """基于规则的轻量改写：拆并列、剥疑问前缀。

    无分隔符且无可剥前缀时，行为与 IdentityRewriter 等价，
    返回 method="identity"（而非误导性的 "expansion"）。
    """
    name = "expansion"

    def __init__(self, max_expansions: int = 2) -> None:
        self._max = max(1, int(max_expansions))

    def rewrite(self, question: str, *, history: list[dict[str, str]] | None = None) -> QueryRewrite:
        base = question.strip()
        if not base:
            return QueryRewrite(original=base, rewritten=base, method="identity")

        expansions: list[str] = []
        for sep in _CLAUSE_SEPARATORS:
            if sep in base:
                expansions.extend(part.strip(_STRIP_CHARS) for part in base.split(sep))
                break

        # 循环剥离前缀，支持复合前缀（如"请问如何配置"→"配置"）。
        stripped = base
        while True:
            matched_prefix = next(
                (p for p in _QUESTION_PREFIXES if stripped.startswith(p)), None
            )
            if matched_prefix is None:
                break
            stripped = stripped[len(matched_prefix):].lstrip()
        if stripped and stripped != base:
            expansions.append(stripped)

        deduped: list[str] = []
        seen = {base}
        for candidate in expansions:
            candidate = candidate.strip(_STRIP_CHARS)
            if candidate and candidate not in seen and len(deduped) < self._max:
                seen.add(candidate)
                deduped.append(candidate)

        if not deduped:
            # 空扩展：行为等同 Identity，标签也随之改为 identity，
            # 避免调用方误以为发生过改写。
            return QueryRewrite(original=base, rewritten=base, method="identity")
        return QueryRewrite(
            original=base,
            rewritten=deduped[0],
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
            logger.warning(
                "LLM 查询改写失败，降级为规则改写 | error=%s", exc
            )
            return self._fallback.rewrite(base)

        parsed = _extract_json_list(response.content)
        queries = [
            str(item).strip()
            for item in parsed
            if str(item).strip() and str(item).strip() != base
        ][: self._max]
        if not queries:
            return self._fallback.rewrite(base)
        return QueryRewrite(original=base, rewritten=queries[0], queries=queries, method="llm")


def _extract_json_list(text: str) -> list[typing.Any]:
    """从模型输出里尽量抠出一个 JSON 数组。

    返回值保证是 list（可能是空），调用方可直接 for 迭代：
      - {"queries": [...]} → 返回 queries 列表；
      - {...} 但无 queries 键 → 返回 []；
      - 裸数组 [...] → 返回该数组；
      - 解析失败 → 返回 []。
    """
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

    if isinstance(data, dict):
        return data.get("queries") or []
    if isinstance(data, list):
        return data
    return []
