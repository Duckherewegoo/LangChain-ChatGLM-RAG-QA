"""检索策略层：工厂函数。"""

from __future__ import annotations

import typing

from model.protocol import ChatModel
from observability import get_logger

from .fusers import DeduplicatingFuser, ReciprocalRankFusion, ResultFuser, ScoreWeightedFuser
from .rewriters import ExpansionRewriter, IdentityRewriter, LLMQueryRewriter, QueryRewriter

logger = get_logger(__name__)


def create_query_rewriter(config: dict[str, typing.Any] | None, *, model: ChatModel | None = None) -> QueryRewriter:
    """构造查询改写器；配置不合法或无 LLM 时退化为规则改写。"""
    config = dict(config or {})
    method = str(config.get("method", "expansion")).strip().lower()
    max_queries = int(config.get("max_queries", 3))

    if method == "llm":
        if model is None:
            logger.warning("未提供模型，查询改写降级为规则扩展。")
            return ExpansionRewriter(max_queries)
        try:
            return LLMQueryRewriter(model, max_queries=max_queries)
        except Exception as exc:
            logger.warning(f"LLM 查询改写初始化失败，降级为规则扩展 | error={exc!s}")
            return ExpansionRewriter(max_queries)
    if method == "identity":
        return IdentityRewriter()
    return ExpansionRewriter(max_queries)


def create_fuser(config: dict[str, typing.Any] | None) -> ResultFuser:
    """构造结果融合器。"""
    config = dict(config or {})
    kind = str(config.get("type", "rrf")).strip().lower()
    if kind == "rrf":
        return ReciprocalRankFusion(k=int(config.get("k", 60)))
    if kind == "weighted":
        return ScoreWeightedFuser(weights=dict(config.get("weights", {})))
    if kind == "dedupe":
        return DeduplicatingFuser()
    raise ValueError(f"未知的结果融合器类型: {kind!r}.")
