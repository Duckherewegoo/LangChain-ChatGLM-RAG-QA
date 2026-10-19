"""检索策略层（Advanced RAG）统一入口。

拆分为多个子模块，通过本文件对外暴露统一接口，保持向后兼容。
"""

from .factory import create_fuser, create_query_rewriter
from .fusers import DeduplicatingFuser, ReciprocalRankFusion, ResultFuser, ScoreWeightedFuser
from .hybrid import HybridRetrieval, RetrievalContext
from .retrievers import BM25Index, InMemoryBM25Index, KeywordRetrievalStrategy, RetrievalStrategy, VectorRetrievalStrategy
from .rewriters import ExpansionRewriter, IdentityRewriter, LLMQueryRewriter, QueryRewriter
from .types import Candidate, QueryRewrite

__all__ = [
    "BM25Index",
    "Candidate",
    "DeduplicatingFuser",
    "ExpansionRewriter",
    "HybridRetrieval",
    "IdentityRewriter",
    "InMemoryBM25Index",
    "KeywordRetrievalStrategy",
    "LLMQueryRewriter",
    "QueryRewrite",
    "QueryRewriter",
    "ReciprocalRankFusion",
    "ResultFuser",
    "RetrievalContext",
    "RetrievalStrategy",
    "ScoreWeightedFuser",
    "VectorRetrievalStrategy",
    "create_fuser",
    "create_query_rewriter",
]
