"""检索策略层：单路检索策略与 BM25 实现。"""

from __future__ import annotations

import abc
import math
import typing

from rag.types import RetrievedDocument

from .types import Candidate

if typing.TYPE_CHECKING:
    from rag.retrieval import RetrievalPipeline


class RetrievalStrategy(abc.ABC):
    """单一检索通道的抽象。"""
    name: str = "abstract"

    @abc.abstractmethod
    def retrieve(
        self, query: str, *, tenant_id: str, top_k: int,
        allowed_sources: typing.Sequence[str] | None = None, owner_id: str | None = None
    ) -> list[Candidate]:
        ...


class VectorRetrievalStrategy(RetrievalStrategy):
    """把请求委托给已有的 RetrievalPipeline，复用其授权与重排。"""
    name = "vector"

    def __init__(self, pipeline: RetrievalPipeline) -> None:
        self._pipeline = pipeline

    def retrieve(
        self, query: str, *, tenant_id: str, top_k: int,
        allowed_sources: typing.Sequence[str] | None = None, owner_id: str | None = None
    ) -> list[Candidate]:
        context = self._pipeline.retrieve(
            query, tenant_id=tenant_id, top_k=top_k,
            allowed_sources=list(allowed_sources) if allowed_sources else None,
            owner_id=owner_id,
        )
        return [
            Candidate(document=doc, score=doc.score, source="vector", rank=rank)
            for rank, doc in enumerate(context.documents)
        ]


class KeywordRetrievalStrategy(RetrievalStrategy):
    """BM25 式关键词检索。"""
    name = "keyword"

    def __init__(self, index: "BM25Index", *, tokenizer: typing.Callable[[str], list[str]] | None = None) -> None:
        self._index = index
        self._tokenize = tokenizer or _default_tokenize

    def retrieve(
        self, query: str, *, tenant_id: str, top_k: int,
        allowed_sources: typing.Sequence[str] | None = None, owner_id: str | None = None
    ) -> list[Candidate]:
        tokens = self._tokenize(query)
        results = self._index.search(
            tokens, tenant_id=tenant_id, owner_id=owner_id,
            allowed_sources=list(allowed_sources) if allowed_sources else None, top_k=top_k
        )
        maximum = max((score for _, score in results), default=1.0) or 1.0
        return [
            Candidate(document=document, score=score / maximum, source="keyword", rank=rank)
            for rank, (document, score) in enumerate(results)
        ]


class BM25Index(abc.ABC):
    """关键词索引的最小契约。"""
    @abc.abstractmethod
    def search(
        self, tokens: list[str], *, tenant_id: str, owner_id: str | None,
        allowed_sources: list[str] | None, top_k: int,
    ) -> list[tuple[RetrievedDocument, float]]:
        ...


class InMemoryBM25Index(BM25Index):
    """可注入打分器与词频提取器的内存索引。"""

    def __init__(
        self, *, tokenizer: typing.Callable[[str], list[str]] | None = None,
        scorer: typing.Callable[[list[str], list[str], dict[str, float]], float] | None = None,
    ) -> None:
        self._documents: list[RetrievedDocument] = []
        self._tokenize = tokenizer or _default_tokenize
        self._scorer = scorer or _tfidf_score
        self._doc_freq: dict[str, int] = {}
        self._total = 0
        self._cached_idf: dict[str, float] = {}
        self._cache_dirty = True

    def add(self, documents: typing.Iterable[RetrievedDocument]) -> None:
        for document in documents:
            self._documents.append(document)
            tokens = self._tokenize(document.content)
            for token in set(tokens):
                self._doc_freq[token] = self._doc_freq.get(token, 0) + 1
            self._total += 1
        self._cache_dirty = True

    def __len__(self) -> int:
        return len(self._documents)

    def search(
        self, tokens: list[str], *, tenant_id: str, owner_id: str | None,
        allowed_sources: list[str] | None, top_k: int,
    ) -> list[tuple[RetrievedDocument, float]]:
        if not tokens or not self._documents:
            return []

        if self._cache_dirty:
            self._cached_idf = _idf_normalizer(self._doc_freq, self._total)
            self._cache_dirty = False

        results: list[tuple[RetrievedDocument, float]] = []
        for document in self._documents:
            if not _matches_scope(document, tenant_id, owner_id, allowed_sources):
                continue
            score = self._scorer(tokens, self._tokenize(document.content), self._cached_idf)
            if score > 0:
                results.append((document, score))

        results.sort(key=lambda pair: pair[1], reverse=True)
        return results[: max(1, int(top_k))]


def _matches_scope(
    document: RetrievedDocument,
    tenant_id: str,
    owner_id: str | None,
    allowed_sources: list[str] | None,
) -> bool:
    """判断文档是否在当前检索主体的可见范围内。

    三条独立红线，任何一条不满足即拒绝：
      1. 租户隔离：document.tenant_id 非空时必须精确匹配请求的 tenant_id。
      2. 归属隔离：请求指定 owner_id 时，document.owner_id 必须严格匹配；
         None 不被视为通配。
      3. 来源白名单：allowed_sources 非空时，document.source 必须命中。

    约束与 RetrievalPipeline._is_authorized 保持一致，避免向量路径和
    关键词路径在同一请求下返回不同的可见范围。
    """
    if document.tenant_id and document.tenant_id != tenant_id:
        return False
    if owner_id is not None:
        if getattr(document, "owner_id", None) != owner_id:
            return False
    if allowed_sources and document.source not in allowed_sources:
        return False
    return True


def _default_tokenize(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    out: list[str] = []
    buffer = ""
    for char in text.lower():
        if "一" <= char <= "鿿":
            if buffer:
                out.append(buffer)
                buffer = ""
            out.append(char)
        elif char.isalnum():
            buffer += char
        elif buffer:
            out.append(buffer)
            buffer = ""
    if buffer:
        out.append(buffer)
    return out


def _idf_normalizer(doc_freq: dict[str, int], total: int) -> dict[str, float]:
    if total <= 0:
        return {}
    return {term: math.log(1 + total / (freq or 1)) for term, freq in doc_freq.items()}


def _tfidf_score(query: list[str], document: list[str], idf: dict[str, float]) -> float:
    if not document or not query:
        return 0.0
    doc_len = math.sqrt(len(document))
    term_freq: dict[str, int] = {}
    for token in document:
        term_freq[token] = term_freq.get(token, 0) + 1
    score = 0.0
    for token in query:
        if token in term_freq:
            score += term_freq[token] * idf.get(token, 0.0)
    return score / doc_len if doc_len else 0.0
