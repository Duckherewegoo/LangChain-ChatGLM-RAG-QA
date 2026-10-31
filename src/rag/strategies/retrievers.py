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
        # 优先使用 pipeline 重排后的 rerank_score；未重排时回退到原始向量分数。
        # 若丢弃 rerank_score，fuser 会看到"未重排的分数"配上"已重排的顺序"，
        # 造成混合检索里排序与分数不匹配，并使 pipeline 里跑过的 cross-encoder
        # 结果被丢弃。
        candidates: list[Candidate] = []
        for rank, doc in enumerate(context.documents):
            score = doc.rerank_score if doc.rerank_score is not None else doc.score
            candidates.append(
                Candidate(document=doc, score=score, source="vector", rank=rank)
            )
        return candidates


class KeywordRetrievalStrategy(RetrievalStrategy):
    """关键词检索：查询 BM25Index 并按最大值归一化分数。"""
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
        # 用 max 归一化把分数映射到 [0, 1]，让第一名恒为 1.0。
        # 注意：这不是加权融合所需的量纲对齐——如果 fuser 用加权求和，
        # keyword 分支会系统性偏高分。加权融合应改用 min-max 或 z-score；
        # RRF（按 rank）不受此影响。
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

    @abc.abstractmethod
    def add(self, documents: typing.Iterable[RetrievedDocument]) -> None:
        ...

    @abc.abstractmethod
    def remove_by_document(self, document_id: str, tenant_id: str | None = None) -> int:
        """删除匹配 document_id（可选 tenant_id）的所有 chunk，返回删除数。"""
        ...


class InMemoryBM25Index(BM25Index):
    """内存关键词索引。

    实现说明：打分函数是 TF-IDF 近似（tf * idf / sqrt(doc_len)），
    不是标准 BM25（缺少 k1/b 参数与 avgdl 归一化）。类名保留 BM25
    以兼容现有引用；如需严格 BM25，可注入自定义 scorer。

    幂等性：add 按 chunk_id 去重，重复摄入不会重复累加。
    删除：remove_by_document 按 document_id 移除，与 vector_store 的
    delete_by_document 对齐。
    """

    def __init__(
        self, *, tokenizer: typing.Callable[[str], list[str]] | None = None,
        scorer: typing.Callable[[list[str], list[str], dict[str, float]], float] | None = None,
    ) -> None:
        self._documents: list[RetrievedDocument] = []
        self._chunk_ids: set[str] = set()
        self._tokenize = tokenizer or _default_tokenize
        self._scorer = scorer or _tfidf_score
        self._doc_freq: dict[str, int] = {}
        self._total = 0
        self._cached_idf: dict[str, float] = {}
        self._cache_dirty = True

    def add(self, documents: typing.Iterable[RetrievedDocument]) -> None:
        for document in documents:
            if document.chunk_id in self._chunk_ids:
                continue  # 幂等：已存在的 chunk 跳过
            self._documents.append(document)
            self._chunk_ids.add(document.chunk_id)
            tokens = self._tokenize(document.content)
            for token in set(tokens):
                self._doc_freq[token] = self._doc_freq.get(token, 0) + 1
            self._total += 1
        self._cache_dirty = True

    def remove_by_document(self, document_id: str, tenant_id: str | None = None) -> int:
        """删除匹配 document_id（可选 tenant_id）的所有 chunk，返回删除数。"""
        remaining: list[RetrievedDocument] = []
        removed = 0
        for document in self._documents:
            matches = document.document_id == document_id and (
                tenant_id is None or document.tenant_id == tenant_id
            )
            if not matches:
                remaining.append(document)
                continue
            for token in set(self._tokenize(document.content)):
                freq = self._doc_freq.get(token, 0) - 1
                if freq <= 0:
                    self._doc_freq.pop(token, None)
                else:
                    self._doc_freq[token] = freq
            self._chunk_ids.discard(document.chunk_id)
            self._total -= 1
            removed += 1
        if removed:
            self._documents = remaining
            self._cache_dirty = True
        return removed

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
      1. 租户隔离：document.tenant_id 必须精确匹配请求的 tenant_id
         （空值不视为通配，避免跨租户泄露）。
      2. 归属隔离：请求指定 owner_id 时，document.owner_id 必须严格匹配；
         None 不被视为通配。
      3. 来源白名单：allowed_sources 非空时，document.source 必须命中。

    与 RetrievalPipeline._is_authorized 覆盖同一可见范围，但各自负责本层
    能做的过滤（pipeline 依赖 store 侧做 tenant/sources，本层在内存全量过滤）。
    """
    if document.tenant_id != tenant_id:
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
