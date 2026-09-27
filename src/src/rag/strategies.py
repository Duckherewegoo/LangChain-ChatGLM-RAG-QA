"""检索策略层（Advanced RAG）。

本模块把"怎么拿到候选文档"这件事从 :class:`~src.rag.retrieval.RetrievalPipeline`
里抽出来，做成可插拔的策略对象。这一层回答四个问题：

1. :class:`QueryRewriter`      -- 要不要把用户的原问题改写成更适合检索的查询？
2. :class:`RetrievalStrategy`  -- 用哪种检索方式（向量 / BM25 / 混合）？
3. :class:`ResultFuser`        -- 多路召回的候选怎么合并、去重、定序？
4. :class:`HybridRetrieval`    -- 把上面三者编排起来，输出统一的 ``RetrievedContext``。

设计取舍
--------
* 所有策略只依赖接口，不依赖具体存储。这保证它们可以被独立测试，
  也允许在测试里用内存版的 fake store 替换 Chroma。
* 多路召回的融合采用可配置的 :class:`ResultFuser`。默认是倒数排名融合
  （RRF），它不要求各路分数在同一量纲上，比直接加权更稳。需要语义驱动的
  融合时换成 :class:`ScoreWeightedFuser`。
* 本层**不**做权限过滤——那是 :class:`RetrievalPipeline` 的职责，策略层
  只对"原始候选"建模，避免重复实现授权逻辑。
* 每一步都产生审计事件，方便在 AB 测试中对比不同策略组合的效果。
"""

from __future__ import annotations

import abc
import dataclasses
import logging
import math
import typing

from pydantic import BaseModel, Field

from src.rag.types import RetrievedContext, RetrievedDocument

if typing.TYPE_CHECKING:  # pragma: no cover - type checking only
    from src.model.manager import ChatModel
    from src.rag.retrieval import RetrievalPipeline

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Candidate:
    """一份召回候选，携带它来自哪一路检索与原始分数。"""

    document: RetrievedDocument
    score: float
    source: str
    rank: int


# ---------------------------------------------------------------------------
# 查询改写
# ---------------------------------------------------------------------------


class QueryRewrite(BaseModel):
    """一次查询改写的结果。"""

    original: str
    rewritten: str
    queries: list[str] = Field(default_factory=list)
    method: str = "identity"

    def all_queries(self) -> list[str]:
        """返回实际要执行的查询，原问题永远保留。"""

        seen = {self.original}
        result = [self.original]
        for query in self.queries:
            if query and query not in seen:
                seen.add(query)
                result.append(query)
        return result

    def changed(self) -> bool:
        return self.method != "identity"


class QueryRewriter(abc.ABC):
    """查询改写策略的抽象基类。"""

    name: str = "abstract"

    @abc.abstractmethod
    def rewrite(self, question: str, *, history: list[dict[str, str]] | None = None) -> QueryRewrite:
        """把用户问题（可附带历史）改写成一次或多次检索查询。"""


class IdentityRewriter(QueryRewriter):
    """原样返回。作为默认实现，确保关闭改写时行为不变。"""

    name = "identity"

    def rewrite(self, question: str, *, history: list[dict[str, str]] | None = None) -> QueryRewrite:
        return QueryRewrite(original=question, rewritten=question, queries=[], method="identity")


class ExpansionRewriter(QueryRewriter):
    """基于规则的轻量改写：拆并列、拆疑问词、提取实体式短语。

    零依赖、离线可用，适合演示与评测基线；不会调用 LLM，所以不存在
    改写环节失败导致整次检索崩掉的风险。如果配置里给了 LLM，会自动
    升级为 :class:`LLMQueryRewriter`。
    """

    name = "expansion"

    def __init__(
        self,
        max_expansions: int = 2,
        *,
        lowercase: bool = True,
    ) -> None:
        self._max = max(1, int(max_expansions))
        self._lowercase = lowercase

    def rewrite(self, question: str, *, history: list[dict[str, str]] | None = None) -> QueryRewrite:
        base = question.strip()
        if not base:
            return QueryRewrite(original=base, rewritten=base, method="identity")

        expansions: list[str] = []
        # 并列拆分：把"如何 A 和 B"拆成两条针对性查询，命中率通常高于原句
        for sep in ("和", "以及", "及", "与", "、", "；", ";"):
            if sep in base:
                expansions.extend(part.strip("？?。.!！ \t") for part in base.split(sep))
                break
        # 去掉常见疑问前缀，让剩余内容更像文档中的陈述句
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
    """用 LLM 做多查询改写（Multi-Query Retrieval）。

    LLM 不可用或输出不合法时**降级为规则改写**而非报错：改写是优化环节，
    不是正确性的必要条件。这遵循"宁可召回少一点，也不要让一次检索崩掉"。
    """

    name = "llm"

    def __init__(self, model: ChatModel, *, max_queries: int = 3) -> None:
        if model is None:
            raise ValueError("LLMQueryRewriter requires a ChatModel instance.")
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
            response = self._model.invoke(
                type(self._model)._request_class(
                    messages=[{"role": "user", "content": prompt}]
                )
                if hasattr(type(self._model), "_request_class")
                else __import__("src.rag.strategies").rag.strategies._simple_request(prompt)
            )
        except Exception as exc:
            logger.warning("query_rewrite_llm_failed", extra={"error": str(exc)})
            return self._fallback.rewrite(base)

        parsed = _extract_json_list(response.content)
        queries = [str(item).strip() for item in parsed if str(item).strip() and item != base][
            : self._max
        ]
        if not queries:
            return self._fallback.rewrite(base)
        return QueryRewrite(
            original=base, rewritten=queries[0], queries=queries, method="llm"
        )


def _simple_request(content: str):
    """构造一个仅含 user 消息的最小请求对象，避免反向依赖 model 模块。"""

    from pydantic import BaseModel as _BM

    class _Req(_BM):
        messages: list[dict[str, str]]
        tools: list[dict[str, typing.Any]] = []

    return _Req(messages=[{"role": "user", "content": content}])


def _extract_json_list(text: str) -> list[typing.Any]:
    """从模型输出里尽量抠出一个 JSON 数组，容错模型自带的代码块包裹。"""

    if not isinstance(text, str):
        return []
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        data = __import__("json").loads(text)
    except ValueError:
        # 兜底：只截取第一个 '[' 到最后一个 ']'，对引号/后缀都更宽容
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return []
        try:
            data = __import__("json").loads(text[start : end + 1])
        except ValueError:
            return []
    return data.get("queries") if isinstance(data, dict) else data


# ---------------------------------------------------------------------------
# 检索策略
# ---------------------------------------------------------------------------


class RetrievalStrategy(abc.ABC):
    """单一检索通道的抽象。可以是一条向量通道，也可以是一条关键词通道。"""

    name: str = "abstract"

    @abc.abstractmethod
    def retrieve(
        self,
        query: str,
        *,
        tenant_id: str,
        top_k: int,
        allowed_sources: typing.Sequence[str] | None = None,
        owner_id: str | None = None,
    ) -> list[Candidate]:
        """返回带来源标记与原始排名的候选。"""


class VectorRetrievalStrategy(RetrievalStrategy):
    """把请求委托给已有的 :class:`RetrievalPipeline`，复用其授权与重排。"""

    name = "vector"

    def __init__(self, pipeline: RetrievalPipeline) -> None:
        if pipeline is None:
            raise ValueError("VectorRetrievalStrategy requires a RetrievalPipeline.")
        self._pipeline = pipeline

    def retrieve(
        self,
        query: str,
        *,
        tenant_id: str,
        top_k: int,
        allowed_sources: typing.Sequence[str] | None = None,
        owner_id: str | None = None,
    ) -> list[Candidate]:
        context = self._pipeline.retrieve(
            query,
            tenant_id=tenant_id,
            top_k=top_k,
            allowed_sources=list(allowed_sources) if allowed_sources else None,
            owner_id=owner_id,
        )
        return [
            Candidate(document=doc, score=doc.score, source="vector", rank=rank)
            for rank, doc in enumerate(context.documents)
        ]


class KeywordRetrievalStrategy(RetrievalStrategy):
    """BM25 式关键词检索。

    实现说明：标准 BM25 需要一个倒排索引。这里提供一个可替换的抽象
    :class:`BM25Index` 与一个基于 :mod:`rank_bm25` 的实现；当该依赖不可
    用时退化为带词频加权的子串命中打分（token 级重叠），保证在纯净环境
    下仍可运行与测试。分数做了 max=1 的归一化，以便和向量通道的余弦
    分数进入同一融合流程。
    """

    name = "keyword"

    def __init__(
        self,
        index: BM25Index,
        *,
        tokenizer: typing.Callable[[str], list[str]] | None = None,
    ) -> None:
        self._index = index
        self._tokenize = tokenizer or _default_tokenize

    def retrieve(
        self,
        query: str,
        *,
        tenant_id: str,
        top_k: int,
        allowed_sources: typing.Sequence[str] | None = None,
        owner_id: str | None = None,
    ) -> list[Candidate]:
        tokens = self._tokenize(query)
        results = self._index.search(
            tokens,
            tenant_id=tenant_id,
            owner_id=owner_id,
            allowed_sources=list(allowed_sources) if allowed_sources else None,
            top_k=top_k,
        )
        maximum = max((score for _, score in results), default=1.0) or 1.0
        return [
            Candidate(document=document, score=score / maximum, source="keyword", rank=rank)
            for rank, (document, score) in enumerate(results)
        ]


class HybridRetrieval:
    """编排查询改写、多路召回与结果融合。

    这是 Advanced RAG 对外暴露的门面。它故意**不**做授权过滤与重排——
    这两个职责分别属于 :class:`RetrievalPipeline` 与 :class:`Reranker`，
    这里只负责把各路的"原始候选"汇总成一个有序列表，再交回管线处理。
    """

    def __init__(
        self,
        rewriter: QueryRewriter,
        strategies: typing.Sequence[tuple[str, RetrievalStrategy]],
        fuser: ResultFuser,
        *,
        per_query_top_k: int = 20,
    ) -> None:
        if not strategies:
            raise ValueError("HybridRetrieval requires at least one RetrievalStrategy.")
        self._rewriter = rewriter
        self._strategies = list(strategies)
        self._fuser = fuser
        self._per_query_top_k = max(1, int(per_query_top_k))

    @property
    def rewriter(self) -> QueryRewriter:
        return self._rewriter

    @property
    def fuser(self) -> ResultFuser:
        return self._fuser

    def retrieve(
        self,
        question: str,
        *,
        tenant_id: str = "default",
        top_k: int = 8,
        allowed_sources: typing.Sequence[str] | None = None,
        owner_id: str | None = None,
    ) -> RetrievalContext:
        """执行完整链路：改写 -> 多路召回 -> 融合。"""

        rewrite = self._rewriter.rewrite(question)
        all_candidates: list[Candidate] = []
        per_query: list[dict[str, typing.Any]] = []

        for query in rewrite.all_queries():
            for label, strategy in self._strategies:
                candidates = strategy.retrieve(
                    query,
                    tenant_id=tenant_id,
                    top_k=self._per_query_top_k,
                    allowed_sources=allowed_sources,
                    owner_id=owner_id,
                )
                per_query.append(
                    {
                        "query": query,
                        "strategy": label,
                        "candidate_count": len(candidates),
                    }
                )
                all_candidates.extend(candidates)

        fused = self._fuser.fuse(all_candidates, top_k=top_k)
        context = RetrievalContext(
            query=question,
            documents=[entry.document for entry in fused],
            search_type="hybrid" if len(self._strategies) > 1 else self._strategies[0][0],
            rewrite=rewrite,
            candidates=all_candidates,
            strategy_notes=per_query,
        )
        logger.info(
            "hybrid_retrieval_completed",
            rewrite_method=rewrite.method,
            queries=len(rewrite.all_queries()),
            strategies=[label for label, _ in self._strategies],
            fused=len(fused),
        )
        return context


class RetrievalContext(RetrievedContext):
    """带策略审计信息的检索上下文，向后兼容 :class:`RetrievedContext`。"""

    rewrite: QueryRewrite = Field(default_factory=lambda: QueryRewrite(original="", rewritten=""))
    candidates: list[Candidate] = Field(default_factory=list)
    strategy_notes: list[dict[str, typing.Any]] = Field(default_factory=list)

    def distinct_sources(self) -> list[str]:
        seen: list[str] = []
        for document in self.documents:
            if document.source not in seen:
                seen.append(document.source)
        return seen


# ---------------------------------------------------------------------------
# 结果融合
# ---------------------------------------------------------------------------


class ResultFuser(abc.ABC):
    """把多路、多查询的候选合并成一个有序的顶级列表。"""

    name: str = "abstract"

    @abc.abstractmethod
    def fuse(self, candidates: list[Candidate], *, top_k: int) -> list[Candidate]:
        """返回去重并重新排序后的前 ``top_k`` 个候选。"""


class ReciprocalRankFusion(ResultFuser):
    """倒数排名融合（RRF）。

    k 是平滑常数，经验值 60。RRF 不要求各路分数同量纲，因此对"向量分数 +
    BM25 分数"这种异构输入天然友好；相比线性加权，它也不需要调权重。
    """

    name = "rrf"

    def __init__(self, k: int = 60) -> None:
        if k <= 0:
            raise ValueError("RRF constant k must be positive.")
        self._k = k

    def fuse(self, candidates: list[Candidate], *, top_k: int) -> list[Candidate]:
        if not candidates:
            return []

        grouped: dict[str, Candidate] = {}
        scores: dict[str, float] = {}
        for candidate in candidates:
            key = candidate.document.chunk_id
            # 同文档保留最高分；审计上记录其最优来源，不影响排序
            if key not in grouped or candidate.score > grouped[key].score:
                grouped[key] = candidate
            scores[key] = scores.get(key, 0.0) + 1.0 / (self._k + candidate.rank + 1)

        ordered = sorted(grouped.values(), key=lambda c: scores[c.chunk_id], reverse=True)
        return ordered[: max(1, int(top_k))]


class ScoreWeightedFuser(ResultFuser):
    """分数加权融合。

    当各路分数已经做过归一化（如都映射到 [0, 1]）且业务上确有优先级
    偏好时使用。如果某个来源的权重为 0，则该路只贡献候选、不贡献分数——
    这可以用来实现"用 A 路做主排序，用 B 路做兜底覆盖"。
    """

    name = "weighted"

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self._weights = dict(weights or {})

    def fuse(self, candidates: list[Candidate], *, top_k: int) -> list[Candidate]:
        if not candidates:
            return []

        grouped: dict[str, Candidate] = {}
        scores: dict[str, float] = {}
        for candidate in candidates:
            key = candidate.document.chunk_id
            weight = float(self._weights.get(candidate.source, 1.0))
            if key not in grouped or candidate.score > grouped[key].score:
                grouped[key] = candidate
            scores[key] = scores.get(key, 0.0) + weight * candidate.score

        ordered = sorted(grouped.values(), key=lambda c: scores[c.chunk_id], reverse=True)
        return ordered[: max(1, int(top_k))]


class DeduplicatingFuser(ResultFuser):
    """只做去重保序，保留各路自身的相对顺序（来源内部稳定，跨来源按首次出现）。

    用于"只想要一个有序合并、不想引入融合偏置"的对照实验。
    """

    name = "dedupe"

    def fuse(self, candidates: list[Candidate], *, top_k: int) -> list[Candidate]:
        seen: dict[str, Candidate] = {}
        for candidate in candidates:
            key = candidate.document.chunk_id
            if key not in seen:
                seen[key] = candidate
        return list(seen.values())[: max(1, int(top_k))]


# ---------------------------------------------------------------------------
# BM25 索引抽象与默认实现
# ---------------------------------------------------------------------------


class BM25Index(abc.ABC):
    """关键词索引的最小契约。实现可基于 rank_bm25、SQLite FTS5 或 Elasticsearch。"""

    @abc.abstractmethod
    def search(
        self,
        tokens: list[str],
        *,
        tenant_id: str,
        owner_id: str | None,
        allowed_sources: list[str] | None,
        top_k: int,
    ) -> list[tuple[RetrievedDocument, float]]:
        """返回 ``(document, raw_score)`` 列表。"""


class InMemoryBM25Index(BM25Index):
    """可注入打分器与词频提取器的内存索引，便于测试与小规模部署。

    打分器接受 ``(query_tokens, document_tokens)``，返回非负分数。缺省为
    TF-IDF 风格：命中词项加权累加，并对文档长度做平方根平滑。这不是严格
    的 Robertson/Spärck Jones BM25，但在纯净 Python 下足够稳，且**行为
    完全由可替换组件决定**——把 ``scorer`` 换成真正的 BM25 公式即可升级，
    不用改调用方。
    """

    def __init__(
        self,
        *,
        tokenizer: typing.Callable[[str], list[str]] | None = None,
        scorer: typing.Callable[[list[str], list[str], dict[str, float]], float] | None = None,
    ) -> None:
        self._documents: list[RetrievedDocument] = []
        self._tokenize = tokenizer or _default_tokenize
        self._scorer = scorer or _tfidf_score
        self._doc_freq: dict[str, int] = {}
        self._total = 0

    def add(self, documents: typing.Iterable[RetrievedDocument]) -> None:
        for document in documents:
            self._documents.append(document)
            tokens = self._tokenize(document.content)
            for token in set(tokens):
                self._doc_freq[token] = self._doc_freq.get(token, 0) + 1
            self._total += 1

    def __len__(self) -> int:
        return len(self._documents)

    def search(
        self,
        tokens: list[str],
        *,
        tenant_id: str,
        owner_id: str | None,
        allowed_sources: list[str] | None,
        top_k: int,
    ) -> list[tuple[RetrievedDocument, float]]:
        if not tokens or not self._documents:
            return []
        norm = _idf_normalizer(self._doc_freq, self._total)
        results: list[tuple[RetrievedDocument, float]] = []
        for document in self._documents:
            if not _matches_scope(document, tenant_id, owner_id, allowed_sources):
                continue
            score = self._scorer(tokens, self._tokenize(document.content), norm)
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
    if document.tenant_id and document.tenant_id != tenant_id:
        return False
    if owner_id and document.owner_id and document.owner_id != owner_id:
        return False
    return not (allowed_sources and document.source not in allowed_sources)


def _default_tokenize(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    # 中文不做分词（没有 jieba 依赖时按字符切分，够用），拉丁部分按非字母切
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


def _tfidf_score(
    query: list[str], document: list[str], idf: dict[str, float]
) -> float:
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


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------


def build_retrieval_strategy(config: dict[str, typing.Any]) -> RetrievalStrategy:
    """根据配置构造单一检索策略。目前仅暴露 vector；hybrid 用 :class:`HybridRetrieval`。"""

    config = dict(config or {})
    kind = str(config.pop("type", "vector")).strip().lower()
    if kind == "vector":
        pipeline = config.get("pipeline")
        if pipeline is None:
            raise ValueError("vector strategy requires a RetrievalPipeline instance (key 'pipeline').")
        return VectorRetrievalStrategy(pipeline)
    raise ValueError(f"Unknown retrieval strategy type: {kind!r}.")


def create_query_rewriter(
    config: dict[str, typing.Any] | None,
    *,
    model: ChatModel | None = None,
) -> QueryRewriter:
    """构造查询改写器；配置不合法或无 LLM 时退化为规则改写。"""

    config = dict(config or {})
    method = str(config.get("method", "expansion")).strip().lower()
    max_queries = int(config.get("max_queries", 3))

    if method == "llm":
        if model is None:
            logger.warning("query_rewriter_llm_disabled_no_model")
            return ExpansionRewriter(max_queries)
        try:
            return LLMQueryRewriter(model, max_queries=max_queries)
        except Exception as exc:
            logger.warning("query_rewriter_fallback_to_expansion", extra={"error": str(exc)})
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
    raise ValueError(f"Unknown result fuser type: {kind!r}.")


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
    "build_retrieval_strategy",
    "create_fuser",
    "create_query_rewriter",
]
