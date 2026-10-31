"""检索策略层：核心数据类型定义。"""

from __future__ import annotations

import dataclasses

from pydantic import BaseModel, Field

from rag.types import RetrievedDocument


@dataclasses.dataclass(frozen=True)
class Candidate:
    """一份召回候选，携带它来自哪一路检索与原始分数。

    字段约定：
      - score：来自该路检索器的原始分数，不同 source 之间不可直接比较。
        仅供 ScoreWeightedFuser 使用；RRF 只用 rank，不读 score。
      - source：检索通道标识（"vector" / "keyword" / 其他自定义实现）。
        用于 ScoreWeightedFuser 查 weights、以及审计。
      - rank：在该路结果中的排名，从 0 开始计数。
        RRF 等融合器依赖该约定计算分母 k + rank + 1。

    注意：document 持有完整 RetrievedDocument（含 content 全文）。
    HybridRetrieval 构造 RetrievalContext 时会复制一份 candidates，
    并把其中的 content 截断到 500 字（见 hybrid._snippet_candidates）；
    最终 documents 仍使用完整原文。fuser 只看 chunk_id / score / rank，
    不读 content，截断不影响融合逻辑。
    """
    document: RetrievedDocument
    score: float
    source: str
    rank: int


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
        """结果是否真的和原问题不同。

        以 all_queries() 的实际长度为准，而非 method：无论 rewriter 标记
        method 为 "expansion" 还是 "llm"，只要 queries 为空，实际可执行的
        查询只剩原问题，应视为"未改变"。
        """
        return len(self.all_queries()) > 1
