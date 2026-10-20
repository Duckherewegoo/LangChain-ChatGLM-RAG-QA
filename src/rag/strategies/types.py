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
      - source：检索通道标识（"vector" / "keyword" / 其他自定义实现）。
      - rank：在该路结果中的排名，从 0 开始计数。
        RRF 等融合器依赖该约定计算分母 k + rank + 1。
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
        """
        结果是否真的和原问题不同。
        注意这与 method != "identity" 不同：ExpansionRewriter 在没有产生扩展时
        method="expansion" 但结果与原问题一致，此时应视为"未改变"。
        """
        return self.method != "identity"
