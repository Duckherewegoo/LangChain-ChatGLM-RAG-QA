"""检索策略层：结果融合器。"""

from __future__ import annotations

import abc

from .types import Candidate


class ResultFuser(abc.ABC):
    """把多路、多查询的候选合并成一个有序的顶级列表。"""
    name: str = "abstract"

    @abc.abstractmethod
    def fuse(self, candidates: list[Candidate], *, top_k: int) -> list[Candidate]:
        ...


class ReciprocalRankFusion(ResultFuser):
    """倒数排名融合（RRF）。"""
    name = "rrf"

    def __init__(self, k: int = 60) -> None:
        if k <= 0:
            raise ValueError("RRF 常数 k 必须大于 0。")
        self._k = k

    def fuse(self, candidates: list[Candidate], *, top_k: int) -> list[Candidate]:
        if not candidates:
            return []
        grouped: dict[str, Candidate] = {}
        scores: dict[str, float] = {}
        for candidate in candidates:
            key = candidate.document.chunk_id
            # 跨 source 分数虽不可直接比较，但"保留哪个 document 版本"
            # （如是否含 rerank_score）需要一个确定性规则。
            # 当前规则：score 高者优先；可能保留 vector 版本。
            if key not in grouped or candidate.score > grouped[key].score:
                grouped[key] = candidate
            scores[key] = scores.get(key, 0.0) + 1.0 / (self._k + candidate.rank + 1)

        ordered = sorted(grouped.values(), key=lambda c: scores[c.chunk_id], reverse=True)
        return ordered[: max(1, int(top_k))]


class ScoreWeightedFuser(ResultFuser):
    """分数加权融合。"""
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
            # 未配置的 source 默认权重 1.0：可能与已配的 vector/keyword
            # 相比更高（如配 vector=0.7 / keyword=0.3，未配的 graph=1.0）。
            # 这是有意的"未配置=全权重"策略；如需严格校验，应在
            # create_fuser 里检查 weights 是否覆盖所有注册 source。
            weight = float(self._weights.get(candidate.source, 1.0))
            if key not in grouped or candidate.score > grouped[key].score:
                grouped[key] = candidate
            scores[key] = scores.get(key, 0.0) + weight * candidate.score

        ordered = sorted(grouped.values(), key=lambda c: scores[c.chunk_id], reverse=True)
        return ordered[: max(1, int(top_k))]


class DeduplicatingFuser(ResultFuser):
    """只做去重保序，不计算分数。

    顺序 = candidates 的插入顺序。HybridRetrieval 按 "query 循环 ×
    strategy 循环" 展开候选，因此排序实际由 strategies 的配置顺序决定——
    排在前面的通道优先。适合"信任某一路"的场景。
    """
    name = "dedupe"

    def fuse(self, candidates: list[Candidate], *, top_k: int) -> list[Candidate]:
        seen: dict[str, Candidate] = {}
        for candidate in candidates:
            key = candidate.document.chunk_id
            if key not in seen:
                seen[key] = candidate
        return list(seen.values())[: max(1, int(top_k))]
