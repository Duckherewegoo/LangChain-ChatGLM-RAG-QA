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
            weight = float(self._weights.get(candidate.source, 1.0))
            if key not in grouped or candidate.score > grouped[key].score:
                grouped[key] = candidate
            scores[key] = scores.get(key, 0.0) + weight * candidate.score

        ordered = sorted(grouped.values(), key=lambda c: scores[c.chunk_id], reverse=True)
        return ordered[: max(1, int(top_k))]


class DeduplicatingFuser(ResultFuser):
    """只做去重保序。"""
    name = "dedupe"

    def fuse(self, candidates: list[Candidate], *, top_k: int) -> list[Candidate]:
        seen: dict[str, Candidate] = {}
        for candidate in candidates:
            key = candidate.document.chunk_id
            if key not in seen:
                seen[key] = candidate
        return list(seen.values())[: max(1, int(top_k))]
