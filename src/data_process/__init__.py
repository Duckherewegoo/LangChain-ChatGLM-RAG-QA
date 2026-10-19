"""文档摄入管线。

提供多格式加载、段落感知切分、与向量存储的端到端摄入能力。
"""

from .chunker import DocumentChunker, compute_document_id, normalize_whitespace
from .loaders import DocumentLoader
from .pipeline import IngestPipeline
from .types import Chunk, IngestStatistics

__all__ = [
    "Chunk",
    "DocumentChunker",
    "DocumentLoader",
    "IngestPipeline",
    "IngestStatistics",
    "compute_document_id",
    "normalize_whitespace",
]
