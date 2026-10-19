"""向量存储抽象与 ChromaDB 实现。"""

from __future__ import annotations

import abc
import logging
import os
import threading
import typing

from pydantic import BaseModel, Field

from src.exceptions import CollectionCompatibilityError

if typing.TYPE_CHECKING:  # 仅用于类型检查
    from collections.abc import Sequence

    from chromadb.api import ClientAPI

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, type[VectorStore]] = {}


class VectorRecord(BaseModel):
    """准备持久化的文档块及其元数据。"""
    document_id: str
    chunk_id: str
    text: str
    title: str = ""
    source: str
    tenant_id: str = "default"
    owner_id: str | None = None
    page: int | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, typing.Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """带分数的搜索结果。"""
    record: VectorRecord
    score: float


class VectorStore(abc.ABC):
    """检索管线使用的向量存储协议。"""

    @abc.abstractmethod
    def add(self, records: Sequence[VectorRecord], embeddings: Sequence[Sequence[float]]) -> int:
        """持久化记录并返回添加的数量。"""
        ...

    @abc.abstractmethod
    def search(
        self,
        query_embedding: Sequence[float],
        *,
        tenant_id: str = "default",
        top_k: int = 8,
        allowed_sources: Sequence[str] | None = None,
    ) -> list[SearchResult]:
        """返回查询向量的前 top_k 个候选。"""
        ...

    @abc.abstractmethod
    def delete_by_document(self, document_id: str, tenant_id: str = "default") -> int:
        """删除属于某文档的所有块。"""
        ...

    @abc.abstractmethod
    def count(self, tenant_id: str = "default") -> int:
        """返回持久化的块数量。"""
        ...

    @abc.abstractmethod
    def dimension(self) -> int | None:
        """返回集合中存储的向量维度（若已知）。"""
        ...

    @abc.abstractmethod
    def describe(self) -> dict[str, typing.Any]:
        """返回用于诊断的存储级元数据。"""
        ...

    def close(self) -> None:
        """释放存储持有的资源。"""
        return


def register_vector_store(name: str, store_type: type[VectorStore]) -> None:
    """注册一个向量存储实现。"""
    _REGISTRY[name.strip().lower()] = store_type


def create_vector_store(store_type: str, **options: typing.Any) -> VectorStore:
    """按注册的类型名称创建向量存储。"""
    implementation = _REGISTRY.get(store_type.strip().lower())
    if implementation is None:
        raise ValueError(
            f"不支持的向量存储类型: {store_type!r}; 可用类型={sorted(_REGISTRY)}"
        )
    return implementation(**options)


class ChromaVectorStore(VectorStore):
    """持久化的 ChromaDB 向量存储。"""

    def __init__(
        self,
        collection: str = "enterprise_knowledge",
        persist_directory: str | os.PathLike[str] = "./data/index",
        tenant_id: str = "default",
        hnsw_space: str = "cosine",
    ) -> None:
        self._collection_name = collection
        self._persist_directory = os.path.abspath(persist_directory)
        self._tenant_id = tenant_id
        self._hnsw_space = hnsw_space
        self._client: ClientAPI | None = None
        self._collection: typing.Any = None
        self._lock = threading.Lock()
        self._known_dimension: int | None = None
        os.makedirs(self._persist_directory, exist_ok=True)

    def _get_collection(self) -> typing.Any:
        if self._collection is not None:
            return self._collection

        with self._lock:
            if self._collection is not None:
                return self._collection

            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self._client = chromadb.PersistentClient(
                path=self._persist_directory,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            metadata = {"hnsw:space": self._hnsw_space}
            try:
                self._collection = self._client.get_collection(name=self._collection_name)
            except Exception:
                self._collection = self._client.create_collection(
                    name=self._collection_name, metadata=metadata
                )
            self._refresh_dimension()
        return self._collection

    def _refresh_dimension(self) -> None:
        collection = self._collection
        metadata = getattr(collection, "metadata", None) or {}
        stored = metadata.get("dimension")
        if isinstance(stored, int):
            self._known_dimension = stored
            return

        try:
            peek = collection.peek(limit=1)
        except Exception:
            peek = None
        embeddings = peek.get("embeddings") if isinstance(peek, dict) else None
        if embeddings and len(embeddings) and len(embeddings[0]):
            self._known_dimension = len(embeddings[0])

    def _record_id(self, tenant_id: str, chunk_id: str) -> str:
        return f"{tenant_id}:{chunk_id}"

    def add(self, records: Sequence[VectorRecord], embeddings: Sequence[Sequence[float]]) -> int:
        if len(records) != len(embeddings):
            raise ValueError("records 和 embeddings 长度必须相同。")

        collection = self._get_collection()
        dimension = len(embeddings[0]) if embeddings else 0
        self._assert_dimension_compatibility(dimension)
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, typing.Any]] = []
        embedded: list[list[float]] = []

        for record, vector in zip(records, embeddings, strict=True):
            if len(vector) != dimension:
                raise ValueError("所有 embeddings 必须具有相同的维度。")
            ids.append(self._record_id(record.tenant_id, record.chunk_id))
            documents.append(record.text)
            metadatas.append(self._serialize_metadata(record))
            embedded.append([float(value) for value in vector])

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embedded,
            metadatas=metadatas,
        )
        if self._known_dimension is None:
            self._known_dimension = dimension
            self._persist_dimension(dimension)
        return len(ids)

    def _assert_dimension_compatibility(self, dimension: int) -> None:
        if self._known_dimension is not None and self._known_dimension != dimension:
            raise CollectionCompatibilityError(
                "无法写入与现有集合维度不同的 embeddings。",
                details={
                    "collection": self._collection_name,
                    "stored_dimension": self._known_dimension,
                    "attempted_dimension": dimension,
                },
            )

    def _persist_dimension(self, dimension: int) -> None:
        try:
            collection = self._get_collection()
            collection.modify(metadata={**(collection.metadata or {}), "dimension": dimension})
        except Exception as exc:
            logger.warning(f"持久化集合维度失败: {exc}")

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        tenant_id: str = "default",
        top_k: int = 8,
        allowed_sources: Sequence[str] | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0:
            return []

        collection = self._get_collection()

        # 修复：ChromaDB 对多个过滤条件必须使用 $and 包裹
        if allowed_sources:
            where_clause = {
                "$and": [
                    {"tenant_id": tenant_id},
                    {"source": {"$in": [str(item) for item in allowed_sources]}}
                ]
            }
        else:
            where_clause = {"tenant_id": tenant_id}

        response = collection.query(
            query_embeddings=[[float(value) for value in query_embedding]],
            n_results=top_k,
            where=where_clause,
        )

        ids = (response.get("ids") or [[]])[0]
        documents = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]

        results: list[SearchResult] = []
        for index, identifier in enumerate(ids):
            metadata = metadatas[index] or {}
            text = documents[index] if index < len(documents) else ""
            results.append(
                SearchResult(
                    record=self._deserialize_record(identifier, metadata, text),
                    score=self._distance_to_similarity(distances[index] if index < len(distances) else None),
                )
            )
        return results

    def delete_by_document(self, document_id: str, tenant_id: str = "default") -> int:
        collection = self._get_collection()
        # 修复：查条件删记录时，不应该传空 ids=[]
        existing = collection.get(
            where={"$and": [{"tenant_id": tenant_id}, {"document_id": document_id}]}
        )
        target_ids = existing.get("ids") or []
        if not target_ids:
            return 0
        collection.delete(ids=list(target_ids))
        return len(target_ids)

    def count(self, tenant_id: str = "default") -> int:
        collection = self._get_collection()
        return int(collection.count())

    def close(self) -> None:
        self._collection = None
        self._client = None

    def dimension(self) -> int | None:
        if self._known_dimension is None:
            try:
                self._refresh_dimension()
            except Exception:
                return None
        return self._known_dimension

    def describe(self) -> dict[str, typing.Any]:
        collection = self._get_collection()
        metadata = getattr(collection, "metadata", None) or {}
        return {
            "type": "chromadb",
            "collection": self._collection_name,
            "persist_directory": self._persist_directory,
            "dimension": self.dimension(),
            "stored_dimension": metadata.get("dimension"),
            "hnsw_space": metadata.get("hnsw:space", self._hnsw_space),
            "tenant_id": self._tenant_id,
        }

    @staticmethod
    def _serialize_metadata(record: VectorRecord) -> dict[str, typing.Any]:
        metadata: dict[str, typing.Any] = {
            "document_id": record.document_id,
            "chunk_id": record.chunk_id,
            "source": record.source,
            "title": record.title,
            "tenant_id": record.tenant_id,
            "page": record.page if record.page is not None else -1,
        }
        if record.owner_id:
            metadata["owner_id"] = record.owner_id
        if record.tags:
            metadata["tags"] = ",".join(record.tags)
        for key, value in record.metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                metadata[key] = value
        return metadata

    @staticmethod
    def _deserialize_record(
        identifier: str, metadata: dict[str, typing.Any], text: str
    ) -> VectorRecord:
        tags_raw = metadata.get("tags")
        tags = [tag for tag in str(tags_raw).split(",") if tag] if isinstance(tags_raw, str) else []

        # 修复：使用异常捕获安全解析 page，防止 numpy 类型导致丢失
        page_value = metadata.get("page")
        page = None
        try:
            if page_value is not None:
                page_num = int(page_value)
                if page_num > 0:
                    page = page_num
        except (ValueError, TypeError):
            pass

        # 修复：把非核心字段塞回 metadata
        core_keys = {"document_id", "chunk_id", "source", "title", "tenant_id", "page", "owner_id", "tags"}
        extra_metadata = {k: v for k, v in metadata.items() if k not in core_keys}

        return VectorRecord(
            document_id=str(metadata.get("document_id", "")),
            chunk_id=str(metadata.get("chunk_id", identifier)),
            text=text,
            title=str(metadata.get("title", "")),
            source=str(metadata.get("source", "")),
            tenant_id=str(metadata.get("tenant_id", "default")),
            owner_id=metadata.get("owner_id"),
            page=page,
            tags=tags,
            metadata=extra_metadata,
        )

    @staticmethod
    def _distance_to_similarity(distance: typing.Any) -> float:
        if distance is None:
            return 0.0
        try:
            value = float(distance)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, 1.0 - value)


register_vector_store("chromadb", ChromaVectorStore)
register_vector_store("chromadb_persistent", ChromaVectorStore)


__all__ = [
    "ChromaVectorStore",
    "SearchResult",
    "VectorRecord",
    "VectorStore",
    "create_vector_store",
    "register_vector_store",
]
