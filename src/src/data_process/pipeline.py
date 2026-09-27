"""Document parsing, normalization, chunking, and ingestion pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import re
import typing
from dataclasses import dataclass

from pydantic import BaseModel, Field

from src.exceptions import DocumentLoadError, IngestError
from src.rag.vector_store import VectorRecord

if typing.TYPE_CHECKING:  # pragma: no cover - type checking only
    from collections.abc import Iterable

    from langchain_core.documents import Document as LangChainDocument
    from langchain_text_splitters import TextSplitter

_MAX_SCAN_DEPTH = 12


class Chunk(BaseModel):
    """A normalized text chunk produced by the ingestion pipeline."""

    document_id: str
    chunk_id: str
    text: str
    title: str = ""
    source: str
    tenant_id: str = "default"
    page: int | None = None
    chunk_index: int = 0
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, typing.Any] = Field(default_factory=dict)

    def to_vector_record(self, owner_id: str | None = None) -> VectorRecord:
        """Create a vector-store record while preserving all chunk metadata."""

        return VectorRecord(
            document_id=self.document_id,
            chunk_id=self.chunk_id,
            text=self.text,
            title=self.title,
            source=self.source,
            tenant_id=self.tenant_id,
            owner_id=owner_id,
            page=self.page,
            tags=list(self.tags),
            metadata={"chunk_index": self.chunk_index, **self.metadata},
        )


class DocumentMeta(BaseModel):
    """Lightweight document descriptor used for inventory and filtering."""

    document_id: str
    source: str
    title: str = ""
    tenant_id: str = "default"
    owner_id: str | None = None
    bytes_size: int = 0
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, typing.Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class IngestStatistics:
    """Outcome of an ingestion run."""

    document_id: str
    source: str
    scanned: int
    chunks_created: int
    chunks_stored: int
    bytes_size: int
    skipped: bool
    reason: str | None = None

    def to_dict(self) -> dict[str, typing.Any]:
        return self.__dict__


class DocumentLoader:
    """Load raw text from files in a tenant- and extension-aware manner."""

    def __init__(self, allowed_extensions: Iterable[str] | None = None) -> None:
        normalized: list[str] = []
        for extension in allowed_extensions or []:
            normalized.append(extension.lower().lstrip("."))
        self._extensions = normalized

    @property
    def allowed_extensions(self) -> list[str]:
        return list(self._extensions)

    def scan(self, root: str | os.PathLike[str], pattern: str = "**/*") -> list[str]:
        """Discover files matching the configured extensions under ``root``."""

        base = os.path.abspath(root)
        if not os.path.exists(base):
            raise DocumentLoadError(f"Input path does not exist: {base}")

        if os.path.isfile(base):
            return [base]

        results: list[str] = []
        for directory, _dirs, files in os.walk(base, followlinks=False):
            if directory.count(os.sep) - base.count(os.sep) > _MAX_SCAN_DEPTH:
                continue
            for filename in files:
                full = os.path.join(directory, filename)
                if self._matches(full):
                    results.append(full)
        results.sort()
        return results

    def load(self, path: str | os.PathLike[str]) -> tuple[str, dict[str, typing.Any]]:
        """Load the text and basic metadata for a supported file."""

        absolute = os.path.abspath(path)
        if not os.path.isfile(absolute):
            raise DocumentLoadError(f"File not found: {absolute}")

        extension = os.path.splitext(absolute)[1].lower().lstrip(".")
        if self._extensions and extension not in self._extensions:
            raise DocumentLoadError(
                f"Unsupported file extension: .{extension}", details={"path": absolute}
            )

        loader = _LOADERS.get(extension, _load_text)
        text, extra = loader(absolute)
        if not text.strip():
            raise DocumentLoadError(
                "Loaded document is empty after normalization.", details={"path": absolute}
            )
        metadata = {"extension": extension, "bytes_size": os.path.getsize(absolute), **extra}
        return text, metadata

    def _matches(self, path: str) -> bool:
        if not self._extensions:
            return True
        return os.path.splitext(path)[1].lower().lstrip(".") in self._extensions


def _load_text(path: str) -> tuple[str, dict[str, typing.Any]]:
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
        try:
            with open(path, encoding=encoding) as stream:
                return stream.read(), {}
        except UnicodeDecodeError:
            continue
    raise DocumentLoadError(
        "Failed to decode file with utf-8, gb18030, or gbk.", details={"path": path}
    )


def _load_json(path: str) -> tuple[str, dict[str, typing.Any]]:
    with open(path, encoding="utf-8") as stream:
        data = json.load(stream)
    if isinstance(data, dict):
        text = data.get("content") or data.get("text")
        extra = {key: value for key, value in data.items() if key not in {"content", "text"}}
        if isinstance(text, str) and text.strip():
            return text, extra
        return json.dumps(data, ensure_ascii=False, indent=2), {}
    if isinstance(data, list):
        rendered = []
        for item in data:
            if isinstance(item, dict):
                rendered.append(json.dumps(item, ensure_ascii=False))
            else:
                rendered.append(str(item))
        return "\n".join(rendered), {}
    return json.dumps(data, ensure_ascii=False), {}


def _load_markdown(path: str) -> tuple[str, dict[str, typing.Any]]:
    text, _extra = _load_text(path)
    return _normalize_whitespace(text), {"format": "markdown"}


def _load_pdf(path: str) -> tuple[str, dict[str, typing.Any]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise DocumentLoadError(
            "PDF support requires pypdf. Install it or convert the document to text.",
            details={"path": path},
        ) from exc

    reader = PdfReader(path)
    pages: list[str] = []
    for page in reader.pages:
        content = page.extract_text() or ""
        pages.append(content)
    return "\n\n".join(pages), {"pages": len(pages)}


def _load_docx(path: str) -> tuple[str, dict[str, typing.Any]]:
    try:
        from docx import Document
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise DocumentLoadError(
            "DOCX support requires python-docx. Install it or convert the document to text.",
            details={"path": path},
        ) from exc

    document = Document(path)
    paragraphs = [paragraph.text for paragraph in document.paragraphs]
    return "\n\n".join(paragraphs), {"paragraphs": len(paragraphs)}


_LOADERS: dict[str, typing.Callable[[str], tuple[str, dict[str, typing.Any]]]] = {
    "txt": _load_text,
    "text": _load_text,
    "md": _load_markdown,
    "markdown": _load_markdown,
    "json": _load_json,
    "jsonl": _load_json,
    "pdf": _load_pdf,
    "docx": _load_docx,
}


class DocumentChunker:
    """Split documents into normalized chunks with stable identifiers."""

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100, min_characters: int = 50) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be between 0 and chunk_size - 1.")
        if min_characters <= 0:
            raise ValueError("min_characters must be positive.")

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._min_characters = min_characters

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def chunk_overlap(self) -> int:
        return self._chunk_overlap

    def split(
        self,
        text: str,
        *,
        source: str,
        title: str = "",
        document_id: str | None = None,
        tenant_id: str = "default",
        tags: Iterable[str] | None = None,
        metadata: dict[str, typing.Any] | None = None,
    ) -> list[Chunk]:
        """Split text into normalized chunks."""

        normalized = _normalize_whitespace(text)
        if not normalized.strip():
            return []

        sections = self._split_sections(normalized)
        document_id = document_id or _document_id(source, normalized)
        chunks: list[Chunk] = []
        for section in sections:
            if len(section) < self._min_characters:
                continue
            chunks.extend(self._chunk_section(section, document_id, index_start=len(chunks),
                                             title=title, source=source, tenant_id=tenant_id,
                                             tags=tags, metadata=metadata))
        return chunks

    def _chunk_section(
        self,
        section: str,
        document_id: str,
        *,
        index_start: int,
        title: str,
        source: str,
        tenant_id: str,
        tags: typing.Iterable[str] | None,
        metadata: dict[str, typing.Any] | None,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        if len(section) <= self._chunk_size:
            chunks.append(self._create_chunk(
                document_id, index_start, section, title=title, source=source,
                tenant_id=tenant_id, tags=tags, metadata=metadata
            ))
            return chunks

        stride = max(1, self._chunk_size - self._chunk_overlap)
        position = 0
        index = index_start
        while position < len(section):
            chunks.append(self._create_chunk(
                document_id, index, section[position:position + self._chunk_size],
                title=title, source=source, tenant_id=tenant_id, tags=tags, metadata=metadata
            ))
            index += 1
            position += stride
            if len(section) - position <= self._chunk_size:
                if position < len(section):
                    chunks.append(self._create_chunk(
                        document_id, index, section[position:],
                        title=title, source=source, tenant_id=tenant_id, tags=tags, metadata=metadata
                    ))
                break
        return chunks

    def _create_chunk(
        self,
        document_id: str,
        index: int,
        text: str,
        *,
        title: str,
        source: str,
        tenant_id: str,
        tags: typing.Iterable[str] | None,
        metadata: dict[str, typing.Any] | None,
    ) -> Chunk:
        return Chunk(
            document_id=document_id,
            chunk_id=f"{document_id}#{index}",
            text=text,
            title=title,
            source=source,
            tenant_id=tenant_id,
            chunk_index=index,
            tags=list(tags or []),
            metadata=dict(metadata or {}),
        )

    def split_langchain_documents(self, documents: Iterable[LangChainDocument]) -> list[Chunk]:
        """Convert LangChain documents into project chunks."""

        results: list[Chunk] = []
        for document in documents:
            source = str(document.metadata.get("source", ""))
            results.extend(
                self.split(
                    document.page_content,
                    source=source,
                    title=str(document.metadata.get("title", "")),
                    tenant_id=str(document.metadata.get("tenant_id", "default")),
                    tags=list(document.metadata.get("tags", []) or []),
                    metadata=document.metadata,
                )
            )
        return results

    def to_text_splitter(self) -> TextSplitter:
        """Create a LangChain text splitter using the same size and overlap."""

        from langchain_text_splitters import CharacterTextSplitter

        return CharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            separator="\n",
            length_function=len,
        )

    def _split_sections(self, text: str) -> list[str]:
        """Respect paragraph and heading boundaries before applying fixed windows."""

        parts = re.split(r"\n{2,}", text)
        sections: list[str] = []
        buffer = ""
        for part in parts:
            cleaned = part.strip()
            if not cleaned:
                continue
            if len(buffer) + len(cleaned) + 1 <= self._chunk_size:
                buffer = f"{buffer}\n{cleaned}".strip() if buffer else cleaned
            else:
                if buffer:
                    sections.append(buffer)
                buffer = cleaned
        if buffer:
            sections.append(buffer)
        return sections


class IngestPipeline:
    """End-to-end ingestion coordinator.

    It wires a loader, chunker, embedder, and vector store together while
    preserving document identity and avoiding duplicate chunk writes through
    stable identifiers.
    """

    def __init__(self, loader: DocumentLoader, chunker: DocumentChunker, vector_store: typing.Any) -> None:
        self._loader = loader
        self._chunker = chunker
        self._vector_store = vector_store

    def ingest_path(
        self,
        path: str | os.PathLike[str],
        *,
        tenant_id: str = "default",
        owner_id: str | None = None,
        embedder: typing.Any,
        title: str | None = None,
        tags: Iterable[str] | None = None,
    ) -> IngestStatistics:
        """Load, parse, chunk, embed, and persist documents under a path."""

        files = self._loader.scan(path) if os.path.isdir(path) else [os.path.abspath(path)]
        if not files:
            raise IngestError(f"No supported documents were found under: {path}")

        total_chunks = 0
        total_stored = 0
        total_bytes = 0
        document_id = _document_id(files[0], None)

        for file_path in files:
            text, metadata = self._loader.load(file_path)
            total_bytes += int(metadata.get("bytes_size", 0))
            chunks = self._chunker.split(
                text,
                source=file_path,
                title=title or os.path.basename(file_path),
                tenant_id=tenant_id,
                tags=tags,
            )
            if not chunks:
                continue

            records = [chunk.to_vector_record(owner_id=owner_id) for chunk in chunks]
            embeddings = embedder.embed_documents([record.text for record in records])
            total_stored += self._vector_store.add(records, embeddings)
            total_chunks += len(chunks)

        return IngestStatistics(
            document_id=document_id,
            source=str(path),
            scanned=len(files),
            chunks_created=total_chunks,
            chunks_stored=total_stored,
            bytes_size=total_bytes,
            skipped=False,
        )

    def remove_document(self, document_id: str, tenant_id: str = "default") -> int:
        """Delete all chunks associated with a document."""

        return self._vector_store.delete_by_document(document_id, tenant_id)


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.splitlines())


def _document_id(source: str, content: str | None) -> str:
    digest = hashlib.sha256()
    digest.update(os.path.abspath(source).encode("utf-8"))
    if content is not None:
        digest.update(b"\x00")
        digest.update(content.encode("utf-8"))
    return digest.hexdigest()[:24]


__all__ = [
    "Chunk",
    "DocumentChunker",
    "DocumentLoader",
    "DocumentMeta",
    "IngestPipeline",
    "IngestStatistics",
]
