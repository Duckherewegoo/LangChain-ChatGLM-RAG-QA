
from __future__ import annotations

import os
import sys
import typing
from collections.abc import Sequence

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model.manager import (  # noqa: E402
    ChatModel,
    ChatRequest,
    ChatResponse,
)
from src.model.registry import clear_registry, register_provider  # noqa: E402
from src.rag.agent import AgentRequest  # noqa: E402
from src.rag.retrieval import RetrievalPipeline  # noqa: E402
from src.rag.service import AnswerRequest, RAGService  # noqa: E402
from src.rag.tools import Guardrails, ToolRegistry, create_rag_tool_registry  # noqa: E402
from src.rag.vector_store import VectorRecord  # noqa: E402
from src.model.router import (  # noqa: E402
    ModelRouterSettings,
    ModelRoutingEntry,
    UseCaseBinding,
)
from src.settings import (  # noqa: E402
    AgentSettings,
    AppSettings,
    RetrievalSettings,
    GuardrailsSettings,
)

_FAKE_PROVIDER = "fake"


class FakeChatModel(ChatModel):
    """Deterministic fake model used for service tests."""

    name = _FAKE_PROVIDER

    def __init__(self, settings) -> None:
        self._settings = settings
        self.tool_calls: list[dict[str, typing.Any]] = []
        self.tools: list[dict[str, typing.Any]] = []
        self.responses: list[ChatResponse] = [
            ChatResponse(content="基于资料，答案是42。", model=settings.model_name)
        ]

    def bind_tools(self, tools: typing.Sequence[dict[str, typing.Any]]) -> ChatModel:
        self.tools = [dict(item) for item in tools]
        return self

    def invoke(self, request: ChatRequest) -> ChatResponse:
        self.tool_calls.append({"messages": list(request.messages), "tools": list(self.tools)})
        if self.responses:
            return self.responses.pop(0)
        return ChatResponse(content="完成", model=self._settings.model_name)

    async def ainvoke(self, request: ChatRequest) -> ChatResponse:
        return self.invoke(request)


class FakeEmbedder:
    """Deterministic embedder that avoids network and model downloads."""

    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(index + 1)] * self.dimension for index in range(len(texts))]

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("query empty")
        return [1.0] * self.dimension


class FakeVectorStore:
    """In-memory vector store that ignores embeddings."""

    def __init__(self, **kwargs: typing.Any) -> None:
        self._records: dict[str, tuple[dict[str, typing.Any], float]] = {}
        self._dimension: int | None = None

    def add(self, records: Sequence[VectorRecord], embeddings: Sequence[Sequence[float]]) -> int:
        if embeddings:
            self._dimension = len(embeddings[0])
        for record, vector in zip(records, embeddings):
            if len(vector) != (self._dimension or len(vector)):
                raise ValueError("embedding dimension mismatch")
            self._records[record.chunk_id] = (record.model_dump(), 0.9)
        return len(records)

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        tenant_id: str = "default",
        top_k: int = 8,
        allowed_sources: Sequence[str] | None = None,
    ) -> list[typing.Any]:
        items: list[typing.Any] = []
        for data, score in self._records.values():
            if data["tenant_id"] != tenant_id:
                continue
            record = VectorRecord(
                **{key: value for key, value in data.items() if key != "text"}, text=data["text"]
            )
            items.append(type("SearchResult", (), {"record": record, "score": score})())
        if allowed_sources:
            items = [item for item in items if item.record.source in allowed_sources]
        return items[:top_k]

    def delete_by_document(self, document_id: str, tenant_id: str = "default") -> int:
        target = [
            chunk_id for chunk_id, (data, _) in self._records.items() if data["document_id"] == document_id
        ]
        for chunk_id in target:
            del self._records[chunk_id]
        return len(target)

    def count(self, tenant_id: str = "default") -> int:
        return sum(1 for _, (data, _) in self._records.items() if data["tenant_id"] == tenant_id)

    def dimension(self) -> int | None:
        return self._dimension

    def describe(self) -> dict[str, typing.Any]:
        return {"type": "fake", "collection": "test", "dimension": self._dimension}

    def close(self) -> None:
        return None


@pytest.fixture
def app_settings() -> AppSettings:
    """Return minimal, test-safe application settings."""

    return AppSettings(
        model_chat={"provider": _FAKE_PROVIDER, "model_name": "fake-model"},
        model_embedding={"provider": "local", "model_name": "fake-embed", "device": "cpu", "batch_size": 8},
        retrieval=RetrievalSettings(
            vector_store={"type": "chromadb", "collection": "test", "persist_directory": "./data/index"},
            top_k=4,
            fetch_multiplier=2,
            rerank={"enabled": False, "model_name": "fake-rerank", "top_k": 2, "min_score": 0.0},
        ),
        agent=AgentSettings(max_iterations=3, max_tool_calls=4),
        guardrails=GuardrailsSettings(max_question_length=500, max_history_messages=4),
        model_router=ModelRouterSettings(
            models={
                "fake-primary": ModelRoutingEntry(
                    name="fake-primary",
                    provider=_FAKE_PROVIDER,
                    model_name="fake-model",
                    api_key=os.environ.get("MODEL_CHAT_API_KEY", "test-key"),
                )
            },
            use_cases={
                "chat": UseCaseBinding(use_case="chat", primary="fake-primary"),
                "agent": UseCaseBinding(use_case="agent", primary="fake-primary"),
            },
            default_use_case="chat",
        ),
    )


@pytest.fixture(autouse=True)
def _register_fake_provider() -> typing.Iterator[None]:
    """Register and remove the deterministic fake provider."""

    register_provider(_FAKE_PROVIDER, lambda settings: FakeChatModel(settings))
    yield
    clear_registry()


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_vector_store() -> FakeVectorStore:
    return FakeVectorStore()


@pytest.fixture
def populated_vector_store(fake_vector_store: FakeVectorStore) -> FakeVectorStore:
    fake_vector_store.add(
        [
            VectorRecord(
                document_id="doc1", chunk_id="doc1#0", text="企业知识库答案42。", source="a.md",
                tenant_id="tenant-a", title="知识A"
            ),
            VectorRecord(
                document_id="doc1", chunk_id="doc1#1", text="无关内容。", source="a.md",
                tenant_id="tenant-a", title="知识A"
            ),
            VectorRecord(
                document_id="doc2", chunk_id="doc2#0", text="另一个租户的机密。", source="secret.md",
                tenant_id="tenant-b", title="机密"
            ),
        ],
        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
    )
    return fake_vector_store


@pytest.fixture
def retrieval_service(
    app_settings: AppSettings,
    fake_embedder: FakeEmbedder,
    populated_vector_store: FakeVectorStore,
) -> RAGService:
    pipeline = RetrievalPipeline(
        vector_store=populated_vector_store,
        embedder=fake_embedder,
        settings=app_settings.retrieval,
    )
    return RAGService(
        retrieval=pipeline,
        chat_model=FakeChatModel(app_settings.model_chat),
        executor=_single_executor(FakeChatModel(app_settings.model_chat)),
        embedder=fake_embedder,
        settings=app_settings.retrieval,
        guardrails=app_settings.guardrails,
    )


@pytest.fixture
def tool_registry(retrieval_service: RAGService) -> ToolRegistry:
    return create_rag_tool_registry(retrieval_service)


@pytest.fixture
def guardrails(app_settings: AppSettings) -> Guardrails:
    return Guardrails(app_settings.guardrails, agent_settings=app_settings.agent)


def _single_executor(model: ChatModel):
    from src.model.fallback import FallbackExecutor

    return FallbackExecutor(primary=model, secondaries=[], max_attempts_per_provider=1)

# ---------------------------------------------------------------------------
# Make ``tests._support`` resolvable regardless of whether ``tests`` is treated
# as a package or a namespace. The system may expose its own top-level
# ``tests`` module, which would otherwise shadow this project's directory. By
# populating the namespace explicitly we guarantee test bodies can always reach
# the shared fakes through the documented ``tests._support`` path.
# ---------------------------------------------------------------------------
import importlib
import tests as _tests_pkg

try:
    _support_module = importlib.import_module("tests._support")
except ImportError:
    import sys as _sys
    _support_path = os.path.join(os.path.dirname(__file__), "_support.py")
    _spec = importlib.util.spec_from_file_location("_support", _support_path)
    _support_module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_support_module)
    _sys.modules["tests._support"] = _support_module

for _name in ("FakeChatModel", "ToolFirstModel", "AuditableModel"):
    if not hasattr(_tests_pkg, _name):
        setattr(_tests_pkg, _name, getattr(_support_module, _name))
