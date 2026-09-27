"""Retrieval-augmented generation components."""

from src.rag.collection_policy import (
    CollectionCompatibilityError,
    CollectionPolicy,
    EmbeddingFingerprint,
    create_collection_policy,
    describe_compatibility,
    fingerprint_embedder,
    validate_compatibility,
)
from src.rag.container import (
    Container,
    get_container,
    initialize_container,
    shutdown_container,
)
from src.rag.reranker import (
    CrossEncoderReranker,
    IdentityReranker,
    Reranker,
    RerankerUnavailable,
    RerankResult,
    create_reranker,
    register_reranker,
)
from src.rag.retrieval import RetrievalPipeline
from src.rag.strategies import (
    BM25Index,
    Candidate,
    DeduplicatingFuser,
    ExpansionRewriter,
    HybridRetrieval,
    IdentityRewriter,
    InMemoryBM25Index,
    KeywordRetrievalStrategy,
    LLMQueryRewriter,
    QueryRewrite,
    QueryRewriter,
    ReciprocalRankFusion,
    ResultFuser,
    RetrievalContext,
    RetrievalStrategy,
    ScoreWeightedFuser,
    VectorRetrievalStrategy,
    build_retrieval_strategy,
    create_fuser,
    create_query_rewriter,
)
from src.rag.tool_sources import (
    LocalToolSource,
    MCPToolSource,
    ToolExecutionError,
    ToolSource,
    ToolSpec,
    create_tool_source,
)
from src.rag.tools import Guardrails, ToolCallRecord, ToolRegistry, create_rag_tool_registry
from src.rag.types import RetrievedContext, RetrievedDocument
from src.rag.vector_store import (
    ChromaVectorStore,
    SearchResult,
    VectorRecord,
    VectorStore,
    create_vector_store,
    register_vector_store,
)


# Heavy or cycle-prone modules are imported lazily to keep the package import cheap.
def __getattr__(name: str):  # type: ignore[no-untyped-def]
    if name in {
        "AgentRunner",
        "AgentRuntimeRequest",
        "AgentResponse",
        "AgentState",
        "AgentDependencies",
        "create_agent_runner",
    }:
        from src.rag.agent import (  # type: ignore[no-redef]  # noqa: F401
            AgentDependencies,
            AgentResponse,
            AgentRunner,
            AgentState,
            create_agent_runner,
        )
        from src.rag.agent import (  # noqa: F401
            AgentRequest as AgentRuntimeRequest,
        )

        return locals()[name]
    if name in {"RAGService", "AnswerRequest", "AnswerResult", "Citation"}:
        from src.rag.service import (  # type: ignore[no-redef]  # noqa: F401
            AnswerRequest,
            AnswerResult,
            Citation,
            RAGService,
        )

        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AgentDefinition",
    "AgentDependencies",
    "AgentResponse",
    "AgentRunner",
    "AgentRuntimeRequest",
    "AgentState",
    "AnswerRequest",
    "AnswerResult",
    "AnswerReviewer",
    "BM25Index",
    "Blackboard",
    "Candidate",
    "ChromaVectorStore",
    "Citation",
    "CollectionCompatibilityError",
    "CollectionPolicy",
    "Container",
    "CrossEncoderReranker",
    "DeduplicatingFuser",
    "EmbeddingFingerprint",
    "ExpansionRewriter",
    "Guardrails",
    "HybridRetrieval",
    "IdentityReranker",
    "IdentityRewriter",
    "InMemoryBM25Index",
    "KeywordRetrievalStrategy",
    "LLMQueryRewriter",
    "LocalToolSource",
    "MCPToolSource",
    "OrchestratorRole",
    "QueryRewrite",
    "QueryRewriter",
    "RAGService",
    "ReasonerRole",
    "ReciprocalRankFusion",
    "RerankResult",
    "Reranker",
    "RerankerUnavailable",
    "ResultFuser",
    "RetrievalContext",
    "RetrievalPipeline",
    "RetrievalStrategy",
    "RetrievedContext",
    "RetrievedDocument",
    "RetrieverRole",
    "ReviewIssue",
    "ReviewReport",
    "ReviewerRole",
    "Role",
    "RoleDefinition",
    "ScoreWeightedFuser",
    "SearchResult",
    "ToolCallRecord",
    "ToolExecutionError",
    "ToolRegistry",
    "ToolSource",
    "ToolSpec",
    "VectorRecord",
    "VectorRetrievalStrategy",
    "VectorStore",
    "build_retrieval_strategy",
    "create_agent_runner",
    "create_collection_policy",
    "create_fuser",
    "create_query_rewriter",
    "create_rag_tool_registry",
    "create_reranker",
    "create_tool_source",
    "create_vector_store",
    "describe_compatibility",
    "enterprise_qa_definition",
    "fingerprint_embedder",
    "get_container",
    "initialize_container",
    "register_reranker",
    "register_vector_store",
    "shutdown_container",
    "validate_compatibility",
]
