import sys

sys.path.insert(0, ".")
import traceback
from src.settings import (
    AppSettings,
    RetrievalSettings,
    AgentSettings,
    GuardrailsSettings,
)
from src.model.router import (
    ModelRouterSettings,
    ModelRoutingEntry,
    UseCaseBinding,
)

try:
    s = AppSettings(
        model_chat={"provider": "fake", "model_name": "fake-model"},
        model_embedding={
            "provider": "local",
            "model_name": "fake-embed",
            "device": "cpu",
            "batch_size": 8,
        },
        retrieval=RetrievalSettings(
            vector_store={
                "type": "chromadb",
                "collection": "test",
                "persist_directory": "./data/index",
            },
            top_k=4,
            fetch_multiplier=2,
            rerank={
                "enabled": False,
                "model_name": "fake-rerank",
                "top_k": 2,
                "min_score": 0.0,
            },
        ),
        agent=AgentSettings(max_iterations=3, max_tool_calls=4),
        guardrails=GuardrailsSettings(
            max_question_length=500, max_history_messages=4
        ),
        model_router=ModelRouterSettings(
            models={
                "fake-primary": ModelRoutingEntry(
                    name="fake-primary",
                    provider="fake",
                    model_name="fake-model",
                    api_key="test-key",
                )
            },
            use_cases={
                "chat": UseCaseBinding(use_case="chat", primary="fake-primary"),
                "agent": UseCaseBinding(use_case="agent", primary="fake-primary"),
            },
            default_use_case="chat",
        ),
    )
    print("OK", s.app_name)
except Exception:
    traceback.print_exc()
