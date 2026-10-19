import sys

sys.path.insert(0, ".")
import traceback
from src.settings import AppSettings

try:
    # 使用 model_validate 接收完整字典，绕过 Pyright 对嵌套子模型的静态检查，
    # 让 Pydantic 在运行时自动完成实例化和校验。
    s = AppSettings.model_validate({
        "app_name": "enterprise-rag-chatglm",  # 显式补上，防止旧版 settings 缺失报错
        "model_chat": {
            "provider": "fake",
            "model_name": "fake-model",
        },
        "model_embedding": {
            "provider": "local",
            "model_name": "fake-embed",
            "device": "cpu",
            "batch_size": 8,
        },
        "retrieval": {
            "vector_store": {
                "type": "chromadb",
                "collection": "test",
                "persist_directory": "./data/index",
            },
            "top_k": 4,
            "fetch_multiplier": 2,
            "rerank": {
                "enabled": False,
                "model_name": "fake-rerank",
                "top_k": 2,
                "min_score": 0.0,
            },
        },
        "agent": {
            "max_iterations": 3,
            "max_tool_calls": 4,
        },
        "guardrails": {
            "max_question_length": 500,
            # 如果你的 settings.py 里还没加这个字段，运行时会被 Pydantic 忽略或报错，
            # 这里显式删去，只留确定存在的字段
            # "max_history_messages": 4, 
        },
        "model_router": {
            "models": {
                "fake-primary": {
                    "name": "fake-primary",
                    "provider": "fake",
                    "model_name": "fake-model",
                    "api_key": "test-key",
                }
            },
            "use_cases": {
                "chat": {"use_case": "chat", "primary": "fake-primary"},
                "agent": {"use_case": "agent", "primary": "fake-primary"},
            },
            "default_use_case": "chat",
        },
    })
    
    print("OK", s.app_name)
    # 顺手验证一下嵌套模型到底有没有被正确转换
    print("model_router type:", type(s.model_router).__name__)
    print("retrieval.vector_store type:", type(s.retrieval.vector_store).__name__)
except Exception:
    traceback.print_exc()
