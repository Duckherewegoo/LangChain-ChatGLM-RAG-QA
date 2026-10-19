"""检索增强生成 (RAG) 组件。

本模块作为 RAG 包的统一入口，仅暴露无副作用的核心数据类型。
具体业务组件（如服务、管线、容器）请从各自对应的子模块中显式导入，
以避免循环依赖和启动时的过度加载。
"""

from src.rag.tool_types import ToolContext, ToolSpec
from src.rag.types import RetrievedContext, RetrievedDocument

# 如果确实有一些经常被外部使用、且没有循环依赖和重型副作用的类，可以在这里暴露：
# from src.rag.collection_policy import EmbeddingFingerprint, CollectionPolicy

__all__ = [
    "RetrievedContext",
    "RetrievedDocument",
    "ToolContext",
    "ToolSpec",
]
