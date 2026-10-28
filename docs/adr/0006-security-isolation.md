---
id: 0006-security-isolation
aliases: []
tags: []
---

# ADR-0006: 向量指纹治理与安全默认策略

- Status: Accepted
- Date: 2026-10-19
- Related:
  - 代码：`src/rag/collection_policy.py`、`src/rag/tools.py` (Guardrails)
  - 配置：`config/settings.yaml`（collection_strategy, allowed_tools）

## Context

企业级 RAG 存在三个致命的安全隐患：
1. **静默数据损坏**：如果运维为了省钱把 Embedding 模型从 1536 维换到了 1024 维，而没重建索引，向量库不会报错，但检索出来的全是不相干的垃圾。这叫静默灾难。
2. **提示词注入与越权**：用户输入恶意 Prompt 诱导大模型去执行未授权的本地操作（如执行命令、读其他目录）。
3. **数据越权**：多租户环境下，A 租户绝对不能查出 B 租户的数据。

## Decision

采取“物理指纹隔离 + 白名单控制 + 启动即拒绝”的安全默认策略：

1. **Embedding 指纹与物理隔离**：
   在 `collection_policy.py` 中，将 `model_name`、`dimension`、`normalize` 打包计算 SHA-256 生成 `EmbeddingFingerprint`。
   集合名称强制映射为 `base_name_{dim}d_{short_id}`。在打开集合前执行 `validate_compatibility`，如果本地存储的维度与当前 Embedder 不一致，直接抛出 `CollectionCompatibilityError`（409），**坚决不允许启动**。
2. **工具默认拒绝**：
   在 `AgentOrchestrator` 或 `AgentRunner` 中，`ToolRegistry` 默认不包含任何工具。必须通过配置显式声明白名单 `allowed_tools`。
3. **输入 Guardrails**：
   在请求刚进入系统时，立刻过一遍 `self._guardrails.check_question()`，校验轮次预算（`check_iteration`）和工具调用次数（`check_tool_calls`），一旦超限直接断死。

## Consequences
**正面**
- **防呆天花板**：即使运维不懂代码，只要换错了 Embedding 模型，系统也会直接拒绝启动并报错，绝不带病运行。
- **极低延迟的安全检查**：指纹和拦截都在微秒级完成，零性能损耗。

**负面**
- **灵活性受限**：如果想换 Embedding 模型，必须经历“建新集合 -> 重新写入数据”的过程，无法原地升级。
