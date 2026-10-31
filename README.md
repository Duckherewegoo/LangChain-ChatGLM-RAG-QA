# 企业级 RAG 智能问答系统 — LangChain-ChatGLM-RAG-QA

> 基于 **LangChain + LangGraph** 构建的检索增强生成（RAG）服务，覆盖从文档摄入、混合检索、多模型路由到多角色协作审核的完整链路。
> 重点在**架构设计与可靠性工程**，而非模型调参。

**版本** `0.3.0` ｜ **Python** `>=3.10, <3.15` ｜ **许可证** [GPL-3.0-or-later](LICENSE)

---

## 目录

- [这个项目解决什么问题](#这个项目解决什么问题)
- [核心能力](#核心能力)
- [能力矩阵](#能力矩阵)
- [架构](#架构)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [调用示例](#调用示例)
- [项目结构](#项目结构)
- [测试](#测试)
- [代码规范](#代码规范)
- [设计哲学](#设计哲学)
- [Vibe-coding Workflow](#vibe-coding-workflow)
- [已知限制](#已知限制)
- [License](#license)

---

## 这个项目解决什么问题

不是又一个"调个 API 拼 prompt"的 Demo。它的核心卖点是**把 RAG 系统当成一个真正的工程产品来做**。

| 模块 | 你能学到的核心知识 |
|---|---|
| 多模型抽象层 | 用 `Protocol` + 工厂模式屏蔽 ChatGLM / DeepSeek / Qwen / 豆包等厂商差异，新增模型零改动接入 |
| 模型路由 + Fallback | 主备模型链、指数退避重试、错误分类（可重试 vs 永久失败） |
| LangGraph Agent Harness | 用状态机实现 `reason → tool → reason` 循环，含预算守卫和审计轨迹 |
| Advanced RAG | 查询改写、BM25 + 向量混合检索、多策略融合、Cross-Encoder 重排 |
| 多角色协作 | Retriever / Reasoner / Reviewer / Orchestrator 分工，共享黑板通信 |
| 答案审核 | LLM 语义审核 + 规则审核双轨降级，检测空答案、无证据、引用越界等硬红线 |
| MCP 协议 | 通过标准 stdio 协议接入外部工具，让 Agent 能力可扩展 |
| 工程化能力 | 租户隔离、Prompt Injection 防护、结构化日志、Prometheus 指标、Pydantic 配置校验 |

---

## 核心能力

- **多模型适配** — 内置智谱 / DeepSeek / 通义千问 / 豆包及通用 OpenAI 兼容适配器，统一 `ChatModel` 协议
- **智能路由与降级** — 按优先级与错误类型自动切换备用模型，避免单点故障
- **混合检索** — 向量召回与 BM25 关键词召回互补，支持多种融合策略
- **结果重排** — Cross-Encoder 精排，提升 top-k 相关性
- **多角色编排** — 检索、推理、审核各司其职，职责边界通过契约约束
- **答案质量把关** — 语义审核与规则审核双轨，命中硬红线自动拦截
- **可观测** — 结构化日志 + Prometheus 指标，调用链可追踪
- **安全默认** — 租户隔离强制生效、Prompt Injection 默认拒绝、工具白名单默认开启

---

## 能力矩阵

项目提供三条产品路径，能力各有取舍。**没有任何一条路径同时具备多角色审核 / LangGraph / 工具调用**——这是演进中的设计取舍，不是缺陷。

| 能力 | `/v1/answer` | `/v1/agent` | `/v1/agent/stream` |
|---|---|---|---|
| 检索 + 生成 | ✅ | ✅ | ✅ |
| 多角色审核 | ❌ | **✅** | ❌ |
| LangGraph 状态机 | ❌ | ❌ | **✅** |
| 工具调用（含 MCP） | ❌ | ❌ | **✅** |
| 引用完整性（chunk_id） | ✅ | ⚠️ | ❌ |
| 审计轨迹 | ❌ | ✅ | ⚠️ 部分 |
| Fallback | ✅ | ✅ | ✅ |

- `/v1/answer`：追求简单快，单轮检索 + 生成，无审核
- `/v1/agent`：追求审核可靠，多角色协作 + 审核重试，无 LangGraph
- `/v1/agent/stream`：追求 Agent 能力，LangGraph + 工具调用，无审核

未来计划把审核节点接入 LangGraph 图，统一到一条路径。

---

## 架构

```text
┌───────────────────────────────────────────────────────┐
│ FastAPI Server                                        │
│ /v1/agent  /v1/answer  /v1/agent/stream              │
│ /v1/retrieve  /v1/ingest  /healthz  /readyz /metrics │
└────────────────────┬──────────────────────────────────┘
                     │
        ┌────────────┼────────────────┐
        ▼            ▼                ▼
┌──────────────┐ ┌────────────────┐ ┌──────────────┐
│ RAGService   │ │AgentOrchestrator│ │ AgentRunner  │
│ (单轮检索生成)│ │ (多角色审核)    │ │ (LangGraph)  │
└──────────────┘ └───────┬────────┘ └──────────────┘
                         │
              retrieve ──▶ reason ──▶ review ──▶ finalize
                │            │           │          （内联逻辑）
                ▼            ▼           ▼
            Retriever    Reasoner    Reviewer
            （黑板通信，每个角色有独立职责契约）

┌───────────────────────────────────────┐
│ 共享层                                  │
├───────────────────────────────────────┤
│ RetrievalPipeline │ FallbackExecutor   │
│ VectorStore (Chroma) │ ModelRouter     │
│ ToolRegistry                          │
└───────────────────────────────────────┘
```

### 多角色协作的黑板通信模型

```text
Blackboard:
├── original_question: str
├── tenant_id: str
├── evidence: list[RetrievedDocument]   ← Retriever 写入
├── answer_draft: str                   ← Reasoner 写入
├── review: ReviewReport                ← Reviewer 写入
├── answer: str                         ← finalize 节点写入最终答案
└── roles_invoked: list[str]            ← 编排器记录
```

每个角色有独立的 `RoleDefinition`，声明：

- **purpose** — 这个角色为什么存在
- **responsibilities** — 它必须做什么
- **must_not** — 它绝对不能做什么（防止越权）
- **output_contract** — 它必须返回什么格式

> `finalize` 是编排器内联的合成逻辑（`orchestrator._build_response`），不是独立角色。未来计划独立为 `Synthesizer` 角色。

---

## 快速开始

### 环境要求

| 项目 | 要求 |
|---|---|
| Python | `>=3.10, <3.15` |
| LLM | 一个可用的 API Key（支持 OpenAI 兼容协议即可） |
| 外部服务 | 无需额外部署——向量库默认 ChromaDB 本地持久化，开箱即用 |

> 本项目**不依赖** MongoDB、Redis 等外部存储。若你看到的文档提到这些，那是其他项目的配置，与本仓库无关。

### 安装

```bash
git clone https://github.com/Duckherewegoo/LangChain-ChatGLM-RAG-QA.git
cd LangChain-ChatGLM-RAG-QA

# 核心依赖（含 pymupdf / pypdf / python-docx / docx2txt 等文档解析依赖，一并安装）
pip install -e .

# 如需使用 MCP 工具源
pip install -e ".[mcp]"

# 开发依赖（测试、lint、类型检查）
pip install -e ".[dev]"
```

> 依赖统一由 `pyproject.toml` 管理，仓库中**不含** `requirements.txt`，请勿手动维护该文件。

### 配置

通过环境变量配置（最小可用集）：

```bash
export MODEL_CHAT_API_KEY="your-api-key"
export MODEL_CHAT_PROVIDER="zhipu"            # 或 deepseek / qwen / doubao / openai_compatible
export MODEL_CHAT_MODEL_NAME="glm-4-plus"
export MODEL_CHAT_BASE_URL="https://open.bigmodel.cn/api/paas/v4/"
```

检索参数、嵌入模型、Agent 预算、安全策略等完整配置项见 `config/settings.yaml`（仓库已提供，直接复制使用即可）。

> 敏感凭证（API Key 等）**禁止**提交到仓库。`.env`、`.env.local`、`config/.env` 已在 `.gitignore` 中排除，仅 `.env.example` 模板允许入库。
> 如果你选择用 `.env` 文件而不是 `export` 提供上述变量，请确保入口处已调用 `load_dotenv()` 加载该文件（如未加载，环境变量对进程不可见）。

### 启动

```bash
enterprise-rag
```

默认监听 `0.0.0.0:8000`。也可以直接运行：

```bash
python -m src.base.main
```

---

## 配置说明

| 变量 | 说明 | 默认值 / 备注 |
|---|---|---|
| `MODEL_CHAT_API_KEY` | LLM API 密钥 | **必填** |
| `MODEL_CHAT_PROVIDER` | 模型厂商 | `zhipu` / `deepseek` / `qwen` / `doubao` / `openai_compatible` |
| `MODEL_CHAT_MODEL_NAME` | 具体模型名 | 依厂商而定 |
| `MODEL_CHAT_BASE_URL` | API 接入地址 | OpenAI 兼容格式 |
| `RETRIEVAL_KEYWORD_ENABLED` | 是否开启 BM25 混合检索 | 默认 `false` |
| `RETRIEVAL_QUERY_REWRITE` | 查询改写策略 | `identity` / `expansion` / `llm` |
| 检索参数 | top-k、相似度阈值等 | 见 `config/settings.yaml` |
| 嵌入模型 | 向量化模型 | 见 `config/settings.yaml` |
| Agent 预算 | 最大步数、超时 | 见 `config/settings.yaml` |
| 安全策略 | 注入防护、白名单 | 见 `config/settings.yaml` |

> **注意**：`RETRIEVAL_FUSION_*` / `RETRIEVAL_KEYWORD_*` 的部分字段当前需通过 `RAG_` 前缀的环境变量配置（例如 `RAG_RETRIEVAL__FUSION__TYPE=rrf`）。这是已知限制，详见下方「已知限制」。

---

## 调用示例

### 可用端点

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/v1/answer` | 单轮检索 + 生成 |
| GET | `/v1/retrieve` | 纯检索（不生成） |
| POST | `/v1/agent` | 多角色编排（含审核） |
| POST | `/v1/agent/stream` | LangGraph 流式 Agent（SSE） |
| GET | `/v1/agent/{id}/trail` | 查询 Agent 审计轨迹（JSONL） |
| POST | `/v1/ingest` | 文档摄入 |
| DELETE | `/v1/documents/{id}` | 删除文档 |
| GET | `/healthz` | 存活探针 |
| GET | `/readyz` | 就绪探针（检查向量库） |
| GET | `/healthz/models` | 模型健康检查（真实 ping） |
| GET | `/metrics` | Prometheus 指标 |

### 示例

```bash
# Agent 对话（多角色审核）
curl -X POST http://localhost:8000/v1/agent \
  -H "Content-Type: application/json" \
  -d '{
    "question": "公司的年假政策是什么？",
    "tenant_id": "tenant_001"
  }'

# 单轮 RAG
curl -X POST http://localhost:8000/v1/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "公司的年假政策是什么？", "tenant_id": "tenant_001"}'

# 健康检查
curl http://localhost:8000/healthz
curl http://localhost:8000/healthz/models

# Prometheus 指标
curl http://localhost:8000/metrics
```

---

## 项目结构

```text
LangChain-ChatGLM-RAG-QA/
├── config/
│   └── settings.yaml            # 主配置文件（密钥走环境变量，不入仓）
├── data/
│   ├── index/                   # 向量索引存储（已排除，见 .gitignore）
│   └── raw/                     # 原始文档（已排除，见 .gitignore）
├── docs/
│   └── adr/                     # ADR 架构文档
├── src/
│   ├── base/                    # 入口、异常、可观测性
│   │   ├── main.py
│   │   ├── exceptions.py
│   │   ├── observability.py
│   │   └── redaction.py
│   ├── api/                     # FastAPI 层
│   │   ├── server.py
│   │   ├── handlers.py
│   │   ├── middleware.py
│   │   └── schemas.py
│   ├── model/                   # 模型抽象层
│   │   ├── protocol.py
│   │   ├── manager.py
│   │   ├── router.py
│   │   ├── fallback.py
│   │   ├── health.py
│   │   ├── errors.py
│   │   ├── registry.py
│   │   ├── routing_types.py
│   │   ├── embedding.py
│   │   └── providers/
│   ├── rag/                     # RAG 核心
│   │   ├── agent.py
│   │   ├── agent_events.py
│   │   ├── orchestrator.py
│   │   ├── retrieval.py
│   │   ├── reranker.py
│   │   ├── reviewer.py
│   │   ├── roles.py
│   │   ├── container.py
│   │   ├── service.py
│   │   ├── tools.py
│   │   ├── tool_types.py
│   │   ├── types.py
│   │   ├── vector_store.py
│   │   ├── collection_policy.py
│   │   ├── strategies/
│   │   └── tool_sources/
│   ├── data_process/            # 文档摄入管线
│   └── settings/                # 配置管理
├── tests/
│   ├── test_agent_harness.py
│   ├── test_model_routing.py
│   ├── test_orchestrator_e2e.py
│   ├── test_pipeline.py
│   └── ab_test/                 # 策略对比测试
├── pyproject.toml
├── LICENSE
└── README.md
```


### 源码一览

| 层次 | 关键文件 | 职责 |
|---|---|---|
| 入口 | `src/base/main.py` | 启动 FastAPI 服务 |
| 横切 | `src/base/exceptions.py` | 异常体系（RAGError + 子类） |
| 横切 | `src/base/observability.py` | 日志 + 指标 + 请求上下文 |
| 横切 | `src/base/redaction.py` | 敏感信息脱敏（递归 + 子串匹配） |
| API | `src/api/server.py` | 应用工厂与路由注册 |
| API | `src/api/handlers.py` | 路由处理函数 |
| API | `src/api/middleware.py` | 全局异常处理 |
| API | `src/api/schemas.py` | 请求/响应模型 |
| 模型抽象 | `src/model/protocol.py` | `ChatModel` Protocol |
| 模型抽象 | `src/model/router.py` | 多模型路由 |
| 模型抽象 | `src/model/fallback.py` | Fallback 执行 |
| 模型抽象 | `src/model/health.py` | 健康检查 |
| 模型抽象 | `src/model/errors.py` | 错误分类 |
| 模型抽象 | `src/model/embedding.py` | 嵌入模型 |
| 厂商适配 | `src/model/providers/*.py` | 智谱 / DeepSeek / 通义 / 豆包 / 通用兼容 |
| RAG 核心 | `src/rag/agent.py` | LangGraph Harness |
| RAG 核心 | `src/rag/agent_events.py` | Agent 事件与审计轨迹 |
| RAG 核心 | `src/rag/orchestrator.py` | 多角色编排 |
| RAG 核心 | `src/rag/retrieval.py` | 检索管线 |
| RAG 核心 | `src/rag/reranker.py` | Cross-Encoder 重排 |
| RAG 核心 | `src/rag/reviewer.py` | 答案审核 |
| RAG 核心 | `src/rag/roles.py` | 角色定义与黑板 |
| RAG 核心 | `src/rag/container.py` | 依赖注入容器 |
| RAG 核心 | `src/rag/service.py` | 单轮 RAG 服务门面 |
| RAG 核心 | `src/rag/tools.py` | 工具注册与护栏 |
| RAG 核心 | `src/rag/vector_store.py` | 向量存储（ChromaDB） |
| RAG 核心 | `src/rag/collection_policy.py` | 集合隔离与指纹治理 |
| 检索策略 | `src/rag/strategies/*.py` | 查询改写 / 融合 / 混合检索 |
| 工具源 | `src/rag/tool_sources/*.py` | 本地工具 + MCP 工具 |
| 文档摄入 | `src/data_process/*.py` | 加载 / 分块 / 摄入管线 |
| 配置 | `src/settings/settings.py` | 配置管理（YAML + 环境变量） |

---

## 测试

```bash
# 全量测试
pytest tests/ -v

# 按模块
pytest tests/test_orchestrator_e2e.py -v
pytest tests/test_pipeline.py -v
```

配置来自 `pyproject.toml`：`asyncio_mode = "auto"`，`testpaths = ["tests"]`，`addopts = "-ra --strict-markers"`，`pythonpath = ["src"]`。

测试策略：

- 所有测试使用**真实组件**，不 mock 内部逻辑
- LLM 调用使用可控的 `FakeModel`（实现 `ChatModel` 协议，走真实代码路径）
- 向量存储使用内存实现，避免测试依赖外部服务
- 覆盖模型路由、Fallback 降级、文档处理、检索隔离、工具白名单、预算守卫、Agent 审计、多角色编排等场景
- **当前缺口**：三条产品路径的端到端冒烟测试不完整。`/v1/answer` 的 `max_history_messages` 字段错误曾长期存活，说明该路径缺少端到端覆盖。这是本项目已知的质量债。

---

## 代码规范

配置来自 `pyproject.toml`：行宽 **130** 字符，目标版本 **py310**，`data/`、`build/`、`dist/` 已排除。

```bash
# ruff 检查
ruff check src/

# ruff 自动修复
ruff check src/ --fix

# 类型检查
pyright src/
```

> 仓库的 dev 依赖中不含 `mypy`；如需本地运行，请自行安装后再执行 `mypy src/`。

代码风格遵循 PEP 8，采用 Python 3.10+ 语法（`X | None`、`list[X]`）。`chromadb`、`sentence_transformers`、`mcp` 的第三方导入已在 `pyproject.toml` 中配置 `ignore_missing_imports`。

---

## 设计哲学

1. **依赖倒置** — 业务层只依赖 `ChatModel` Protocol，不耦合具体厂商 SDK
2. **开闭原则** — 新增模型 = 写一个 Adapter + 注册一行，不改调用方
3. **故障透明** — 所有异常（Python 内置 / 自定义 / SDK）都有明确映射和兜底策略
4. **审计可追溯** — Agent 的每次决策、工具调用、模型调用全部进入审计轨迹
5. **安全默认** — 租户隔离强制生效、Prompt Injection 默认拒绝、工具白名单默认开启
6. **配置即代码** — 行为参数全部可配置，支持环境变量覆盖，无需改代码

---

## Vibe-coding Workflow

本项目由 AI 和我协作完成：我负责核心思想、架构决策、人工审查，AI 负责大段代码实现。这个过程让我总结出 AI 辅助编码的三类典型错误，以及对应的审查方法。

### 错误类型 1：单文件正确，跨文件错位

AI 写每个函数都正确，但组合起来错。典型案例（均已修复，保留作方法论示例）：

- `ChatRequest` 有 `tools` 字段，`_payload` 却读 `self._bound_tools`——Agent 的 tool calling 从未传给模型；
- `ToolSpec` 是纯数据模型，`_wrap_spec` 却读 `spec.source`——每次工具调用都 `AttributeError`；
- `fingerprint_embedder` 校验 dimension，`create_embedding_model` 却懒加载——`/readyz` 启动必崩；
- `compute_document_id` 的 docstring 说"无论是否有 content 必须同一个"，实现却内容敏感——`/v1/ingest` 返回的 ID 用 DELETE 删 0 条；
- `retrieve_knowledge` 的 `allowed_sources` 用并集而非交集——模型可以扩大权限；
- `retrieve_knowledge` 的 `tenant_id` / `owner_id` 用 `setdefault`——模型可以跨租户访问。

**审查方法**：不逐文件读，逐链路对照契约。画一条"输入 → 中间态 → 输出"的数据流，每个跳点检查"上游给了什么，下游假设什么"。

### 错误类型 2：注释承诺 > 实现

AI 写注释时倾向"描述意图"，实现时可能没跟上。至少 6 处 docstring 描述的断言与实现不符：

- `QueryRewrite.changed()` docstring 说"未产生扩展时视为未改变"，实现却 `return self.method != "identity"`；
- `reviewer` docstring 说"解析失败降级规则审核"，实现却不降级；
- `collection_policy._safe` 注释说"截断保留唯一后缀"，实现却 off-by-one 输出 64 字符。

**审查方法**：每个 docstring 都是一条断言。读代码时对照 docstring，不一致就是 bug（或注释要改）。

### 错误类型 3：配置声明 > 消费

AI 写 schema 时倾向"把可能用到的都写上"，handler 实现时只接一部分。11 个字段声明但未消费：

- `AnswerRequest.model` / `use_case` / `additional_context`；
- `AgentRequest.use_case` / `model` / `use_model_fallback`；
- `Settings.score_threshold` / `hybrid_weight` / `rerank.top_k` / `keyword.top_k`；
- `RetrievedContext.truncated`。

**审查方法**：每个 `Field(...)` 都要有消费点。用 `grep -rn "field_name"` 确认至少有一处读它，否则删掉。

### 我从这个项目学到的方法

- **Import smoke test 是 CI 第一步**：批量重构后立刻跑 `python -c "import 所有模块"`，比单元测试更早暴露断裂；
- **跨文件契约用"链路图"审，不用"文件列表"审**：`request.tools` 被忽略这类问题，逐文件永远发现不了；
- **主动写 Known Gaps**：列出来的短板比藏起来的 bug 更值钱——它证明"你知道边界在哪"。

---

## 已知限制

### 架构演进中

三条产品路径能力不齐（见[能力矩阵](#能力矩阵)）。没有任何一条路径同时具备多角色审核 / LangGraph / 工具调用。未来计划把审核节点接入 LangGraph 图，统一到一条路径。

### 能力未完成

- **混合检索**：BM25 索引已接写入链路，但删除 / 幂等未接；默认 `RETRIEVAL_KEYWORD_ENABLED=false`，需配置开启；
- **MCP 超时**：`MCPToolSource.timeout_seconds` 声明但 `run_coro` 未使用，MCP 服务端挂死时请求会永久阻塞；
- **健康检查**：`/healthz/models` 依赖 `model.bind_tools` 的当前实现，可能修改 model 实例状态，建议修复方向是返回新实例而非 `self`；
- **字段消费**：11 个字段声明但未消费（见上文 [Vibe-coding Workflow](#vibe-coding-workflow) 错误类型 3）。

### 配置契约漂移

- **异常类双重定义**：`ConfigurationError` / `CollectionCompatibilityError` / `ToolExecutionError` 各有两个定义（`settings` / `collection_policy` / `tool_sources` vs `base/exceptions.py`）。这是重构遗留，待统一；
- **max_retries 双源**：`model_chat.max_retries` 和 `model_router.models.*.max_retries` 各管一段，容器只读前者；
- **ProviderHealth.provider 语义**：存的是配置 key（如 `"primary"`），不是 provider 名（如 `"zhipu"`）；
- **attempted_models 语义**：部分路径返回全量配置列表，不是"本次尝试过的模型"；
- **RETRIEVAL_FUSION_\* 环境变量**：`_normalize_env_config` 未对 `fusion_*` / `keyword_*` 做二级嵌套，需用 `RAG_RETRIEVAL__FUSION__TYPE` 形式配置。

### 安全与测试

- **无鉴权**：个人展示项目，`/v1/ingest` 和 `/v1/documents` 无认证，`body.path` 无路径白名单。生产部署必须加 API key + `allowed_roots` 路径限制；
- **摄入健壮性**：`.jsonl` 文件加载路径使用 `json.load`，对 JSONL 格式会失败；`_load_text` 的 `OSError` 未转 `DocumentLoadError`，单个权限失败可能阻断批量；
- **PDF 页码**：当前 PDF 解析不保留页码信息到 chunk 级，引用只到文档级；
- **测试缺口**：三条路径的端到端冒烟测试不完整。

---

## License

本项目以 **GNU GPL v3.0 或更高版本（GPL-3.0-or-later）** 发布，详见 [LICENSE](LICENSE)。

- 使用、复制、修改、分发本软件均须遵守 GPL v3 条款
- 修改后的源码在分发时**必须**以相同许可证开源
- 本软件按"原样"提供，**不附带任何担保**，作者不对使用后果承担责任
- 完整的权限、条件与免责声明以 [LICENSE](LICENSE) 文件为准
