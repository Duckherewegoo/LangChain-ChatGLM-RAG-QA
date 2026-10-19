# 企业级 RAG 智能问答系统

> 基于 **LangChain + LangGraph** 构建的检索增强生成（RAG）服务，覆盖从文档摄入、混合检索、多模型路由到多角色协作审核的完整链路。
> 重点在**架构设计与可靠性工程**，而非模型调参。

**版本** `0.2.0` ｜ **Python** `>=3.10, <3.15` ｜ **许可证** [GPL-3.0-or-later](LICENSE)

______________________________________________________________________

## 目录

- [这个项目解决什么问题](#%E8%BF%99%E4%B8%AA%E9%A1%B9%E7%9B%AE%E8%A7%A3%E5%86%B3%E4%BB%80%E4%B9%88%E9%97%AE%E9%A2%98)
- [核心能力](#%E6%A0%B8%E5%BF%83%E8%83%BD%E5%8A%9B)
- [架构](#%E6%9E%B6%E6%9E%84)
- [快速开始](#%E5%BF%AB%E9%80%9F%E5%BC%80%E5%A7%8B)
- [配置说明](#%E9%85%8D%E7%BD%AE%E8%AF%B4%E6%98%8E)
- [调用示例](#%E8%B0%83%E7%94%A8%E7%A4%BA%E4%BE%8B)
- [项目结构](#%E9%A1%B9%E7%9B%AE%E7%BB%93%E6%9E%84)
- [测试](#%E6%B5%8B%E8%AF%95)
- [代码规范](#%E4%BB%A3%E7%A0%81%E8%A7%84%E8%8C%83)
- [设计哲学](#%E8%AE%BE%E8%AE%A1%E5%93%B2%E5%AD%A6)
- [已知限制](#%E5%B7%B2%E7%9F%A5%E9%99%90%E5%88%B6)
- [License](#license)

______________________________________________________________________

## 这个项目解决什么问题

不是又一个"调个 API 拼 prompt"的 Demo。它的核心卖点是**把 RAG 系统当成一个真正的工程产品来做**。

| 模块 | 你能学到的核心知识 |
|---|---|
| --- | --- |
| 多模型抽象层 | 用 `Protocol` + 工厂模式屏蔽 ChatGLM / DeepSeek / Qwen / 豆包等厂商差异，新增模型零改动接入 |
| 模型路由 + Fallback | 主备模型链、指数退避重试、错误分类（可重试 vs 永久失败） |
| LangGraph Agent Harness | 用状态机实现 `reason → tool → reason` 循环，含预算守卫和审计轨迹 |
| Advanced RAG | 查询改写、BM25 + 向量混合检索、多策略融合、Cross-Encoder 重排 |
| 多角色协作 | Retriever / Reasoner / Reviewer / Orchestrator 分工，共享黑板通信 |
| 答案审核 | LLM 语义审核 + 规则审核双轨降级，检测空答案、无证据、引用越界等硬红线 |
| MCP 协议 | 通过标准 stdio 协议接入外部工具，让 Agent 能力可扩展 |
| 工程化能力 | 租户隔离、Prompt Injection 防护、结构化日志、Prometheus 指标、Pydantic 配置校验 |

______________________________________________________________________

## 核心能力

- **多模型适配** — 内置智谱 / DeepSeek / 通义千问 / 豆包及通用 OpenAI 兼容适配器，统一 `ChatModel` 协议
- **智能路由与降级** — 按优先级与错误类型自动切换备用模型，避免单点故障
- **混合检索** — 向量召回与 BM25 关键词召回互补，支持多种融合策略
- **结果重排** — Cross-Encoder 精排，提升 top-k 相关性
- **多角色编排** — 检索、推理、审核、合成各司其职，职责边界通过契约约束
- **答案质量把关** — 语义审核与规则审核双轨，命中硬红线自动拦截
- **可观测** — 结构化日志 + Prometheus 指标，调用链可追踪
- **安全默认** — 租户隔离强制生效、Prompt Injection 默认拒绝、工具白名单默认开启

______________________________________________________________________

## 架构

```
                    ┌──────────────────────────────────────────────────┐
                    │               FastAPI Server                     │
                    │   /v1/agent  /v1/answer  /v1/ingest  /health    │
                    └────────────────────┬─────────────────────────────┘
                                         │
                    ┌────────────────────▼─────────────────────────────┐
                    │              AgentOrchestrator                    │
                    │                                                  │
                    │    retrieve ──▶ reason ──▶ review ──▶ finalize   │
                    │       │            │           │           │      │
                    │       ▼            ▼           ▼           ▼      │
                    │   Retriever    Reasoner    Reviewer  Synthesizer  │
                    │   （黑板通信，每个角色有独立职责契约）             │
                    └─────┬───────────┬───────────┬────────────────────┘
                          │           │           │
             ┌────────────▼──┐  ┌─────▼─────┐  ┌──▼────────────┐
             │  检索管线       │  │ 模型执行    │  │ 工具系统        │
             │  • 向量召回     │  │ • 路由      │  │ • 本地工具      │
             │  • BM25         │  │ • Fallback  │  │ • MCP 工具      │
             │  • 融合         │  │ • Tool Call │  │ • 白名单        │
             │  • Cross-Encoder│  │ • 预算守卫   │  │ • 租户隔离      │
             │  • 查询改写     │  │            │  │ • 审计          │
             └────────────────┘  └───────────┘  └────────────────┘
```

### 多角色协作的黑板通信模型

```
Blackboard:
  ├── question: str
  ├── tenant_id: str
  ├── evidence: list[RetrievedDocument]   ← Retriever 写入
  ├── draft_answer: str                   ← Reasoner 写入
  ├── review_report: ReviewReport         ← Reviewer 写入
  ├── final_answer: str                   ← Synthesizer 写入
  └── roles_invoked: list[str]            ← 编排器记录
```

每个角色有独立的 `RoleDefinition`，声明：

- **purpose** — 这个角色为什么存在
- **responsibilities** — 它必须做什么
- **must_not** — 它绝对不能做什么（防止越权）
- **output_contract** — 它必须返回什么格式

______________________________________________________________________

## 快速开始

### 环境要求

| 项目 | 要求 |
|---|---|
| --- | --- |
| Python | `>=3.10, <3.15` |
| LLM | 一个可用的 API Key（支持 OpenAI 兼容协议即可） |
| 外部服务 | 无需额外部署——向量库默认 ChromaDB 本地持久化，开箱即用 |

> 本项目**不依赖** MongoDB、Redis 等外部存储。若你看到的文档提到这些，那是其他项目的配置，与本仓库无关。

### 安装

```bash
git clone https://github.com/Duckherewegoo/enterprise-rag-chatglm.git
cd enterprise-rag-chatglm

# 核心依赖
pip install -e .

# 如需使用 MCP 工具源
pip install -e ".[mcp]"

# 开发依赖（测试、lint、类型检查）
pip install -e ".[dev]"
```

> `requirements.txt` 不在仓库中，依赖统一由 `pyproject.toml` 管理，请勿手动维护 `requirements.txt`。

### 配置

通过环境变量配置（最小可用集）：

```bash
export MODEL_CHAT_API_KEY="your-api-key"
export MODEL_CHAT_PROVIDER="zhipu"            # 或 deepseek / qwen / doubao / openai_compatible
export MODEL_CHAT_MODEL_NAME="glm-4-plus"
export MODEL_CHAT_BASE_URL="https://open.bigmodel.cn/api/paas/v4/"
```

检索参数、嵌入模型、Agent 预算、安全策略等完整配置项见 `config/settings.yaml`。

> 敏感凭证（API Key 等）**禁止**提交到仓库。`.env`、`.env.local`、`config/.env` 已在 `.gitignore` 中排除，仅 `.env.example` 模板允许入库。

### 启动

```bash
python -m src.main
```

默认监听 `0.0.0.0:8000`。

______________________________________________________________________

## 配置说明

| 变量 | 说明 | 默认值 / 备注 |
|---|---|---|
| --- | --- | --- |
| `MODEL_CHAT_API_KEY` | LLM API 密钥 | **必填** |
| `MODEL_CHAT_PROVIDER` | 模型厂商 | `zhipu` / `deepseek` / `qwen` / `doubao` / `openai_compatible` |
| `MODEL_CHAT_MODEL_NAME` | 具体模型名 | 依厂商而定 |
| `MODEL_CHAT_BASE_URL` | API 接入地址 | OpenAI 兼容格式 |
| 检索参数 | top-k、相似度阈值等 | 见 `config/settings.yaml` |
| 嵌入模型 | 向量化模型 | 见 `config/settings.yaml` |
| Agent 预算 | 最大步数、超时 | 见 `config/settings.yaml` |
| 安全策略 | 注入防护、白名单 | 见 `config/settings.yaml` |

______________________________________________________________________

## 调用示例

```bash
# Agent 对话（返回完整审计轨迹）
curl -X POST http://localhost:8000/v1/agent \
  -H "Content-Type: application/json" \
  -d '{
    "question": "公司的年假政策是什么？",
    "tenant_id": "tenant_001",
    "session_id": "sess_001"
  }'

# 健康检查
curl http://localhost:8000/v1/health

# Prometheus 指标
curl http://localhost:8000/metrics
```

______________________________________________________________________

## 项目结构

```
enterprise-rag-chatglm/
├── config/
│   └── settings.yaml            # 主配置文件（密钥走环境变量，不入仓）
├── data/
│   ├── index/                   # 向量索引存储（已排除，见 .gitignore）
│   └── raw/                    # 原始文档（已排除，见 .gitignore）
├── docs/                       # 补充文档
├── src/
│   ├── main.py                 # 入口
│   ├── api/                    # FastAPI 层
│   │   ├── server.py           # 应用工厂 + 路由注册
│   │   ├── handlers.py         # 路由处理函数
│   │   ├── middleware.py       # 全局异常处理
│   │   └── schemas.py          # 请求/响应模型
│   ├── model/                  # 模型抽象层
│   │   ├── protocol.py         # ChatModel Protocol 定义
│   │   ├── manager.py          # 模型管理器 + 工厂
│   │   ├── router.py           # 多模型路由
│   │   ├── fallback.py         # Fallback 执行器
│   │   ├── health.py           # 模型健康检查
│   │   ├── errors.py           # 错误分类器
│   │   ├── registry.py         # Provider 注册表
│   │   ├── routing_types.py    # 路由配置数据类型
│   │   └── providers/          # 各厂商适配器
│   ├── rag/                    # RAG 核心
│   │   ├── agent.py            # LangGraph Harness
│   │   ├── agent_events.py     # Agent 事件定义
│   │   ├── orchestrator.py     # 多角色编排器
│   │   ├── retrieval.py        # 检索管线
│   │   ├── reranker.py         # Cross-Encoder 重排
│   │   ├── reviewer.py         # 答案审核器
│   │   ├── roles.py            # 角色定义
│   │   ├── container.py        # 依赖注入容器
│   │   ├── service.py          # RAG 服务门面
│   │   ├── tools.py            # 工具注册与执行
│   │   ├── tool_types.py       # 工具类型定义
│   │   ├── types.py            # 领域类型
│   │   ├── vector_store.py     # 向量存储封装
│   │   ├── collection_policy.py# 集合/租户隔离策略
│   │   ├── strategies/         # 检索策略（查询改写 / 融合 / 混合）
│   │   └── tool_sources/       # 本地工具源与 MCP 工具源
│   ├── data_process/           # 文档摄入管线（加载 / 分块 / 类型）
│   ├── settings/               # 配置管理
│   ├── model/embedding.py      # 嵌入模型封装
│   ├── observability.py        # 日志 + 指标
│   └── exceptions.py           # 异常体系
├── tests/                      # 测试套件
│   ├── ab_test/               # 策略对比测试
│   └── ...
├── pyproject.toml
├── LICENSE
└── README.md
```

> 目录树已按 `.gitignore` 的排除规则精简：`__pycache__/`、`data/index/*`、`data/raw/*`、`.env`、测试缓存等不再列出。被忽略的目录仅保留结构说明，其下实际内容不进入版本库。

### 源码一览

| 层次 | 关键文件 | 职责 |
|---|---|---|
| --- | --- | --- |
| 入口 | `src/main.py` | 启动 FastAPI 服务 |
| API | `src/api/server.py` | 应用工厂与路由注册 |
| API | `src/api/handlers.py` | 路由处理函数 |
| API | `src/api/middleware.py` | 全局异常处理 |
| API | `src/api/schemas.py` | 请求/响应模型 |
| 模型抽象 | `src/model/protocol.py` | `ChatModel` Protocol |
| 模型抽象 | `src/model/manager.py` | 模型管理器 |
| 模型抽象 | `src/model/router.py` | 多模型路由 |
| 模型抽象 | `src/model/fallback.py` | Fallback 执行 |
| 模型抽象 | `src/model/health.py` | 健康检查 |
| 模型抽象 | `src/model/errors.py` | 错误分类 |
| 模型抽象 | `src/model/registry.py` | Provider 注册 |
| 模型抽象 | `src/model/routing_types.py` | 路由数据类型 |
| 模型抽象 | `src/model/embedding.py` | 嵌入模型 |
| 厂商适配 | `src/model/providers/zhipu.py` | 智谱 |
| 厂商适配 | `src/model/providers/deepseek.py` | DeepSeek |
| 厂商适配 | `src/model/providers/qwen.py` | 通义千问 |
| 厂商适配 | `src/model/providers/doubao.py` | 豆包 |
| 厂商适配 | `src/model/providers/openai_compatible.py` | 通用兼容 |
| RAG 核心 | `src/rag/agent.py` | LangGraph Harness |
| RAG 核心 | `src/rag/agent_events.py` | Agent 事件 |
| RAG 核心 | `src/rag/orchestrator.py` | 多角色编排 |
| RAG 核心 | `src/rag/retrieval.py` | 检索管线 |
| RAG 核心 | `src/rag/reranker.py` | Cross-Encoder 重排 |
| RAG 核心 | `src/rag/reviewer.py` | 答案审核 |
| RAG 核心 | `src/rag/roles.py` | 角色定义 |
| RAG 核心 | `src/rag/container.py` | 依赖注入 |
| RAG 核心 | `src/rag/service.py` | 服务门面 |
| RAG 核心 | `src/rag/tools.py` | 工具注册执行 |
| RAG 核心 | `src/rag/tool_types.py` | 工具类型 |
| RAG 核心 | `src/rag/types.py` | 领域类型 |
| RAG 核心 | `src/rag/vector_store.py` | 向量存储 |
| RAG 核心 | `src/rag/collection_policy.py` | 集合隔离策略 |
| 检索策略 | `src/rag/strategies/factory.py` | 策略工厂 |
| 检索策略 | `src/rag/strategies/hybrid.py` | 混合检索 |
| 检索策略 | `src/rag/strategies/fusers.py` | 融合策略 |
| 检索策略 | `src/rag/strategies/retrievers.py` | 检索器 |
| 检索策略 | `src/rag/strategies/rewriters.py` | 查询改写 |
| 检索策略 | `src/rag/strategies/types.py` | 策略类型 |
| 工具源 | `src/rag/tool_sources/base.py` | 工具源基类 |
| 工具源 | `src/rag/tool_sources/local.py` | 本地工具 |
| 工具源 | `src/rag/tool_sources/mcp.py` | MCP 工具 |
| 工具源 | `src/rag/tool_sources/factory.py` | 工具源工厂 |
| 文档摄入 | `src/data_process/loaders.py` | 文档加载 |
| 文档摄入 | `src/data_process/chunker.py` | 文本分块 |
| 文档摄入 | `src/data_process/pipeline.py` | 摄入管线 |
| 文档摄入 | `src/data_process/types.py` | 数据类型 |
| 配置 | `src/settings/settings.py` | 配置管理 |
| 横切 | `src/observability.py` | 日志与指标 |
| 横切 | `src/exceptions.py` | 异常体系 |
| 测试 | `tests/test_agent_harness.py` | Agent 状态机测试 |
| 测试 | `tests/test_model_routing.py` | 模型路由测试 |
| 测试 | `tests/test_orchestrator_e2e.py` | 编排端到端测试 |
| 测试 | `tests/test_pipeline.py` | 摄入管线测试 |
| 测试 | `tests/ab_test/` | 策略对比测试 |

______________________________________________________________________

## 测试

```bash
# 全量测试
pytest tests/ -v

# 按模块
pytest tests/test_orchestrator.py -v
pytest tests/test_pipeline.py -v
```

配置来自 `pyproject.toml`：`asyncio_mode = "auto"`，`testpaths = ["tests"]`，`addopts = "-ra --strict-markers"`，`pythonpath = ["."]`。

测试策略：

- 所有测试使用**真实组件**，不 mock 内部逻辑
- LLM 调用使用可控的 `FakeModel`（实现 `ChatModel` 协议，走真实代码路径）
- 向量存储使用内存实现，避免测试依赖外部服务
- 覆盖模型路由、Fallback 降级、文档处理、检索隔离、工具白名单、预算守卫、Agent 审计、多角色编排等场景

______________________________________________________________________

## 代码规范

配置来自 `pyproject.toml`：行宽 **130** 字符，目标版本 **py310**，`data/`、`build/`、`dist/`、`tests` 已排除。

```bash
# ruff 检查
ruff check src/

# ruff 自动修复
ruff check src/ --fix

# 类型检查
mypy src/
```

代码风格遵循 PEP 8，采用 Python 3.10+ 语法（`X | None`、`list[X]`）。`chromadb`、`sentence_transformers`、`mcp` 的第三方导入已在 `pyproject.toml` 中配置 `ignore_missing_imports`。

______________________________________________________________________

## 设计哲学

1. **依赖倒置** — 业务层只依赖 `ChatModel` Protocol，不耦合具体厂商 SDK
1. **开闭原则** — 新增模型 = 写一个 Adapter + 注册一行，不改调用方
1. **故障透明** — 所有异常（Python 内置 / 自定义 / SDK）都有明确映射和兜底策略
1. **审计可追溯** — Agent 的每次决策、工具调用、模型调用全部进入审计轨迹
1. **安全默认** — 租户隔离强制生效、Prompt Injection 默认拒绝、工具白名单默认开启
1. **配置即代码** — 行为参数全部可配置，支持环境变量覆盖，无需改代码

______________________________________________________________________

## 已知限制

- **版本标注为 `0.2.0`**，尚未达到 `1.0` 稳定版，API 与配置项在后续版本中可能变动
- **MCP 工具源**为可选功能，需额外安装 `.[mcp]` 依赖组
- 文档摄入依赖 `pypdf` 与 `python-docx`，处理扫描件或复杂排版文档时效果有限
- 答案审核的语义红线依赖 LLM 判断，存在误判可能，规则审核作为兜底

______________________________________________________________________

## License

本项目以 **GNU GPL v3.0 或更高版本（GPL-3.0-or-later）** 发布，详见 [LICENSE](LICENSE)。

- 使用、复制、修改、分发本软件均须遵守 GPL v3 条款
- 修改后的源码在分发时**必须**以相同许可证开源
- 本软件按"原样"提供，**不附带任何担保**，作者不对使用后果承担责任
- 完整的权限、条件与免责声明以 [LICENSE](LICENSE) 文件为准
