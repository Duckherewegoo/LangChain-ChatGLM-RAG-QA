# 企业级 RAG 智能问答系统

> 基于 LangChain + LangGraph + ChatGLM 构建的检索增强生成（RAG）Agent 实战项目，覆盖从文档摄入到多角色协作审核的完整链路。

## 这个项目能让你学到什么

不是又一个"调个 API 拼 prompt"的 Demo。这个项目的核心卖点是**把 RAG 系统当成一个真正的工程产品来做**，重点在架构设计和可靠性工程，而不是模型调参。

| 模块 | 你学到的核心知识 |
|---|---|
| 多模型抽象层 | 如何用 Protocol + 工厂模式屏蔽 ChatGLM/DeepSeek/Qwen 等厂商差异，新增模型零改动接入 |
| 模型路由 + Fallback | 如何做主备模型链、指数退避重试、错误分类（可重试 vs 永久失败） |
| LangGraph Agent Harness | 如何用状态机实现 `reason → tool → reason` 循环，含预算守卫和审计轨迹 |
| Advanced RAG | 查询改写、BM25 + 向量混合检索、RRF/加权/去重三套融合策略、Cross-Encoder 重排 |
| 多角色协作 | Retriever / Reasoner / Reviewer / Orchestrator 如何分工协作，共享黑板通信 |
| 答案审核 | LLM 语义审核 + 规则审核双轨降级，检测空答案、无证据、引用越界等硬红线 |
| MCP 协议 | 如何通过标准 stdio 协议接入外部工具，让 Agent 能力可扩展 |
| 工程化能力 | 租户隔离、Prompt Injection 防护、结构化日志、Prometheus 指标、Pydantic 配置校验 |

## 架构

```
                    ┌──────────────────────────────────────────────┐
                    │              FastAPI Server                   │
                    │  /v1/agent  /v1/answer  /v1/ingest  /health  │
                    └──────────────────┬───────────────────────────┘
                                       │
                    ┌──────────────────▼───────────────────────────┐
                    │           AgentOrchestrator                   │
                    │                                              │
                    │   retrieve ──▶ reason ──▶ review ──▶ finalize │
                    │      │           │          │           │     │
                    │      ▼           ▼          ▼           ▼     │
                    │  Retriever  Reasoner  Reviewer  Synthesizer  │
                    │  (黑板通信，每个角色有独立职责契约)            │
                    └────┬──────────┬──────────┬───────────────────┘
                         │          │          │
              ┌──────────▼──┐  ┌────▼─────┐  ┌──▼──────────┐
              │ 检索管线      │  │ 模型执行   │  │ 工具系统      │
              │ • 向量召回    │  │ • 路由      │  │ • 本地工具    │
              │ • BM25        │  │ • Fallback  │  │ • MCP 工具    │
              │ • RRF 融合    │  │ • Tool Call │  │ • 白名单      │
              │ • Cross-Enc   │  │ • 预算守卫   │  │ • 租户隔离    │
              │ • 查询改写    │  │            │  │ • 审计        │
              └─────────────┘  └──────────┘  └─────────────┘
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
- `purpose`：这个角色为什么存在
- `responsibilities`：它必须做什么
- `must_not`：它绝对不能做什么（防止越权）
- `output_contract`：它必须返回什么格式

## 快速开始

### 环境要求

- Python 3.10+
- 一个可用的 LLM API Key（支持 OpenAI 兼容协议即可）

### 安装

```bash
git clone <repo-url>
cd enterprise-rag-chatglm

pip install -r requirements.txt
```

### 配置

复制配置模板并填入你的 API Key：

```bash
cp config/.env.example config/.env
# 编辑 config/.env，填入 MODEL_CHAT_API_KEY 等
```

或直接通过环境变量配置：

```bash
export MODEL_CHAT_API_KEY="your-api-key"
export MODEL_CHAT_PROVIDER="zhipu"          # 或 deepseek / qwen / openai
export MODEL_CHAT_MODEL_NAME="glm-4-plus"
export MODEL_CHAT_BASE_URL="https://open.bigmodel.cn/api/paas/v4/"
```

完整配置项见 `config/settings.yaml`，包含检索参数、嵌入模型、Agent 预算、安全策略等。

### 启动

```bash
python -m src.main
```

默认监听 `0.0.0.0:8000`。

### 调用示例

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

## 测试

```bash
# 全量测试
pytest tests/ -v

# 按模块
pytest tests/test_orchestrator.py -v
pytest tests/test_pipeline.py -v
```

测试策略说明：
- 所有测试使用**真实组件**，不 mock 内部逻辑
- LLM 调用使用可控的 `FakeModel`（实现 `ChatModel` 协议，走真实代码路径）
- 向量存储使用内存实现，避免测试依赖外部服务
- 覆盖模型路由、Fallback 降级、文档处理、检索隔离、工具白名单、预算守卫、Agent 审计、多角色编排等场景

## 代码规范

```bash
# ruff 检查
ruff check src/

# ruff 自动修复
ruff check src/ --fix
```

代码风格遵循 PEP 8，行宽 130 字符，Python 3.10+ 语法（`X | None`、`list[X]`）。

## 项目结构

```
enterprise-rag-chatglm/
├── config/
│   ├── settings.yaml       # 主配置文件
│   └── .env.example        # 环境变量模板
├── src/
│   ├── main.py             # 入口
│   ├── api/                # FastAPI 层
│   │   ├── server.py       # 应用工厂 + 路由注册
│   │   ├── handlers.py     # 路由处理函数
│   │   ├── middleware.py   # 全局异常处理
│   │   └── schemas.py      # 请求/响应模型
│   ├── model/              # 模型抽象层
│   │   ├── manager.py      # ChatModel Protocol + 工厂
│   │   ├── router.py       # 多模型路由
│   │   ├── fallback.py     # Fallback 执行器
│   │   ├── health.py       # 模型健康检查
│   │   ├── errors.py       # 错误分类器
│   │   ├── registry.py     # Provider 注册表
│   │   ├── routing_types.py# 路由配置数据类型
│   │   └── providers/      # 各厂商适配器
│   ├── rag/                # RAG 核心
│   │   ├── agent.py        # LangGraph Harness
│   │   ├── orchestrator.py # 多角色编排器
│   │   ├── strategies.py   # 查询改写 + 混合检索 + 融合
│   │   ├── tools.py        # 工具注册与执行
│   │   ├── tool_sources/   # 本地/MCP 工具源
│   │   ├── reviewer.py     # 答案审核器
│   │   ├── roles.py        # 角色定义
│   │   ├── container.py    # 依赖注入容器
│   │   ├── service.py      # RAG 服务门面
│   │   └── ...
│   ├── data_process/       # 文档摄入管线
│   ├── settings/           # 配置管理
│   ├── observability.py    # 日志 + 指标
│   └── exceptions.py       # 异常体系
├── tests/                  # 测试套件
├── data/
│   ├── index/              # 向量索引存储
│   └── raw/                # 原始文档
├── pyproject.toml
├── requirements.txt
└── README.md
```

## 设计哲学

1. **依赖倒置**：业务层只依赖 `ChatModel` Protocol，不耦合具体厂商 SDK
2. **开闭原则**：新增模型 = 写一个 Adapter + 注册一行，不改调用方
3. **故障透明**：所有异常（Python 内置 / 自定义 / SDK）都有明确映射和兜底策略
4. **审计可追溯**：Agent 的每次决策、工具调用、模型调用全部进入审计轨迹
5. **安全默认**：租户隔离强制生效、Prompt Injection 默认拒绝、工具白名单默认开启
6. **配置即代码**：行为参数全部可配置，支持环境变量覆盖，无需改代码

## License

GPL-3.0
