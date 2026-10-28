---
id: 0005-roles-permission
aliases: []
tags: []
---

# ADR-0005: 基于共享黑板（Blackboard）的多角色协作契约

- Status: Accepted
- Date: 2026-10-19（commit `947f596`：引入多角色编排与黑板通信）
- Related:
  - 代码：`src/rag/roles.py`、`src/rag/orchestrator.py`、`src/rag/agent.py`
  - 设计原则：单一职责（SRP）、契约编程（Design by Contract）

## Context

在实现“检索 -> 推理 -> 审核”的 RAG 链路时，如果让大模型在一个 Prompt 里同时完成这些事，会导致：
1. **角色混淆与幻觉**：模型在检索时自己编造事实，或者在推理时忽略引用。
2. **不可控的重试**：如果审核失败，传统的单体 Agent 很难精确回退到“重新推理”而不是“重新检索”。
3. **黑盒状态**：多 Agent 框架（如 AutoGen / CrewAI）通过自然语言互相通信，状态在聊天记录中游走，难以进行确定性的代码级拦截和审计。

**架构要求**：必须实现多角色分离，但绝不能引入庞大且不可控的自主 Agent 编排框架。我们需要的是**高度确定的流水线**。

## Decision

采用 **“硬编码流水线 + 共享黑板 + 契约约束”** 的设计模式：

1. **职责分离**：定义 Retriever（只找不说）、Reasoner（只说找的）、Reviewer（只审不改）、Orchestrator（只调度不做计算）四个核心角色。
2. **共享黑板（Blackboard）**：
   角色之间不直接通信。所有状态由 Pydantic 模型 `Blackboard` 统一承载（包含 `evidence`、`answer_draft`、`review` 等）。角色只读写自己关心的键。
3. **契约数据化（RoleDefinition）**：
   拒绝把约束只写在 Prompt 里。每个角色必须声明不可变的 `RoleDefinition`，包含 `purpose`（目的）、`responsibilities`（职责）、`must_not`（绝对禁止事项）和 `output_contract`（输出格式）。系统会在运行期读取这些契约进行校验和审计。
4. **纯声明角色（无状态）**：
   `Role` 子类不持有 LLM 或工具引用。它们只负责通过 `build_prompt` 组装上下文，真正的模型调用由 `AgentOrchestrator` 统一执行，确保模型调用被集中管控。

## Alternatives Considered

- **A. 采用 AutoGen 或 CrewAI 等纯自主 Agent 框架**：**否决**。企业级应用绝对不能接受“模型自己决定该找谁”的黑盒执行。流水线必须 100% 可控。
- **B. 把所有的规则都写进 System Prompt**：**否决**。Prompt 不是强约束，大模型极易产生“指令遗忘”。把 `must_not` 提取成 Python 数据结构，可以直接被单元测试和编排器拦截（例如：强行掐断 Reviewer 试图生成新内容的动作）。
- **C. 让模型自己通过复杂的 JSON 来互相传递状态**：**否决**。传递成本极高，且一旦模型输出格式崩溃，整个链路断裂。共享内存黑板（Blackboard）是最稳定、最高效的数据交换方式。

## Consequences

**正面**
- **绝对的可预测性**：Orchestrator 的流水线是硬编码的（retrieve -> reason -> review）。不会有“两个 Agent 无限循环聊天”的 Token 黑洞。
- **极高的可测试性**：因为 `RoleDefinition` 和 `Blackboard` 是纯 Pydantic/Dataclass，开发者可以极速编写单元测试，验证“Reasoner 是否违规引用了外部信息”或“Reviewer 是否输出了符合 Contract 的 JSON”。
- **审计级安全**：`AgentDefinition` 和所有的 `RoleDefinition` 被统一导出，直接进入 API 的 `/agent/definition` 响应。

**负面（接受的代价）**
- **灵活性降低**：如果未来要增加一个 `Planner` 角色动态决定走哪条分支，硬编码的流水线需要改动代码（调整状态机），而不是像自主 Agent 那样改改 Prompt 就行。
- **黑板的键名协议脆弱性**：`Blackboard` 中的字典键（如 `evidence`、`answer_draft`）是隐式契约。如果一个角色将数据写成了 `documents` 而不是 `evidence`，下游就会取不到值。虽然 Pydantic 提供了基础类型保护，但无法保护业务语义。

**备注：优雅的“包装”**
AI 生成代码时，通常会把约束全塞进 System Prompt。在重构时，人类架构师强制要求将约束抽象为 `RoleDefinition` 数据类。这种做法的精妙之处在于：**“禁止做什么”比“要求做什么”更适合作为系统的红线，且更容易被工程化度量。**
