---
id: 0001-dip-factory-pattern
aliases: []
tags: []
---

# ADR-0001: 依赖倒置与多模型工厂设计

- Status: Accepted
- Date: 2026-10-19（commit `315035f`：引入 Protocol 契约并全面解耦）
- Related:
  - 代码：`src/model/protocol.py`、`src/model/manager.py`、`src/model/registry.py`、`src/model/providers/`
  - 设计原则：依赖倒置（DIP）、开闭原则（OCP）

## Context

**痛点**：企业级 RAG 系统绝不能绑定在某一家大模型厂商的战车上。
- 业务需求可能在 Zhipu、DeepSeek、Qwen、豆包之间随时切换（为了成本、并发或效果评测）。
- 不同厂商的 SDK、参数命名、错误码、返回结构甚至流式输出的格式都完全不同。如果业务代码直接 `import zhipu` 并调用，一旦切换厂商，整个 RAG 链路（改写、推理、重排、审核）都要跟着大改。

**早期尝试**：项目 v1.0 版本是直接调用的，结果各种模型参数和异常处理渗透到了业务层的每一个角落，代码被污染成了意大利面条。

## Decision

全面引入**依赖倒置原则（DIP）**与**工厂模式（Factory Pattern）**，建设模型抽象层。

1. **协议契约（Protocol）**：定义 `ChatModel` 和 `EmbeddingModel` Protocol（基于 `typing.Protocol`），规定系统唯一的语言，如 `async chat(messages, tools) -> ChatResponse`。
2. **适配器（Adapter）**：把所有第三方厂商 SDK 隔离在 `providers/` 目录下，各自实现 Protocol 接口，将不同厂商的异常统一映射为系统内的领域异常。
3. **注册表（Registry）与工厂**：通过 `registry.py` 动态注册 Provider。业务层通过 `ModelManager` 按需获取具体的模型实例，完全不感知 SDK 的存在。

## Alternatives Considered

- **A. 使用 LangChain 的 `ChatOpenAI` 等现成封装**：**否决**。LangChain 的封装太重，且各家的自定义能力（如特殊的 Tool Calling 格式）容易被抹平。我们需要极致的轻量和绝对的控制权。
- **B. 简单的字典映射（`if provider == 'zhipu': return ZhipuClient()`）**：**否决**。一旦模型增加到 5 个以上，字典映射会变成巨大的屎山，且违背开闭原则。
- **C. 多继承或抽象基类（ABC）**：未采纳。Python 生态更推荐鸭子类型。`typing.Protocol` 是结构化子类型，不强求显式继承，对第三方 SDK 的兼容性更好。

## Consequences

**正面**
- **开闭原则达成**：新增一个厂商（如 Moonshot Kimi），只需在 `providers/` 下加一个文件，在 `registry.py` 注册一行，业务代码 **零改动**。
- **测试极简**：因为有了 Protocol，测试时可以直接写一个 `FakeChatModel` 实现该协议，彻底摆脱对真实 API 和网络环境的依赖（见 `tests/ab_test/`）。
- **彻底解耦**：业务层（RAG）不再关心底层是谁在提供算力。

**负面（接受的代价）**
- **抽象泄漏风险**：有些厂商 SDK 带有极其特殊的参数（比如 Zhipu 独有的某种系统提示词），强行抹平进通用 Protocol 可能会导致部分厂商的高级特性用不了。
- **前置开发成本高**：必须在一开始就设计出合理的数据类（`ChatResponse`, `Usage`），如果设计不当，后期加字段会很痛苦。

**备注：AI 时代的架构师价值**
虽然这个项目在 9 月底由 AI 生成了第一版原型（commit `f0015e7`），但在 10 月 19 日的重构中，将底层彻底翻转为依赖倒置架构，是人类架构师做出的核心判断。**AI 可以一秒写出工厂，但 AI 不会主动告诉你“这里必须解耦”。**
