---
id: 0000-index
aliases: []
tags: []
---

# 架构决策记录（ADR）索引 — Enterprise RAG System

> 本目录记录了一个基于 LangChain + LangGraph 构建的企业级 RAG 问答系统的核心架构决策。
> 重点聚焦于：依赖解耦、高可用路由、高级检索、状态机编排与安全兜底。

## 决策列表

| 编号 | 标题 | 状态 | 一句话总结 |
|---|---|---|---|
| [0001](0001-dependency-inversion.md) | 依赖倒置与多模型工厂 | Accepted | 基于 Protocol 与工厂模式解耦厂商 SDK，实现模型无缝切换 |
| [0002](0002-model-routing-fallback.md) | 模型路由与 Fallback 降级 | Accepted | 细粒度异常分类 + 主备降级链，保障高可用 |
| [0003](0003-advanced-rag.md) | 高级 RAG 检索管线 | Accepted | 向量+BM25 混合检索，辅以 RRF 融合与 Cross-Encoder 重排 |
| [0004](0004-ab-testing-invariants.md) | 策略矩阵与不变量测试 | Accepted | 以业务红线为断言，用 Fake 组件实现极速 CI 测试 |
| [0005](0005-multi-role-blackboard.md) | 多角色黑板通信与契约 | Accepted | 基于 RoleDefinition 契约，用双轨状态机编排多角色 |
| [0006](0006-security-isolation.md) | 向量指纹治理与安全默认 | Accepted | Embedding 指纹维度校验 + 租户物理隔离 + Guardrails |
| [0007](0007-dual-track-review.md) | 双轨审核与重试路由 | Accepted | 规则保底 + LLM 语义审核，基于 ReviewReport 动态控制重试 |

## 总结：AI 时代的架构师底线
本项目虽是借助 AI 极速生成的初版，但后续的深度重构注入了极强的工业级约束。
从防御性的 `EmbeddingFingerprint`，到手写容错的 JSON 解析器，再到底层拦截重试的 `retryable` 机制。
这证明了：**AI 可以写出跑得通的代码，但系统的防弹能力与架构底线，永远由人类工程师设计。**
