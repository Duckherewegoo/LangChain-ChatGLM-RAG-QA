---
id: 0004-ab_test
aliases: []
tags: []
---

# ADR-0004: 基于策略矩阵与业务不变量的 A/B 测试架构

- Status: Accepted
- Date: 2026-10-19（commit `a4f67ed`：重构测试套件并引入 A/B 测试）
- Related:
  - 代码：`tests/ab_test/`、`src/rag/orchestrator.py`
  - 依赖：`pytest`、`FakeChatModel`、`FakeRetrieval`

## Context

RAG 系统包含大量的策略组合（检索策略 × 审核策略 × 模型路由）。
如果使用传统的单元测试，不仅写起来繁琐，而且只要一改 Prompt 或者换个模型，断言就会大面积崩溃。如果直接去调真实的大模型 API 做集成测试，成本极高且结果具有随机性（LLM 每次回答不一样，无法做精确断言）。

此外，业务需求要求：**无论底层走的是什么策略（快速模式、安全模式、普通模式），系统的底线行为绝不能变。**（比如：绝对不能把 Prompt 注入当成正常问题回答，绝对不能对没有证据的问题瞎编）。

## Decision

引入**基于策略矩阵的 A/B 测试架构**，并采用**业务不变量（Business Invariants）**作为核心断言标准。

1. **彻底解耦的 Fake 依赖**：
   - 实现 `FakeChatModel`，走真实的 `ChatModel` Protocol 路径，但直接返回写死的字符串。
   - 实现 `FakeRetrieval`，直接返回预设的文档集合。这保证了测试既不消耗 Token，也不依赖外部 DB。
2. **全矩阵运行（Cartesian Product）**：
   遍历所有的 `Strategy` 和所有的 `TestCase`（如普通问答、无证据、注入攻击），组合执行并生成对比表（表头包含耗时、审核状态、轮次等）。
3. **业务不变量断言（不测文本，测行为）**：
   - `invariant 1`：所有策略在所有用例上不能崩溃，必须有 Answer。
   - `invariant 2`：对于 `no_evidence` 用例，所有策略必须输出包含“无法”或“没有”的拒答文案。
   - `invariant 3`：对于 `prompt_injection` 用例，必须在 Guardrails 阶段就被 `input_rejected` 拦截。
   - `invariant 4`：对于 `fast` 策略，不允许产生审核轮次。

## Alternatives Considered

- **A. 真实调用 LLM 做集成测试**：**否决**。每次 CI 跑都要花真金白银，而且 LLM 的随机性会导致测试用例像掷骰子一样时红时绿（Flaky Tests）。
- **B. 用 `unittest.mock.MagicMock` 全局打桩**：**否决**。Mock 会破坏真实的调用栈，导致测试通过但线上依然崩溃。使用 `FakeChatModel` 实现了真实的接口契约，保证了链路是走通的。
- **C. 单纯对比两个版本的生成结果（传统的 A/B Test）**：**否决**。对于不同策略，它们的目标不同（如 fast 不审核，safe 强审核），不能用同一套输出标准去比对。应该比对“它们是否遵守了各自路径的预期逻辑”。

## Consequences

**正面**
- **测试即文档**：只要看 `test_ab_comparison` 里的 5 个不变量断言，立刻就能明白这个 RAG 系统的绝对红线是什么。
- **极速 CI**：跑完所有策略和用例的矩阵，耗时极短，完全可以放在每次 Commit 前执行。
- **勇敢重构的底气**：有了这些不变量兜底，开发者可以大胆地去优化 Retrieval 或 Reviewer 的代码，只要不变量没破，系统就不会出大问题。

**负面（接受的代价）**
- **编写成本高**：设计 `FakeChatModel` 和处理各种用例的预期结果，比直接写 `assert result == "xxx"` 要复杂得多。
- **无法测试 LLM 的“语义质量”**：这种测试只能保证“逻辑跑通”和“安全兜底”，不能保证模型回答得是否优美（后者只能靠人工评测或 LLM-as-a-Judge）。

**备注：AI 时代的测试护城河**
AI 很容易写出“能跑就行”的粗糙代码。但当开发者把“A/B 矩阵”和“业务不变量”这些架构要求写进 Prompt 后，AI 就被迫搭建出了这种极具商业级水准的测试护城河。这是软件工程规范在 AI 辅助生成中的完美应用。
