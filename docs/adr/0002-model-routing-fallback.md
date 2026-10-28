---
id: 0002-error-fallback-router
aliases: []
tags: []
---

# ADR-0002: 模型路由、细粒度错误分类与 Fallback 降级机制

- Status: Accepted
- Date: 2026-10-19（commit `315035f`：引入 ModelRouter 与 FallbackExecutor）
- Related:
  - 代码：`src/model/errors.py`、`src/model/fallback.py`、`src/model/router.py`
  - 配置：`settings/settings.py`（ModelRouterSettings）
  - 设计原则：容错设计（Fault Tolerance）、故障隔离（Fault Isolation）

## Context

企业级 LLM 应用面临极其脆弱的网络和 API 环境：
1. **厂商不稳定**：随时可能遇到 `429 限流`、`502 网关错误`、或者几秒以上的长 `timeout`。
2. **错误的致命程度不同**：如果是网络超时，换个模型或等几秒就好了；但如果是 `401 API Key 错误`、或者触发了 `400 内容安全审查`，你在所有模型上重试都会失败，盲目重试只会浪费时间和 token。
3. **场景差异**：有些任务（如意图识别、查询改写）用便宜、极速的小模型即可；有些任务（如最终答案合成）必须用高智商的大模型。

**早期教训**：市面上许多开源项目（包括 AI 默认生成的代码）只会粗暴地写 `try-except Exception`，然后无脑重试三次。这在企业级场景中是灾难，不仅掩盖了真实的配置错误（如 Key 失效），还会因为重试引发更严重的限流雪崩。

## Decision

采取 **“场景路由 + 细粒度异常分类 + 状态机回退链”** 的综合容错策略：

1. **基于用例的模型路由（Use Case Routing）**：
   通过 `settings.yaml` 绑定不同的 `use_case`（如 `chat`、`agent`），为其配置主备模型（Primary / Secondary）。框架不硬编码模型用途。
2. **细粒度领域异常体系（Domain Exceptions）**：
   在 `errors.py` 中定义 `ModelTimeoutError`、`ModelRateLimitError`、`ModelAuthenticationError` 等明确异常，并为每个异常绑定 `status_code` 和 `error_code`。
3. **启发式错误分类器（Error Classifier）**：
   提供 `classify_model_error` 函数。对于已知的异常直接判定；对于第三方 SDK 抛出的未知异常，利用正则表达式和关键词匹配（如 `"rate limit"`, `"invalid api key"`, `"timed out"`）将其归类，并打上 `retryable`（是否可重试）的布尔标签。
4. **状态机回退执行（FallbackExecutor）**：
   在 `fallback.py` 中实现顺序回退链。对于瞬时故障（`retryable=True`），在同一模型内重试；对于永久故障（如 Auth、Content Policy），**直接中断当前模型，不浪费重试次数**，平滑降级到下一个备用模型。

## Alternatives Considered

- **A. 使用现成的重试库（如 `tenacity`）**：**未完全采纳**。虽然底层可以使用，但无法解决“根据错误类型决定是否切换模型”的需求。必须自研 `FallbackExecutor` 来记录完整的 `FallbackAttempt` 轨迹。
- **B. 大锅烩重试（无论什么错误都无脑重试 3 次）**：**否决**。这是新手最爱写的逻辑，会导致遇到 `401` 或内容违规时白等几秒钟。
- **C. 依赖 HTTP 状态码强判断**：**未完全采纳**。因为很多大模型厂商的 SDK 没有抛出标准的 HTTP 异常，只能靠 `str(exc)` 里的内容做启发式兜底。

## Consequences

**正面**
- **高可用与高鲁棒性**：单点模型宕机（如智谱挂了）完全不影响业务，静默切到备模型（如 DeepSeek）。
- **精确的耗时控制**：当遇到内容安全审查（Content Policy）直接失败退出，不浪费时间重试。
- **极高的可观测性**：通过 `FallbackResult.attempts`，能精准记录每一次尝试的耗时、模型名、错误类型。这对于排查“昨晚为什么一次对话花了 10 秒”极其重要。

**负面（接受的代价）**
- **启发式分类的局限性**：如果厂商修改了报错文案（比如把 `"invalid api key"` 改成了 `"authentication rejected"`），正则匹配可能会失效，导致误判为 `unknown`（不可重试）或 `connection`（可重试）。
- **代码量激增**：手动维护异常体系、映射表、分类器和回退执行器，比简单的 `try-except` 多了数百行代码。
- **上下文一致性的挑战**：如果模型 A 生成了一半突然断网，切换到模型 B 重新生成时，要确保之前注入的 System Prompt 和 History 能被 B 完美接收。

**备注：架构师的底线**
AI 极速生成代码时，默认只会写最基础的 `try-except`。是开发者在最初的 Prompt 中强制规定了“必须要有细粒度异常、必须要有 Fallback 链”，才让 AI 生成了这套工业级组件。这证明了在 AI 时代，**系统质量的上限依然由人类架构师的约束力决定**。
