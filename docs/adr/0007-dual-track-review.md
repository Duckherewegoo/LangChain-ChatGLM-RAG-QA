---
id: 0007-dual-track-review
aliases: []
tags: []
---

# ADR-0007: 基于 ReviewReport 的双轨审核与重试路由

- Status: Accepted
- Date: 2026-10-19
- Related:
  - 代码：`src/rag/reviewer.py`、`src/rag/orchestrator.py`

## Context

大模型生成答案后，如何决定“直接返回”、“让它重新生成”、还是“直接拒答”？
如果用纯规则（如正则），识别不了“语义矛盾”；如果用纯 LLM 审核，成本高且输出格式极易崩溃（比如返回 Markdown 包裹的 JSON）。

## Decision

设计 `ReviewReport` 结构化数据模型，实现双轨降级与智能重试路由：

1. **双轨审核降级**：
   - 优先尝试 `_review_with_llm`（语义审核）。
   - 只要模型异常或输出无法解析，立刻降级到 `_review_with_rules`（规则审核）。规则审核负责死守三道硬红线：`empty_answer`（答案为空）、`no_evidence`（无证据）、`citation_out_of_range`（引用越界）。
2. **容错 JSON 解析**：
   使用 `json.JSONDecoder().raw_decode` 编写 `_extract_json_object`，能从容剔除大模型可能输出的 ` ```json ` 标记或多余前言，精准抓取 JSON 主体。
3. **基于 `retryable` 的路由控制**：
   报告中的每一个 `ReviewIssue` 都带有 `severity`（blocking/warning）和 `retryable`（是否可重试）。
   `Orchestrator` 读取报告：
   - 若存在 `blocking` 且 `retryable=True` -> 触发 `reasoner` 重新生成（最多 N 轮）。
   - 若存在 `blocking` 且 `retryable=False` -> 直接掐断，生成 `_safe_refusal`。
   - 若只有 `warning` -> 放行，附加提示。

## Consequences
**正面**
- **绝不死循环**：像“空答案”这种重试一万次也没用的错误，被标记为 `retryable=False`，直接拦截，不浪费一分钱 Token。
- **绝对稳定**：就算大模型审核环节完全瘫痪，正则规则引擎也能兜住底线，保证系统仍能输出结果。

**负面**
- **代码膨胀**：手写 JSON 容错提取和两层审核机制，增加了代码复杂度和维护成本。
