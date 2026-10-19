"""A/B 对比的策略构建器。

每个策略是一个 Callable[(model, retrieval) -> AgentOrchestrator]，
差异只体现在审核与护栏配置上，模型和检索器由用例注入。
"""

from __future__ import annotations

import typing

from src.model.fallback import FallbackExecutor
from src.rag.orchestrator import AgentOrchestrator
from src.rag.reviewer import AnswerReviewer
from src.rag.tools import ToolRegistry
from src.settings import AgentSettings, GuardrailsSettings, ReviewSettings

StrategyBuilder = typing.Callable[[typing.Any, typing.Any], AgentOrchestrator]


def build_fast(model: typing.Any, retrieval: typing.Any) -> AgentOrchestrator:
    """fast：单模型 + 禁用审核。延迟最低，但对质量问题不设防。"""
    return AgentOrchestrator(
        executor=FallbackExecutor(
            primary=model, secondaries=[], max_attempts_per_provider=1
        ),
        retrieval=retrieval,
        reviewer=AnswerReviewer(model=None),
        registry=ToolRegistry(),
        agent_settings=AgentSettings(
            review=ReviewSettings(enabled=False, use_llm=False)
        ),
        guardrails_settings=GuardrailsSettings(
            prompt_injection_action="reject"
        ),
    )


def build_safe(model: typing.Any, retrieval: typing.Any) -> AgentOrchestrator:
    """safe：单模型 + 启用规则审核。延迟略增，能拦住引用越界与空答案。"""
    return AgentOrchestrator(
        executor=FallbackExecutor(
            primary=model, secondaries=[], max_attempts_per_provider=1
        ),
        retrieval=retrieval,
        reviewer=AnswerReviewer(model=None),
        registry=ToolRegistry(),
        agent_settings=AgentSettings(
            review=ReviewSettings(
                enabled=True, use_llm=False, max_retries=1
            )
        ),
        guardrails_settings=GuardrailsSettings(
            prompt_injection_action="reject"
        ),
    )


STRATEGIES: dict[str, StrategyBuilder] = {
    "fast": build_fast,
    "safe": build_safe,
}
