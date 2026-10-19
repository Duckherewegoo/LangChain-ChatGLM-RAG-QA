"""A/B 对比主测试。

跑所有策略 × 所有用例，打印对比表并断言关键不变量。
使用 -s 运行可以看到表格：

    pytest tests/ab_test/ -v -s
"""

from __future__ import annotations

import dataclasses
import time

import pytest

from src.exceptions import ValidationError
from src.rag.orchestrator import OrchestratorRequest

from .cases import ABCase, CASES
from .fakes import FakeChatModel, FakeRetrieval
from .strategies import STRATEGIES, StrategyBuilder


@dataclasses.dataclass
class ABResult:
    """单个策略 × 单个用例的执行结果。"""

    strategy: str
    case: str
    duration_ms: float
    answer: str
    review_status: str
    review_rounds: int
    roles_invoked: list[str]
    event_count: int
    succeeded: bool
    error: str | None = None


def _run_one(
    strategy_name: str,
    builder: StrategyBuilder,
    case: ABCase,
) -> ABResult:
    """在隔离的 orchestrator 实例上跑一条用例，捕获异常并归一化结果。"""
    model = FakeChatModel(content="根据[1]，年假需提前申请。")
    retrieval = FakeRetrieval(case.documents)
    orchestrator = builder(model, retrieval)

    started = time.perf_counter()
    try:
        response = orchestrator.run(
            OrchestratorRequest(question=case.question, tenant_id="t1")
        )
    except ValidationError as exc:
        # 输入被 guardrails 拦截，属于预期路径，不算失败
        elapsed_ms = (time.perf_counter() - started) * 1000
        return ABResult(
            strategy=strategy_name,
            case=case.name,
            duration_ms=elapsed_ms,
            answer=str(exc),
            review_status="input_rejected",
            review_rounds=0,
            roles_invoked=[],
            event_count=0,
            succeeded=True,
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        return ABResult(
            strategy=strategy_name,
            case=case.name,
            duration_ms=elapsed_ms,
            answer="",
            review_status="error",
            review_rounds=0,
            roles_invoked=[],
            event_count=0,
            succeeded=False,
            error=str(exc),
        )

    elapsed_ms = (time.perf_counter() - started) * 1000
    summary = response.trail.summarize()
    return ABResult(
        strategy=strategy_name,
        case=case.name,
        duration_ms=elapsed_ms,
        answer=response.answer,
        review_status=response.review_status,
        review_rounds=response.review_rounds,
        roles_invoked=list(response.roles_invoked),
        event_count=int(summary["event_count"]),
        succeeded=True,
    )


def _format_table(results: list[ABResult]) -> str:
    header = (
        f"{'strategy':<8} {'case':<18} {'ms':>8} "
        f"{'review':<14} {'rounds':>6} {'events':>6} {'ok':<4}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(
            f"{r.strategy:<8} {r.case:<18} {r.duration_ms:>8.2f} "
            f"{r.review_status:<14} {r.review_rounds:>6} "
            f"{r.event_count:>6} {'yes' if r.succeeded else 'NO':<4}"
        )
    return "\n".join(lines)


def test_ab_comparison(capsys: pytest.CaptureFixture[str]) -> None:
    """跑所有策略 × 所有用例，打印对比表并断言关键不变量。"""
    all_results: list[ABResult] = []
    for strategy_name, builder in STRATEGIES.items():
        for case in CASES:
            all_results.append(_run_one(strategy_name, builder, case))

    table = _format_table(all_results)
    with capsys.disabled():
        print("\n" + table + "\n")

    # 不变量 1：所有策略在所有用例上都必须成功执行（输入拦截视为合法路径）
    for r in all_results:
        assert r.succeeded, f"{r.strategy}/{r.case} 执行失败: {r.error}"
        assert r.answer, f"{r.strategy}/{r.case} 返回空答案"

    # 不变量 2：no_evidence 用例必须触发安全拒答
    for r in all_results:
        if r.case != "no_evidence":
            continue
        assert "无法" in r.answer or "没有" in r.answer, (
            f"{r.strategy} 对无证据问题未拒答：{r.answer!r}"
        )

    # 不变量 3：prompt_injection 用例必须被 guardrails 拦截
    for r in all_results:
        if r.case != "prompt_injection":
            continue
        assert r.review_status == "input_rejected", (
            f"{r.strategy} 未拦截注入，review_status={r.review_status}"
        )

    # 不变量 4：fast 策略对普通用例不应产生审核轮次
    for r in all_results:
        if r.strategy != "fast" or r.case == "prompt_injection":
            continue
        assert r.review_status in ("skipped", "not_applicable"), (
            f"fast 策略不应启用审核，实际 review_status={r.review_status}"
        )

    # 不变量 5：safe 策略在 no_evidence 上应触发审核拒绝或 warning
    for r in all_results:
        if r.strategy != "safe" or r.case != "no_evidence":
            continue
        assert r.review_status in ("rejected", "warning"), (
            f"safe 策略对无证据问题应触发审核，实际 {r.review_status}"
        )
