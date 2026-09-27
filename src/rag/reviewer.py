"""答案发布前审核（Reviewer）。

审核发生在 harness 生成答案**之后**、响应返回用户**之前**，是整个流水线里
"事实性兜底"的最后一道闸。它回答四个问题：

1. 答案里的断言是否都有证据支撑？（事实性）
2. 引用编号是否真实存在、且指向正确片段？（引用完整性）
3. 是否存在敏感信息泄露或越权披露？（合规）
4. 术语、数值是否与证据一致？（口径一致性）

设计要点
--------
* 审核结果是**结构化的** :class:`ReviewReport`，不是一段自由文本。这是
  关键差异：结构化结果可以被单测覆盖，也能让 harness 自动决定"重试"
  还是"拒答"，而不是靠正则去抠模型的评语。
* 审核走**双层降级**：先尝试用模型做语义审核；模型不可用或输出不合法
  时退化为基于规则的确定性审核。规则审核覆盖三条硬红线——空答案、
  无证据、引用越界——这三条不需要 LLM 也能判。审核环节绝不能因为
  LLM 抖动而让整次请求失败，所以这里用的是"降级"而不是"抛错"。
* 报告里带 ``retryable`` 与 ``severity`` 两个维度。harness 据此决定：
  blocking 且 retryable -> 重做；blocking 且不可重试 -> 安全拒答；
  warning -> 放行并附提示。
* 审核器的提示词与 :class:`~src.rag.roles.ReviewerRole` 的契约是同一份，
  不是两份副本：角色负责"声明"，审核器负责"执行"。
"""

from __future__ import annotations

import dataclasses
import logging
import typing

from pydantic import BaseModel, Field

from src.model.manager import ChatModel, ChatRequest, ChatResponse
from src.rag.roles import Blackboard, ReviewerRole

logger = logging.getLogger(__name__)

_SEVERITIES = {"blocking", "warning", "info"}


@dataclasses.dataclass(frozen=True)
class ReviewIssue:
    """一条审核问题。不可变，便于审计与测试比对。"""

    code: str
    severity: str
    message: str
    suggestion: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "suggestion": self.suggestion,
        }


class ReviewReport(BaseModel):
    """一次审核的结构化结论。"""

    approved: bool
    issues: list[dict[str, str]] = Field(default_factory=list)
    summary: str = ""
    model: str | None = None
    strategy: str = "rule"  # "llm" or "rule"

    @property
    def blocking(self) -> list[dict[str, str]]:
        return [issue for issue in self.issues if issue.get("severity") == "blocking"]

    @property
    def warnings(self) -> list[dict[str, str]]:
        return [issue for issue in self.issues if issue.get("severity") == "warning"]

    def retryable(self) -> bool:
        """是否有理由让 harness 重做一次推理。

        规则：只要存在 blocking 问题，就值得重试一次（让模型拿到反馈后再
        生成一遍）。warning 不触发重试，避免无谓消耗。
        """

        return bool(self.blocking)

    def to_dict(self) -> dict[str, typing.Any]:
        return {
            "approved": self.approved,
            "issues": list(self.issues),
            "summary": self.summary,
            "model": self.model,
            "strategy": self.strategy,
            "retryable": self.retryable(),
        }


# ---------------------------------------------------------------------------
# 审核器
# ---------------------------------------------------------------------------


class AnswerReviewer:
    """对答案做发布前质检。

    用法::

        report = reviewer.review(question, answer, evidence)
        if report.blocking:
            ... 按策略重试或拒答 ...

    审核器**永不抛出业务异常**：它要么返回 LLM 报告，要么返回规则报告。
    这一保证由 :meth:`review` 最外层的兜底保证——任何意外错误都会被转
    成"warning 级"的报告，确保审核是增强项，不是新的单点故障。
    """

    def __init__(
        self,
        model: ChatModel | None = None,
        *,
        max_evidence_chars: int = 6000,
        on_issue: typing.Callable[[ReviewIssue], None] | None = None,
    ) -> None:
        self._model = model
        self._max_evidence_chars = max_evidence_chars
        self._on_issue = on_issue
        self._role = ReviewerRole()

    @property
    def role(self) -> ReviewerRole:
        return self._role

    def review(
        self,
        question: str,
        answer: str,
        evidence: typing.Sequence[dict[str, typing.Any]],
    ) -> ReviewReport:
        """审核答案，优先用 LLM，失败则退化为规则审核。"""

        normalized = (answer or "").strip()
        if self._model is not None and normalized:
            try:
                return self._review_with_llm(question, normalized, list(evidence))
            except Exception as exc:
                logger.warning("answer_review_llm_failed", extra={"error": str(exc)})

        return self._review_with_rules(question, normalized, list(evidence))

    # ---- LLM 审核 ---------------------------------------------------------

    def _review_with_llm(
        self,
        question: str,
        answer: str,
        evidence: list[dict[str, typing.Any]],
    ) -> ReviewReport:
        prompt = self._role.build_prompt(_blackboard_for(question, answer, evidence))
        response = self._model.invoke(
            ChatRequest(
                messages=[{"role": "user", "content": prompt}],
                tools=[],
            )
        )
        report = _parse_review_response(response)
        report.model = getattr(response, "model", None) or getattr(self._model, "name", None)
        report.strategy = "llm"
        self._emit_issues(report)
        return report

    # ---- 规则审核 ---------------------------------------------------------

    def _review_with_rules(
        self,
        question: str,
        answer: str,
        evidence: list[dict[str, typing.Any]],
    ) -> ReviewReport:
        """不依赖 LLM 的确定性审核，覆盖三条硬红线。"""

        issues: list[ReviewIssue] = []

        if not answer:
            issues.append(
                ReviewIssue(
                    code="empty_answer",
                    severity="blocking",
                    message="答案为空，无法交付给用户。",
                    suggestion="基于已有证据重新生成答案，或明确说明无法确认。",
                )
            )

        if not evidence:
            issues.append(
                ReviewIssue(
                    code="no_evidence",
                    severity="blocking",
                    message="没有可用证据却产生了答案。",
                    suggestion="先执行检索，或返回无法确认声明。",
                )
            )
        else:
            # 引用越界检测：模型给出的 [n] 必须落在证据索引内
            import re

            used = {int(ref) for ref in re.findall(r"\[(\d+)\]", answer)}
            valid_range = range(1, len(evidence) + 1)
            out_of_range = sorted(ref for ref in used if ref not in valid_range)
            if out_of_range:
                issues.append(
                    ReviewIssue(
                        code="citation_out_of_range",
                        severity="blocking",
                        message=(
                            f"答案引用了不存在的编号 {out_of_range}，有效范围是 1..{len(evidence)}。"
                        ),
                        suggestion="修正引用编号，使其与证据序号一致。",
                    )
                )

            # 空洞兜底：答案既无引用、也没有任何来源暗示时告警
            if not re.search(r"\[\d+\]", answer):
                issues.append(
                    ReviewIssue(
                        code="no_inline_citation",
                        severity="warning",
                        message="答案中未发现内联引用标注。",
                        suggestion="为关键结论补上 [n] 形式的事实来源编号。",
                    )
                )

        approved = not any(issue.severity == "blocking" for issue in issues)
        report = ReviewReport(
            approved=approved,
            issues=[issue.to_dict() for issue in issues],
            summary="规则审核通过。" if approved else "规则审核发现阻断性问题。",
            strategy="rule",
        )
        self._emit_issues(report)
        return report

    def _emit_issues(self, report: ReviewReport) -> None:
        if not self._on_issue:
            return
        for raw in report.issues:
            try:
                self._on_issue(
                    ReviewIssue(
                        code=raw.get("code", "unknown"),
                        severity=raw.get("severity", "warning"),
                        message=raw.get("message", ""),
                        suggestion=raw.get("suggestion", ""),
                    )
                )
            except Exception:
                logger.exception("review_issue_listener_failed")


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _blackboard_for(
    question: str,
    answer: str,
    evidence: list[dict[str, typing.Any]],
) -> Blackboard:
    """为角色提示词构造最小黑板。"""

    return Blackboard(
        request_id="review",
        original_question=question,
        answer_draft=answer,
        evidence=evidence,
    )


def _parse_review_response(response: ChatResponse) -> ReviewReport:
    """把模型输出解析成 ReviewReport；解析失败即视为审核未通过。"""


    text = (getattr(response, "content", "") or "").strip()
    data = _extract_json_object(text)
    if not isinstance(data, dict):
        return ReviewReport(
            approved=False,
            issues=[
                {
                    "code": "review_output_unparseable",
                    "severity": "blocking",
                    "message": "审核模型未返回可解析的 JSON。",
                    "suggestion": "重试一次；持续失败时降级为规则审核。",
                }
            ],
            summary="审核输出无法解析。",
            strategy="llm",
        )

    raw_issues = data.get("issues") or []
    issues: list[dict[str, str]] = []
    for item in raw_issues:
        if not isinstance(item, dict):
            continue
        severity = str(item.get("severity", "warning")).lower()
        if severity not in _SEVERITIES:
            severity = "warning"
        issues.append(
            {
                "code": str(item.get("code", "generic")),
                "severity": severity,
                "message": str(item.get("message", "")),
                "suggestion": str(item.get("suggestion", "")),
            }
        )

    approved = bool(data.get("approved", False)) and not any(
        item.get("severity") == "blocking" for item in issues
    )
    return ReviewReport(
        approved=approved,
        issues=issues,
        summary=str(data.get("summary", "")),
        strategy="llm",
    )


def _extract_json_object(text: str) -> typing.Any:
    """容错解析：剥离 ```json 代码块，定位首个完整 JSON 对象。"""

    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    decoder = __import__("json").JSONDecoder()
    decoder.scan_once  # 引用以触发属性访问检查
    start = text.find("{")
    # 逐个候选起始位置尝试，容忍模型输出里的前后缀文本
    while start != -1:
        decoder = __import__("json").JSONDecoder()
        index = start
        while index < len(text):
            try:
                obj, _end = decoder.raw_decode(text, index)
            except ValueError:
                index += 1
                continue
            return obj
        start = text.find("{", start + 1)
    return None


__all__ = [
    "AnswerReviewer",
    "ReviewIssue",
    "ReviewReport",
]
