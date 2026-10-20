"""答案发布前审核（Reviewer）。

审核发生在 harness 生成答案之后、响应返回用户之前，是整个流水线里
"事实性兜底"的最后一道闸。它回答四个问题：

1. 答案里的断言是否都有证据支撑？（事实性）
2. 引用编号是否真实存在、且指向正确片段？（引用完整性）
3. 是否存在敏感信息泄露或越权披露？（合规）
4. 术语、数值是否与证据一致？（口径一致性）

设计要点：
1. 审核结果是结构化的 ReviewReport，不是一段自由文本。这让 harness 能
   自动决定"重试"还是"拒答"，而不是靠正则去抠模型的评语。
2. 审核走双层降级：先尝试用模型做语义审核；模型不可用或输出不合法时
   退化为基于规则的确定性审核。
3. 报告里带 retryable 与 severity 两个维度。harness 据此决定：
   blocking 且 retryable -> 重做；blocking 且不可重试 -> 安全拒答；
   warning -> 放行并附提示。
4. 审核器的提示词与 ReviewerRole 的契约是同一份，角色负责"声明"，
   审核器负责"执行"。
"""

from __future__ import annotations

import json
import re
import typing

from pydantic import BaseModel, Field

from model.protocol import ChatModel, ChatRequest, ChatResponse
from observability import get_logger
from rag.roles import Blackboard, ReviewerRole

logger = get_logger(__name__)

_SEVERITIES = {"blocking", "warning", "info"}
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


class ReviewIssue(BaseModel):
    """一条审核问题。不可变，便于审计与测试比对。"""

    model_config = {"frozen": True}

    code: str
    severity: str
    message: str
    suggestion: str = ""
    retryable: bool = True

    def to_dict(self) -> dict[str, typing.Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "suggestion": self.suggestion,
            "retryable": self.retryable,
        }


class ReviewReport(BaseModel):
    """一次审核的结构化结论。"""

    approved: bool
    issues: list[dict[str, typing.Any]] = Field(default_factory=list)
    summary: str = ""
    model: str | None = None
    strategy: str = "rule"

    @property
    def blocking(self) -> list[dict[str, typing.Any]]:
        return [issue for issue in self.issues if issue.get("severity") == "blocking"]

    @property
    def warnings(self) -> list[dict[str, typing.Any]]:
        return [issue for issue in self.issues if issue.get("severity") == "warning"]

    def retryable(self) -> bool:
        """是否有理由让 harness 重做一次推理。

        规则：只要存在一条 "blocking 且标注为可重试" 的问题，就触发重试。
        这是为了避免空答案之类不可恢复的问题让 harness 无谓重试烧钱。
        """
        return any(
            issue.get("severity") == "blocking" and issue.get("retryable", True)
            for issue in self.issues
        )

    def to_dict(self) -> dict[str, typing.Any]:
        return {
            "approved": self.approved,
            "issues": list(self.issues),
            "summary": self.summary,
            "model": self.model,
            "strategy": self.strategy,
            "retryable": self.retryable(),
        }


class AnswerReviewer:
    """对答案做发布前质检。"""

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
        evidence_list = list(evidence)

        if self._model is not None and normalized:
            try:
                return self._review_with_llm(question, normalized, evidence_list)
            except Exception as exc:
                logger.warning(f"LLM 审核失败，降级为规则审核 | error={exc!s}")

        try:
            return self._review_with_rules(question, normalized, evidence_list)
        except Exception as exc:
            logger.exception(f"规则审核意外失败 | error={exc!s}")
            return ReviewReport(
                approved=True,
                issues=[
                    ReviewIssue(
                        code="reviewer_internal_error",
                        severity="warning",
                        message=f"审核器内部异常: {exc}",
                        suggestion="本次审核已跳过，请人工复核。",
                        retryable=False,
                    ).to_dict()
                ],
                summary="审核器内部异常，已放行并标记警告。",
                strategy="rule",
            )

    def _review_with_llm(
        self,
        question: str,
        answer: str,
        evidence: list[dict[str, typing.Any]],
    ) -> ReviewReport:
        # Pyright 无法跨方法追踪 self._model 的窄化，显式断言
        assert self._model is not None, "_review_with_llm 需要已配置的模型"
        prompt = self._role.build_prompt(_blackboard_for(question, answer, evidence))
        response = self._model.invoke(
            ChatRequest(messages=[{"role": "user", "content": prompt}], tools=[])
        )
        report = _parse_review_response(response)
        report.model = response.model or getattr(self._model, "name", None)
        report.strategy = "llm"
        self._emit_issues(report)
        return report

    def _review_with_rules(
        self,
        question: str,
        answer: str,
        evidence: list[dict[str, typing.Any]],
    ) -> ReviewReport:
        """不依赖 LLM 的确定性审核，覆盖三条硬红线。

        question 当前未被使用，保留参数是为了和 _review_with_llm 保持接口对称，
        未来若加入"问题与答案相关性"规则会用到它。
        """
        issues: list[ReviewIssue] = []

        if not answer:
            issues.append(
                ReviewIssue(
                    code="empty_answer",
                    severity="blocking",
                    message="答案为空，无法交付给用户。",
                    suggestion="基于已有证据重新生成答案，或明确说明无法确认。",
                    retryable=False,
                )
            )

        if not evidence:
            issues.append(
                ReviewIssue(
                    code="no_evidence",
                    severity="blocking",
                    message="没有可用证据却产生了答案。",
                    suggestion="先执行检索，或返回无法确认声明。",
                    retryable=True,
                )
            )
        else:
            used = {int(ref) for ref in _CITATION_PATTERN.findall(answer)}
            valid_range = range(1, len(evidence) + 1)
            out_of_range = sorted(ref for ref in used if ref not in valid_range)
            if out_of_range:
                issues.append(
                    ReviewIssue(
                        code="citation_out_of_range",
                        severity="blocking",
                        message=(
                            f"答案引用了不存在的编号 {out_of_range}，"
                            f"有效范围是 1..{len(evidence)}。"
                        ),
                        suggestion="修正引用编号，使其与证据序号一致。",
                        retryable=True,
                    )
                )

            if not used:
                issues.append(
                    ReviewIssue(
                        code="no_inline_citation",
                        severity="warning",
                        message="答案中未发现内联引用标注。",
                        suggestion="为关键结论补上 [n] 形式的事实来源编号。",
                        retryable=False,
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
                        retryable=raw.get("retryable", True),
                    )
                )
            except Exception:
                logger.exception("审核问题监听器回调失败。")


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
                    "retryable": True,
                }
            ],
            summary="审核输出无法解析。",
            strategy="llm",
        )

    raw_issues = data.get("issues") or []
    issues: list[dict[str, typing.Any]] = []
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
                "retryable": bool(item.get("retryable", True)),
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
        text = text[3:]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start == -1:
        return None

    decoder = json.JSONDecoder()
    for index in range(start, len(text)):
        if text[index] != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(text, index)
            return obj
        except json.JSONDecodeError:
            continue
    return None


__all__ = [
    "AnswerReviewer",
    "ReviewIssue",
    "ReviewReport",
]
