"""Agent 角色体系：明确"这个 Agent 到底是干什么的"。

整个系统对外只暴露一种产品语义——"企业知识库问答"。但实现上它由若干
**职责单一**的角色协作完成。每个角色都是一段带约束的系统提示 + 一组
能力边界，而**不是一个独立进程**：角色之间在同一张 LangGraph 里用消息
传递协作，没有引入额外编排框架。

角色清单
--------
:class:`RetrieverRole`
    只做检索规划与证据汇总。它决定"该问知识库什么问题"，但**不**回答。
    对应 harness 的 ``retrieve`` 阶段。
:class:`ReasonerRole`
    拿到证据后做推理与答案综合。它**必须**引用证据，不能凭空回答。
    对应 harness 的 ``reason`` 阶段。
:class:`ReviewerRole`
    审核最终答案的事实性、引用完整性、合规性、口径一致性。它有权力要求
    上一环节重做，但不能自己编造证据。对应 harness 的 ``review`` 阶段。
:class:`OrchestratorRole`
    把请求路由到合适的角色，决定何时收工。它不做任何业务计算。

设计要点
--------
* 每个角色的提示词都用 :class:`RoleDefinition` 数据类描述，包含
  ``name`` / ``purpose`` / ``responsibilities`` / ``must_not`` /
  ``output_contract``。这不是文档装饰——**它是 Agent 的契约**，会进入
  harness 的系统提示与审计事件，运行期可见。
* 角色协作走"共享黑板"模式：一个 ``dict`` 承载各角色产出，角色只读写
  自己关心的键。这比直接传参更利于审计，也比多进程轻得多。
* 角色是**纯声明**，不持有 LLM 或工具引用。真正调用模型的是
  :class:`~src.rag.agent.Harness`，角色只提供"该怎么问模型"。这避免了
  "上帝对象"——一个角色同时知道怎么检索、怎么审核、怎么调模型。
"""

from __future__ import annotations

import dataclasses
import typing
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 契约：一个角色到底是干什么的
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RoleDefinition:
    """不可变的角色契约。

    ``must_not`` 和 ``output_contract`` 不是建议——harness 会读取它们来做
    校验与审计。把"禁止做什么"写成数据，而不是散落在提示词字符串里，
    才能被单元测试覆盖。
    """

    name: str
    title: str
    purpose: str
    responsibilities: tuple[str, ...]
    must_not: tuple[str, ...]
    output_contract: str

    def describe(self) -> dict[str, typing.Any]:
        return {
            "name": self.name,
            "title": self.title,
            "purpose": self.purpose,
            "responsibilities": list(self.responsibilities),
            "must_not": list(self.must_not),
            "output_contract": self.output_contract,
        }


@dataclasses.dataclass(frozen=True)
class AgentDefinition:
    """整个 Agent 的对外声明。

    这是回答"这个 Agent 到底是干什么的"的**唯一权威来源**。它被序列化成
    API 的 ``/agent/definition`` 响应、写入审计事件、也进入 harness 的
    系统提示。三层来源一致，避免"文档说一套、代码做一套"。
    """

    name: str
    version: str
    product_purpose: str
    input_contract: str
    output_contract: str
    roles: tuple[Role, ...]
    escalation: str

    def role_names(self) -> list[str]:
        return [role.definition.name for role in self.roles]

    def describe(self) -> dict[str, typing.Any]:
        return {
            "name": self.name,
            "version": self.version,
            "product_purpose": self.product_purpose,
            "input_contract": self.input_contract,
            "output_contract": self.output_contract,
            "roles": [role.definition.describe() for role in self.roles],
            "escalation": self.escalation,
        }


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------


class Role(ABC):
    """一个职责单一的 Agent 角色。

    子类只实现两件事：声明自己是谁（``definition``），以及把自己的任务
    描述成模型能执行的提示词（``build_prompt``）。真正调模型、解析输出
    的是 harness，子类不持有模型引用。
    """

    definition: RoleDefinition

    @abstractmethod
    def build_prompt(self, blackboard: Blackboard) -> str:
        """根据本角色关心的上下文，构造一次模型调用要用的系统提示片段。"""

    def can_handle(self, blackboard: Blackboard) -> bool:
        """该角色是否应该在这一轮介入。默认：永远介入。"""

        return True

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.definition.name!r})"


class Blackboard(BaseModel):
    """角色间共享的、可审计的工作区。

    键的命名即协议：``original_question`` / ``rewritten_queries`` /
    ``evidence`` / ``answer`` / ``review``。角色只读写自己声明的键，这
    是一种廉价的、可读的协作契约，比强类型通道更适合迭代。
    """

    request_id: str
    original_question: str = ""
    tenant_id: str = "default"
    rewritten_queries: list[str] = Field(default_factory=list)
    evidence: list[dict[str, typing.Any]] = Field(default_factory=list)
    answer: str | None = None
    answer_draft: str | None = None
    review: dict[str, typing.Any] | None = None
    review_rounds: int = 0
    roles_invoked: list[str] = Field(default_factory=list)
    errors: list[dict[str, typing.Any]] = Field(default_factory=list)
    metadata: dict[str, typing.Any] = Field(default_factory=dict)

    def record_error(self, stage: str, message: str) -> None:
        self.errors.append({"stage": stage, "message": message})

    def snapshot(self) -> dict[str, typing.Any]:
        """返回一份可序列化的副本，用于审计与 API 响应。"""

        return self.model_dump(exclude_none=True)


# ---------------------------------------------------------------------------
# 具体角色
# ---------------------------------------------------------------------------


class RetrieverRole(Role):
    """检索规划者：决定检索什么、验证召回是否足够。

    它不做答案综合。召回不足时必须明确标记，而不是让下游硬编。
    """

    definition = RoleDefinition(
        name="retriever",
        title="检索规划者",
        purpose="把用户问题转化为可执行的检索策略，并判断召回结果是否足以支撑回答。",
        responsibilities=(
            "规划检索查询，必要时做多查询改写",
            "汇总多路召回，去重并保留来源元信息",
            "判断证据是否充足、来源是否可信",
        ),
        must_not=(
            "不得自行编造或补全文档中不存在的事实",
            "不得在证据不足时给出确定性结论",
            "不得绕过权限范围扩大检索",
        ),
        output_contract="一份带来源的候选证据列表，以及一条充足性判定。",
    )

    def build_prompt(self, blackboard: Blackboard) -> str:
        queries = blackboard.rewritten_queries or [blackboard.original_question]
        return (
            "你是企业知识库的检索规划者。你的唯一产出是检索计划与证据清单，不要回答用户。\n"
            f"原始问题：{blackboard.original_question}\n"
            f"已生成的检索查询：{queries}\n"
            "请说明：1) 你打算如何拆分这个问题；2) 哪些信息是当前证据无法覆盖的。"
        )

    def can_handle(self, blackboard: Blackboard) -> bool:
        # 已经有答案草稿时不再检索
        return blackboard.answer_draft is None


class ReasonerRole(Role):
    """答案综合者：基于证据生成答案，必须引用来源。

    这是用户最终看到的"回答者"。它的硬约束是：没有证据就承认不知道。
    """

    definition = RoleDefinition(
        name="reasoner",
        title="答案综合者",
        purpose="基于已授权证据综合答案，并逐条引用来源。",
        responsibilities=(
            "只依据已提供的证据作答",
            "为关键结论标注来源编号",
            "证据冲突时并列呈现，不擅自选择",
        ),
        must_not=(
            "不得引用证据之外的知识或常识补全",
            "不得编造来源编号",
            "不得输出未经验证的确定性结论",
        ),
        output_contract="一段带来源标注的自然语言答案，或明确的无法回答声明。",
    )

    def build_prompt(self, blackboard: Blackboard) -> str:
        evidence = blackboard.evidence or []
        evidence_text = "\n".join(
            f"[{index + 1}] {item.get('source', '未知来源')}: "
            f"{(item.get('snippet') or item.get('content') or '')[:500]}"
            for index, item in enumerate(evidence)
        ) or "（当前无可用证据）"
        return (
            "你是企业知识库的答案综合者。只依据下列带编号证据作答，并为关键结论标注编号。\n"
            "证据之外的信息一律不得使用。证据不足时，明确说明无法确认。\n"
            "证据之间存在冲突时，并列呈现不同说法，不要自行裁定。\n\n"
            f"原始问题：{blackboard.original_question}\n\n"
            f"证据：\n{evidence_text}\n\n"
            "请给出答案："
        )

    def can_handle(self, blackboard: Blackboard) -> bool:
        return bool(blackboard.evidence)


class ReviewerRole(Role):
    """答案审核者：事实性、引用完整性、合规性、口径一致性。

    审核是**结构化的、可重试的**——它会返回一组 :class:`ReviewIssue`，
    而不是一段不可解析的评语。这让它可以被单元测试覆盖，也让 harness
    能据此决定是否重做。
    """

    definition = RoleDefinition(
        name="reviewer",
        title="答案审核者",
        purpose="对答案做发布前质检，输出可机器处理的问题清单。",
        responsibilities=(
            "核对答案中的每个断言是否都有对应证据",
            "检查引用编号是否真实存在且指向正确片段",
            "检查是否存在越权披露、敏感信息泄露、合规风险",
            "检查术语与数值是否与证据一致",
        ),
        must_not=(
            "不得自行补充或改写答案内容",
            "不得引入新的事实（只做判定）",
            "不得绕过既定的拒答策略",
        ),
        output_contract=(
            "一个 JSON 对象：{\"approved\": bool, \"issues\": ["
            "{\"code\": str, \"severity\": \"blocking\"|\"warning\", "
            "\"message\": str, \"suggestion\": str}], \"summary\": str}。"
        ),
    )

    def build_prompt(self, blackboard: Blackboard) -> str:
        evidence = blackboard.evidence or []
        evidence_text = "\n".join(
            f"[{index + 1}] {item.get('source', '未知来源')}: "
            f"{(item.get('snippet') or item.get('content') or '')[:400]}"
            for index, item in enumerate(evidence)
        ) or "（无证据）"
        return (
            "你是企业知识库的发布前审核者。你只能判定，不能改写答案。\n"
            "逐项检查：①每个断言是否有对应证据；②引用编号是否存在且指向正确片段；"
            "③是否泄露敏感信息或越权披露；④术语与数值是否与证据一致。\n"
            "严格只输出 JSON，键为 approved(boolean)、issues(array)、summary(string)。"
            "issues 中每条含 code、severity（blocking/warning）、message、suggestion。\n"
            f"原始问题：{blackboard.original_question}\n"
            f"答案：\n{blackboard.answer_draft or ''}\n\n"
            f"证据：\n{evidence_text}\n"
        )


class OrchestratorRole(Role):
    """请求编排者：决定下一步该谁上。

    它把请求路由到合适的角色、维护轮次预算、决定何时收工。它自己不计算。
    """

    definition = RoleDefinition(
        name="orchestrator",
        title="请求编排者",
        purpose="按角色契约编排多角色协作，并控制整体执行预算。",
        responsibilities=(
            "决定下一轮由哪个角色处理",
            "维护轮次与重试预算",
            "在角色全部完成后汇总最终产出",
        ),
        must_not=(
            "不得自行检索或生成业务答案",
            "不得绕过某个角色的否决（如审核不通过）",
            "不得泄露内部角色分工给最终用户",
        ),
        output_contract="一个执行计划与每轮的路由决策记录。",
    )

    def build_prompt(self, blackboard: Blackboard) -> str:
        return (
            "你是请求编排者。只输出本轮应调度的角色与理由，不要回答业务问题。\n"
            f"当前黑板状态：问题={blackboard.original_question!r}, "
            f"已有证据={len(blackboard.evidence)}条, "
            f"已有草稿={'是' if blackboard.answer_draft else '否'}, "
            f"审核轮次={blackboard.review_rounds}。"
        )


# ---------------------------------------------------------------------------
# Agent 声明（系统唯一权威来源）
# ---------------------------------------------------------------------------


def enterprise_qa_definition() -> AgentDefinition:
    """构造本系统的 Agent 声明。

    集中在一处定义，避免"文档、提示词、API 响应"三套口径互相打架。
    """

    return AgentDefinition(
        name="enterprise-knowledge-agent",
        version="1.0.0",
        product_purpose=(
            "基于企业授权知识库的检索增强问答：先检索、再推理、最后审核，"
            "所有对外答案均须可追溯至已授权文档。"
        ),
        input_contract="用户自然语言问题，可选附带租户、用户、来源白名单。",
        output_contract="带来源引用的答案，或明确的无法确认声明；附带完整执行轨迹。",
        roles=(
            OrchestratorRole(),
            RetrieverRole(),
            ReasonerRole(),
            ReviewerRole(),
        ),
        escalation=(
            "审核未通过时最多重试一次（回退到重新推理）；仍不通过则按策略返回"
            "安全拒答，并在响应中标记审核状态。"
        ),
    )


__all__ = [
    "AgentDefinition",
    "Blackboard",
    "OrchestratorRole",
    "ReasonerRole",
    "RetrieverRole",
    "ReviewerRole",
    "Role",
    "RoleDefinition",
    "enterprise_qa_definition",
]
