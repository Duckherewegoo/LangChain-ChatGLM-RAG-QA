"""多模型路由的纯数据类型定义。

本模块刻意不依赖 settings 或 model.manager。
其存在的目的是为了打破 settings -> router -> manager -> settings 的循环导入链。

由于 Pydantic v2 的 BaseSettings 在解析前向引用失败时会拒绝实例化，
将路由数据结构放置在独立的叶子模块中，是保持依赖方向单向且干净的唯一解法。
"""

from __future__ import annotations

import typing

from pydantic import BaseModel, Field, model_validator

if typing.TYPE_CHECKING:
    from settings import ModelChatSettings


class ModelCapabilitySettings(BaseModel):
    """模型需具备的能力声明。"""

    supports_tools: bool = False
    supports_streaming: bool = True
    max_context_tokens: int | None = None


class ModelRoutingEntry(BaseModel):
    """单个模型的路由配置元数据。

    将配置转换为具体的 ModelChatSettings 的逻辑通过 to_provider_settings 方法实现。
    该方法内部采用局部导入以彻底避免循环依赖。
    """

    name: str = Field(..., min_length=1)
    provider: str = Field(..., min_length=1)
    model_name: str = Field(..., min_length=1)
    base_url: str | None = None
    api_key: str = ""

    temperature: float = Field(default=0.2, ge=0.0)
    max_tokens: int | None = Field(default=None, gt=0)
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=1, ge=0)

    capabilities: ModelCapabilitySettings = Field(default_factory=ModelCapabilitySettings)

    def to_provider_settings(self) -> "ModelChatSettings":
        """转换为具体厂商可用的底层 ModelChatSettings。

        采用局部导入以避免 routing_types -> settings -> routing_types 的循环导入。
        """
        from settings import ModelChatSettings

        return ModelChatSettings(
            provider=self.provider,
            model_name=self.model_name,
            base_url=self.base_url,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
        )


class UseCaseBinding(BaseModel):
    """将业务用例映射到一个有序的模型调用链。"""

    use_case: str = Field(..., min_length=1)
    primary: str = Field(..., min_length=1)
    fallbacks: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_chain(self) -> UseCaseBinding:
        """校验主模型和备用模型之间不能重复。"""
        chain = [self.primary, *self.fallbacks]
        if len(chain) != len(set(chain)):
            raise ValueError("主模型和备用模型列表中不能包含重复的模型名称。")
        return self


class ModelRouterSettings(BaseModel):
    """包含命名模型与用例绑定的路由配置容器。"""

    models: dict[str, ModelRoutingEntry] = Field(default_factory=dict)
    use_cases: dict[str, UseCaseBinding] = Field(default_factory=dict)
    default_use_case: str = "chat"

    @model_validator(mode="after")
    def _validate_use_cases(self) -> ModelRouterSettings:
        """校验用例绑定的有效性。"""
        for binding_key, binding in self.use_cases.items():
            if binding_key != binding.use_case:
                raise ValueError(
                    f"用例字典的键 {binding_key!r} 与内部定义的 use_case {binding.use_case!r} 不一致。"
                )

            for reference in [binding.primary, *binding.fallbacks]:
                if reference not in self.models:
                    raise ValueError(
                        f"用例 {binding.use_case!r} 引用了未定义的模型 {reference!r}。"
                    )

        if self.use_cases and self.default_use_case not in self.use_cases:
            raise ValueError(
                f"默认用例 {self.default_use_case!r} 未在 use_cases 中找到绑定。"
            )
        return self


class ModelEmbeddingSettings(BaseModel):
    """Embedding 模型参数。"""

    provider: str = Field(
        default="local",
        min_length=1,
        description="Embedding 提供者，预留用于未来的远程嵌入模型。",
    )
    model_name: str = Field(default="BAAI/bge-small-zh-v1.5", min_length=1)
    device: str = Field(default="cpu", min_length=1)
    batch_size: int = Field(default=32, gt=0, le=512)
    normalize: bool = Field(
        default=True, description="是否对向量进行 L2 归一化以适配余弦相似度计算。"
    )


__all__ = [
    "ModelCapabilitySettings",
    "ModelEmbeddingSettings",
    "ModelRouterSettings",
    "ModelRoutingEntry",
    "UseCaseBinding",
]
