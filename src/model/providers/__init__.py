"""模型提供者实现包。

导入此包将注册所有内置的模型提供者。
一个提供者只有在其对应的模块被导入后才会被注册，以此隔离可选 SDK 的依赖。
"""

from model.providers import (  # noqa: F401 - 仅用于触发注册副作用
    deepseek,
    doubao,
    openai_compatible,
    qwen,
    zhipu,
)
