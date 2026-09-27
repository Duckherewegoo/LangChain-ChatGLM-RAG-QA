"""Model provider implementations.

Importing this package registers all built-in providers. A provider is only
registered after its module is imported, isolating optional SDK dependencies.
"""

from src.model.providers import (  # noqa: F401  - registration side effects
    deepseek,
    doubao,
    openai_compatible,
    qwen,
    zhipu,
)
