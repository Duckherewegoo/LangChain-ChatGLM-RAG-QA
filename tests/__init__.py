"""企业 RAG 测试套件。

本包标记 ``tests/`` 为常规 Python 包（而非 namespace package），
从而让 ``tests._support`` 等共享测试工具可以通过标准导入访问，
无需在 ``conftest.py`` 中通过 ``importlib`` 手工注入 ``sys.modules``。

子目录说明：
  - ``ab_test/``      A/B 对比测试（策略 × 用例矩阵）
  - ``_support.py``   共享的 Fake 实现（模型、Embedding、向量库）
  - ``conftest.py``   全局 fixture 与 provider 生命周期管理
"""
