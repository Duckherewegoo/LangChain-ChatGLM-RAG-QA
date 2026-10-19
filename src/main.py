"""应用入口。

本模块组装可观测性、配置与 FastAPI 应用，并启动 Uvicorn 服务。
进程生命周期错误会被显式记录；在绑定端口之前不会静默崩溃。
"""

from __future__ import annotations

import sys

from src.observability import (
    configure_observability,
    get_logger,
    shutdown_observability,
)
from src.settings import load_settings

from dotenv import load_dotenv
load_dotenv()  # 读取 .env

logger = get_logger(__name__)


def main() -> int:
    """加载配置、构建应用并运行 HTTP 服务。"""

    try:
        settings = load_settings()
    except Exception as exc:
        print(f"[FATAL] 配置加载失败: {exc}", file=sys.stderr)
        return 2

    # 先配置可观测性，后续所有错误都能进入统一的日志管道
    configure_observability(
        settings.log_level, settings.observability.metrics_enabled
    )

    try:
        from src.api.server import create_app
        from src.rag.container import initialize_container
    except Exception as exc:
        logger.exception(f"应用初始化失败 | error={exc}")
        return 3

    try:
        container = initialize_container(settings)
        app = create_app(container)
    except Exception as exc:
        logger.exception(f"容器或应用装配失败 | error={exc}")
        return 3

    import uvicorn

    try:
        uvicorn.run(
            app,
            host=settings.api.host,
            port=settings.api.port,
            log_level=settings.log_level.lower(),
            access_log=True,
        )
    except KeyboardInterrupt:
        logger.info("应用被中断")
        return 0
    except Exception as exc:
        logger.exception(f"服务启动失败 | error={exc}")
        return 4
    finally:
        try:
            shutdown_observability()
        except Exception as exc:
            print(f"[WARN] 可观测性关闭失败: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
