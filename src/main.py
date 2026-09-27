"""Application entrypoint.

This module wires observability, configuration, and the FastAPI application, then
starts a Uvicorn server. Process lifecycle errors are logged explicitly; the
application never crashes silently before binding a port.
"""

from __future__ import annotations

import sys

from src.observability import (
    configure_observability,
    get_logger,
)

# Conditional import: shutdown_observability is optional — if the observability
# module is a stripped-down build (e.g. embedded deployment), we still start.
try:
    from src.observability import shutdown_observability
except ImportError:
    shutdown_observability = None  # type: ignore[assignment]

logger = get_logger(__name__)


def main() -> int:
    """Load configuration, build the app, and run the HTTP server."""

    try:
        settings = load_settings()
    except Exception as exc:
        print(f"[FATAL] configuration error: {exc}", file=sys.stderr)
        return 2

    configure_observability(settings.log_level, settings.observability.metrics_enabled)
    try:
        from src.api.server import create_app
        from src.rag.container import initialize_container
    except Exception as exc:
        logger.exception("application_failed_to_initialize", extra={"error": str(exc)})
        return 3

    container = initialize_container(settings)
    app = create_app(container)

    import uvicorn

    server_logger = logger.bind(component="uvicorn") if hasattr(logger, "bind") else logger
    try:
        uvicorn.run(
            app,
            host=settings.host,
            port=settings.port,
            log_level=settings.log_level.lower(),
            access_log=True,
        )
    except KeyboardInterrupt:
        logger.info("application_interrupted")
        return 0
    except Exception as exc:
        server_logger.exception("server_failed_to_start", error=str(exc))
        return 4
    finally:
        if shutdown_observability is not None:
            try:
                shutdown_observability()
            except Exception as exc:
                print(f"[WARN] observability shutdown failed: {exc}", file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
