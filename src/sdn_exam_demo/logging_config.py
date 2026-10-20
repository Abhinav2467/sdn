from __future__ import annotations

import logging
import os

DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(level: str | None = None) -> None:
    resolved = (level or os.getenv("SDN_DEMO_LOG_LEVEL") or "INFO").upper()
    logging.basicConfig(level=resolved, format=DEFAULT_LOG_FORMAT)
