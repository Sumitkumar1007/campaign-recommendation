from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from project_paths import LOG_DIR, ensure_parent_dir


def setup_logging(log_file: str | Path | None, logger_name: str) -> logging.Logger:
    if log_file is None:
        log_path = LOG_DIR / f"{logger_name}.log"
    else:
        log_path = Path(log_file)

    ensure_parent_dir(log_path)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info("Logging initialized. log_file=%s", log_path)
    return logger


@contextmanager
def log_step(logger: logging.Logger, step_name: str, **fields: object) -> Iterator[None]:
    field_text = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("START %s%s", step_name, f" | {field_text}" if field_text else "")
    start = time.perf_counter()
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - start
        logger.exception("FAILED %s | elapsed_seconds=%.2f", step_name, elapsed)
        raise
    else:
        elapsed = time.perf_counter() - start
        logger.info("END %s | elapsed_seconds=%.2f", step_name, elapsed)
