from __future__ import annotations

import logging as py_logging


def get_logger(name: str = "optimal_quoting", level: int = py_logging.INFO) -> py_logging.Logger:
    """
    Small project logger helper.
    Uses stdlib logging and avoids duplicate handlers.
    """
    logger = py_logging.getLogger(name)

    if logger.handlers:
        logger.setLevel(level)
        return logger

    logger.setLevel(level)

    handler = py_logging.StreamHandler()
    fmt = py_logging.Formatter("[%(levelname)s] %(name)s - %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)

    logger.propagate = False
    return logger
