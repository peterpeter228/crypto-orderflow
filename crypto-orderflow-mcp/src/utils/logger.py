"""Backward-compatible logger import.

Some earlier revisions referenced ``src.utils.logger.get_logger``.
The canonical implementation lives in :mod:`src.utils.logging` and is re-exported
from :mod:`src.utils`.

Keeping this module avoids ``ModuleNotFoundError`` when users deploy mixed
versions or have stale imports.
"""

from __future__ import annotations

from .logging import get_logger, setup_logging

__all__ = [
    "get_logger",
    "setup_logging",
]
