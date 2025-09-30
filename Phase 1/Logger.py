"""Logging facade for the Phase 1 trading infrastructure prototype.

The README identifies logging as a core dependency required by multiple layers
of the bot stack.  This module keeps the logging contract compact and focused:
components can depend on the :class:`Logger` interface without committing to a
specific backend while the default implementation delegates to Python's
:mod:`logging` module.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional


class Logger(ABC):
    """Minimal logging interface shared across the application.

    The methods intentionally mirror the most common severity levels required
    for early experimentation.  Additional levels (debug, critical, …) can be
    added once the prototype matures.
    """

    @abstractmethod
    def info(self, message: str, *, extra: Optional[dict] = None) -> None:
        """Log an informational *message* about the bot's current state."""

    @abstractmethod
    def warn(self, message: str, *, extra: Optional[dict] = None) -> None:
        """Log a warning *message* to highlight recoverable issues."""

    @abstractmethod
    def error(
        self,
        message: str,
        *,
        exc: Optional[BaseException] = None,
        extra: Optional[dict] = None,
    ) -> None:
        """Log an error *message* optionally enriched by the triggering *exc*."""


class StandardLogger(Logger):
    """Concrete :class:`Logger` implementation delegating to ``logging``.

    Parameters
    ----------
    name:
        Logger name used by :func:`logging.getLogger`.  Naming loggers by module
        keeps log output organised once the project grows beyond Phase 1.
    level:
        Optional log level to configure the underlying handler.
    """

    def __init__(self, name: str = "project_ex_nihilo", level: int = logging.INFO) -> None:
        self._logger = logging.getLogger(name)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
        self._logger.setLevel(level)

    def info(self, message: str, *, extra: Optional[dict] = None) -> None:
        self._logger.info(message, extra=extra)

    def warn(self, message: str, *, extra: Optional[dict] = None) -> None:
        self._logger.warning(message, extra=extra)

    def error(
        self,
        message: str,
        *,
        exc: Optional[BaseException] = None,
        extra: Optional[dict] = None,
    ) -> None:
        self._logger.error(message, exc_info=exc, extra=extra)


def get_logger(name: str = "project_ex_nihilo", level: int = logging.INFO) -> Logger:
    """Factory helper returning a :class:`Logger` instance.

    Using a dedicated function avoids scattering implementation details across
    the code base.  Consumers can later swap this for a more advanced logging
    provider without refactoring every import.
    """

    return StandardLogger(name=name, level=level)