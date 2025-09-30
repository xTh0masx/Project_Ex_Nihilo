"""Abstract persistence interfaces for the trading research platform.

The storage component is another core dependency highlighted by the README.  It
provides the bridge between the data layer and downstream strategy logic by
persisting raw market data, executed trades and trained models.  The concrete
implementation can vary—from local files during Phase 1 experiments to hosted
cloud solutions—therefore this module only defines the abstract contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol

try:  # Optional import; dataclasses are only available on Python >=3.7
    from dataclasses import dataclass
except ImportError:  # pragma: no cover - legacy Python guard
    dataclass = None  # type: ignore


if dataclass is not None:
    @dataclass
    class TradeRecord:
        """Lightweight trade representation used by the storage interface."""

        symbol: str
        side: str
        quantity: float
        price: float
        timestamp: int
else:  # pragma: no cover - legacy Python guard
    class TradeRecord(Protocol):  # type: ignore[misc]
        symbol: str
        side: str
        quantity: float
        price: float
        timestamp: int


class Storage(ABC):
    """Abstract interface for persisting and retrieving trading artefacts."""

    @abstractmethod
    def save_ohlcv(self, symbol: str, timeframe: str, data: Any) -> None:
        """Persist OHLCV *data* for a specific trading *symbol* and *timeframe*."""

    @abstractmethod
    def load_ohlcv(self, symbol: str, timeframe: str) -> Any:
        """Load previously saved OHLCV data for *symbol* and *timeframe*."""

    @abstractmethod
    def save_trade(self, trade: TradeRecord) -> None:
        """Persist a single executed *trade* event."""

    @abstractmethod
    def save_model(self, tag: str, obj: Any) -> None:
        """Store a serialisable model object identified by *tag*."""

    @abstractmethod
    def load_model(self, tag: str) -> Any:
        """Load a model previously stored under *tag*."""