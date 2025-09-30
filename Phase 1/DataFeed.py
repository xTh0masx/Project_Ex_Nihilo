"""Market data interfaces for the early research phase of the project.

The README describes a ''DataFeed'' abstraction that the rest of the platform
relies on for market snapshots and historical candles. This module codifies the
contract and provides leight-weigt event containers. Concrete exchange
integrations (Binance, Bybit, Yahoo Finance, ...) can be implemented later while
staying compatible with the Strategy and Storage Layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Optional

@dataclass(slots=True)
class TickerEvent:
    """Lightweight representation of a live ticker update."""

    ts: int
    symbol: str
    price: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume: Optional[float] = None

@dataclass(slots=True)
class CandleEvent:
    """Container for OHLCV data compatible with the project class diagram."""

    ts: int
    symbol: str
    o: float
    h: float
    l: float
    c: float
    v: float

class DataFeed(ABC):
    """Abstract base class for market data providers."""

    @abstractmethod
    def subscribe_ticker(self, symbol: str) -> None:
        """Subscribe to live ticker updates for *symbol*."""

    @abstractmethod
    def subscribe_ohlcv(self, symbol: str, timeframe: str) -> None:
        """Subscribe to OHLCV updates for *symbol* at the desired *timeframe*."""

    @abstractmethod
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500): -> Iterable[CandleEvent]:
        """Fetch historical candles for *symbol* and *timeframe*."""

    @abstractmethod
    def close(self) -> None:
        """Release network connections and other resources."""

    # --------------------------------------------------------------------------------------------------
    def on_ticker(self, event: TickerEvent) -> None: # pragma: no cover - hook point
        """Event hook triggered for every incoming :class:`TickerEvent`."""

    def on_ohlcv(self, event: CandleEvent) -> None: # pragma: no cover - hook point
        """Event hook triggered for every incoming :class:`CandleEvent`."""


class BinanceDataFeed(DataFeed):
    """Placeholder for a Binance implementation.

    The real implementation will coordinate with REST/WebSocket clients to deliver updates in the format defined above
    """

    def subscribe_ticker(self, symbol: str) -> None: # pragma: no cover - placeholder
        raise NotImplementedError

    def subscribe_ohlcv(self, symbol: str, timeframe: str) -> None: # pragma: no cover - placeholder
        raise NotImplementedError

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> Iterable[CandleEvent]: # pragma: no cover
        raise NotImplementedError

    def close(self) -> None: # pragma: no cover - placeholder
        raise NotImplementedError



    class BybitDataFeed(DataFeed):
        """Placeholder for a Bybit implementation."""

        def subscribe_ticker(self, symbol: str) -> None:  # pragma: no cover - placeholder
            raise NotImplementedError

        def subscribe_ohlcv(self, symbol: str, timeframe: str) -> None:  # pragma: no cover - placeholder
            raise NotImplementedError

        def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> Iterable[
            CandleEvent]:  # pragma: no cover
            raise NotImplementedError

        def close(self) -> None:  # pragma: no cover - placeholder
            raise NotImplementedError



    class YahooDataFeed(DataFeed):
        """Placeholder for a Yahoo Finance implementation."""

        def subscribe_ticker(self, symbol: str) -> None:  # pragma: no cover - placeholder
            raise NotImplementedError

        def subscribe_ohlcv(self, symbol: str, timeframe: str) -> None:  # pragma: no cover - placeholder
            raise NotImplementedError

        def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> Iterable[
            CandleEvent]:  # pragma: no cover
            raise NotImplementedError

        def close(self) -> None:  # pragma: no cover - placeholder
            raise NotImplementedError