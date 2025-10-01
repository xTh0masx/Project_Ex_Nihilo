"""Market data interfaces for the early research phase of the project.

The README describes a ''DataFeed'' abstraction that the rest of the platform
relies on for market snapshots and historical candles. This module codifies the
contract and provides leight-weigt event containers. Concrete exchange
integrations (Binance, Bybit, Yahoo Finance, ...) can be implemented later while
staying compatible with the Strategy and Storage Layer.
"""

from __future__ import annotations

import importlib.util
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

_BINANCE_SPEC = importlib.util.find_spec("binance")

if _BINANCE_SPEC is not None: # pragma: no cover - optional dependency
    from binance.client import Client
    from binance.streams import ThreadedWebsocketManager
else: # pragma: no cover - executed when dependency is missing
    Client = ThreadedWebsocketManager = None # type: ignore

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
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> Iterable[CandleEvent]:\
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
    """Binance-backend implementation of :class: 'DataFeed'.

    the feed wraps the 'python-binance <https://binance.readthedocs.io/>'_clients
    to access the public REST and WebSocket endpoints of Binance:

    * REST ''GET /api/v3/klines'' (via :meth:'binance.client.Client.get_klines')
    for historical OHLCV data.
    * WebSocket streams ''btcusdt@trade'' and ''btcusdt@kline_<inverval>'' (via
    :class: 'binance.streams.ThreadedWebsocketManager') for live ticker and candle updates.

    The REST client requires an API key only for private endpoints; the implemented
    methods work with public market data and therefore operate without credentials.
    Install ''python-binance'' and optionally provide ''api_key''/''api_secret'' when
    instantiating the class to enable authenticated requests or user-specific rate limits.
    """

    def __init__(
            self,
            api_key: Optional[str] = None,
            api_secret: Optional[str] = None,
            base_url: Optional[str] = None,
            enable_websockets: bool = True,
    ) -> None:
        if Client is None: # pragma: no cover - dependency guard
            raise RuntimeError(
                "python-binance is required for BinanceDataFeed. Install it via 'pip install python-binance'."
            )

        self.rest_client = Client(
            api_key=api_key,
            api_secret=api_secret,
            requests_params={"timeout": 10},
            base_url=base_url,
        )
        self._ws_manager: Optional[ThreadedWebsocketManager] = None
        self._ws_running = False

        if enable_websockets:
            if ThreadedWebsocketManager is None: # pragma: no cover - dependency guard

                raise RuntimeError(
                    "python-binance is required for BinanceDataFeed. "
                )
            self._ws_manager = ThreadedWebsocketManager(
                api_key=api_key,
                api_secret=api_secret,
                tld="com",
                testnet=base_url is not None,
            )
            self._ws_manager.start()
            self._ws_running = True

    # -----------------------------------------------------------------------
    def _ensure_websocket(self) -> ThreadedWebsocketManager:
        if self._ws_manager is None or not self._ws_running:
            raise RuntimeError("Websocket support is disabled for this BinanceDataFeed instance.")
        return self._ws_manager

    def _stream_symbol(self, symbol: str) ->str:
        return symbol.lower()

    def _handle_trade(self, symbol: str) -> Callable[[dict], None]:
        uppercase = symbol.upper()

        def _callback(message: dict) -> None:
            price = float(message["p"])
            volume = float(message.get("q", 0.0)) or None
            event = TickerEvent(
                ts=int(message["T"]),
                symbol=uppercase,
                price=price,
                volume=volume,
            )
            self.on_ticker(event)
        return _callback

    def _handle_kline(self, symbol: str) -> Callable[[dict], None]:
        uppercase = symbol.upper()

        def _callback(message: dict) -> None:
            payload = message["k"]
            event = CandleEvent(
                ts=int(payload["t"]),
                symbol=uppercase,
                o=float(payload["o"]),
                h=float(payload["h"]),
                l=float(payload["l"]),
                c=float(payload["c"]),
                v=float(payload["v"]),
            )
            self.on_ohlcv(event)

        return _callback

    # --------------------------------------------------------------------------------
    def subscribe_ticker(self, symbol:str) ->None:
        manager = self._ensure_websocket()
        manager.start_trade_socket(
            callback=self._handle_trade(symbol),
            symbol=self._stream_symbol(symbol),
        )

    def subscribe_ohlcv(self, symbol: str, timeframe: str) -> None: # pragma: no cover - placeholder
        manager = self._ensure_websocket()
        manager.start_kline_socket(
            callback=self._handle_kline(symbol),
            symbol=self._stream_symbol(symbol),
            interval=timeframe,
        )

    # -------------------------------------------------------------------------------
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> Iterable[CandleEvent]:
        candles = self._rest_client.get_klines(
            symbol=symbol.upper(),
            interval=timeframe,
            limit=limit,
        )
        for open_time, o, h, l, c, v, *_ in candles:
            yield CandleEvent(
                ts=int(open_time),
                symbol=symbol.upper(),
                o=float(o),
                h=float(h),
                l=float(l),
                c=float(c),
                v=float(v),
            )
    # --------------------------------------------------------------------------------
    def close(self) -> None: # pragma: no cover - placeholder
        if self._ws_manager is not None and self._ws_running:
            self._ws_manager.stop()
            self._ws_running = False
        self.rest_client.close_connection()



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