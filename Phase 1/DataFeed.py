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
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

from Logger import Logger, get_logger

_BINANCE_SPEC = importlib.util.find_spec("binance")

if _BINANCE_SPEC is not None: # pragma: no cover - optional dependency
    from binance.client import Client
    from binance.error import BinanceAPIException
    from binance.streams import ThreadedWebsocketManager
else: # pragma: no cover - executed when dependency is missing
    Client = ThreadedWebsocketManager = None # type: ignore
    BinanceAPIException = Exception # type: ignore[assignment]

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

def _default_logger(name: str) -> logging.Logger:
    """Return a shared application logger, falling back to :mod: 'logging'."""

    try: # Prefer the project specific logger when available.
        from Logger import get_logger # type: ignore
    except Exception: # pragma: no cover - logging fallback
        base_logger = logging.getLogger(name)
        if not base_logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            base_logger.addHandler(handler)
        base_logger.setLevel(logging.INFO)

        class _Adapter:
            def __init__(self, logger: logging.Logger) -> None:
                self._logger = logger

            def info(self, message: str, *, extra: Optional[dict] = None) -> None:
                self._logger.info(message, extra=extra)

            def warn(self, message: str, *, extra: Optional[dict] = None) -> None:
                self._logger.warning(message, extra=extra)

            def error(self, message: str, *, exc: Optional[Exception] = None, extra: Optional[dict] = None) -> None:
                self._logger.error(message, exc_info=exc, extra=extra)

        return _Adapter(base_logger)
    else:
        return get_logger(name)

class DataFeed(ABC):
    """Abstract base class for market data providers."""

    def __init__(self) -> None:
        self._ticker_listeners: List[Callable[[TickerEvent], None]] = []
        self._ohlcv_listeners: List[Callable[[CandleEvent], None]] = []

    @abstractmethod
    def subscribe_ticker(self, symbol: str) -> None:
        """Subscribe to live ticker updates for *symbol*."""

    @abstractmethod
    def subscribe_ohlcv(self, symbol: str, timeframe: str) -> None:
        """Subscribe to OHLCV updates for *symbol* at the desired *timeframe*."""

    @abstractmethod
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> Iterable[CandleEvent]:
        """Fetch historical candles for *symbol* and *timeframe*."""

    @abstractmethod
    def close(self) -> None:
        """Release network connections and other resources."""

    # --------------------------------------------------------------------------------------------------
    def add_ticker_listener(self, callback: Callable[[TickerEvent], None]) -> None:
        """Register a listener invoked for every :class: 'TickerEvent' event."""

        self._ticker_listeners.append(callback)

    def add_ohlcv_listener(self, callback: Callable[[CandleEvent], None]) -> None:
        """Register a listener invoked for every :class: 'CandleEvent' event."""

        self._ohlcv_listeners.append(callback)

    def on_ticker(self, event: TickerEvent) -> None: # pragma: no cover - hook point
        """Event hook triggered for every incoming :class:`TickerEvent`."""

        for listener in list(self._ticker_listeners):
            listener(event)

    def on_ohlcv(self, event: CandleEvent) -> None: # pragma: no cover - hook point
        """Event hook triggered for every incoming :class:`CandleEvent`."""

        for listener in list(self._ohlcv_listeners):
            listener(event)

    def ping(self) -> bool: # pragma: no cover - default implementation
        """Health probe for the feed; sublasses can override with concrete logic."""
        return True


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
        logger: Logger = get_logger(),
        metrics_sink: Optional[Callable[[str, float], None]] = None,
    ) -> None:
        super().__init__()
        if Client is None: # pragma: no cover - dependency guard
            raise RuntimeError(
                "python-binance is required for BinanceDataFeed. Install it via 'pip install python-binance'."
            )

        self._logger = logger or _default_logger("project_ex_nihilo.datafeed.binance")
        self._metrics_sink = metrics_sink
        self._retry_attempts = 0
        self._socket_restarts: Dict[str, int] = {}

        self._rest_client = Client(
            api_key=api_key,
            api_secret=api_secret,
            requests_params={"timeout": 10},
            base_url=base_url,
        )
        self._logger.info(
            "Initialised Binance REST client",
            extra={"base_url": base_url or "https://api.binance.com"},
        )
        self._ws_manager: Optional[ThreadedWebsocketManager] = None
        self._ws_running = False
        self._socket_keys: Dict[str, str] = {}

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
            self._logger.info(
                "Started Binance WebSocket manager",
                extra={"websocket_enabled": True, "testnet": base_url is not None},
            )

    # -----------------------------------------------------------------------
    def _ensure_websocket(self) -> ThreadedWebsocketManager:
        if self._ws_manager is None or not self._ws_running:
            raise RuntimeError("Websocket support is disabled for this BinanceDataFeed instance.")
        return self._ws_manager

    def _stream_symbol(self, symbol: str) -> str:
        return symbol.lower()

    def _emit_metric(self, name: str, value: float) -> None:
        if self._metrics_sink is not None:
            try:
                self._metrics_sink(name, value)
            except Exception:  # pragma: no cover - metrics sinks must not crash ingestion
                self._logger.warn("Metrics sink raised an exception", extra={"metric": name})

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
            try:
                self.on_ticker(event)
            except Exception as exc: # pragma: no cover - defensive logging
                self._logger.error(
                    "Ticker callback raised an exception",
                    exc=exc,
                    extra={"symbol": uppercase},
                )
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
            try:
                self.on_ohlcv(event)
            except Exception as exc: # pragma: no cover - defensive logging
                self._logger.error(
                    "OHLCV callback raised an exception",
                    exc=exc,
                    extra={"symbol": uppercase},
                )

        return _callback

    # --------------------------------------------------------------------------------
    def subscribe_ticker(self, symbol:str) ->None:
        manager = self._ensure_websocket()
        handler = self._handle_trade(symbol)
        stream_id = self._stream_identifier("trade", symbol)

        def _starter(callback: Callable[[dict], None]) -> str:
            return manager.start_trade_socket(
                callback=callback,
                symbol=self._stream_symbol(symbol),
            )

        self._start_socket(stream_id, _starter, handler)


    def subscribe_ohlcv(self, symbol: str, timeframe: str) -> None: # pragma: no cover - placeholder
        manager = self._ensure_websocket()
        handler = self._handle_kline(symbol)
        stream_id = self._stream_identifier("kline", symbol, timeframe)

        def _starter(callback: Callable[[dict], None]) -> str:
            return manager.start_kline_socket(
                callback=callback,
                symbol=self._stream_symbol(symbol),
                interval=timeframe,
            )

        self._start_socket(stream_id, _starter, handler)

    # -------------------------------------------------------------------------------
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> Iterable[CandleEvent]:
        uppercase = symbol.upper()
        try:
            candles = self._rest_client.get_klines(
                symbol=symbol.upper(),
                interval=timeframe,
                limit=limit,
            )
        except BinanceAPIException as exc: # pragma: no cover - depends on external API
            self._logger.warn(
                "Revocerable Binance API error while fetching klines",
                extra={
                    "symbol": uppercase,
                    "timeframe": timeframe,
                    "limit": limit,
                    "status_code": getattr(exc, "status_code", None),
                    "error_code": getattr(exc, "code", None),
                },
            )
            raise
        if not candles:
            self._logger.warn(
                "Binance API returned no candles",
                extra={"symbol": uppercase, "timeframe": timeframe, "limit": limit},
            )
            return []

        for open_time, o, h, l, c, v, *_ in candles:
            yield CandleEvent(
                ts=int(open_time),
                symbol=uppercase,
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
        self._rest_client.close_connection()

        # --------------------------------------------------------------------------------

    def ping(self) -> bool:
        """Check connectivity with the Binance REST API."""

        try:
            self._with_retries(self._rest_client.ping, attempts=2, initial_delay=0.5)
            return True
        except Exception as exc:  # pragma: no cover - network interaction
            self._logger.error("Binance REST ping failed", exc=exc)
            return False

        # --------------------------------------------------------------------------------

    @property
    def retry_count(self) -> int:
        return self._retry_attempts

    def metrics_snapshot(self) -> Dict[str, Any]:
        """Return the latest ingestion metrics collected by the feed."""

        return {
            "rest_retries": self._retry_attempts,
            "socket_restarts": dict(self._socket_restarts),
            "active_sockets": len(self._socket_keys),
        }

    # --------------------------------------------------------------------------------
    def _stream_identifier(self, kind: str, symbol: str, timeframe: Optional[str] = None) -> str:
        base = f"{kind}:{symbol.upper()}"
        if timeframe:
            base = f"{base}:{timeframe}"
        return base

    def _start_socket(
            self,
            stream_id: str,
            starter: Callable[[Callable[[dict], None]], str],
            handler: Callable[[dict], None],
    ) -> None:
        def _bootstrap() -> None:
            callback = self._wrap_socket_callback(stream_id, handler, _bootstrap)
            key = starter(callback)
            self._socket_keys[stream_id] = key

        _bootstrap()

    def _wrap_socket_callback(
            self,
            stream_id: str,
            handler: Callable[[dict], None],
            starter: Callable[[], None],
    ) -> Callable[[dict], None]:
        def _callback(message: dict) -> None:
            if self._is_disconnect_message(message):
                self._logger.warn("WebSocket disconnected", extra={"stream": stream_id})
                self._restart_socket(stream_id, starter)
                return
            try:
                handler(message)
            except Exception as exc:  # pragma: no cover - handler level safety net
                self._logger.error(
                    "WebSocket handler failed; restarting stream",
                    exc=exc,
                    extra={"stream": stream_id},
                )
                self._restart_socket(stream_id, starter)

        return _callback

    def _restart_socket(self, stream_id: str, starter: Callable[[], None]) -> None:
        manager = self._ensure_websocket()
        key = self._socket_keys.get(stream_id)
        if key is not None:
            try:
                manager.stop_socket(key)
            except Exception:  # pragma: no cover - best effort cleanup
                self._logger.warn("Failed to stop websocket during restart", extra={"stream": stream_id})

        time.sleep(1.0)
        starter()
        self._socket_restarts[stream_id] = self._socket_restarts.get(stream_id, 0) + 1
        self._emit_metric("binance.socket_restart", float(self._socket_restarts[stream_id]))

    def _is_disconnect_message(self, message: Optional[dict]) -> bool:
        if message is None:
            return True
        if not isinstance(message, dict):
            return False
        if message.get("e") == "error":
            return True
        if message.get("code") in {"-1001", -1001}:  # Binance "Connection reset" code
            return True
        if message.get("m") == "Connection was closed":
            return True
        return False

    def _with_retries(
            self,
            func: Callable[[], Any],
            *,
            attempts: int = 3,
            initial_delay: float = 1.0,
            max_delay: float = 8.0,
    ) -> Any:
        delay = initial_delay
        last_exc: Optional[BaseException] = None
        for attempt in range(1, attempts + 1):
            try:
                return func()
            except Exception as exc:  # pragma: no cover - depends on network behaviour
                last_exc = exc
                self._retry_attempts += 1
                self._emit_metric("binance.rest_retry", float(self._retry_attempts))
                self._logger.warn(
                    "Binance REST request failed; retrying",
                    extra={"attempt": attempt, "max_attempts": attempts},
                )
                if attempt == attempts:
                    break
                time.sleep(delay)
                delay = min(delay * 2, max_delay)
        assert last_exc is not None
        self._logger.error(
            "Binance REST request exhausted retries",
            exc=last_exc,
        )
        raise last_exc