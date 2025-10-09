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

    # ---------------------------------------------------------------------
    def ping(self) -> bool:  # pragma: no cover - default implementation
        """Health probe hook allowing orchestrators to monitor the storage backend."""

        return True

def _default_logger(name: str) -> "logging.Logger":
    """Return a project logger compatible with :class:`Logger`."""

    try:
        from Logger import get_logger  # type: ignore
    except Exception:  # pragma: no cover - logging fallback
        base_logger = logging.getLogger(name)
        if not base_logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            base_logger.addHandler(handler)
        base_logger.setLevel(logging.INFO)

        class _Adapter:
            def __init__(self, logger: "logging.Logger") -> None:
                self._logger = logger

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

        return _Adapter(base_logger)
    else:
        return get_logger(name)

try:  # pragma: no cover - optional dependency
    import mysql.connector  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    mysql = None  # type: ignore
else:  # pragma: no cover - optional dependency
    mysql = mysql.connector  # type: ignore

class MySQLStorage(Storage):
    """Lightweight MySQL-backed storage with in-memory fallbacks for tests."""

    def __init__(
            self,
            *,
            dsn: Optional[Dict[str, Any]] = None,
            connection: Optional[Any] = None,
            logger: Optional[Any] = None,
            metrics_sink: Optional[Callable[[str, float], None]] = None,
    ) -> None:
        self._logger = logger or _default_logger("project_ex_nihilo.storage.mysql")
        self._metrics_sink = metrics_sink
        self._dsn = dsn or {}
        self._connection = connection or self._connect(self._dsn)
        self._ohlcv: Dict[Tuple[str, str], List[Any]] = {}
        self._trades: List[TradeRecord] = []
        self._models: Dict[str, Any] = {}
        self._candles_counter = 0

    # ------------------------------------------------------------------
    def save_ohlcv(self, symbol: str, timeframe: str, data: Any) -> None:
        key = (symbol.upper(), timeframe)
        batch: List[Any]
        if isinstance(data, Iterable) and not isinstance(data, (str, bytes)):
            batch = list(data)
        else:
            batch = [data]

        stored = self._ohlcv.setdefault(key, [])
        stored.extend(batch)
        self._candles_counter += len(batch)
        self._emit_metric("storage.candles_stored", float(self._candles_counter))
        self._logger.info(
            "Stored OHLCV batch",
            extra={"symbol": symbol.upper(), "timeframe": timeframe, "count": len(batch)},
        )

    def load_ohlcv(self, symbol: str, timeframe: str) -> List[Any]:
        key = (symbol.upper(), timeframe)
        return list(self._ohlcv.get(key, []))

    def save_trade(self, trade: TradeRecord) -> None:
        self._trades.append(trade)

    def save_model(self, tag: str, obj: Any) -> None:
        self._models[tag] = obj

    def load_model(self, tag: str) -> Any:
        return self._models.get(tag)

    # ------------------------------------------------------------------
    def ping(self) -> bool:
        connection = self._connection
        if connection is None:
            return True

        ping = getattr(connection, "ping", None)
        if callable(ping):
            try:
                ping(reconnect=True)  # type: ignore[call-arg]
                return True
            except TypeError:  # pragma: no cover - connector specific signature
                try:
                    ping()
                    return True
                except Exception as exc:  # pragma: no cover - connector specific failure
                    self._logger.error("MySQL ping failed", exc=exc)
                    return False
            except Exception as exc:  # pragma: no cover - connector specific failure
                self._logger.error("MySQL ping failed", exc=exc)
                return False

        return True

    def metrics_snapshot(self) -> Dict[str, Any]:
        return {"candles_stored": self._candles_counter}

    # ------------------------------------------------------------------
    def _emit_metric(self, name: str, value: float) -> None:
        if self._metrics_sink is None:
            return
        try:
            self._metrics_sink(name, value)
        except Exception:  # pragma: no cover - metrics hooks must not crash ingestion
            self._logger.warn("Metrics sink raised an exception", extra={"metric": name})

    def _connect(self, dsn: Dict[str, Any]) -> Optional[Any]:
        if not dsn:
            return None
        if mysql is None:
            self._logger.warn("mysql-connector not installed; using in-memory storage only")
            return None
        try:
            return mysql.connect(**dsn)
        except Exception as exc:  # pragma: no cover - connector runtime dependent
            self._logger.error("Failed to establish MySQL connection", exc=exc)
            return None