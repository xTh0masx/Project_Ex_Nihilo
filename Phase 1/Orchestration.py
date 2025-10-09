"""Coordinator that wires together data feeds, storage and monitoring hooks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover - used only for static type checking
    from DataFeed import CandleEvent, DataFeed
    from Logger import Logger
    from Storage import Storage


def _default_logger(name: str) -> "logging.Logger":
    try:
        from Logger import get_logger  # type: ignore
    except Exception:  # pragma: no cover - fallback when custom logger is unavailable
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


class DataIngestionOrchestrator:
    """High-level orchestrator coordinating feeds, storage and monitoring."""

    def __init__(
        self,
        feed: "DataFeed",
        storage: "Storage",
        *,
        logger: Optional[Any] = None,
        metrics_sink: Optional[Callable[[str, float], None]] = None,
    ) -> None:
        self.feed = feed
        self.storage = storage
        self._logger = logger or _default_logger("project_ex_nihilo.orchestrator")
        self._metrics_sink = metrics_sink
        self._last_ts: Dict[Tuple[str, str], int] = {}
        self._live_timeframes: Dict[str, str] = {}
        self._candles_stored = 0
        self._health_failures = 0
        self._alert_hooks: List[Callable[[Dict[str, bool]], None]] = []

        self.feed.add_ohlcv_listener(self._handle_live_candle)

    # ------------------------------------------------------------------
    def register_alert_hook(self, callback: Callable[[Dict[str, bool]], None]) -> None:
        """Register a callable invoked when a health check fails."""

        self._alert_hooks.append(callback)

    # ------------------------------------------------------------------
    def start_stream(self, symbol: str, timeframe: str) -> None:
        """Subscribe to live candles for *symbol* and cache the *timeframe*."""

        symbol_upper = symbol.upper()
        self._live_timeframes[symbol_upper] = timeframe
        self.feed.subscribe_ohlcv(symbol, timeframe)

    # ------------------------------------------------------------------
    def store_candles(
        self,
        symbol: str,
        timeframe: str,
        candles: Iterable["CandleEvent"],
    ) -> bool:
        """Validate a batch of candles and persist them via the storage backend."""

        batch = list(candles)
        if not self._validate_batch(symbol, timeframe, batch):
            return False

        self.storage.save_ohlcv(symbol, timeframe, batch)
        self._candles_stored += len(batch)
        self._last_ts[(symbol.upper(), timeframe)] = batch[-1].ts
        self._emit_metric("orchestrator.candles_stored", float(self._candles_stored))
        self._logger.info(
            "Stored candle batch",
            extra={"symbol": symbol.upper(), "timeframe": timeframe, "count": len(batch)},
        )
        return True

    # ------------------------------------------------------------------
    def health_check(self) -> Dict[str, bool]:
        """Run feed/storage health checks and trigger alert hooks on failure."""

        statuses: Dict[str, bool] = {}
        try:
            statuses["feed"] = bool(self.feed.ping())
        except Exception as exc:  # pragma: no cover - defensive monitoring
            self._logger.error("Feed health probe raised", exc=exc)
            statuses["feed"] = False
        try:
            statuses["storage"] = bool(self.storage.ping())
        except Exception as exc:  # pragma: no cover - defensive monitoring
            self._logger.error("Storage health probe raised", exc=exc)
            statuses["storage"] = False

        if not all(statuses.values()):
            self._health_failures += 1
            self._emit_metric("orchestrator.health_failures", float(self._health_failures))
            for hook in list(self._alert_hooks):
                try:
                    hook(dict(statuses))
                except Exception:  # pragma: no cover - alert hooks must not break monitoring
                    self._logger.warn("Alert hook raised an exception", extra={"hook": repr(hook)})

        return statuses

    # ------------------------------------------------------------------
    def report_metrics(self) -> Dict[str, Any]:
        """Emit a metrics snapshot via logging for observability."""

        snapshot = self.metrics_snapshot()
        self._logger.info("Ingestion metrics snapshot", extra={"metrics": snapshot})
        return snapshot

    def metrics_snapshot(self) -> Dict[str, Any]:
        """Return collected metrics across orchestrator, feed and storage."""

        snapshot = {
            "candles_stored": self._candles_stored,
            "health_failures": self._health_failures,
        }

        feed_metrics = getattr(self.feed, "metrics_snapshot", None)
        if callable(feed_metrics):
            snapshot["feed"] = feed_metrics()

        storage_metrics = getattr(self.storage, "metrics_snapshot", None)
        if callable(storage_metrics):
            snapshot["storage"] = storage_metrics()

        return snapshot

    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        """Stop the feed and release resources."""

        try:
            self.feed.close()
        except Exception:  # pragma: no cover - defensive shutdown
            self._logger.warn("Feed close raised", extra={"feed": repr(self.feed)})

    # ------------------------------------------------------------------
    def _handle_live_candle(self, event: "CandleEvent") -> None:
        symbol_upper = event.symbol.upper()
        timeframe = self._live_timeframes.get(symbol_upper)
        if timeframe is None:
            self._logger.warn(
                "Dropping live candle without registered timeframe",
                extra={"symbol": symbol_upper},
            )
            return

        if not self._validate_incremental(symbol_upper, timeframe, event.ts):
            return

        self.storage.save_ohlcv(symbol_upper, timeframe, [event])
        self._candles_stored += 1
        self._emit_metric("orchestrator.candles_stored", float(self._candles_stored))

    def _validate_batch(
        self,
        symbol: str,
        timeframe: str,
        candles: List["CandleEvent"],
    ) -> bool:
        if not candles:
            self._logger.warn(
                "Received empty candle batch",
                extra={"symbol": symbol.upper(), "timeframe": timeframe},
            )
            return False

        timestamps = [candle.ts for candle in candles]
        if any(t2 <= t1 for t1, t2 in zip(timestamps, timestamps[1:])):
            self._logger.warn(
                "Detected non-monotonic candle timestamps",
                extra={
                    "symbol": symbol.upper(),
                    "timeframe": timeframe,
                    "timestamps": timestamps,
                },
            )
            return False

        return True

    def _validate_incremental(self, symbol: str, timeframe: str, ts: int) -> bool:
        key = (symbol, timeframe)
        last_ts = self._last_ts.get(key)
        if last_ts is not None and ts <= last_ts:
            self._logger.warn(
                "Out-of-order live candle detected",
                extra={"symbol": symbol, "timeframe": timeframe, "last_ts": last_ts, "ts": ts},
            )
            return False

        self._last_ts[key] = ts
        return True

    def _emit_metric(self, name: str, value: float) -> None:
        if self._metrics_sink is None:
            return
        try:
            self._metrics_sink(name, value)
        except Exception:  # pragma: no cover - metrics sink must not affect ingestion
            self._logger.warn("Metrics sink raised an exception", extra={"metric": name})