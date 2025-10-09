"""Command line utility for ingesting Binance market data into MySQL.

This module wires together the configuration, logging, data feed and storage
components shipped with the Phase 1 prototype.  It exposes a small CLI that can
backfill historical candles and, optionally, keep storing live updates via the
Binance WebSocket streams.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Sequence

try:  # Optional dependency – only required when writing to MySQL.
    import mysql.connector
except Exception:  # pragma: no cover - optional dependency guard
    mysql = None  # type: ignore
else:  # pragma: no cover - namespace normalisation
    mysql = mysql.connector

from .Configuration import Config
from .DataFeed import BinanceDataFeed, CandleEvent
from .Logger import Logger, get_logger
from .Storage import Storage


class MySQLStorage(Storage):
    """Simple MySQL-backed storage implementation for OHLCV data."""

    def __init__(
        self,
        *,
        host: str,
        user: str,
        password: str,
        database: str,
        port: int = 3306,
        table: str = "ohlcv",
    ) -> None:
        if mysql is None:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "MySQL support requires 'mysql-connector-python'. Install the package to enable storage."
            )

        self._connection = mysql.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            port=port,
            autocommit=True,
        )
        self._table = table

    # ------------------------------------------------------------------
    def save_ohlcv(self, symbol: str, timeframe: str, data: Sequence[CandleEvent]) -> None:
        if not data:
            return

        rows = [
            (
                candle.ts,
                symbol.upper(),
                timeframe,
                candle.o,
                candle.h,
                candle.l,
                candle.c,
                candle.v,
            )
            for candle in data
        ]

        placeholders = (
            "INSERT INTO `{table}` (open_time, symbol, timeframe, open, high, low, close, volume)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
            " ON DUPLICATE KEY UPDATE open=%s, high=%s, low=%s, close=%s, volume=%s"
        ).format(table=self._table)

        params = [
            (
                open_time,
                symbol_value,
                timeframe_value,
                o,
                h,
                l,
                c,
                v,
                o,
                h,
                l,
                c,
                v,
            )
            for open_time, symbol_value, timeframe_value, o, h, l, c, v in rows
        ]

        cursor = self._connection.cursor()
        try:
            cursor.executemany(placeholders, params)
        finally:  # pragma: no branch - cleanup
            cursor.close()

    # ------------------------------------------------------------------
    def load_ohlcv(self, symbol: str, timeframe: str) -> Iterable[CandleEvent]:  # pragma: no cover - not used
        query = (
            "SELECT open_time, open, high, low, close, volume FROM `{table}`"
            " WHERE symbol=%s AND timeframe=%s ORDER BY open_time ASC"
        ).format(table=self._table)

        cursor = self._connection.cursor()
        try:
            cursor.execute(query, (symbol.upper(), timeframe))
            for open_time, o, h, l, c, v in cursor.fetchall():
                yield CandleEvent(
                    ts=int(open_time),
                    symbol=symbol.upper(),
                    o=float(o),
                    h=float(h),
                    l=float(l),
                    c=float(c),
                    v=float(v),
                )
        finally:  # pragma: no branch - cleanup
            cursor.close()

    # ------------------------------------------------------------------
    def save_trade(self, trade):  # pragma: no cover - placeholder
        raise NotImplementedError("Trade persistence is not implemented for Phase 1.")

    # ------------------------------------------------------------------
    def save_model(self, tag: str, obj):  # pragma: no cover - placeholder
        raise NotImplementedError("Model persistence is not implemented for Phase 1.")

    # ------------------------------------------------------------------
    def load_model(self, tag: str):  # pragma: no cover - placeholder
        raise NotImplementedError("Model loading is not implemented for Phase 1.")

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()


def _interval_to_milliseconds(interval: str) -> int:
    unit = interval[-1]
    value = int(interval[:-1])
    factors = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
    if unit not in factors:
        raise ValueError(f"Unsupported Binance interval '{interval}'.")
    return value * factors[unit]


def _coerce_start(start: str) -> int:
    if isinstance(start, (int, float)):
        value = int(start)
    else:
        text = str(start).strip()
        if text.isdigit():
            value = int(text)
        else:
            try:
                dt = datetime.fromisoformat(text)
            except ValueError as exc:
                raise ValueError(f"Unable to parse start time '{start}'.") from exc
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            value = int(dt.timestamp())

    if value > 10**12:  # already milliseconds
        return value
    return value * 1000


def _chunk_request(
    feed: BinanceDataFeed,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int,
) -> List[CandleEvent]:
    candles = feed.rest_client.get_klines(
        symbol=symbol.upper(),
        interval=interval,
        limit=limit,
        startTime=start_ms,
        endTime=end_ms,
    )

    return [
        CandleEvent(
            ts=int(open_time),
            symbol=symbol.upper(),
            o=float(o),
            h=float(h),
            l=float(l),
            c=float(c),
            v=float(v),
        )
        for open_time, o, h, l, c, v, *_ in candles
    ]


def sync_historical(
    feed: BinanceDataFeed,
    storage: MySQLStorage,
    logger: Logger,
    *,
    symbol: str,
    interval: str,
    start: str,
    limit: int = 1000,
) -> None:
    """Backfill historical OHLCV data into MySQL."""

    timeframe_ms = _interval_to_milliseconds(interval)
    start_ms = _coerce_start(start)
    now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)

    window = timeframe_ms * limit
    current_start = start_ms

    while current_start < now_ms:
        current_end = min(current_start + window - timeframe_ms, now_ms)
        candles = _chunk_request(feed, symbol, interval, current_start, current_end, limit)

        if not candles:
            logger.info(f"No more candles returned after {datetime.fromtimestamp(current_start/1000, tz=timezone.utc)}.")
            break

        storage.save_ohlcv(symbol, interval, candles)
        logger.info(
            "Stored %s candles for %s (%s) covering %s -> %s",
            len(candles),
            symbol,
            interval,
            datetime.fromtimestamp(candles[0].ts / 1000, tz=timezone.utc).isoformat(),
            datetime.fromtimestamp(candles[-1].ts / 1000, tz=timezone.utc).isoformat(),
        )

        current_start = candles[-1].ts + timeframe_ms


def _attach_live_ingestion(
    feed: BinanceDataFeed,
    storage: MySQLStorage,
    logger: Logger,
    symbol: str,
    interval: str,
) -> None:
    def _on_candle(event: CandleEvent) -> None:
        storage.save_ohlcv(symbol, interval, [event])
        logger.info(
            "Saved live candle for %s (%s) at %s",
            event.symbol,
            interval,
            datetime.fromtimestamp(event.ts / 1000, tz=timezone.utc).isoformat(),
        )

    feed.on_ohlcv = _on_candle  # type: ignore[assignment]
    feed.subscribe_ohlcv(symbol, interval)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Binance data ingestion utility")
    parser.add_argument("--config", required=True, help="Path to the configuration file (JSON or YAML).")
    parser.add_argument("--symbol", required=True, help="Trading pair to ingest, e.g. BTCUSDT.")
    parser.add_argument("--interval", required=True, help="Binance kline interval, e.g. 1m, 1h.")
    parser.add_argument(
        "--start",
        required=True,
        help="Historical start timestamp (ISO string or unix seconds/milliseconds).",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Keep a WebSocket subscription open to ingest new candles after the backfill.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    config = Config.load(args.config)
    logger = get_logger("binance_ingestion")

    api_key = config.get("exchange.api_key")
    api_secret = config.get("exchange.api_secret")
    base_url = config.get("exchange.base_url")

    db_settings = {
        "host": config.get("database.host", "localhost"),
        "port": int(config.get("database.port", 3306)),
        "user": config.get("database.user", "root"),
        "password": config.get("database.password", ""),
        "database": config.get("database.name", "trading"),
        "table": config.get("database.table", "ohlcv"),
    }

    storage = MySQLStorage(**db_settings)
    feed = BinanceDataFeed(api_key=api_key, api_secret=api_secret, base_url=base_url)

    try:
        sync_historical(
            feed,
            storage,
            logger,
            symbol=args.symbol,
            interval=args.interval,
            start=args.start,
        )

        if args.live:
            _attach_live_ingestion(feed, storage, logger, args.symbol, args.interval)
            logger.info("Live ingestion started – press Ctrl+C to stop.")
            while True:  # pragma: no cover - integration behaviour
                time.sleep(1)
    except KeyboardInterrupt:  # pragma: no cover - interactive session
        logger.info("Interrupted by user. Shutting down...")
    finally:
        feed.close()
        storage.close()

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())