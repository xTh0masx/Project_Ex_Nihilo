"""Download BTC hourly candles from Yahoo Finance and store them in MySQL.

The script can be executed as a standalone utility.  It fetches historical
BTC-USD candles using ``yfinance`` with an hourly interval and writes the
resulting OHLCV records into a MySQL table.  Connection parameters can be
provided either via command line arguments or via the environment variables
``MYSQL_HOST``, ``MYSQL_PORT``, ``MYSQL_USER``, ``MYSQL_PASSWORD`` and
``MYSQL_DATABASE``.

Example
-------
.. code-block:: bash

    python btc_ohlcv.py \
        --host localhost \
        --user trader \
        --password secret \
        --database trading

The table structure matches the Phase 1 storage schema
(``symbol``, ``timeframe``, ``timestamp``, ``open``, ``high``, ``low``,
``close``, ``volume``) and will be created automatically when missing.  Use
``--print-ddl`` to display the SQL definition if you want to compare it with an
existing table before loading data.
"""

from __future__ import annotations

import argparse
import os
from typing import Generator, Iterable, Sequence, Tuple

import mysql.connector
import pandas as pd
import yfinance as yf

Row = Tuple[str, str, int, float, float, float, float, float]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTC-USD", help="Trading pair to download (default: %(default)s)")
    parser.add_argument(
        "--interval",
        default="1d",
        help="Candle interval requested from Yahoo Finance (default: %(default)s)",
    )
    parser.add_argument(
        "--start",
        default="2014-09-17",
        help="Start date for the historical backfill in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--end",
        default="2025-10-17",
        help="End date for the historical backfill in YYYY-MM-DD format",
    )
    parser.add_argument("--host", default=os.getenv("MYSQL_HOST", "localhost"), help="MySQL server host")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("MYSQL_PORT", "3306")),
        help="MySQL server port (default: %(default)s)",
    )
    parser.add_argument("--user", default=os.getenv("MYSQL_USER"), help="MySQL user name")
    parser.add_argument("--password", default=os.getenv("MYSQL_PASSWORD"), help="MySQL user password")
    parser.add_argument("--database", default=os.getenv("MYSQL_DATABASE"), help="Target MySQL database name")
    parser.add_argument(
        "--table",
        default="ohlcv",
        help="Destination table name (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2_000,
        help="Number of rows inserted per batch (default: %(default)s)",
    )
    parser.add_argument(
        "--print-ddl",
        action="store_true",
        help="Print the CREATE TABLE statement that the loader expects and exit",
    )
    return parser.parse_args()


def _render_table_ddl(table: str) -> str:
    return f"""
    CREATE TABLE IF NOT EXISTS `{table}` (
        symbol VARCHAR(32) NOT NULL,
        timeframe VARCHAR(16) NOT NULL,
        timestamp BIGINT NOT NULL,
        open DOUBLE NOT NULL,
        high DOUBLE NOT NULL,
        low DOUBLE NOT NULL,
        close DOUBLE NOT NULL,
        volume DOUBLE NOT NULL,
        PRIMARY KEY (symbol, timeframe, timestamp)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """.strip()


def _fetch_hourly_data(symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    frame = ticker.history(start=start, end=end, interval=interval, auto_adjust=False)
    if frame.empty:
        raise ValueError(
            "No OHLCV data received from Yahoo Finance. Check the symbol, interval and date range."
        )

    frame = frame.reset_index()
    timestamp_column = "Datetime" if "Datetime" in frame.columns else "Date"
    frame["timestamp"] = pd.to_datetime(frame[timestamp_column], utc=True).view("int64") // 1_000_000_000

    frame = frame.rename(columns=str.lower)
    expected_columns = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = expected_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns in Yahoo Finance response: {sorted(missing)}")

    ordered_columns = ["timestamp", "open", "high", "low", "close", "volume"]
    return frame[ordered_columns]


def _connect_mysql(host: str, port: int, user: str | None, password: str | None, database: str | None):
    if not user:
        raise ValueError("MySQL user is required (pass via --user or MYSQL_USER environment variable).")
    if not database:
        raise ValueError("MySQL database name is required (pass via --database or MYSQL_DATABASE environment variable).")

    return mysql.connector.connect(host=host, port=port, user=user, password=password, database=database)


def _ensure_table(connection, table: str) -> None:
    statement = _render_table_ddl(table)
    cursor = connection.cursor()
    try:
        cursor.execute(statement)
    finally:
        cursor.close()


def _iter_rows(frame: pd.DataFrame, symbol: str, timeframe: str) -> Iterable[Row]:
    for row in frame.itertuples(index=False):
        yield (
            symbol.upper(),
            timeframe,
            int(row.timestamp),
            float(row.open),
            float(row.high),
            float(row.low),
            float(row.close),
            float(row.volume),
        )


def _chunk(rows: Sequence[Row], size: int) -> Generator[Sequence[Row], None, None]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _insert_rows(connection, table: str, rows: Sequence[Row], batch_size: int) -> int:
    if not rows:
        return 0

    query = (
        f"INSERT INTO `{table}` (symbol, timeframe, timestamp, open, high, low, close, volume) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
        "ON DUPLICATE KEY UPDATE open=VALUES(open), high=VALUES(high), low=VALUES(low), "
        "close=VALUES(close), volume=VALUES(volume)"
    )

    inserted = 0
    for batch in _chunk(rows, batch_size):
        cursor = connection.cursor()
        try:
            cursor.executemany(query, batch)
            connection.commit()
            inserted += len(batch)
        finally:
            cursor.close()

    return inserted


def main() -> None:
    args = _parse_args()

    if args.print_ddl:
        print(_render_table_ddl(args.table))
        return

    data = _fetch_hourly_data(args.symbol, args.start, args.end, args.interval)

    connection = _connect_mysql(args.host, args.port, args.user, args.password, args.database)
    try:
        _ensure_table(connection, args.table)
        rows = list(_iter_rows(data.sort_values("timestamp"), args.symbol, args.interval))
        stored = _insert_rows(connection, args.table, rows, args.batch_size)
    finally:
        connection.close()

    print(f"Stored {stored} {args.interval} candles for {args.symbol.upper()} into MySQL table '{args.table}'.")


if __name__ == "__main__":
    main()