"""Fetch and persist BTC OHLCV data on rolling schedule.

This module downloads BTC-USD OHLCV data from Yahoo Finance and stores it in
three MySQL tables (daily, hourly and minute level). When executed directly it
keeps the tables up-to-date by polling Yahoo Finance every minute and interesting any new rows.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Dict

import yfinance as yf
import pandas as pd
import mysql.connector
from mysql.connector import Error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def to_python_datetime(timestamp):
    dt = timestamp.to_pydatetime()
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    return dt


@dataclass(frozen=True)
class IntervalConfig:
    """Describe how a given OHLCV interval maps to the database schema."""

    table: str
    time_column: str
    period: str
    interval: str
    to_time: Callable

DB_CONFIG: Dict[str, str | int] = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "Digimon@4123",
    "database": "ex_nihilo",
    "auth_plugin": "mysql_native_password",
}

INTERVAL_CONFIGS: Dict[str, IntervalConfig] = {
    "daily": IntervalConfig(
        table="yahoo_finance_data",
        time_column="quote_date",
        period="2y",
        interval="1d",
        to_time=lambda idx: to_python_datetime(idx).date(),
    ),
    "hourly": IntervalConfig(
        table="yahoo_finance_data_hourly",
        time_column="quote_datetime",
        period="60d",
        interval="1h",
        to_time=lambda idx: to_python_datetime(idx),
    ),
    "minute": IntervalConfig(
        table="yahoo_finance_data_minute",
        time_column="quote_datetime",
        period="7d",
        interval="1m",
        to_time=lambda idx: to_python_datetime(idx),
    ),
}

def get_connection():
    """Create a fresh MySQL connection using the configured credentials."""

    return mysql.connector.connect(**DB_CONFIG)

def ensure_tables(cursor):
    """Create the OHLCV storage tables if they do not already exist."""

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS yahoo_finance_data (
            quote_date DATE PRIMARY KEY,
            open DECIMAL(18, 8),
            high DECIMAL(18, 8),
            low DECIMAL(18, 8),
            close DECIMAL(18, 8),
            volume BIGINT
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS yahoo_finance_data_hourly (
            quote_datetime DATETIME PRIMARY KEY,
            open DECIMAL(18, 8),
            high DECIMAL(18, 8),
            low DECIMAL(18, 8),
            close DECIMAL(18, 8),
            volume BIGINT
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS yahoo_finance_data_minute (
            quote_datetime DATETIME PRIMARY KEY,
            open DECIMAL(18, 8),
            high DECIMAL(18, 8),
            low DECIMAL(18, 8),
            close DECIMAL(18, 8),
            volume BIGINT
        )
        """
    )

def store_history(cursor, connection, table_name, time_column, data, to_time):
    """Persist a slice of OHLCV data using an upsert statement."""

    if data.empty:
        logging.info("No data returned for table %s", table_name)
        return

    insert_statement = f"""
        INSERT INTO {table_name} ({time_column}, open, high, low, close, volume)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            open = VALUES(open),
            high = VALUES(high),
            low = VALUES(low),
            close = VALUES(close),
            volume = VALUES(volume)
    """

    rows = []
    for idx, row in data.iterrows():
        if any(pd.isna(row[column]) for column in PRICE_COLUMNS if column in row):
            continue

        volume = row.get("Volume", pd.NA)
        volume_value = None if pd.isna(volume) else int(round(float(volume)))

        rows.append(
        (
            to_time(idx),
            float(row["Open"]),
            float(row["High"]),
            float(row["Low"]),
            float(row["Close"]),
            volume_value,
        )
    )

    if not rows:
        logging.info("No valid rows to store for table %s", table_name)
        return

    cursor.executemany(insert_statement, rows)
    connection.commit()
    #print(f"{cursor.rowcount} rows inserted into {table_name}.")
    logging.info("%s rows upserted into table %s", cursor.rowcount, table_name)

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
PRICE_COLUMNS = ["OPEN", "HIGH", "LOW", "CLOSE"]

def normalise_history_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Coerce Yahoo Finance frames into a predictable column layout."""

    if data.empty:
        return data

    normalised = data.copy()

    if isinstance(normalised.columns, pd.MultiIndex):
        normalised.columns = normalised.columns.get_level_values(-1)

    rename_map = {}
    for column in normalised.columns:
        column_key = str(column).strip().lower()
        if column_key == "adj close":
            rename_map[column] = "Adj Close"
        elif column_key in {'open', 'high', 'low', 'close', 'volume'}:
            rename_map[column] = column_key.capitalize()

    if rename_map:
        normalised.rename(columns=rename_map, inplace=True)

    for required in REQUIRED_COLUMNS:
        if required not in normalised.columns:
            normalised[required] = pd.NA

    return normalised

def fetch_history(symbol: str, *, period:str, interval: str):
    """Download fresh OHLCV candles from Yahoo Finance."""

    data = yf.download(
        tickers=symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
    )

    return normalise_history_frame(data)

def refresh_interval(cursor, connection, symbol:str, config: IntervalConfig):
    """Fetch and store OHLCV for a single interval configuration."""

    logging.info("Updating %s data", config.interval)
    data = fetch_history(symbol, period=config.period, interval=config.interval)
    if not data.empty:
        subset = [column for column in PRICE_COLUMNS if column in data.columns]
        if not subset:
            logging.warning(
                "Skipping %s update because price columns are missing", config.interval
            )
            return
        data = data.dropna(subset=subset)
    store_history(cursor, connection, config.table, config.time_column, data, config.to_time)

def update_all_intervals():
    """Connect to the database and update all configured OHLCV intervals."""

    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor()
        ensure_tables(cursor)

        symbol = "BTC-USD"
        for config in INTERVAL_CONFIGS.values():
            refresh_interval(cursor, connection, symbol, config)

    except Error as error:
        logging.exception("Failed to fetch/store BTC OHLCV data: %s", error)
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()

def run_scheduler(poll_interval_seconds: int = 60):
    """Continuously update OHLCV tables on a fixed interval."""

    logging.info("Starting BTC OHLCV updater (interval=%ss)", poll_interval_seconds)
    try:
        while True:
            start = time.perf_counter()
            update_all_intervals()
            elapsed = time.perf_counter() - start
            sleep_for = max(poll_interval_seconds - elapsed, 0)
            logging.info("Sleeping for %.2f seconds", sleep_for)
            time.sleep(sleep_for)

    except KeyboardInterrupt:
        logging.info("Stopping BTC OHLCV updater")

if __name__ == "__main__":
    run_scheduler()
