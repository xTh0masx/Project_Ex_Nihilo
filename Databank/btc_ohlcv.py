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
import mysql.connector
from mysql.connector import Error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
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
        to_time=lambda idx: to_python_datetime(idx).date(),
    ),
    "minute": IntervalConfig(
        table="yahoo_finance_data_minute",
        time_column="quote_datetime",
        period="7d",
        interval="1m",
        to_time=lambda idx: to_python_datetime(idx).date(),
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

    rows = [
        (
            to_time(idx),
            float(row["Open"]),
            float(row["High"]),
            float(row["Low"]),
            float(row["Close"]),
            float(row["Volume"]),
        )
        for idx, row in data.iterrows()
    ]

    cursor.executemany(insert_statement, rows)
    connection.commit()
    #print(f"{cursor.rowcount} rows inserted into {table_name}.")
    logging.info("%s rows upserted into table %s", cursor.rowcount, table_name)

def refresh_interval(cursor, connection, ticker, config: IntervalConfig):
    """Fetch and store OHLCV for a single interval configuration."""

    logging.info("Updating %s data", config.interval)
    data = ticker.history(period=config.period, interval=config.interval)
    store_history(cursor, connection, config.table, config.time_column, data, config.to_time)

def update_all_intervals():
    """Connect to the database and update all configured OHLCV intervals."""

    connection = None
    cursor = None
    try:
        connection = get_connection()
        cursor = connection.cursor()
        ensure_tables(cursor)

        ticker = yf.Ticker("BTC-USD")
        for config in INTERVAL_CONFIGS.values():
            refresh_interval(cursor, connection, ticker, config)

    except Error as error:
        logging.exception("Failed to fetch/store BTC OHLCV data: %s", error)
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()

def run_schedule(poll_interval_seconds: int = 60):
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
    run_schedule()
