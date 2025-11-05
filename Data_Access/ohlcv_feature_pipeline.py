"""Pipeline for loading OHLCV data from MySQL, engineering features, and
producing model-ready datasets.

The goal is to keep the setup as simple as possible: edit the configuration
variables in this file with your own database credentials, date range, and
output preferences, then run the script with ``python
Data_Access/ohlcv_feature_pipeline.py``. No command line flags or environment
variables are required once the values below are filled in.

The script purposefully keeps dependencies minimal (pandas, numpy,
SQLAlchemy) so it can run in most environments without requiring the
full TA-Lib binary distribution.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
from sqlalchemy import create_engine


# ---------------------------------------------------------------------------
# Configuration dataclasses


@dataclass
class DatabaseConfig:
    """Settings required to connect to the MySQL instance."""

    host: str
    user: str
    password: str
    database: str
    port: int = 3306
    table: str = ""
    timestamp_column: str = "timestamp"

    def make_engine_url(self) -> str:
        """Build a SQLAlchemy-compatible connection URL."""

        safe_user = quote_plus(self.user)
        safe_password = quote_plus(self.password)

        return (
            f"mysql+mysqlconnector://{safe_user}:{safe_password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


@dataclass
class PipelineConfig:
    """Parameters controlling the feature/label pipeline."""

    start: Optional[str] = None
    end: Optional[str] = None
    horizon: int = 5  # minutes / bars ahead to predict
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    output_dir: str = "data/processed"
    output_format: str = "parquet"

    def validate(self) -> None:
        """Validate the ratios and output format."""

        if not 0 < self.train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1")
        if not 0 < self.val_ratio < 1:
            raise ValueError("val_ratio must be between 0 and 1")
        if self.train_ratio + self.val_ratio >= 1:
            raise ValueError("train_ratio + val_ratio must be less than 1")
        if self.output_format not in {"parquet", "csv"}:
            raise ValueError("output_format must be either 'parquet' or 'csv'")


# ---------------------------------------------------------------------------
# Data loading and feature engineering helpers


def load_ohlcv_data(cfg: DatabaseConfig, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    """Load OHLCV records from MySQL between optional date bounds."""

    engine_url = cfg.make_engine_url()
    logging.info("Connecting to MySQL using %s", engine_url)
    engine = create_engine(engine_url)

    where_clauses = []
    params = {}
    timestamp_col = cfg.timestamp_column

    if start:
        where_clauses.append(f"{timestamp_col} >= :start")
        params["start"] = start
    if end:
        where_clauses.append(f"{timestamp_col} <= :end")
        params["end"] = end

    where_sql = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    query = f"SELECT * FROM {cfg.table}{where_sql} ORDER BY {timestamp_col}"
    logging.info("Running query: %s", query)

    frame = pd.read_sql(query, engine, params=params)

    if timestamp_col not in frame.columns:
        raise ValueError(
            f"Expected a '{timestamp_col}' column in the OHLCV table; available columns: "
            f"{sorted(frame.columns.tolist())}"
        )

    frame[timestamp_col] = pd.to_datetime(frame[timestamp_col], utc=True)
    frame = frame.sort_values(timestamp_col).reset_index(drop=True)
    frame.rename(columns={timestamp_col: "timestamp"}, inplace=True)
    frame.set_index("timestamp", inplace=True)

    expected_cols = {"open", "high", "low", "close", "volume"}
    missing = expected_cols - set(frame.columns)
    if missing:
        raise ValueError(f"Missing expected columns in OHLCV data: {missing}")

    return frame


def compute_true_range(frame: pd.DataFrame) -> pd.Series:
    """Calculate the True Range used for volatility measures."""

    prev_close = frame["close"].shift(1)
    ranges = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Manual RSI implementation to avoid heavy dependencies."""

    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute returns, momentum, and volatility features."""

    engineered = frame.copy()

    # 1-bar log return and multi-horizon returns
    engineered["log_return_1"] = np.log(engineered["close"]) - np.log(engineered["close"].shift(1))
    engineered["return_5"] = engineered["close"].pct_change(5)
    engineered["return_15"] = engineered["close"].pct_change(15)

    # Exponential moving averages and their ratios
    ema_short = engineered["close"].ewm(span=12, adjust=False).mean()
    ema_long = engineered["close"].ewm(span=26, adjust=False).mean()
    engineered["ema_12"] = ema_short
    engineered["ema_26"] = ema_long
    engineered["ema_ratio"] = ema_short / ema_long - 1

    # RSI and rolling volatility
    engineered["rsi_14"] = compute_rsi(engineered["close"], window=14)
    engineered["volatility_30"] = engineered["log_return_1"].rolling(window=30).std()

    # Average true range based volatility and volume trend
    tr = compute_true_range(engineered)
    engineered["atr_14"] = tr.rolling(window=14).mean()
    engineered["volume_zscore_30"] = (
        (engineered["volume"] - engineered["volume"].rolling(30).mean())
        / engineered["volume"].rolling(30).std()
    )

    engineered.dropna(inplace=True)
    return engineered


def label_forward_return(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Create a forward-looking return target column."""

    if horizon <= 0:
        raise ValueError("horizon must be a positive integer number of bars")

    labeled = frame.copy()
    future_close = labeled["close"].shift(-horizon)
    labeled[f"target_return_{horizon}"] = (future_close / labeled["close"]) - 1
    labeled = labeled.iloc[:-horizon]
    return labeled


# ---------------------------------------------------------------------------
# Dataset splitting and persistence helpers


def split_dataset(
    frame: pd.DataFrame, train_ratio: float, val_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataframe chronologically into train/val/test sets."""

    n_rows = len(frame)
    if n_rows == 0:
        raise ValueError("Cannot split an empty dataframe")

    train_end = math.floor(n_rows * train_ratio)
    val_end = math.floor(n_rows * (train_ratio + val_ratio))

    train = frame.iloc[:train_end]
    val = frame.iloc[train_end:val_end]
    test = frame.iloc[val_end:]

    return train, val, test


def save_dataframe(frame: pd.DataFrame, path: str, fmt: str) -> None:
    """Persist a dataframe in the requested format."""

    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    if fmt == "parquet":
        frame.to_parquet(path)
    elif fmt == "csv":
        frame.to_csv(path)
    else:
        raise ValueError(f"Unsupported format '{fmt}'")


def persist_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: str,
    fmt: str,
) -> None:
    """Persist train/validation/test dataframes to disk."""

    suffix = "parquet" if fmt == "parquet" else "csv"
    save_dataframe(train, os.path.join(output_dir, f"train.{suffix}"), fmt)
    save_dataframe(val, os.path.join(output_dir, f"validation.{suffix}"), fmt)
    save_dataframe(test, os.path.join(output_dir, f"test.{suffix}"), fmt)


def run_pipeline(db_cfg: DatabaseConfig, pipeline_cfg: PipelineConfig) -> None:
    """Execute the full feature pipeline using the provided configuration."""

    pipeline_cfg.validate()

    raw = load_ohlcv_data(db_cfg, pipeline_cfg.start, pipeline_cfg.end)
    logging.info("Loaded %d OHLCV rows", len(raw))

    features = engineer_features(raw)
    logging.info("Engineered %d rows after feature computation", len(features))

    labeled = label_forward_return(features, pipeline_cfg.horizon)
    logging.info("Labeled dataset contains %d rows", len(labeled))

    train, val, test = split_dataset(
        labeled, pipeline_cfg.train_ratio, pipeline_cfg.val_ratio
    )
    logging.info(
        "Split counts - train: %d, validation: %d, test: %d",
        len(train),
        len(val),
        len(test),
    )

    persist_splits(train, val, test, pipeline_cfg.output_dir, pipeline_cfg.output_format)
    logging.info("Saved processed datasets to %s", pipeline_cfg.output_dir)


# ---------------------------------------------------------------------------
# Simple in-file configuration


# Fill in these values with your own database credentials and preferences.

# Map the available candle frequencies to the corresponding table names in MySQL.
# Update the mapping if your schema uses different table names.
TABLE_CONFIG_BY_FREQUENCY = {
    "daily": ("yahoo_finance_data", "quote_date"),
    "minute": ("yahoo_finance_data_minute", "quote_datetime"),
    "hourly": ("yahoo_finance_data_hourly", "quote_datetime"),
}


def resolve_table_name(frequency: str) -> Tuple[str, str]:
    """Return the table name and timestamp column for the requested frequency."""

    try:
        return TABLE_CONFIG_BY_FREQUENCY[frequency]
    except KeyError as exc:  # pragma: no cover - defensive configuration guard
        available = ", ".join(sorted(TABLE_CONFIG_BY_FREQUENCY))
        raise ValueError(
            f"Unknown frequency '{frequency}'. Choose one of: {available}."
        ) from exc


# Select which table (by its frequency) you want the pipeline to use.
SELECTED_FREQUENCY = "daily"  # change to "daily" or "hourly" or "minute" as needed
SELECTED_TABLE, SELECTED_TIMESTAMP_COLUMN = resolve_table_name(SELECTED_FREQUENCY)


DATABASE_SETTINGS = DatabaseConfig(
    host="127.0.0.1",
    port=3306,
    user="root",
    password="Digimon@4123",
    database="ex_nihilo",
    table=SELECTED_TABLE,
    timestamp_column=SELECTED_TIMESTAMP_COLUMN,
)


PIPELINE_SETTINGS = PipelineConfig(
    start=None,  # e.g. "2023-01-01"
    end=None,  # e.g. "2023-01-31"
    horizon=5,
    train_ratio=0.7,
    val_ratio=0.15,
    output_dir="data/processed",
    output_format="parquet",
)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    run_pipeline(DATABASE_SETTINGS, PIPELINE_SETTINGS)


if __name__ == "__main__":
    main()
