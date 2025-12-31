# streamlit run "Phase 1\\dashboard2.py"

"""Professional Streamlit dashboard for BTC datasets and trade analytics.

This dashboard focuses on clear layout, robust database handling, and
reusable utilities so teams can monitor price data, inspect trade history,
and run quick what-if simulations without editing the script.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os
import sys
from typing import Dict, Iterable, List, Optional
from urllib.parse import quote_plus

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

# Ensure repository modules can be imported when launched via ``streamlit run``
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Databank.btc_ohlcv import update_all_intervals

DATASETS: Dict[str, Dict[str, str]] = {
    "Minute": {"table": "yahoo_finance_data_minute", "time_column": "quote_datetime"},
    "Hourly": {"table": "yahoo_finance_data_hourly", "time_column": "quote_datetime"},
    "Daily": {"table": "yahoo_finance_data", "time_column": "quote_date"},
}


def _env(key: str, default: str | int) -> str | int:
    """Read a configuration value from the environment with a fallback."""

    return os.getenv(key.upper(), default)


DB_CONFIG: Dict[str, str | int] = {
    "host": _env("MYSQL_HOST", "127.0.0.1"),
    "port": int(_env("MYSQL_PORT", 3306)),
    "user": _env("MYSQL_USER", "root"),
    "password": _env("MYSQL_PASSWORD", "Digimon@4123"),
    "database": _env("MYSQL_DATABASE", "ex_nihilo"),
    "auth_plugin": "mysql_native_password",
}

TRADE_TABLE = os.getenv("TRADE_TABLE", "bot_trades")

_ENGINE: Engine | None = None


def _engine() -> Engine:
    """Create (or reuse) a SQLAlchemy engine for pandas queries."""

    global _ENGINE
    if _ENGINE is None:
        user = quote_plus(str(DB_CONFIG["user"]))
        password = quote_plus(str(DB_CONFIG["password"]))
        host = DB_CONFIG["host"]
        port = DB_CONFIG["port"]
        database = DB_CONFIG["database"]
        auth_plugin = DB_CONFIG.get("auth_plugin")
        query = f"?auth_plugin={quote_plus(str(auth_plugin))}" if auth_plugin else ""
        url = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}{query}"
        _ENGINE = create_engine(url, pool_pre_ping=True)
    return _ENGINE


def _table_columns(table: str) -> List[str]:
    with _engine().connect() as connection:
        rows = connection.execute(
            text(
                """
                SELECT COLUMN_NAME
                FROM information_schema.columns
                WHERE table_schema = :schema AND table_name = :table
                ORDER BY ORDINAL_POSITION
                """
            ),
            {"schema": DB_CONFIG["database"], "table": table},
        ).fetchall()
    return [row[0] for row in rows]


def _resolve_time_column(table: str, preferred: str) -> str:
    columns = _table_columns(table)
    if not columns:
        raise RuntimeError(f"Table '{table}' does not exist in schema {DB_CONFIG['database']}")
    if preferred in columns:
        return preferred

    for candidate in ("quote_datetime", "quote_date", "timestamp", "quote_time"):
        if candidate in columns:
            return candidate

    raise RuntimeError(
        f"None of the expected timestamp columns ({preferred}) exist on table '{table}'."
    )


def _query_dataframe(query: str, *, parse_dates: Iterable[str] | None = None) -> pd.DataFrame:
    with _engine().connect() as connection:
        return pd.read_sql_query(text(query), con=connection, parse_dates=list(parse_dates or []))


@st.cache_data(ttl=120)
def load_ohlcv(granularity: str) -> pd.DataFrame:
    config = DATASETS[granularity]
    time_column = _resolve_time_column(config["table"], config["time_column"])
    query = (
        f"SELECT `{time_column}` AS timestamp, open, high, low, close, volume "
        f"FROM `{config['table']}` ORDER BY `{time_column}`"
    )
    frame = _query_dataframe(query, parse_dates=["timestamp"])
    frame.set_index("timestamp", inplace=True)
    frame.sort_index(inplace=True)
    numeric_cols = ["open", "high", "low", "close", "volume"]
    frame[numeric_cols] = frame[numeric_cols].apply(pd.to_numeric, errors="coerce")
    frame.dropna(subset=numeric_cols, inplace=True)
    return frame


def _detect_trade_columns(connection, table: str) -> List[str]:
    rows = connection.execute(
        text(
            """
            SELECT COLUMN_NAME
            FROM information_schema.columns
            WHERE table_schema = :schema AND table_name = :table
            ORDER BY ORDINAL_POSITION
            """
        ),
        {"schema": DB_CONFIG["database"], "table": table},
    ).fetchall()
    return [row[0] for row in rows]


@st.cache_data(ttl=60)
def load_trades() -> pd.DataFrame:
    if not TRADE_TABLE:
        return pd.DataFrame()

    try:
        connection = _engine().connect()
    except SQLAlchemyError:
        return pd.DataFrame()

    with connection:
        exists = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_schema = :schema AND table_name = :table
                """
            ),
            {"schema": DB_CONFIG["database"], "table": TRADE_TABLE},
        ).scalar()
        if not exists:
            return pd.DataFrame()

        available_columns = _detect_trade_columns(connection, TRADE_TABLE)
        timestamp_column = next(
            (col for col in ("executed_at", "created_at", "timestamp") if col in available_columns),
            None,
        )
        if not timestamp_column:
            return pd.DataFrame()

        select_columns = [timestamp_column]
        preferred = ["side", "direction", "price", "execution_price", "quantity", "qty", "pnl", "pnl_usd", "strategy"]
        for column in preferred:
            if column in available_columns and column not in select_columns:
                select_columns.append(column)

        query = (
            f"SELECT {', '.join(f'`{col}`' for col in select_columns)} FROM `{TRADE_TABLE}` "
            f"ORDER BY `{timestamp_column}`"
        )
        try:
            trades = pd.read_sql_query(text(query), con=connection, parse_dates=[timestamp_column])
        except SQLAlchemyError:
            return pd.DataFrame()

    trades.rename(columns={timestamp_column: "timestamp"}, inplace=True)
    if "direction" in trades.columns and "side" not in trades.columns:
        trades.rename(columns={"direction": "side"}, inplace=True)
    if "execution_price" in trades.columns and "price" not in trades.columns:
        trades.rename(columns={"execution_price": "price"}, inplace=True)
    if "qty" in trades.columns and "quantity" not in trades.columns:
        trades.rename(columns={"qty": "quantity"}, inplace=True)
    if "pnl_usd" in trades.columns and "pnl" not in trades.columns:
        trades.rename(columns={"pnl_usd": "pnl"}, inplace=True)

    trades.sort_values("timestamp", inplace=True)
    return trades


def _render_summary_cards(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.info("No price data available for the selected configuration.")
        return

    latest_price = frame["close"].iloc[-1]
    start_ts, end_ts = frame.index[0], frame.index[-1]
    price_change = (frame["close"].iloc[-1] - frame["close"].iloc[0]) / frame["close"].iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Latest close", f"${latest_price:,.2f}")
    col2.metric("Date range", f"{start_ts:%Y-%m-%d} → {end_ts:%Y-%m-%d}")
    col3.metric("Change", f"{price_change * 100:+.2f}%")


def _render_price_chart(frame: pd.DataFrame, *, title: str) -> None:
    if frame.empty:
        st.warning("No candlesticks to display.")
        return

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=frame.index,
            open=frame["open"],
            high=frame["high"],
            low=frame["low"],
            close=frame["close"],
            name="Price",
        )
    )
    fig.add_trace(go.Scatter(x=frame.index, y=frame["close"].rolling(20).mean(), name="MA 20"))
    fig.add_trace(go.Scatter(x=frame.index, y=frame["close"].rolling(50).mean(), name="MA 50"))

    fig.update_layout(height=520, title=title, xaxis_title="Timestamp", yaxis_title="Price (USD)")
    st.plotly_chart(fig, width="stretch")


def _render_volume_chart(frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    fig = go.Figure()
    fig.add_bar(x=frame.index, y=frame["volume"], name="Volume", marker_color="#2E8BC0")
    fig.update_layout(height=200, margin=dict(t=30), xaxis_title="Timestamp", yaxis_title="Volume")
    st.plotly_chart(fig, width="stretch")


def _render_trade_performance(trades: pd.DataFrame) -> None:
    st.subheader("Trade performance")
    if trades.empty:
        st.info("No trades found in the configured table.")
        return

    trades["pnl"].fillna(0, inplace=True)
    total_trades = len(trades)
    profitable = (trades["pnl"] > 0).sum()
    total_pnl = trades["pnl"].sum()
    win_rate = (profitable / total_trades) * 100 if total_trades else 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("Trades", total_trades)
    col2.metric("Win rate", f"{win_rate:.1f}%")
    col3.metric("Total PnL", f"${total_pnl:,.2f}")

    st.dataframe(trades.tail(200), width="stretch")



def _simulate_bot_trades(
    frame: pd.DataFrame,
    *,
    starting_capital: float = 2000.0,
    max_trade_usd: float = 1000.0,
    max_trades: int = 500,
) -> pd.DataFrame:
    closes = frame["close"].copy()
    fast = closes.rolling(window=5, min_periods=5).mean()
    slow = closes.rolling(window=20, min_periods=20).mean()

    capital = starting_capital
    trades: List[Dict[str, float | str | pd.Timestamp]] = []
    previous_signal: Optional[str] = None

    for idx in range(len(closes) - 1):
        if len(trades) >= max_trades or capital <= 0:
            break

        signal = "hold"
        if not pd.isna(fast.iloc[idx]) and not pd.isna(slow.iloc[idx]):
            signal = "buy" if fast.iloc[idx] > slow.iloc[idx] else "sell"

        if signal == "hold" or signal == previous_signal:
            continue

        entry_price = closes.iloc[idx]
        exit_price = closes.iloc[idx + 1]
        trade_size = min(max_trade_usd, capital)
        quantity = trade_size / entry_price

        if signal == "buy":
            pnl = quantity * (exit_price - entry_price)
        else:
            pnl = quantity * (entry_price - exit_price)

        capital += pnl
        trades.append(
            {
                "timestamp": closes.index[idx + 1],
                "side": signal,
                "quantity": quantity,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "remaining_capital": capital,
            }
        )
        previous_signal = signal

    return pd.DataFrame(trades)


def _render_simulation(frame: pd.DataFrame) -> None:
    st.subheader("Quick moving-average simulation")
    st.caption("A lightweight backtest to sanity-check incoming data and latency.")

    if frame.empty:
        st.info("Load price data to run the simulation.")
        return

    capital = st.number_input("Starting capital", value=2000.0, min_value=100.0, step=100.0)
    risk_cap = st.number_input("Max capital per trade", value=1000.0, min_value=50.0, step=50.0)
    max_trades = st.number_input("Maximum trades", value=200, min_value=1, max_value=2000, step=25)

    if st.button("Run simulation", type="primary"):
        trades = _simulate_bot_trades(
            frame, starting_capital=capital, max_trade_usd=risk_cap, max_trades=int(max_trades)
        )
        if trades.empty:
            st.warning("No trades were generated; adjust thresholds or load more data.")
            return

        st.success(
            f"Simulation finished with {len(trades)} trades and total PnL ${trades['pnl'].sum():,.2f}."
        )
        st.dataframe(trades, width="stretch")



def _render_sync_controls() -> None:
    st.subheader("Data maintenance")
    st.caption("Keep your databank fresh without leaving the dashboard.")

    if st.button("Sync OHLCV databank now"):
        try:
            update_all_intervals()
            load_ohlcv.clear()
            st.success("Databank updated and caches cleared.")
        except Exception as exc:  # pragma: no cover - defensive for live dashboard
            st.error(f"Failed to refresh databank: {exc}")



def main() -> None:
    st.set_page_config(page_title="BTC Professional Dashboard", layout="wide")
    st.title("BTC Operations Dashboard")
    st.caption("Monitor market structure, trade execution health, and quick strategy probes.")

    st.sidebar.header("Configuration")
    granularity = st.sidebar.radio("Dataset", list(DATASETS.keys()), index=1)
    show_volume = st.sidebar.checkbox("Show volume chart", value=True)
    limit_rows = st.sidebar.number_input("Rows to load (0 = all)", value=5000, min_value=0, step=500)
    start_date = st.sidebar.date_input("Start date", value=None)
    end_date = st.sidebar.date_input("End date", value=None)

    try:
        frame = load_ohlcv(granularity)
    except Exception as exc:  # pragma: no cover - keep dashboard responsive
        st.error(f"Failed to load OHLCV data: {exc}")
        return

    if limit_rows and limit_rows > 0:
        frame = frame.tail(int(limit_rows))

    if start_date:
        frame = frame.loc[pd.to_datetime(start_date) :]
    if end_date:
        frame = frame.loc[: pd.to_datetime(end_date) + pd.Timedelta(days=1)]

    _render_summary_cards(frame)
    _render_price_chart(frame, title=f"BTC/USD {granularity} candles")
    if show_volume:
        _render_volume_chart(frame)

    st.divider()
    trades = load_trades()
    _render_trade_performance(trades)

    st.divider()
    _render_simulation(frame)

    st.divider()
    _render_sync_controls()


if __name__ == "__main__":
    main()