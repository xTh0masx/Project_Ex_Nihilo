"""Streamlit dashboard for exploring BTC OHLCV datasets and trades."""
#to run streamlit copypaste -> streamlit run "Phase 1\\bitcoin_dashboard.py"

from __future__ import annotations

from datetime import timedelta
import os
from typing import Dict, Iterable, List
from urllib.parse import quote_plus

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def _env(key: str, default: str | int) -> str | int:
    """Read a database configuration value from the environment."""

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
    """Fetch the ordered column names for a given table."""

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
    """Ensure the timestamp column exists, falling back to similar names."""

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


DATASETS = {
    "Minute": {
        "table": "yahoo_finance_data_minute",
        "time_column": "quote_datetime",
    },
    "Hourly": {
        "table": "yahoo_finance_data_hourly",
        "time_column": "quote_datetime",
    },
    "Daily": {
        "table": "yahoo_finance_data",
        "time_column": "quote_date",
    },
}


def _query_dataframe(query: str, *, parse_dates: Iterable[str] | None = None) -> pd.DataFrame:
    """Execute a SQL query and return the resulting dataframe."""

    with _engine().connect() as connection:
        return pd.read_sql_query(text(query), con=connection, parse_dates=list(parse_dates or []))


@st.cache_data(ttl=60)
def load_ohlcv(granularity: str) -> pd.DataFrame:
    """Load OHLCV rows for the selected granularity into a dataframe."""

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


@st.cache_data(ttl=30)
def load_trades() -> pd.DataFrame:
    """Load trade executions from the configured table if it exists."""

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
        preferred = [
            "side",
            "direction",
            "price",
            "execution_price",
            "quantity",
            "qty",
            "size",
            "pnl",
            "pnl_usd",
            "strategy",
        ]
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


def _split_trades_by_side(trades: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    side_series = trades.get("side")
    if side_series is None:
        empty = trades.iloc[0:0]
        return {"buy": empty, "sell": empty}

    normalized = side_series.astype(str).str.upper()
    buys = trades[normalized.isin({"BUY", "LONG"})]
    sells = trades[normalized.isin({"SELL", "SHORT"})]
    return {"buy": buys, "sell": sells}


def render_candlestick(frame: pd.DataFrame, trades: pd.DataFrame, title: str) -> None:
    """Render a candlestick chart with optional trade markers."""

    fig = go.Figure(
        data=
        [
            go.Candlestick(
                x=frame.index,
                open=frame["open"],
                high=frame["high"],
                low=frame["low"],
                close=frame["close"],
                name="BTC-USD",
            )
        ]
    )

    if not trades.empty:
        grouped = _split_trades_by_side(trades)
        for label, data in grouped.items():
            if data.empty:
                continue
            price_series = data.get("price")
            if price_series is None or price_series.isna().all():
                price_series = frame.reindex(data["timestamp"], method="nearest")["close"]
            fig.add_trace(
                go.Scatter(
                    x=data["timestamp"],
                    y=price_series,
                    mode="markers",
                    marker=dict(
                        size=10,
                        symbol="triangle-up" if label == "buy" else "triangle-down",
                        color="#16a34a" if label == "buy" else "#dc2626",
                        line=dict(width=1, color="#111"),
                    ),
                    name=f"{label.title()} trade",
                    text=data.get("strategy", data.get("quantity")),
                    hovertemplate="Timestamp: %{x}<br>Price: %{y:.2f}<extra></extra>",
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Timestamp",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=60, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_summary(frame: pd.DataFrame) -> None:
    """Display quick statistics for the selected data slice."""

    latest_close = frame["close"].iloc[-1]
    change_series = frame["close"].pct_change()
    day_change = 0.0 if change_series.isna().all() else change_series.iloc[-1] * 100
    high = frame["high"].max()
    low = frame["low"].min()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest close", f"${latest_close:,.2f}")
    col2.metric("Change", f"{day_change:+.2f}%")
    col3.metric("High", f"${high:,.2f}")
    col4.metric("Low", f"${low:,.2f}")


def render_trades_table(trades: pd.DataFrame) -> None:
    """Show the recent trades and aggregated PnL, if available."""

    if trades.empty:
        st.info("No trades recorded in the configured table yet.")
        return

    latest_trades = trades.tail(200).copy()
    latest_trades["timestamp"] = latest_trades["timestamp"].dt.tz_localize(None)
    st.dataframe(latest_trades, hide_index=True, use_container_width=True)

    if "pnl" in trades.columns:
        st.metric("Cumulative PnL", f"${trades['pnl'].sum():,.2f}")


def main():  # pragma: no cover - Streamlit entrypoint
    st.set_page_config(page_title="BTC OHLCV Dashboard", layout="wide")
    st.title("Bitcoin OHLCV Dashboard")

    st.sidebar.header("Dataset selection")
    dataset = st.sidebar.selectbox("Granularity", list(DATASETS.keys()))

    if st.sidebar.button("Refresh data", use_container_width=True):
        load_ohlcv.clear()
    if st.sidebar.button("Refresh trades", use_container_width=True):
        load_trades.clear()

    try:
        frame = load_ohlcv(dataset)
    except SQLAlchemyError as exc:
        st.error(f"Failed to load data from MySQL: {exc}")
        return

    if frame.empty:
        st.info("No data available for the selected dataset.")
        return

    trades = load_trades()

    min_ts = frame.index.min().to_pydatetime()
    max_ts = frame.index.max().to_pydatetime()
    default_start = max_ts - timedelta(days=30)
    start, end = st.slider(
        "Date range",
        min_value=min_ts,
        max_value=max_ts,
        value=(max(min_ts, default_start), max_ts),
        format="YYYY-MM-DD",
    )
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    frame_slice = frame.loc[start_ts:end_ts]
    if frame_slice.empty:
        st.warning("No data found for the selected range.")
        return

    trades_slice = (
        trades[(trades["timestamp"] >= start_ts) & (trades["timestamp"] <= end_ts)]
        if not trades.empty
        else trades
    )

    render_summary(frame_slice)
    render_candlestick(frame_slice, trades_slice, f"BTC-USD – {dataset} candles")

    tab_data, tab_trades = st.tabs(["Data preview", "Trades"])
    with tab_data:
        st.dataframe(frame_slice.tail(500), use_container_width=True)
    with tab_trades:
        render_trades_table(trades_slice)

    st.caption(
        "Data source: Yahoo Finance via the local MySQL databank. Refreshing "
        "the dataset will invalidate the one-minute cache."
    )


if __name__ == "__main__":  # pragma: no cover - Streamlit runner
    main()