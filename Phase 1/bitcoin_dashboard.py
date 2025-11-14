"""Streamlit dashboard for exploring BTC OHLCV datasets."""

from __future__ import annotations

import mysql.connector
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from mysql.connector import Error


DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "Digimon@4123",
    "database": "ex_nihilo",
    "auth_plugin": "mysql_native_password",
}


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


@st.cache_data(ttl=60)
def load_ohlcv(granularity: str) -> pd.DataFrame:
    """Load OHLCV rows for the selected granularity into a dataframe."""

    config = DATASETS[granularity]
    connection = None
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        query = (
            f"SELECT {config['time_column']} AS timestamp, open, high, low, close, volume "
            f"FROM {config['table']} ORDER BY {config['time_column']}"
        )
        frame = pd.read_sql(query, con=connection, parse_dates=["timestamp"])
        frame.set_index("timestamp", inplace=True)
        frame.sort_index(inplace=True)
        numeric_cols = ["open", "high", "low", "close", "volume"]
        frame[numeric_cols] = frame[numeric_cols].apply(pd.to_numeric, errors="coerce")
        frame.dropna(subset=numeric_cols, inplace=True)
        return frame
    finally:
        if connection is not None and connection.is_connected():
            connection.close()


def render_candlestick(frame: pd.DataFrame, title: str) -> None:
    """Render a candlestick chart for the supplied dataframe."""

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
    fig.update_layout(
        title=title,
        xaxis_title="Timestamp",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=False,
    )

    st.plotly_chart(fig, use_container_width=True)


def main():  # pragma: no cover - Streamlit entrypoint
    st.set_page_config(page_title="BTC OHLCV Dashboard", layout="wide")
    st.title("Bitcoin OHLCV Dashboard")

    st.sidebar.header("Dataset selection")
    dataset = st.sidebar.selectbox("Granularity", list(DATASETS.keys()))

    if st.sidebar.button("Refresh data", use_container_width=True):
        load_ohlcv.clear()

    try:
        frame = load_ohlcv(dataset)
    except Error as exc:
        st.error(f"Failed to load data from MySQL: {exc}")
        return

    if frame.empty:
        st.info("No data available for the selected dataset.")
        return

    render_candlestick(frame, f"BTC-USD – {dataset} candles")
    st.caption(
        "Data source: Yahoo Finance via the local MySQL databank. Refreshing "
        "the dataset will invalidate the one-minute cache."
    )


if __name__ == "__main__":  # pragma: no cover - Streamlit runner
    main()