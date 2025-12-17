#streamlit run UI_SMA/app_sma_paper_dashboard.py


from __future__ import annotations
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from UI_SMA.yahoo_feed import YahooFeedConfig, fetch_ohlcv
from UI_SMA.paper_models import PaperConfig
from UI_SMA.paper_engine import run_paper_sma_crossover
from UI_SMA.equity import trades_to_df, df_to_csv_bytes


def plot_price_and_equity(curve: pd.DataFrame, trades_df: pd.DataFrame, title: str):
    fig = go.Figure()

    # Preis
    fig.add_trace(go.Scatter(x=curve.index, y=curve["close"], name="Close"))

    # SMAs (optional)
    if "sma_fast" in curve.columns and curve["sma_fast"].notna().any():
        fig.add_trace(go.Scatter(x=curve.index, y=curve["sma_fast"], name="SMA fast"))
    if "sma_slow" in curve.columns and curve["sma_slow"].notna().any():
        fig.add_trace(go.Scatter(x=curve.index, y=curve["sma_slow"], name="SMA slow"))

    # Trades Marker
    if not trades_df.empty:
        buys = trades_df[trades_df["side"] == "BUY"]
        sells = trades_df[trades_df["side"] == "SELL"]
        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=buys["timestamp"], y=buys["decision_price"], mode="markers",
                name="BUY (decision)", marker=dict(symbol="triangle-up", size=12)
            ))
        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=sells["timestamp"], y=sells["decision_price"], mode="markers",
                name="SELL (decision)", marker=dict(symbol="triangle-down", size=12)
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Price (USD)",
        height=520,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Equity Chart separat (klarer)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=curve.index, y=curve["equity"], name="Equity"))
    fig2.update_layout(
        title="Equity Curve (Cash + BTC Wert)",
        xaxis_title="Time",
        yaxis_title="USD",
        height=320,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(fig2, use_container_width=True)


def main():
    st.set_page_config(page_title="BTC Paper Trading – SMA Crossover", layout="wide")
    st.title("BTC Paper Trading – SMA Crossover (Yahoo Finance)")

    with st.sidebar:
        st.header("Feed")
        symbol = st.text_input("Symbol", "BTC-USD")
        period = st.selectbox("Period", ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "max"], index=3)
        interval = st.selectbox("Interval", ["1m", "2m", "5m", "15m", "30m", "1h", "1d"], index=3)

        st.header("SMA")
        fast = st.number_input("Fast SMA", min_value=2, max_value=500, value=5, step=1)
        slow = st.number_input("Slow SMA", min_value=3, max_value=1000, value=20, step=1)

        st.header("Paper Portfolio")
        initial_cash = st.number_input("Initial cash (USD)", min_value=10.0, value=2000.0, step=100.0)
        max_usd_per_buy = st.number_input("Max USD per BUY", min_value=10.0, value=1000.0, step=100.0)

        st.header("Costs")
        fee_rate = st.number_input("Fee rate (e.g. 0.001 = 0.1%)", min_value=0.0, value=0.001, step=0.0005, format="%.6f")
        slippage_rate = st.number_input("Slippage (e.g. 0.0005 = 0.05%)", min_value=0.0, value=0.0005, step=0.0005, format="%.6f")

        run = st.button("Load & Run", type="primary")

    if not run:
        st.info("Parameter links setzen und **Load & Run** klicken.")
        return

    if fast >= slow:
        st.error("Fast SMA muss kleiner sein als Slow SMA.")
        return

    cfg = YahooFeedConfig(symbol=symbol, period=period, interval=interval)
    df = fetch_ohlcv(cfg)
    if df.empty:
        st.error("Keine Daten von Yahoo Finance erhalten. Bitte Period/Interval ändern.")
        return

    paper_cfg = PaperConfig(
        initial_cash=float(initial_cash),
        max_usd_per_buy=float(max_usd_per_buy),
        fee_rate=float(fee_rate),
        slippage_rate=float(slippage_rate),
    )

    result = run_paper_sma_crossover(df, fast=int(fast), slow=int(slow), cfg=paper_cfg)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Final equity", f"${result.final_equity:,.2f}")
    col2.metric("PnL", f"${result.pnl_abs:,.2f}")
    col3.metric("PnL %", f"{result.pnl_pct*100:+.2f}%")
    col4.metric("Trades", len(result.trades))

    trades_df = trades_to_df(result.trades)

    plot_price_and_equity(
        result.equity_curve,
        trades_df,
        f"{symbol} – SMA({fast}) / SMA({slow}) – Paper Trading ({period} @ {interval})"
    )

    st.subheader("Trades (mit Trade-ID, Fees, Slippage)")
    if trades_df.empty:
        st.warning("Keine Trades generiert.")
    else:
        st.dataframe(trades_df.tail(300), use_container_width=True)

        st.subheader("Letzte Entscheidung – einfach erklärt")
        st.markdown(trades_df.iloc[-1]["plain_explanation"])

    st.subheader("Downloads")
    equity_df = result.equity_curve.reset_index().rename(columns={"index": "timestamp"})
    st.download_button(
        "Download Trades CSV",
        data=df_to_csv_bytes(trades_df),
        file_name="paper_trades.csv",
        mime="text/csv",
        disabled=trades_df.empty,
    )
    st.download_button(
        "Download Equity Curve CSV",
        data=df_to_csv_bytes(equity_df),
        file_name="paper_equity_curve.csv",
        mime="text/csv",
        disabled=equity_df.empty,
    )


if __name__ == "__main__":
    main()
