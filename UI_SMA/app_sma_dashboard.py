#py -m streamlit run UI_SMA/app_sma_dashboard.py


from __future__ import annotations
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from UI_SMA.yahoo_feed import YahooFeedConfig, fetch_ohlcv
from UI_SMA.sma_strategy import add_sma_columns
from UI_SMA.trade_simulator import simulate_sma_crossover


def plot_chart(df: pd.DataFrame, trades: pd.DataFrame, title: str):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="BTC-USD"
    ))

    if "sma_fast" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["sma_fast"], name="SMA fast"))
    if "sma_slow" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["sma_slow"], name="SMA slow"))

    if not trades.empty:
        buys = trades[trades["side"] == "BUY"]
        sells = trades[trades["side"] == "SELL"]

        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=buys["timestamp"], y=buys["price"], mode="markers",
                name="BUY", marker=dict(symbol="triangle-up", size=12)
            ))
        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=sells["timestamp"], y=sells["price"], mode="markers",
                name="SELL", marker=dict(symbol="triangle-down", size=12)
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=False,
        height=650,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.set_page_config(page_title="BTC SMA Crossover (Yahoo Finance)", layout="wide")
    st.title("BTC Spot Bot – SMA Crossover (Yahoo Finance Live)")

    with st.sidebar:
        st.header("Yahoo Finance Feed")
        symbol = st.text_input("Symbol", "BTC-USD")
        period = st.selectbox("Period", ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "max"], index=2)
        interval = st.selectbox("Interval", ["1m", "2m", "5m", "15m", "30m", "1h", "1d"], index=3)

        st.header("SMA Strategy")
        fast = st.number_input("Fast SMA", min_value=2, max_value=500, value=5, step=1)
        slow = st.number_input("Slow SMA", min_value=3, max_value=1000, value=20, step=1)

        st.header("Simulation / Position Sizing")
        initial_cash = st.number_input("Initial cash (USD)", min_value=10.0, value=2000.0, step=100.0)
        risk_per_trade = st.number_input("Max USD per trade", min_value=10.0, value=1000.0, step=100.0)

        run = st.button("Load & Run", type="primary")

    if not run:
        st.info("Konfiguriere links die Parameter und klicke **Load & Run**.")
        return

    if fast >= slow:
        st.error("Fast SMA muss kleiner sein als Slow SMA.")
        return

    cfg = YahooFeedConfig(symbol=symbol, period=period, interval=interval)
    df = fetch_ohlcv(cfg)

    if df.empty:
        st.error("Keine Daten von Yahoo Finance erhalten. Bitte Period/Interval ändern.")
        return

    df = add_sma_columns(df, fast=fast, slow=slow)

    result = simulate_sma_crossover(
        df,
        fast=fast,
        slow=slow,
        initial_cash=float(initial_cash),
        risk_per_trade_usd=float(risk_per_trade),
    )

    events = [{
        "timestamp": e.timestamp,
        "side": e.side,
        "price": e.price,
        "units_after": e.units,
        "cash_after": e.cash_after,
        "sma_fast": e.sma_fast,
        "sma_slow": e.sma_slow,
        "explanation": e.explanation,
        "plain_explanation": e.plain_explanation,  # 🆕
    } for e in result.events]

    trades_df = pd.DataFrame(events)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Final equity", f"${result.equity_last:,.2f}")
    col2.metric("PnL", f"${result.pnl_abs:,.2f}")
    col3.metric("PnL %", f"{result.pnl_pct*100:+.2f}%")
    col4.metric("Trades", len(trades_df))

    plot_chart(df, trades_df, f"{symbol} – SMA({fast}) / SMA({slow}) – {period} @ {interval}")

    st.subheader("Explainable Trade Log")
    if trades_df.empty:
        st.warning("Keine Trades generiert (keine Crossovers im Zeitraum / Warmup zu kurz).")
    else:
        st.dataframe(trades_df.tail(200), use_container_width=True)

        st.subheader("Letzte Entscheidung – technisch")
        st.write(trades_df.iloc[-1]["explanation"])

        st.subheader("Letzte Entscheidung – einfach erklärt")
        st.markdown(trades_df.iloc[-1]["plain_explanation"])


if __name__ == "__main__":
    main()
