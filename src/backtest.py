import pandas as pd
import numpy as np

def backtest_sma(df: pd.DataFrame, start_cash=1000.0):
    cash, btc = start_cash, 0.0
    equity_curve = []
    position = 0  # 0=flat, 1=long
    last_price = None

    for _, r in df.iterrows():
        price = float(r["Close"])
        sig = r["signal"]

        # Regel: nur auf neue Kreuzsignale reagieren
        if sig == "BUY" and position == 0:
            btc = cash / price
            cash = 0
            position = 1
        elif sig == "SELL" and position == 1:
            cash = btc * price
            btc = 0
            position = 0

        equity_curve.append(cash + btc * price)
        last_price = price

    df = df.copy()
    df["equity"] = equity_curve
    total_return = (df["equity"].iloc[-1] / start_cash) - 1
    max_dd = ((df["equity"].cummax() - df["equity"]) / df["equity"].cummax()).max()
    trades = ((df["signal"] == "BUY") | (df["signal"] == "SELL")).sum()
    return {"return": float(total_return), "max_drawdown": float(max_dd), "trades": int(trades), "equity_series": df["equity"]}
