import yfinance as yf
import pandas as pd

def fetch_btc(interval="1h", lookback_days=365):
    t = yf.Ticker("BTC-USD")
    df = t.history(period=f"{lookback_days}d", interval=interval, actions=False)
    # Liefert OHLCV mit DatetimeIndex (TZ-aware). Bei Bedarf lücken prüfen:
    df = df.dropna(subset=["Open","High","Low","Close","Volume"])
    return df

