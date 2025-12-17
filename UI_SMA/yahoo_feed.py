from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class YahooFeedConfig:
    symbol: str = "BTC-USD"
    period: str = "7d"       # z.B. "1d", "5d", "1mo", "3mo", "1y", "max"
    interval: str = "15m"    # z.B. "1m","2m","5m","15m","1h","1d"


def fetch_ohlcv(cfg: YahooFeedConfig) -> pd.DataFrame:
    """
    Lädt OHLCV direkt von Yahoo Finance (ohne DB).
    Gibt DataFrame mit Index=Datetime und Spalten: open, high, low, close, volume
    """
    t = yf.Ticker(cfg.symbol)
    df = t.history(period=cfg.period, interval=cfg.interval, auto_adjust=False, actions=False)

    if df is None or df.empty:
        return pd.DataFrame()

    # yfinance liefert meist: Open High Low Close Volume
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    rename = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    for k, v in rename.items():
        if k in df.columns:
            df.rename(columns={k: v}, inplace=True)
        elif v not in df.columns:
            df[v] = pd.NA

    df = df[["open", "high", "low", "close", "volume"]]
    df.dropna(subset=["close"], inplace=True)
    df.sort_index(inplace=True)
    return df
