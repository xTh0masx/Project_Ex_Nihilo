import pandas as pd
import numpy as np

def sma(df, n, col="close"):
    return df[col].rolling(n).mean()

def rsi(df, n=14, col="close"):
    delta = df[col].diff()
    up = delta.clip(lower=0).rolling(n).mean()
    down = -delta.clip(upper=0).rolling(n).mean()
    rs = up / (down.replace(0, 1e-9))
    return 100 - (100 / (1 + rs))

def atr(df, n=14):
    hl = (df["high"] - df["low"]).abs()
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n).mean()
