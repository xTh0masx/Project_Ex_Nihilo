from .indicators import sma
import pandas as pd

def signal_sma_crossover(df: pd.DataFrame, short=20, long=50) -> pd.DataFrame:
    out = df.copy()
    out["smaS"] = sma(out["Close"], short)
    out["smaL"] = sma(out["Close"], long)
    out["prev_diff"] = (out["smaS"] - out["smaL"]).shift(1)
    out["diff"] = out["smaS"] - out["smaL"]

    buy = (out["prev_diff"] <= 0) & (out["diff"] > 0)
    sell = (out["prev_diff"] >= 0) & (out["diff"] < 0)

    out["signal"] = "HOLD"
    out.loc[buy, "signal"] = "BUY"
    out.loc[sell, "signal"] = "SELL"
    return out
