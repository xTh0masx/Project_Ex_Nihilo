from __future__ import annotations
import pandas as pd


def add_sma_columns(df: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    out = df.copy()
    out["sma_fast"] = out["close"].rolling(window=fast, min_periods=fast).mean()
    out["sma_slow"] = out["close"].rolling(window=slow, min_periods=slow).mean()
    return out


def detect_crossover(prev_fast: float, prev_slow: float, curr_fast: float, curr_slow: float) -> str:
    """
    Liefert: "buy" wenn fast von unten nach oben kreuzt,
             "sell" wenn fast von oben nach unten kreuzt,
             "" sonst.
    """
    if any(pd.isna(x) for x in [prev_fast, prev_slow, curr_fast, curr_slow]):
        return ""
    if prev_fast <= prev_slow and curr_fast > curr_slow:
        return "buy"
    if prev_fast >= prev_slow and curr_fast < curr_slow:
        return "sell"
    return ""
