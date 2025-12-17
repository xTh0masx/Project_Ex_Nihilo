from __future__ import annotations
import pandas as pd


def explain_buy(ts: pd.Timestamp, price: float, prev_fast: float, prev_slow: float, fast: float, slow: float) -> str:
    return (
        f"BUY weil SMA_fast von unten nach oben kreuzt. "
        f"Vorher: fast={prev_fast:.2f} ≤ slow={prev_slow:.2f}. "
        f"Jetzt: fast={fast:.2f} > slow={slow:.2f}. "
        f"Preis={price:.2f}."
    )


def explain_sell(ts: pd.Timestamp, price: float, prev_fast: float, prev_slow: float, fast: float, slow: float) -> str:
    return (
        f"SELL weil SMA_fast von oben nach unten kreuzt. "
        f"Vorher: fast={prev_fast:.2f} ≥ slow={prev_slow:.2f}. "
        f"Jetzt: fast={fast:.2f} < slow={slow:.2f}. "
        f"Preis={price:.2f}."
    )
