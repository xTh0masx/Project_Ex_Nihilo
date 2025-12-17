from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import pandas as pd

from UI_SMA.sma_strategy import add_sma_columns, detect_crossover
from UI_SMA.explain import explain_buy, explain_sell
from UI_SMA.plain_language import plain_buy_explanation, plain_sell_explanation



@dataclass
class TradeEvent:
    timestamp: pd.Timestamp
    side: str
    price: float
    units: float
    cash_after: float
    sma_fast: float
    sma_slow: float
    explanation: str           # technisch
    plain_explanation: str     # 🆕 Alltagssprache



@dataclass
class TradeRunResult:
    events: List[TradeEvent]
    final_cash: float
    final_units: float
    equity_last: float
    pnl_abs: float
    pnl_pct: float


def simulate_sma_crossover(
    df: pd.DataFrame,
    *,
    fast: int,
    slow: int,
    initial_cash: float = 2000.0,
    risk_per_trade_usd: float = 1000.0,
) -> TradeRunResult:
    """
    Long-only Spot:
    - BUY bei bullish crossover, wenn keine Position offen ist
    - SELL bei bearish crossover, wenn Position offen ist
    """
    if df.empty:
        return TradeRunResult([], initial_cash, 0.0, initial_cash, 0.0, 0.0)

    frame = add_sma_columns(df, fast=fast, slow=slow)

    cash = float(initial_cash)
    units = 0.0
    events: List[TradeEvent] = []

    # wir brauchen mindestens 2 bars, um prev vs curr zu vergleichen
    for i in range(1, len(frame)):
        row_prev = frame.iloc[i - 1]
        row = frame.iloc[i]
        ts = frame.index[i]
        price = float(row["close"])

        prev_fast, prev_slow = row_prev["sma_fast"], row_prev["sma_slow"]
        curr_fast, curr_slow = row["sma_fast"], row["sma_slow"]

        signal = detect_crossover(prev_fast, prev_slow, curr_fast, curr_slow)
        if not signal:
            continue

        # BUY nur wenn keine Position
        if signal == "buy" and units <= 0 and cash > 0:
            size = min(risk_per_trade_usd, cash)
            buy_units = size / price if price > 0 else 0.0
            if buy_units <= 0:
                continue
            cash -= buy_units * price
            units += buy_units
            expl = explain_buy(ts, price, float(prev_fast), float(prev_slow), float(curr_fast), float(curr_slow))
            events.append(
                TradeEvent(
                    ts, "BUY", price, units, cash,
                    float(curr_fast), float(curr_slow),
                    expl,
                    plain_buy_explanation()
                )
            )

        # SELL nur wenn Position offen
        if signal == "sell" and units > 0:
            cash += units * price
            expl = explain_sell(ts, price, float(prev_fast), float(prev_slow), float(curr_fast), float(curr_slow))
            units = 0.0
            events.append(
                TradeEvent(
                    ts, "SELL", price, units, cash,
                    float(curr_fast), float(curr_slow),
                    expl,
                    plain_sell_explanation()
                )
            )

    last_price = float(frame["close"].iloc[-1])
    equity_last = cash + units * last_price
    pnl_abs = equity_last - initial_cash
    pnl_pct = (pnl_abs / initial_cash) if initial_cash else 0.0

    return TradeRunResult(
        events=events,
        final_cash=cash,
        final_units=units,
        equity_last=equity_last,
        pnl_abs=pnl_abs,
        pnl_pct=pnl_pct,
    )
