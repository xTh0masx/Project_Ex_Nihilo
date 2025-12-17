from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import pandas as pd

from UI_SMA.sma_strategy import add_sma_columns, detect_crossover
from UI_SMA.explain import explain_buy, explain_sell

# Falls du die einfache Sprache schon eingebaut hast:
try:
    from UI_SMA.plain_language import plain_buy_explanation, plain_sell_explanation
except Exception:  # fallback, falls Datei (noch) nicht existiert
    def plain_buy_explanation() -> str:
        return "Einfach erklärt: Kurzfristiger Durchschnitt steigt über langfristigen → Kauf."

    def plain_sell_explanation() -> str:
        return "Einfach erklärt: Kurzfristiger Durchschnitt fällt unter langfristigen → Verkauf."

from UI_SMA.paper_models import PaperConfig, PaperState, PaperTrade


@dataclass
class PaperRunResult:
    trades: List[PaperTrade]
    equity_curve: pd.DataFrame  # index timestamp, cols: close, cash, units, equity, sma_fast, sma_slow
    final_equity: float
    pnl_abs: float
    pnl_pct: float


def _apply_slippage(price: float, side: str, slippage_rate: float) -> float:
    """
    BUY: du bekommst etwas schlechteren Preis (höher)
    SELL: du bekommst etwas schlechteren Preis (tiefer)
    """
    if price <= 0:
        return price
    if side.upper() == "BUY":
        return price * (1 + slippage_rate)
    return price * (1 - slippage_rate)


def _fee(notional_usd: float, fee_rate: float) -> float:
    return max(notional_usd * fee_rate, 0.0)


def run_paper_sma_crossover(
    df: pd.DataFrame,
    *,
    fast: int,
    slow: int,
    cfg: PaperConfig,
) -> PaperRunResult:
    if df.empty:
        empty_curve = pd.DataFrame(columns=["close", "cash", "units", "equity", "sma_fast", "sma_slow"])
        return PaperRunResult([], empty_curve, cfg.initial_cash, 0.0, 0.0)

    frame = add_sma_columns(df, fast=fast, slow=slow)

    state = PaperState(cash=float(cfg.initial_cash), units=0.0)
    trades: List[PaperTrade] = []
    trade_id = 0

    rows = []
    for i in range(len(frame)):
        ts = frame.index[i]
        row = frame.iloc[i]
        close = float(row["close"])
        sma_fast = row.get("sma_fast", pd.NA)
        sma_slow = row.get("sma_slow", pd.NA)

        # Default: kein Trade, aber Equity loggen
        equity = state.cash + state.units * close
        rows.append({
            "timestamp": ts,
            "close": close,
            "cash": state.cash,
            "units": state.units,
            "equity": equity,
            "sma_fast": float(sma_fast) if pd.notna(sma_fast) else None,
            "sma_slow": float(sma_slow) if pd.notna(sma_slow) else None,
        })

        if i == 0:
            continue

        prev = frame.iloc[i - 1]
        prev_fast, prev_slow = prev.get("sma_fast", pd.NA), prev.get("sma_slow", pd.NA)
        curr_fast, curr_slow = sma_fast, sma_slow

        signal = detect_crossover(prev_fast, prev_slow, curr_fast, curr_slow)
        if not signal:
            continue

        # BUY: nur wenn keine Position
        if signal == "buy" and not state.has_position and state.cash > 0:
            # Budget begrenzen
            budget = min(cfg.max_usd_per_buy, state.cash)
            if budget <= 0 or close <= 0:
                continue

            side = "BUY"
            fill_price = _apply_slippage(close, side, cfg.slippage_rate)

            # Bei BUY ist notional = budget (was wir investieren)
            fee_usd = _fee(budget, cfg.fee_rate)
            usable = max(budget - fee_usd, 0.0)
            qty = usable / fill_price if fill_price > 0 else 0.0
            if qty <= 0:
                continue

            state.cash -= budget
            state.units += qty

            trade_id += 1
            equity_after = state.cash + state.units * close

            explanation = explain_buy(ts, close, float(prev_fast), float(prev_slow), float(curr_fast), float(curr_slow))
            plain = plain_buy_explanation()

            trades.append(PaperTrade(
                trade_id=trade_id,
                timestamp=ts,
                side=side,
                decision_price=close,
                fill_price=float(fill_price),
                fee_usd=float(fee_usd),
                qty=float(qty),
                cash_after=float(state.cash),
                units_after=float(state.units),
                equity_after=float(equity_after),
                explanation=explanation,
                plain_explanation=plain,
            ))

        # SELL: nur wenn Position offen
        if signal == "sell" and state.has_position and close > 0:
            side = "SELL"
            fill_price = _apply_slippage(close, side, cfg.slippage_rate)

            # Notional = qty * fill_price
            notional = state.units * fill_price
            fee_usd = _fee(notional, cfg.fee_rate)
            proceeds = max(notional - fee_usd, 0.0)

            sold_qty = state.units
            state.units = 0.0
            state.cash += proceeds

            trade_id += 1
            equity_after = state.cash  # units=0

            explanation = explain_sell(ts, close, float(prev_fast), float(prev_slow), float(curr_fast), float(curr_slow))
            plain = plain_sell_explanation()

            trades.append(PaperTrade(
                trade_id=trade_id,
                timestamp=ts,
                side=side,
                decision_price=close,
                fill_price=float(fill_price),
                fee_usd=float(fee_usd),
                qty=float(sold_qty),
                cash_after=float(state.cash),
                units_after=float(state.units),
                equity_after=float(equity_after),
                explanation=explanation,
                plain_explanation=plain,
            ))

    equity_curve = pd.DataFrame(rows).set_index("timestamp")
    final_equity = float(equity_curve["equity"].iloc[-1]) if not equity_curve.empty else cfg.initial_cash
    pnl_abs = final_equity - cfg.initial_cash
    pnl_pct = (pnl_abs / cfg.initial_cash) if cfg.initial_cash else 0.0

    return PaperRunResult(
        trades=trades,
        equity_curve=equity_curve,
        final_equity=final_equity,
        pnl_abs=float(pnl_abs),
        pnl_pct=float(pnl_pct),
    )
