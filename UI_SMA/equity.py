from __future__ import annotations
import pandas as pd


def trades_to_df(trades) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame([{
        "trade_id": t.trade_id,
        "timestamp": t.timestamp,
        "side": t.side,
        "decision_price": t.decision_price,
        "fill_price": t.fill_price,
        "fee_usd": t.fee_usd,
        "qty": t.qty,
        "cash_after": t.cash_after,
        "units_after": t.units_after,
        "equity_after": t.equity_after,
        "explanation": t.explanation,
        "plain_explanation": t.plain_explanation,
    } for t in trades])


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")
