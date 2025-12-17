from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass
class PaperConfig:
    initial_cash: float = 2000.0
    max_usd_per_buy: float = 1000.0
    fee_rate: float = 0.001       # 0.1% pro Trade
    slippage_rate: float = 0.0005 # 0.05% Preis “schlechter”


@dataclass
class PaperState:
    cash: float
    units: float  # BTC units gehalten

    @property
    def has_position(self) -> bool:
        return self.units > 0


@dataclass
class PaperTrade:
    trade_id: int
    timestamp: pd.Timestamp
    side: str              # "BUY" / "SELL"
    decision_price: float  # close price auf dem Signal
    fill_price: float      # nach Slippage
    fee_usd: float
    qty: float
    cash_after: float
    units_after: float
    equity_after: float
    explanation: str
    plain_explanation: str
