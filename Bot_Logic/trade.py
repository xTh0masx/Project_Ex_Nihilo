"""Lightweight trading loop for a single spot position.

The goal is to demonstrate the basic lifecycle of a trade using the
``SpotProfitStopStrategy``. The bot opens one position at the first observed
price, then watches subsequent prices until a profit target or stop loss is
hit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from strategy import Action, SpotProfitStopStrategy


@dataclass
class TradeResult:
    """Summary of a completed or abandoned trade."""

    status: str
    entry_price: float
    exit_price: Optional[float]
    profit_pct: Optional[float]
    steps: int


class TradeBot:
    """Execute a single-asset spot trade using a profit/stop-loss rule set."""

    def __init__(
        self,
        strategy: Optional[SpotProfitStopStrategy] = None,
        symbol: str = "BTC-USD",
    ) -> None:
        self.strategy = strategy or SpotProfitStopStrategy()
        self.symbol = symbol

    def run(self, prices: Iterable[float]) -> TradeResult:
        """Run the trading loop until take-profit, stop-loss, or exhaustion.

        Args:
            prices: Iterable of observed prices. The first price becomes the
                entry price. Subsequent values are used to monitor the trade.

        Returns:
            A :class:`TradeResult` describing how the trade finished.
        """

        iterator = iter(prices)
        try:
            entry_price = float(next(iterator))
        except StopIteration as exc:  # pragma: no cover - defensive guard
            raise ValueError("At least one price is required to enter a trade") from exc

        print(f"Entering {self.symbol} at {entry_price:.2f}")
        steps = 0

        for price in iterator:
            steps += 1
            action = self.strategy.evaluate(entry_price, float(price))
            print(
                f"Step {steps}: price={float(price):.2f} | action={action.name}")

            if action is Action.TAKE_PROFIT:
                profit_pct = (float(price) / entry_price) - 1
                return TradeResult(
                    status="profit_target",
                    entry_price=entry_price,
                    exit_price=float(price),
                    profit_pct=profit_pct,
                    steps=steps,
                )

            if action is Action.STOP_LOSS:
                profit_pct = (float(price) / entry_price) - 1
                return TradeResult(
                    status="stop_loss",
                    entry_price=entry_price,
                    exit_price=float(price),
                    profit_pct=profit_pct,
                    steps=steps,
                )

        return TradeResult(
            status="incomplete",
            entry_price=entry_price,
            exit_price=None,
            profit_pct=None,
            steps=steps,
        )