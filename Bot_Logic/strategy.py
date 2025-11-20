"""Simple spot trading strategy definitions.

This module provides a minimal rule set for entering a single spot position
with a fixed take‑profit and stop‑loss threshold. The strategy itself remains
stateless; it only evaluates the current price relative to the entry price.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class Action(Enum):
    """Discrete actions the strategy can request from the trading loop."""

    HOLD = auto()
    TAKE_PROFIT = auto()
    STOP_LOSS = auto()


@dataclass(frozen=True)
class SpotProfitStopStrategy:
    """Rule set to manage a single long position.

    The strategy assumes you are already in a spot position and monitors price
    changes until either the profit target or stop loss is reached.
    """

    target_profit: float = 0.01  # 1% default take-profit
    stop_loss: float = 0.10  # 10% default stop-loss

    def evaluate(self, entry_price: float, current_price: float) -> Action:
        """Return the next action based on the current price.

        Args:
            entry_price: Price where the position was opened.
            current_price: Latest observed market price.

        Returns:
            An :class:`Action` indicating whether to hold, take profit, or stop
            due to losses.
        """

        if current_price >= entry_price * (1 + self.target_profit):
            return Action.TAKE_PROFIT

        if current_price <= entry_price * (1 - self.stop_loss):
            return Action.STOP_LOSS

        return Action.HOLD
