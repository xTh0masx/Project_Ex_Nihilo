"""Simple spot trading strategy definitions.

This module provides a minimal rule set for entering a single spot position
with a fixed take‑profit and stop‑loss threshold. The strategy itself remains
stateless; it only evaluates the current price relative to the entry price.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


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
    Optional trailing and volatility-aware stops tighten exits as the trade progress.
    """

    target_profit: float = 0.01  # 1% default take-profit
    stop_loss: float = 0.05  # 5% default stop-loss
    trailing_stop_loss: Optional[float] = 0.03 # 3% trail from the peak
    volatility_lookback: int = 20 # window to measure pct-change volatility
    volatility_stop_multiplier: float = 1.5 # widen stop during choppier regimes

    def evaluate(
        self,
        entry_price: float,
        current_price: float,
        *,
        peak_price: Optional[float] = None,
        recent_volatility: Optional[float] = None,
    ) -> Action:
        """Return the next action based on the current price.

        Args:
            entry_price: Price where the position was opened.
            current_price: Latest observed market price.
            peak_price: Highest observed price since entry, used for trailing stops.
            recent_volatility: Standard deviation of recent percentage changes;
            when supplied, it tightens the stop in volatile conditions.

        Returns:
            An :class:`Action` indicating whether to hold, take profit, or stop
            due to losses.
        """

        if current_price >= entry_price * (1 + self.target_profit):
            return Action.TAKE_PROFIT

        stop_threshold = entry_price * (1 - self.stop_loss)

        if self.trailing_stop_loss is not None and peak_price is not None:
            trailing_threshold = peak_price * (1 - self.trailing_stop_loss)
            stop_threshold = max(stop_threshold, trailing_threshold)

        if recent_volatility is not None:
            adaptive_threshold = entry_price * (1 - recent_volatility * self.volatility_stop_multiplier)
            stop_threshold = max(stop_threshold, adaptive_threshold)

        if current_price <= stop_threshold:
            return Action.STOP_LOSS

        return Action.HOLD
