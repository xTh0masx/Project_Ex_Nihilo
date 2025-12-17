"""Lightweight trading loop for a single spot position.

The goal is to demonstrate the basic lifecycle of a trade using the
``SpotProfitStopStrategy``. The bot opens one position at the first observed
price, then watches subsequent prices until a profit target or stop loss is
hit.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from Logic.nn_inference import NeuralPricePredictor
from Bot_Logic.strategy import Action, SpotProfitStopStrategy


@dataclass
class TradeResult:
    """Summary of a completed or abandoned trade."""

    status: str
    entry_price: float
    exit_price: Optional[float]
    profit_pct: Optional[float]
    steps: int
    model_prediction: Optional[float] = None


class TradeBot:
    """Execute a single-asset spot trade using a profit/stop-loss rule set."""

    def __init__(
        self,
        strategy: Optional[SpotProfitStopStrategy] = None,
        symbol: str = "BTC-USD",
        predictor: Optional[NeuralPricePredictor] = None,
        prediction_exit_threshold: float = 0.0,
        entry_prediction_threshold: float = 0.0,
        max_bars_held: Optional[int] = None,
    ) -> None:
        self.strategy = strategy or SpotProfitStopStrategy()
        self.symbol = symbol
        self.predictor = predictor
        self.prediction_exit_threshold = prediction_exit_threshold
        self.entry_prediction_threshold = entry_prediction_threshold
        self.max_bars_held = max_bars_held

    def run(self, prices: Iterable[float]) -> TradeResult:
        """Run the trading loop until take-profit, stop-loss, or exhaustion.

        Args:
            prices: Iterable of observed prices. The first price becomes the
                entry price. Subsequent values are used to monitor the trade.

        Returns:
            A :class:`TradeResult` describing how the trade finished.
        """

        iterator = iter(prices)
        price_history = []
        latest_prediction: Optional[float] = None
        entry_price: Optional[float] = None
        peak_price: Optional[float] = None
        steps = 0

        for price in iterator:
            float_price = float(price)
            price_history.append(float_price)

            if entry_price is None:
                if self.predictor is not None and self.entry_prediction_threshold > 0:
                    if len(price_history) < self.predictor.lookback:
                        continue
                    latest_prediction = self.predictor.predict_from_prices(price_history)
                    if latest_prediction is None or latest_prediction < self.entry_prediction_threshold:
                        continue

                entry_price = float_price
                peak_price = float_price
                print(f"Entering {self.symbol} at {entry_price:.2f}")
                continue

            steps += 1
            peak_price = max(peak_price or float_price, float_price)

            if self.predictor is not None:
                latest_prediction = self.predictor.predict_from_prices(price_history)
                if latest_prediction is not None:
                    print(
                        f"Model prediction: {latest_prediction:+.4%} expected return over next step"
                    )
                    if (
                        self.prediction_exit_threshold > 0
                        and latest_prediction <= -self.prediction_exit_threshold
                    ):
                        profit_pct = (float_price / entry_price) - 1
                        return TradeResult(
                            status="model_exit",
                            entry_price=entry_price,
                            exit_price=float_price,
                            profit_pct=profit_pct,
                            steps=steps,
                            model_prediction=latest_prediction,
                        )

            if self.max_bars_held is not None and steps >= self.max_bars_held:
                profit_pct = (float_price / entry_price) - 1
                return TradeResult(
                    status="timeout_exit",
                    entry_price=entry_price,
                    exit_price=float_price,
                    profit_pct=profit_pct,
                    steps=steps,
                    model_prediction=latest_prediction,
                )

            recent_volatility = _recent_volatility(price_history, self.strategy.volatility_lookback)
            action = self.strategy.evaluate(
                entry_price,
                float_price,
                peak_price=peak_price,
                recent_volatility=recent_volatility,
            )
            print(f"Step {steps}: price={float_price:.2f} | action={action.name}")

            if action is Action.TAKE_PROFIT:
                profit_pct = (float_price / entry_price) - 1
                return TradeResult(
                    status="profit_target",
                    entry_price=entry_price,
                    exit_price=float_price,
                    profit_pct=profit_pct,
                    steps=steps,
                    model_prediction=latest_prediction,
                )

            if action is Action.STOP_LOSS:
                profit_pct = (float_price / entry_price) - 1
                return TradeResult(
                    status="stop_loss",
                    entry_price=entry_price,
                    exit_price=float_price,
                    profit_pct=profit_pct,
                    steps=steps,
                    model_prediction=latest_prediction,
                )

        if entry_price is None:
            return TradeResult(
                status="no_entry",
                entry_price=0.0,
                exit_price=None,
                profit_pct=None,
                steps=0,
            )

        return TradeResult(
            status="incomplete",
            entry_price=entry_price,
            exit_price=None,
            profit_pct=None,
            steps=steps,
            model_prediction=latest_prediction,
        )

def _recent_volatility(prices: Iterable[float], lookback: int) -> Optional[float]:
    """Return the standard deviation of recent percentage changes."""

    series = list(prices)
    if len(series) < 2:
        return None

    changes = []
    for prev, curr in zip(series[-(lookback + 1) : -1], series[-lookback:]):
        if prev == 0:
            continue
        changes.append((curr - prev) / prev)

    if not changes:
        return None

    mean_change = sum(changes) / len(changes)
    variance = sum((c - mean_change) ** 2 for c in changes) / len(changes)
    return variance ** 0.5