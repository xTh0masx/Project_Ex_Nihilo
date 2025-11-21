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
from strategy import Action, SpotProfitStopStrategy


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
    ) -> None:
        self.strategy = strategy or SpotProfitStopStrategy()
        self.symbol = symbol
        self.predictor = predictor
        self.prediction_exit_threshold = prediction_exit_threshold

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
        price_history = [entry_price]
        latest_prediction: Optional[float] = None

        for price in iterator:
            steps += 1
            float_price = float(price)
            price_history.append(float_price)

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

            action = self.strategy.evaluate(entry_price, float_price)
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

        return TradeResult(
            status="incomplete",
            entry_price=entry_price,
            exit_price=None,
            profit_pct=None,
            steps=steps,
        )