"""Replay historical candles as if they were a live ticker feed.

The simulator reveals one bar at a time inside a chosen date window, allowing
neural-network predictions to drive entries and exits without any look-ahead
bias. It is designed to power dashboard replays where each past candle arrives
with a configurable delay, mirroring a live ticker stream.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from Logic.nn_inference import NeuralPricePredictor
from strategy import Action, SpotProfitStopStrategy


@dataclass
class ReplayCandle:
    """A single candle emitted by the historical stream."""

    timestamp: pd.Timestamp
    close: float


@dataclass
class ReplayTrade:
    """Lifecycle of one simulated trade during the replay."""

    entry_time: pd.Timestamp
    entry_price: float
    exit_time: Optional[pd.Timestamp]
    exit_price: Optional[float]
    status: str
    profit_pct: Optional[float]
    bars_held: int
    model_prediction: Optional[float] = None


@dataclass
class ReplaySummary:
    """Collection of all trades and aggregate performance."""

    trades: List[ReplayTrade] = field(default_factory=list)
    total_return: float = 0.0
    cumulative_profit_pct: float = 0.0


class HistoricalPriceStreamer:
    """Iterate through a date-bounded slice of OHLCV data one candle at a time."""

    def __init__(self, frame: pd.DataFrame, start: datetime, end: datetime) -> None:
        if "close" not in frame.columns:
            raise ValueError("Input frame must contain a 'close' column")

        bounded = frame.loc[start:end]
        if bounded.empty:
            raise ValueError(
                f"No price data available between {start!s} and {end!s}; got empty slice"
            )

        self.series = bounded["close"].astype(float)

    def stream(self, delay_seconds: float = 0.0) -> Iterable[ReplayCandle]:
        """Yield one candle at a time, optionally pausing between steps."""

        for timestamp, price in self.series.items():
            yield ReplayCandle(timestamp=pd.Timestamp(timestamp), close=float(price))
            if delay_seconds > 0:
                time.sleep(delay_seconds)


class NeuralReplayTrader:
    """Simulate trading decisions on a sequentially revealed price stream."""

    def __init__(
        self,
        predictor: NeuralPricePredictor,
        strategy: Optional[SpotProfitStopStrategy] = None,
        entry_threshold: float = 0.001,
        prediction_exit_threshold: float = 0.002,
    ) -> None:
        self.predictor = predictor
        self.strategy = strategy or SpotProfitStopStrategy()
        self.entry_threshold = entry_threshold
        self.prediction_exit_threshold = prediction_exit_threshold

    def simulate(
        self,
        streamer: HistoricalPriceStreamer,
        *,
        delay_seconds: float = 0.0,
    ) -> ReplaySummary:
        """Run the replay loop until the stream is exhausted."""

        price_history: List[float] = []
        trades: List[ReplayTrade] = []
        active_trade: Optional[ReplayTrade] = None

        for bar in streamer.stream(delay_seconds=delay_seconds):
            price_history.append(bar.close)
            prediction: Optional[float] = None

            if len(price_history) >= self.predictor.lookback:
                prediction = self.predictor.predict_from_prices(price_history)

            if active_trade is None:
                if prediction is not None and prediction >= self.entry_threshold:
                    active_trade = ReplayTrade(
                        entry_time=bar.timestamp,
                        entry_price=bar.close,
                        exit_time=None,
                        exit_price=None,
                        status="open",
                        profit_pct=None,
                        bars_held=0,
                        model_prediction=prediction,
                    )
                continue

            # Manage an existing position
            active_trade.bars_held += 1
            current_profit = (bar.close / active_trade.entry_price) - 1

            if prediction is not None:
                active_trade.model_prediction = prediction
                if self.prediction_exit_threshold > 0 and prediction <= -self.prediction_exit_threshold:
                    trades.append(
                        ReplayTrade(
                            entry_time=active_trade.entry_time,
                            entry_price=active_trade.entry_price,
                            exit_time=bar.timestamp,
                            exit_price=bar.close,
                            status="model_exit",
                            profit_pct=current_profit,
                            bars_held=active_trade.bars_held,
                            model_prediction=prediction,
                        )
                    )
                    active_trade = None
                    continue

            action = self.strategy.evaluate(active_trade.entry_price, bar.close)
            if action is Action.TAKE_PROFIT:
                trades.append(
                    ReplayTrade(
                        entry_time=active_trade.entry_time,
                        entry_price=active_trade.entry_price,
                        exit_time=bar.timestamp,
                        exit_price=bar.close,
                        status="profit_target",
                        profit_pct=current_profit,
                        bars_held=active_trade.bars_held,
                        model_prediction=prediction,
                    )
                )
                active_trade = None
                continue

            if action is Action.STOP_LOSS:
                trades.append(
                    ReplayTrade(
                        entry_time=active_trade.entry_time,
                        entry_price=active_trade.entry_price,
                        exit_time=bar.timestamp,
                        exit_price=bar.close,
                        status="stop_loss",
                        profit_pct=current_profit,
                        bars_held=active_trade.bars_held,
                        model_prediction=prediction,
                    )
                )
                active_trade = None
                continue

        if active_trade is not None:
            trades.append(
                ReplayTrade(
                    entry_time=active_trade.entry_time,
                    entry_price=active_trade.entry_price,
                    exit_time=streamer.series.index[-1],
                    exit_price=float(streamer.series.iloc[-1]),
                    status="incomplete",
                    profit_pct=(streamer.series.iloc[-1] / active_trade.entry_price) - 1,
                    bars_held=active_trade.bars_held,
                    model_prediction=active_trade.model_prediction,
                )
            )

        total_return = sum(trade.profit_pct or 0.0 for trade in trades)
        summary = ReplaySummary(
            trades=trades,
            total_return=total_return,
            cumulative_profit_pct=total_return,
        )
        return summary


def simulate_date_window(
    frame: pd.DataFrame,
    predictor_dir: Path,
    *,
    start: datetime,
    end: datetime,
    delay_seconds: float = 0.0,
    entry_threshold: float = 0.001,
    prediction_exit_threshold: float = 0.002,
) -> ReplaySummary:
    """Convenience wrapper to run a full replay given a date range and model."""

    predictor = NeuralPricePredictor(predictor_dir)
    streamer = HistoricalPriceStreamer(frame, start=start, end=end)
    trader = NeuralReplayTrader(
        predictor,
        entry_threshold=entry_threshold,
        prediction_exit_threshold=prediction_exit_threshold,
    )
    return trader.simulate(streamer, delay_seconds=delay_seconds)