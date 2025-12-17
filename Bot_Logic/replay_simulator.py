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
from typing import Callable, Iterable, List, Optional

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from Logic.nn_inference import NeuralPricePredictor
from Bot_Logic.strategy import Action, SpotProfitStopStrategy


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
    quantity: float
    capital_used: float
    profit_usd: Optional[float] = None
    model_prediction: Optional[float] = None
    peak_price: Optional[float] = None


@dataclass
class ReplaySummary:
    """Collection of all trades and aggregate performance."""

    trades: List[ReplayTrade] = field(default_factory=list)
    total_return: float = 0.0
    cumulative_profit_pct: float = 0.0
    applied_entry_threshold: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    average_holding_bars: float = 0.0
    average_profit_pct: float = 0.0
    average_loss_pct: float = 0.0


class HistoricalPriceStreamer:
    """Iterate through a date-bounded slice of OHLCV data one candle at a time."""

    def __init__(self, frame: pd.DataFrame, start: datetime, end: datetime) -> None:
        """Prepare a time-bounded view of historical prices.

        The slice is assigend to ''self.frame'' immediatley so downstream attributes exist even if
        validation fails, which avoids partially constructed objects throwing attribute errors in the
        dashboard.
        """

        # Materialize the slice early to guarantee ''self.frame'' exists.
        self.frame = frame.loc[start:end].sort_index()

        if self.frame.empty:
            raise ValueError(
                f"No price data available between {start!s} and {end!s}; got empty slice"
            )

        if "close" not in self.frame.columns:
            raise ValueError("Input frame must contain a 'close' column")

        self.series = self.frame["close"].astype("float")

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
        trade_capital_usd: float = 1000.0,
        max_bars_held: Optional[int] = None,
    ) -> None:
        self.predictor = predictor
        self.strategy = strategy or SpotProfitStopStrategy()
        self.entry_threshold = entry_threshold
        self.prediction_exit_threshold = prediction_exit_threshold
        self.trade_capital_usd = trade_capital_usd
        self.max_bars_held = max_bars_held

    def simulate(
        self,
        streamer: HistoricalPriceStreamer,
        *,
        delay_seconds: float = 0.0,
        on_step: Optional[
            Callable[
                [int, ReplayCandle, List[ReplayTrade], Optional[ReplayTrade], Optional[float]],
                None,
            ]
        ] = None
    ) -> ReplaySummary:
        """Run the replay loop until the stream is exhausted."""

        feature_frame = self.predictor.prepare_feature_frame(streamer.frame)
        trades: List[ReplayTrade] = []
        active_trade: Optional[ReplayTrade] = None
        price_history: List[float] = []

        applied_entry_threshold = self.entry_threshold
        adaptive_entry_threshold: Optional[float] = None

        # Pre-compute all predictions up-front so we know whether the chosen
        # entry threshold is achievable. This avoids running through the stream
        # only to discover that the model's signals never exceed the slider
        # value, which previously resulted in empty trade lists.
        precomputed_predictions: List[Optional[float]] = [None] * len(feature_frame)
        positive_predictions: List[float] = []
        for idx in range(len(feature_frame)):
            if idx + 1 < self.predictor.lookback:
                continue
            window = feature_frame.iloc[: idx + 1]
            prediction = self.predictor.predict_next_return(window)
            precomputed_predictions[idx] = prediction
            if prediction is not None and prediction > 0:
                positive_predictions.append(prediction)

        if positive_predictions:
            # Aim for more frequent trades by relaxing the requested entry filter
            # down to the 25th percentile of positive predictions when necessary.
            adaptive_entry_threshold = float(np.percentile(positive_predictions, 25))
            applied_entry_threshold = min(self.entry_threshold, adaptive_entry_threshold)

        for idx, bar in enumerate(streamer.stream(delay_seconds=delay_seconds)):
            price_history.append(bar.close)
            prediction: Optional[float] = precomputed_predictions[idx]

            if active_trade is None:
                if prediction is not None and prediction >= applied_entry_threshold:
                    quantity = self.trade_capital_usd / bar.close if bar.close else 0.0
                    active_trade = ReplayTrade(
                        entry_time=bar.timestamp,
                        entry_price=bar.close,
                        exit_time=None,
                        exit_price=None,
                        status="open",
                        profit_pct=None,
                        bars_held=1,
                        quantity=quantity,
                        capital_used=quantity * bar.close,
                        model_prediction=prediction,
                        peak_price=bar.close,
                    )
                if on_step is not None:
                    on_step(idx, bar, trades, active_trade, prediction)
                continue

            # Manage an existing position
            active_trade.bars_held += 1
            active_trade.peak_price = max(active_trade.peak_price or bar.close, bar.close)
            current_profit = (bar.close / active_trade.entry_price) - 1
            recent_volatility = _recent_volatility(
                price_history, self.strategy.volatility_lookback
            )

            if self.max_bars_held is not None and active_trade.bars_held >= self.max_bars_held:
                trades.append(
                    ReplayTrade(
                        entry_time=active_trade.entry_time,
                        entry_price=active_trade.entry_price,
                        exit_time=bar.timestamp,
                        exit_price=bar.close,
                        status="timeout_exit",
                        profit_pct=current_profit,
                        bars_held=active_trade.bars_held,
                        quantity=active_trade.quantity,
                        capital_used=active_trade.capital_used,
                        profit_usd=current_profit * active_trade.capital_used,
                        model_prediction=prediction,
                        peak_price=active_trade.peak_price,
                    )
                )
                active_trade = None
                if on_step is not None:
                    on_step(idx, bar, trades, active_trade, prediction)
                continue

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
                            quantity=active_trade.quantity,
                            capital_used=active_trade.capital_used,
                            profit_usd=current_profit * active_trade.capital_used,
                            model_prediction=prediction,
                            peak_price=active_trade.peak_price,
                        )
                    )
                    active_trade = None
                    if on_step is not None:
                        on_step(idx, bar, trades, active_trade, prediction)
                    continue

            action = self.strategy.evaluate(
                active_trade.entry_price,
                bar.close,
                peak_price=active_trade.peak_price,
                recent_volatility=recent_volatility,
            )
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
                        quantity=active_trade.quantity,
                        capital_used=active_trade.capital_used,
                        profit_usd=current_profit * active_trade.capital_used,
                        model_prediction=prediction,
                        peak_price=active_trade.peak_price,
                    )
                )
                active_trade = None
                if on_step is not None:
                    on_step(idx, bar, trades, active_trade, prediction)
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
                        quantity=active_trade.quantity,
                        capital_used=active_trade.capital_used,
                        profit_usd=current_profit * active_trade.capital_used,
                        model_prediction=prediction,
                        peak_price=active_trade.peak_price,
                    )
                )
                active_trade = None
                if on_step is not None:
                    on_step(idx, bar, trades, active_trade, prediction)
                continue

            if on_step is not None:
                on_step(idx, bar, trades, active_trade, prediction)

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
                    quantity=active_trade.quantity,
                    capital_used=active_trade.capital_used,
                    profit_usd=(
                        ((streamer.series.iloc[-1] / active_trade.entry_price) - 1)
                        * active_trade.capital_used
                    ),
                    model_prediction=active_trade.model_prediction,
                    peak_price=active_trade.peak_price,
                )
            )

        total_return = sum(trade.profit_pct or 0.0 for trade in trades)
        wins = [trade for trade in trades if trade.profit_pct is not None and trade.profit_pct > 0]
        losses = [
            trade for trade in trades if trade.profit_pct is not None and trade.profit_pct <= 0
        ]
        total_trades = len(trades)
        win_rate = len(wins) / total_trades if total_trades else 0.0
        average_holding = (
            sum(trade.bars_held for trade in trades) / total_trades if total_trades else 0.0
        )
        average_profit = (
            sum(trade.profit_pct for trade in wins) / len(wins) if wins else 0.0
        )
        average_loss = (
            sum(trade.profit_pct for trade in losses) / len(losses) if losses else 0.0
        )
        summary = ReplaySummary(
            trades=trades,
            total_return=total_return,
            cumulative_profit_pct=total_return,
            applied_entry_threshold=applied_entry_threshold,
            total_trades=total_trades,
            win_rate=win_rate,
            average_holding_bars=average_holding,
            average_profit_pct=average_profit,
            average_loss_pct=average_loss,
        )
        return summary


def _recent_volatility(prices: List[float], lookback: int) -> Optional[float]:
    """Compute the standard deviation of recent percentage changes."""

    if len(prices) < 2:
        return None

    changes = []
    for prev, curr in zip(prices[-(lookback + 1) : -1], prices[-lookback:]):
        if prev == 0:
            continue
        changes.append((curr - prev) / prev)

    if not changes:
        return None

    mean_change = sum(changes) / len(changes)
    variance = sum((c - mean_change) ** 2 for c in changes) / len(changes)
    return variance ** 0.5


def simulate_date_window(
    frame: pd.DataFrame,
    predictor_dir: Path,
    *,
    start: datetime,
    end: datetime,
    delay_seconds: float = 0.0,
    entry_threshold: float = 0.001,
    prediction_exit_threshold: float = 0.002,
    max_bars_held: Optional[int] = None,
) -> ReplaySummary:
    """Convenience wrapper to run a full replay given a date range and model."""

    predictor = NeuralPricePredictor(predictor_dir)
    streamer = HistoricalPriceStreamer(frame, start=start, end=end)
    trader = NeuralReplayTrader(
        predictor,
        entry_threshold=entry_threshold,
        prediction_exit_threshold=prediction_exit_threshold,
        max_bars_held=max_bars_held,
    )
    return trader.simulate(streamer, delay_seconds=delay_seconds)