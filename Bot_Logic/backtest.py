"""Minimal backtest harness for the spot trading bot."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from Logic.nn_inference import NeuralPricePredictor, create_minimal_demo_model
from Bot_Logic.replay_simulator import HistoricalPriceStreamer, NeuralReplayTrader
from trade import TradeBot


def demo_run() -> None:
    """Run the bot against a tiny synthetic price series.

    The price path is crafted to trigger the 1% take-profit condition. Adjust
    the list to explore different outcomes; e.g., insert a sharp drop to
    observe the 10% stop-loss exit.
    """

    base_price = 100.0
    prices = [base_price * (1+ 0.0002 * step) for step in range(65)]

    model_dir = Path("model/demo_predictor")
    if not model_dir.exists():
        print("Demo model not found; creating a lightweight demo artefact...")
        create_minimal_demo_model(model_dir)

    predictor = NeuralPricePredictor(model_dir)
    bot = TradeBot(predictor=predictor, prediction_exit_threshold=0.002)
    result = bot.run(prices)

    print("\nFinal result:")
    print(result)

def demo_replay() -> None:
    """Replay a historical window one candle at a time using the neural bot.

    This mirrors a dashboard playback: only candles within the configured
    window are revealed, with the neural predictor suggesting entries and the
    rule-based strategy handling exits.
    """

    import pandas as pd

    start = datetime(2025, 10, 31)
    end = datetime(2025, 11, 21)
    date_index = pd.date_range(start=start, end=end, freq="D")

    base_price = 30000.0
    closes = [base_price * (1 + 0.0008 * idx + (0.01 if idx % 5 == 0 else 0)) for idx in range(len(date_index))]
    frame = pd.DataFrame({"close": closes}, index=date_index)

    model_dir = Path("models/demo_predictor")
    if not model_dir.exists():
        print("Demo model not found; creating a lightweight demo artefact...")
        create_minimal_demo_model(model_dir)

    predictor = NeuralPricePredictor(model_dir)
    trader = NeuralReplayTrader(predictor, entry_threshold=0.001, prediction_exit_threshold=0.002)
    streamer = HistoricalPriceStreamer(frame, start=start, end=end)
    summary = trader.simulate(streamer, delay_seconds=0)

    print("\nReplay summary:")
    for trade in summary.trades:
        print(
            f"{trade.status:12s} | entry={trade.entry_time.date()} @ {trade.entry_price:,.2f}"
            f" -> exit={trade.exit_time.date() if trade.exit_time else '-'} @ {trade.exit_price:,.2f}"
            f" | pnl={trade.profit_pct:+.2%} | bars={trade.bars_held}"
        )
    print(f"Total PnL across trades: {summary.cumulative_profit_pct:+.2%}")

if __name__ == "__main__":
    demo_replay()