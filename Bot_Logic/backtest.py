"""Minimal backtest harness for the spot trading bot."""
from __future__ import annotations

from trade import TradeBot


def demo_run() -> None:
    """Run the bot against a tiny synthetic price series.

    The price path is crafted to trigger the 1% take-profit condition. Adjust
    the list to explore different outcomes; e.g., insert a sharp drop to
    observe the 10% stop-loss exit.
    """

    # Start at 100.00, drift upward past the 1% profit target.
    prices = [100.0, 100.3, 100.7, 101.5, 101.2]

    bot = TradeBot()
    result = bot.run(prices)

    print("\nFinal result:")
    print(result)


if __name__ == "__main__":
    demo_run()