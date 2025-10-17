"""Fetch BTC-USD OHLCV data from Yahoo Finance and print it."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, List
from urllib.parse import urlencode
from urllib.request import urlopen


@dataclass
class Candle:
    """Represents a single OHLCV candle."""

    date: datetime
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: int

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "Candle":
        return cls(
            date=datetime.strptime(row["Date"], "%Y-%m-%d"),
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(row["Close"]),
            adj_close=float(row["Adj Close"]),
            volume=int(float(row["Volume"])),
        )


@dataclass
class YahooFinanceClient:
    ticker: str

    def build_url(self, start: datetime, end: datetime, interval: str = "1d") -> str:
        # Yahoo Finance expects timestamps in seconds since epoch in UTC and the end
        # timestamp is exclusive, so we add one day to include the final date.
        start_utc = start.replace(tzinfo=timezone.utc)
        end_utc = end.replace(tzinfo=timezone.utc) + timedelta(days=1)

        params = {
            "period1": int(start_utc.timestamp()),
            "period2": int(end_utc.timestamp()),
            "interval": interval,
            "events": "history",
            "includeAdjustedClose": "true",
        }
        return f"https://query1.finance.yahoo.com/v7/finance/download/{self.ticker}?{urlencode(params)}"

    def fetch(self, start: datetime, end: datetime, interval: str = "1d") -> Iterable[Candle]:
        url = self.build_url(start, end, interval)
        with urlopen(url) as response:
            decoded = response.read().decode("utf-8")

        reader = csv.DictReader(decoded.splitlines())
        candles: List[Candle] = []
        for row in reader:
            if "null" in row.values():
                # Skip rows with missing data
                continue
            candles.append(Candle.from_row(row))

        if not candles:
            raise RuntimeError("No data returned from Yahoo Finance.")

        return candles


def main() -> None:
    start = datetime(2020, 1, 1)
    end = datetime(2020, 12, 31)
    client = YahooFinanceClient(ticker="BTC-USD")

    candles = client.fetch(start=start, end=end, interval="1d")

    print("Date        Open        High        Low         Close       Adj Close   Volume")
    for candle in candles:
        print(
            f"{candle.date:%Y-%m-%d}  "
            f"{candle.open:10.2f}  "
            f"{candle.high:10.2f}  "
            f"{candle.low:10.2f}  "
            f"{candle.close:10.2f}  "
            f"{candle.adj_close:10.2f}  "
            f"{candle.volume:10d}"
        )


if __name__ == "__main__":
    main()