import time
import pandas as pd
from .fetch import fetch_btc
from .strategy import signal_sma_crossover
from .explain import explain_row
from .db import engine

def run_paper(interval_sec=300, short=20, long=50):
    while True:
        df = fetch_btc(interval="1h", lookback_days=60)
        df = signal_sma_crossover(df, short=short, long=long)
        last = df.iloc[-1]
        reason = explain_row(last, short=short, long=long)

        with engine.begin() as conn:
            conn.execute(
                "INSERT INTO signals (ts, symbol, signal, reason, s1_value, s2_value, price) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                [pd.Timestamp.utcnow(), "BTC-USD", last["signal"], reason,
                 float(last["smaS"]), float(last["smaL"]), float(last["Close"])]
            )
        print(f"{pd.Timestamp.utcnow()} → {last['signal']} @ {last['Close']:.2f} | {reason}")
        time.sleep(interval_sec)

