import pandas as pd
from src.strategy import signal_sma_crossover

def test_signal_columns_exist():
    df = pd.DataFrame({"Close": [1,2,3,4,5]})
    out = signal_sma_crossover(df, short=2, long=3)
    assert set(["smaS","smaL","signal"]).issubset(out.columns)
