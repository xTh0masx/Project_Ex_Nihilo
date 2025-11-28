def explain_row(row, short=20, long=50):
    sS, sL, p = row["smaS"], row["smaL"], row["Close"]
    if row["signal"] == "BUY":
        return (f"Kaufsignal: SMA{short} ({sS:.2f}) kreuzt "
                f"SMA{long} ({sL:.2f}) von unten. Preis={p:.2f}.")
    if row["signal"] == "SELL":
        return (f"Verkaufssignal: SMA{short} ({sS:.2f}) kreuzt "
                f"SMA{long} ({sL:.2f}) von oben. Preis={p:.2f}.")
    return (f"Kein Trade: SMA{short}={sS:.2f}, SMA{long}={sL:.2f}; "
            f"kein Kreuz. Preis={p:.2f}.")

