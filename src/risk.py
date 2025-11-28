def size_position(account_value, entry, sl, risk_pct=0.01):
    risk_amt = account_value * risk_pct
    per_unit_risk = max(entry - sl, 1e-9)
    qty = risk_amt / per_unit_risk
    return max(qty, 0)
