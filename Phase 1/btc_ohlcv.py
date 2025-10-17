import yfinance as yf
btcusd  = yf.Ticker("BTC-USD")
data = btcusd.history(period="max")
print(data.to_string())