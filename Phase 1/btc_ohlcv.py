import yfinance as yf
btcusd  = yf.Ticker("BTC-USD")
data = btcusd.history(interval="1h", period="max")
print(data.to_string())