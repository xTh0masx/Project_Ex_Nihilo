import yfinance as yf
btcusd  = yf.Ticker("BTC-USD")
data = btcusd.history(interval="1m", period="max")
print(data.to_string())