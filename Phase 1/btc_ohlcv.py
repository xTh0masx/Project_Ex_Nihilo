import yfinance as yf
import mysql
import mysql.connector
from mysql.connector import Error

# MySQL connection
try:
    connection = mysql.connector.connect(
        host="127.0.0.1",
        port=3306,
        user="root",
        password="Digimon@4123",
        database="yahoo_finance_extraction"
    )

    if connection.is_connected():
        print("Connected to MySQL")

    cursor = connection.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS EX_NIHILO (
            quote_date DATE PRIMARY KEY,
            open DECIMAL (18,8),
            high DECIMAL (18,8),
            low DECIMAL (18,8),
            close DECIMAL (18,8),
            volume BIGINT,
        )
        """
    )

    btcusd = yf.Ticker("BTC-USD")
    data = btcusd.history(period="max", interval="1d")

    insert_stm = """
        INSERT INTO EX_NIHILO (quote_date, open, high, low, close, volume)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            open=VALUES(open),
            high=VALUES(high),
            low=VALUES(low),
            close=VALUES(close),
            volume=VALUES(volume)
        """

    rows = [
        (
            idx.date(),
            float(row["Open"]),
            float(row["High"]),
            float(row["Low"]),
            float(row["Close"]),
            float(row["Volume"])
        )
        for idx, row in data.iterrows()
    ]

    cursor.executemany(insert_stm, rows)
    connection.commit()
    print(f"{cursor.rowcount} rows inserted.")

finally:
    if cursor:
        cursor.close()
    if connection and connection.is_connected():
        connection.close()

# Data Extraction from Yahoo Finance
#btcusd  = yf.Ticker("BTC-USD")
#data = btcusd.history(interval="1d", period="max")
#print(data.to_string())

