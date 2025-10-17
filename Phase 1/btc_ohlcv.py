import yfinance as yf
import mysql.connector
from mysql.connector import Error

# MySQL connection
try:
    connection = mysql.connector.connect(user='root',
                                         password='Digimon@4123',
                                         host='localhost',
                                         database='yahoo_finance_extraction')

    if connection.is_connected():
        print("Connected to MySQL")
except Error as e:
    print("Error while connecting to MySQL", e)

cursor = connection.cursor()

# Data Extraction from Yahoo Finance
btcusd  = yf.Ticker("BTC-USD")
data = btcusd.history(interval="1d", period="max")
print(data.to_string())

connection.commit()
cursor.close()
connection.close()
