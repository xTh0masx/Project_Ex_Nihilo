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
    )

    if connection.is_connected():
        print("Connected to MySQL")

    cursor = connection.cursor()
    cursor.execute("CREATE TABLE test (id int(3), text varchar(255), datetime datetime)")
    cursor.close()

    cursor = connection.cursor()
    cursor.execute("INSERT INTO test (id, text, datetime) VALUES (%s, %s, %s)")
    cursor.close()
    connection.commit()

    cursor = connection.cursor()
    cursor.execute("SELECT * FROM test")
    result = cursor.fetchall()
    cursor.close()
    for data in result:
        print("Nummer: " + str(data[0]) + "; Text: " + (data[1]))

except Error as e:
    print("Error while connecting to MySQL", e)

# Data Extraction from Yahoo Finance
#btcusd  = yf.Ticker("BTC-USD")
#data = btcusd.history(interval="1d", period="max")
#print(data.to_string())


connection.close()
