import yfinance as yf
import mysql.connector
from mysql.connector import Error


def to_python_datetime(timestamp):
    dt = timestamp.to_pydatetime()
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    return dt


def store_history(cursor, connection, table_name, time_column, data, to_time):
    insert_statement = f"""
        INSERT INTO {table_name} ({time_column}, open, high, low, close, volume)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            open = VALUES(open),
            high = VALUES(high),
            low = VALUES(low),
            close = VALUES(close),
            volume = VALUES(volume)
    """

    rows = [
        (
            to_time(idx),
            float(row["Open"]),
            float(row["High"]),
            float(row["Low"]),
            float(row["Close"]),
            float(row["Volume"])
        )
        for idx, row in data.iterrows()
    ]

    cursor.executemany(insert_statement, rows)
    connection.commit()
    print(f"{cursor.rowcount} rows inserted into {table_name}.")



# MySQL connection
connection = None
cursor = None

try:
    connection = mysql.connector.connect(
        host='127.0.0.1',
        port=3306,
        user='root',
        password='Digimon@4123',
        database='ex_nihilo',
        auth_plugin='mysql_native_password'
    )

    if connection.is_connected():
        print("Connected to MySQL")

    cursor = connection.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS yahoo_finance_data (
            quote_date DATE PRIMARY KEY,
            open DECIMAL (18, 8),
            high DECIMAL (18, 8),
            low DECIMAL (18, 8),
            close DECIMAL (18, 8),
            volume BIGINT
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS yahoo_finance_data_hourly (
            quote_datetime DATETIME PRIMARY KEY,
            open DECIMAL (18, 8),
            high DECIMAL (18, 8),
            low DECIMAL (18, 8),
            close DECIMAL (18, 8),
            volume BIGINT
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS yahoo_finance_data_minute (
            quote_datetime DATETIME PRIMARY KEY,
            open DECIMAL (18, 8),
            high DECIMAL (18, 8),
            low DECIMAL (18, 8),
            close DECIMAL (18, 8),
            volume BIGINT
        )
        """
    )

    btcusd = yf.Ticker("BTC-USD")
    daily_data = btcusd.history(period="max", interval="1d")
    store_history(
        cursor,
        connection,
        "yahoo_finance_data",
        "quote_date",
        daily_data,
        lambda idx: to_python_datetime(idx).date()
    )

    hourly_data = btcusd.history(period="max", interval="1h")
    store_history(
        cursor,
        connection,
        "yahoo_finance_data_hourly",
        "quote_datetime",
        hourly_data,
        lambda idx: to_python_datetime(idx)
    )

    minute_data = btcusd.history(period="max", interval="1m")
    store_history(
        cursor,
        connection,
        "yahoo_finance_data_minute",
        "quote_datetime",
        minute_data,
        lambda idx: to_python_datetime(idx)
    )

    # print(data.to_string())

#    insert_stm = """
    #        INSERT INTO yahoo_finance_data (quote_date, open, high, low, close, volume)
    #   VALUES (%s, %s, %s, %s, %s, %s)
    #   ON DUPLICATE KEY UPDATE
    #       open=VALUES(open),
    #       high=VALUES(high),
    ##       low=VALUES(low),
    #      close=VALUES(close),
    #       volume=VALUES(volume)
    #   """

#    rows = [
    #    (
     #       idx.date(),
     #       float(row["Open"]),
     #       float(row["High"]),
     #       float(row["Low"]),
     #       float(row["Close"]),
     #       float(row["Volume"])
     #   )
     #   for idx, row in data.iterrows()
#    ]

#    cursor.executemany(insert_stm, rows)
#    connection.commit()
#    print(f"{cursor.rowcount} rows inserted.")

except Error as error:
    print(f"Failed to fetch/store BTC OHLCV data: \n{error}")

finally:
    if cursor:
        cursor.close()
    if connection and connection.is_connected():
        connection.close()
