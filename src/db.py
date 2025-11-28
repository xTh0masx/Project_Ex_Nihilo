import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL  # <- wichtig

load_dotenv()

DB_USER = os.getenv("root")
DB_PASS = os.getenv("Digimon@4123")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "3307"))
DB_NAME = os.getenv("trading")

# Debug-Ausgabe (kannst du später löschen)
print("DB_HOST =", DB_HOST)
print("DB_PORT =", DB_PORT)
print("DB_USER =", DB_USER)
print("DB_NAME =", DB_NAME)

url = URL.create(
    drivername="mysql+pymysql",
    username=DB_USER,
    password=DB_PASS,   # URL.create kümmert sich um @, !, : usw.
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME,
    query={"charset": "utf8mb4"},
)

engine = create_engine(
    url,
    pool_recycle=3600,
    pool_pre_ping=True,
)

