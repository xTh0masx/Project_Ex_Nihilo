from src.db import engine

with engine.connect() as conn:
    r = conn.execute("SELECT 1").fetchone()
    print("DB Connection OK:", r)


