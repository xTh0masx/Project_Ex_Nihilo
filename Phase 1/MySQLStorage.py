"""MySQL based :class:`Storage` implementation used by the prototype.

The abstract :class:`~Phase 1.Storage.Storage` contract requires a concrete
backend to persist OHLCV data, executed trades and serialised models.  This
module wires the interface to a MySQL database using the lightweight connection
pooling utilities provided by :mod:`mysql.connector`.  The implementation keeps
runtime dependencies optional: when the MySQL driver is not available an
``ImportError`` is raised with a helpful message.

The class expects a :class:`~Phase 1.Configuration.Config` instance (or a plain
mapping) containing the usual connection credentials under the
``storage.mysql`` namespace::

    {
        "storage": {
            "mysql": {
                "host": "localhost",
                "port": 3306,
                "user": "root",
                "password": "secret",
                "database": "trading"
            }
        }
    }

The schema helper :func:`create_schema` can be used during development to
create the required tables.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Mapping, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import mysql.connector
    from mysql.connector import pooling
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "mysql_storage requires the 'mysql-connector-python' package. Install it to enable MySQL support."
    ) from exc

try:  # Optional dependency for DataFrame interop
    import pandas as pd
except Exception:  # pragma: no cover - pandas is optional
    pd = None  # type: ignore

from .Configuration import Config
from .Storage import Storage, TradeRecord


class MySQLStorage(Storage):
    """Persist trading artefacts inside a MySQL database."""

    def __init__(self, config: Config | Mapping[str, Any], pool_name: str = "phase1_pool", pool_size: int = 5) -> None:
        if not isinstance(config, Config):
            config = Config(config)  # type: ignore[arg-type]

        self._config = config
        settings = self._build_connection_settings()
        settings.setdefault("pool_name", pool_name)
        settings.setdefault("pool_size", pool_size)
        settings.setdefault("charset", "utf8mb4")
        settings.setdefault("collation", "utf8mb4_unicode_ci")
        settings.setdefault("autocommit", False)

        self._pool = pooling.MySQLConnectionPool(**settings)

    # -----------------------------------------------------------------------------------------
    def _build_connection_settings(self) -> Mapping[str, Any]:
        """Read connection options from the configuration object."""

        prefix = "storage.mysql"
        get = self._config.get

        host = get(f"{prefix}.host", "localhost")
        port = int(get(f"{prefix}.port", 3306) or 3306)
        user = get(f"{prefix}.user")
        password = get(f"{prefix}.password")
        database = get(f"{prefix}.database")

        if not user:
            raise ValueError("MySQL configuration requires 'storage.mysql.user'.")
        if database is None:
            raise ValueError("MySQL configuration requires 'storage.mysql.database'.")

        return {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database,
        }

    # -----------------------------------------------------------------------------------------
    @contextmanager
    def _connection(self) -> Iterator[mysql.connector.connection_cext.CMySQLConnection]:
        """Context manager returning a pooled MySQL connection."""

        conn = self._pool.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # -----------------------------------------------------------------------------------------
    @staticmethod
    def _prepare_ohlcv_rows(
        symbol: str, timeframe: str, data: Any
    ) -> Sequence[Tuple[str, str, int, float, float, float, float, float]]:
        """Normalise *data* into a sequence of value tuples ready for insertion."""

        if pd is not None and isinstance(data, pd.DataFrame):
            required_columns = {"timestamp", "open", "high", "low", "close", "volume"}
            missing = required_columns.difference(data.columns)
            if missing:
                raise ValueError(f"DataFrame is missing required OHLCV columns: {sorted(missing)}")

            iterator: Iterable[Tuple[int, float, float, float, float, float]] = (
                (
                    int(row.timestamp),
                    float(row.open),
                    float(row.high),
                    float(row.low),
                    float(row.close),
                    float(row.volume),
                )
                for row in data.sort_values("timestamp").itertuples(index=False)
            )
        else:
            iterator = []
            for item in data or []:
                if isinstance(item, Mapping):
                    timestamp = int(item["timestamp"])
                    values = (
                        timestamp,
                        float(item["open"]),
                        float(item["high"]),
                        float(item["low"]),
                        float(item["close"]),
                        float(item["volume"]),
                    )
                else:
                    # Expecting tuple/list in order (timestamp, open, high, low, close, volume)
                    timestamp = int(item[0])
                    values = (
                        timestamp,
                        float(item[1]),
                        float(item[2]),
                        float(item[3]),
                        float(item[4]),
                        float(item[5]),
                    )
                iterator.append(values)

        rows = [
            (symbol, timeframe, ts, op, hi, lo, cl, vol)
            for ts, op, hi, lo, cl, vol in iterator
        ]
        return rows

    # -----------------------------------------------------------------------------------------
    def save_ohlcv(self, symbol: str, timeframe: str, data: Any) -> None:
        rows = self._prepare_ohlcv_rows(symbol, timeframe, data)
        if not rows:
            return

        query = (
            "INSERT INTO ohlcv (symbol, timeframe, timestamp, open, high, low, close, volume) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
            "ON DUPLICATE KEY UPDATE "
            "open = VALUES(open), high = VALUES(high), low = VALUES(low), "
            "close = VALUES(close), volume = VALUES(volume)"
        )

        with self._connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.executemany(query, rows)
            finally:
                cursor.close()

    # -----------------------------------------------------------------------------------------
    def load_ohlcv(self, symbol: str, timeframe: str) -> Any:
        query = (
            "SELECT timestamp, open, high, low, close, volume "
            "FROM ohlcv WHERE symbol = %s AND timeframe = %s ORDER BY timestamp ASC"
        )

        with self._connection() as conn:
            cursor = conn.cursor(dictionary=True)
            try:
                cursor.execute(query, (symbol, timeframe))
                rows = cursor.fetchall()
            finally:
                cursor.close()

        if pd is not None:
            return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])

        return rows

    # -----------------------------------------------------------------------------------------
    def save_trade(self, trade: TradeRecord) -> None:
        payload = json.dumps(
            {
                "symbol": trade.symbol,
                "side": trade.side,
                "quantity": float(trade.quantity),
                "price": float(trade.price),
                "timestamp": int(trade.timestamp),
            }
        )

        query = "INSERT INTO trades (symbol, trade_timestamp, payload) VALUES (%s, %s, %s)"
        params = (trade.symbol, int(trade.timestamp), payload)

        with self._connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(query, params)
            finally:
                cursor.close()

    # -----------------------------------------------------------------------------------------
    def save_model(self, tag: str, obj: Any) -> None:
        payload = json.dumps(obj)
        query = (
            "INSERT INTO models (tag, payload) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE payload = VALUES(payload), updated_at = CURRENT_TIMESTAMP"
        )

        with self._connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(query, (tag, payload))
            finally:
                cursor.close()

    # -----------------------------------------------------------------------------------------
    def load_model(self, tag: str) -> Any:
        query = "SELECT payload FROM models WHERE tag = %s"

        with self._connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(query, (tag,))
                row = cursor.fetchone()
            finally:
                cursor.close()

        if not row:
            raise KeyError(f"Model with tag '{tag}' not found")

        payload = row[0]
        return json.loads(payload)

    # -----------------------------------------------------------------------------------------
    def create_schema(self) -> None:
        """Create database tables required by the storage backend."""

        statements = [
            """
            CREATE TABLE IF NOT EXISTS ohlcv (
                symbol VARCHAR(32) NOT NULL,
                timeframe VARCHAR(16) NOT NULL,
                timestamp BIGINT NOT NULL,
                open DOUBLE NOT NULL,
                high DOUBLE NOT NULL,
                low DOUBLE NOT NULL,
                close DOUBLE NOT NULL,
                volume DOUBLE NOT NULL,
                PRIMARY KEY (symbol, timeframe, timestamp)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """,
            """
            CREATE TABLE IF NOT EXISTS trades (
                id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
                symbol VARCHAR(32) NOT NULL,
                trade_timestamp BIGINT NOT NULL,
                payload LONGTEXT NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (id),
                INDEX idx_trades_symbol_ts (symbol, trade_timestamp)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """,
            """
            CREATE TABLE IF NOT EXISTS models (
                tag VARCHAR(64) NOT NULL,
                payload LONGTEXT NOT NULL,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                PRIMARY KEY (tag)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
            """,
        ]

        with self._connection() as conn:
            cursor = conn.cursor()
            try:
                for statement in statements:
                    cursor.execute(statement)
            finally:
                cursor.close()