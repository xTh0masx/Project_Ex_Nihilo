"""Streamlit dashboard for exploring BTC OHLCV datasets and trades."""
#to run streamlit copypaste -> streamlit run "Phase 1\\bitcoin_dashboard.py"

from __future__ import annotations

from datetime import datetime, timedelta
import threading
import time
from pathlib import Path
import os
import sys
from typing import Dict, Iterable, List, Tuple
from urllib.parse import quote_plus

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Databank.btc_ohlcv import update_all_intervals
from Bot_Logic.strategy import Action, SpotProfitStopStrategy
from Bot_Logic.replay_simulator import HistoricalPriceStreamer, NeuralReplayTrader
from Logic.nn_inference import NeuralPricePredictor, create_minimal_demo_model

def _env(key: str, default: str | int) -> str | int:
    """Read a database configuration value from the environment."""

    return os.getenv(key.upper(), default)


DB_CONFIG: Dict[str, str | int] = {
    "host": _env("MYSQL_HOST", "127.0.0.1"),
    "port": int(_env("MYSQL_PORT", 3306)),
    "user": _env("MYSQL_USER", "root"),
    "password": _env("MYSQL_PASSWORD", "Digimon@4123"),
    "database": _env("MYSQL_DATABASE", "ex_nihilo"),
    "auth_plugin": "mysql_native_password",
}

TRADE_TABLE = os.getenv("TRADE_TABLE", "bot_trades")


_ENGINE: Engine | None = None


def _engine() -> Engine:
    """Create (or reuse) a SQLAlchemy engine for pandas queries."""

    global _ENGINE
    if _ENGINE is None:
        user = quote_plus(str(DB_CONFIG["user"]))
        password = quote_plus(str(DB_CONFIG["password"]))
        host = DB_CONFIG["host"]
        port = DB_CONFIG["port"]
        database = DB_CONFIG["database"]
        auth_plugin = DB_CONFIG.get("auth_plugin")
        query = f"?auth_plugin={quote_plus(str(auth_plugin))}" if auth_plugin else ""
        url = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}{query}"
        _ENGINE = create_engine(url, pool_pre_ping=True)
    return _ENGINE


def _table_columns(table: str) -> List[str]:
    """Fetch the ordered column names for a given table."""

    with _engine().connect() as connection:
        rows = connection.execute(
            text(
                """
                SELECT COLUMN_NAME
                FROM information_schema.columns
                WHERE table_schema = :schema AND table_name = :table
                ORDER BY ORDINAL_POSITION
                """
            ),
            {"schema": DB_CONFIG["database"], "table": table},
        ).fetchall()
    return [row[0] for row in rows]


def _resolve_time_column(table: str, preferred: str) -> str:
    """Ensure the timestamp column exists, falling back to similar names."""

    columns = _table_columns(table)
    if not columns:
        raise RuntimeError(f"Table '{table}' does not exist in schema {DB_CONFIG['database']}")
    if preferred in columns:
        return preferred

    for candidate in ("quote_datetime", "quote_date", "timestamp", "quote_time"):
        if candidate in columns:
            return candidate

    raise RuntimeError(
        f"None of the expected timestamp columns ({preferred}) exist on table '{table}'."
    )

class TradingBotRunner:
    """Background trading loop that stays active while the dashboard is open."""

    def __init__(
        self,
        engine: Engine,
        *,
        price_table: str,
        time_column: str,
        poll_interval_seconds: int = 60,
    ) -> None:
        self.engine = engine
        self.price_table = price_table
        self.time_column = time_column
        self.poll_interval_seconds = poll_interval_seconds
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.strategy = SpotProfitStopStrategy()
        self.last_timestamp: pd.Timestamp | None = None
        self.entry_price: float | None = None
        self.trade_count = 0
        self.last_error: str | None = None
        self.last_loop_started_at: datetime | None = None

    def is_running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self) -> None:
        if self.is_running():
            return

        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if not self.is_running():
            return

        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1)

    def _run_loop(self) -> None:
        try:
            self._ensure_trade_table()
        except Exception as exc:  # pragma: no cover - defensive guard
            self.last_error = str(exc)
            return

        while not self.stop_event.is_set():
            self.last_loop_started_at = datetime.utcnow()
            try:
                update_all_intervals()
                latest = self._fetch_latest_price()
                if latest is not None:
                    self._maybe_trade(latest)
                    self.last_error = None
            except Exception as exc:  # pragma: no cover - runtime safety
                self.last_error = str(exc)
            finally:
                time.sleep(self.poll_interval_seconds)

    def _ensure_trade_table(self) -> None:
        if not TRADE_TABLE:
            raise RuntimeError("TRADE_TABLE environment variable must be set for live trading")

        ddl = f"""
            CREATE TABLE IF NOT EXISTS `{TRADE_TABLE}` (
                id INT AUTO_INCREMENT PRIMARY KEY,
                timestamp DATETIME NOT NULL,
                side VARCHAR(10) NOT NULL,
                price DECIMAL(18, 8) NOT NULL,
                entry_price DECIMAL(18, 8) NOT NULL,
                exit_price DECIMAL(18, 8) NOT NULL,
                pnl DECIMAL(18, 8) NOT NULL,
                strategy VARCHAR(64) NOT NULL
            )
        """
        with self.engine.begin() as connection:
            connection.execute(text(ddl))

    def _fetch_latest_price(self) -> Tuple[pd.Timestamp, float] | None:
        query = text(
            f"""
            SELECT `{self.time_column}` AS ts, close
            FROM `{self.price_table}`
            ORDER BY `{self.time_column}` DESC
            LIMIT 1
            """
        )
        with self.engine.connect() as connection:
            row = connection.execute(query).mappings().first()
            if not row:
                return None

        timestamp = pd.to_datetime(row["ts"])
        price = float(row["close"])
        return timestamp, price

    def _maybe_trade(self, latest: Tuple[pd.Timestamp, float]) -> None:
        timestamp, price = latest

        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            return

        self.last_timestamp = timestamp

        if self.entry_price is None:
            self.entry_price = price
            return

        action = self.strategy.evaluate(self.entry_price, price)
        if action is Action.HOLD:
            return

        side = "SELL" if action is Action.TAKE_PROFIT else "STOP"
        pnl = price - self.entry_price
        self._record_trade(timestamp, side, price, self.entry_price, price, pnl)
        self.entry_price = price
        self.trade_count += 1

    def _record_trade(
        self,
        timestamp: pd.Timestamp,
        side: str,
        price: float,
        entry_price: float,
        exit_price: float,
        pnl: float,
    ) -> None:
        insert = text(
            f"""
            INSERT INTO `{TRADE_TABLE}` (timestamp, side, price, entry_price, exit_price, pnl, strategy)
            VALUES (:timestamp, :side, :price, :entry_price, :exit_price, :pnl, :strategy)
            """
        )

        with self.engine.begin() as connection:
            connection.execute(
                insert,
                {
                    "timestamp": timestamp.to_pydatetime(),
                    "side": side,
                    "price": price,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "strategy": "spot_profit_stop",
                },
            )

        load_trades.clear()


DATASETS = {
    "Minute": {
        "table": "yahoo_finance_data_minute",
        "time_column": "quote_datetime",
    },
    "Hourly": {
        "table": "yahoo_finance_data_hourly",
        "time_column": "quote_datetime",
    },
    "Daily": {
        "table": "yahoo_finance_data",
        "time_column": "quote_date",
    },
}


def _query_dataframe(query: str, *, parse_dates: Iterable[str] | None = None) -> pd.DataFrame:
    """Execute a SQL query and return the resulting dataframe."""

    with _engine().connect() as connection:
        return pd.read_sql_query(text(query), con=connection, parse_dates=list(parse_dates or []))


@st.cache_data(ttl=60)
def load_ohlcv(granularity: str) -> pd.DataFrame:
    """Load OHLCV rows for the selected granularity into a dataframe."""

    config = DATASETS[granularity]
    time_column = _resolve_time_column(config["table"], config["time_column"])
    query = (
        f"SELECT `{time_column}` AS timestamp, open, high, low, close, volume "
        f"FROM `{config['table']}` ORDER BY `{time_column}`"
    )
    frame = _query_dataframe(query, parse_dates=["timestamp"])
    frame.set_index("timestamp", inplace=True)
    frame.sort_index(inplace=True)
    numeric_cols = ["open", "high", "low", "close", "volume"]
    frame[numeric_cols] = frame[numeric_cols].apply(pd.to_numeric, errors="coerce")
    frame.dropna(subset=numeric_cols, inplace=True)
    return frame


def _detect_trade_columns(connection, table: str) -> List[str]:
    rows = connection.execute(
        text(
            """
            SELECT COLUMN_NAME
            FROM information_schema.columns
            WHERE table_schema = :schema AND table_name = :table
            ORDER BY ORDINAL_POSITION
            """
        ),
        {"schema": DB_CONFIG["database"], "table": table},
    ).fetchall()
    return [row[0] for row in rows]


@st.cache_data(ttl=30)
def load_trades() -> pd.DataFrame:
    """Load trade executions from the configured table if it exists."""

    if not TRADE_TABLE:
        return pd.DataFrame()

    try:
        connection = _engine().connect()
    except SQLAlchemyError:
        return pd.DataFrame()

    with connection:
        exists = connection.execute(
            text(
                """
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_schema = :schema AND table_name = :table
                """
            ),
            {"schema": DB_CONFIG["database"], "table": TRADE_TABLE},
        ).scalar()
        if not exists:
            return pd.DataFrame()

        available_columns = _detect_trade_columns(connection, TRADE_TABLE)
        timestamp_column = next(
            (col for col in ("executed_at", "created_at", "timestamp") if col in available_columns),
            None,
        )
        if not timestamp_column:
            return pd.DataFrame()

        select_columns = [timestamp_column]
        preferred = [
            "side",
            "direction",
            "price",
            "execution_price",
            "quantity",
            "qty",
            "size",
            "pnl",
            "pnl_usd",
            "strategy",
        ]
        for column in preferred:
            if column in available_columns and column not in select_columns:
                select_columns.append(column)

        query = (
            f"SELECT {', '.join(f'`{col}`' for col in select_columns)} FROM `{TRADE_TABLE}` "
            f"ORDER BY `{timestamp_column}`"
        )
        try:
            trades = pd.read_sql_query(text(query), con=connection, parse_dates=[timestamp_column])
        except SQLAlchemyError:
            return pd.DataFrame()

    trades.rename(columns={timestamp_column: "timestamp"}, inplace=True)
    if "direction" in trades.columns and "side" not in trades.columns:
        trades.rename(columns={"direction": "side"}, inplace=True)
    if "execution_price" in trades.columns and "price" not in trades.columns:
        trades.rename(columns={"execution_price": "price"}, inplace=True)
    if "qty" in trades.columns and "quantity" not in trades.columns:
        trades.rename(columns={"qty": "quantity"}, inplace=True)
    if "pnl_usd" in trades.columns and "pnl" not in trades.columns:
        trades.rename(columns={"pnl_usd": "pnl"}, inplace=True)

    trades.sort_values("timestamp", inplace=True)
    return trades

def _update_databank() -> None:
    """Pull fresh BTC OHLCV data into the MySQL databank using btc_ohlcv."""

    try:
        update_all_intervals()
        st.session_state["databank_last_updated_at"] = datetime.now()
    except Exception as exc:  # pragma: no cover - runtime protection
        st.session_state["databank_last_updated_error"] = str(exc)
        st.warning(f"Failed to update databank automatically: {exc}")


def _auto_refresh(interval_seconds: int = 60) -> None:
    """Periodically refresh cached data to keep charts up to date."""

    now = datetime.now()
    next_refresh_at = st.session_state.get("data_autorefresh_next_at")

    if next_refresh_at is None or now >= next_refresh_at:
        _update_databank()
        load_ohlcv.clear()
        load_trades.clear()
        st.session_state["data_autorefresh_at"] = now
        st.session_state["data_autorefresh_next_at"] = now + timedelta(seconds=interval_seconds)

    interval_ms = interval_seconds * 1000
    st.markdown(
        f"<script>setTimeout(() => window.location.reload(), {interval_ms});</script>",
        unsafe_allow_html=True,
    )

    refreshed_at = st.session_state.get("data_autorefresh_at")
    last_message = refreshed_at.strftime("%Y-%m-%d %H:%M:%S UTC") if refreshed_at else "pending"
    databank_at = st.session_state.get("databank_last_updated_at")
    databank_message = databank_at.strftime("%Y-%m-%d %H:%M:%S UTC") if databank_at else "pending"
    st.caption(
        "Auto-refresh enabled: databank updates via btc_ohlcv and cache clears "
        f"every {interval_seconds} seconds (next refresh scheduled at "
        f"{st.session_state.get('data_autorefresh_next_at', 'pending')}, last databank update "
        f"{databank_message}, last refresh {last_message})."
    )


def _simulate_bot_trades(
    frame: pd.DataFrame,
    *,
    starting_capital: float = 2000.0,
    max_trade_usd: float = 1000.0,
    max_trades: int = 1000,
) -> pd.DataFrame:
    """Run a simple strategy until profitability or limits are reached."""

    closes = frame["close"].copy()
    fast = closes.rolling(window=5, min_periods=5).mean()
    slow = closes.rolling(window=20, min_periods=20).mean()

    capital = starting_capital
    trades: List[Dict[str, float | str | pd.Timestamp]] = []
    previous_signal = None

    for idx in range(len(closes) - 1):
        if len(trades) >= max_trades or capital <= 0:
            break

        signal = "hold"
        if not pd.isna(fast.iloc[idx]) and not pd.isna(slow.iloc[idx]):
            signal = "buy" if fast.iloc[idx] > slow.iloc[idx] else "sell"

        if signal == "hold" or signal == previous_signal:
            continue

        entry_price = closes.iloc[idx]
        exit_price = closes.iloc[idx + 1]
        trade_size = min(max_trade_usd, capital)
        quantity = trade_size / entry_price

        if signal == "buy":
            pnl = quantity * (exit_price - entry_price)
        else:
            pnl = quantity * (entry_price - exit_price)

        capital += pnl
        trades.append(
            {
                "timestamp": closes.index[idx + 1],
                "side": signal,
                "quantity": quantity,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "remaining_capital": capital,
            }
        )
        previous_signal = signal

        if capital - starting_capital > 0:
            break

    return pd.DataFrame(trades)

def _replay_trades_to_frame(trades: List) -> pd.DataFrame:
    """Convert replay trades into a DataFrame for display/plotting."""

    if not trades:
        return pd.DataFrame()

    rows = []
    for trade in trades:
        profit_pct = trade.profit_pct or 0.0
        exit_price = trade.exit_price if trade.exit_price is not None else trade.entry_price
        pnl_usd = (
            trade.profit_usd
            if trade.profit_usd is not None
            else profit_pct * getattr(trade, "capital_used", trade.entry_price)
        )
        rows.append(
            {
                "timestamp": trade.exit_time or trade.entry_time,
                "entry_time": trade.entry_time,
                "exit_time": trade.exit_time,
                "side": "buy",
                "quantity": getattr(trade, "quantity", 1.0),
                "capital_used": getattr(trade, "capital_used", trade.entry_price),
                "entry_price": trade.entry_price,
                "exit_price": exit_price,
                "status": trade.status,
                "pnl_pct": profit_pct * 100,
                "pnl_usd": pnl_usd,
                "bars": trade.bars_held,
                "model_prediction": trade.model_prediction,
            }
        )
    frame = pd.DataFrame(rows)
    frame.sort_values("timestamp", inplace=True)
    return frame

def _build_trade_ledger(trades: pd.DataFrame) -> pd.DataFrame:
    """Create a tab-friendly ledger with capital and sizing details."""

    if trades.empty:
        return trades

    ledger = trades.copy()
    ledger["timestamp"] = ledger["exit_time"].fillna(ledger["entry_time"])
    ledger["units"] = ledger.get("quantity", 1.0)
    ledger["used_capital"] = ledger.get("capital_used", ledger["entry_price"])
    ledger["pnl_usd"] = ledger.get(
        "pnl_usd", ledger.get("pnl_pct", 0.0) / 100 * ledger["used_capital"]
    )

    columns = [
        "timestamp",
        "side",
        "units",
        "entry_price",
        "exit_price",
        "pnl_usd",
        "used_capital",
    ]

    return ledger[columns]


def _render_replay_tables(table_placeholder, trades: pd.DataFrame) -> None:
    """Render both the raw trade feed and the capital-aware ledger."""

    with table_placeholder.container():
        if trades.empty:
            st.info("No trades were opened during this replay window.")
            return

        ledger_frame = _build_trade_ledger(trades)
        trades_tab, ledger_tab = st.tabs(["Trades", "Trade ledger"])
        trades_tab.dataframe(trades.tail(200), hide_index=True, width="stretch")

        if ledger_frame.empty:
            ledger_tab.info("No closed trades recorded during this replay window.")
        else:
            ledger_tab.dataframe(ledger_frame.tail(200), hide_index=True, width="stretch")

def _split_trades_by_side(trades: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    side_series = trades.get("side")
    if side_series is None:
        empty = trades.iloc[0:0]
        return {"buy": empty, "sell": empty}

    normalized = side_series.astype(str).str.upper()
    buys = trades[normalized.isin({"BUY", "LONG"})]
    sells = trades[normalized.isin({"SELL", "SHORT"})]
    return {"buy": buys, "sell": sells}

def _get_trading_runner() -> TradingBotRunner | None:
    """Instantiate or reuse a background trading bot tied to the minute feed."""

    minute_config = DATASETS.get("Minute")
    if minute_config is None:
        return None

    try:
        time_column = _resolve_time_column(minute_config["table"], minute_config["time_column"])
    except RuntimeError as exc:
        st.warning(str(exc))
        return None

    runner: TradingBotRunner | None = st.session_state.get("live_trading_runner")
    if runner is None or runner.time_column != time_column:
        runner = TradingBotRunner(
            _engine(),
            price_table=minute_config["table"],
            time_column=time_column,
            poll_interval_seconds=60,
        )
        st.session_state["live_trading_runner"] = runner

    return runner

def render_candlestick(
    frame: pd.DataFrame,
    trades: pd.DataFrame,
    title: str,
    *,
    key: str | None = None,
    target: DeltaGenerator | None = None,
) -> None:
    """Render a candlestick chart with optional trade markers."""

    fig = go.Figure(
        data=
        [
            go.Candlestick(
                x=frame.index,
                open=frame["open"],
                high=frame["high"],
                low=frame["low"],
                close=frame["close"],
                name="BTC-USD",
            )
        ]
    )

    if not trades.empty:
        grouped = _split_trades_by_side(trades)
        for label, data in grouped.items():
            if data.empty:
                continue
            price_series = data.get("price")
            if price_series is None or price_series.isna().all():
                price_series = frame.reindex(data["timestamp"], method="nearest")["close"]
            fig.add_trace(
                go.Scatter(
                    x=data["timestamp"],
                    y=price_series,
                    mode="markers",
                    marker=dict(
                        size=10,
                        symbol="triangle-up" if label == "buy" else "triangle-down",
                        color="#16a34a" if label == "buy" else "#dc2626",
                        line=dict(width=1, color="#111"),
                    ),
                    name=f"{label.title()} trade",
                    text=data.get("strategy", data.get("quantity")),
                    hovertemplate="Timestamp: %{x}<br>Price: %{y:.2f}<extra></extra>",
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Timestamp",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=60, b=10),
    )

    plot_target = target if target is not None else st
    plot_target.plotly_chart(fig, width="stretch", key=key)

def render_neural_replay(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> None:
    """Allow users to replay a historical window with neural guidance.

    The replay alway uses the minute-level dataset so the bot only sees
    and trades candles inside the selected time window.
    Bars are revealed sequentially with optional pacing to mirror live conditions.
    """

    st.subheader("Neural network live replay")
    st.write(
        "Simulate a live ticker using past data. Only candles between the selected"
        " start and end dates are revealed to the bot. It opens long positions when"
        " the neural model predicts upside and exits via profit target, stop loss,"
        " or a defensive model-driven stop."
    )

    try:
        minute_frame = load_ohlcv("Minute")
    except SQLAlchemyError as exc:
        st.error(f"Failed to load minute data for neural replay: {exc}")
        return

    frame_slice = minute_frame.loc[start_ts:end_ts]
    if frame_slice.empty:
        st.info("No minute-level data available in the selected window.")
        return

    dafault_model_dir = PROJECT_ROOT / "models" / "btc_usd"
    model_input = st.text_input("Model directory", str(dafault_model_dir))
    model_dir = Path(model_input)
    if not model_dir.exists():
        model_dir = PROJECT_ROOT / model_dir
    entry_threshold = st.slider("Prediction to enter (return %)", 0.05, 2.0, 0.1, step=0.05) / 100
    exit_threshold = st.slider("Prediction to force-exit (return %)", 0.05, 2.0, 0.2, step=0.05) / 100
    trade_capital = st.number_input(
        "Capital per trade (USD)", value=1000.0, min_value=10.0, step=100.0
    )
    delay_seconds = st.number_input("Seconds per candle (visual replay pacing)", value=1.0, min_value=0.0, step=0.5)

    if not model_dir.exists():
        try:
            create_minimal_demo_model(model_dir)
            st.success("No trained model found; created lightweight demo checkpoint automatically.")
        except Exception as exc: # pragma: no cover - runtime safety for dashboard
            st.error("Model director not found. Train a model or point to an existing checkpoint.")
            st.caption(str(exc))
            return

    try:
        predictor = NeuralPricePredictor(model_dir)
    except Exception as exc:  # pragma: no cover - runtime safety for dashboard
        st.error(f"Failed to load neural predictor: {exc}")
        return

    streamer = HistoricalPriceStreamer(frame_slice, start=start_ts, end=end_ts)
    trader = NeuralReplayTrader(
        predictor,
        entry_threshold=entry_threshold,
        prediction_exit_threshold=exit_threshold,
        trade_capital_usd=trade_capital,
    )

    st.caption(
        "The replay animates each candle in real time. Watch the chart and table"
        " below update as the neural model opens or closes trades."
    )

    progress_bar = st.progress(0)
    status_placeholder = st.empty()
    chart_placeholder = st.empty()
    table_placeholder = st.empty()

    if st.button("Run neural replay", width="stretch"):
        run_id = st.session_state.get("neural_replay_run_id", 0) + 1
        st.session_state["neural_replay_run_id"] = run_id
        total_candles = len(frame_slice)
        render_every = max(1, total_candles // 120)

        def _on_step(idx, candle, trades, active_trade, prediction):
            completed = idx + 1
            progress_bar.progress(int((completed / total_candles) * 100))
            prediction_text = f"{(prediction * 100):+.2f}%" if prediction is not None else "pending"
            status_placeholder.info(
                f"Candle {completed}/{total_candles} @ {candle.timestamp}: "
                f"price ${candle.close:,.2f} | prediction {prediction_text}"
            )

            if (completed % render_every != 0) and (completed != total_candles):
                return

            trades_to_render = list(trades)
            if active_trade is not None:
                trades_to_render.append(active_trade)

            replay_frame_live = _replay_trades_to_frame(trades_to_render)
            render_candlestick(
                frame_slice.iloc[:completed],
                replay_frame_live,
                "Live replay progress",
                target=chart_placeholder,
                key=f"neural-replay-live-{run_id}-{completed}",
            )

            _render_replay_tables(table_placeholder, replay_frame_live)

        summary = trader.simulate(
            streamer, delay_seconds=delay_seconds, on_step=_on_step
        )
        replay_frame = _replay_trades_to_frame(summary.trades)

        if summary.applied_entry_threshold < entry_threshold:
            st.info(
                "Model signals were weaker than the requested entry filter; "
                f"the bot lowered the threshold to {summary.applied_entry_threshold * 100:.3f}% "
                "(25th percentile of positive predictions) to encourage moire trades in the window."
            )

        progress_bar.progress(100)
        status_placeholder.success("Replay complete")

        trade_count = len(summary.trades)
        total_pnl_usd = replay_frame["pnl_usd"].sum() if not replay_frame.empty else 0.0

        st.metric("Trades executed", trade_count)
        st.metric("Total PnL (approx, USD)", f"${total_pnl_usd:,.2f}")

        with chart_placeholder.container():
            render_candlestick(frame_slice, replay_frame, "Replay candles with neural trades", key="neural-replay-summary",)
        _render_replay_tables(table_placeholder, replay_frame)

def render_summary(frame: pd.DataFrame) -> None:
    """Display quick statistics for the selected data slice."""

    latest_close = frame["close"].iloc[-1]
    change_series = frame["close"].pct_change()
    day_change = 0.0 if change_series.isna().all() else change_series.iloc[-1] * 100
    high = frame["high"].max()
    low = frame["low"].min()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest close", f"${latest_close:,.2f}")
    col2.metric("Change", f"{day_change:+.2f}%")
    col3.metric("High", f"${high:,.2f}")
    col4.metric("Low", f"${low:,.2f}")


def render_trades_table(trades: pd.DataFrame) -> None:
    """Show the recent trades and aggregated PnL, if available."""

    if trades.empty:
        st.info("No trades recorded in the configured table yet.")
        return

    latest_trades = trades.tail(200).copy()
    latest_trades["timestamp"] = latest_trades["timestamp"].dt.tz_localize(None)
    st.dataframe(latest_trades, hide_index=True, width="stretch")

    if "pnl" in trades.columns:
        st.metric("Cumulative PnL", f"${trades['pnl'].sum():,.2f}")

def render_live_trading_controls(runner: TradingBotRunner | None) -> None:
    """Show controls and status for the always-on trading bot."""

    st.subheader("Live trading bot")

    if runner is None:
        st.info("Minute dataset unavailable; cannot start the live trading bot.")
        return

    status = "Running" if runner.is_running() else "Stopped"
    last_loop = (
        runner.last_loop_started_at.strftime("%Y-%m-%d %H:%M:%S UTC")
        if runner.last_loop_started_at
        else "pending"
    )
    st.caption(
        f"Status: {status} | Trades executed: {runner.trade_count} | "
        f"Last loop started: {last_loop}"
    )

    if runner.entry_price is not None:
        st.caption(f"Active entry price: ${runner.entry_price:,.2f}")

    if runner.last_error:
        st.warning(f"Live bot error: {runner.last_error}")

    col_start, col_stop = st.columns(2)
    if col_start.button("Start live bot", disabled=runner.is_running(), width="stretch"):
        runner.start()
        st.success("Live trading bot started. It will trade and refresh the databank automatically.")

    if col_stop.button("Stop live bot", disabled=not runner.is_running(), width="stretch"):
        runner.stop()
        st.info("Live trading bot stopped.")

def _detect_active_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Return trades that appear to be still active/open."""

    if trades.empty:
        return trades

    status_col = next((col for col in trades.columns if col.lower() == "status"), None)
    if status_col:
        normalized = trades[status_col].astype(str).str.lower()
        return trades[normalized.isin({"open", "active", "pending", "working", "in_progress"})]

    close_col = next(
        (col for col in trades.columns if col.lower() in {"closed_at", "exit_price", "close_price", "close_time"}),
        None,
    )
    if close_col:
        return trades[trades[close_col].isna()]

    return trades.iloc[0:0]

def main():  # pragma: no cover - Streamlit entrypoint
    st.set_page_config(page_title="BTC OHLCV Dashboard", layout="wide")
    st.title("Bitcoin OHLCV Dashboard - Ex Nihilo")

    _auto_refresh(interval_seconds=60)

    st.sidebar.header("Dataset selection")
    dataset = st.sidebar.selectbox("Granularity", list(DATASETS.keys()))

    if st.sidebar.button("Refresh data", width="stretch"):
        _update_databank()
        load_ohlcv.clear()
    if st.sidebar.button("Refresh trades", width="stretch"):
        load_trades.clear()

    try:
        frame = load_ohlcv(dataset)
    except SQLAlchemyError as exc:
        st.error(f"Failed to load data from MySQL: {exc}")
        return

    if frame.empty:
        st.info("No data available for the selected dataset.")
        return

    trades = load_trades()

    min_ts = frame.index.min().to_pydatetime()
    max_ts = frame.index.max().to_pydatetime()
    default_start_dt = max(min_ts, max_ts - timedelta(days=30))

    col_start_date, col_end_date = st.columns(2)
    start_date = col_start_date.date_input(
        "Start date",
        value=default_start_dt.date(),
        min_value=min_ts.date(),
        max_value=max_ts.date(),
    )
    end_date = col_end_date.date_input(
        "End date",
        value=max_ts.date(),
        min_value=start_date,
        max_value=max_ts.date(),
    )
    col_start_time, col_end_time = st.columns(2)
    start_time = col_start_time.time_input("Start time", value=default_start_dt.time())
    end_time = col_end_time.time_input("End time", value=max_ts.time())

    start_dt = datetime.combine(start_date, start_time)
    end_dt = datetime.combine(end_date, end_time)

    start_dt = max(start_dt, min_ts)
    end_dt = min(end_dt, max_ts)

    if start_dt > end_dt:
        st.warning("Start date/time must be before end date/time.")
        return

    start_ts = pd.Timestamp(start_dt)
    end_ts = pd.Timestamp(end_dt)

    frame_slice = frame.loc[start_ts:end_ts]
    if frame_slice.empty:
        st.warning("No data found for the selected range.")
        return

    trades_slice = (
        trades[(trades["timestamp"] >= start_ts) & (trades["timestamp"] <= end_ts)]
        if not trades.empty
        else trades
    )

    render_summary(frame_slice)
    render_candlestick(frame_slice, trades_slice, f"BTC-USD – {dataset} candles", key=f"candles-{dataset.lower()}",)

    tab_data, tab_trades = st.tabs(["Data preview", "Trades"])
    with tab_data:
        st.dataframe(frame_slice.tail(500), width="stretch")
    with tab_trades:
        render_trades_table(trades_slice)

        live_runner = _get_trading_runner()
        render_live_trading_controls(live_runner)

        st.subheader("Neural network bot simulation")
        st.write(
            "The bot starts with $2,000 and risks up to $100 per trade. It trades "
            "until it reaches profitability or exhausts its trade budget."
        )

        col_capital, col_active = st.columns([1, 2])
        starting_capital = 2000.0
        realized_pnl = trades_slice["pnl"].sum() if "pnl" in trades_slice.columns else 0.0
        col_capital.metric("Starting capital", f"${starting_capital:,.2f}")
        col_capital.metric("Capital after selected trades", f"${starting_capital + realized_pnl:,.2f}")
        col_capital.metric("Actual PnL (selected trades)", f"${realized_pnl:,.2f}")

        active_trades = _detect_active_trades(trades_slice)
        if active_trades.empty:
            col_active.info("No active bot trades detected for the selected range.")
        else:
            col_active.dataframe(active_trades, hide_index=True, width="stretch")

        if st.button("TRADE WITH BOT", width="stretch"):
            bot_trades = _simulate_bot_trades(frame_slice)
            st.session_state["bot_trades_result"] = bot_trades

        bot_trades = st.session_state.get("bot_trades_result")
        if bot_trades is not None:
            if bot_trades.empty:
                st.info("Bot could not find a profitable setup in the selected window.")
            else:
                total_pnl = bot_trades["pnl"].sum()
                st.metric("Simulated Bot PnL", f"${total_pnl:,.2f}")
                display_trades = bot_trades.rename(columns={"quantity": "Units purchased"})
                st.dataframe(display_trades, hide_index=True, width="stretch")

        render_neural_replay(start_ts, end_ts)

    st.caption(
        "Data source: Yahoo Finance via the local MySQL databank. Refreshing "
        "the dataset will invalidate the one-minute cache."
    )


if __name__ == "__main__":  # pragma: no cover - Streamlit runner
    main()