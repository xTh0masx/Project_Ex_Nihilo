# Project_Ex_Nihilo

First concept idea of the project. - Note: exact definition of profit and caped loss will be done as soon as project has started.

The first goal is for the program to execute standard spot trading strategies with Bitcoin.
Depending on the coin’s volatility, the bot should aim a minimum profit/loss (P/L) of +10%. 
The next step is to enable leveraged trading with a maximum leverage of x50.
In this mode, the achievable P/L should be capped at ±50% to prevent a total loss.

Website to gather information:
- https://finance.yahoo.com/markets/crypto/all/
- https://www.binance.com/en
- https://www.bybit.com/en/
- trading view
- 

## Configuration

### Phase 1 ingestion service

1. Copy the example configuration so you can customise it locally:

   ```bash
   cp "Phase 1/settings.example.yml" "Phase 1/settings.yml"
   cp "Phase 1/.env.example" "Phase 1/.env"
   ```

2. Fill in the required secrets either by editing `Phase 1/settings.yml` directly or by
   exporting the corresponding environment variables before launching the ingestion
   service. The configuration references the following variables:

   - `BINANCE_API_KEY`
   - `BINANCE_API_SECRET`
   - `BINANCE_BASE_URL` (optional – defaults to the public API)
   - `MYSQL_HOST`
   - `MYSQL_PORT`
   - `MYSQL_USER`
   - `MYSQL_PASSWORD`
   - `MYSQL_DATABASE`

   Example of exporting them in your shell:

   ```bash
   export BINANCE_API_KEY=your-key
   export BINANCE_API_SECRET=your-secret
   export MYSQL_USER=project_user
   export MYSQL_PASSWORD=strong-password
   export MYSQL_DATABASE=project_ex_nihilo
   ```

   Any variables left unset will fall back to the defaults defined in
   `Phase 1/settings.yml` if one is provided.


## Work process

### 1. Structural Plan - Data, Method
### 2. Data Gathering
### 3. Data Processing
### 4. Sandbox Beta Version 
### 5. Bot Logics for spot trading
### 6. First Bot Concept according to Data
### X. Bot Logics for factor (Leverage) trading

Zeitreiheanalyse
Metadaten Dollar/Euro - S&P500 - NASDAQ - Interestrate
Patternanalyse
Bestimmte Trading Situationen -> Nur Standard szenarie infos finden und tranieren
load data Yahoo finance into R-Studio with Code :D

### Klassendiagramm

classDiagram
direction LR

%% === Core Abhängigkeiten ===
class Config {
  +load(path:str) : Config
  +get(key:str, default) : any
}

class Logger {
  +info(msg:str)
  +warn(msg:str)
  +error(msg:str, exc:Exception)
}

class Storage {
  +save_ohlcv(symbol:str, df) : None
  +load_ohlcv(symbol:str, tf:str) : DataFrame
  +save_trade(trade:Trade) : None
  +save_model(tag:str, obj:any) : None
  +load_model(tag:str) : any
}

%% === Data Layer ===
class DataFeed {
  <<abstract>>
  +subscribe_ticker(symbol:str) : None
  +subscribe_ohlcv(symbol:str, timeframe:str) : None
  +fetch_ohlcv(symbol:str, timeframe:str, limit:int) : DataFrame
  +close() : None
  #on_ticker(event:TickerEvent) : None
  #on_ohlcv(event:CandleEvent) : None
}

class BinanceDataFeed
class BybitDataFeed
class YahooDataFeed

DataFeed <|-- BinanceDataFeed
DataFeed <|-- BybitDataFeed
DataFeed <|-- YahooDataFeed

class TickerEvent { +ts:int; +symbol:str; +price:float; +bid:float; +ask:float; +volume:float }
class CandleEvent { +ts:int; +symbol:str; +o:float; +h:float; +l:float; +c:float; +v:float }

%% === Strategy Layer ===
class Strategy {
  <<abstract>>
  +on_start(ctx:Context) : None
  +on_bar(bar:CandleEvent) : None
  +on_tick(tick:TickerEvent) : None
  +on_end(ctx:Context) : None
  +get_state() : dict
  +set_state(state:dict) : None
}

class RuleBasedStrategy {
  -params:StrategyParams
  -indicators:Indicators
  +on_bar(bar) : None
  +signal() : Signal
}

class MLStrategy {
  -model:any
  -feature_builder:FeatureBuilder
  +on_bar(bar) : None
  +signal() : Signal
  +train(df) : None
  +predict(features) : float
}

Strategy <|-- RuleBasedStrategy
Strategy <|-- MLStrategy

class Indicators {
  +sma(series, n:int) : Series
  +ema(series, n:int) : Series
  +rsi(series, n:int) : Series
  +atr(df, n:int) : Series
}

class FeatureBuilder {
  +transform(df) : DataFrame
  +latest(features_from_state) : ndarray
}

class Signal { +side:Side; +confidence:float; +price:float }
class Side { <<enum>> LONG; SHORT; FLAT }

%% === Portfolio & Risiko ===
class Portfolio {
  +cash:float
  +positions:dict~str, Position~
  +update_mark_to_market(tick:TickerEvent) : None
  +exposure(symbol:str) : float
  +available_cash() : float
}

class Position {
  +symbol:str
  +size:float
  +avg_price:float
  +unrealized_pnl:float
  +update(price:float) : None
  +is_flat() : bool
}

class RiskManager {
  +max_position_usd:float
  +max_daily_loss:float
  +stop_loss_pct:float
  +take_profit_pct:float
  +validate_order(order:Order, portfolio:Portfolio) : bool
  +protective_orders(entry:Order) : list~Order~
}

Portfolio "1" o-- "*" Position
RiskManager --> Portfolio

%% === Execution Layer ===
class ExchangeAdapter {
  <<abstract>>
  +fetch_market_rules(symbol:str) : MarketRules
  +create_order(order:Order) : OrderResult
  +cancel_order(id:str) : None
  +get_open_orders(symbol:str) : list~Order~
  +get_balance() : dict
}

class CCXTAdapter
class BinanceAdapter
class BybitAdapter

ExchangeAdapter <|-- CCXTAdapter
ExchangeAdapter <|-- BinanceAdapter
ExchangeAdapter <|-- BybitAdapter

class ExecutionEngine {
  -adapter:ExchangeAdapter
  -risk:RiskManager
  +route(signal:Signal, symbol:str, portfolio:Portfolio) : list~OrderResult~
  +place_market(symbol:str, side:Side, qty:float) : OrderResult
  +place_limit(symbol:str, side:Side, qty:float, px:float) : OrderResult
}

class MarketRules { +min_qty:float; +step_size:float; +tick_size:float; +min_notional:float }

class Order { +id:str; +symbol:str; +side:Side; +type:OrderType; +qty:float; +price:float }
class OrderType { <<enum>> MARKET; LIMIT; STOP; TAKE_PROFIT }
class OrderResult { +order:Order; +status:str; +filled:float; +avg_price:float }

ExecutionEngine --> ExchangeAdapter
ExecutionEngine --> RiskManager

%% === Orchestrierung ===
class Bot {
  -feed:DataFeed
  -strategy:Strategy
  -exec:ExecutionEngine
  -portfolio:Portfolio
  -logger:Logger
  -storage:Storage
  +start() : None
  +stop() : None
  +on_candle(event:CandleEvent) : None
  +on_ticker(event:TickerEvent) : None
}

Bot --> DataFeed
Bot --> Strategy
Bot --> ExecutionEngine
Bot --> Portfolio
Bot --> Storage
Bot --> Logger

%% === Backtesting / Paper ===
class Backtester {
  -strategy:Strategy
  -broker:PaperBroker
  +run(df_ohlcv:DataFrame) : BacktestReport
}

class PaperBroker {
  -rules:MarketRules
  +submit(order:Order) : OrderResult
  +mark_to_market(price:float) : None
}

class BacktestReport { +trades:list~Trade~; +equity_curve:Series; +stats:dict }
class Trade { +entry:OrderResult; +exit:OrderResult; +pnl:float; +bars_held:int }

Backtester --> PaperBroker
Backtester --> Strategy
Backtester --> BacktestReport
PaperBroker --> MarketRules

%% === Monitoring ===
class Metrics {
  +record_latency(ms:float)
  +record_fill(order_id:str, slippage:float)
  +equity(equity:float)
}

Bot --> Metrics
