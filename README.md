**Required packages to be installed:**
1. For Algorithm 1 - 6: It is required to install pandas package
from cmd run the code: pip install pandas
2. For Algorithm 7 - 8: It is required to install joblib, xgboost package
from cmd run the code:
  pip install pypi
  pip install joblib
  pip install xgboost
3. For Algorithm 9: It is required to install hyperopt package
from cmd run the code:
  pip install hyperopt

**Algorithm 1: Data Preprocessing and Selection of Traded Instruments**

**Goal:** Our goal here is to identify which NSE stocks, futures to apply the breakout strategy to. We will focus on instruments with sufficient 1)liquidity 2)volatility and 3)trendliness.

**Inputs:**
- Universe of NSE instruments
- Intraday OHLCV data (O: open, H: high, L: low, C: close, V: volume) for each instrument
- Chosen intraday time frame (e.g., 5-min bars)
- Minimum liquidity threshold (e.g., average volume over X days)
- Minimum volatility threshold (e.g., ATR > certain value)
- Optional trend filter (e.g., ADX > Y)
"""

**Algorithm 2 Option A: Computing Fibinacci Levels for Intraday Breakouts**

**Goal:** Our goal here is for each chosen instrument,we will identify key Fibonacci retracements and extensions from a recent swing (e.g., the morning session swing) to determine breakout points.
 
**Inputs:**
- Intraday price series (O, H, L, C)
- A reference swing to measure Fibonacci ratios (e.g., the day's first significant swing high and swing low)
- Common Fibonacci ratios: '[0.382, 0.5, 0.618, 1.0, 1.272, 1.618, ...]'

**Algorithm 3: Calculation of Swing Index (SI) and Accumulative Swing Index (ASI)**

**Goal:** Our goal here is to utilize Swing Index (SI) and Accumulative Swing Index (ASI) calculations to confirm that momentum supports the breakout levels. The ASI can help verify if a bareakout is backed by a meaningful swing.This serves as a filter. We will only engage in breakouts if the ASI confirms a significant swing point.

**Inputs:**
Intraday OHLC data for the current session
Previous bar's close and previous ASI value
SI, ASI formula parameters

**Algorithm 4: Breakout Confimation and Entry Logic with Additional Indicators**

**Goal:**
Our goal here is to trigger breakout trades only when Fibonacci-based barekout conditions align with 1)trend confirmation (via EMA), 2)volume participation, 3)momentum signals (ASI and MACD), and 4)avoid trades at extreme RSI levels (overbought or oversold)

**Inputs:**
current_price: Current bar's closing price
fibonacci_levels: Dictionary of Fibonacci levels (e.g., {0.32: price, 0.5: price, 0.618: price, ...})
ASI_value: Current Accumulative Swing Index value
has_HSP, has_LSP: Booleans indicating recent High or Low Swing Points confirmed by ASI
ema_50: 50-period EMA value at current bar
ema_slope(ema_50): Slope or trend direction inferred from ema_50 (positive slope = uptrend, negative slope = downtrend)
current_volume, avg_volume: Current and average volumes for volume spike check
macd_hist, previous_macd_hist: Current and previous MACD histogram values for momentum confirmation
rsi: Current RSI value
risk_per_trade, ATR: For position sizing and stop calculation
fibonacci_levels[0.618_downside_level]: Downward breakout Fibonacci level (mirroring the 0.618 ratio)
session_end_time: Time at which all positions must be closed

**Algorithm 5: Dynamic Stop-Loss, Profit Target, and Time-Based Exit**
**Goal:** Our goal here is once we are in a position, we will manage it using a volatility-based trailing stop, set profit targets at next Fibonacci extension levels, and ensure no positions are held past the session end time.

**Inputs:**
Open positions
ATR or another volatility measure
End-of-day exit requirement (no overnight holding)
Trailing increment (e.g., ATR * factor for trailing stop)

**Algorithm 6: Performance Tracking an d Post-Session Optimization**

**Goal:**

Our goal here is after the session ends, evaluae key performance metrics (e.g., win rate, average Profit and Loss, maximum drawdown, Sharpe ratio).And after that we will use these insights to adjust parameters for future sessions.

**Inputs:**

Trade logs of the day (e.g., entry prices, exit prices, timestamps, profit and loss per trade)
Historical performance database for reference

**Algorithm 7: Backtesting the Intraday Fibonacci Breakout Strategy**

**Goal:**

Our goal here is to evaluate the strategy's performance on historical intraday data. The backtest helps fine-tune parameters(e.g., ATR multiples for stops, specific Fibonacci levels, ASI thresholds). And we will also identify edge cases before going live if the opportunity allowed.

**Inputs:**

Historical intraday OHLCV data for all selected instruments

**Strategy parameters:**

Liquidity, Volatility thresholds
ATR multiplier for stops
Chosen Fibonacci ratios (e.g., [0.618, 1.272, ...])
SI, ASI thresholds for breakout confirmation
Designated historical period for backtesting (e.g., last 6 months)
Chosen time interval (e.g., 5-min bars)
Transaction cost assumptions (commissions, slippage)

**Algorithm 8: Parameter Optimization Loop**
**Goal:**

Our goal here is to systematically test differet parameter sets to find the combination that provides the best performance metrics. This involves running multiple backtests, each with a unique set of parameter, and then comparing result.
**Inputs:**

A range of values for each key parameter

**Example:**

ATR multipliers = [1.0, 1.5, 2.0]
ADX filters = [20, 25, 30]
Different Fibonacci ratio subsets (e.g., [0.382,0.618, 1.272] vs. [0.5,0.618, 1.618])
Backtest algorithm from previous steps

**Algorithm 9: Forward Testing and Paper Trading**
**Goal:**

Our goal here is after identify a promising parameter set, verifying its robustness by applying it to out-of-sample historical data or running it in a paper-trading environment. This step helps confirm that the strategy's good performance is not just a result of overfitting.

**Inputs:**

Out-of-sample historical data or live market feed (for paper trading)

Best parameters derived from the optimization step
