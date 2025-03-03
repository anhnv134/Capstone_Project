#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Improved Walk-Forward Intraday Fibonacci Breakout Backtest & Optimization
===========================================================================
This script implements:
  • Algorithm 7: A backtest for an intraday Fibonacci breakout strategy.
  • Algorithm 8: A parameter optimization loop (via Hyperopt) to fine-tune key parameters.

Key steps in Algorithm 7:
  - Load and resample historical OHLCV data.
  - Compute technical indicators (ATR, RSI, ADX, MACD, ASI, EMA) using Wilder’s methods.
  - Determine the morning swing and derive Fibonacci levels.
  - Generate trade signals when price breaks above the long Fibonacci level or below the short Fibonacci level.
  - Optionally use an XGBoost classifier as a gate.
  - Manage trades dynamically with ATR-based stops and a profit target.

Algorithm 8 splits the data into training and testing segments and uses Hyperopt to optimize:
  - ATR multiplier for stops,
  - Minimum swing threshold, and
  - ML gating probability threshold.
"""

import os, glob, math
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, time, timedelta
from xgboost import XGBClassifier
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION (Modified for Improved Performance)
# ============================================================================
class StrategyConfig:
    # Data and Logging
    DATA_PATH       = "E:/VuAnhData/colab/2021_selectedTickers/"
    START_DT        = pd.to_datetime("2021-01-01")
    END_DT          = pd.to_datetime("2021-12-31")
    LOG_PATH        = "./improved_strategy_log.txt"

    # Session & Resampling
    MARKET_OPEN     = time(9, 15)
    MARKET_CLOSE    = time(15, 30)
    SKIP_OPEN_MIN   = 15       # Skip first 15 minutes after open
    SKIP_CLOSE_MIN  = 5
    RESAMPLE_FREQ   = "5T"

    # Indicator Periods
    ATR_PERIOD      = 14
    RSI_PERIOD      = 14
    EMA_PERIOD      = 50
    ADX_PERIOD      = 14

    # Fibonacci & Swing Parameters (Algorithm 7)
    PIVOT_WIN       = 2
    MIN_SWING       = 0.01
    FIB_LEVELS      = [0.236, 0.382, 0.618, 0.764]  # Morning Fibonacci ratios
    LABEL_OFFSET    = 5    # Look 5 bars ahead to create training labels
    PROFIT_THRESH   = 0.0

    # XGBoost Model Parameters (Optional ML Gate)
    XGB_ESTIMATORS  = 180
    XGB_MAX_DEPTH   = 4
    XGB_LR          = 0.04

    # ML Gating Parameters
    ML_PROB_THRESH  = 0.8

    # Risk & Money Management
    RISK_PER_TRADE  = 0.04    # Risk 4% of account per trade
    ACCOUNT_SIZE    = 100000
    ATR_MULTIPLIER  = 0.8     # Tighter stops: 0.8 ATR multiplier (can be optimized)
    SLIPPAGE_RATE   = 0.0001
    COMMISSION_RATE = 0.0005
    PROFIT_MULTIPLIER = 2.0   # Profit target = 2 x ATR stop distance

    # Additional Entry Filters (Relaxed for more trades)
    RSI_UPPER       = 95      # Only enter long if RSI < 95
    RSI_LOWER       = 5       # Only enter short if RSI > 5
    ADX_MIN         = 15      # Require ADX >= 15
    MACD_FILTER     = -999    # MACD filter disabled
    EMA_SLOPE_MIN   = 0.0     # No filtering by EMA slope
    MIN_VOLUME      = 8000
    VOL_FACTOR      = 0.7     # Lower volume multiplier

    # Walk-Forward & Optimization (Algorithm 8)
    TRAIN_MONTHS    = 3
    TEST_MONTHS     = 1
    MAX_OPT_EVALS   = 30      # Broader search

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def log_message(msg: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text = f"[{now}] {msg}"
    print(text)
    try:
        with open(StrategyConfig.LOG_PATH, "a") as f:
            f.write(text + "\n")
    except Exception:
        pass

# ============================================================================
# TIME FILTER FUNCTION
# ============================================================================
def time_filter_ok(dt: pd.Timestamp) -> bool:
    open_dt = dt.replace(hour=StrategyConfig.MARKET_OPEN.hour,
                         minute=StrategyConfig.MARKET_OPEN.minute,
                         second=0, microsecond=0)
    close_dt = dt.replace(hour=StrategyConfig.MARKET_CLOSE.hour,
                          minute=StrategyConfig.MARKET_CLOSE.minute,
                          second=0, microsecond=0)
    if dt < open_dt + timedelta(minutes=StrategyConfig.SKIP_OPEN_MIN):
        return False
    if dt > close_dt - timedelta(minutes=StrategyConfig.SKIP_CLOSE_MIN):
        return False
    return True

# ============================================================================
# DATA LOADING & RESAMPLING
# ============================================================================
def load_data(folder: str) -> pd.DataFrame:
    log_message(f"Loading CSV data from {folder} ...")
    required_cols = {'<date>', '<time>', '<open>', '<high>', '<low>', '<close>', '<volume>'}
    all_csvs = glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)
    dfs = []
    for file in all_csvs:
        try:
            df = pd.read_csv(file)
        except Exception as ex:
            log_message(f"Error reading {file}: {ex}")
            continue
        if not required_cols.issubset(set(df.columns)):
            continue
        df["Datetime"] = pd.to_datetime(df["<date>"] + " " + df["<time>"], errors="coerce")
        df.set_index("Datetime", inplace=True)
        df.rename(columns={"<open>": "Open", "<high>": "High", "<low>": "Low",
                           "<close>": "Close", "<volume>": "Volume"}, inplace=True)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().sort_index()
        df = df[(df.index >= StrategyConfig.START_DT) & (df.index <= StrategyConfig.END_DT)]
        df = df[df.index.map(lambda t: time_filter_ok(t))]
        if df.empty:
            continue
        ticker = os.path.splitext(os.path.basename(file))[0]
        df["Ticker"] = ticker
        dfs.append(df)
    if not dfs:
        log_message("No valid data loaded.")
        return pd.DataFrame()
    data = pd.concat(dfs).sort_index()
    data = data[~data.index.duplicated(keep="first")]
    log_message(f"Loaded {len(data)} rows of data.")
    return data

def resample_data(df: pd.DataFrame, freq: str = StrategyConfig.RESAMPLE_FREQ) -> pd.DataFrame:
    if df.empty:
        return df
    resampled = []
    for tick in df["Ticker"].unique():
        sub = df[df["Ticker"] == tick].copy()
        agg = sub.resample(freq).agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
            "Ticker": "first"
        }).dropna()
        resampled.append(agg)
    return pd.concat(resampled).sort_index()

# ============================================================================
# TECHNICAL INDICATOR FUNCTIONS (Wilder's Methods)
# ============================================================================
def calc_atr(df: pd.DataFrame, period: int = StrategyConfig.ATR_PERIOD) -> pd.Series:
    temp = df.copy()
    temp["range1"] = temp["High"] - temp["Low"]
    temp["range2"] = (temp["High"] - temp["Close"].shift(1)).abs()
    temp["range3"] = (temp["Low"] - temp["Close"].shift(1)).abs()
    temp["TR"] = temp[["range1", "range2", "range3"]].max(axis=1)
    return temp["TR"].ewm(alpha=1/period, adjust=False).mean()

def calc_rsi(df: pd.DataFrame, period: int = StrategyConfig.RSI_PERIOD) -> pd.Series:
    diff = df["Close"].diff()
    gain = diff.clip(lower=0)
    loss = -diff.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def calc_adx(df: pd.DataFrame, period: int = StrategyConfig.ADX_PERIOD) -> pd.Series:
    temp = df.copy()
    temp["up"] = temp["High"] - temp["High"].shift(1)
    temp["down"] = temp["Low"].shift(1) - temp["Low"]
    temp["+DM"] = np.where((temp["up"] > temp["down"]) & (temp["up"] > 0), temp["up"], 0)
    temp["-DM"] = np.where((temp["down"] > temp["up"]) & (temp["down"] > 0), temp["down"], 0)
    temp["ATR"] = calc_atr(temp, period)
    alpha = 1 / period
    temp["+DI"] = 100 * temp["+DM"].ewm(alpha=alpha, adjust=False).mean() / temp["ATR"]
    temp["-DI"] = 100 * temp["-DM"].ewm(alpha=alpha, adjust=False).mean() / temp["ATR"]
    temp["DX"] = 100 * (abs(temp["+DI"] - temp["-DI"]) / (temp["+DI"] + temp["-DI"] + 1e-9))
    return temp["DX"].ewm(alpha=alpha, adjust=False).mean()

def calc_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - sig_line
    return macd_line, sig_line, hist

def calc_asi(df: pd.DataFrame, limit: float = 1.0) -> pd.Series:
    df = df.copy()
    df["Close_prev"] = df["Close"].shift(1)
    df["Open_prev"] = df["Open"].shift(1)
    df["K"] = np.maximum((df["High"] - df["Close_prev"]).abs(),
                         (df["Low"] - df["Close_prev"]).abs())
    def compute_M(row):
        if row["Close"] >= row["Close_prev"]:
            return (row["Close"] - row["Close_prev"]) + 0.5*(row["Close"] - row["Open"]) + 0.25*(row["Close_prev"] - row["Open"])
        else:
            return (row["Close"] - row["Close_prev"]) - 0.5*(row["Close"] - row["Open"]) - 0.25*(row["Close_prev"] - row["Open"])
    df["M"] = df.apply(compute_M, axis=1)
    def compute_R(row):
        up_move = row["High"] - row["Close_prev"]
        down_move = row["Close_prev"] - row["Low"]
        if up_move > down_move and up_move > 0:
            R = (row["High"] - row["Close_prev"]) - 0.5*(row["Close_prev"] - row["Low"]) + 0.25*(row["Close_prev"] - row["Open"])
        elif down_move > up_move and down_move > 0:
            R = (row["Close_prev"] - row["Low"]) - 0.5*(row["High"] - row["Close_prev"]) + 0.25*(row["Close_prev"] - row["Open"])
        else:
            R = (row["High"] - row["Low"]) + 0.25*(row["Close_prev"] - row["Open"])
        return R if R != 0 else 1e-10
    df["R"] = df.apply(compute_R, axis=1)
    df["SI"] = 50.0 * (df["M"] / df["R"]) * (df["K"] / limit)
    return df["SI"].cumsum().fillna(0)

def ema_slope(series: pd.Series, window: int = 1) -> pd.Series:
    return series.diff(window)

# ============================================================================
# DATASET BUILDING & MODEL TRAINING
# ============================================================================
def create_training_dataset(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df["ATR"] = calc_atr(df, StrategyConfig.ATR_PERIOD)
    df["RSI"] = calc_rsi(df, StrategyConfig.RSI_PERIOD)
    samples = []
    grouped = df.groupby([df.index.normalize(), "Ticker"])
    for (day, tick), group in grouped:
        fibs = compute_morning_fib(group)
        long_level = fibs.get(0.236)
        short_level = fibs.get(0.764)
        if long_level is None or short_level is None:
            continue
        bars = group.index
        for i, ts in enumerate(bars):
            if i + StrategyConfig.LABEL_OFFSET >= len(bars):
                break
            row = group.loc[ts]
            fut_price = group.iloc[i + StrategyConfig.LABEL_OFFSET]["Close"]
            if row["Close"] > long_level:
                label = 1 if (fut_price - row["Close"] > StrategyConfig.PROFIT_THRESH) else 0
                samples.append({
                    "Close": row["Close"],
                    "Volume": row["Volume"],
                    "RSI": row["RSI"],
                    "ATR": row["ATR"],
                    "Label": label,
                    "Ticker": tick,
                    "Signal": "LONG"
                })
            if row["Close"] < short_level:
                label = 1 if (row["Close"] - fut_price > StrategyConfig.PROFIT_THRESH) else 0
                samples.append({
                    "Close": row["Close"],
                    "Volume": row["Volume"],
                    "RSI": row["RSI"],
                    "ATR": row["ATR"],
                    "Label": label,
                    "Ticker": tick,
                    "Signal": "SHORT"
                })
    return pd.DataFrame(samples)

def compute_morning_fib(df: pd.DataFrame) -> dict:
    morning = df.between_time(StrategyConfig.MARKET_OPEN, time(12, 0))
    if len(morning) < 2:
        return {}
    high = morning["High"].max()
    low = morning["Low"].min()
    rng = high - low
    if rng < StrategyConfig.MIN_SWING:
        return {}
    fibs = {}
    for level in StrategyConfig.FIB_LEVELS:
        fibs[level] = low + rng * level
    return fibs

def train_model(training_df: pd.DataFrame):
    if training_df.empty:
        return None
    features = [col for col in training_df.columns if col not in ["Label", "Ticker", "Signal"]]
    X = training_df[features].values
    y = training_df["Label"].values
    model = XGBClassifier(n_estimators=StrategyConfig.XGB_ESTIMATORS,
                          max_depth=StrategyConfig.XGB_MAX_DEPTH,
                          learning_rate=StrategyConfig.XGB_LR,
                          eval_metric="logloss",
                          use_label_encoder=False,
                          random_state=42)
    model.fit(X, y)
    return model

# ============================================================================
# BACKTESTING FUNCTION (Algorithm 7 Implementation)
# ============================================================================
def run_backtest(df: pd.DataFrame, model) -> list:
    trades = []
    if df.empty:
        return trades
    groups = df.groupby([df.index.normalize(), "Ticker"])
    for (day, tick), day_data in groups:
        if len(day_data) < 5:
            continue
        fib_levels = compute_morning_fib(day_data)
        long_fib = fib_levels.get(0.236)
        short_fib = fib_levels.get(0.764)
        if long_fib is None or short_fib is None:
            continue
        day_data["ATR"] = calc_atr(day_data, StrategyConfig.ATR_PERIOD)
        day_data["RSI"] = calc_rsi(day_data, StrategyConfig.RSI_PERIOD)
        day_data["ADX"] = calc_adx(day_data, StrategyConfig.ADX_PERIOD)
        macd_line, sig_line, macd_hist = calc_macd(day_data)
        day_data["MACD_Hist"] = macd_hist  # MACD filter disabled
        day_data["EMA50"] = day_data["Close"].ewm(span=StrategyConfig.EMA_PERIOD, adjust=False).mean()
        day_data["EMA_Slope"] = ema_slope(day_data["EMA50"], window=1)
        vol_thresh = max(StrategyConfig.MIN_VOLUME, day_data["Volume"].mean() * StrategyConfig.VOL_FACTOR)
        open_positions = []
        for ts, row in day_data.iterrows():
            # End-of-day: close open positions at market close.
            if ts.time() >= StrategyConfig.MARKET_CLOSE:
                for pos in open_positions:
                    close_trade(trades, pos, row["Close"], ts)
                open_positions = []
                break
            price = row["Close"]
            # First, update open positions with dynamic ATR-based stops.
            still_open = []
            for pos in open_positions:
                side = pos["side"]
                curr_stop = pos["stop_price"]
                new_stop = price - row["ATR"] * StrategyConfig.ATR_MULTIPLIER if side == "LONG" else price + row["ATR"] * StrategyConfig.ATR_MULTIPLIER
                # Also check for profit target.
                if side == "LONG":
                    pos["stop_price"] = max(curr_stop, new_stop)
                    if price >= pos["profit_target"]:
                        close_trade(trades, pos, price, ts)
                    elif price < pos["stop_price"]:
                        close_trade(trades, pos, pos["stop_price"], ts)
                    else:
                        still_open.append(pos)
                else:
                    pos["stop_price"] = min(curr_stop, new_stop)
                    if price <= pos["profit_target"]:
                        close_trade(trades, pos, price, ts)
                    elif price > pos["stop_price"]:
                        close_trade(trades, pos, pos["stop_price"], ts)
                    else:
                        still_open.append(pos)
            open_positions = still_open

            # Optional ML gating: predict probability using the XGBoost model.
            features = [price, row["Volume"], row["RSI"], row["ATR"]]
            ml_prob = 0.5
            if model is not None:
                try:
                    ml_prob = model.predict_proba(np.array(features).reshape(1, -1))[:,1][0]
                except Exception:
                    ml_prob = 0.5

            if np.isnan(price) or np.isnan(row["ATR"]) or row["ATR"] <= 0 or price <= 0:
                continue
            stop_dist = row["ATR"] * StrategyConfig.ATR_MULTIPLIER
            denom = stop_dist * price
            if np.isnan(denom) or denom <= 0:
                continue

            # Additional Filters: relaxed RSI, ADX, and volume.
            if row["RSI"] >= StrategyConfig.RSI_UPPER or row["RSI"] <= StrategyConfig.RSI_LOWER:
                continue
            if row["ADX"] < 15:
                continue
            if row["Volume"] < vol_thresh:
                continue

            risk_capital = StrategyConfig.ACCOUNT_SIZE * StrategyConfig.RISK_PER_TRADE
            if ml_prob > StrategyConfig.ML_PROB_THRESH:
                risk_capital *= 1.5
            qty = math.floor(risk_capital / denom)
            if qty <= 0:
                continue

            # Compute profit target as a multiple of the stop distance.
            profit_target_long = price + stop_dist * StrategyConfig.PROFIT_MULTIPLIER
            profit_target_short = price - stop_dist * StrategyConfig.PROFIT_MULTIPLIER

            if price > long_fib:
                pos = {
                    "ticker": tick,
                    "side": "LONG",
                    "entry_price": price,
                    "quantity": qty,
                    "stop_price": price - stop_dist,
                    "profit_target": profit_target_long,
                    "entry_time": ts
                }
                open_positions.append(pos)
            if price < short_fib:
                pos = {
                    "ticker": tick,
                    "side": "SHORT",
                    "entry_price": price,
                    "quantity": qty,
                    "stop_price": price + stop_dist,
                    "profit_target": profit_target_short,
                    "entry_time": ts
                }
                open_positions.append(pos)
        if open_positions:
            final_price = day_data.iloc[-1]["Close"]
            final_time = day_data.index[-1]
            for pos in open_positions:
                close_trade(trades, pos, final_price, final_time)
    return trades

def close_trade(trade_list: list, pos: dict, exit_price: float, exit_time: datetime):
    entry_price = pos["entry_price"]
    qty = pos["quantity"]
    side = pos["side"]
    entry_adj = entry_price + (entry_price * StrategyConfig.SLIPPAGE_RATE if side == "LONG" else -entry_price * StrategyConfig.SLIPPAGE_RATE)
    exit_adj = exit_price + (exit_price * StrategyConfig.SLIPPAGE_RATE if side == "LONG" else -exit_price * StrategyConfig.SLIPPAGE_RATE)
    profit = (exit_adj - entry_adj) * qty if side == "LONG" else (entry_adj - exit_adj) * qty
    net_profit = profit - (entry_price * qty * StrategyConfig.COMMISSION_RATE + exit_price * qty * StrategyConfig.COMMISSION_RATE)
    trade_list.append({
        "Ticker": pos["ticker"],
        "Side": side,
        "EntryTime": pos["entry_time"],
        "EntryPrice": entry_price,
        "ExitTime": exit_time,
        "ExitPrice": exit_price,
        "Quantity": qty,
        "GrossPnL": round(profit, 2),
        "NetPnL": round(net_profit, 2)
    })

# ============================================================================
# PERFORMANCE EVALUATION
# ============================================================================
def evaluate_performance(trades: list) -> dict:
    if not trades:
        return dict(net_pl=0.0, total_trades=0, win_rate=0.0, avg_pnl=0.0,
                    max_drawdown=0.0, sharpe=0.0, profit_factor=0.0, return_pct=0.0)
    df = pd.DataFrame(trades).sort_values("ExitTime")
    pnl = df["NetPnL"].values
    net_pl = pnl.sum()
    total = len(pnl)
    wins = (pnl > 0).sum()
    win_rate = (wins / total) * 100 if total > 0 else 0.0
    avg_pnl = pnl.mean()
    cum = np.cumsum(pnl)
    drawdown = np.max(np.maximum.accumulate(cum) - cum)
    std_pnl = pnl.std(ddof=1)
    sharpe = (avg_pnl / std_pnl) * math.sqrt(total) if std_pnl > 1e-9 else 0.0
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    profit_factor = gains / losses if losses > 1e-9 else 0.0
    ret_pct = (net_pl / StrategyConfig.ACCOUNT_SIZE) * 100.0
    return dict(net_pl=round(net_pl, 2), total_trades=total, win_rate=round(win_rate, 2),
                avg_pnl=round(avg_pnl, 2), max_drawdown=round(drawdown, 2),
                sharpe=round(sharpe, 3), profit_factor=round(profit_factor, 3),
                return_pct=round(ret_pct, 2))

def compute_final_stats(df: pd.DataFrame, model) -> dict:
    trades = run_backtest(df, model)
    return evaluate_performance(trades)

# ============================================================================
# WALK-FORWARD EVALUATION & HYPEROPTIMIZATION (Algorithm 8)
# ============================================================================
def generate_segments(start_dt, end_dt, train_m, test_m):
    segments = []
    current_start = start_dt
    current_train_end = current_start + pd.offsets.MonthEnd(train_m)
    while True:
        test_start = current_train_end + timedelta(days=1)
        test_end = test_start + pd.offsets.MonthEnd(test_m)
        if test_end > end_dt:
            break
        segments.append((current_start, current_train_end, test_start, test_end))
        current_start = test_start
        current_train_end = test_start + pd.offsets.MonthEnd(train_m)
    return segments

def walk_forward(df: pd.DataFrame) -> dict:
    segs = generate_segments(StrategyConfig.START_DT, StrategyConfig.END_DT,
                             StrategyConfig.TRAIN_MONTHS, StrategyConfig.TEST_MONTHS)
    results = []
    for i, (train_start, train_end, test_start, test_end) in enumerate(segs):
        log_message(f"Segment {i}: Train [{train_start.date()} -> {train_end.date()}], Test [{test_start.date()} -> {test_end.date()}]")
        train_data = df.loc[(df.index >= train_start) & (df.index <= train_end)]
        if train_data.empty:
            log_message("No training data; skipping segment.")
            continue
        train_ds = create_training_dataset(train_data)
        if len(train_ds) < 10:
            log_message("Not enough training samples; skipping segment.")
            continue
        model = train_model(train_ds)
        if model is None:
            log_message("Model training failed; skipping segment.")
            continue
        test_data = df.loc[(df.index >= test_start) & (df.index <= test_end)]
        if test_data.empty:
            log_message("No testing data; skipping segment.")
            continue
        stats = compute_final_stats(test_data, model)
        stats.update({"Segment": i, "TrainStart": train_start, "TrainEnd": train_end,
                      "TestStart": test_start, "TestEnd": test_end})
        results.append(stats)
        log_message(f"Segment {i} stats: {stats}")
    if not results:
        return {}
    df_res = pd.DataFrame(results)
    summary = dict(
        net_pl = df_res["net_pl"].sum(),
        total_trades = df_res["total_trades"].sum(),
        win_rate = df_res["win_rate"].mean(),
        avg_pnl = df_res["avg_pnl"].mean(),
        max_drawdown = df_res["max_drawdown"].max(),
        sharpe = df_res["sharpe"].mean(),
        profit_factor = df_res["profit_factor"].mean(),
        return_pct = df_res["return_pct"].mean()
    )
    return summary

GLOBAL_DF = None
def opt_objective(params):
    StrategyConfig.ATR_MULTIPLIER = params["atr_mult"]
    StrategyConfig.MIN_SWING = params["swing_min"]
    StrategyConfig.ML_PROB_THRESH = params["ml_prob"]
    global GLOBAL_DF
    perf = walk_forward(GLOBAL_DF)
    if not perf:
        return {"loss": 9999, "status": STATUS_OK, "params": params, "sharpe": 0, "netpl": 0}
    loss_val = - (perf.get("sharpe", 0) + 0.0001 * perf.get("net_pl", 0))
    return {"loss": loss_val, "status": STATUS_OK, "params": params,
            "sharpe": perf.get("sharpe", 0), "netpl": perf.get("net_pl", 0)}

def run_optimization(df: pd.DataFrame):
    global GLOBAL_DF
    GLOBAL_DF = df
    space = {
        "atr_mult": hp.uniform("atr_mult", 1.0, 2.0),
        "swing_min": hp.uniform("swing_min", 0.005, 0.05),
        "ml_prob": hp.uniform("ml_prob", 0.7, 0.95)
    }
    trials = Trials()
    best = fmin(fn=opt_objective, space=space, algo=tpe.suggest,
                max_evals=StrategyConfig.MAX_OPT_EVALS, trials=trials)
    log_message(f"Hyperopt raw best: {best}")
    losses = [t["result"]["loss"] for t in trials.trials]
    best_index = np.argmin(losses)
    best_result = trials.trials[best_index]["result"]
    log_message(f"Optimized Params: {best_result['params']}")
    log_message(f"Optimized Sharpe: {best_result['sharpe']}, NetPL: {best_result['netpl']}")
    final_perf = walk_forward(df)
    log_message(f"Final Walk-Forward Performance: {final_perf}")
    return best_result["params"], final_perf

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    log_message("=== Loading Data ===")
    raw_data = load_data(StrategyConfig.DATA_PATH)
    if raw_data.empty:
        log_message("No data loaded. Exiting.")
        return
    log_message("=== Resampling Data ===")
    bars = resample_data(raw_data, StrategyConfig.RESAMPLE_FREQ)
    if bars.empty:
        log_message("No resampled bars. Exiting.")
        return
    log_message("=== Running Single Walk-Forward Evaluation ===")
    single_perf = walk_forward(bars)
    log_message(f"Single WF Performance: {single_perf}")
    log_message("=== Running Hyperopt Parameter Optimization ===")
    best_params, opt_perf = run_optimization(bars)
    log_message(f"Optimal Parameters Found: {best_params}")
    log_message(f"Optimized Performance: {opt_perf}")

if __name__ == "__main__":
    main()
