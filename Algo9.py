#Algorithm 9

"""
Multi-Group Trading Simulation:
  - We perform in-sample and out-of-sample backtesting for two groups of tickers.
    (Group A and Group B) of tickers.
  - Each group uses its own in-sample and out-of-sample directories and date ranges.
  - Then, using the out-of-sample data, a live-trading (paper trading) simulation is launched separately for each group.



Key Steps:
  - We use in-sample data to optimize strategy parameters.
  - After that, we apply the best parameters to out-of-sample data and evaluate performance.
  - If the out-of-sample performance is acceptable, (e.g., sharpe > 0.5), we connect to a simulated live feed and run paper trading.

"""
#import necessary libraries
import os
import glob
import math
import time
import json
import copy

import threading  # threading for concurrent simulation
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, time as dtime, timedelta
from xgboost import XGBClassifier
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================

class BaseConfig:
    MARKET_OPEN         = dtime(9, 15)
    MARKET_CLOSE        = dtime(15, 30)
    SKIP_OPEN           = 15
    SKIP_CLOSE          = 5
    RESAMPLE_FREQ       = "5T"
    ATR_PERIOD          = 14
    RSI_PERIOD          = 14
    EMA_PERIOD          = 50
    ADX_PERIOD          = 14
    MIN_SWING           = 0.01
    FIB_LEVELS          = [0.236, 0.382, 0.618, 0.764]
    LABEL_OFFSET        = 5
    PROFIT_THRESH       = 0.0
    XGB_ESTIMATORS      = 180
    XGB_MAX_DEPTH       = 4
    XGB_LR              = 0.04
    ML_THRESHOLD        = 0.8
    RISK_PER_TRADE      = 0.04
    ACCOUNT_SIZE        = 100000
    ATR_MULTIPLIER      = 0.8
    SLIPPAGE            = 0.0001
    COMMISSION          = 0.0005
    PROFIT_MULTIPLIER   = 2.0
    RSI_UPPER           = 95
    RSI_LOWER           = 5
    ADX_MIN             = 15
    MIN_VOLUME          = 5000#8000
    VOL_FACTOR          = 0.7
    TRAIN_MONTHS        = 3
    TEST_MONTHS         = 1
    MAX_OPT_EVALS       = 100



class GroupAConfig(BaseConfig):
    DATA_PATH = "E:/VuAnhData/test/2021_selectedTickers/"
    OOS_PATH  = "E:/VuAnhData/test/2022_selectedTickers/"
    START_DT  = pd.to_datetime("2021-01-01")
    END_DT    = pd.to_datetime("2021-12-31")
    OOS_START = pd.to_datetime("2022-01-01")
    OOS_END   = pd.to_datetime("2022-12-31")
    TICKERS   = [

        'ZENSARTECH', 'TATAMTRDVR', 'TATAMOTORS', 'TATACOMM',
        'RESPONIND', 'OIL', 'NIITLTD', 'M&M', 'JSWSTEEL', 'IRCTC',
        'INTELLECT', 'INFY', 'HINDALCO', 'HDFCBANK', 'HDFC',
        'DHANI', 'DELTACORP', 'DBL', 'CAMLINFINE', 'BALRAMCHIN',
        'AUBANK', 'APOLLO', 'ADANIPORTS', 'ABFRL'
    ]

class GroupBConfig(BaseConfig):

    DATA_PATH = "E:/VuAnhData/test/2021_lxchem/"
    OOS_PATH  = "E:/VuAnhData/test/2022_lxchem/"
    START_DT  = pd.to_datetime("2021-01-01")
    END_DT    = pd.to_datetime("2021-12-31")
    OOS_START = pd.to_datetime("2022-01-01")
    OOS_END   = pd.to_datetime("2022-12-31")
    TICKERS   = [

        'KRBL', 'M&M', 'LXCHEM', 'OIL', 'PARAGMILK', 'RALLIS',
        'SHANTIGEAR', 'SPARC', 'STLTECH', 'TIRUMALCHM',
        'ADANIPOWER', 'ARVIND', 'BALRAMCHIN', 'CYBERTECH', 'DCAL',
        'DHAMPURSUG', 'FCL', 'GAEL', 'GOKEX', 'GUJGASLTD',
        'GUJALKALI', 'INOXLEISUR', 'ISEC', 'KCP'
    ]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def log_message(msg: str, group: str = None):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if group:
        text = f"[{now}] [{group}] {msg}"
    else:
        text = f"[{now}] {msg}"
    print(text)
    try:
        with open(BaseConfig.LOG_PATH, "a") as f:
            f.write(text + "\n")
    except Exception:
        pass

def is_trading_time(dt: pd.Timestamp, config: BaseConfig) -> bool:
    open_time = dt.replace(hour=config.MARKET_OPEN.hour, minute=config.MARKET_OPEN.minute, second=0, microsecond=0)
    close_time = dt.replace(hour=config.MARKET_CLOSE.hour, minute=config.MARKET_CLOSE.minute, second=0, microsecond=0)
    return dt >= open_time + timedelta(minutes=config.SKIP_OPEN) and dt <= close_time - timedelta(minutes=config.SKIP_CLOSE)

def get_group_label(config: BaseConfig) -> str:
    if config == GroupAConfig:
        return "Group A"
    elif config == GroupBConfig:
        return "Group B"
    else:
        return "Unknown Group"


# =============================================================================
# DATA LOADING AND RESAMPLING FUNCTIONS
# =============================================================================

def load_csv_data(folder: str, config: BaseConfig) -> pd.DataFrame:
    log_message(f"Loading CSV data from {folder} ...", group=get_group_label(config))
    file_list = glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)
    all_data = []
    for file in file_list:
        try:
            df = pd.read_csv(file)
        except Exception as e:
            log_message(f"Error reading {file}: {e}", group=get_group_label(config))
            continue
        req_cols = {'<date>', '<time>', '<open>', '<high>', '<low>', '<close>', '<volume>'}
        if not req_cols.issubset(set(df.columns)):
            continue
        df["Datetime"] = pd.to_datetime(df["<date>"] + " " + df["<time>"], errors="coerce")
        df.set_index("Datetime", inplace=True)
        df.rename(columns={"<open>": "Open", "<high>": "High", "<low>": "Low",
                           "<close>": "Close", "<volume>": "Volume"}, inplace=True)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().sort_index()
        df = df[(df.index >= config.START_DT) & (df.index <= config.END_DT)]
        df = df[df.index.map(lambda t: is_trading_time(t, config))]
        if df.empty:
            continue
        ticker = os.path.splitext(os.path.basename(file))[0]
        if ticker in config.TICKERS:
            df["Ticker"] = ticker
            all_data.append(df)
    if not all_data:
        log_message("No valid data found.", group=get_group_label(config))
        return pd.DataFrame()
    data = pd.concat(all_data).sort_index()
    data = data[~data.index.duplicated(keep="first")]
    log_message(f"Loaded {len(data)} rows.", group=get_group_label(config))
    return data

def resample_market_data(df: pd.DataFrame, freq: str, config: BaseConfig) -> pd.DataFrame:
    if df.empty:
        return df
    resampled_list = []
    for ticker in df["Ticker"].unique():
        sub_df = df[df["Ticker"] == ticker].copy()
        resampled = sub_df.resample(freq).agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
            "Ticker": "first"
        }).dropna()
        resampled_list.append(resampled)
    return pd.concat(resampled_list).sort_index()

# =============================================================================
# TECHNICAL INDICATOR FUNCTIONS
# =============================================================================

def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    d = df.copy()
    d["range1"] = d["High"] - d["Low"]
    d["range2"] = (d["High"] - d["Close"].shift(1)).abs()
    d["range3"] = (d["Low"] - d["Close"].shift(1)).abs()
    d["TR"] = d[["range1", "range2", "range3"]].max(axis=1)
    return d["TR"].ewm(alpha=1/period, adjust=False).mean()

def compute_rsi(df: pd.DataFrame, period: int) -> pd.Series:
    diff = df["Close"].diff()
    gain = diff.clip(lower=0)
    loss = -diff.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_adx(df: pd.DataFrame, period: int) -> pd.Series:
    d = df.copy()
    d["up"] = d["High"] - d["High"].shift(1)
    d["down"] = d["Low"].shift(1) - d["Low"]
    d["+DM"] = np.where((d["up"] > d["down"]) & (d["up"] > 0), d["up"], 0)
    d["-DM"] = np.where((d["down"] > d["up"]) & (d["down"] > 0), d["down"], 0)
    atr = compute_atr(d, period)
    alpha = 1/period
    d["+DI"] = 100 * d["+DM"].ewm(alpha=alpha, adjust=False).mean() / atr
    d["-DI"] = 100 * d["-DM"].ewm(alpha=alpha, adjust=False).mean() / atr
    d["DX"] = 100 * (abs(d["+DI"] - d["-DI"]) / (d["+DI"] + d["-DI"] + 1e-9))
    return d["DX"].ewm(alpha=alpha, adjust=False).mean()

def compute_macd(df: pd.DataFrame, fast=12, slow=26, signal=9):
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - sig_line
    return macd_line, sig_line, histogram

def compute_morning_fib(df: pd.DataFrame) -> dict:
    morning = df.between_time(BaseConfig.MARKET_OPEN, dtime(12, 0))
    if len(morning) < 2:
        return {}
    high = morning["High"].max()
    low = morning["Low"].min()
    rng = high - low
    if rng < BaseConfig.MIN_SWING:
        return {}
    return {lvl: low + rng * lvl for lvl in BaseConfig.FIB_LEVELS}

def calc_ema_slope(series: pd.Series, window: int = 1) -> pd.Series:
    return series.diff(window)


# =============================================================================
# DATASET BUILDING AND MODEL TRAINING FUNCTIONS
# =============================================================================

def build_training_dataset(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df["ATR"] = compute_atr(df, BaseConfig.ATR_PERIOD)
    df["RSI"] = compute_rsi(df, BaseConfig.RSI_PERIOD)
    samples = []
    groups = df.groupby([df.index.normalize(), "Ticker"])
    for (date_val, ticker), group in groups:
        fibs = compute_morning_fib(group)
        long_lvl = fibs.get(0.236)
        short_lvl = fibs.get(0.764)
        if long_lvl is None or short_lvl is None:
            continue
        for i, ts in enumerate(group.index):
            if i + BaseConfig.LABEL_OFFSET >= len(group):
                break
            row = group.loc[ts]
            fut_price = group.iloc[i + BaseConfig.LABEL_OFFSET]["Close"]
            if row["Close"] > long_lvl:
                label = 1 if (fut_price - row["Close"] > BaseConfig.PROFIT_THRESH) else 0
                samples.append({
                    "Close": row["Close"],
                    "Volume": row["Volume"],
                    "RSI": row["RSI"],
                    "ATR": row["ATR"],
                    "Label": label,
                    "Ticker": ticker,
                    "Signal": "LONG"
                })
            if row["Close"] < short_lvl:
                label = 1 if (row["Close"] - fut_price > BaseConfig.PROFIT_THRESH) else 0
                samples.append({
                    "Close": row["Close"],
                    "Volume": row["Volume"],
                    "RSI": row["RSI"],
                    "ATR": row["ATR"],
                    "Label": label,
                    "Ticker": ticker,
                    "Signal": "SHORT"
                })
    return pd.DataFrame(samples)

def train_model_xgb(train_df: pd.DataFrame):
    if train_df.empty:
        return None
    features = [col for col in train_df.columns if col not in ["Label", "Ticker", "Signal"]]
    X = train_df[features].values
    y = train_df["Label"].values
    model = XGBClassifier(
        n_estimators=BaseConfig.XGB_ESTIMATORS,
        max_depth=BaseConfig.XGB_MAX_DEPTH,
        learning_rate=BaseConfig.XGB_LR,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    )
    model.fit(X, y)
    return model


# =============================================================================
# BACKTESTING FUNCTION
# =============================================================================

def run_strategy_backtest(df: pd.DataFrame, model) -> list:
    trades = []
    if df.empty:
        return trades
    daily_groups = df.groupby([df.index.normalize(), "Ticker"])
    for (date_val, ticker), day_df in daily_groups:
        if len(day_df) < 5:
            continue
        fibs = compute_morning_fib(day_df)
        long_fib = fibs.get(0.236)
        short_fib = fibs.get(0.764)
        if long_fib is None or short_fib is None:
            continue
        day_df["ATR"] = compute_atr(day_df, BaseConfig.ATR_PERIOD)
        day_df["RSI"] = compute_rsi(day_df, BaseConfig.RSI_PERIOD)
        day_df["ADX"] = compute_adx(day_df, BaseConfig.ADX_PERIOD)
        macd_line, sig_line, macd_hist = compute_macd(day_df)
        day_df["MACD_Hist"] = macd_hist
        day_df["EMA50"] = day_df["Close"].ewm(span=BaseConfig.EMA_PERIOD, adjust=False).mean()
        day_df["EMA_Slope"] = calc_ema_slope(day_df["EMA50"], 1)
        vol_thresh = max(BaseConfig.MIN_VOLUME, day_df["Volume"].mean() * BaseConfig.VOL_FACTOR)
        positions = []
        for ts, row in day_df.iterrows():
            if ts.time() >= BaseConfig.MARKET_CLOSE:
                for pos in positions:
                    finalize_trade(trades, pos, row["Close"], ts)
                positions = []
                break
            current_price = row["Close"]
            updated_positions = []
            for pos in positions:
                if pos["side"] == "LONG":
                    new_stop = current_price - row["ATR"] * BaseConfig.ATR_MULTIPLIER
                    pos["stop"] = max(pos["stop"], new_stop)
                    if current_price >= pos["profit_target"]:
                        finalize_trade(trades, pos, current_price, ts)
                    elif current_price < pos["stop"]:
                        finalize_trade(trades, pos, pos["stop"], ts)
                    else:
                        updated_positions.append(pos)
                else:
                    new_stop = current_price + row["ATR"] * BaseConfig.ATR_MULTIPLIER
                    pos["stop"] = min(pos["stop"], new_stop)
                    if current_price <= pos["profit_target"]:
                        finalize_trade(trades, pos, current_price, ts)
                    elif current_price > pos["stop"]:
                        finalize_trade(trades, pos, pos["stop"], ts)
                    else:
                        updated_positions.append(pos)
            positions = updated_positions

            features = [current_price, row["Volume"], row["RSI"], row["ATR"]]
            ml_prob = 0.5
            if model is not None:
                try:
                    ml_prob = model.predict_proba(np.array(features).reshape(1, -1))[:, 1][0]
                except Exception:
                    ml_prob = 0.5

            if math.isnan(current_price) or math.isnan(row["ATR"]) or row["ATR"] <= 0:
                continue
            stop_dist = row["ATR"] * BaseConfig.ATR_MULTIPLIER
            denom = stop_dist * current_price
            if math.isnan(denom) or denom <= 0:
                continue

            if row["RSI"] >= BaseConfig.RSI_UPPER or row["RSI"] <= BaseConfig.RSI_LOWER:
                continue
            if row["ADX"] < BaseConfig.ADX_MIN:
                continue
            if row["Volume"] < vol_thresh:
                continue

            trade_capital = BaseConfig.ACCOUNT_SIZE * BaseConfig.RISK_PER_TRADE
            if ml_prob > BaseConfig.ML_THRESHOLD:
                trade_capital *= 1.5
            qty = math.floor(trade_capital / denom)
            if qty <= 0:
                continue

            target_long = current_price + stop_dist * BaseConfig.PROFIT_MULTIPLIER
            target_short = current_price - stop_dist * BaseConfig.PROFIT_MULTIPLIER

            if current_price > long_fib:
                pos = {"ticker": ticker, "side": "LONG", "entry": current_price, "qty": qty,
                       "stop": current_price - stop_dist, "profit_target": target_long, "time": ts}
                positions.append(pos)
            if current_price < short_fib:
                pos = {"ticker": ticker, "side": "SHORT", "entry": current_price, "qty": qty,
                       "stop": current_price + stop_dist, "profit_target": target_short, "time": ts}
                positions.append(pos)
        if positions:
            final_price = day_df.iloc[-1]["Close"]
            final_time = day_df.index[-1]
            for pos in positions:
                finalize_trade(trades, pos, final_price, final_time)
    return trades

def finalize_trade(trade_list: list, pos: dict, exit_price: float, exit_time: datetime):
    entry_price = pos["entry"]
    qty = pos["qty"]
    side = pos["side"]
    adj_entry = entry_price + (entry_price * BaseConfig.SLIPPAGE if side == "LONG" else -entry_price * BaseConfig.SLIPPAGE)
    adj_exit = exit_price + (exit_price * BaseConfig.SLIPPAGE if side == "LOGN" else -exit_price * BaseConfig.SLIPPAGE)
    profit = (adj_exit - adj_entry) * qty if side == "LONG" else (adj_entry - adj_exit) * qty
    net_profit = profit - (entry_price * qty * BaseConfig.COMMISSION + exit_price * qty * BaseConfig.COMMISSION)
    trade_list.append({"Ticker": pos["ticker"], "side": side, "EntryTime": pos["time"],
                       "EntryPrice": entry_price, "ExitTime": exit_time, "ExitPrice": exit_price,
                       "Quantity": qty, "GrossPnL": round(profit, 2), "NetPnL": round(net_profit, 2)})


# =============================================================================
# PERFORMANCE EVALUATION FUNCTIONS
# =============================================================================

def evaluate_perf(trades: list) -> dict:
    if not trades:
        return {"net_pl": 0, "trades": 0, "win_rate": 0, "avg_pnl": 0,
                "max_dd": 0, "sharpe": 0, "profit_factor": 0, "return": 0}
    df = pd.DataFrame(trades).sort_values("ExitTime")
    pnl = df["NetPnL"].values
    tot = len(pnl)
    net_pl = pnl.sum()
    wins = np.sum(pnl > 0)
    win_rate = (wins / tot) * 100
    avg_pnl = np.mean(pnl)
    cum = np.cumsum(pnl)
    max_dd = np.max(np.maximum.accumulate(cum) - cum)
    std_pnl = np.std(pnl, ddof=1)
    sharpe = (avg_pnl / std_pnl) * math.sqrt(tot) if std_pnl > 1e-9 else 0
    gains = np.sum(pnl[pnl > 0])
    losses = -np.sum(pnl[pnl < 0])
    profit_factor = gains / losses if losses > 1e-9 else 0
    ret_pct = (net_pl / BaseConfig.ACCOUNT_SIZE) * 100
    return {"net_pl": round(net_pl, 2), "trades": tot, "win_rate": round(win_rate, 2),
             "avg_pnl": round(avg_pnl, 2), "max_dd": round(max_dd, 2),
            "sharpe": round(sharpe, 3), "profit_factor": round(profit_factor, 3),
            "return": round(ret_pct, 2)}

def compute_stats(df: pd.DataFrame, model) -> dict:
    trades = run_strategy_backtest(df, model)
    return evaluate_perf(trades)


# =============================================================================
# WALK-FORWARD EVALUATION AND OPTIMIZATION FUNCTIONS
# =============================================================================

def generate_segments(start_dt, end_dt, train_m, test_m):
    segments = []
    current = start_dt
    while True:
        train_end = current + pd.offsets.MonthEnd(train_m)
        test_start = train_end + timedelta(days=1)
        test_end = test_start + pd.offsets.MonthEnd(test_m)
        if test_end > end_dt:
            break
        segments.append((current, train_end, test_start, test_end))
        current = test_start
    return segments

def walk_forward_eval(df: pd.DataFrame, config: BaseConfig) -> dict:
    segments = generate_segments(config.START_DT, config.END_DT, config.TRAIN_MONTHS, config.TEST_MONTHS)
    results = []
    for i, (train_start, train_end, test_start, test_end) in enumerate(segments):
        log_message(f"Segment {i}: Train [{train_start.date()} -> {train_end.date()}], Test [{test_start.date()} -> {test_end.date()}]")
        train_data = df.loc[(df.index >= train_start) & (df.index <= train_end)]
        if train_data.empty:
            log_message("No training data; skip segment.")
            continue
        train_ds = build_training_dataset(train_data)
        if len(train_ds) < 10:
            log_message("Insufficient training samples; skip segment.")
            continue
        model = train_model_xgb(train_ds)
        if model is None:
            log_message("Model training failed; skip segment.")
            continue
        test_data = df.loc[(df.index >= test_start) & (df.index <= test_end)]
        if test_data.empty:
            log_message("No testing data; skip segment.")
            continue
        seg_stats = compute_stats(test_data, model)
        seg_stats.update({"Segment": i, "TrainStart": train_start, "TrainEnd": train_end,
                          "TestStart": test_start, "TestEnd": test_end})
        results.append(seg_stats)
        log_message(f"Segment {i} stats: {seg_stats}")
    if not results:
        return {}
    res_df = pd.DataFrame(results)
    summary = {"net_pl": res_df["net_pl"].sum(),
                "trades": res_df["trades"].sum(),
               "win_rate": res_df["win_rate"].mean(),
               "avg_pnl": res_df["avg_pnl"].mean(),
               "max_dd": res_df["max_dd"].max(),
               "sharpe": res_df["sharpe"].mean(),
               "profit_factor": res_df["profit_factor"].mean(),
               "return": res_df["return"].mean()}
    return summary

def make_opt_objective(config: BaseConfig, df: pd.DataFrame):
    def opt_objective(params):
        config.ATR_MULTIPLIER = params["atr_mult"]
        config.MIN_SWING = params["swing_min"]
        config.ML_THRESHOLD = params["ml_prob"]
        config.RSI_UPPER = params["rsi_upper"]
        config.RSI_LOWER = params["rsi_lower"]
        config.ADX_MIN = params["adx_filter"]
        config.PROFIT_MULTIPLIER = params["profit_mult"]
        perf = walk_forward_eval(df, config)
        if not perf:
            return {"loss": 9999, "status": STATUS_OK, "params": params, "sharpe": 0, "netpl": 0}
        loss_val = - (perf.get("sharpe", 0) + 0.0001 * perf.get("net_pl", 0))
        return {"loss": loss_val, "status": STATUS_OK, "params": params,
                "sharpe": perf.get("sharpe", 0), "netpl": perf.get("net_pl", 0)}
    return opt_objective

def run_optimization(df: pd.DataFrame, config: BaseConfig) -> (dict, dict):
    space = {
        "atr_mult": hp.uniform("atr_mult", 0.5, 2.5),
        "swing_min": hp.uniform("swing_min", 0.005, 0.05),
        "ml_prob": hp.uniform("ml_prob", 0.7, 0.95),
        "adx_filter": hp.choice("adx_filter", [15, 20, 25, 30]),
        "rsi_upper": hp.uniform("rsi_upper", 80, 100),
        "rsi_lower": hp.uniform("rsi_lower", 0, 20),
        "profit_mult": hp.uniform("profit_mult", 1.0, 3.0)
    }
    trials = Trials()
    opt_fn = make_opt_objective(config, df)
    best = fmin(fn=opt_fn, space=space, algo=tpe.suggest,
                max_evals=config.MAX_OPT_EVALS, trials=trials)
    log_message(f"Hyperopt raw best: {best}")
    losses = [t["result"]["loss"] for t in trials.trials]
    best_index = np.argmin(losses)
    best_result = trials.trials[best_index]["result"]
    log_message(f"Optimized Params: {best_result['params']}")
    log_message(f"Optimized Sharpe: {best_result['sharpe']}, NetPL: {best_result['netpl']}")
    final_summary = walk_forward_eval(df, config)
    log_message(f"Final Walk-Forward Performance: {final_summary}")
    with open('best_params.json', 'w') as f:
        json.dump(best_result["params"], f)
    return best_result["params"], final_summary


# =============================================================================
# LIVE TRADING SIMULATION (PAPER TRADING)
# =============================================================================

def load_oos_data(config: BaseConfig) -> pd.DataFrame:
    log_message("Loading Out-of-Sample Data from OOS path...", group=get_group_label(config))
    oos_config = copy.deepcopy(config)
    oos_config.START_DT = config.OOS_START
    oos_config.END_DT = config.OOS_END
    return load_csv_data(config.OOS_PATH, oos_config)

def connect_to_live_feed(config: BaseConfig):
    data = load_oos_data(config)
    market_data = resample_market_data(data, config.RESAMPLE_FREQ, config)
    log_message("Starting live feed simulation from out-of-sample data...", group=get_group_label(config))
    def live_generator(df: pd.DataFrame):
        for ts, row in df.iterrows():
            yield ts, row
            time.sleep(1)  # simulate a 1-second delay
    return live_generator(market_data)

def start_live_paper_trading(live_feed, params: dict, config: BaseConfig):
    config.ATR_MULTIPLIER = params["atr_mult"]
    config.MIN_SWING = params["swing_min"]
    config.ML_THRESHOLD = params["ml_prob"]
    group_label = get_group_label(config)
    log_message("Commencing paper trading simulation...", group=group_label)
    trades = []
    for ts, row in live_feed:
        log_message(f"Live bar at {ts}: Close = {row['Close']}", group=group_label)
        # Here we would insert real-time signal generation and trade management logic.
    performance = evaluate_perf(trades)
    log_message(f"Paper Trading Performance Metrics: {performance}", group=group_label)
    return trades, performance


# =============================================================================
# MAIN EXECUTION FOR MULTI-GROUP TRADING
# =============================================================================

def main():
    # ========================= Group A In-Sample =========================
    log_message("=== Group A: In-Sample Optimization ===", group="Group A")
    data_A = load_csv_data(GroupAConfig.DATA_PATH, GroupAConfig)
    if data_A.empty:
        log_message("Group A: In-sample data not available.", group="Group A")
    else:
        bars_A = resample_market_data(data_A, GroupAConfig.RESAMPLE_FREQ, GroupAConfig)
        params_A, stats_A = run_optimization(bars_A, GroupAConfig)
        log_message(f"Group A Optimized Parameters: {params_A}", group="Group A")
        log_message(f"Group A In-Sample Performance: {stats_A}", group="Group A")

        # ------------------------- Group A Out-of-Sample -------------------------
        log_message("=== Group A: Out-of-Sample Evaluation ===", group="Group A")
        # we use load_oos data() to adjust the data range to OOS
        oos_data_A = load_oos_data(GroupAConfig)
        if oos_data_A.empty:
            log_message("Group A: OOS data not available.", group="Group A")
        else:
            oos_bars_A = resample_market_data(oos_data_A, GroupAConfig.RESAMPLE_FREQ, GroupAConfig)
            oos_performance_A = compute_stats(oos_bars_A, params_A)
            log_message(f"Group A OOS Performance: {oos_performance_A}", group="Group A")

    # ========================= Group B In-Sample =========================
    log_message("=== Group B: In-Sample Optimization ===", group="Group B")
    data_B = load_csv_data(GroupBConfig.DATA_PATH, GroupBConfig)
    if data_B.empty:
        log_message("Group B: In-Sample data not available.", group="Group B")
    else:
        bars_B = resample_market_data(data_B, GroupBConfig.RESAMPLE_FREQ, GroupBConfig)
        params_B, stats_B = run_optimization(bars_B, GroupBConfig)
        log_message(f"Group B Optimized Parameters: {params_B}", group="Group B")
        log_message(f"Group B In-Sample Performance: {stats_B}", group="Group B")

        # ------------------------- Group B Out-of-Sample -------------------------
        log_message("=== Group B: Out-of-Sample Evaluation ===", group="Group B")
        # We use load_oos_data() for OOS data, which adjusts the data range
        oos_data_B = load_oos_data(GroupBConfig)
        if oos_data_B.empty:
            log_message("Group B: OOS data not available.", group="Group B")
        else:
            oos_bars_B = resample_market_data(oos_data_B, GroupBConfig.RESAMPLE_FREQ, GroupBConfig)
            oos_performance_B = compute_stats(oos_bars_B, params_B)
            log_message(f"Group B OOS Performance: {oos_performance_B}", group="Group B")

    # --------------------- Live Trading Simulation ---------------------
    if (not data_A.empty) and (not data_B.empty):
        log_message("=== Starting Combined Live Trading Simulation ===")
        live_feed_A = connect_to_live_feed(GroupAConfig)
        live_feed_B = connect_to_live_feed(GroupBConfig)
        thread_A = threading.Thread(target=start_live_paper_trading, args=(live_feed_A, params_A, GroupAConfig))
        thread_B = threading.Thread(target=start_live_paper_trading, args=(live_feed_B, params_B, GroupBConfig))
        thread_A.start()
        thread_B.start()
        thread_A.join()
        thread_B.join()
    else:
        log_message("Insufficient data for live trading simulation.")

if __name__ == "__main__":
    main()