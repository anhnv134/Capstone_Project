#Algorithm 7 Version 11

#import necessary libraries
import os
import glob
import math
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, time, timedelta
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import warnings
warnings.filterwarnings("ignore")

#####################################
# CONFIGURE LOGGING
#####################################
LOG_DIR = "../test/algo_7_11"
# Log file path
LOG_FILE = os.path.join(LOG_DIR, "Algo7_version11_logfile_9.txt")

def log_message(message, log_file=LOG_FILE):
    """
    Logs a message with a timestamp to the specified log file and prints it.

    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{timestamp}] {message}"
    print(msg)
    try:
        with open(log_file, 'a') as f:
            f.write(msg + "\n")
    except FileNotFoundError:
        print(f"ERROR: Log file directory does not exist: {os.path.dirname(log_file)}")
    except Exception as e:
        print(f"ERROR: Failed to write to log file. Exception: {e}")

##############################################################################
# GLOBAL CONFIGURATION , PTHS FOR MACHINE LEARNING AND BACKTEST
##############################################################################
DATA_FOLDER_2021    = "../2021_11Tickers"
#DATA_FOLDER_2022    = "/content/drive/My Drive/2022"
MODEL_SAVE_PATH     = "../colab/Algorithm_7/fib_ml_model_1.pkl"

START_DATE          = pd.to_datetime("2021-01-01")
END_DATE            = pd.to_datetime("2021-12-31")
TRAIN_CUTOFF        = pd.to_datetime("2021-07-01")

SESSION_START       = time(9, 15)
SESSION_END         = time(15, 35)
SKIP_OPEN_MINS      = 30
SKIP_CLOSE_MINS     = 5
LUNCH_START         = time(12, 0)
LUNCH_END           = time(13, 0)

FIB_RATIOS          = [0.382, 0.5, 0.618, 1.0, 1.272, 1.618, 2.0]
MIN_MORNING_SWING   = 0.6
RSI_PERIOD          = 14
MACD_FAST           = 12
MACD_SLOW           = 26
MACD_SIGNAL         = 9
ADX_PERIOD          = 14
ATR_PERIOD          = 14
EMA_PERIOD_50       = 50
VOLUME_FACTOR       = 1.5
LABEL_LOOKAHEAD_BARS= 7 #10
PROFIT_THRESHOLD    = 0.0

BACKTEST_START_DATE = pd.to_datetime("2021-01-04")
BACKTEST_END_DATE   = pd.to_datetime("2021-12-31")

##############################################################################
# Algorithm 7 parameters
##############################################################################
PARAMS = {

    'min_volume': 20000,
    'min_atr': 0.8,
    'min_adx': 30,
    'bar_interval_5m': '5min',
    'bar_interval_15m': '15min',
    'adx_15m_threshold': 45,
    'adx_filter_threshold': 45,
    'multi_bar_confirm': 3,
    'min_morning_swing': 0.6,
    'fib_ratios': [0.382, 0.5, 0.618, 1.0, 1.272, 1.618, 2.0],
    'atr_stop_multiplier': 1.8,
    'rsi_upper_bound': 65,
    'rsi_lower_bound': 35,
    'macd_fast_period': 12,
    'macd_slow_period': 26,
    'macd_signal_period': 9,
    'volume_factor': 1.5,
    'risk_per_trade_fraction': 0.01,
    'assumed_account_size': 100000,
    'partial_exit_levels': [0.618, 1.0, 1.272, 1.618],
    'partial_exit_ratio': 0.8,

    'avoid_new_trades_after': time(14, 0),
    'session_end_time': SESSION_END,
    'skip_open_mins': SKIP_OPEN_MINS,
    'skip_close_mins': SKIP_CLOSE_MINS,
    'lunch_start': LUNCH_START,
    'lunch_end': LUNCH_END,

    'ema_period': 50,
    'min_ema_slope': 0.0,

    'slippage_perc': 0.0001,
    'commission_perc': 0.0005,
    'max_positions': 5,

    'use_higher_timeframe_ema': True,
    'daily_ema_period': 20,
    'strong_signal_adx_threshold': 45,
    'strong_signal_volume_factor': 2.0
}
#############################################################################
# Time Filter
##############################################################################


def time_filter_ok(dt: pd.Timestamp) -> bool:
    t = dt.time()

    if PARAMS['lunch_start'] <= t < PARAMS['lunch_end']:
        return False


    open_t = dt.replace(hour=9, minute=15, second=0, microsecond=0)
    if dt < open_t + timedelta(minutes=PARAMS['skip_open_mins']):
        return False

    # Avoid new trades after 14:00
    if t >= time(14, 0):
        return False

    close_t = dt.replace(hour=15, minute=35, second=0, microsecond=0)
    if dt > close_t - timedelta(minutes=PARAMS['skip_close_mins']):
        return False

    return True
##############################################################################
# Recursive Folder Loader
##############################################################################

def load_csv_folder_recursive(folder_path: str) -> pd.DataFrame:
    log_message(f"Scanning folder: {folder_path}")
    if not os.path.isdir(folder_path):

        log_message(f"Directory not found: {folder_path}")
        return pd.DataFrame()

    all_items = sorted(os.listdir(folder_path))
    log_message(f"Items found in {folder_path}: {all_items}")

    subfolders = []
    for item in all_items:
        path_sub = os.path.join(folder_path, item)
        if os.path.isdir(path_sub) and ("Cash Data" in item):
            subfolders.append(path_sub)
    log_message(f"found {len(subfolders)} subfolders that match 'Cash Data' in {folder_path}.")


    df_parts = []
    for sf in subfolders:
        csv_list = glob.glob(os.path.join(sf, "*.csv"))
        log_message(f"Subfolder '{sf}': found {len(csv_list)} CSV file(s).")


        if not csv_list:
            continue

        for cfile in csv_list:
            try:
                dftmp = pd.read_csv(cfile)
            except Exception as e:
                log_message(f"Error reading {cfile}: {e}")
                continue
            needed_cols = {'<date>', '<time>', '<open>', '<high>', '<low>', '<close>', '<volume>'}
            if not needed_cols.issubset(dftmp.columns):
                log_message(f"Missing columns in {cfile}, skipping.")
                continue

            dftmp['DateTime'] = pd.to_datetime(dftmp['<date>'] + " " + dftmp['<time>'], errors='coerce')
            dftmp.set_index('DateTime', inplace=True)
            dftmp.sort_index(inplace=True)

            rename_map = {
                '<open>': 'Open',
                '<high>': 'High',
                '<low>':  'Low',
                '<close>':'Close',
                '<volume>':'Volume'
            }
            dftmp.rename(columns=rename_map, inplace=True)
            dftmp = dftmp[['Open','High','Low','Close','Volume']].dropna()
            dftmp = dftmp[~dftmp.index.duplicated(keep='first')]

            dftmp = dftmp.loc[(dftmp.index>= START_DATE) & (dftmp.index<= END_DATE)]
            dftmp = dftmp[dftmp.index.map(time_filter_ok)]
            if not dftmp.empty:
                df_parts.append(dftmp)

    if not df_parts:
        log_message(f"No CSV data found in subfolders of {folder_path}")
        return pd.DataFrame()

    merged = pd.concat(df_parts).sort_index()
    merged = merged[~merged.index.duplicated(keep='first')]
    rowcount = len(merged)
    log_message(f"Loaded {rowcount} rows from subfolders under: {folder_path}")
    return merged



###########################################################
# RESAMPLE => 5min for ML Above is gpt , below is me
############################################################
def resample_5m_bars_for_ml(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
      return df


    agg_spec = {
        'Open': 'first',
        'High': 'max',
        'Low' : 'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    df_5m = df.resample('5T').agg(agg_spec).dropna()
    df_5m = df_5m.between_time(SESSION_START, SESSION_END)

    return df_5m

##############################################################################
# INDICATORS FOR MACHINE LEARNING
##############################################################################

def wilder_atr_ml(df: pd.DataFrame, period=14)->pd.Series: #We use adjust=False to replicate Wilder's smoothing as it was originally defined

    df = df.copy()
    df['H-L'] = df['High'] - df['Low']
    df['H-Cp'] = (df['High'] - df['Close'].shift(1)).abs()
    df['L-Cp'] = (df['Low']  - df['Close'].shift(1)).abs()
    df['TR'] = df[['H-L','H-Cp','L-Cp']].max(axis=1)
    atr = df['TR'].ewm(alpha=1.0/period, adjust=False).mean()
    return atr

def compute_adx_ml(df: pd.DataFrame,  period=14)->pd.Series:

    df = df.copy()
    df['upMove']   = df['High'] - df['High'].shift(1)
    df['downMove'] = df['Low'].shift(1) - df['Low']
    df['+DM'] = np.where((df['upMove'] > df['downMove']) & (df['upMove'] > 0), df['upMove'], 0)
    df['-DM'] = np.where((df['downMove'] > df['upMove']) & (df['downMove'] > 0), df['downMove'], 0)

    df['ATR'] = wilder_atr(df[['High','Low','Close']], period=period)

    alpha = 1.0 / period
    df['+DI'] = 100 * df['+DM'].ewm(alpha=alpha).mean() / df['ATR']
    df['-DI'] = 100 * df['-DM'].ewm(alpha=alpha).mean() / df['ATR']

    df['DX'] = 100 * ( (df['+DI'] - df['-DI']).abs() / (df['+DI'] + df['-DI'] + 1e-9) )
    df['ADX'] = df['DX'].ewm(alpha=alpha).mean()
    return df['ADX']

def compute_rsi_ml(df: pd.DataFrame,  period=14)->pd.Series:

    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = (-1 * delta.clip(upper=0))
    ema_up = up.ewm(alpha=1/period, adjust=False).mean()
    ema_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ema_up / (ema_down + 1e-9)
    rsi = 100 - (100 / (1+rs))
    return rsi




def compute_macd_ml(df: pd.DataFrame,  fast=12, slow=26, signal=9):

    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist

def get_morning_swing_ml(df_day: pd.DataFrame):
    if df_day.empty:
        return None
    s_t = df_day.index[0].replace(hour=9, minute=30)
    c_t = df_day.index[0].replace(hour=11, minute=30)
    sub = df_day.between_time(s_t.time(), c_t.time())
    if sub.empty:
        return None
    hi_idx = sub['High'].idxmax()
    lo_idx = sub['Low'].idxmin()
    if pd.isna(hi_idx) or pd.isna(lo_idx):
        return None
    return (sub.loc[lo_idx,'Low'], sub.loc[hi_idx,'High'], lo_idx, hi_idx)


def compute_fib_levels_for_day_ml(df_day: pd.DataFrame)->dict:
    info = get_morning_swing_ml(df_day)
    if not info:
        return {}
    low_val, high_val, _, _ = info
    rng = high_val - low_val
    if rng< MIN_MORNING_SWING:
        return {}
    fib_dict= {}
    up_swing= (low_val< high_val)
    for r in FIB_RATIOS:
        if up_swing:
            fib_dict[r] = low_val + rng*r
        else:
            fib_dict[r] = high_val - rng*r
    return fib_dict

##############################################################################
# BUILD DATASET (INCLUDES BOTH LONG AND SHORT)
##############################################################################

def build_labeled_dataset(df_5m: pd.DataFrame)->pd.DataFrame:
    """
    Creates labeled samples for both LONG and SHORT signals.
      - LONG: (Close>fib_618 and ADX>30)
      - SHORT: (Close<fib_382 and ADX>30)
      Then checks 10 bars ahead to see if the move was profitable for that direction.
    """

    if df_5m.empty:
        return pd.DataFrame()

    df_5m['ATR'] = wilder_atr_ml(df_5m, ATR_PERIOD)
    df_5m['ADX'] = compute_adx_ml(df_5m, ADX_PERIOD)
    mfast, mslow, msignal = compute_macd_ml(df_5m, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    df_5m['MACD_Line']   = mfast
    df_5m['MACD_Signal'] = mslow
    df_5m['MACD_Hist']   = msignal
    df_5m['RSI']         = compute_rsi_ml(df_5m, RSI_PERIOD)
    df_5m['EMA_50']      = df_5m['Close'].ewm(span=EMA_PERIOD_50, adjust=False).mean()


    samples = []
    days_list = sorted(df_5m.index.normalize().unique())
    for day in days_list:
        day_df = df_5m[df_5m.index.normalize()== day]
        if day_df.empty:
            continue


        fib_info= compute_fib_levels_for_day_ml(day_df)
        if not fib_info:
            continue

        fib_618= fib_info.get(0.618, None)
        fib_382= fib_info.get(0.382, None)

        bar_list= day_df.index
        for i, tstamp in enumerate(bar_list):
            row= day_df.loc[tstamp]
            fut_idx= i + LABEL_LOOKAHEAD_BARS
            if fut_idx>= len(bar_list):
                break
            future_tstamp= bar_list[fut_idx]
            future_close = day_df.loc[future_tstamp, 'Close']

            # LONG secnario
            if fib_618 and (row['Close']> fib_618) and (row['ADX']> 45):#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

                feats_long= {
                    'Close': row['Close'],
                    'Volume': row['Volume'],
                    'RSI': row['RSI'],
                    'MACD_Line': row['MACD_Line'],
                    'MACD_Signal': row['MACD_Signal'],
                    'ADX': row['ADX'],
                    'ATR': row['ATR'],
                    'EMA_50': row['EMA_50'],
                    'signal_type': 'LONG'
                }
                pl_long= future_close - row['Close']
                feats_long['label'] = 1 if (pl_long> PROFIT_THRESHOLD) else 0
                samples.append(feats_long)


            #SHORT scenario
            if fib_382 and (row['Close']< fib_382) and (row['ADX']> 45):
                feats_short= {
                    'Close': row['Close'],
                    'Volume': row['Volume'],
                    'RSI': row['RSI'],
                    'MACD_Line': row['MACD_Line'],
                    'MACD_Signal': row['MACD_Signal'],
                    'ADX': row['ADX'],
                    'ATR': row['ATR'],
                    'EMA_50': row['EMA_50'],
                    'signal_type': 'SHORT'

                }
                pl_short= row['Close'] - future_close  # profit if price goes down
                feats_short['label']= 1 if (pl_short> PROFIT_THRESHOLD) else 0
                samples.append(feats_short)

    return pd.DataFrame(samples)

def train_and_save_model(df_samples: pd.DataFrame):
    if df_samples.empty:
        log_message("No samples found. Cannot train a model.")
        return

    # We exclude signal_type from the features
    feat_cols= [c for c in df_samples.columns if c not in ['label','signal_type']]
    X= df_samples[feat_cols].values
    y= df_samples['label'].values

    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)
    model= XGBClassifier(
        n_estimators=180,
        max_depth=5,
        learning_rate=0.04,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    preds= model.predict(X_test)
    probs= model.predict_proba(X_test)[:,1]
    acc= accuracy_score(y_test, preds)
    try:
        auc= roc_auc_score(y_test, probs)
    except:
        auc= 0.0

    log_message(f"ML Train Accuracy: {acc:.4f}")
    log_message(f"ML Train ROC AUC:  {auc:.4f}")
    log_message("ML Classification Report (Train Split):")
    log_message(classification_report(y_test, preds))

    joblib.dump(model, MODEL_SAVE_PATH)
    log_message(f"Model saved to {MODEL_SAVE_PATH}")

def quick_out_of_sample_test(df_5m_test: pd.DataFrame):
    if df_5m_test.empty:
        log_message("Test data is empty, skipping out-of-sample check.")
        return

    test_samples= build_labeled_dataset(df_5m_test)
    if test_samples.empty:
        log_message("No labeled test samples in out-of-sample data, skipping.")
        return

    try:
        model= joblib.load(MODEL_SAVE_PATH)
    except Exception as e:
        log_message(f"Could not load model file: {e}")
        return

    feat_cols= [c for c in test_samples.columns if c not in ['label','signal_type']]
    X_test= test_samples[feat_cols].values
    y_test= test_samples['label'].values

    pred_labels= model.predict(X_test)
    try:
        pred_probs= model.predict_proba(X_test)[:,1]
    except:
        pred_probs = np.zeros(len(pred_labels))

    acc= accuracy_score(y_test, pred_labels)
    try:
        auc= roc_auc_score(y_test, pred_probs)
    except:
        auc= 0.0

    log_message(f"Out-of-sample Accuracy: {acc:.4f}")
    log_message(f"Out-of-sample ROC AUC:  {auc:.4f}")
    log_message("Out-of-sample Classification Report:")
    log_message(classification_report(y_test, pred_labels))

##############################################################################
# BACKTEST
##############################################################################

def resample_bars(df: pd.DataFrame, interval: str) -> pd.DataFrame:

    if df.empty:
        return df
    agg_spec = {
        'Open': 'first',
        'High':   'max',
        'Low':    'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    df_res = df.resample(interval).agg(agg_spec).dropna()
    df_res = df_res.between_time(time(9,15), time(15,35))
    return df_res

def compute_daily_ema_filter(df_intra: pd.DataFrame, ema_period=20) -> pd.DataFrame:
    df_daily = df_intra.resample('1D').agg({
        'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'
    }).dropna()
    if df_daily.empty:
        return pd.DataFrame()
    df_daily['DailyEMA'] = df_daily['Close'].ewm(span=ema_period, adjust=False).mean()
    df_daily_intrp = df_daily[['DailyEMA']].reindex(df_intra.index, method='ffill')
    return df_daily_intrp



def wilder_atr(df: pd.DataFrame, period=14,)->pd.Series: #We use adjust=False to replicate Wilder's smoothing as it was originally defined

    df = df.copy()
    df['H-L'] = df['High'] - df['Low']
    df['H-Cp'] = (df['High'] - df['Close'].shift(1)).abs()
    df['L-Cp'] = (df['Low']  - df['Close'].shift(1)).abs()
    df['TR'] = df[['H-L','H-Cp','L-Cp']].max(axis=1)
    atr = df['TR'].ewm(alpha=1.0/period, adjust=False).mean()
    return atr


def compute_adx(df: pd.DataFrame, period=14)->pd.Series:

    df = df.copy()
    df['upMove']   = df['High'] - df['High'].shift(1)
    df['downMove'] = df['Low'].shift(1) - df['Low']
    df['+DM'] = np.where((df['upMove'] > df['downMove']) & (df['upMove'] > 0), df['upMove'], 0)
    df['-DM'] = np.where((df['downMove'] > df['upMove']) & (df['downMove'] > 0), df['downMove'], 0)

    df['ATR'] = wilder_atr(df[['High','Low','Close']], period=period)

    alpha = 1.0 / period
    df['+DI'] = 100 * df['+DM'].ewm(alpha=alpha).mean() / df['ATR']
    df['-DI'] = 100 * df['-DM'].ewm(alpha=alpha).mean() / df['ATR']

    df['DX'] = 100 * ( (df['+DI'] - df['-DI']).abs() / (df['+DI'] + df['-DI'] + 1e-9) )
    df['ADX'] = df['DX'].ewm(alpha=alpha).mean()
    return df['ADX']
def compute_macd(df: pd.DataFrame, fast=12, slow=26, signal=9):

    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist

def compute_rsi(df: pd.DataFrame, period=14)->pd.Series:

    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(alpha=1/period, adjust=False).mean()
    ema_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ema_up / (ema_down + 1e-9)
    rsi = 100 - (100 / (1+rs))
    return rsi

def compute_asi(df: pd.DataFrame, limit_move=None)->pd.Series:

    df = df.copy()

    # Shifted values for previous day
    df['Close_prev'] = df['Close'].shift(1)
    df['Open_prev']  = df['Open'].shift(1)
    df['High_prev']  = df['High'].shift(1)
    df['Low_prev']   = df['Low'].shift(1)

    # Default limit_move to 1 if not provided
    if limit_move is None:
        limit_move = 1.0

    # K factor: largest of (|High - ClosePrev|, |Low - ClosePrev|)
    df['K'] = np.maximum(
        (df['High'] - df['Close_prev']).abs(),
        (df['Low'] - df['Close_prev']).abs()
    )

    # 1) M (sign-based approach)
    def wilder_m(row):

        C = row['Close']
        Cp = row['Close_prev']
        O = row['Open']
        # If today's Close is >= yesterday's Close
        if (C >= Cp):
            return (C - Cp) + 0.5 * (C - O) + 0.25 * (Cp - O)
        else:
            return (C - Cp) - 0.5 * (C - O) - 0.25 * (Cp - O)

    df['M'] = df.apply(wilder_m, axis=1)

    # 2) R (piecewise)
    def wilder_r(row):

        H = row['High']
        L = row['Low']
        Cp = row['Close_prev']
        O = row['Open']

        move_up = (H - Cp)
        move_down = (Cp - L)

        # Condition 1
        if move_up > move_down and move_up > 0:
            R = (H - Cp) - 0.5*(Cp - L) + 0.25*(Cp - O)
        # Condition 2
        elif move_down > move_up and move_down > 0:
            R = (Cp - L) - 0.5*(H - Cp) + 0.25*(Cp - O)
        else:
            # Default
            R = (H - L) + 0.25*(Cp - O)

        # Avoid division-by-zero
        if R == 0:
            R = 1e-10  # small epsilon
        return R

    df['R'] = df.apply(wilder_r, axis=1)

    # 3) SI = 50 * (M / R) * (K / limit_move)
    df['SI'] = 50.0 * (df['M'] / df['R']) * (df['K'] / limit_move)

    # 4) ASI = cumulative sum of SI
    df['ASI'] = df['SI'].cumsum().fillna(0)

    # Return the ASI as a Pandas Series
    return df['ASI']

def detect_hsp_lsp(asi_series: pd.Series, window=3):
    n= len(asi_series)
    is_hsp= [False]*n
    is_lsp= [False]*n
    for i in range(n):
        if i< window or i>= (n- window):
            continue
        val= asi_series.iloc[i]
        left_chunk= asi_series.iloc[i- window:i]
        right_chunk= asi_series.iloc[i+1: i+1+ window]
        if val> left_chunk.min() and val> right_chunk.min():
            is_hsp[i]= True
        if val< left_chunk.min() and val< right_chunk.min():
            is_lsp[i]= True
    return pd.Series(is_hsp, index=asi_series.index), pd.Series(is_lsp, index=asi_series.index)


def get_morning_swing(df_day: pd.DataFrame):
    if df_day.empty:
        return None
    s_t = df_day.index[0].replace(hour=9, minute=30)
    c_t = df_day.index[0].replace(hour=11, minute=30)
    sub = df_day.between_time(s_t.time(), c_t.time())
    if sub.empty:
        return None
    hi_idx = sub['High'].idxmax()
    lo_idx = sub['Low'].idxmin()
    if pd.isna(hi_idx) or pd.isna(lo_idx):
        return None
    return (sub.loc[lo_idx,'Low'], sub.loc[hi_idx,'High'], lo_idx, hi_idx)

def compute_fib_levels_for_day(df_day: pd.DataFrame)->dict:
    info= get_morning_swing(df_day)
    if not info:
        return {}
    m_low, m_high, lo_t, hi_t= info
    rng= m_high- m_low
    if rng< PARAMS['min_morning_swing']:
        return {}
    up_swing= (lo_t< hi_t)
    fib_d= {}
    for r in PARAMS['fib_ratios']:
        if up_swing:
            fib_d[r]= m_low+ rng*r
        else:
            fib_d[r]= m_high- rng*r
    fib_d['up_swing']= up_swing
    fib_d['swing_low']= m_low
    fib_d['swing_high']= m_high
    return fib_d



def extract_trading_days(df: pd.DataFrame)->list:
    return sorted(df.index.normalize().unique())

def compute_transaction_cost(notional: float, slip: float, comm: float)->float:
    return notional*(slip+ comm)

def record_trade(trades_list: list, pos_dict: dict, exit_price: float, exit_time: pd.Timestamp):
    entry_px= pos_dict['entry_price']
    qty     = pos_dict['quantity']
    side    = pos_dict['side']
    notional= entry_px* qty
    if side=='LONG':
        pl= (exit_price- entry_px)* qty
    else:
        pl= (entry_px- exit_price)* qty

    e_cost= compute_transaction_cost(notional, PARAMS['slippage_perc'], PARAMS['commission_perc'])
    x_cost= compute_transaction_cost(exit_price* qty, PARAMS['slippage_perc'], PARAMS['commission_perc'])
    net_pl= pl- (e_cost+ x_cost)

    trades_list.append({
        'instrument': pos_dict['instrument'],
        'side': side,
        'entry_time': pos_dict['entry_time'],
        'exit_time': exit_time,
        'entry_price': entry_px,
        'exit_price': exit_price,
        'quantity': qty,
        'gross_pl': pl,
        'net_pl': net_pl
    })

##############################################################################
# BACKTEST - CALLS WITH MACHINE LEARNING MODEL
##############################################################################

def run_backtest_for_instrument(df_5m: pd.DataFrame,
                                df_15m: pd.DataFrame,
                                instrument_name: str,
                                trades_out: list,
                                df_daily_ema: pd.DataFrame=None,
                                model=None):
    """
    Bar-by-bar 'Algorithm 7' backtest on a single instrument.It also calls the Machine
    Learning model to confirm whether each bar is predicted profitable.
    If the model says '0' => we skip the trade, even if the rule-based logic says OK.
    """
    if df_5m.empty or df_15m.empty:
        return
    if model is None:
        log_message("No ML model provided: skipping ML confirmation.")

        # We'll proceed with the old rules only, or we can return.

    # Basic indicators on 5-min
    df_5m['ATR']= wilder_atr(df_5m[['High','Low','Close']], period=14)
    df_5m['ADX']= compute_adx(df_5m[['High','Low','Close']], period=14)
    m_line, m_signal, m_hist= compute_macd(df_5m, 12, 26, 9)
    df_5m['MACD_Hist']= m_hist
    df_5m['RSI']= compute_rsi(df_5m, 14)
    df_5m['ASI']= compute_asi(df_5m)
    hsp, lsp= detect_hsp_lsp(df_5m['ASI'], window=3)
    df_5m['has_HSP']= hsp
    df_5m['has_LSP']= lsp

    # 15-min ADX
    df_15m['ADX_15']= compute_adx(df_15m[['High','Low','Close']], period=14)

    # We'll build a helper function that, for a given bar row + "LONG" or "SHORT", returns the same features used in training:
    def build_ml_features_for_bar(row, signal_type:str='LONG'):
        # replicate columns used in build_labeled_dataset
        feats = {}
        feats['Close']       = row['Close']
        feats['Volume']      = row['Volume']
        feats['RSI']         = row['RSI']
        feats['MACD_Line']   = m_line[row.name]  # or row['MACD_Line'] if we stored it
        feats['MACD_Signal'] = m_signal[row.name] # or row['MACD_Signal']
        feats['ADX']         = row['ADX']
        feats['ATR']         = row['ATR']
        feats['EMA_50']      = row['EMA_50']
        feats['signal_type'] = 1 if signal_type=='LONG' else 0

        return feats

    days_list= extract_trading_days(df_5m)
    for day in days_list:
        d5= df_5m[df_5m.index.normalize()== day].copy()
        d15= df_15m[df_15m.index.normalize()== day].copy()
        if d5.empty or d15.empty:
            continue

        d5['VolMean20']=  d5['Volume'].rolling(20).mean().ffill().fillna(0)



        d5['EMA_50']= d5['Close'].ewm(span= PARAMS['ema_period'], adjust=False).mean()
        d5['EMA_50_Slope']= d5['EMA_50'].diff(1)


        #Basic filter
        if d5['ATR'].median()< PARAMS['min_atr']:
            continue
        if d5['Volume'].median()< PARAMS['min_volume']:
            continue
        if d5['ADX'].median()< PARAMS['min_adx']:
            continue

        fib_info= compute_fib_levels_for_day(d5)
        if not fib_info:
            log_message(f"No fib levels for {instrument_name} on {day}")#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            continue
        if PARAMS['use_higher_timeframe_ema'] and (df_daily_ema is not None) and (not df_daily_ema.empty):
            daily_vals= []
            for ix in d5.index:
                if ix in df_daily_ema.index:
                    daily_vals.append(df_daily_ema.at[ix,'DailyEMA'])
                else:
                    daily_vals.append(np.nan)
            d5['DailyEMA']= daily_vals
        else:
            d5['DailyEMA']= np.nan

        open_positions= []
        consecutive_above_618=0#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        consecutive_below_382=0

        d15_idx= d15.index
        for bar_time, row in d5.iterrows():
            if d15_idx.empty:
                continue
            if bar_time< d15_idx[0] or bar_time> d15_idx[-1]:
                continue
            pos15= d15_idx.searchsorted(bar_time, side='right')-1
            if pos15<0 or pos15>= len(d15_idx):
                continue

            adx_15_val= d15['ADX_15'].iloc[pos15]
            if adx_15_val< PARAMS['adx_15m_threshold']:
                continue

            cpx= row['Close']
            vol_spike= (row['Volume']> row['VolMean20']* PARAMS['volume_factor'])



            adx_5m= row['ADX']
            rsi_val= row['RSI']
            atr_val= row['ATR']
            has_h= bool(row['has_HSP'])
            has_l= bool(row['has_LSP'])
            ema_50= row['EMA_50']
            slope_50= row['EMA_50_Slope']
            macd_val= row['MACD_Hist']
            daily_ema_val= row['DailyEMA']

            uptrend= (cpx> ema_50) and (slope_50> PARAMS['min_ema_slope'])
            downtrend= (cpx< ema_50) and (slope_50< -PARAMS['min_ema_slope'])

            # overbought/oversold avoidance
            if adx_5m< PARAMS['adx_filter_threshold']:
                continue
            #if fib_info.get(1.272, None) is None:
                #log_message(f"No fib info for fib_1272")
                #continue
            fib_618= fib_info.get(0.618, None)
            fib_382= fib_info.get(0.382, None)

            # multi-bar confirm
            if uptrend and fib_618 and (cpx> fib_618):
                if has_h or consecutive_above_618>0:
                    consecutive_above_618+=1
                else:
                    consecutive_above_618=0
            else:
                consecutive_above_618=0

            if downtrend and fib_382 and (cpx< fib_382):
                if has_l or consecutive_below_382>0:
                    consecutive_below_382+=1
                else:
                    consecutive_below_382=0
            else:
                consecutive_below_382=0

            if PARAMS['use_higher_timeframe_ema'] and not pd.isna(daily_ema_val):
                if uptrend and (cpx<= daily_ema_val):
                    continue
                if downtrend and (cpx>= daily_ema_val):
                    continue

            risk_scale= PARAMS['risk_per_trade_fraction']
            if (adx_15_val> PARAMS['strong_signal_adx_threshold']) or \
               (row['Volume']> row['VolMean20']* PARAMS['strong_signal_volume_factor']):
                risk_scale*=1.5
            acct_cap= PARAMS['assumed_account_size']* risk_scale

            if len(open_positions)< PARAMS['max_positions']:
                # LONG
                if uptrend and vol_spike and (rsi_val< PARAMS['rsi_upper_bound']) and (macd_val>0):
                    if fib_618 and (cpx> fib_618) and (consecutive_above_618>= PARAMS['multi_bar_confirm']):
                        # =========== Machine Learning for LONG    =============
                        if model is not None:
                            feats = build_ml_features_for_bar(row, 'LONG')
                            # We convert feats -> DataFrame row with same columns as train set

                            use_cols = ['Close','Volume','RSI','MACD_Line','MACD_Signal','ADX','ATR','EMA_50']
                            Xrow = pd.DataFrame([feats], columns=use_cols)
                            pred_label = model.predict(Xrow)[0]  # 0 or 1
                            if pred_label == 0:
                                # If ML model says not profitable, skip the trade
                                continue
                            # else model says 1 => proceed
                        # normal rule-based:
                        stop_dist= PARAMS['atr_stop_multiplier']* atr_val
                        if stop_dist<0.01:
                            stop_dist=0.01
                        qnty= math.floor(acct_cap/(stop_dist* cpx))
                        if qnty>0:
                            posd= {
                                'instrument': instrument_name,
                                'side':'LONG',
                                'entry_price': cpx,
                                'quantity': qnty,
                                'stop_price': cpx- stop_dist,
                                'fib_info': fib_info,
                                'partial_exits_done':0,
                                'entry_time': bar_time
                            }
                            open_positions.append(posd)

                # SHORT
                if downtrend and vol_spike and (rsi_val> PARAMS['rsi_lower_bound']) and (macd_val<0):
                    if fib_382 and (cpx< fib_382) and (consecutive_below_382>= PARAMS['multi_bar_confirm']):
                        # =========== ML PREDICTION CHECK FOR SHORT =============
                        if model is not None:
                            feats = build_ml_features_for_bar(row, 'SHORT')
                            use_cols = ['Close','Volume','RSI','MACD_Line','MACD_Signal','ADX','ATR','EMA_50']
                            Xrow = pd.DataFrame([feats], columns=use_cols)
                            pred_label = model.predict(Xrow)[0]
                            if pred_label == 0:
                                continue
                        # normal rule-based:
                        stop_dist= PARAMS['atr_stop_multiplier']* atr_val
                        if stop_dist<0.01:
                            stop_dist=0.01
                        qnty= math.floor(acct_cap/(stop_dist* cpx))
                        if qnty>0:
                            posd= {
                                'instrument': instrument_name,
                                'side':'SHORT',
                                'entry_price': cpx,
                                'quantity': qnty,
                                'stop_price': cpx+ stop_dist,
                                'fib_info': fib_info,
                                'partial_exits_done':0,
                                'entry_time': bar_time
                            }
                            open_positions.append(posd)

            # manage open positions
            to_remove= []
            for i_pos, posdict in enumerate(open_positions):
                side= posdict['side']
                stp= posdict['stop_price']
                fibs= posdict['fib_info']
                pexits= PARAMS['partial_exit_levels']
                dyn_stop= PARAMS['atr_stop_multiplier']* atr_val

                if side=='LONG':
                    candidate_stop= cpx- dyn_stop
                    if candidate_stop> stp:
                        posdict['stop_price']= candidate_stop

                    # partial exits
                    for lvl_i in pexits:
                        if posdict['partial_exits_done']>= len(pexits):
                            break
                        desired= pexits[posdict['partial_exits_done']]
                        fib_lev= fibs.get(desired, None)
                        if fib_lev and (cpx>= fib_lev):
                            part_q= math.floor(posdict['quantity']* PARAMS['partial_exit_ratio'])

                            if part_q>0 and posdict['quantity']> part_q:
                                subpos= posdict.copy()
                                subpos['quantity']= part_q
                                record_trade(trades_out, subpos, fib_lev, bar_time)
                                posdict['quantity']-= part_q
                                posdict['partial_exits_done']+=1
                            break

                    # stop out
                    if cpx< posdict['stop_price']:
                        record_trade(trades_out, posdict, posdict['stop_price'], bar_time)
                        to_remove.append(i_pos)
                        continue

                    # final fib
                    final_fib= fibs.get(2.0, None)#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                    if final_fib and (cpx>= final_fib):
                        record_trade(trades_out, posdict, final_fib, bar_time)
                        to_remove.append(i_pos)
                        continue

                else: # SHORT
                    candidate_stop= cpx+ dyn_stop
                    if candidate_stop< stp:
                        posdict['stop_price']= candidate_stop

                    # partial exits
                    for lvl_i in pexits:
                        if posdict['partial_exits_done']>= len(pexits):
                            break
                        desired= pexits[posdict['partial_exits_done']]
                        fib_lev= fibs.get(desired, None)
                        if fib_lev and (cpx<= fib_lev):
                            part_q= math.floor(posdict['quantity']* PARAMS['partial_exit_ratio'])
                            if part_q>0 and posdict['quantity']> part_q:
                                subpos= posdict.copy()
                                subpos['quantity']= part_q
                                record_trade(trades_out, subpos, fib_lev, bar_time)
                                posdict['quantity']-= part_q
                                posdict['partial_exits_done']+=1
                            break

                    # stop out
                    if cpx> posdict['stop_price']:
                        record_trade(trades_out, posdict, posdict['stop_price'], bar_time)
                        to_remove.append(i_pos)
                        continue

                    # final fib@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                    final_fib= fibs.get(2.0, None)
                    if final_fib and (cpx<= final_fib):
                        record_trade(trades_out, posdict, final_fib, bar_time)
                        to_remove.append(i_pos)
                        continue

            # remove positions that are closed
            new_opens= []
            for j, pval in enumerate(open_positions):
                if j not in to_remove:
                    new_opens.append(pval)
            open_positions= new_opens


        # End-of-day close any leftover
        if open_positions:
            eod_px= d5.iloc[-1]['Close']
            eod_tm= d5.index[-1]
            for leftover in open_positions:
                record_trade(trades_out, leftover, eod_px, eod_tm)
            open_positions= []

def compute_performance_metrics(trades: list)->dict:
    if len(trades)==0:
        return {
            'net_pl': 0.0,
            'total_trades': 0,
            'win_rate': 0.0,
            'average_pl': 0.0,
            'max_drawdown': 0.0,
            'sharpe': 0.0
        }
    df_tr= pd.DataFrame(trades).sort_values('exit_time')
    pl_arr= df_tr['net_pl'].values
    net_pl= pl_arr.sum()
    total_tr= len(pl_arr)
    wins= (pl_arr>0).sum()
    w_rate= (wins/ total_tr)*100 if total_tr>0 else 0.0
    avg_pl= pl_arr.mean()

    eq= np.cumsum(pl_arr)
    peak= eq[0]
    max_dd= 0
    for val in eq:
        if val> peak:
            peak= val
        dd= peak- val
        if dd> max_dd:
            max_dd= dd

    std_ret= pl_arr.std(ddof=1)
    if std_ret< 1e-9:
        shrp= 0.0
    else:
        shrp= (avg_pl/ std_ret)* np.sqrt(total_tr)

    return {
        'net_pl': round(net_pl,2),
        'total_trades': total_tr,
        'win_rate': round(w_rate,2),
        'average_pl': round(avg_pl,2),
        'max_drawdown': round(max_dd,2),
        'sharpe': round(shrp,3)
    }

def main_backtest_all_instruments(data_dir: str, model=None):
    """
    Walk subfolders, load each ticker, resample to 5m and 15m, run the backtest
    with machine learning
    """
    subfolders= []
    for item in sorted(os.listdir(data_dir)):
        path_sub= os.path.join(data_dir, item)
        if os.path.isdir(path_sub) and ("Cash Data" in item) and ("2021" in item):
            subfolders.append(path_sub)

    ticker_csv_map= {}
    for sf in subfolders:
        csv_files= glob.glob(os.path.join(sf,"*.csv"))
        for cfile in csv_files:
            base= os.path.basename(cfile)
            tkr_name= os.path.splitext(base)[0]
            ticker_csv_map.setdefault(tkr_name,[]).append(cfile)

    all_trades= []
    for tkr, cfiles in ticker_csv_map.items():
        log_message(f"Now processing TICKER {tkr}, with {len(cfiles)} CSV files...")

        df_parts= []
        for cf in cfiles:
            dfx= pd.read_csv(cf)
            needed_cols = {'<date>', '<time>', '<open>', '<high>', '<low>', '<close>', '<volume>'}
            if not needed_cols.issubset(dfx.columns):
                log_message(f"Missing columns in {cf}, skipping.")
                continue
            dfx['DateTime'] = pd.to_datetime(dfx['<date>']+" "+dfx['<time>'], errors='coerce')
            dfx.set_index('DateTime', inplace=True)
            dfx.sort_index(inplace=True)

            rename_map = {
                '<open>': 'Open',
                '<high>': 'High',
                '<low>':  'Low',
                '<close>':'Close',
                '<volume>':'Volume'
            }
            dfx.rename(columns=rename_map, inplace=True)
            dfx = dfx[['Open','High','Low','Close','Volume']].dropna()
            dfx = dfx[~dfx.index.duplicated(keep='first')]
            dfx = dfx.loc[(dfx.index>= BACKTEST_START_DATE)&(dfx.index<= BACKTEST_END_DATE)]
            dfx = dfx[dfx.index.map(time_filter_ok)]
            if not dfx.empty:
                df_parts.append(dfx)

        if not df_parts:
            continue

        df_merged= pd.concat(df_parts).sort_index()
        df_merged= df_merged[~df_merged.index.duplicated(keep='first')]
        if df_merged.empty:
            continue

        log_message(f"{tkr}: merged data has {len(df_merged)} rows after filtering.")

        df_daily_ema= pd.DataFrame()
        if PARAMS['use_higher_timeframe_ema']:
            df_daily_ema= compute_daily_ema_filter(df_merged, PARAMS['daily_ema_period'])
            if df_daily_ema.empty:
                log_message(f"No daily EMA available for {tkr}; skipping daily filter usage.")

        df_5m= resample_bars(df_merged, PARAMS['bar_interval_5m'])
        df_15m= resample_bars(df_merged, PARAMS['bar_interval_15m'])
        if df_5m.empty or df_15m.empty:
            continue

        ticker_trades= []
        # --------------------------------------------------
        # Wrap the backtest call in try/except
        # --------------------------------------------------
        try:
          run_backtest_for_instrument(
              df_5m=df_5m,
              df_15m=df_15m,
              instrument_name=tkr,
              trades_out=ticker_trades,
              df_daily_ema=df_daily_ema,
              model=model
          )
        except Exception as e:
          log_message(f"ERROR running backtest for {tkr}: {repr(e)}")





        # Here   we pass 'model' into run_backtest_for_instrument@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        #run_backtest_for_instrument(df_5m, df_15m, tkr, ticker_trades, df_daily_ema, model=model)

        log_message(f"{tkr}: {len(ticker_trades)} trades found.")
        stats= compute_performance_metrics(ticker_trades)
        log_message(f"{tkr}: {stats}")
        all_trades.extend(ticker_trades)

    final_stats= compute_performance_metrics(all_trades)
    log_message(f"FINAL RESULTS ACROSS ALL TICKERS: {final_stats}")

    df_all= pd.DataFrame(all_trades)
    out_path= "../colab/Algorithm_7/Algorithm7_version11_Trades_1.csv"
    df_all.to_csv(out_path, index=False)
    log_message("All trades saved to 'Algorithm7_version11_Trades_1.csv'")
    return all_trades, final_stats

##############################################################################
# MAIN
##############################################################################

def main():
    """
    1) Load 2021 & 2022 data for Machine Learning (recursive).
    2) Split at TRAIN_CUTOFF.
    3) Build labeled dataset & train XGB model (LONG & SHORT).
    4) Quick out-of-sample test.
    5) Load the trained model & run Algorithm 7 backtest on 2021, using model predictions.

    """
    # Step 1: Load data from 2021 & 2022
    log_message("Loading 2021 CSVs (recursive) for ML building...")
    df_2021 = load_csv_folder_recursive(DATA_FOLDER_2021)

    #log_message("Loading 2022 CSVs (recursive) for ML building...")
    #df_2022 = load_csv_folder_recursive(DATA_FOLDER_2022)

    df_all  = df_2021.sort_index()
    #df_all = pd.concat([df_2021, df_2022]).sort_index()
    df_all  = df_all[~df_all.index.duplicated(keep='first')]

    if df_all.empty:
        log_message("No combined data loaded for ML. Exiting.")
        return

    # Step 2: split by TRAIN_CUTOFF
    log_message(f"Splitting data by TRAIN_CUTOFF={TRAIN_CUTOFF}")
    df_train = df_all.loc[df_all.index< TRAIN_CUTOFF].copy()
    df_test  = df_all.loc[df_all.index>= TRAIN_CUTOFF].copy()

    # Step 3: Build labeled dataset from df_train 5-min
    log_message("Resampling training data to 5-min bars for labeling...")
    df_5m_train = resample_5m_bars_for_ml(df_train)
    if df_5m_train.empty:
        log_message("No 5-min train data. Exiting.")
        return

    log_message("Building labeled samples (LONG+SHORT) from the training protion...")
    train_samples = build_labeled_dataset(df_5m_train)
    if train_samples.empty:
        log_message("No training samples. Exiting.")
        return

    log_message("Training ML model on the in-sample portion...")
    train_and_save_model(train_samples)
    log_message("ML model training done.")

    # Step 4: Quick out-of-sample test
    log_message("Resampling test data to 5-min bars for a quick check...")
    df_5m_test = resample_5m_bars_for_ml(df_test)
    quick_out_of_sample_test(df_5m_test)

    # Step 5: Load the model & run the final backtest on 2021 using the model
    log_message("Now running the final backtest on 2021 data folder (with ML predictions).")
    try:
        ml_model = joblib.load(MODEL_SAVE_PATH)
        log_message("ML model loaded successfully for backtest usage.")
    except Exception as e:
        log_message(f"Could not load ML model for backtest. Error: {e}")
        ml_model = None  # fallback - if we can't load, we'll just do rule-based.

    base_data_directory= "../2021"
    trades, final_stats= main_backtest_all_instruments(base_data_directory, model=ml_model)
    print("============== BACKTEST COMPLETE ==============")
    for k,v in final_stats.items():
        print(f"{k}: {v}")

if __name__=="__main__":
    main()