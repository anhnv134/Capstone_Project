
"""
**DataDrivenThresholds.csv**"""

# import necessary libraries
import os
import numpy as np
import pandas as pd
from datetime import datetime

#####################################
# CONFIGURE LOGGING
#####################################
LOG_DIR = "../colab"

# Log file path
LOG_FILE = os.path.join(LOG_DIR, "Algo1_logfile.txt")


# ---------------------------
# LOG FUNCTION DEFINITIONS
# ---------------------------

def log_message(message, log_file=LOG_FILE):
    """
    Logs a message with a timestamp to the specified log file and prints it.

    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    try:
        with open(log_file, 'a') as f:
            f.write(full_message + "\n")
    except FileNotFoundError:
        print(f"ERROR: Log file directory does not exist: {os.path.dirname(log_file)}")
    except Exception as e:
        print(f"ERROR: Failed to write to log file. Exception: {e}")


###############################################################################
# PART 1: INDICATORS
###############################################################################
def wilder_atr(df, period=14):  # We use adjust=True:pandas default
    """
    Computes Wilder's ATR. Requires columns: High, Low, Close.
    """
    df = df.copy()
    df['H-L'] = df['High'] - df['Low']
    df['H-Cp'] = (df['High'] - df['Close'].shift(1)).abs()
    df['L-Cp'] = (df['Low'] - df['Close'].shift(1)).abs()
    df['TR'] = df[['H-L', 'H-Cp', 'L-Cp']].max(axis=1)
    atr = df['TR'].ewm(alpha=1.0 / period, adjust=False).mean()
    return atr


def compute_adx(df, period=14):
    """
    ADX using Wilder's approach. Returns a Series 'ADX'.
    """
    df = df.copy()
    df['upMove'] = df['High'] - df['High'].shift(1)
    df['downMove'] = df['Low'].shift(1) - df['Low']
    df['+DM'] = np.where((df['upMove'] > df['downMove']) & (df['upMove'] > 0), df['upMove'], 0)
    df['-DM'] = np.where((df['downMove'] > df['upMove']) & (df['downMove'] > 0), df['downMove'], 0)

    df['ATR'] = wilder_atr(df[['High', 'Low', 'Close']], period=period)

    alpha = 1.0 / period
    df['+DI'] = 100 * df['+DM'].ewm(alpha=alpha).mean() / df['ATR']
    df['-DI'] = 100 * df['-DM'].ewm(alpha=alpha).mean() / df['ATR']

    df['DX'] = 100 * ((df['+DI'] - df['-DI']).abs() / (df['+DI'] + df['-DI'] + 1e-9))
    df['ADX'] = df['DX'].ewm(alpha=alpha).mean()
    return df['ADX']


def compute_rsi(df, period=14):
    """
    RSI (Wilder). Returns a Series 'RSI'.
    """
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(alpha=1 / period, adjust=False).mean()
    ema_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = ema_up / (ema_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


###############################################################################
# PART 2: AUTO-THRESHOLDS ONLY
###############################################################################
def auto_thresholds(df):
    """
    Automatically derive thresholds
    - ADX threshold = 60th percentile
    - Volume spike threshold = mean + std
    - RSI upper /lower = 90th/10th percentile
    """
    result = {}

    # ADX threshold from 60th percentile (if ADX is present)
    if 'ADX' in df.columns and df['ADX'].notna().sum() > 0:
        adx_60 = df['ADX'].quantile(0.6)
        result['adx_threshold'] = adx_60
    else:
        result['adx_threshold'] = 20.0  # fallback

    # Volume spike threshold = mean + std
    if 'Volume' in df.columns and df['Volume'].notna().sum() > 0:
        vol_mean = df['Volume'].mean()
        vol_std = df['Volume'].std()
        result['volume_spike_threshold'] = vol_mean + vol_std
    else:
        result['volume_spike_threshold'] = 9999999

    # RSI upper & lower from 90th / 10th percentile
    if 'RSI' in df.columns and df['RSI'].notna().sum() > 0:
        rsi_upper = df['RSI'].quantile(0.9)
        rsi_lower = df['RSI'].quantile(0.1)
        result['rsi_upper'] = rsi_upper
        result['rsi_lower'] = rsi_lower
    else:
        result['rsi_upper'] = 80
        result['rsi_lower'] = 20

    return result


###############################################################################
# PART 3: MAIN - SAVE THRESHOLDS TO CSV
###############################################################################
def main():
    """
    Loads 'Enriched_5Min_Data.csv',
    computes data-driven thresholds (ADX, Volume, RSI) for each ticker,
    and then saves them to a CSV file called 'DataDrivenThresholds.csv'.
    """

    data_path = "../colab/2021_selectedTickers/Enriched_5Min_Data.csv"
    if not os.path.exists(data_path):
        log_message(f"[INFO] CSV not found at: {data_path}")
        return

    # Load data
    df_5m_all = pd.read_csv(data_path)
    df_5m_all['Timestamp'] = pd.to_datetime(df_5m_all['Timestamp'], errors='coerce')
    df_5m_all.dropna(subset=['Timestamp'], inplace=True)
    df_5m_all.sort_values(['Ticker', 'Timestamp'], inplace=True)

    threshold_rows = []  # We'll store one row per ticker

    # Process each Ticker separately
    for ticker in df_5m_all['Ticker'].unique():
        subdf = df_5m_all[df_5m_all['Ticker'] == ticker].copy()
        subdf.sort_values('Timestamp', inplace=True)

        # If ADX missing or all NaN, compute it
        if 'ADX' not in subdf.columns or subdf['ADX'].isna().all():
            subdf['ADX'] = compute_adx(subdf[['High', 'Low', 'Close']].copy())

        # If RSI missing or all NaN, compute it
        if 'RSI' not in subdf.columns or subdf['RSI'].isna().all():
            subdf['RSI'] = compute_rsi(subdf)

        # Now derive thresholds
        thr_dict = auto_thresholds(subdf)
        threshold_rows.append({
            'Ticker': ticker,
            'ADX_Threshold': thr_dict['adx_threshold'],
            'Volume_Spike_Threshold': thr_dict['volume_spike_threshold'],
            'RSI_Upper': thr_dict['rsi_upper'],
            'RSI_Lower': thr_dict['rsi_lower']
        })

    # Convert to DataFrame & save
    df_thresholds = pd.DataFrame(threshold_rows)
    df_thresholds.sort_values('Ticker', inplace=True)

    out_csv = "../colab/2021_selectedTickers/DataDrivenThresholds.csv"
    df_thresholds.to_csv(out_csv, index=False)

    log_message("====== DATA-DRIVEN THRESHOLDS PER TICKER (SAVED) ======")
    for _, row in df_thresholds.iterrows():
        tkr = row['Ticker']
        log_message(f"\nTicker: {tkr}")
        log_message(f"  ADX threshold        : {row['ADX_Threshold']:.2f}")
        log_message(f"  Volume spike thr.    : {row['Volume_Spike_Threshold']:.2f}")
        log_message(f"  RSI upper (90th pct) : {row['RSI_Upper']:.2f}")
        log_message(f"  RSI lower (10th pct) : {row['RSI_Lower']:.2f}")

    log_message(f"\n[INFO] Thresholds saved to: {out_csv}")


if __name__ == "__main__":
    main()