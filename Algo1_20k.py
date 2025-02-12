"""
##**Volume = 20,000**

**ATR = 0.7**

**ADX = 25**

**Tickers =46**
"""

# Import necessary libraries
import os
import glob
import pandas as pd
import numpy as np
from collections import defaultdict
import logging
from datetime import datetime

LOG_DIR = "C:/Users/vuanh/Downloads/colab"
base_dir = "C:/Users/vuanh/Downloads/2021"
# Log file path
LOG_FILE = os.path.join(LOG_DIR, "Algo1_logfile_v1.txt")


# ---------------------------
# FUNCTION DEFINITIONS
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


##############################
#  USER-DEFINED PATHS
##############################

BASE_2021_DIR = "C:/Users/vuanh/Downloads/2021"
BASE_2022_DIR = "E:/VuAnhData/2022"

TIMEFRAME = "5min"

# Defatult Rolling Periods
ATR_PERIOD = 14
ADX_PERIOD = 14
ROLLING_VOL_BARS = 100


############################################################
# STEP A: LOAD & CLEAN SINGLE CSV
############################################################
def load_and_clean_csv(csv_path: str) -> pd.DataFrame:
    """

    Loads one CSV that contains intrady data for a single ticker & month.
    Expects:
       <data>, <time>, <open>, <high>, <low>, <close>, <volume>, (<ticker>)
    Merges <date>+<time> into a DateTiem index; drops incomplete rows.
    """

    if not os.path.exists(csv_path):
        log_message(f"CSV path does not exist: {csv_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        log_message(f"Error reading CSV: '{csv_path}': {e}")
        return pd.DataFrame()

    needed_cols = {'<date>', '<time>', '<open>', '<high>', '<low>', '<close>', '<volume>'}

    if not needed_cols.issubset(df.columns):
        log_message(f"Required Columns missing in {csv_path}")
        return pd.DataFrame()

    # Merge date/time => single DateTime
    df['DateTime'] = pd.to_datetime(
        df['<date>'] + " " + df['<time>'],
        errors='coerce'
    )
    df.set_index('DateTime', inplace=True)
    df.drop(['<date>', '<time>'], axis=1, errors='ignore', inplace=True)

    # convert to numeric
    numeric_cols = ['<open>', '<high>', '<low>', '<close>', '<volume>']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Optional rename
    rename_map = {
        '<open>': 'Open',
        '<high>': 'High',
        '<low>': 'Low',
        '<close>': 'Close',
        '<volume>': 'Volume',
        '<ticker>': 'Ticker'
    }
    df.rename(columns=rename_map, inplace=True)

    # Sort & drop rows missing O/H/L/C
    df.sort_index(inplace=True)
    df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], how='any', inplace=True)

    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]
    return df


############################################################
# STEP B; COMBINE 12 MONTHS => ANNUAL, TICKER-BY-TICKER
############################################################
def combine_annual_data(base_dir: str) -> dict:
    """
    1) Finds subfolders (one per month).
    2) Builds {ticker => list of monthly CSVs}.
    3) Loads & merges => single DF per ticker for the full year.
    Returns {ticker: df_combined}.
    """
    # check directory validity
    if not os.path.isdir(base_dir):
        log_message(f"Base directory '{base_dir}' does not exist.")
        return {}

    month_folders = []
    for item in sorted(os.listdir(base_dir)):
        subp = os.path.join(base_dir, item)
        if os.path.isdir(subp):
            month_folders.append(subp)

    log_message(f"Found {len(month_folders)} monthly folders under '{base_dir}':")
    for mf in month_folders:
        log_message(f"   -> {os.path.basename(mf)}")

    # Gather CSV paths
    ticker_to_csvs = defaultdict(list)
    for mfolder in month_folders:
        csv_files = glob.glob(os.path.join(mfolder, "*.csv"))
        log_message(f" {os.path.basename(mfolder)} => {len(csv_files)} CSV files found.")
        for cpath in csv_files:
            ticker_name = os.path.splitext(os.path.basename(cpath))[0]
            ticker_to_csvs[ticker_name].append(cpath)

    combined_map = {}
    for ticker, cpaths in ticker_to_csvs.items():
        df_list = []
        for cpath in cpaths:
            df_part = load_and_clean_csv(cpath)
            if not df_part.empty:
                df_list.append(df_part)
        if not df_list:
            continue
        df_full = pd.concat(df_list, axis=0).sort_index()
        df_full = df_full[~df_full.index.duplicated(keep='first')]
        combined_map[ticker] = df_full
    log_message(f"[INFO] Total distinct tickers across subfolders: {len(ticker_to_csvs)}")
    log_message(f"[INFO] Tickers with valid data after merging: {len(combined_map)}")
    return combined_map


############################################################
# STEP C: RESAMPLE => 5min
############################################################
def resample_to_5min(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample to 5-min bars:
      - Open: first
      - High: max
      - Low: min
      - Close: last
      - Volume: sum
    Drops bars missing O/H/L/C.
    """

    agg_spec = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    df_5m = df.resample('5min').agg(agg_spec)
    df_5m.dropna(subset=['Open', 'High', 'Low', 'Close'], how='any', inplace=True)
    return df_5m


def resample_all_to_5min(annual_map: dict) -> dict:
    """
    For each ticker: produce 5-min DataFrame
    Return {ticker: df_5m} for non-empty results.
    """
    out_map = {}
    for tkr, df_yr in annual_map.items():
        if df_yr.empty:
            log_message(f"No Data to resample for ticker: {tkr}")
            continue
        df_5 = resample_to_5min(df_yr)
        if not df_5.empty:
            out_map[tkr] = df_5
    return out_map


############################################################
# STEP D: INDICATOR CALCULATIONS (ATR, ADX)
############################################################
def calculate_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Calculate the Average True Range (ATR) with Wilder's smoothing.
    :param df: DataFrame with 'High', 'Low', and 'Close' columns.
    :param period: Lookback period for ATR calculation.
    :return: ATR as a Pandas Series.
    """
    # Calculate True Range (TR)
    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift(1)).abs()
    lc = (df['Low'] - df['Close'].shift(1)).abs()
    tr = np.maximum.reduce([hl, hc, lc])

    # Convert TR to Pandas Series before applying Wilder's smoothing
    tr_series = pd.Series(tr, index=df.index)

    # Apply Wilder's smoothing to TR
    atr = tr_series.ewm(alpha=1 / period, adjust=False).mean()

    return atr


def calculate_adx(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Calculate the Average Directional Index (ADX).with Wilder's smoothing.
    :param df: DataFrame with 'High', 'Low', and 'Close' columns.
    :param period: Lookback period for ADX calculation.
    :return: ADX as a Pandas Series.
    """
    # Calcualte directional movements
    up_move = df['High'].diff()
    down_move = -df['Low'].diff()

    plus_dm = np.where((up_move > 0) & (up_move > down_move), up_move, 0)
    minus_dm = np.where((down_move > 0) & (down_move > up_move), down_move, 0)

    # Calculate True Range (TR)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift(1))
    low_close = np.abs(df['Low'] - df['Close'].shift(1))
    tr = np.maximum.reduce([high_low, high_close, low_close])

    # convert numpy array to Pandas Series
    tr_series = pd.Series(tr, index=df.index)
    plus_dm_series = pd.Series(plus_dm, index=df.index)
    minus_dm_series = pd.Series(minus_dm, index=df.index)

    # Smooth TR, +DM, and -DM using Wilder's smoothing
    tr_smoothed = tr_series.ewm(alpha=1 / period, adjust=False).mean()
    plus_dm_smoothed = plus_dm_series.ewm(alpha=1 / period, adjust=False).mean()
    minus_dm_smoothed = minus_dm_series.ewm(alpha=1 / period, adjust=False).mean()

    # Calculate +DI and -DI
    plus_di = (plus_dm_smoothed / tr_smoothed) * 100
    minus_di = (minus_dm_smoothed / tr_smoothed) * 100

    # Calculate Directional Index (DX)
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100

    # Calculate ADX
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    return pd.Series(adx, index=df.index)


############################################################
# STEP E; GATHER FINAL STATISTICS => CSV
############################################################
def gather_final_stats_5min(df_5m: pd.DataFrame) -> dict:
    """
    For a given 5-min DataFrame, compute:
      - final rolling avarage volume (over ROLLING_VOL_BARS),
      - final ATR(14),
      - final ADX(14).
    Retrun them in a dictionary. If insufficient data, return empty dict.
    """
    if len(df_5m) < max(ROLLING_VOL_BARS, ATR_PERIOD + 1, ADX_PERIOD + 1):
        return {}

    # Rolling Volume
    vol_roll = df_5m['Volume'].rolling(ROLLING_VOL_BARS).mean().dropna()
    if vol_roll.empty:
        return {}

    final_vol = vol_roll.iloc[-1]

    # ATR
    atr_ser = calculate_atr(df_5m, ATR_PERIOD).dropna()
    if atr_ser.empty:
        return {}
    final_atr = atr_ser.iloc[-1]

    # ADX
    adx_ser = calculate_adx(df_5m, ADX_PERIOD).dropna()
    if adx_ser.empty:
        return {}
    final_adx = adx_ser.iloc[-1]

    return {
        'FinalRollingVolume': final_vol,
        'FinalATR': final_atr,
        'FinalADX': final_adx
    }


############################################################
# MAIN FUNCTION
############################################################

def main():
    log_message("Starting the main function....")

    # 1) Combine all months => annual DataFrames
    combined_map = combine_annual_data(base_dir)
    log_message(f"[INFO] Combined full-year data for {len(combined_map)} tickers.")

    # 2) Resample each ticker => 5-min bars
    five_min_map = resample_all_to_5min(combined_map)
    log_message(f"[INFO] Resampled to 5-min for {len(five_min_map)} tickers.")

    # 3) For each ticker, gather final metrics

    results_list = []
    for tkr, df_5m in five_min_map.items():
        stats_dict = gather_final_stats_5min(df_5m)
        if not stats_dict:
            # insufficient data or empty dict
            continue
        row = {
            'Ticker': tkr,
            'FinalRollingVolume': stats_dict['FinalRollingVolume'],
            'FinalATR': stats_dict['FinalATR'],
            'FinalADX': stats_dict['FinalADX']
        }
        results_list.append(row)

    # 4) Store the stats in a CSV for distribution analysis
    stats_df = pd.DataFrame(results_list)
    stats_df.sort_values('FinalRollingVolume', ascending=False, inplace=True)
    try:
        stats_df.to_csv("/content/drive/MyDrive/Capstone Share/colab/Algo1_final_intraday_stats.csv", index=False)
        log_message(f"[INFO] Exported Algo1_final_intraday_stats.csv with {len(stats_df)} rows.")
    except Exception as e:
        log_message(f"[ERROR] Error exporting Algo1_final_intraday_stats.csv: {e}")

    # 5) Now we can investigate distributions of volume, ATR, ADX to pick  thresholds
    log_message("=== DISTRIBUTION ANALYSIS  (placeholder) ===")
    if not stats_df.empty:
        log_message(f"\n{stats_df[['FinalRollingVolume', 'FinalATR', 'FinalADX']].describe()}")
        # print(f"\n{stats_df[['FinalRollingVolume','FinalATR','FinalADX']].describe()}")
        # Then  set thresholds based on these distributions

    # 6) Once thresholds decided, filter or do the next step

    chosen_thresholds = {
        'Volume': 20_000,
        'ATR': 0.7,
        'ADX': 25

    }

    # Filter
    filtered_df = stats_df[
        (stats_df['FinalRollingVolume'] >= chosen_thresholds['Volume']) &
        (stats_df['FinalATR'] >= chosen_thresholds['ATR']) &
        (stats_df['FinalADX'] >= chosen_thresholds['ADX'])

        ]
    filtered_tickers = filtered_df['Ticker'].tolist()

    log_message("\n=== Tickers passing custom thresholds ===")
    log_message(
        f"Thresholds: Volume>={chosen_thresholds['Volume']}, ATR>={chosen_thresholds['ATR']}, ADX>={chosen_thresholds['ADX']}")
    log_message(f"Number passing: {len(filtered_tickers)}")
    log_message(f"Tickers: {filtered_tickers}")
    # print(f"Candidate Tickers: {filtered_tickers}")
    # print(f"Total: {len(filtered_tickers)}")

    # Save filtered data to a CSV file
    filtered_df.to_csv("C:/Users/vuanh/Downloads/colab/Algo1_filtered_tickers.csv", index=False)
    log_message(f"[INFO] Exported Algo1_filtered_tickers.csv with {len(filtered_df)} rows.")
    log_message("End of the main function.")


if __name__ == "__main__":
    main()
# Check the log
#!cat "/content/drive/MyDrive/Capstone Share/colab/Algo1_logfile.txt"

"""**Consistency Test**"""

