"""##**Algorithm 2 Option A: Computing Fibinacci Levels for Intraday Breakouts**



<div style="text-align: justiry;">

**Goal:** Our goal here is for each chosen instrument,we will identify key Fibonacci retracements and extensions from a recent swing (e.g., the morning session swing) to determine breakout points.
</div>

**Inputs:**
- Intraday price series (O, H, L, C)
- A reference swing to measure Fibonacci ratios (e.g., the day's first significant swing high and swing low)
- Common Fibonacci ratios: '[0.382, 0.5, 0.618, 1.0, 1.272, 1.618, ...]'

**Pseudocode Steps:**

for instrument in instruments_to_trade:
    data = load_intraday_data(instrument)

    # identify a recent major swing (morning session example)
    morning_high = max price between session start and a cutoff time
    morning_low = min price in the same period

    # ensure which is swing high and which is swing low (trend direction)


    if morning_high_time < morning_low_time:
        # up-swing: low->high
        swing_low = morning_low
        swing_high = morning_high
    else:
        #
        # Down-swing: high->low
        swing_low = morning_lowswing
        swing_high = morning_high

    swing_range = swing_high - swing_low

    fibonacci_levels = {}
    for ratio in [0.382, 0.5, 0.618, 1.0, 1.272, 1.618]:
        if uptrend assumption:
            fib_level = swing_low + (swing_range * ratio)
        else:
            fib_level = swing_high - (swing_range * ratio)
        fibonacci_levels[ratio] = fib_level

**Output**: A dictionary of fibonacci levels keyed by ratio, providing potential breakout trigger points.
"""

# Algorithm 2

# Import necessary libraries
import os
import glob
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import time, datetime
import logging

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


# Utility Function for Safe Execution
def safe_execute(func, *args, **kwargs):
    """Safely execute a function with exception handling."""
    try:

        return func(*args, **kwargs)
    except Exception as e:
        log_message(f"Error executing {func.__name__}: {e}")
        return None


################################################################################
# USER SETTINGS
################################################################################
BASE_2021_DIR = "../colab/2021_selectedTickers"
CHOSEN_TICKERS = ['HINDALCO', 'TATAMOTORS', 'JSWSTEEL', 'DELTACORP', 'OIL', 'INFY',
                  'RESPONIND', 'HDFCBANK', 'DHANI', 'ADANIPORTS', 'ABFRL', 'IRCTC',
                  'APOLLO', 'AUBANK', 'BALRAMCHIN', 'HDFC', 'TATACOMM', 'TATAMTRDVR',
                  'ZENSARTECH', 'DBL', 'NIITLTD', 'CAMLINFINE', 'M&M', 'INTELLECT']

TIMEFRAME = "5min"  # We need 5-min resampled data
ATR_PERIOD = 14
ADX_PERIOD = 14
ROLLING_VOL_BARS = 100  # not used here, but keep from previous steps
FIB_RATIOS = [0.382, 0.5, 0.618, 1.0, 1.272, 1.618]


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


################################################################################
# STEP D: ALGORITHM 2 - FIBONACCI CALS
################################################################################
def get_morning_swing(day_data: pd.DataFrame, session_start=time(9, 15), cutoff=time(11, 0)):
    morning_df = day_data.between_time(session_start.strftime("%H:%M"), cutoff.strftime("%H:%M"))

    if morning_df.empty:
        return None
    high_idx = morning_df['High'].idxmax()
    low_idx = morning_df['Low'].idxmin()
    if pd.isna(high_idx) or pd.isna(low_idx):
        return None

    morning_high = morning_df['High'].loc[high_idx]
    morning_low = morning_df['Low'].loc[low_idx]
    return (morning_low, morning_high, low_idx, high_idx)


def compute_fibinacci_levels(swing_low, swing_high, is_up_swing) -> dict:
    swing_range = swing_high - swing_low
    fib_dict = {}
    for ratio in FIB_RATIOS:

        if is_up_swing:
            fib_val = swing_low + (swing_range * ratio)
        else:
            fib_val = swing_high - (swing_range * ratio)
        fib_dict[ratio] = fib_val
    return fib_dict


def compute_fibs_for_morning_swings(five_min_map: dict,
                                    instruments: list,
                                    session_start=time(9, 15),
                                    cutoff=time(11, 0)) -> dict:
    """
    Returns nested dict:
      {
        'TICKER': {
           date_1: {
              'fib_levels': {0.382: x, 0.5: y, ...},

              'up_swing': bool,
              'swing_low': val,
              'swing_high': val,
              'low_time': Timestamp,
              'high_time': Timestamp
           },
           date_2: { ... }, ...
           ...
        },
        ...
      }
    """
    all_results = {}
    for tkr in instruments:
        if tkr not in five_min_map:
            continue
        df_5m = five_min_map[tkr]
        if df_5m.empty:
            continue

        tkr_dict = {}
        unique_dates = df_5m.index.normalize().unique()
        for d in unique_dates:
            day_data = df_5m[df_5m.index.normalize() == d]
            if day_data.empty:
                continue

            info = get_morning_swing(day_data, session_start, cutoff)
            if not info:
                continue
            morning_low, morning_high, low_time, high_time = info

            is_up = (low_time < high_time)

            fibs = compute_fibinacci_levels(morning_low, morning_high, is_up)
            tkr_dict[d] = {

                'fib_levels': fibs,
                'up_swing': is_up,
                'swing_low': morning_low,
                'swing_high': morning_high,
                'low_time': low_time,
                'high_time': high_time
            }
        all_results[tkr] = tkr_dict
    return all_results

"""##**Algorithm 3: Calculation of Swing Index (SI) and Accumulative Swing Index (ASI)**

**Goal:**

<div style="text-align: justify;">

Our goal here is to utilize Swing Index (SI) and Accumulative Swing Index (ASI) calculations to confirm that momentum supports the breakout levels. The ASI can help verify if a bareakout is backed by a meaningful swing.This serves as a filter. We will only engage in breakouts if the ASI confirms a significant swing point.

</div>

**Inputs:**
- Intraday OHLC data for the current session
- Previous bar's close and previous ASI value
- SI, ASI formula parameters

**Pseudocode**

"
Given current bar (O,H,L,C) and previous bar close (C_prev):
  R = max(H - L, abs(H - C_prev), abs(L - C_prev)) # true range variation
  A = abs(H - C_prev)
  B = abs(L - C_prev)
  C_val = abs (H - L ) # just naming C_val to not confuse with C price

  # SI calculation
  # Sign determines if we add or subtract SI from ASI:
  # If C > C_prev: sign = +1 else sign = -1

  if C > C_prev:
      SI = 50 * ((C - C_prev) + (0.5*(C - 0)) + (0.25*(C_prev - 1))) / R
  else:
      SI = -50 * ((C_prev - C) + (0.5*(0 - C)) + (0.25*(0 - C_prev))) / R

  # Accumulate:
  ASI_new = ASI_old + SI
""

Steps (Pseudocode):

ASI = 0 # start of day or carry over from previous day if needed
C_prev = previous close (e.g., last bar before session start)

for each bar in intraday_data:
    compute SI using above formula
    ASI = ASI + SI
    C_prev = current bar's close

# After computing ASI for the current session up to now:
# Identify HSP/LSP: look for local maxima/minima in ASI
HSP if ASI makes a local peak above previous swings
LSP if ASI makes a local trough below previous swings
"

**Output**: Current ASI value, identification of any new High Swing Point (HSP) or Low Swing Point (LSP).
"""

# Algorithm 3
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


################################################################################
# ALGORITHM 3 - SWING INDEX (SI) & ACCUMULATIVE SWING INDEX (ASI)
################################################################################
def compute_SI_ASI(df: pd.DataFrame,
                   asi_initial: float = 0.0,
                   c_prev_initial: float = None) -> pd.DataFrame:
    """
    Computes the Swing Index (SI) and Accumulative Swing Index (ASI) for each bar.

    Parameters

    ------------------
    df : pd.DataFrame
        Must contain columns: ['Open', 'High', 'Low', 'Close'] (5-min bars).
    asi_initial : float
        Starting ASI value (e.g., 0 at session start or carryover from previous day).
    c_prev_initial : float
        Previous close price before this DataFrame starts (if None, use first row's Close).
        P

    Returns
    -------
    df_out : pd.DataFrame
        Same as input of but with extra columns: ['SI', 'ASI'].
    """

    df_out = df.copy()

    if c_prev_initial is None:
        c_prev_initial = df_out['Close'].iloc[0]

    si_list = []
    asi_list = []

    asi_running = asi_initial
    c_prev = c_prev_initial

    for idx, row in df_out.iterrows():
        O = row['Open']
        H = row['High']
        L = row['Low']
        C = row['Close']

        R = max(H - L, abs(H - c_prev), abs(L - c_prev))

        if C > c_prev:
            # SI formula (positive scenario)
            numerator = (C - c_prev) + 0.5 * (C - O) + 0.25 * (c_prev - O)
            si = 50.0 * numerator / R if R != 0 else 0.0

        else:
            # SI formula (negative scenario)
            numerator = (c_prev - C) + 0.5 * (O - C) + 0.25 * (O - c_prev)
            si = -50.0 * numerator / R if R != 0 else 0.0

        asi_running += si
        si_list.append(si)
        asi_list.append(asi_running)

        c_prev = C

    df_out['SI'] = si_list
    df_out['ASI'] = asi_list
    return df_out


def find_local_swing_points(asi_series: pd.Series, lookback: int = 2) -> pd.DataFrame:
    """
    Identifies local maxima (High Swing Point) and local minima (Low Swing Point)
    in the ASI by checking if the ASI is the highest/lowest among neighbors.

    Parameters
    ----------
    asi_series : pd.Series
        The ASI values (one per bar).
    loopback : int
        How many bars on each side to check for local max/min.

    Returns
    -------
    pd.DataFrame
        Contains 'ASI', 'is_HSP'(bool), 'is_LSP'(bool).
    """

    n = len(asi_series)
    is_hsp = [False] * n
    is_lsp = [False] * n

    for i in range(n):
        val = asi_series.iloc[i]
        start = max(i - lookback, 0)
        end = min(i + lookback + 1, n)  # +1 because end is exclusive
        window = asi_series.iloc[start:end]

        # Mark as HSP if it's the unique max
        if val == window.max() and window.value_counts()[val] == 1:
            is_hsp[i] = True
        # Mark as LSP if it's the unique min
        if val == window.min() and window.value_counts()[val] == 1:
            is_lsp[i] = True

    return pd.DataFrame({
        'ASI': asi_series.values,

        'is_HSP': is_hsp,
        'is_LSP': is_lsp
    }, index=asi_series.index)


################################################################################
# MAIN EXECUTION (ALGO 2 + ALGO 3)
################################################################################
def main():
    # =========================
    # STEP 1: COMBINE + RESAMPLE
    # =========================
    annual_map = combine_annual_data(BASE_2021_DIR)
    log_message(f"[INFO] Found annual data for {len(annual_map)} tickers in 2021_selectedTickers folder.")

    map_5m = resample_all_to_5min(annual_map)

    log_message(f"[INFO] Resampled to 5-min => {len(map_5m)} tickers remain non-empty.")

    # =========================
    # STEP 2: ALGORITHM 2 - FIBONACCI
    # =========================
    fib_results = compute_fibs_for_morning_swings(

        five_min_map=map_5m,
        instruments=CHOSEN_TICKERS,
        session_start=time(9, 15),
        cutoff=time(11, 0)
    )

    for tkr, ddict in fib_results.items():
        log_message(f"{tkr}: {len(ddict)} trading days with a valid morning swing & fib levels.")

    # Example CSV export (Algorithm 2 results)
    rows_algo2 = []
    for tkr, date_map in fib_results.items():
        for dt, info in date_map.items():
            row = {
                'Ticker': tkr,
                'Date': dt.strftime("%Y-%m-%d"),
                'UpSwing': info['up_swing'],
                'SwingLow': info['swing_low'],
                'SwingHigh': info['swing_high'],
                'LowTime': info['low_time'].strftime("%Y-%m-%d %H:%M:%S"),
                'HighTime': info['high_time'].strftime("%Y-%m-%d %H:%M:%S")
            }
            for ratio, price in info['fib_levels'].items():
                row[f"Fib_{ratio}"] = price
            rows_algo2.append(row)

    df_algo2 = pd.DataFrame(rows_algo2)
    df_algo2.sort_values(['Ticker', 'Date'], inplace=True)
    df_algo2.to_csv("../colab/2021_selectedTickers/Algorithm2_FibonacciLevels.csv",
                    index=False)
    log_message(f"[INFO] Fibonacci results saved to 'Algorithm2_FibonacciLevels.csv'.")

    # =========================
    # STEP 3: ALGORITHM 3 - SI & ASI
    # =========================
    # We'll compute SI/ASI for eac ticker's entire 5-min data,
    # then detect HSP/LSP points. Finally, we store to a new CSV.

    all_rows_asi = []
    for tkr in CHOSEN_TICKERS:
        if tkr not in map_5m:
            continue
        df_5m = map_5m[tkr]
        if df_5m.empty:
            continue

        # 1) Compute SI/ASI
        df_si_asi = compute_SI_ASI(df_5m, asi_initial=0.0, c_prev_initial=None)

        # 2) Find local swing points in the ASI
        df_sp = find_local_swing_points(df_si_asi['ASI'], lookback=2)

        # 3) Merge results for final usuage
        df_combined = df_si_asi.join(df_sp[['is_HSP', 'is_LSP']])
        # We'll add Ticker as a column
        df_combined['Ticker'] = tkr

        # We'll keep columns: DateTime (from index), O/H/L/C, SI, ASI, is_HSP, is_LSP
        # Convert index to string for export
        df_combined.reset_index(inplace=True)
        df_combined.rename(columns={'DateTime': 'Timestamp'}, inplace=True)

        # Collect rows for CSV
        for _, row in df_combined.iterrows():
            out_row = {
                'Ticker': tkr,
                'Timestamp': row['Timestamp'].strftime("%Y-%m-%d %H:%M:%S"),

                'Open': row['Open'],
                'High': row['High'],
                'Low': row['Low'],
                'Close': row['Close'],
                'Volume': row['Volume'],
                'SI': row['SI'],
                'ASI': row['ASI'],
                'is_HSP': row['is_HSP'],
                'is_LSP': row['is_LSP']
            }
            all_rows_asi.append(out_row)

    # Create final DataFrame for Algorithm 3
    df_algo3 = pd.DataFrame(all_rows_asi)
    # Sort by Ticker, then Timestamp
    df_algo3.sort_values(['Ticker', 'Timestamp'], inplace=True)
    df_algo3.to_csv("../colab/2021_selectedTickers/Algorithm3_SI_ASI.csv",
                    index=False)
    log_message(f"[INFO] SI & ASI results (Algorithm 3) saved to 'Algorithm3_SI_ASI.csv'.")


if __name__ == "__main__":
    main()
