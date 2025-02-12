"""##**Algorithm 4: Breakout Confimation and Entry Logic with Additional Indicators**

**Goal:**

<div style="text-align: justify;">

Our goal here is to trigger breakout trades only when Fibonacci-based barekout conditions align with 1)trend confirmation (via EMA), 2)volume participation, 3)momentum signals (ASI and MACD), and 4)avoid trades at extreme RSI levels (overbought or oversold)

</div>

**Inputs:**

- `current_price`: Current bar's closing price
- `fibonacci_levels`: Dictionary of Fibonacci levels (e.g., `{0.32: price, 0.5: price, 0.618: price, ...}`)
- `ASI_value`: Current Accumulative Swing Index value
- `has_HSP, has_LSP`: Booleans indicating recent High or Low Swing Points confirmed by ASI
- `ema_50`: 50-period EMA value at current bar
- `ema_slope(ema_50)`: Slope or trend direction inferred from `ema_50`
  *(positive slope = uptrend, negative slope = downtrend)*
- `current_volume, avg_volume`: Current and average volumes for volume spike check
- `macd_hist, previous_macd_hist`: Current and previous MACD histogram values for momentum confirmation
- `rsi`: Current RSI value
- `risk_per_trade, ATR`: For position sizing and stop calculation
- `fibonacci_levels[0.618_downside_level]`: Downward breakout Fibonacci level (mirroring the 0.618 ratio)
- `session_end_time`: Time at which all positions must be closed

**Pseudocode Steps:**

for instrument in instruments_to_trade:
    data = load_intraday_data(instrument)
    fibonacci_levels = precomputed_fib_levels[instrument]
    current_price = last_bar_close(data)

    #Retrieve ASI-related information
    ASI_value = get_current_ASI_value(data)
    has_HSP = detect_HSP(ASI_value) # True if a recent High Swing Point formed
    has_LSP = detect_LSP(ASI_value) # True if a recent Low Swing Point formed

    # Additional indicators
    ema_50 = get_ema(data, period=50)
    current_volume = last_bar_volume(data)
    avg_volume = average_volume(data, period=20)
    macd_hist = get_macd_histogram(data, fast=12, slow=26, signal=9)
    previous_macd_hist = get_previous_macd_hist(data)
    rsi = get_RSI(data, period=14)

    # Conditions :

    # Trend conditions:
    uptrend_condition = (current_price > ema_50) and (ema_slope(ema_50) > 0)
    downtrend_condition = (current_price < ema_50) and (ema_slope(ema_50) < 0)

    # Volume condition: ensure breakout is accompanied by a volume spike
    volume_condition = (current_volume > avg_volume * 1.2)

    # MACD conditions: ensure MACD histogram supports the trend direction
    macd_condition_long = (macd_hist > 0) and (macd_hist > previous_macd_hist)
    macd_condition_short = (macd_hist < 0) and (macd_hist < previous_macd_hist)

    # RSI conditions: avoid extreme overbought or oversold levels
    rsi_condigion_long = (rsi < 80)
    rsi_condition_short = (rsi > 20)

    # LONG Breakout Condition:
    # Price must cross above the 0.618 Fibonacci level, ASI must show a High Swing Point,
    # trend must be aligned to the upside, volume must confirm participation,
    # MACD msut show upward momentum, and RSI must not be too hgih.
    if (current_price > fibonacci_levels[0.618])
        and has_HSP
        and uptrend_condition
        and volume_condition
        and macd_condition_long
        and rsi_condition_long):

        quantity = position_sizing(instrument, risk_per_trade, ATR)
        place_order(instrument, 'BUY', quantity, current_market_price)

    # SHORT Breakout Condition:
    # Price crosses below 0.618 downside Fibonacci level, ASI shows a Low Swing Point,
    # trend aligned to the downside, good volume spike,
    # MACD downward momentum, and RSI not oversold.
    if (current_price < fibonacci_levels[0.618_downside_level])
        and has_LSP
        and downtrend_condition
        and volume_condition
        and macd_condition_short
        and rsi_condition_short):

        qunatity = position_sizing(instrument, risk_per_trade, ATR)
        place_order(instrument, 'SELL', quantity, current_market_price)

**Output:**

<p align="justify">


Trade execution occurs only when a confluence of Fibonacci breakout signals, ASI momentum confirmations, trend alignment (EMA), volume spikes, MACD momentum direction, and favourable RSI conditions are met. This reduces false positives and improves overall trade quality.
</p>
"""

# Algorithm 4

# Import necessary libraries
import os
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta

#####################################
# CONFIGURE LOGGING
#####################################
LOG_DIR = "C:/Users/vuanh/Downloads/colab"

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
# USER SETTINGS
###############################################################################
CHOSEN_TICKERS = ['HINDALCO', 'TATAMOTORS', 'JSWSTEEL', 'DELTACORP', 'OIL', 'INFY',
                  'RESPONIND', 'HDFCBANK', 'DHANI', 'ADANIPORTS', 'ABFRL', 'IRCTC',
                  'APOLLO', 'AUBANK', 'BALRAMCHIN', 'HDFC', 'TATACOMM', 'TATAMTRDVR',
                  'ZENSARTECH', 'DBL', 'NIITLTD', 'CAMLINFINE', 'M&M', 'INTELLECT']

FIB_KEY = 0.618
ATR_PERIOD = 14

# Lunch break window in NSE
LUNCH_START = time(12, 0)
LUNCH_END = time(13, 0)


###############################################################################
# PART A. UTILITY FUNCTIONS
###############################################################################
def wilder_atr(df, period=14,
               adjust=False):  # We use adjust=False to replicate Wilder's smoothing as it was originally defined
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


def compute_macd(df, fast=12, slow=26, signal=9):
    """
    Computes MACD line, signal line, and histogram.
    Returns three Series: (macd_line, signal_line, macd_hist).
    We will use df['Close'] for calculation.
    """
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist


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


def compute_ema_slope(series, window=1):
    """
    Quick slope approximation: slope = difference over 'window' bars.
    slope = (EMA_50[t] - EMA_50[t - 1])
    """
    return series.diff(periods=window)


def compute_asi(df, limit_move=None):
    """
    Compute Welles Wilder's Accumulative Swing Index (ASI  using a literal, piecewise approach.
    This method follows Wilder's original logic from:
    'New Concepts in Technical Trading Systems' (1978).

    Parameters
    __________
    df : pd.DataFrame
        Must have columns: 'Open', 'High', 'Low', 'Close'.
    limit_move : float or None, this is optional
        The maximum allowable price move for the instrument (futures-style limit move).
        If None, defaults to 1.0, effectively skipping the limit factor.

    Returns
    _______

    pd.Series
        A Pandas Series of ASI values (cumulative sum of SI).


    Notes
    -----
    Wilder's original formula for the Swing Index (SI) is:
        SI_t = 50 * (M / R) * (K / T)

    where:
      - M : A sign-based measure of price movement.
      - R : A piecewise range factor.
      - K : max(|High[t] - Close[t-1]|, |Low[t] - Close[t-1]|).
      - T : The limit_move. If not using limit moves, T can be 1.
      - 50 is a scaling constant.

    Then:
        ASI_t = ASI_(t-1) + SI_t.

    The piecewise logic (as in Wilder's book):

    1) M (sign-based):
       If today's Close >= yesterday's Close:
           M = (Close - Close_prev)
               + 0.5 * (Close - Open)
               + 0.25 * (Close_prev - Open)
         else:
           M = (Close - Close_prev)
               - 0.5 * (Close - Open)
               - 0.25 * (Close_prev - Open)

    2) R (piecewise):
       Let move_up   = (High - Close_prev)
           move_dwon = (Close_prev - Low)

       If move_up > move_down and move_up > 0:
           R = (High - Close_prev)
               - 0.5 * (Close_prev - Low)
               + 0.25 * (Close_prev - Open)
       elif move_down > move_up and move_down > 0:
           R = (Close_prev - Low)
               - 0.5 * (High - Close_prev)
               + 0.25 * (Close_prev - Open)

       else:
           # default if neither condition above is satisfied
           R = (High - Low)
               + 0.25 * (Close_prev - Open)
    The use of a tine epsilon prevents division by zero.
    # Compute the piecewise ASI
    asi_series = compute_asi(df, limit_move=5.0)
    df['ASI'] = asi_series
    """
    df = df.copy()

    # Shifted values for previous day
    df['Close_prev'] = df['Close'].shift(1)
    df['Open_prev'] = df['Open'].shift(1)
    df['High_prev'] = df['High'].shift(1)
    df['Low_prev'] = df['Low'].shift(1)

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
            R = (H - Cp) - 0.5 * (Cp - L) + 0.25 * (Cp - O)
        # Condition 2
        elif move_down > move_up and move_down > 0:
            R = (Cp - L) - 0.5 * (H - Cp) + 0.25 * (Cp - O)
        else:
            # Default
            R = (H - L) + 0.25 * (Cp - O)

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


def detect_hsp_rolling(series, window=3):
    """
    Returns a boolean Series indicating which bars (indices) are High Swing Points (HSP).

    A bar i is considered an HSP if:
      1) i has at least 'window' bars before it and 'window' bars after it.
      2) series[i] > all of the 'window' bars before i, and
      2) series[i] > all of the 'window' bars after i.
      3) series[i] is strictly higher [no ties].


    Parameters
    ----------
    series : pd.Series (float)
        The numeric values (e.g., ASI, or price, or any oscillator).
    window : int
        Number of bars on each side to confirm a local maximum.

    Returns
    -------
    pd.Series of bool
        True where the bar is an HSP, False otherwise.
    """

    n = len(series)
    is_hsp = [False] * n

    for i in range(n):
        # We need at least 'window' bars before AND after index i
        if i < window or i > n - window - 1:
            continue

        # Current value
        val = series.iloc[i]

        # 'window' bars on the left (past)
        left_chunk = series.iloc[i - window: i]
        # 'window' bars on the right (future)
        right_chunk = series.iloc[i + 1: i + 1 + window]

        # Check if 'val' is strictly greater than all left_chunk and right_chunk
        if all(val > x for x in left_chunk) and all(val > x for x in right_chunk):
            is_hsp[i] = True

    return pd.Series(is_hsp, index=series.index, name='is_HSP')


def detect_lsp_rolling(series: pd.Series, window: int = 3) -> pd.Series:
    """
    Returns a boolean Series indicating which bars (indices) are Low Swing Points (LSPs).

    A bar i is considered an LSP if:
      1) i has at least 'window' bars before it and 'window' bars after it.
      2) series[i] < all of the 'window' bars before i, and
         AND series[i] < all of the 'window' bars after i.
      3) series[i] is strictly lower (no ties) than its neighbors in that local region.

    Parameters
    ----------
    series : pd.Series
        The numeric values (e.g. ASI, RSI, or price) from which to detect LSPs.
    window : int
        How many bars on each side to confirm a local minimum.

    Returns
    -------
    pd.Series (bool)
        A boolean Series (same index as 'series'),
        where True indicates the bar is a Low Swing Point (LSP).
    """
    n = len(series)
    is_lsp = [False] * n

    for i in range(n):
        # We need at least 'window' bars before index i
        # and 'window' bars after index i
        if i < window or i > n - window - 1:
            continue

        # Current bar's value
        val = series.iloc[i]

        # Bars immediately before i
        left_chunk = series.iloc[i - window: i]

        # Bars immediately after i
        right_chunk = series.iloc[i + 1: i + 1 + window]

        # Check if 'val' is strictly less than all bars in left_chunk & right_chunk
        if all(val < x for x in left_chunk) and all(val < x for x in right_chunk):
            is_lsp[i] = True

    return pd.Series(is_lsp, index=series.index, name='is_LSP')


def time_filter_ok(ts, open_skip=5, close_skip=5,
                   lunch_start=LUNCH_START, lunch_end=LUNCH_END):
    """
    Returns False if the timestamp is within:
      - the first 'open_skip' minutes after market open
      - the last 'close_skip' minutes before market close
      - the lunch window
    Otherwise returns True
    """
    t = ts.time()
    # Lunch window filter
    if lunch_start <= t < lunch_end:
        return False

    open_dt = ts.replace(hour=9, minute=15, second=0, microsecond=0)
    close_dt = ts.replace(hour=15, minute=30, second=0, microsecond=0)

    # Skip first X minutes after open
    if ts < open_dt + timedelta(minutes=open_skip):
        return False
    # Skip last X minutes before close
    if ts > close_dt - timedelta(minutes=close_skip):
        return False

    return True


def dynamic_volume_threshold(volume_ser, n_std=2.0):
    """Returns a threshold of mean(volume) + n_std * stdev(volume)."""
    return volume_ser.mean() + n_std * volume_ser.std()


def dynamic_atr_threshold(atr_ser, quantile=0.7):
    """Returns the ATR value at the given quantile."""
    return atr_ser.quantile(quantile)


def position_sizing(risk_per_trade, current_atr, stop_multiple=0.8):
    """
    Returns an integer number of shares (or contracts) to trade, based on:
      - risk_per_trade: The maximum risk we're willing to lose on this trade (e.g., 1000).
      - current_atr: ATR for the current bar (float).
      - stop_multiple: multiple of ATR that we'll  place our  stop from the entry price.

    Formula:
        stop_distance = stop_multiple * current_atr
        position_size = risk_per_trade / stop_distance
    Rounded down to ensure we  don't exceed risk_per_trade if the stop is hit.

    Returns:
      int: The position size
    """
    if current_atr <= 0:
        return 0

    stop_distance = stop_multiple * current_atr
    if stop_distance <= 0:
        return 0

    size = risk_per_trade / stop_distance
    size = int(size)  # floor to ensure we don't exceed risk
    return size


###############################################################################
# PART B. ADVANCED ALGO WITH FIB + EMA + ASI + MACD + RSI
###############################################################################
def advanced_breakout_signals(
        df_5m,
        df_15m,
        fib_level_up_key='Fib_0.618',
        fib_level_dn_key='Fib_0.618_downside',
        adx_threshold=20,
        volume_spike_threshold=9999999.0,
        rsi_max_long=80,
        rsi_min_short=20,
        partial_exit_ratio=0.5,
        trailing_stop_mult=0.1,  # original is 1.0
        transaction_cost_per_trade=10.0,
        risk_per_trade=500.0  # original is 1000
):
    """
    Generates signals only when:
      1. Price breaks above (or below) a key Fib level (0.618).
      2. ASI indicates HSP or LSP.
      3. Trend alignment: current_price vs. EMA(50) and slope sign.
      4. Volume spike vs. average volume.
      5. MACD histogram confirms momentum.
      6. RSI is not extreme.
      7. Additional conditions on ADX or ATR may still apply if needed.
    """

    # Define market close time
    market_close = time(15, 40)

    # Copy to avoid altering original
    df_5m = df_5m.copy()

    # Basic sanity checks
    required_cols = {
        'Timestamp', 'High', 'Low', 'Close',
        'Volume', 'ATR', 'VolMean', 'VolStd',
        'ASI', 'MACD_Hist', 'RSI', 'EMA_50', 'EMA_50_Slope',
        fib_level_up_key, fib_level_dn_key,
        'has_HSP', 'has_LSP'
    }
    missing_cols = required_cols - set(df_5m.columns)
    if missing_cols:
        log_message(f"[ERROR] Missing columns in df_5m: {missing_cols}")
        return []

    # Compute ATR threshold from 5-min data
    atr_cutoff = dynamic_atr_threshold(df_5m['ATR'].dropna(), quantile=0.7)

    # Also compute ADX on 15-min data
    df_15m = df_15m.copy()
    if not {'High', 'Low', 'Close', 'Timestamp'}.issubset(df_15m.columns):
        log_message("[ERROR] df_15m missing required columns (Timestamp, High, Low, Close).")
        return []

    df_15m.set_index('Timestamp', inplace=True)
    df_15m['ADX'] = compute_adx(df_15m[['High', 'Low', 'Close']], 14)
    df_15m.reset_index(inplace=True)

    trades = []
    position = None
    entry_price = None
    stop_loss = None
    position_size = 0
    partial_exit_done = False

    # Iterate over each bar in 5-min data
    for i, row in df_5m.iterrows():
        ts = row['Timestamp']
        current_price = row['Close']

        # 1) Time Filter
        if not time_filter_ok(ts):
            continue

        # 2) Volume condition:
        if pd.isna(row['Volume']) or (row['Volume'] < volume_spike_threshold):
            continue

        # 3) ATR Filter
        if pd.isna(row['ATR']) or (row['ATR'] < atr_cutoff):
            continue

        # 4) ADX filter from 15-min data
        df_15m_cut = df_15m[df_15m['Timestamp'] <= ts]
        if df_15m_cut.empty:
            continue
        adx_val = df_15m_cut.iloc[-1]['ADX']
        if pd.isna(adx_val) or (adx_val < adx_threshold):
            continue

        # 5) Pull out required indicator values
        fib_up = row[fib_level_up_key]  # e.g., row['Fib_0.619']
        fib_dn = row[fib_level_dn_key]  # e.g., row['Fib_0.618_downside']
        has_hsp = bool(row['has_HSP'])
        has_lsp = bool(row['has_LSP'])
        ema_50 = row['EMA_50']
        ema_slope_50 = row['EMA_50_Slope']
        macd_hist = row['MACD_Hist']
        rsi_val = row['RSI']
        asi_val = row['ASI']  # (if needed for debugging or more conditions)

        # 6) Trend conditions
        uptrend_condition = (current_price > ema_50) and (ema_slope_50 > 0)
        downtrend_condition = (current_price < ema_50) and (ema_slope_50 < 0)

        """
        7) MACD conditions
        We also want to see that macd_hist is above 0 (for bullish) or below 0 (for bearish).
        We compre macd_hist to the previous bar, but that requires accessing i-1 safely.
        """
        if i > 0:
            prev_macd_hist = df_5m.iloc[i - 1]['MACD_Hist']
        else:
            prev_macd_hist = 0
        macd_condition_long = (macd_hist > 0) and (macd_hist > prev_macd_hist)
        macd_condition_short = (macd_hist < 0) and (macd_hist < prev_macd_hist)

        # 8) RSI conditions: to avoid overbought or oversold extremes
        rsi_condition_long = (rsi_val < rsi_max_long)
        rsi_condition_short = (rsi_val > rsi_min_short)

        # --------------------------
        # TO CHECK ENTRY CONDITIONS
        # --------------------------
        breakout_long = (
                (current_price > fib_up) and
                has_hsp and
                uptrend_condition and
                macd_condition_long and
                rsi_condition_long
        )

        breakout_short = (
                (current_price < fib_dn) and
                has_lsp and
                downtrend_condition and
                macd_condition_short and
                rsi_condition_short
        )

        # ----------------------------------------------------
        # ENTRY LOGIC with Position Sizing
        # -----------------------------------------------------
        # ENTER position if none
        if position is None:
            if breakout_long:
                atr_val = row['ATR']
                # use position sizing logic
                qty = position_sizing(risk_per_trade, atr_val, trailing_stop_mult)

                if qty == 0:
                    continue  # ATR or risk calculation invalid

                position = 'long'
                entry_price = current_price
                position_size = qty

                # Set the initial stop loss at (entry - trailing_stop_mult * ATR)
                stop_loss = entry_price - (trailing_stop_mult * atr_val)
                partial_exit_done = False

                trades.append({
                    'type': 'ENTRY',
                    'side': 'BUY',
                    'timestamp': ts,
                    'price': entry_price,
                    'size': position_size,
                    'stop_loss': stop_loss,
                    'transaction_cost': transaction_cost_per_trade
                })

            elif breakout_short:
                atr_val = row['ATR']
                if atr_val <= 0:
                    continue
                qty = position_sizing(risk_per_trade, atr_val, trailing_stop_mult)

                position = 'short'
                entry_price = current_price
                position_size = qty

                # Set the initial stop loss at (entry + trailing_stop_mult * ATR)
                stop_loss = entry_price + (trailing_stop_mult * atr_val)
                partial_exit_done = False
                trades.append({
                    'type': 'ENTRY',
                    'side': 'SELL',
                    'timestamp': ts,
                    'price': entry_price,
                    'size': position_size,
                    'stop_loss': stop_loss,
                    'transaction_cost': transaction_cost_per_trade
                })

        else:
            # If already in a position, we manage trailing stop and partial exists
            current_atr = row['ATR']

            if position == 'long':
                # Update trailing stop (only upwards)
                new_stop = current_price - (trailing_stop_mult * current_atr)
                if new_stop > stop_loss:
                    stop_loss = new_stop

                # Stop-out condition
                if current_price < stop_loss:
                    trades.append({
                        'type': 'EXIT',
                        'side': 'SELL_STOP',
                        'timestamp': ts,
                        'price': stop_loss,
                        'size': position_size,
                        'transaction_cost': transaction_cost_per_trade
                    })
                    position = None
                    continue

                # Partial exit if price has moved at least +1 * ATR from entry
                if (not partial_exit_done) and (current_price >= entry_price + current_atr):
                    qty_exit = int(position_size * partial_exit_ratio)
                    trades.append({
                        'type': 'PARTIAL_EXIT',
                        'side': 'SELL',
                        'timestamp': ts,
                        'price': current_price,
                        'size': qty_exit,
                        'transaction_cost': transaction_cost_per_trade
                    })
                    position_size -= qty_exit
                    partial_exit_done = True

            elif position == 'short':
                # Update trailing stop (only downwards)
                new_stop = current_price + (trailing_stop_mult * current_atr)
                if new_stop < stop_loss:
                    stop_loss = new_stop

                # Stop-out condition
                if current_price > stop_loss:
                    trades.append({
                        'type': 'EXIT',
                        'side': 'BUY_STOP',
                        'timestamp': ts,
                        'price': stop_loss,
                        'size': position_size,
                        'transaction_cost': transaction_cost_per_trade
                    })
                    position = None
                    continue

                # Partial exit if price has moved at least 1 * ATR from entry (downside)
                if (not partial_exit_done) and (current_price <= entry_price - current_atr):
                    qty_exit = int(position_size * partial_exit_ratio)
                    trades.append({
                        'type': 'PARTIAL_EXIT',
                        'side': 'BUY',
                        'timestamp': ts,
                        'price': current_price,
                        'size': qty_exit,
                        'transaction_cost': transaction_cost_per_trade
                    })
                    position_size -= qty_exit
                    partial_exit_done = True

    # Close any open position at end-of-day
    if position is not None:
        last_bar = df_5m.iloc[-1]
        final_exit_price = last_bar['Close']
        trades.append({
            'type': 'EXIT_EOD',
            'side': ('SELL' if position == 'long' else 'BUY'),
            'timestamp': last_bar['Timestamp'],
            'price': final_exit_price,
            'size': position_size,
            'transaction_cost': transaction_cost_per_trade
        })

    return trades


###############################################################################
# PART C. MAIN FUNCTION
###############################################################################
def main():
    """
    Full workflow:
      1. Read 5-min enriched data from CSV.
      2. For each Ticerk, filter rows, ensure 'Timestamp' is present, convert to datetime.
      3. Resample to 15-min for ADX calculations (or any higher timeframe steps if 15-min not enough).
      4. Call 'advanced_breakout_signals' to get trades.
      5. Compile results into a single DataFrame -> CSV.
    """

    enriched_data_path_5m = "C:/Users/vuanh/Downloads/colab/2021_selectedTickers/Enriched_5Min_Data.csv"
    signals_out_path = "C:/Users/vuanh/Downloads/colab/2021_selectedTickers/Algo4_Trades.csv"

    log_message("[INFO] Loading enriched 5-min data....")
    df_5m_all = pd.read_csv(enriched_data_path_5m)
    cols_str = ", ".join(df_5m_all.columns)

    log_message(f"[DEBUG] Initial columns: {cols_str}")

    # rename or fix the time column
    if 'Timestamp' not in df_5m_all.columns:
        if 'Datetime' in df_5m_all.columns:
            df_5m_all.rename(columns={'Datetime': 'Timestamp'}, inplace=True)
        elif 'timestamp' in df_5m_all.columns:
            df_5m_all.rename(columns={'timestamp': 'Timestamp'}, inplace=True)
        else:
            raise KeyError("No recognized time column found. Please ensure 'Timestamp' is available.")

    # Convert to datetime
    df_5m_all['Timestamp'] = pd.to_datetime(df_5m_all['Timestamp'], errors='coerce')
    if df_5m_all['Timestamp'].isna().sum() > 0:
        log_message("[WARNING] Some invalid timestamps detected; dropping those rows...")
        df_5m_all.dropna(subset=['Timestamp'], inplace=True)

    sample_timestamps_str = str(df_5m_all['Timestamp'].head())
    log_message(f"[DEBUG] Parsed Timestamps: Sample:\n{sample_timestamps_str}")

    # sort entire Data Frame by (Ticker, Timestamp)
    df_5m_all.sort_values(['Ticker', 'Timestamp'], inplace=True)

    # 2) Load the thresholds from the CSV we generated
    thresholds_csv = "C:/Users/vuanh/Downloads/colab/2021_selectedTickers/DataDrivenThresholds.csv"
    if not os.path.exists(thresholds_csv):
        log_message(F"[WARNING] Thresholds CSV not found: {thresholds_csv}. Using default constants.")
        ticker_threshold_map = {}
    else:
        df_thr = pd.read_csv(thresholds_csv)
        # We build a dictionary
        ticker_threshold_map = {}
        for _, row in df_thr.iterrows():
            tkr = row['Ticker']
            ticker_threshold_map[tkr] = {
                'adx_threshold': float(row['ADX_Threshold']),
                'volume_spike_threshold': float(row['Volume_Spike_Threshold']),
                'rsi_upper': float(row['RSI_Upper']),
                'rsi_lower': float(row['RSI_Lower'])
            }

    # -----------------------------------------------------------------------
    # If we do not already have EMA_50, MACD_Hist, RSI, ASI,etc.
    # We compute them before the main loop, for each ticker.
    # -----------------------------------------------------------------------
    # We'll do a groupby on Ticker to compute indicators:
    def compute_indicators(subdf):
        subdf = subdf.copy()
        subdf.sort_values('Timestamp', inplace=True)
        # 1) Compute ATR
        if 'ATR' not in subdf.columns:
            subdf['ATR'] = wilder_atr(subdf[['High', 'Low', 'Close']], period=ATR_PERIOD)

        # 2) Compute 50 EMA + slope
        if 'EMA+50' not in subdf.columns:
            subdf['EMA_50'] = subdf['Close'].ewm(span=50, adjust=False).mean()
        if 'EMA_50_Slope' not in subdf.columns:
            subdf['EMA_50_Slope'] = compute_ema_slope(subdf['EMA_50'], window=1)

        # 3) Compute MACD histogram
        if 'MACD_Hist' not in subdf.columns:
            macd_line, signal_line, macd_hist = compute_macd(subdf, fast=12, slow=26, signal=9)
            subdf['MACD_Hist'] = macd_hist

        # 4) Compute RSI
        if 'RSI' not in subdf.columns:
            subdf['RSI'] = compute_rsi(subdf, period=14)

        # 5) Compute ASI
        if 'ASI' not in subdf.columns:
            subdf['ASI'] = compute_asi(subdf)

        # 5) Detect has_HSP/has_LSP in a rolling manner.
        if 'ASI' not in subdf.columns:
            log_message("[WARNING]: ASI missing. We cannot detect HSP/LSP.")
        else:
            if 'has_HSP' not in subdf.columns:
                subdf['has_HSP'] = detect_hsp_rolling(subdf['ASI'], window=3)
            if 'has_LSP' not in subdf.columns:
                subdf['has_LSP'] = detect_lsp_rolling(subdf['ASI'], window=3)

        # 7) Compute rolling volume stats if not present
        if 'VolMean' not in subdf.columns:
            subdf['VolMean'] = subdf['Volume'].rolling(20).mean()
        if 'VolStd' not in subdf.columns:
            subdf['VolStd'] = subdf['Volume'].rolling(20).std()

        # 8) define fib_0.618, fib_0.618_downside from existing columns
        if 'Fib_0.618' not in subdf.columns:
            log_message("[WARNING] 'Fib_0.618' column not found. Skipping.")
        if 'Fib_0.618_downside' not in subdf.columns:
            if 'Fib_0.382' in subdf.columns:
                log_message("[INFO] Creating Fib_0.618_Downside fallback from Fib_0.382.")
                subdf['Fib_0.618_downside'] = subdf['Fib_0.382']
            else:
                subdf['Fib_0.618_downside'] = subdf['Close'] * 0.99

        return subdf

    df_5m_all = df_5m_all.groupby('Ticker', group_keys=False).apply(lambda x: compute_indicators(x))

    df_5m_all.to_csv("C:/Users/vuanh/Downloads/colab/2021_selectedTickers/Enriched_5Min_Data.csv",
                     index=False)

    all_trades = []

    # We now loop over each Tciekr
    for ticker in CHOSEN_TICKERS:
        df_5m_tkr = df_5m_all.loc[df_5m_all['Ticker'] == ticker].copy()

        if df_5m_tkr.empty:
            log_message(f"[INFO] No data for {ticker}. Skipping.")
            continue

        # Double-check Timestamp existence
        if 'Timestamp' not in df_5m_tkr.columns:
            log_message(f"[ERROR] 'Timestamp' missing for {ticker}. Skipping.")
            continue

        # We will convert once more in case of unexpected merges.
        df_5m_tkr['Timestamp'] = pd.to_datetime(df_5m_tkr['Timestamp'], errors='coerce')
        df_5m_tkr.dropna(subset=['Timestamp'], inplace=True)
        if df_5m_tkr.empty:
            log_message(f"[INFO] All timestamps invalid for {ticker}. Skipping.")
            continue

        # Sort by time
        df_5m_tkr.sort_values('Timestamp', inplace=True)

        # Create 15-min bars for ADX (resample)
        df_5m_tkr.set_index('Timestamp', inplace=True)
        agg_dict = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        }
        # If volume column exists, include sum
        if 'Volume' in df_5m_tkr.columns:
            agg_dict['Volume'] = 'sum'

        df_15m_tkr = df_5m_tkr.resample('15min').agg(agg_dict)
        df_15m_tkr.reset_index(inplace=True)

        # Reset index on 5-min data for iteration
        df_5m_tkr.reset_index(inplace=True)

        # 4) Retrieve thresholds for this ticker (or fallback)
        thr_dict = ticker_threshold_map.get(ticker, None)
        if thr_dict is None:
            log_message(f"[WARNING] No thresholds found for {ticker}. Using default constants.")
            adx_thr = 20
            vol_thr = 9999999.0
            rsi_up = 80
            rsi_down = 20
        else:
            log_message(f"[INFO] Using thresholds for {ticker}: {thr_dict}.")
            adx_thr = thr_dict['adx_threshold']
            vol_thr = thr_dict['volume_spike_threshold']
            rsi_up = thr_dict['rsi_upper']
            rsi_down = thr_dict['rsi_lower']

        # Now we call  the advanced breakout function
        trades_for_ticker = advanced_breakout_signals(
            df_5m_tkr,
            df_15m_tkr,
            fib_level_up_key='Fib_0.618',
            fib_level_dn_key='Fib_0.618_downside',
            adx_threshold=adx_thr,  # original value is 20
            volume_spike_threshold=vol_thr,  # original value is 1.2
            rsi_max_long=rsi_up,  # original value is 80
            rsi_min_short=rsi_down,  # original value is 20
            partial_exit_ratio=0.5,
            trailing_stop_mult=1.0,
            transaction_cost_per_trade=10.0,
            risk_per_trade=1000.0
        )

        # Tag each trade with the ticker
        for trade_record in trades_for_ticker:
            trade_record['Ticker'] = ticker

        all_trades.extend(trades_for_ticker)

    # Convert trades list -> DataFrame, save
    df_trades = pd.DataFrame(all_trades)
    if not df_trades.empty:
        df_trades.sort_values(['timestamp', 'Ticker'], inplace=True)
        df_trades.to_csv(signals_out_path, index=False)
        log_message(f"[INFO] Generated {len(df_trades)} trade records. Saved to {signals_out_path}.")
    else:
        log_message("[INFO] No trades generated for any ticker.")


# Entry point
if __name__ == "__main__":
    main()