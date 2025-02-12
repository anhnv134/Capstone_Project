"""##**Algorithm 5: Dynamic Stop-Loss, Profit Target, and Time-Based Exit**

**Goal:**
Our goal here is once we are in a position, we will manage it using a volatility-based trailing stop, set profit targets at next Fibonacci extension levels, and ensure no positions are held past the session end time.


**Inputs:**
- Open positions
- `ATR` or another volatility measure
- End-of-day exit requirement (no overnight holding)
- Trailing increment (e.g., `ATR * factor` for trailing stop)


**Pseudocode Steps:**

for each open position:
    entry_price = position.entry_price
    # set initial stop (e.g., entry_price - (ATR*X) for long)
    stop_price = calculate_initial_stop(entry_price, ATR, direction=long_or_short)

    # profit target example: use next Fibonacci extension (e.g., 1.272)
    #If long:
    profit_target = fibonacci_levels[1.272]

    # Update stops dynamically:
    # If price moves favorably (distance_to_entry > ATR), raise stop incrementally:
    if direction == 'LONG':
       unrealized_profit = current_price - entry_price
       if unrealized_profit > ATR:
           # move stop up by a fraction of unrealized gain or to the last LSP-based SAR
           stop_price = max(stop_price, entry_price + (unrealized_profit/2)) # just example logic
    else:  # for short positions:
       unrealized_profit = entry_price - current_price
       if unrealized_profit > ATR:
           # Move stop down by a fraction of unrealized gain
           stop_price = min(stop_price, entry_price - (unrealized_profit / 2))



    # check conditions to exit
    if current_price <= stop_price and direction == 'LONG':
        close_position(instrument)
    if current_price >= stop_price and direction == 'SHORT':
        close_position(instrument)

    if current_price >= profit_target and direction == 'LONG':
        close_position(instrument)
    if current_price <= profit_target and direction == 'SHORT':
        close_position(instrument)

    # End-of-day flat:
    if current_time >= session_end_time:
        close_position(instrument)

**Output:**

Positions managed dynamically. This ensures no overnight exposure and protecting profits.
"""

# Algorithm 5

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

# Fibonacci levels
FIB_BREAKOUT_UP = 'Fib_0.618'
FIB_BREAKOUT_DOWN = 'Fib_0.618_downside'
FIB_TARGET = 'Fib_1.272'

ATR_PERIOD = 14
SESSION_END = time(15, 40)  # Exit any open trades after market close
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
# HELPER :MULTI-BAR BREAKOUT CONFIRMATION FUNCTIONS
###############################################################################

def multi_bar_breakout_confirm(df, idx, fib_price, bars=2, buffer_pct=0.0, direction='above'):
    """
    Returns True if the last 'bars' bars' Close meets the condition relative to fib_price.
    If direction='above', all 'bars' closes must be above fib_price*(1+buffer_pct).
    If direction='below', all 'bars' closes must be below fib_price*(1-buffer_pct).
    bufer_pct is a decimal (e.g. 0.003  => 0.3%)

    """
    if idx < bars - 1:
        return False

    # We'll define the threshold
    if direction == 'above':
        threshold = fib_price * (1 + buffer_pct)
    else:
        threshold = fib_price * (1 - buffer_pct)

    recent_df = df.iloc[idx - bars + 1: idx + 1]

    if direction == 'above':
        return all(recent_df['Close'] > threshold)
    else:  # direction='below'
        return all(recent_df['Close'] < threshold)


###############################################################################
# PART B. ADVANCED ALGO 5
###############################################################################
def advanced_breakout_signals_algo5(
        df_5m,
        df_15m,
        fib_up_key=FIB_BREAKOUT_UP,
        fib_down_key=FIB_BREAKOUT_DOWN,
        fib_target_key=FIB_TARGET,
        adx_threshold=35,
        volume_spike_threshold=9999999.0,
        rsi_max_long=80,
        rsi_min_short=20,
        partial_exit_ratio=0.5,
        trailing_stop_mult=0.1,
        transaction_cost_per_trade=10.0,
        risk_per_trade=500.0,
        bars_confirm=2,
        fib_buffer=0.01
):
    """

      1. Checks fib_up_key, fib_down_key breakouts
      2. ADX Filter
      3. RSI Filter
      4. partial exit, trailing stop, fib target
      5. end-of-day forced exit

    """

    # Copy to avoid altering original
    df_5m = df_5m.copy()

    # Verified Required columns
    required = {
        'Timestamp', 'High', 'Low', 'Close',
        'Volume', 'ATR', 'VolMean', 'VolStd',
        'ASI', 'MACD_Hist', 'RSI', 'EMA_50', 'EMA_50_Slope',
        fib_up_key, fib_down_key, fib_target_key,
        'has_HSP', 'has_LSP'
    }
    missing = required - set(df_5m.columns)
    if missing:
        log_message(f"[ERROR] Missing columns in df_5m: {missing}")
        return []

    # ATR-based Filter
    atr_cut = dynamic_atr_threshold(df_5m['ATR'].dropna(), quantile=0.7)

    #  ADX from 15-min
    df_15m = df_15m.copy()
    if not {'High', 'Low', 'Close', 'Timestamp'}.issubset(df_15m.columns):
        log_message("[ERROR] df_15m missing required columns (Timestamp, High, Low, Close).")
        return []

    df_15m.set_index('Timestamp', inplace=True)
    df_15m['ADX'] = compute_adx(df_15m[['High', 'Low', 'Close']], period=14)
    df_15m.reset_index(inplace=True)

    trades = []
    position = None
    entry_price = None
    stop_loss = None
    position_size = 0
    partial_exit_done = False
    fib_target = None

    # Iterate over each bar in 5-min data
    for i, row in df_5m.iterrows():

        ts = row['Timestamp']
        # 1) Time Filter
        if not time_filter_ok(ts):
            continue
        current_price = row['Close']

        # 2) Volume condition:
        if pd.isna(row['Volume']) or (row['Volume'] < volume_spike_threshold):
            continue

        # 3) ATR Filter
        if pd.isna(row['ATR']) or (row['ATR'] < atr_cut):
            continue

        # 4) ADX filter from 15-min data
        df_15m_cut = df_15m[df_15m['Timestamp'] <= ts]
        if df_15m_cut.empty:
            continue
        adx_val = df_15m_cut.iloc[-1]['ADX']
        if pd.isna(adx_val) or (adx_val < adx_threshold):
            continue

        # 5) gather indicators
        fib_up = row[fib_up_key]
        fib_dn = row[fib_down_key]
        fib_ext = row[fib_target_key]
        has_hsp = bool(row['has_HSP'])
        has_lsp = bool(row['has_LSP'])
        ema_50 = row['EMA_50']
        slope_50 = row['EMA_50_Slope']
        macd_hist = row['MACD_Hist']
        rsi_val = row['RSI']
        asi_val = row['ASI']  # (if needed for debugging or more conditions)

        # 6) Trend conditions
        uptrend_condition = (current_price > ema_50) and (slope_50 > 0)
        downtrend_condition = (current_price < ema_50) and (slope_50 < 0)

        """
        7) MACD conditions
        We also want to see that macd_hist is above 0 (for bullish) or below 0 (for bearish).
        We compre macd_hist to the previous bar, but that requires accessing i-1 safely.
        """
        if i > 0:
            prev_macd_hist = df_5m.iloc[i - 1]['MACD_Hist']
        else:
            prev_macd_hist = 0
        macd_long = (macd_hist > 0) and (macd_hist > prev_macd_hist)
        macd_short = (macd_hist < 0) and (macd_hist < prev_macd_hist)

        # 8) RSI conditions: to avoid overbought or oversold extremes
        rsi_long = (rsi_val < rsi_max_long)
        rsi_short = (rsi_val > rsi_min_short)

        # --------------------------
        # TO CHECK ENTRY CONDITIONS
        # --------------------------
        breakout_long = (
                multi_bar_breakout_confirm(df_5m, i, fib_up, bars=bars_confirm, buffer_pct=fib_buffer,
                                           direction='above')

                and has_hsp
                and uptrend_condition
                and macd_long
                and rsi_long
        )

        breakout_short = (
                multi_bar_breakout_confirm(df_5m, i, fib_dn, bars=bars_confirm, buffer_pct=fib_buffer,
                                           direction='below')

                and has_lsp
                and downtrend_condition
                and macd_short
                and rsi_short
        )

        # ----------------------------------------------------
        # ENTRY LOGIC with Position Sizing
        # -----------------------------------------------------
        # ENTER position if none
        if position is None:
            if breakout_long:
                atr_val = row['ATR']
                if atr_val <= 0:
                    continue
                # use position sizing logic
                qty = position_sizing(risk_per_trade, atr_val, trailing_stop_mult)

                if qty <= 0:
                    continue  # ATR or risk calculation invalid

                position = 'long'
                entry_price = current_price
                position_size = qty

                # Set the initial stop loss at (entry - trailing_stop_mult * ATR)
                stop_loss = entry_price - (trailing_stop_mult * atr_val)
                partial_exit_done = False
                fib_target = fib_ext

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
                if qty <= 0:
                    continue

                position = 'short'
                entry_price = current_price
                position_size = qty

                # Set the initial stop loss at (entry + trailing_stop_mult * ATR)
                stop_loss = entry_price + (trailing_stop_mult * atr_val)
                partial_exit_done = False
                fib_target = fib_ext
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

            # end-of-day forced exit
            if ts.time() >= SESSION_END:
                trades.append({
                    'type': 'EXIT_EOD',
                    'side': ('SELL' if position == 'long' else 'BUY'),
                    'timestamp': ts,
                    'price': current_price,
                    'size': position_size,
                    'transaction_cost': transaction_cost_per_trade
                })
                position = None
                continue

            if position == 'long':

                # Partial exit if price has moved at least +1 * ATR from entry
                if (not partial_exit_done) and (current_price >= entry_price + current_atr):
                    qty_exit = int(position_size * partial_exit_ratio)
                    if qty_exit > 0:
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

                # if unrealized profit > ATR => move stop up by half that profit
                profit = current_price - entry_price
                if profit > current_atr:
                    stop_loss = max(stop_loss, entry_price + (profit * 0.5))

                # Update trailing stop (only upwards)
                new_stop = current_price - (trailing_stop_mult * current_atr)
                if new_stop > stop_loss:
                    stop_loss = new_stop

                # Stop-out condition
                if current_price < stop_loss:
                    trades.append({
                        'type': 'EXIT_STOP',
                        'side': 'SELL_STOP',
                        'timestamp': ts,
                        'price': stop_loss,
                        'size': position_size,
                        'transaction_cost': transaction_cost_per_trade
                    })
                    position = None
                    continue
                # fib target check
                if fib_target and (current_price >= fib_target):
                    trades.append({
                        'type': 'EXIT_PROFIT',
                        'side': 'SELL',
                        'timestamp': ts,
                        'price': fib_target,
                        'size': position_size,
                        'transaction_cost': transaction_cost_per_trade
                    })
                    position = None
                    continue

            elif position == 'short':
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

                profit = entry_price - current_price
                if profit > current_atr:
                    stop_loss = min(stop_loss, entry_price - (profit * 0.5))
                # Update trailing stop (only downwards)
                new_stop = current_price + (trailing_stop_mult * current_atr)
                if new_stop < stop_loss:
                    stop_loss = new_stop

                # Stop-out condition
                if current_price > stop_loss:
                    trades.append({
                        'type': 'EXIT_STOP',
                        'side': 'BUY_STOP',
                        'timestamp': ts,
                        'price': stop_loss,
                        'size': position_size,
                        'transaction_cost': transaction_cost_per_trade
                    })
                    position = None
                    continue
                # fib target check
                if fib_target and (current_price <= fib_target):
                    trades.append({
                        'type': 'EXIT_PROFIT',
                        'side': 'BUY',
                        'timestamp': ts,
                        'price': fib_target,
                        'size': position_size,
                        'transaction_cost': transaction_cost_per_trade
                    })
                    position = None
                    continue

    # final forced exit if still open at last bar
    if position is not None:
        last_bar = df_5m.iloc[-1]
        final_exit_price = last_bar['Close']
        trades.append({
            'type': 'EXIT_EOD_FINAL',
            'side': ('SELL' if position == 'long' else 'BUY'),
            'timestamp': last_bar['Timestamp'],
            'price': last_bar['Close'],
            'size': position_size,
            'transaction_cost': transaction_cost_per_trade
        })

    return trades


###############################################################################
# PART C. MAIN
###############################################################################
def main():
    data_5m_path = "C:/Users/vuanh/Downloads/colab/2021_selectedTickers/Enriched_5Min_Data.csv"
    out_path = "C:/Users/vuanh/Downloads/colab/2021_selectedTickers/Algo5_Trades.csv"

    if not os.path.exists(data_5m_path):
        log_message(f"[ERROR] 5-min data CSV not found: {data_5m_path}")
        return

    df_5m_all = pd.read_csv(data_5m_path)
    if 'Timestamp' not in df_5m_all.columns:
        raise KeyError("No 'Timestamp' column in data CSV")

    df_5m_all['Timestamp'] = pd.to_datetime(df_5m_all['Timestamp'], errors='coerce')
    df_5m_all.dropna(subset=['Timestamp'], inplace=True)
    df_5m_all.sort_values(['Ticker', 'Timestamp'], inplace=True)

    # Load thresholds
    thresholds_csv = "C:/Users/vuanh/Downloads/colab/2021_selectedTickers/DataDrivenThresholds.csv"
    if not os.path.exists(thresholds_csv):
        log_message("[WARNING] No thresholds CSV, using fallback defaults.")
        ticker_threshold_map = {}
    else:
        df_thr = pd.read_csv(thresholds_csv)
        ticker_threshold_map = {}
        for _, row in df_thr.iterrows():
            tkr = row['Ticker']
            ticker_threshold_map[tkr] = {
                'adx_threshold': float(row['ADX_Threshold']),
                'volume_spike_threshold': float(row['Volume_Spike_Threshold']),
                'rsi_upper': float(row['RSI_Upper']),
                'rsi_lower': float(row['RSI_Lower'])
            }

    all_trades = []

    for ticker in CHOSEN_TICKERS:
        df_sub = df_5m_all.loc[df_5m_all['Ticker'] == ticker].copy()
        if df_sub.empty:
            log_message(f"[WARNING] No data for {ticker}")
            continue

        # Build 15-min data
        df_sub.set_index('Timestamp', inplace=True)
        agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
        if 'Volume' in df_sub.columns:
            agg_dict['Volume'] = 'sum'
        df_15m = df_sub.resample('15min').agg(agg_dict)
        df_15m.reset_index(inplace=True)
        df_sub.reset_index(inplace=True)

        # fallback thresholds
        thr = ticker_threshold_map.get(ticker, None)
        if thr is None:
            adx_thr = 45
            vol_thr = 9999999
            rsi_up = 80
            rsi_down = 20
        else:
            adx_thr = thr['adx_threshold']
            vol_thr = thr['volume_spike_threshold']
            rsi_up = thr.get('rsi_upper', 50)
            rsi_down = thr.get('rsi_lower', 50)

        # call the advanced function
        trades_for_ticker = advanced_breakout_signals_algo5(
            df_5m=df_sub,
            df_15m=df_15m,
            fib_up_key=FIB_BREAKOUT_UP,
            fib_down_key=FIB_BREAKOUT_DOWN,
            fib_target_key=FIB_TARGET,
            adx_threshold=adx_thr,  # original value is adx_thr
            volume_spike_threshold=vol_thr,
            rsi_max_long=rsi_up,
            rsi_min_short=rsi_down,
            partial_exit_ratio=0.5,  # original is 0.5
            trailing_stop_mult=0.1,  # orginal is 1.0
            transaction_cost_per_trade=10.0,
            # bars_confirm  = 2,
            fib_buffer=0.008,  # 0.008
            risk_per_trade=500.0  # original is 1000.0
        )

        for trd in trades_for_ticker:
            trd['Ticker'] = ticker
        all_trades.extend(trades_for_ticker)

    df_trades = pd.DataFrame(all_trades)
    if not df_trades.empty:
        df_trades.sort_values(['timestamp', 'Ticker'], inplace=True)
        df_trades.to_csv(out_path, index=False)
        log_message(f"[INFO] Generated {len(df_trades)} trades => {out_path}")
    else:
        log_message(f"[INFO] No trades generated for any ticker.")


if __name__ == "__main__":
    main()
