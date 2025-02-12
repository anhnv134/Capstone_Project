"""**Enriched_5Min_Data.csv**"""

# Combine Enriched_5Min_Data.csv
# Combine Algorithm2_FibinacciLevels.csv and Algorithm3_SI_ASI.csv

# import necessary libraries
import pandas as pd
from datetime import datetime

# Load data from CSV files
fib_path = "C:/Users/vuanh/Downloads/colab/2021_selectedTickers/Algorithm2_FibonacciLevels.csv"
si_asi_path = "C:/Users/vuanh/Downloads/colab/2021_selectedTickers/Algorithm3_SI_ASI.csv"


# Load Algorithm 2 Fibonacci Levels
def load_fib_levels(file_path):
    fib_df = pd.read_csv(file_path, parse_dates=['Date', 'LowTime', 'HighTime'])
    fib_dict = {}
    for _, row in fib_df.iterrows():
        ticker = row['Ticker']
        date = row['Date'].date()
        fib_info = {
            'up_swing': row['UpSwing'],
            'swing_low': row['SwingLow'],
            'swing_high': row['SwingHigh'],
            'low_time': row['LowTime'],
            'high_time': row['HighTime'],
            'fib_levels': {
                0.382: row['Fib_0.382'],
                0.5: row['Fib_0.5'],
                0.618: row['Fib_0.618'],
                1.0: row['Fib_1.0'],
                1.272: row['Fib_1.272'],
                1.618: row['Fib_1.618'],
            }
        }

        if ticker not in fib_dict:
            fib_dict[ticker] = {}
        fib_dict[ticker][date] = fib_info
    return fib_dict


# Load Algorithm 3 SI/ASI data
def load_si_asi_data(file_path):
    si_asi_df = pd.read_csv(file_path, parse_dates=['Timestamp'])
    si_asi_df['Date'] = si_asi_df['Timestamp'].dt.date
    return si_asi_df


# Merge SI/ASI data with Fibonacci Levels
def merge_data(si_asi_df, fib_dict):
    enriched_data = []
    for ticker in si_asi_df['Ticker'].unique():
        ticker_df = si_asi_df[si_asi_df['Ticker'] == ticker]
        for date in ticker_df['Date'].unique():
            daily_df = ticker_df[ticker_df['Date'] == date].copy()
            if ticker in fib_dict and date in fib_dict[ticker]:
                fib_info = fib_dict[ticker][date]
                for key, value in fib_info['fib_levels'].items():
                    daily_df[f'Fib_{key}'] = value
                daily_df['UpSwing'] = fib_info['up_swing']
                daily_df['SwingLow'] = fib_info['swing_low']
                daily_df['SwingHigh'] = fib_info['swing_high']
                daily_df['LowTime'] = fib_info['low_time']
                daily_df['HighTime'] = fib_info['high_time']
                enriched_data.append(daily_df)
    return pd.concat(enriched_data, ignore_index=True) if enriched_data else pd.DataFrame()


# Main Execution
if __name__ == "__main__":
    # Load data
    fib_results = load_fib_levels(fib_path)
    si_asi_df = load_si_asi_data(si_asi_path)

    # Merge data
    enriched_df = merge_data(si_asi_df, fib_results)

    # Save enriched data to a CSV file for further use
    enriched_df.to_csv("C:/Users/vuanh/Downloads/colab/2021_selectedTickers/Enriched_5Min_Data.csv",
                       index=False)
    print(f"Enriched data saved to Enriched_5Min_Data.csv with {len(enriched_df)} rows.")