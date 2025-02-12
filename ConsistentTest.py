# import necessary libraries
import os
import glob
import logging
from datetime import datetime

#####################################
# CONFIGURE LOGGING
#####################################
LOG_DIR = "C:/Users/vuanh/Downloads/colab"

# Log file path
LOG_FILE = os.path.join(LOG_DIR, "Algo1_logfile.txt")


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


#####################################
# USER DEFINED PATHS
#####################################
BASE_2021_DIR = "C:/Users/vuanh/Downloads/2021"
BASE_2022_DIR = "E:/VuAnhData/2022"

#####################################
# CANDIDATE TICKERS
#####################################
candidate_tickers = ['HINDALCO', 'TATAMOTORS', 'SUPRIYA', 'JSWSTEEL', 'MOTHERSUMI',
                     'DELTACORP', 'OIL', 'INFY', 'RESPONIND', 'HDFCBANK', 'POONAWALLA',
                     'DHANI', 'ADANIPORTS', 'ABFRL', 'IRCTC', 'APOLLO', 'AUBANK', 'BALRAMCHIN',
                     'HDFC', 'TATACOMM', 'TATAMTRDVR', 'ZENSARTECH', 'LATENTVIEW', 'DBL',
                     'NIITLTD', 'CAMLINFINE', 'M&M', 'INTELLECT']


#####################################
# FIND MONTHLY FOLDERS
#####################################
def get_monthly_folders(base_dir: str) -> list:
    """
    Returns a sorted list of full paths to subfolders
    under base_dir (one per month).
    e.g, 2021/Cash Data January 2021, etc.

    """
    # Basci directory validation
    if not os.path.isdir(base_dir):
        log_message(f"Base directory '{base_dir}' does not exist.")
        return []
    month_folders = []
    for item in sorted(os.listdir(base_dir)):
        subp = os.path.join(base_dir, item)
        if os.path.isdir(subp):
            month_folders.append(subp)
    return month_folders


#####################################
# CHECK COVERAGE
#####################################
def coverage_in_one_year(tickers, year_base_dir) -> dict:
    """
    For each ticker in 'tickers', finds how many monthly CSVs exist
    in the 12 subfolders under 'year_base_dir'.
    Returns {ticker: count_of_months_found}.
    """

    # Identify monthly subfolders
    month_folders = get_monthly_folders(year_base_dir)
    coverage_map = {tkr: 0 for tkr in tickers}  # start each at 0

    for mdir in month_folders:
        log_message(f"Checking folder: {mdir}")
        # Gather all CSV files in this month's folder
        csv_files = glob.glob(os.path.join(mdir, "*.csv"))
        # Build a set of base filenames, e.g. "TICKER" from "TICKER.CSV"
        found_tickers = set()

        for cfile in csv_files:
            base = os.path.splitext(os.path.basename(cfile))[0]
            found_tickers.add(base)

        # For each candidate ticker, see if it's found
        for tkr in tickers:
            # Because some tickers might have special characters
            # we do exact match
            # If real tickers have parentheses or unusual characters,
            # we ensure they match the CSV naming exactly.
            # For standard naming: TKR.csv => base=TKR
            if tkr in found_tickers:
                coverage_map[tkr] += 1
    return coverage_map


def main():
    """
    Main Function to check coverage for candidate tickers across 2021 and 2022.

    """
    if not os.path.isdir(BASE_2021_DIR) or not os.path.isdir(BASE_2022_DIR):
        log_message("Either BASE_2021_DIR or BASE_2022_DIR does not exist.Existing")
        return
    # 1) Gather coverage in 2021
    coverage_2021 = coverage_in_one_year(candidate_tickers, BASE_2021_DIR)
    # 2) Gather coverage in 2022
    coverage_2022 = coverage_in_one_year(candidate_tickers, BASE_2022_DIR)

    # We require at lease 12 months coverage in each year.
    MIN_MONTHS_REQUIRED = 12

    consistent_tickers = []
    for tkr in candidate_tickers:
        c21 = coverage_2021[tkr]
        c22 = coverage_2022[tkr]
        log_message(f"{tkr}: months in 2021={c21}, months in 2022={c22}")
        # If tickers meets coverage in both 2021 and 2022
        if (c21 >= MIN_MONTHS_REQUIRED) and (c22 >= MIN_MONTHS_REQUIRED):
            consistent_tickers.append(tkr)

    log_message(f"\n=== Tickers with consistent coverage in both 2021 & 2022 ===")
    log_message(f"{consistent_tickers}")
    log_message(f"Total: {len(consistent_tickers)}")
    # print(f"Consistent Tickers: {consistent_tickers}")
    # print(f"Total Consisten Tickers: {len(consistent_tickers)}")


if __name__ == "__main__":
    main()