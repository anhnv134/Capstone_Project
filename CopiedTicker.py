"""##**Copied Selected Tickers**"""

# import libraries
import os
import shutil
from collections import defaultdict
from datetime import datetime

# ----------------------------
# USER CONFIGURATIONS
# ----------------------------

# Base directories for 2021 and 2022 data
BASE_2021_DIR = "C:/Users/vuanh/Downloads/2021"
# BASE_2022_DIR = "/content/drive/My Drive/2022"

# Target directory where the organized 2021_24Tickers data will reside
TARGET_DIR = "C:/Users/vuanh/Downloads/colab/2021_selectedTickers"

# List of monthly subfolders (ensure these match exactly with our actual folder names)
monthly_folders = [

    "Cash Data January 2021",
    "Cash Data February 2021",
    "Cash Data March 2021",
    "Cash Data April 2021",
    "Cash Data May 2021",
    "Cash Data June 2021",
    "Cash Data July 2021",
    "Cash Data August 2021",
    "Cash Data September 2021",
    "Cash Data October 2021",
    "Cash Data November 2021",
    "Cash Data December 2021"
]

# List of specified tickers
tickers = ['HINDALCO', 'TATAMOTORS', 'JSWSTEEL', 'DELTACORP', 'OIL', 'INFY',
           'RESPONIND', 'HDFCBANK', 'DHANI', 'ADANIPORTS', 'ABFRL', 'IRCTC',
           'APOLLO', 'AUBANK', 'BALRAMCHIN', 'HDFC', 'TATACOMM', 'TATAMTRDVR',
           'ZENSARTECH', 'DBL', 'NIITLTD', 'CAMLINFINE', 'M&M', 'INTELLECT']

# Log file path
LOG_DIR = "C:/Users/vuanh/Downloads/colab"
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


def ensure_directory_exists(path):
    """
    Ensures that the given directory exists. Creates it if it doesn't.

    """
    if not os.path.exists(path):
        os.makedirs(path)
        log_message(f"Created directory: {path}")
    else:
        log_message(f"Directory already exists: {path}")


def ensure_monthly_folders_exist(source_dir, target_dir, monthly_folders):
    """
    Ensures that all monthly folders exist within the target directory.
    Creates them if they do not exist.
    """

    for folder in monthly_folders:
        source_folder_path = os.path.join(source_dir, folder)
        target_folder_path = os.path.join(target_dir, folder)

        # Debugging: Print the folder paths being checked
        print(f"Checking target folder: {target_folder_path}")

        # Check if source monthly folder exists
        if not os.path.exists(source_folder_path):
            log_message(f"ERROR: Soruce folder does not exist: {source_folder_path}")
            continue

        # Ensure target monthly folder exists
        ensure_directory_exists(target_folder_path)


def copy_ticker_csvs(source_dir, target_dir, monthly_folders, tickers):
    """
    Copies specified tickers' CSV files from source to target monthly folders.
    Logs any missing CSVs.
    """

    missing_files = defaultdict(list)
    copied_files = defaultdict(list)

    for folder in monthly_folders:
        source_folder_path = os.path.join(source_dir, folder)
        target_folder_path = os.path.join(target_dir, folder)

        for ticker in tickers:
            source_csv = os.path.join(source_folder_path, f"{ticker}.csv")
            target_csv = os.path.join(target_folder_path, f"{ticker}.csv")

            if os.path.exists(source_csv):
                try:
                    shutil.copy(source_csv, target_csv)
                    copied_files[ticker].append(folder)
                    log_message(f"Copied {ticker}.csv to {folder}")
                except Exception as e:
                    log_message(f"ERROR: Failed to copy {ticker}.csv to {folder}. Exception: {e}")
            else:
                missing_files[ticker].append(folder)
                log_message(f"WARNING: {ticker}.csv not found in {folder}")

    # Summary
    log_message("\n=== Summary of CSV Organization ===")

    if copied_files:
        log_message("\nTickers successfully copied to the following folders:")
        for ticker, folders in copied_files.items():
            folders_str = ', '.join(folders)
            log_message(f"{ticker}: {folders_str}")

    else:
        log_message("nNo CSVs were copied.")

    if missing_files:
        log_message("\nTickers missing in the following folders:")
        for ticker, folders in missing_files.items():
            folders_str = ', '.join(folders)
            log_message(f"{ticker}: {folders_str}")

    else:
        log_message("\nAll specified tickers have their CSVs in every monthly folder.")


def main():
    # Step 0: Ensure the target base directory exists
    ensure_directory_exists(TARGET_DIR)

    # Step 1: Ensure all monthly folders exist in the target directory
    log_message("=== Step 1: Ensuring Target Monthly Folders Exist ===")
    ensure_monthly_folders_exist(BASE_2021_DIR, TARGET_DIR, monthly_folders)

    # Step 2: Copy specified tickers' CSVs
    log_message("\n=== Step 2: Copying Specified Tickers' CSVs ===")
    copy_ticker_csvs(BASE_2021_DIR, TARGET_DIR, monthly_folders, tickers)
    log_message("\n=== CSV Organization Completed ===")


if __name__ == "__main__":
    main()
