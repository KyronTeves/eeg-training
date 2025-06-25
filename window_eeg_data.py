"""
Segment raw EEG CSV data into overlapping windows for model training.

- Loads labeled EEG data
- Filters by session type
- Extracts EEG channels
- Segments into overlapping windows for supervised learning
- Saves windowed data and labels as .npy files for model training

Input: Labeled EEG CSV file
Output: Windowed EEG data (.npy), windowed labels (.npy)
"""

import logging

import numpy as np
import pandas as pd

from utils import (
    load_config,
    window_data,
    setup_logging,
    check_no_nan,
    check_labels_valid,
)

setup_logging()  # Set up consistent logging to file and console


def load_and_filter_data(config: dict) -> pd.DataFrame:
    """
    Load raw EEG data and filter it based on session types specified in the config.

    Args:
        config: Dictionary with configuration parameters.

    Returns:
        Filtered pandas DataFrame.
    """
    try:
        raw_data = pd.read_csv(config["OUTPUT_CSV"])
        if raw_data.empty:
            logging.warning("Raw data file is empty. No data to process.")
            return pd.DataFrame()
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        logging.error("Failed to load or parse raw data CSV: %s", e)
        raise

    if "session_type" in raw_data.columns:
        logging.info("Available session types: %s", raw_data["session_type"].unique())
        use_sessions = config["USE_SESSION_TYPES"]
        filtered_data = raw_data[raw_data["session_type"].isin(use_sessions)]
        logging.info(
            "Using session types: %s, samples: %d",
            use_sessions,
            len(filtered_data),
        )
        return filtered_data

    logging.warning("'session_type' column not found. Using all data.")
    return raw_data


def process_and_window_data(df: pd.DataFrame, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Process the DataFrame to extract EEG data and labels, then create overlapping windows.

    Args:
        df: DataFrame containing the EEG data.
        config: Dictionary with configuration parameters.

    Returns:
        A tuple containing windowed data (X_windows) and labels (y_windows).
    """
    eeg_cols = [col for col in df.columns if col.startswith("ch_")]
    X = df[eeg_cols].values
    labels = df["label"].values

    check_no_nan(X, name="EEG data")
    check_labels_valid(labels, valid_labels=config["LABELS"], name="Labels")

    n_channels = config["N_CHANNELS"]
    if X.shape[1] != n_channels:
        raise ValueError(
            f"Expected {n_channels} channels, but found {X.shape[1]} in the data."
        )

    X = X.reshape(-1, n_channels)
    labels = labels.reshape(-1, 1)

    return window_data(X, labels, config["WINDOW_SIZE"], config["STEP_SIZE"])


def save_windowed_data(
    X_windows: np.ndarray, y_windows: np.ndarray, config: dict
) -> None:
    """
    Save the windowed data and labels to .npy files.

    Args:
        X_windows: The windowed feature data.
        y_windows: The windowed labels.
        config: Dictionary with configuration parameters.
    """
    try:
        np.save(config["WINDOWED_NPY"], X_windows)
        np.save(config["WINDOWED_LABELS_NPY"], y_windows)
        logging.info(
            "Saved windowed data to %s and %s",
            config["WINDOWED_NPY"],
            config["WINDOWED_LABELS_NPY"],
        )
    except (OSError, ValueError) as e:
        logging.error("Failed to save windowed data: %s", e)
        raise


def main():
    """
    Main function to orchestrate the windowing process.
    """
    config = load_config()
    filtered_df = load_and_filter_data(config)

    if filtered_df.empty:
        logging.info("No data to process. Exiting.")
        return

    X_windows, y_windows = process_and_window_data(filtered_df, config)

    logging.info(
        "Created %d windows of shape %s",
        X_windows.shape[0],
        X_windows.shape[1:],
    )

    save_windowed_data(X_windows, y_windows, config)


if __name__ == "__main__":
    main()
