"""
window_eeg_data.py

Segment raw EEG CSV data into overlapping windows for model training.

Loads, filters, and windows EEG data for supervised learning, saving windowed EEG and label arrays
for downstream model training.

Typical usage:
    $ python window_eeg_data.py
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from utils import handle_errors  # Import error handler
from utils import (check_labels_valid, check_no_nan, load_config,
                   setup_logging, window_data)

setup_logging()  # Set up consistent logging to file and console


def load_and_filter_data(config: dict[str, Any]) -> pd.DataFrame:
    """
    Loads EEG data from CSV and filters by session types if specified in config.

    Args:
        config (dict): Configuration parameters.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    csv_file = config["OUTPUT_CSV"]

    logging.info("Loading EEG data from: %s", csv_file)

    try:
        raw_data = pd.read_csv(csv_file)
        if raw_data.empty:
            logging.warning("Data file is empty. No data to process.")
            return pd.DataFrame()
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        logging.error("Failed to load or parse data CSV: %s", e)
        logging.info(
            "Make sure to run collect_data.py first to generate training data."
        )
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


def process_and_window_data(
    df: pd.DataFrame, config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts EEG and labels from DataFrame, then creates overlapping windows.

    Args:
        df (pd.DataFrame): Input DataFrame.
        config (dict): Configuration dictionary.

    Returns:
        tuple[np.ndarray, np.ndarray]: (x_windows, y_windows)
    """
    eeg_cols = [col for col in df.columns if col.startswith("ch_")]
    x = df[eeg_cols].values
    labels = df["label"].values

    check_no_nan(x, name="EEG data")
    check_labels_valid(labels, valid_labels=config["LABELS"], name="Labels")

    n_channels = config["N_CHANNELS"]
    if x.shape[1] != n_channels:
        raise ValueError(
            f"Expected {n_channels} channels, but found {x.shape[1]} in the data."
        )

    x = x.reshape(-1, n_channels)
    labels = labels.reshape(-1, 1)

    return window_data(x, labels, config["WINDOW_SIZE"], config["STEP_SIZE"])


def save_windowed_data(
    x_windows: np.ndarray, y_windows: np.ndarray, config: dict[str, Any]
) -> None:
    """
    Saves windowed EEG data and labels to .npy files.

    Args:
        x_windows (np.ndarray): Windowed EEG data.
        y_windows (np.ndarray): Windowed labels.
        config (dict): Configuration dictionary.
    """
    try:
        np.save(config["WINDOWED_NPY"], x_windows)
        np.save(config["WINDOWED_LABELS_NPY"], y_windows)
        logging.info(
            "Saved windowed data to %s and %s",
            config["WINDOWED_NPY"],
            config["WINDOWED_LABELS_NPY"],
        )
    except (OSError, ValueError) as e:
        logging.error("Failed to save windowed data: %s", e)
        raise


@handle_errors
def main() -> None:
    """
    Main entry point for windowing EEG data for model training.

    Loads config and data, windows the data, and saves output files.
    """
    config = load_config()
    filtered_df = load_and_filter_data(config)

    if filtered_df.empty:
        logging.info("No data to process. Exiting.")
        return

    x_windows, y_windows = process_and_window_data(filtered_df, config)

    logging.info(
        "Created %d windows of shape %s",
        x_windows.shape[0],
        x_windows.shape[1:],
    )

    save_windowed_data(x_windows, y_windows, config)


if __name__ == "__main__":
    main()
