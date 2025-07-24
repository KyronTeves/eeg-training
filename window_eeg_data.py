"""window_eeg_data.py.

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

from utils import (
    check_labels_valid,
    check_no_nan,
    handle_errors,  # Import error handler
    load_config,
    setup_logging,
    window_data,
)

setup_logging()  # Set up consistent logging to file and console
logger = logging.getLogger(__name__)


def load_and_filter_data(config: dict[str, Any]) -> pd.DataFrame:
    """Load EEG data from CSV and filters by session types if specified in config.

    Args:
        config (dict[str, Any]): Configuration parameters.

    Returns:
        pd.DataFrame: Filtered DataFrame.

    """
    csv_file = config["OUTPUT_CSV"]

    logger.info("Loading EEG data from: %s", csv_file)

    try:
        raw_data = pd.read_csv(csv_file)
        if raw_data.empty:
            logger.warning("Data file is empty. No data to process.")
            return pd.DataFrame()
    except (FileNotFoundError, pd.errors.EmptyDataError):
        logger.exception("Failed to load or parse data CSV.")
        logger.info("Make sure to run collect_data.py first to generate training data.")
        raise

    if "session_type" in raw_data.columns:
        logger.info("Available session types: %s", raw_data["session_type"].unique())
        use_sessions = config["USE_SESSION_TYPES"]
        filtered_data = raw_data[raw_data["session_type"].isin(use_sessions)]
        logger.info(
            "Using session types: %s, samples: %d",
            use_sessions,
            len(filtered_data),
        )
        return filtered_data

    logger.warning("'session_type' column not found. Using all data.")
    return raw_data


def process_and_window_data(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract EEG and labels from DataFrame, then creates overlapping windows.

    Args:
        df (pd.DataFrame): Input DataFrame.
        config (dict[str, Any]): Configuration dictionary.

    Raises:
        ValueError: If the data is not valid.

    Returns:
        tuple[np.ndarray, np.ndarray]: (x_windows, y_windows)

    """
    eeg_cols = [col for col in df.columns if col.startswith("ch_")]
    x = df[eeg_cols].to_numpy()
    labels = df["label"].to_numpy()

    check_no_nan(x, name="EEG data")
    check_labels_valid(labels, valid_labels=config["LABELS"], name="Labels")

    n_channels = config["N_CHANNELS"]
    if x.shape[1] != n_channels:
        msg = f"Expected {n_channels} channels, but found {x.shape[1]} in the data."
        raise ValueError(msg)

    x = x.reshape(-1, n_channels)
    labels = labels.reshape(-1, 1)

    return window_data(x, labels, config["WINDOW_SIZE"], config["STEP_SIZE"])


def save_windowed_data(
    x_windows: np.ndarray,
    y_windows: np.ndarray,
    config: dict[str, Any],
) -> None:
    """Save windowed EEG data and labels to .npy files.

    Args:
        x_windows (np.ndarray): Windowed EEG data.
        y_windows (np.ndarray): Windowed labels.
        config (dict[str, Any]): Configuration dictionary.

    """
    try:
        np.save(config["WINDOWED_NPY"], x_windows)
        np.save(config["WINDOWED_LABELS_NPY"], y_windows)
        logger.info(
            "Saved windowed data to %s and %s",
            config["WINDOWED_NPY"],
            config["WINDOWED_LABELS_NPY"],
        )
    except (OSError, ValueError):
        logger.exception("Failed to save windowed data.")
        raise


@handle_errors
def main() -> None:
    """Window EEG data for model training.

    Load config and data, window the data, and save output files.
    """
    config = load_config()
    filtered_df = load_and_filter_data(config)

    if filtered_df.empty:
        logger.info("No data to process. Exiting.")
        return

    x_windows, y_windows = process_and_window_data(filtered_df, config)

    logger.info(
        "Created %d windows of shape %s",
        x_windows.shape[0],
        x_windows.shape[1:],
    )

    save_windowed_data(x_windows, y_windows, config)


if __name__ == "__main__":
    main()
