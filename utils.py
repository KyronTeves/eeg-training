"""
utils.py

Utility functions for the EEG training system.
- Configuration loading
- Data windowing
- Calibration data collection and model fine-tuning
"""

import json
import logging
import os
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler


def load_config(path: str = "config.json") -> dict:
    """Load configuration from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def window_data(
    data: np.ndarray, labels: np.ndarray, window_size: int, step_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment data and labels into overlapping windows.

    Args:
        data: EEG data array of shape (samples, channels)
        labels: Array of labels (samples, 1)
        window_size: Number of samples per window
        step_size: Step size between windows
    Returns:
        Tuple of (X_windows, y_windows)
    """
    X_windows = []
    y_windows = []
    for start in range(0, len(data) - window_size + 1, step_size):
        window = data[start : start + window_size]
        window_labels = labels[start : start + window_size]
        # Use the most frequent label in the window as the label
        unique, counts = np.unique(window_labels, return_counts=True)
        window_label = unique[np.argmax(counts)]
        X_windows.append(window)
        y_windows.append(window_label)

    X_windows = np.array(X_windows)
    y_windows = np.array(y_windows)

    # Data quality assessment
    logging.info(
        "Data quality: Range [%.3f, %.3f], Std: %.3f, NaN count: %d",
        np.min(X_windows),
        np.max(X_windows),
        np.std(X_windows),
        np.isnan(X_windows).sum(),
    )

    return X_windows, y_windows


def collect_calibration_data(
    board, channels, window_size, labels, seconds_per_class=10, sample_rate=250
):
    """
    Collect labeled calibration data for each class from the user.
    Returns X_calib (windows, window, channels), y_calib (windows,)
    """
    calib_X = []
    calib_y = []
    for label in labels:
        input(
            f"Calibrating '{label}': Press Enter to record for {seconds_per_class} seconds..."
        )
        data = []
        start_time = time.time()
        while time.time() - start_time < seconds_per_class:
            eeg = board.get_current_board_data(window_size)
            if eeg.shape[1] >= window_size:
                eeg_window = eeg[channels, -window_size:].T
                data.append(eeg_window)
            time.sleep(window_size / sample_rate)
        calib_X.extend(data)
        calib_y.extend([label] * len(data))
        logging.info("Collected %d windows for label '%s'.", len(data), label)
    return np.array(calib_X), np.array(calib_y)


def run_session_calibration(
    X_calib,
    y_calib,
    base_model_path,
    base_scaler_path,
    label_encoder_path,
    out_model_path,
    out_scaler_path,
    epochs=3,
    batch_size=16,
):
    """
    Windows, preprocesses, and fine-tunes the model/scaler for the session.
    """
    _ = base_scaler_path  # Dummy assignment to suppress unused argument warning
    le = joblib.load(label_encoder_path)
    y_calib_encoded = le.transform(y_calib)
    y_calib_cat = to_categorical(y_calib_encoded)
    scaler = StandardScaler()
    X_calib_flat = X_calib.reshape(-1, X_calib.shape[-1])
    scaler.fit(X_calib_flat)
    X_calib_scaled = scaler.transform(X_calib_flat).reshape(X_calib.shape)
    # Prepare for EEGNet: (batch, window, channels) -> (batch, channels, window, 1)
    X_calib_eegnet = np.expand_dims(X_calib_scaled, -1)
    X_calib_eegnet = np.transpose(X_calib_eegnet, (0, 2, 1, 3))
    model = load_model(base_model_path)
    # Recompile model with new optimizer to avoid Keras variable mismatch error
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        X_calib_eegnet, y_calib_cat, epochs=epochs, batch_size=batch_size, verbose=1
    )
    model.save(out_model_path)
    joblib.dump(scaler, out_scaler_path)
    logging.info(
        "Session calibration complete. Model saved to %s, scaler saved to %s.",
        out_model_path,
        out_scaler_path,
    )


def setup_logging(logfile: str = "eeg_training.log"):
    """
    Set up logging to both console and file with automatic rotation.
    Should be called at the start of each script to ensure consistent logging.
    Args:
        logfile: Path to the log file (default: 'eeg_training.log').
    """

    # Create rotating file handler (10MB max, keep 5 backup files)
    file_handler = RotatingFileHandler(
        logfile,
        maxBytes=10 * 1024 * 1024,  # 10MB per file
        backupCount=5,  # Keep 5 old files (eeg_training.log.1, .2, etc.)
        encoding="utf-8",
    )

    # Console handler for immediate feedback
    console_handler = logging.StreamHandler()

    # Set up formatting
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO, handlers=[console_handler, file_handler]
    )


def cleanup_old_logs(logfile: str = "eeg_training.log", max_size_mb: int = 50):
    """
    Manual cleanup function for oversized log files.
    Call this if log rotation wasn't previously enabled.
    """

    if not os.path.exists(logfile):
        return

    file_size_mb = os.path.getsize(logfile) / (1024 * 1024)

    if file_size_mb > max_size_mb:
        # Archive the current log
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"{logfile}.archive_{timestamp}"

        os.rename(logfile, archive_name)

        logging.info("Large log file archived to %s (%.1fMB)", archive_name, file_size_mb)
        logging.info("Starting fresh log file with rotation enabled")


def check_no_nan(X, name="data"):
    """
    Check for NaN values in a numpy array and log/raise error if found.
    Args:
        X: numpy array to check.
        name: Name of the data (for error messages).
    Raises:
        ValueError: If any NaN values are found in X.
    """
    if np.isnan(X).any():
        logging.error("%s contains NaN values.", name)
        raise ValueError(f"{name} contains NaN values.")


def check_labels_valid(labels, valid_labels=None, name="labels"):
    """
    Check for NaN values and, if valid_labels is provided, for invalid label values.
    Args:
        labels: Array of labels to check.
        valid_labels: Optional set/list of valid label values.
        name: Name of the label array (for error messages).
    Raises:
        ValueError: If any NaN or invalid label values are found.
    """
    if pd.isnull(labels).any():
        logging.error("%s contain NaN values.", name)
        raise ValueError(f"{name} contain NaN values.")
    if valid_labels is not None:
        invalid = set(labels) - set(valid_labels)
        if invalid:
            logging.error("%s contain invalid values: %s", name, invalid)
            raise ValueError(f"{name} contain invalid values: {invalid}")
