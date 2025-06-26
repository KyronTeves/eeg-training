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
from scipy.signal import welch
from scipy.stats import mode
from sklearn.preprocessing import StandardScaler
from EEGModels import ShallowConvNet


def extract_features(window: np.ndarray, fs: int = 250) -> np.ndarray:
    """
    Extract features from a single EEG window.

    Args:
        window (np.ndarray): EEG window of shape (window_size, n_channels).
        fs (int): Sampling frequency.

    Returns:
        np.ndarray: 1D array of features.
    """
    features = []
    n_channels = window.shape[1]

    # Define frequency bands
    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 100),
    }

    for i in range(n_channels):
        channel_data = window[:, i]

        # --- Spectral Features (Band Power) ---
        # Use nperseg that is appropriate for the window size
        nperseg = min(256, len(channel_data))
        freqs, psd = welch(channel_data, fs=fs, nperseg=nperseg)

        total_power = np.sum(psd)
        if total_power == 0:
            # Avoid division by zero if signal is flat
            band_powers = [0.0] * len(bands)
        else:
            band_powers = [
                np.sum(psd[(freqs >= fmin) & (freqs < fmax)]) / total_power
                for fmin, fmax in bands.values()
            ]

        features.extend(band_powers)

        # --- Statistical Features ---
        features.append(np.mean(channel_data))
        features.append(np.var(channel_data))
        features.append(np.std(channel_data))

    return np.array(features)


def load_config(path: str = "config.json") -> dict:
    """Load configuration from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def window_data(
    data: np.ndarray, labels: np.ndarray, window_size: int, step_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized segmentation of data and labels into overlapping windows.
    Uses numpy stride tricks for efficient window creation.
    """
    n_windows = (len(data) - window_size) // step_size + 1
    if n_windows <= 0:
        return np.empty((0, window_size, data.shape[1])), np.empty((0,))

    # Create windows using stride tricks
    shape = (n_windows, window_size, data.shape[1])
    strides = (data.strides[0] * step_size, data.strides[0], data.strides[1])
    x_windows = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

    # For labels, use the mode (most frequent) in each window
    label_windows = np.lib.stride_tricks.as_strided(
        labels,
        shape=(n_windows, window_size, 1),
        strides=(labels.strides[0] * step_size, labels.strides[0], labels.strides[1]),
    )
    # Compute mode along window axis
    y_windows = mode(label_windows, axis=1, keepdims=False)[0].reshape(-1)

    # Data quality assessment
    logging.info(
        "Data quality: Range [%.3f, %.3f], Std: %.3f, NaN count: %d",
        np.min(x_windows),
        np.max(x_windows),
        np.std(x_windows),
        np.isnan(x_windows).sum(),
    )
    return x_windows, y_windows


def collect_calibration_data(
    board, channels, window_size, labels, seconds_per_class=10, sample_rate=250
):
    """
    Collect labeled calibration data for each class from the user.
    Returns x_calib (windows, window, channels), y_calib (windows,)
    """
    calib_x = []
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
        calib_x.extend(data)
        calib_y.extend([label] * len(data))
        logging.info("Collected %d windows for label '%s'.", len(data), label)
    return np.array(calib_x), np.array(calib_y)


def run_session_calibration(
    x_calib,
    y_calib,
    base_model_path,
    label_encoder_path,
    out_model_path,
    out_scaler_path,
    epochs=3,
    batch_size=16,
    model_type="EEGNet",
    n_channels=None,
    window_size=None,
    dropout_rate=None,
):
    """
    Windows, preprocesses, and fine-tunes the model/scaler for the session.
    Supports both EEGNet and ShallowConvNet.
    """
    le = joblib.load(label_encoder_path)
    y_calib_encoded = le.transform(y_calib)
    y_calib_cat = to_categorical(y_calib_encoded)
    scaler = StandardScaler()
    x_calib_flat = x_calib.reshape(-1, x_calib.shape[-1])
    scaler.fit(x_calib_flat)
    x_calib_scaled = scaler.transform(x_calib_flat).reshape(x_calib.shape)
    # Prepare for model: (batch, window, channels) -> (batch, channels, window, 1)
    x_calib_model = np.expand_dims(x_calib_scaled, -1)
    x_calib_model = np.transpose(x_calib_model, (0, 2, 1, 3))
    if model_type == "EEGNet":
        model = load_model(base_model_path)
    elif model_type == "ShallowConvNet":
        # Rebuild model from scratch to avoid optimizer issues
        if n_channels is None or window_size is None or dropout_rate is None:
            raise ValueError("n_channels, window_size, dropout_rate must be provided for ShallowConvNet calibration.")
        model = ShallowConvNet(
            nb_classes=y_calib_cat.shape[1],
            Chans=n_channels,
            Samples=window_size,
            dropoutRate=dropout_rate,
        )
        model.load_weights(base_model_path)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    # Recompile model with new optimizer to avoid Keras variable mismatch error
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        x_calib_model, y_calib_cat, epochs=epochs, batch_size=batch_size, verbose=1
    )
    model.save(out_model_path)
    joblib.dump(scaler, out_scaler_path)
    logging.info(
        f"Session calibration complete for {model_type}. Model saved to %s, scaler saved to %s.",
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


def check_no_nan(x, name="data"):
    """
    Check for NaN values in a numpy array and log/raise error if found.
    Args:
        x: numpy array to check.
        name: Name of the data (for error messages).
    Raises:
        ValueError: If any NaN values are found in x.
    """
    if np.isnan(x).any():
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
