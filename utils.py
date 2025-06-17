"""
utils.py

Utility functions for the EEG training system.
- Configuration loading
- Data windowing
- Calibration data collection and model fine-tuning
"""

import json
import logging
import time
from typing import Tuple

import joblib
import numpy as np
from keras.models import load_model
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
    return np.array(X_windows), np.array(y_windows)


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
            f"Get ready for calibration: {label}. Press Enter to start recording {seconds_per_class} seconds..."
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
