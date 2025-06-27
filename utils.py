"""
utils.py

Utility functions for the EEG training system.

Input: Various (EEG data, config files, etc.)
Process: Configuration loading, data windowing, calibration data collection, logging setup, validation utilities.
Output: Processed data, configuration dicts, logging setup, validation results.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from EEGModels import ShallowConvNet


def extract_features(window: np.ndarray, fs: int = 250) -> np.ndarray:
    """
    Extract features from a single EEG window for tree-based models.

    Input: window (np.ndarray) - shape (window_size, n_channels), fs (int) - sampling frequency
    Process: Computes band powers and statistical features for each channel
    Output: 1D np.ndarray of features (length: n_channels * 8)
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


def load_config(path: str = None) -> dict:
    """
    Load configuration from a JSON file. If path is None, use the CONFIG_PATH environment variable or default to 'config.json'.
    Input: path (str) - path to config file
    Process: Reads and parses JSON
    Output: Config dictionary
    """
    if path is None:
        path = os.environ.get("CONFIG_PATH", "config.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def window_data(
    data: np.ndarray, labels: np.ndarray, window_size: int, step_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment data and labels into overlapping windows using stride tricks.

    Input: data (np.ndarray), labels (np.ndarray), window_size (int), step_size (int)
    Process: Creates overlapping windows, computes majority label per window
    Output: (x_windows, y_windows)
    """
    n_windows = (len(data) - window_size) // step_size + 1
    if n_windows <= 0:
        return np.empty((0, window_size, data.shape[1])), np.empty((0,))

    # Create windows using stride tricks
    shape = (n_windows, window_size, data.shape[1])
    strides = (data.strides[0] * step_size, data.strides[0], data.strides[1])
    x_windows = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

    # For labels, use the majority label in each window
    label_windows = np.lib.stride_tricks.as_strided(
        labels,
        shape=(n_windows, window_size, 1),
        strides=(labels.strides[0] * step_size, labels.strides[0], labels.strides[1]),
    )
    # Compute majority label using np.unique
    y_windows = np.array(
        [
            np.unique(window, return_counts=True)[0][
                np.argmax(np.unique(window, return_counts=True)[1])
            ]
            for window in label_windows.reshape(n_windows, -1)
        ]
    )

    # Data quality assessment
    logging.info(
        "Data quality: Range [%.3f, %.3f], Std: %.3f, NaN count: %d",
        np.min(x_windows),
        np.max(x_windows),
        np.std(x_windows),
        np.isnan(x_windows).sum(),
    )
    return x_windows, y_windows


def setup_logging(logfile: str = "eeg_training.log"):
    """
    Set up logging to both console and file with rotation.

    Input: logfile (str)
    Process: Configures logging handlers and formatters
    Output: None (side effect: logging configured)
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
    logging.basicConfig(level=logging.INFO, handlers=[console_handler, file_handler])


def cleanup_old_logs(logfile: str = "eeg_training.log", max_size_mb: int = 50):
    """
    Archive and rotate log file if it exceeds max_size_mb.

    Input: logfile (str), max_size_mb (int)
    Process: Checks file size, renames if too large
    Output: None (side effect: log file archived)
    """

    if not os.path.exists(logfile):
        return

    file_size_mb = os.path.getsize(logfile) / (1024 * 1024)

    if file_size_mb > max_size_mb:
        # Archive the current log
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"{logfile}.archive_{timestamp}"

        os.rename(logfile, archive_name)

        logging.info(
            "Large log file archived to %s (%.1fMB)", archive_name, file_size_mb
        )
        logging.info("Starting fresh log file with rotation enabled")


def check_no_nan(x, name="data"):
    """
    Check for NaN values in a numpy array and raise error if found.

    Input: x (np.ndarray), name (str)
    Process: Checks for NaN, logs and raises if found
    Output: None (raises ValueError if NaN found)
    """
    if np.isnan(x).any():
        logging.error("%s contains NaN values.", name)
        raise ValueError(f"{name} contains NaN values.")


def check_labels_valid(labels, valid_labels=None, name="labels"):
    """
    Check for NaN and invalid label values in an array.

    Input: labels (array-like), valid_labels (list/None), name (str)
    Process: Checks for NaN and invalid values, logs and raises if found
    Output: None (raises ValueError if invalid)
    """
    if pd.isnull(labels).any():
        logging.error("%s contain NaN values.", name)
        raise ValueError(f"{name} contain NaN values.")
    if valid_labels is not None:
        invalid = set(labels) - set(valid_labels)
        if invalid:
            logging.error("%s contain invalid values: %s", name, invalid)
            raise ValueError(f"{name} contain invalid values: {invalid}")


def collect_lsl_calib_data(lsl_stream_handler, config, seconds_per_class):
    """
    Collect calibration data from an LSL stream for each label class.

    Input: lsl_stream_handler (object), config (dict), seconds_per_class (int)
    Process: Prompts user for each label, collects EEG windows for a specified duration per class
    Output: Tuple of numpy arrays (calib_x, calib_y) containing calibration data and corresponding labels
    """
    label_classes = config["LABELS"]
    window_size = config["WINDOW_SIZE"]
    sample_rate = config["SAMPLING_RATE"]
    calib_x, calib_y = [], []
    for label in label_classes:
        input(
            f"Calibrating '{label}': Press Enter to record for {seconds_per_class} seconds..."
        )
        data = []
        start_time = time.time()
        while time.time() - start_time < seconds_per_class:
            eeg = lsl_stream_handler.get_window(window_size, timeout=2.0)
            if eeg is not None and eeg.shape[0] == window_size:
                eeg_window = eeg
                data.append(eeg_window)
            time.sleep(window_size / sample_rate)
        calib_x.extend(data)
        calib_y.extend([label] * len(data))
        logging.info("Collected %d windows for label '%s'.", len(data), label)
    return np.array(calib_x), np.array(calib_y)


def calibrate_deep_models(
    config, x_model, y_cat, scaler, session_tag, save_dir, verbose
):
    """
    Calibrate and save deep learning models (EEGNet and ShallowConvNet) for EEG classification.

    Input:
        config (dict): Configuration dictionary.
        x_model (np.ndarray): Preprocessed EEG data for model input.
        y_cat (np.ndarray): One-hot encoded labels.
        scaler (StandardScaler): Fitted scaler for data normalization.
        session_tag (str): Unique session identifier.
        save_dir (str): Directory to save models and scalers.
        verbose (bool): Verbosity flag for model training.

    Process:
        Loads pretrained models, fine-tunes them on calibration data, and saves updated models and scalers.

    Output:
        None (side effect: models and scalers are saved to disk)
    """
    label_classes = config["LABELS"]
    channels = (
        config.get("EEG_CHANNELS")
        or config.get("CHANNELS")
        or list(range(config["N_CHANNELS"]))
    )
    window_size = config["WINDOW_SIZE"]
    # EEGNet
    eegnet_path = config["MODEL_EEGNET"]
    eegnet_out = config.get("MODEL_EEGNET_SESSION") or os.path.join(
        save_dir, f"eeg_direction_model_session_{session_tag}.h5"
    )
    scaler_out = config.get("SCALER_EEGNET_SESSION") or os.path.join(
        save_dir, f"eeg_scaler_session_{session_tag}.pkl"
    )
    model = load_model(eegnet_path)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        x_model,
        y_cat,
        epochs=config.get("CALIB_EPOCHS", 3),
        batch_size=config.get("CALIB_BATCH_SIZE", 16),
        verbose=verbose,
    )
    model.save(eegnet_out)
    joblib.dump(scaler, scaler_out)
    logging.info("EEGNet session model saved to %s", eegnet_out)
    # ShallowConvNet
    shallow_path = config["MODEL_SHALLOW"]
    shallow_out = os.path.join(save_dir, f"eeg_shallow_model_session_{session_tag}.h5")
    scaler_shallow_out = config.get("SCALER_TREE_SESSION") or os.path.join(
        save_dir, f"eeg_scaler_tree_session_{session_tag}.pkl"
    )
    shallow = ShallowConvNet(
        nb_classes=len(label_classes),
        Chans=len(channels),
        Samples=window_size,
    )
    shallow.load_weights(shallow_path)
    shallow.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    shallow.fit(
        x_model,
        y_cat,
        epochs=config.get("CALIB_EPOCHS", 3),
        batch_size=config.get("CALIB_BATCH_SIZE", 16),
        verbose=verbose,
    )
    shallow.save(shallow_out)
    joblib.dump(scaler, scaler_shallow_out)
    logging.info("ShallowConvNet session model saved to %s", shallow_out)


def calibrate_tree_models(config, x_calib, y_calib_encoded, session_tag, save_dir):
    """
    Calibrate and save tree-based models (Random Forest, XGBoost) for EEG classification.

    Input:
        config (dict): Configuration dictionary.
        x_calib (np.ndarray): Calibration EEG data windows.
        y_calib_encoded (np.ndarray): Encoded calibration labels.
        session_tag (str): Unique session identifier.
        save_dir (str): Directory to save models and scalers.

    Process:
        Extracts features, scales data, trains models, and saves them.

    Output:
        None (side effect: models and scalers are saved to disk)
    """
    sample_rate = config["SAMPLING_RATE"]
    x_feat = np.array([extract_features(w, fs=sample_rate) for w in x_calib])
    scaler_tree = StandardScaler()
    x_feat_scaled = scaler_tree.fit_transform(x_feat)
    scaler_tree_out = config.get("SCALER_TREE_SESSION") or os.path.join(
        save_dir, f"eeg_scaler_tree_session_{session_tag}.pkl"
    )
    joblib.dump(scaler_tree, scaler_tree_out)
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(x_feat_scaled, y_calib_encoded)
    rf_out = config.get("MODEL_RF_SESSION") or os.path.join(
        save_dir, f"eeg_rf_model_session_{session_tag}.pkl"
    )
    joblib.dump(rf, rf_out)
    logging.info("Random Forest session model saved to %s", rf_out)
    # XGBoost
    xgb = XGBClassifier(
        n_estimators=100, use_label_encoder=False, eval_metric="mlogloss"
    )
    xgb.fit(x_feat_scaled, y_calib_encoded)
    xgb_out = config.get("MODEL_XGB_SESSION") or os.path.join(
        save_dir, f"eeg_xgb_model_session_{session_tag}.pkl"
    )
    joblib.dump(xgb, xgb_out)
    logging.info("XGBoost session model saved to %s", xgb_out)


def calibrate_all_models_lsl(
    lsl_stream_handler,
    config_path="config.json",
    seconds_per_class=None,
    session_tag=None,
    save_dir="models",
    verbose=True,
):
    """
    Unified, LSL-aware calibration for all models (EEGNet, ShallowConvNet, RF, XGBoost).
    Uses config for parameters and saves session-specific models/scalers.
    """

    # --- Load config ---
    config = load_config(config_path)
    config = {k.upper(): v for k, v in config.items()}
    if seconds_per_class is None:
        seconds_per_class = config.get("CALIBRATION_SECONDS_PER_CLASS", 10)
    if session_tag is None:
        session_tag = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(save_dir, exist_ok=True)

    # --- Collect calibration data using LSL ---
    x_calib, y_calib = collect_lsl_calib_data(
        lsl_stream_handler, config, seconds_per_class
    )
    check_no_nan(x_calib, name="calibration data")
    check_labels_valid(y_calib, valid_labels=config["LABELS"])

    # --- Encode labels ---
    le = LabelEncoder()
    le.fit(config["LABELS"])
    y_calib_encoded = le.transform(y_calib)
    le_path = config.get("LABEL_ENCODER_SESSION") or os.path.join(
        save_dir, f"eeg_label_encoder_session_{session_tag}.pkl"
    )
    label_classes_path = config.get("LABEL_CLASSES_NPY_SESSION") or os.path.join(
        save_dir, f"eeg_label_classes_session_{session_tag}.npy"
    )
    joblib.dump(le, le_path)
    np.save(label_classes_path, config["LABELS"])

    # --- Prepare data for deep models ---
    scaler = StandardScaler()
    x_calib_flat = x_calib.reshape(-1, x_calib.shape[-1])
    scaler.fit(x_calib_flat)
    x_calib_scaled = scaler.transform(x_calib_flat).reshape(x_calib.shape)
    x_model = np.expand_dims(x_calib_scaled, -1)
    x_model = np.transpose(x_model, (0, 2, 1, 3))
    y_cat = to_categorical(y_calib_encoded)

    calibrate_deep_models(
        config, x_model, y_cat, scaler, session_tag, save_dir, verbose
    )
    calibrate_tree_models(config, x_calib, y_calib_encoded, session_tag, save_dir)

    print("Session calibration complete. All models saved.")
