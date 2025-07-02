"""
utils.py

Utility functions for the EEG training system.

Provides configuration loading, data windowing, calibration data collection, logging setup, and
validation utilities for EEG data processing and model training.

Typical usage:
    from utils import load_config, window_data, setup_logging
"""

import json
import logging
import os
import time
import functools
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any, Callable, Dict, Optional, Tuple, List

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
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
    Extracts features from a single EEG window for tree-based models.

    Args:
        window (np.ndarray): EEG window, shape (window_size, n_channels).
        fs (int): Sampling frequency.

    Returns:
        np.ndarray: 1D array of features (length: n_channels * 8).
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


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads configuration from a JSON file.

    Args:
        path (str, optional): Path to config file. If None, uses CONFIG_PATH env var or 'config.json'.

    Returns:
        dict: Config dictionary.
    """
    if path is None:
        path = os.environ.get("CONFIG_PATH", "config.json")
    return load_json(path)


def window_data(
    data: np.ndarray, labels: np.ndarray, window_size: int, step_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segments data and labels into overlapping windows using stride tricks.

    Args:
        data (np.ndarray): EEG data.
        labels (np.ndarray): Labels.
        window_size (int): Window size.
        step_size (int): Step size.

    Returns:
        tuple: (x_windows, y_windows)
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


def setup_logging(
    logfile: str = "eeg_training.log", default_level: Optional[str] = None
) -> None:
    """
    Sets up logging to both console and file with rotation.

    Args:
        logfile (str): Log file name.
        default_level (str, optional): Default log level.

    Returns:
        None
    """
    # Determine log level
    level_str = os.environ.get("LOG_LEVEL", default_level or "INFO")
    level = getattr(logging, level_str.upper(), logging.INFO)

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
    logging.basicConfig(level=level, handlers=[console_handler, file_handler])


def cleanup_old_logs(logfile: str = "eeg_training.log", max_size_mb: int = 50) -> None:
    """
    Archives and rotates log file if it exceeds max_size_mb.

    Args:
        logfile (str): Log file name.
        max_size_mb (int): Max size in MB before rotating.

    Returns:
        None
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


def check_no_nan(x: np.ndarray, name: str = "data") -> None:
    """
    Checks for NaN values in a numpy array and raises error if found.

    Args:
        x (np.ndarray): Array to check.
        name (str): Name for logging.

    Raises:
        ValueError: If NaN found.
    """
    if np.isnan(x).any():
        logging.error("%s contains NaN values.", name)
        raise ValueError(f"{name} contains NaN values.")


def check_labels_valid(
    labels: np.ndarray, valid_labels: Optional[List[Any]] = None, name: str = "labels"
) -> None:
    """
    Checks for NaN and invalid label values in an array.

    Args:
        labels (array-like): Labels to check.
        valid_labels (list, optional): List of valid labels.
        name (str): Name for logging.

    Raises:
        ValueError: If invalid labels or NaN found.
    """
    if pd.isnull(labels).any():
        logging.error("%s contain NaN values.", name)
        raise ValueError(f"{name} contain NaN values.")
    if valid_labels is not None:
        invalid = set(labels) - set(valid_labels)
        if invalid:
            logging.error("%s contain invalid values: %s", name, invalid)
            raise ValueError(f"{name} contain invalid values: {invalid}")


def collect_lsl_calib_data(
    lsl_stream_handler: Any,
    label_classes: list[str],
    window_size: int,
    sample_rate: int,
    seconds_per_class: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collects calibration data from an LSL stream for each label class.

    Args:
        lsl_stream_handler (object): LSL handler.
        label_classes (list[str]): List of label classes.
        window_size (int): Window size.
        sample_rate (int): Sample rate.
        seconds_per_class (int): Seconds to record per class.

    Returns:
        tuple: (calib_x, calib_y) arrays.
    """
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
    config: dict,
    x_model: np.ndarray,
    y_cat: np.ndarray,
    scaler: StandardScaler,
    save_dir: str,
    verbose: bool,
) -> None:
    """
    Calibrates and saves deep learning models (EEGNet and ShallowConvNet) for EEG classification.

    Args:
        config (dict): Configuration dictionary.
        x_model (np.ndarray): Model input data.
        y_cat (np.ndarray): One-hot labels.
        scaler (StandardScaler): Scaler instance.
        save_dir (str): Directory to save models.
        verbose (bool): Verbosity flag.

    Returns:
        None
    """
    # EEGNet
    model = load_model(config["MODEL_EEGNET"])
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
    # Save EEGNet model and scaler to generic session paths (no timestamp)
    eegnet_out = os.path.join(save_dir, "eeg_direction_model_session.h5")
    scaler_out = os.path.join(save_dir, "eeg_scaler_session.pkl")
    model.save(eegnet_out)
    joblib.dump(scaler, scaler_out)
    logging.info("EEGNet session model saved to %s", eegnet_out)
    # ShallowConvNet
    shallow = ShallowConvNet(
        nb_classes=len(config["LABELS"]),
        Chans=len(
            config.get("EEG_CHANNELS")
            or config.get("CHANNELS")
            or list(range(config["N_CHANNELS"]))
        ),
        Samples=config["WINDOW_SIZE"],
    )
    shallow.load_weights(config["MODEL_SHALLOW"])
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
    # Save ShallowConvNet model to generic session path (no timestamp)
    shallow_out = os.path.join(save_dir, "eeg_shallow_model_session.h5")
    scaler_shallow_out = os.path.join(save_dir, "eeg_scaler_tree_session.pkl")
    shallow.save(shallow_out)
    joblib.dump(scaler, scaler_shallow_out)
    logging.info("ShallowConvNet session model saved to %s", shallow_out)


def calibrate_tree_models(
    sample_rate: int,
    x_calib: np.ndarray,
    y_calib_encoded: np.ndarray,
    scaler_tree_out: str,
    rf_out: str,
    xgb_out: str,
) -> None:
    """
    Calibrates and saves tree-based models (Random Forest, XGBoost) for EEG classification.

    Args:
        sample_rate (int): Sample rate.
        x_calib (np.ndarray): Calibration data.
        y_calib_encoded (np.ndarray): Encoded labels.
        scaler_tree_out (str): Path to save scaler.
        rf_out (str): Path to save RF model.
        xgb_out (str): Path to save XGB model.

    Returns:
        None
    """
    x_feat = np.array([extract_features(w, fs=sample_rate) for w in x_calib])
    scaler_tree = StandardScaler()
    x_feat_scaled = scaler_tree.fit_transform(x_feat)
    joblib.dump(scaler_tree, scaler_tree_out)
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(x_feat_scaled, y_calib_encoded)
    joblib.dump(rf, rf_out)
    logging.info("Random Forest session model saved to %s", rf_out)
    # XGBoost
    xgb = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="mlogloss")
    xgb.fit(x_feat_scaled, y_calib_encoded)
    joblib.dump(xgb, xgb_out)
    logging.info("XGBoost session model saved to %s", xgb_out)


def calibrate_all_models_lsl(
    lsl_stream_handler: Any,
    config_path: str = "config.json",
    seconds_per_class: Optional[int] = None,
    save_dir: str = "models",
    verbose: bool = True,
) -> None:
    """
    Unified, LSL-aware calibration for all models (EEGNet, ShallowConvNet, RF, XGBoost).
    Uses config for parameters and saves session-specific models/scalers.

    Args:
        lsl_stream_handler (Any): LSL handler.
        config_path (str): Path to config file.
        seconds_per_class (int, optional): Seconds per class.
        save_dir (str): Directory to save models.
        verbose (bool): Verbosity flag.

    Returns:
        None
    """

    # --- Load config ---
    config = load_config(config_path)
    config = {k.upper(): v for k, v in config.items()}
    if seconds_per_class is None:
        seconds_per_class = config.get("CALIBRATION_SECONDS_PER_CLASS", 10)
    os.makedirs(save_dir, exist_ok=True)

    # --- Collect calibration data using LSL ---
    x_calib, y_calib = collect_lsl_calib_data(
        lsl_stream_handler,
        config["LABELS"],
        config["WINDOW_SIZE"],
        config["SAMPLING_RATE"],
        seconds_per_class
    )
    check_no_nan(x_calib, name="calibration data")
    check_labels_valid(y_calib, valid_labels=config["LABELS"])

    # --- Encode labels ---
    le = LabelEncoder()
    le.fit(config["LABELS"])
    y_calib_encoded = le.transform(y_calib)
    # Save label encoder and class list to generic session paths (no timestamp)
    le_path = os.path.join(save_dir, "eeg_label_encoder.pkl")
    label_classes_path = os.path.join(save_dir, "eeg_label_classes.npy")
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
        config,
        x_model,
        y_cat,
        scaler,
        save_dir,
        verbose,
    )
    scaler_tree_out = os.path.join(save_dir, "eeg_scaler_tree.pkl")
    rf_out = os.path.join(save_dir, "eeg_rf_model.pkl")
    xgb_out = os.path.join(save_dir, "eeg_xgb_model.pkl")
    calibrate_tree_models(
        config["SAMPLING_RATE"],
        x_calib,
        y_calib_encoded,
        scaler_tree_out,
        rf_out,
        xgb_out,
    )

    print("Session calibration complete. All models saved.")


def square(x: Any) -> Any:
    """
    Computes the element-wise square of a tensor using TensorFlow.

    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor: Squared tensor.
    """
    return tf.math.square(x)


def log(x: Any) -> Any:
    """
    Computes the element-wise natural logarithm of a tensor using TensorFlow, with values clipped to avoid log(0).

    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor: Log-transformed tensor.
    """
    return tf.math.log(tf.clip_by_value(x, 1e-7, tf.reduce_max(x)))


CUSTOM_OBJECTS = {"square": square, "log": log}


class EEGSystemError(Exception):
    """Base exception for EEG system errors."""


class DataLoadError(EEGSystemError):
    """Raised when data loading fails."""


class ModelLoadError(EEGSystemError):
    """Raised when model loading fails."""


class ConfigError(EEGSystemError):
    """Raised when configuration is invalid or missing."""


def handle_errors(main_func: Callable) -> Callable:
    """
    Decorator to catch and log uncaught exceptions in script entry points.

    Args:
        main_func (Callable): Main function to wrap.

    Returns:
        Callable: Wrapped function.
    """

    @functools.wraps(main_func)
    def wrapper(*args, **kwargs):
        try:
            return main_func(*args, **kwargs)
        except EEGSystemError as e:
            logging.error("EEGSystemError: %s", e)
            raise
        except Exception as e:
            logging.exception("Unhandled exception: %s", e)
            raise

    return wrapper


def save_json(data: Dict[str, Any], path: str) -> None:
    """
    Saves a dictionary as a JSON file.

    Args:
        data (dict): Data to save.
        path (str): File path.

    Returns:
        None
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def load_json(path: str) -> Dict[str, Any]:
    """
    Loads a dictionary from a JSON file.

    Args:
        path (str): File path.

    Returns:
        dict: Loaded data.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
