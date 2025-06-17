"""
Segment raw EEG data into overlapping windows for model training.

- Loads raw CSV data and filters by session type.
- Uses only EEG channel columns for features.
- Uses a utility function for windowing.
- Saves windowed data and labels as .npy files.
- Uses logging for status and error messages.
"""

import numpy as np
import pandas as pd
import logging
from utils import load_config, window_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("eeg_training.log", mode='a')
    ]
)

# Load configuration from config.json
config = load_config()

N_CHANNELS = config["N_CHANNELS"]
WINDOW_SIZE = config["WINDOW_SIZE"]
STEP_SIZE = config["STEP_SIZE"]
RAW_CSV = config["OUTPUT_CSV"]
WINDOWED_NPY = config["WINDOWED_NPY"]
WINDOWED_LABELS_NPY = config["WINDOWED_LABELS_NPY"]
SESSION_TYPES = config["SESSION_TYPES"]
USE_SESSION_TYPES = config["USE_SESSION_TYPES"]

# Load raw data
try:
    raw_data = pd.read_csv(RAW_CSV)
except FileNotFoundError:
    logging.error("Raw data file %s not found.", RAW_CSV)
    raise
except pd.errors.EmptyDataError:
    logging.error("Raw data file %s is empty.", RAW_CSV)
    raise
except Exception as e:
    logging.error("Failed to load raw data: %s", e)
    raise

# Filter by session_type if present
if 'session_type' in raw_data.columns:
    logging.info("Available session types: %s", raw_data['session_type'].unique())
    raw_data = raw_data[raw_data['session_type'].isin(USE_SESSION_TYPES)]
    logging.info("Using session types: %s, samples: %d", USE_SESSION_TYPES, len(raw_data))

# Use only EEG channel columns (ch_*) for features
eeg_cols = [col for col in raw_data.columns if col.startswith('ch_')]
X = raw_data[eeg_cols].values
labels = raw_data['label'].values

# Data validation checks
if np.isnan(X).any():
    logging.error("EEG data contains NaN values.")
    raise ValueError("EEG data contains NaN values.")
if pd.isnull(labels).any():
    logging.error("Labels contain NaN values.")
    raise ValueError("Labels contain NaN values.")
valid_labels = set(config["LABELS"])
if not set(np.unique(labels.flatten())).issubset(valid_labels):
    logging.error("Found labels outside expected set: %s", valid_labels)
    raise ValueError("Found labels outside expected set: %s" % valid_labels)

# Reshape X to [n_samples, n_channels]
if X.shape[1] != N_CHANNELS:
    logging.error("Expected %d channels, but got %d columns per sample.", N_CHANNELS, X.shape[1])
    raise ValueError("Expected %d channels, but got %d columns per sample." % (N_CHANNELS, X.shape[1]))
X = X.reshape(-1, N_CHANNELS)
labels = labels.reshape(-1, 1)

# Use the utility function for windowing
X_windows, y_windows = window_data(X, labels, WINDOW_SIZE, STEP_SIZE)

# Windowed data validation
if X_windows.shape[1:] != (WINDOW_SIZE, N_CHANNELS):
    logging.error("Windowed data shape mismatch.")
    raise ValueError("Windowed data shape mismatch.")
if X_windows.shape[0] != y_windows.shape[0]:
    logging.error("Number of windows and labels do not match.")
    raise ValueError("Number of windows and labels do not match.")

logging.info("Windowed data shape: %s, Labels shape: %s", X_windows.shape, y_windows.shape)

try:
    np.save(WINDOWED_NPY, X_windows)
    np.save(WINDOWED_LABELS_NPY, y_windows)
    logging.info("Saved windowed data to %s and %s", WINDOWED_NPY, WINDOWED_LABELS_NPY)
except Exception as e:
    logging.error("Failed to save windowed data: %s", e)
    raise
