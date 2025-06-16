import numpy as np
import pandas as pd
import logging
from utils import load_config, window_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
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
except Exception as e:
    logging.error(f"Failed to load raw data: {e}")
    raise

# Filter by session_type if present
if 'session_type' in raw_data.columns:
    logging.info(f"Available session types: {raw_data['session_type'].unique()}")
    raw_data = raw_data[raw_data['session_type'].isin(USE_SESSION_TYPES)]
    logging.info(f"Using session types: {USE_SESSION_TYPES}, samples: {len(raw_data)}")

# Use only EEG channel columns (ch_*) for features
eeg_cols = [col for col in raw_data.columns if col.startswith('ch_')]
X = raw_data[eeg_cols].values
labels = raw_data['label'].values

# Reshape X to [n_samples, n_channels]
if X.shape[1] != N_CHANNELS:
    logging.error(f"Expected {N_CHANNELS} channels, but got {X.shape[1]} columns per sample.")
    raise ValueError(f"Expected {N_CHANNELS} channels, but got {X.shape[1]} columns per sample.")
X = X.reshape(-1, N_CHANNELS)
labels = labels.reshape(-1, 1)

# Use the utility function for windowing
X_windows, y_windows = window_data(X, labels, WINDOW_SIZE, STEP_SIZE)

logging.info(f"Windowed data shape: {X_windows.shape}, Labels shape: {y_windows.shape}")

# Save as .npy for fast loading in training
try:
    np.save(WINDOWED_NPY, X_windows)
    np.save(WINDOWED_LABELS_NPY, y_windows)
    logging.info(f"Saved windowed data to {WINDOWED_NPY} and {WINDOWED_LABELS_NPY}")
except Exception as e:
    logging.error(f"Failed to save windowed data: {e}")
    raise
