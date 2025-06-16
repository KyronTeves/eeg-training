import numpy as np
import pandas as pd
from utils import load_config, window_data

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
raw_data = pd.read_csv(RAW_CSV)

# Filter by session_type if present
if 'session_type' in raw_data.columns:
    print(f"Available session types: {raw_data['session_type'].unique()}")
    raw_data = raw_data[raw_data['session_type'].isin(USE_SESSION_TYPES)]
    print(f"Using session types: {USE_SESSION_TYPES}, samples: {len(raw_data)}")

# Use only EEG channel columns (ch_*) for features
eeg_cols = [col for col in raw_data.columns if col.startswith('ch_')]
X = raw_data[eeg_cols].values
labels = raw_data['label'].values

# Reshape X to [n_samples, n_channels]
if X.shape[1] != N_CHANNELS:
    raise ValueError(f"Expected {N_CHANNELS} channels, but got {X.shape[1]} columns per sample.")
X = X.reshape(-1, N_CHANNELS)
labels = labels.reshape(-1, 1)

# Use the utility function for windowing
X_windows, y_windows = window_data(X, labels, WINDOW_SIZE, STEP_SIZE)

print(f"Windowed data shape: {X_windows.shape}, Labels shape: {y_windows.shape}")

# Save as .npy for fast loading in training
np.save(WINDOWED_NPY, X_windows)
np.save(WINDOWED_LABELS_NPY, y_windows)
print(f"Saved windowed data to {WINDOWED_NPY} and {WINDOWED_LABELS_NPY}")
