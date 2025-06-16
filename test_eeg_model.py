"""
Test trained models on held-out EEG data windows.

- Loads test data and models.
- Windows the data using the utility function.
- Evaluates Conv1D, Random Forest, and XGBoost models.
- Prints predictions and actual labels for comparison.
- Uses logging for status and error messages.
"""

import numpy as np
import pandas as pd
import joblib
import logging
from tensorflow.keras.models import load_model # type: ignore
from EEGModels import EEGNet

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from utils import load_config, window_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("eeg_training.log", mode='a')
    ]
)

config = load_config()

N_CHANNELS = config["N_CHANNELS"]
WINDOW_SIZE = config["WINDOW_SIZE"]
STEP_SIZE = config["STEP_SIZE"]
CSV_FILE = config["OUTPUT_CSV"]
TEST_SESSION_TYPES = config["TEST_SESSION_TYPES"]
NUM_TEST_SAMPLES = config["NUM_TEST_SAMPLES"]

try:
    logging.info(f"Loading data from {CSV_FILE} ...")
    df = pd.read_csv(CSV_FILE)
    test_df = df[df['session_type'].isin(TEST_SESSION_TYPES)]
    logging.info(f"Test samples: {len(test_df)}")
except FileNotFoundError:
    logging.error(f"Test data file {CSV_FILE} not found.")
    raise
except pd.errors.EmptyDataError:
    logging.error(f"Test data file {CSV_FILE} is empty.")
    raise
except Exception as e:
    logging.error(f"Failed to load or filter test data: {e}")
    raise

# Use the utility function for windowing
eeg_cols = [col for col in test_df.columns if col.startswith('ch_')]
X = test_df[eeg_cols].values
labels = test_df['label'].values

# Data validation checks
if np.isnan(X).any():
    logging.error("EEG data contains NaN values.")
    raise ValueError("EEG data contains NaN values.")
if pd.isnull(labels).any():
    logging.error("Labels contain NaN values.")
    raise ValueError("Labels contain NaN values.")
valid_labels = set(config["LABELS"])
if not set(np.unique(labels.flatten())).issubset(valid_labels):
    logging.error(f"Found labels outside expected set: {valid_labels}")
    raise ValueError(f"Found labels outside expected set: {valid_labels}")

X = X.reshape(-1, N_CHANNELS)
labels = labels.reshape(-1, 1)
X_windows, y_windows = window_data(X, labels, WINDOW_SIZE, STEP_SIZE)

# Windowed data validation
if X_windows.shape[1:] != (WINDOW_SIZE, N_CHANNELS):
    logging.error("Windowed data shape mismatch.")
    raise ValueError("Windowed data shape mismatch.")
if X_windows.shape[0] != y_windows.shape[0]:
    logging.error("Number of windows and labels do not match.")
    raise ValueError("Number of windows and labels do not match.")

logging.info(f"Test windows: {X_windows.shape}")

try:
    le = joblib.load(config["LABEL_ENCODER"])
    scaler = joblib.load(config["SCALER_CNN"])
    model = load_model(config["MODEL_CNN"])
    rf = joblib.load(config["MODEL_RF"])
    xgb = joblib.load(config["MODEL_XGB"])
except FileNotFoundError as fnf:
    logging.error(f"Model or encoder file not found: {fnf}")
    raise
except Exception as e:
    logging.error(f"Failed to load models or encoders: {e}")
    raise

X_windows_flat = X_windows.reshape(-1, N_CHANNELS)
X_windows_scaled = scaler.transform(X_windows_flat).reshape(X_windows.shape)
X_windows_flat_scaled = X_windows_scaled.reshape(X_windows.shape[0], -1)

# Prepare data for EEGNet: (batch, window, channels) -> (batch, channels, window, 1)
X_windows_eegnet = np.expand_dims(X_windows_scaled, -1)
X_windows_eegnet = np.transpose(X_windows_eegnet, (0, 2, 1, 3))

num_samples = min(NUM_TEST_SAMPLES, X_windows.shape[0])
indices = np.random.choice(X_windows.shape[0], num_samples, replace=False)

for idx in indices:
    actual_label = y_windows[idx]
    sample_eegnet = X_windows_eegnet[idx].reshape(1, N_CHANNELS, WINDOW_SIZE, 1)
    pred_eegnet = model.predict(sample_eegnet)
    pred_label_eegnet = le.inverse_transform([np.argmax(pred_eegnet)])[0]
    logging.info(f"Actual label:   {actual_label}")
    logging.info(f"EEGNet Predicted label: {pred_label_eegnet}")
    # Random Forest
    sample_rf = X_windows_flat_scaled[idx].reshape(1, -1)
    pred_rf = rf.predict(sample_rf)
    pred_label_rf = le.inverse_transform(pred_rf)[0]
    # XGBoost
    pred_xgb = xgb.predict(sample_rf)
    pred_label_xgb = le.inverse_transform(pred_xgb)[0]
    logging.info(f"Random Forest Predicted label: {pred_label_rf}")
    logging.info(f"XGBoost Predicted label: {pred_label_xgb}")
    logging.info("-")
