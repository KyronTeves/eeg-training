"""
Test trained models on held-out EEG data windows.

- Loads test data and models.
- Windows the data using the utility function.
- Evaluates Conv1D, Random Forest, and XGBoost models.
- Prints predictions and actual labels for comparison.
- Uses logging for status and error messages.
"""

import logging
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from keras.models import load_model  # type: ignore

from utils import load_config, window_data, setup_logging, check_no_nan, check_labels_valid

setup_logging()

config = load_config()

N_CHANNELS = config["N_CHANNELS"]
WINDOW_SIZE = config["WINDOW_SIZE"]
STEP_SIZE = config["STEP_SIZE"]
CSV_FILE = config["OUTPUT_CSV"]
TEST_SESSION_TYPES = config["TEST_SESSION_TYPES"]
NUM_TEST_SAMPLES = config["NUM_TEST_SAMPLES"]

try:
    logging.info("Loading data from %s ...", CSV_FILE)
    df = pd.read_csv(CSV_FILE)
    test_df = df[df["session_type"].isin(TEST_SESSION_TYPES)]
    logging.info("Test samples: %d", len(test_df))
except (pd.errors.EmptyDataError, OSError, ValueError, KeyError) as e:
    logging.error("Failed to load or filter test data: %s", e)
    raise

# Use the utility function for windowing
eeg_cols = [col for col in test_df.columns if col.startswith("ch_")]
X = test_df[eeg_cols].values
labels = test_df["label"].values

check_no_nan(X, name="EEG data")
check_labels_valid(labels, valid_labels=config["LABELS"], name="Labels")

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

logging.info("Test windows: %s", X_windows.shape)

# Proceed with all windowed test data as originally intended
try:
    le = joblib.load(config["LABEL_ENCODER"])
    scaler = joblib.load(config["SCALER_CNN"])
    model = load_model(config["MODEL_CNN"])
    rf = joblib.load(config["MODEL_RF"])
    xgb = joblib.load(config["MODEL_XGB"])
except (ImportError, OSError, AttributeError) as e:
    logging.error("Failed to load models or encoders: %s", e)
    raise

X_windows_flat = X_windows.reshape(-1, N_CHANNELS)
X_windows_scaled = scaler.transform(X_windows_flat).reshape(X_windows.shape)
X_windows_flat_scaled = X_windows_scaled.reshape(X_windows.shape[0], -1)

# Prepare data for EEGNet: (batch, window, channels) -> (batch, channels, window, 1)
X_windows_eegnet = np.expand_dims(X_windows_scaled, -1)
X_windows_eegnet = np.transpose(X_windows_eegnet, (0, 2, 1, 3))

num_samples = min(NUM_TEST_SAMPLES, X_windows.shape[0])
indices = np.random.choice(X_windows.shape[0], num_samples, replace=False)

CORRECT = 0
ENSEMBLE_CORRECT = 0
RULE_BASED_CORRECT = 0
for idx in indices:
    actual_label = y_windows[idx]
    sample_eegnet = X_windows_eegnet[idx].reshape(1, N_CHANNELS, WINDOW_SIZE, 1)
    pred_eegnet = model.predict(sample_eegnet)
    pred_label_eegnet = le.inverse_transform([np.argmax(pred_eegnet)])[0]
    # Random Forest
    sample_rf = X_windows_flat_scaled[idx].reshape(1, -1)
    pred_rf = rf.predict(sample_rf)
    pred_label_rf = le.inverse_transform(pred_rf)[0]
    # XGBoost
    pred_xgb = xgb.predict(sample_rf)
    pred_label_xgb = le.inverse_transform(pred_xgb)[0]
    # Hard voting ensemble
    votes = [pred_label_eegnet, pred_label_rf, pred_label_xgb]
    final_pred = Counter(votes).most_common(1)[0][0]
    ensemble_match = final_pred == actual_label
    if ensemble_match:
        ENSEMBLE_CORRECT += 1
    # Rule-based ensemble
    if pred_label_eegnet in ["left", "right", "up", "down"]:
        rule_pred = pred_label_eegnet
    elif pred_label_rf == "neutral" or pred_label_xgb == "neutral":
        rule_pred = "neutral"
    else:
        rule_pred = final_pred
    rule_match = rule_pred == actual_label
    if rule_match:
        RULE_BASED_CORRECT += 1
    # Individual EEGNet accuracy
    match = pred_label_eegnet == actual_label
    if match:
        CORRECT += 1
    logging.info("Actual label:   %s", actual_label)
    logging.info("EEGNet Predicted label: %s | Match: %s", pred_label_eegnet, match)
    logging.info("Random Forest Predicted label: %s", pred_label_rf)
    logging.info("XGBoost Predicted label: %s", pred_label_xgb)
    logging.info("Ensemble (hard voting) label: %s | Match: %s", final_pred, ensemble_match)
    logging.info("Ensemble (rule-based) label: %s | Match: %s", rule_pred, rule_match)
    logging.info("-")
logging.info(
    "EEGNet accuracy on %d test samples: %d/%d (%.2f%%)",
    num_samples,
    CORRECT,
    num_samples,
    100 * CORRECT / num_samples,
)
logging.info(
    "Ensemble (hard voting) accuracy on %d test samples: %d/%d (%.2f%%)",
    num_samples,
    ENSEMBLE_CORRECT,
    num_samples,
    100 * ENSEMBLE_CORRECT / num_samples,
)
logging.info(
    "Ensemble (rule-based) accuracy on %d test samples: %d/%d (%.2f%%)",
    num_samples,
    RULE_BASED_CORRECT,
    num_samples,
    100 * RULE_BASED_CORRECT / num_samples,
)
