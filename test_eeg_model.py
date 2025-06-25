"""
Evaluate trained EEGNet, Random Forest, and XGBoost models on held-out EEG data windows.

- Loads test data and trained models
- Applies windowing and scaling
- Computes predictions for each model
- Reports accuracy and ensemble results for comparison

Input: Labeled EEG CSV file, trained model files
Output: Evaluation metrics, predictions, and logs
"""

import logging
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from keras.models import load_model  # type: ignore
from sklearn.metrics import classification_report, confusion_matrix

from utils import (
    load_config,
    window_data,
    setup_logging,
    check_no_nan,
    check_labels_valid,
    extract_features,
)

setup_logging()  # Set up consistent logging to file and console

config = load_config()

N_CHANNELS = config["N_CHANNELS"]
WINDOW_SIZE = config["WINDOW_SIZE"]
STEP_SIZE = config["STEP_SIZE"]
CSV_FILE = config["OUTPUT_CSV"]
TEST_SESSION_TYPES = config["TEST_SESSION_TYPES"]
SAMPLING_RATE = config["SAMPLING_RATE"]


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

check_no_nan(X, name="EEG data")  # Validate no NaNs in EEG data
check_labels_valid(
    labels, valid_labels=config["LABELS"], name="Labels"
)  # Validate labels

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

# --- Feature Extraction for Tree-based Models ---
logging.info("Extracting features for tree-based models...")
X_features = np.array(
    [extract_features(window, SAMPLING_RATE) for window in X_windows]
)
logging.info("Feature extraction complete. Feature shape: %s", X_features.shape)


# Proceed with all windowed test data as originally intended
try:
    le = joblib.load(config["LABEL_ENCODER"])
    scaler_cnn = joblib.load(config["SCALER_CNN"])
    scaler_tree = joblib.load(config["SCALER_TREE"])
    model_cnn = load_model(config["MODEL_CNN"])
    rf = joblib.load(config["MODEL_RF"])
    xgb = joblib.load(config["MODEL_XGB"])
except (ImportError, OSError, AttributeError) as e:
    logging.error("Failed to load models or encoders: %s", e)
    raise

# --- Scaling ---
# CNN
X_windows_flat = X_windows.reshape(-1, N_CHANNELS)
X_windows_scaled_cnn = scaler_cnn.transform(X_windows_flat).reshape(X_windows.shape)
X_windows_eegnet = np.expand_dims(X_windows_scaled_cnn, -1)
X_windows_eegnet = np.transpose(X_windows_eegnet, (0, 2, 1, 3))

# Tree-based
X_features_scaled = scaler_tree.transform(X_features)


# --- Predictions ---
logging.info("Generating predictions for all models...")
pred_cnn_prob = model_cnn.predict(X_windows_eegnet)
pred_cnn_labels = np.argmax(pred_cnn_prob, axis=1)

pred_rf_labels = rf.predict(X_features_scaled)
pred_xgb_labels = xgb.predict(X_features_scaled)

y_true_labels = le.transform(y_windows.ravel())

# --- Ensemble Prediction (Hard Voting) ---
pred_ensemble_labels = []
for i in range(len(y_true_labels)):
    # Get votes from the string labels
    vote_cnn = le.inverse_transform([pred_cnn_labels[i]])[0]
    vote_rf = le.inverse_transform([pred_rf_labels[i]])[0]
    vote_xgb = le.inverse_transform([pred_xgb_labels[i]])[0]

    votes = [vote_cnn, vote_rf, vote_xgb]
    final_pred = Counter(votes).most_common(1)[0][0]
    pred_ensemble_labels.append(final_pred)

# Convert string labels back to numeric for classification_report
pred_ensemble_numeric = le.transform(pred_ensemble_labels)


# --- Evaluation ---
logging.info("--- EEGNet Evaluation ---")
logging.info("Confusion Matrix:\n%s", confusion_matrix(y_true_labels, pred_cnn_labels))
logging.info(
    "Classification Report:\n%s",
    classification_report(y_true_labels, pred_cnn_labels, target_names=le.classes_),
)

logging.info("--- Random Forest Evaluation ---")
logging.info("Confusion Matrix:\n%s", confusion_matrix(y_true_labels, pred_rf_labels))
logging.info(
    "Classification Report:\n%s",
    classification_report(y_true_labels, pred_rf_labels, target_names=le.classes_),
)

logging.info("--- XGBoost Evaluation ---")
logging.info("Confusion Matrix:\n%s", confusion_matrix(y_true_labels, pred_xgb_labels))
logging.info(
    "Classification Report:\n%s",
    classification_report(y_true_labels, pred_xgb_labels, target_names=le.classes_),
)

logging.info("--- Ensemble (Hard Voting) Evaluation ---")
logging.info(
    "Confusion Matrix:\n%s", confusion_matrix(y_true_labels, pred_ensemble_numeric)
)
logging.info(
    "Classification Report:\n%s",
    classification_report(
        y_true_labels, pred_ensemble_numeric, target_names=le.classes_
    ),
)
