"""
Perform real-time EEG direction prediction using trained EEGNet, Random Forest, and XGBoost
models.

- Streams live EEG data
- Applies windowing and scaling
- Predicts direction in real time using all models
- Supports ensemble voting and prints predictions to console

Input: Live EEG data stream, trained model files
Output: Real-time predictions and logs
"""

import logging
import os
import time
from collections import Counter

import joblib
import numpy as np
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
from brainflow.exit_codes import BrainFlowError
from keras.models import load_model
import tensorflow as tf

from utils import (
    collect_calibration_data,
    load_config,
    run_session_calibration,
    setup_logging,
    check_no_nan,
)

# Enable TensorFlow debug mode for better error messages
tf.data.experimental.enable_debug_mode()

# Ensure eager execution is enabled for Keras/TensorFlow compatibility
try:
    tf.config.run_functions_eagerly(True)
except RuntimeError:
    try:
        tf.compat.v1.enable_eager_execution()
    except RuntimeError:
        pass

setup_logging()

config = load_config()

N_CHANNELS = config["N_CHANNELS"]
WINDOW_SIZE = config["WINDOW_SIZE"]

params = BrainFlowInputParams()
params.serial_port = config["COM_PORT"]

try:
    board = BoardShim(BoardIds.CYTON_DAISY_BOARD.value, params)
    logging.info("Preparing session...")
    board.prepare_session()
    board.start_stream()
except FileNotFoundError as fnf:
    logging.error("Could not find BrainFlow board or driver: %s", fnf)
    exit(1)
except (OSError, ValueError, KeyError, BrainFlowError) as e:
    logging.error("Failed to start board session: %s", e)
    exit(1)

# Calibration step before prediction loop
LABELS = config["LABELS"]
CHANNELS = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD.value)
logging.info("Starting session calibration. Please follow the prompts.")
X_calib, y_calib = collect_calibration_data(
    board, CHANNELS, WINDOW_SIZE, LABELS, seconds_per_class=10, sample_rate=250
)
run_session_calibration(
    X_calib,
    y_calib,
    base_model_path=config["MODEL_CNN"],
    base_scaler_path=config["SCALER_CNN"],
    label_encoder_path=config["LABEL_ENCODER"],
    out_model_path="models/eeg_direction_model_session.h5",
    out_scaler_path="models/eeg_scaler_session.pkl",
    epochs=3,
    batch_size=16,
)
cnn = load_model("models/eeg_direction_model_session.h5")
scaler_cnn = joblib.load("models/eeg_scaler_session.pkl")
logging.info("Session calibration complete. Using session-specific model and scaler.")

required_files = [
    config["LABEL_ENCODER"],
    config["MODEL_RF"],
    config["SCALER_TREE"],
    config["MODEL_XGB"],
    "models/eeg_direction_model_session.h5",
    "models/eeg_scaler_session.pkl",
]
for f in required_files:
    if not os.path.exists(f):
        logging.error(
            "Required file missing: %s. Ensure all models and encoders are present.", f
        )
        exit(1)

try:
    rf = joblib.load(config["MODEL_RF"])
    xgb = joblib.load(config["MODEL_XGB"])
    scaler_tree = joblib.load(config["SCALER_TREE"])
    cnn = load_model("models/eeg_direction_model_session.h5")
    scaler_cnn = joblib.load("models/eeg_scaler_session.pkl")
    le = joblib.load(config["LABEL_ENCODER"])
except FileNotFoundError as fnf:
    logging.error("Model or encoder file not found: %s", fnf)
    exit(1)
except (
    OSError,
    joblib.externals.loky.process_executor.TerminatedWorkerError,
    ImportError,
    AttributeError,
) as e:
    logging.error("Failed to load models or encoders: %s", e)
    exit(1)

# Prompt user for prediction display mode after calibration
print("\nChoose prediction display mode:")
print("1. EEGNet only")
print("2. Ensemble (EEGNet, Random Forest, XGBoost)")
mode = input("Enter 1 or 2: ").strip()
show_ensemble = (mode == "2")

try:
    while True:
        eeg_window_scaled_tree = (
            None  # Initialize at the start of each loop (not a constant)
        )
        data = board.get_current_board_data(WINDOW_SIZE)
        if data.shape[1] >= WINDOW_SIZE:
            eeg_window = data[CHANNELS, -WINDOW_SIZE:]
            check_no_nan(eeg_window, name="Real-time EEG window")
            eeg_window = eeg_window.T
            # Tree-based models: flatten and scale
            eeg_window_flat = eeg_window.flatten().reshape(1, -1)
            eeg_window_scaled_tree = scaler_tree.transform(eeg_window_flat)
            # CNN: scale and reshape
            eeg_window_scaled_cnn = scaler_cnn.transform(eeg_window)
            eeg_window_cnn = np.expand_dims(eeg_window_scaled_cnn, axis=0)
            eeg_window_cnn = np.expand_dims(eeg_window_cnn, axis=-1)
            eeg_window_cnn = np.transpose(eeg_window_cnn, (0, 2, 1, 3))

            # Predictions
            pred_rf = rf.predict(eeg_window_scaled_tree)
            prob_rf = rf.predict_proba(eeg_window_scaled_tree).max()
            pred_label_rf = le.inverse_transform(pred_rf)[0]

            pred_xgb = xgb.predict(eeg_window_scaled_tree)
            prob_xgb = xgb.predict_proba(eeg_window_scaled_tree).max()
            pred_label_xgb = le.inverse_transform(pred_xgb)[0]

            pred_cnn = cnn.predict(eeg_window_cnn, verbose=0)
            prob_cnn = pred_cnn.max()
            pred_label_cnn = le.inverse_transform([np.argmax(pred_cnn)])[0]

            # Hard voting ensemble
            votes = [pred_label_cnn, pred_label_rf, pred_label_xgb]
            final_pred = Counter(votes).most_common(1)[0][0]

            # Print output based on user choice
            if show_ensemble:
                print(
                    f"\n--- Real-Time Prediction ---\n"
                    f"EEGNet:        {pred_label_cnn} (conf: {prob_cnn:.2f})\n"
                    f"Random Forest: {pred_label_rf} (conf: {prob_rf:.2f})\n"
                    f"XGBoost:       {pred_label_xgb} (conf: {prob_xgb:.2f})\n"
                    f"Ensemble:      {final_pred} | Votes: {votes}\n"
                )
                logging.info(
                    "Ensemble Prediction: %s | Votes: %s | EEGNet: %s (%.2f) | RF: %s (%.2f) | XGB: %s (%.2f)",
                    final_pred, votes,
                    pred_label_cnn, prob_cnn,
                    pred_label_rf, prob_rf,
                    pred_label_xgb, prob_xgb
                )
            else:
                print(
                    f"\n--- Real-Time Prediction (EEGNet Only) ---\n"
                    f"EEGNet: {pred_label_cnn} (conf: {prob_cnn:.2f})\n"
                )
                logging.info(
                    "EEGNet Only Prediction: %s (%.2f)", pred_label_cnn, prob_cnn
                )
        else:
            logging.info(
                "Waiting for enough data... (current samples: %d)", data.shape[1]
            )
        time.sleep(1)
except KeyboardInterrupt:
    logging.info("\nStopping...")
except (OSError, RuntimeError, ValueError) as e:
    logging.error("Error during real-time prediction loop: %s", e)
finally:
    board.stop_stream()
    board.release_session()
    logging.info("Session closed.")
