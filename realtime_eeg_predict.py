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
    extract_features,
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
x_calib, y_calib = collect_calibration_data(
    board, CHANNELS, WINDOW_SIZE, LABELS, seconds_per_class=10, sample_rate=250
)
run_session_calibration(
    x_calib,
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


try:
    # Performance monitoring variables
    prediction_times = []
    recent_predictions = []

    while True:
        # Start timing for this prediction
        start_time = time.time()

        data = board.get_current_board_data(WINDOW_SIZE)
        if data.shape[1] >= WINDOW_SIZE:
            eeg_window = data[CHANNELS, -WINDOW_SIZE:].T  # Transpose
            check_no_nan(eeg_window, name="Real-time EEG window")

            # --- Feature Extraction & Scaling ---
            # CNN
            eeg_window_scaled_cnn = scaler_cnn.transform(eeg_window)
            eeg_window_cnn = np.expand_dims(eeg_window_scaled_cnn, axis=0)
            eeg_window_cnn = np.expand_dims(eeg_window_cnn, axis=-1)
            eeg_window_cnn = np.transpose(eeg_window_cnn, (0, 2, 1, 3))

            # Tree-based
            SAMPLING_RATE = 250  # TODO: Add to config
            features = extract_features(eeg_window, SAMPLING_RATE).reshape(1, -1)
            features_scaled = scaler_tree.transform(features)

            # --- Predictions ---
            pred_cnn_prob = cnn.predict(eeg_window_cnn, verbose=0)[0]
            pred_label_cnn = le.inverse_transform([np.argmax(pred_cnn_prob)])[0]

            pred_rf_prob = rf.predict_proba(features_scaled)[0]
            pred_label_rf = le.inverse_transform([np.argmax(pred_rf_prob)])[0]

            pred_xgb_prob = xgb.predict_proba(features_scaled)[0]
            pred_label_xgb = le.inverse_transform([np.argmax(pred_xgb_prob)])[0]

            # --- Weighted Voting Ensemble ---
            # Combine probabilities from all models
            combined_probs = pred_cnn_prob + pred_rf_prob + pred_xgb_prob
            final_pred_idx = np.argmax(combined_probs)
            final_pred_label = le.inverse_transform([final_pred_idx])[0]
            final_pred_conf = combined_probs[final_pred_idx] / 3  # Average confidence

            # --- Display Output ---
            print(
                f"\n--- Real-Time Prediction ---\n"
                f"EEGNet:        {pred_label_cnn} (conf: {np.max(pred_cnn_prob):.2f})\n"
                f"Random Forest: {pred_label_rf} (conf: {np.max(pred_rf_prob):.2f})\n"
                f"XGBoost:       {pred_label_xgb} (conf: {np.max(pred_xgb_prob):.2f})\n"
                f"Ensemble:      {final_pred_label} (avg conf: {final_pred_conf:.2f})\n"
            )

            # Performance monitoring
            pred_time = time.time() - start_time
            prediction_times.append(pred_time)
            recent_predictions.append(final_pred_label)

            # Enhanced logging with performance and pattern info
            if len(recent_predictions) >= 10:
                pattern_info = (
                    f"Last 10: {dict(Counter(recent_predictions[-10:]))}"
                )
                avg_time = sum(prediction_times[-10:]) / len(
                    prediction_times[-10:]
                )
                logging.info(
                    "Ensemble Prediction: %s | Avg time: %.3fs | %s | EEGNet: %s (%.2f) | RF: %s (%.2f) | XGB: %s (%.2f)",
                    final_pred_label,
                    avg_time,
                    pattern_info,
                    pred_label_cnn,
                    np.max(pred_cnn_prob),
                    pred_label_rf,
                    np.max(pred_rf_prob),
                    pred_label_xgb,
                    np.max(pred_xgb_prob),
                )
            else:
                logging.info(
                    "Ensemble Prediction: %s | EEGNet: %s (%.2f) | RF: %s (%.2f) | XGB: %s (%.2f)",
                    final_pred_label,
                    pred_label_cnn,
                    np.max(pred_cnn_prob),
                    pred_label_rf,
                    np.max(pred_rf_prob),
                    pred_label_xgb,
                    np.max(pred_xgb_prob),
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
