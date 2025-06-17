"""
Perform real-time EEG prediction using trained models.

- Loads trained models and scalers.
- Streams live EEG data from the board.
- Applies windowing and predicts direction in real time.
- Supports Random Forest, XGBoost, or both.
- Uses logging for status and error messages.
"""

import logging
import os
import time

import joblib
import numpy as np
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
from keras.models import load_model

from utils import (collect_calibration_data, load_config,
                   run_session_calibration)

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
except (OSError, ValueError, KeyError) as e:
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
    X_calib, y_calib,
    base_model_path=config["MODEL_CNN"],
    base_scaler_path=config["SCALER_CNN"],
    label_encoder_path=config["LABEL_ENCODER"],
    out_model_path="models/eeg_direction_model_session.h5",
    out_scaler_path="models/eeg_scaler_session.pkl",
    epochs=3,
    batch_size=16
)
cnn = load_model("models/eeg_direction_model_session.h5")
scaler_cnn = joblib.load("models/eeg_scaler_session.pkl")
logging.info("Session calibration complete. Using session-specific model and scaler.")

logging.info("Select model for real-time prediction:")
logging.info("1: Random Forest\n2: XGBoost\n3: Both\n4: EEGNet (CNN)")
model_choice = input("Enter choice (1/2/3/4): ").strip()
use_rf = model_choice in ['1', '3']
use_xgb = model_choice in ['2', '3']
use_cnn = model_choice == '4'

required_files = [config["LABEL_ENCODER"]]
if use_rf:
    required_files += [config["MODEL_RF"], config["SCALER_TREE"]]
if use_xgb:
    required_files += [config["MODEL_XGB"], config["SCALER_TREE"]]
if use_cnn:
    required_files += [config["MODEL_CNN"], config["SCALER_CNN"]]
for f in required_files:
    if not os.path.exists(f):
        logging.error("Required file missing: %s. Ensure all models and encoders are present.", f)
        exit(1)

scaler_tree = None  # Ensure scaler_tree is always defined
eeg_window_scaled_tree = None  # Ensure eeg_window_scaled_tree is always defined
try:
    if use_rf:
        rf = joblib.load(config["MODEL_RF"])
        scaler_tree = joblib.load(config["SCALER_TREE"])
    if use_xgb:
        xgb = joblib.load(config["MODEL_XGB"])
        scaler_tree = joblib.load(config["SCALER_TREE"])
    if use_cnn:
        cnn = load_model(config["MODEL_CNN"])
        scaler_cnn = joblib.load(config["SCALER_CNN"])
    le = joblib.load(config["LABEL_ENCODER"])
except FileNotFoundError as fnf:
    logging.error("Model or encoder file not found: %s", fnf)
    exit(1)
except (OSError, joblib.externals.loky.process_executor.TerminatedWorkerError, ImportError, AttributeError) as e:
    logging.error("Failed to load models or encoders: %s", e)
    exit(1)

try:
    while True:
        data = board.get_current_board_data(WINDOW_SIZE)
        if data.shape[1] >= WINDOW_SIZE:
            eeg_window = data[CHANNELS, -WINDOW_SIZE:]
            eeg_window = eeg_window.T
            if use_rf or use_xgb:
                eeg_window_flat = eeg_window.flatten().reshape(1, -1)
                eeg_window_scaled_tree = scaler_tree.transform(eeg_window_flat)
            if use_cnn:
                eeg_window_scaled_cnn = scaler_cnn.transform(eeg_window)  # (window, channels)
                eeg_window_cnn = np.expand_dims(eeg_window_scaled_cnn, axis=0)   # (1, window, channels)
                eeg_window_cnn = np.expand_dims(eeg_window_cnn, axis=-1)         # (1, window, channels, 1)
                eeg_window_cnn = np.transpose(eeg_window_cnn, (0, 2, 1, 3))      # (1, channels, window, 1)
            if use_rf:
                pred_rf = rf.predict(eeg_window_scaled_tree)
                prob_rf = rf.predict_proba(eeg_window_scaled_tree).max()
                pred_label_rf = le.inverse_transform(pred_rf)[0]
                logging.info("Random Forest Prediction: %s (confidence: %.2f)", pred_label_rf, prob_rf)
            if use_xgb:
                pred_xgb = xgb.predict(eeg_window_scaled_tree)
                prob_xgb = xgb.predict_proba(eeg_window_scaled_tree).max()
                pred_label_xgb = le.inverse_transform(pred_xgb)[0]
                logging.info("XGBoost Prediction: %s (confidence: %.2f)", pred_label_xgb, prob_xgb)
            if use_cnn:
                pred_cnn = cnn.predict(eeg_window_cnn)
                prob_cnn = pred_cnn.max()
                pred_label_cnn = le.inverse_transform([np.argmax(pred_cnn)])[0]
                logging.info("EEGNet Prediction: %s (confidence: %.2f)", pred_label_cnn, prob_cnn)
        else:
            logging.info("Waiting for enough data... (current samples: %d)", data.shape[1])
        time.sleep(1)
except KeyboardInterrupt:
    logging.info("\nStopping...")
except (OSError, RuntimeError, ValueError) as e:
    logging.error("Error during real-time prediction loop: %s", e)
finally:
    board.stop_stream()
    board.release_session()
    logging.info("Session closed.")
