"""
Perform real-time EEG prediction using trained models.

- Loads trained models and scalers.
- Streams live EEG data from the board.
- Applies windowing and predicts direction in real time.
- Supports Random Forest, XGBoost, or both.
- Uses logging for status and error messages.
"""

import time
import numpy as np
import joblib
import os
import logging
from utils import load_config
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import load_model # type: ignore

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
    logging.error(f"Could not find BrainFlow board or driver: {fnf}")
    exit(1)
except Exception as e:
    logging.error(f"Failed to start board session: {e}")
    exit(1)

CHANNELS = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD.value)

logging.info("Started EEG stream. Waiting for data to accumulate...")
time.sleep(3)

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
        logging.error(f"Required file missing: {f}. Please ensure all models and encoders are present.")
        exit(1)

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
    logging.error(f"Model or encoder file not found: {fnf}")
    exit(1)
except Exception as e:
    logging.error(f"Failed to load models or encoders: {e}")
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
                eeg_window_scaled_cnn = scaler_cnn.transform(eeg_window)
                eeg_window_cnn = np.expand_dims(eeg_window_scaled_cnn, axis=0)  # (1, window, channels)
                eeg_window_cnn = np.transpose(eeg_window_cnn, (0, 2, 1, 3))      # (1, channels, window, 1)
            if use_rf:
                pred_rf = rf.predict(eeg_window_scaled_tree)
                prob_rf = rf.predict_proba(eeg_window_scaled_tree).max()
                pred_label_rf = le.inverse_transform(pred_rf)[0]
                logging.info(f"Random Forest Prediction: {pred_label_rf} (confidence: {prob_rf:.2f})")
            if use_xgb:
                pred_xgb = xgb.predict(eeg_window_scaled_tree)
                prob_xgb = xgb.predict_proba(eeg_window_scaled_tree).max()
                pred_label_xgb = le.inverse_transform(pred_xgb)[0]
                logging.info(f"XGBoost Prediction: {pred_label_xgb} (confidence: {prob_xgb:.2f})")
            if use_cnn:
                pred_cnn = cnn.predict(eeg_window_cnn)
                prob_cnn = pred_cnn.max()
                pred_label_cnn = le.inverse_transform([np.argmax(pred_cnn)])[0]
                logging.info(f"EEGNet Prediction: {pred_label_cnn} (confidence: {prob_cnn:.2f})")
        else:
            logging.info(f"Waiting for enough data... (current samples: {data.shape[1]})")
        time.sleep(1)
except KeyboardInterrupt:
    logging.info("\nStopping...")
except Exception as e:
    logging.error(f"Error during real-time prediction loop: {e}")
finally:
    board.stop_stream()
    board.release_session()
    logging.info("Session closed.")
