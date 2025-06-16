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
logging.info("1: Random Forest\n2: XGBoost\n3: Both")
model_choice = input("Enter choice (1/2/3): ").strip()
use_rf = model_choice in ['1', '3']
use_xgb = model_choice in ['2', '3']

required_files = [config["MODEL_RF"], config["MODEL_XGB"], config["SCALER_TREE"], config["LABEL_ENCODER"]]
for f in required_files:
    if not os.path.exists(f):
        logging.error(f"Required file missing: {f}. Please ensure all models and encoders are present.")
        exit(1)

try:
    rf = joblib.load(config["MODEL_RF"])
    xgb = joblib.load(config["MODEL_XGB"])
    scaler = joblib.load(config["SCALER_TREE"])
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
            eeg_window_flat = eeg_window.flatten().reshape(1, -1)
            eeg_window_scaled = scaler.transform(eeg_window_flat)
            if use_rf:
                pred_rf = rf.predict(eeg_window_scaled)
                prob_rf = rf.predict_proba(eeg_window_scaled).max()
                pred_label_rf = le.inverse_transform(pred_rf)[0]
                logging.info(f"Random Forest Prediction: {pred_label_rf} (confidence: {prob_rf:.2f})")
            if use_xgb:
                pred_xgb = xgb.predict(eeg_window_scaled)
                prob_xgb = xgb.predict_proba(eeg_window_scaled).max()
                pred_label_xgb = le.inverse_transform(pred_xgb)[0]
                logging.info(f"XGBoost Prediction: {pred_label_xgb} (confidence: {prob_xgb:.2f})")
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
