"""
calibration_utils.py

Modular calibration utilities for EEG session adaptation.
- collect_calibration_data: Collects labeled calibration data from the user via the board.
- run_session_calibration: Windows, preprocesses, and fine-tunes the model/scaler for the session.
"""

import time
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from EEGModels import EEGNet
from sklearn.preprocessing import StandardScaler
import logging
import os

def collect_calibration_data(board, CHANNELS, WINDOW_SIZE, LABELS, seconds_per_class=10, sample_rate=250):
    """
    Collect labeled calibration data for each class from the user.
    Returns X_calib (windows, window, channels), y_calib (windows,)
    """
    calib_X = []
    calib_y = []
    for label in LABELS:
        input(f"Get ready for calibration: {label}. Press Enter to start recording {seconds_per_class} seconds...")
        data = []
        start_time = time.time()
        while time.time() - start_time < seconds_per_class:
            eeg = board.get_current_board_data(WINDOW_SIZE)
            if eeg.shape[1] >= WINDOW_SIZE:
                eeg_window = eeg[CHANNELS, -WINDOW_SIZE:].T
                data.append(eeg_window)
            time.sleep(WINDOW_SIZE / sample_rate)
        calib_X.extend(data)
        calib_y.extend([label] * len(data))
        logging.info(f"Collected {len(data)} windows for label '{label}'.")
    return np.array(calib_X), np.array(calib_y)

def run_session_calibration(X_calib, y_calib, base_model_path, base_scaler_path, label_encoder_path, out_model_path, out_scaler_path, epochs=3, batch_size=16):
    """
    Windows, preprocesses, and fine-tunes the model/scaler for the session.
    """
    le = joblib.load(label_encoder_path)
    y_calib_encoded = le.transform(y_calib)
    y_calib_cat = to_categorical(y_calib_encoded)
    scaler = StandardScaler()
    X_calib_flat = X_calib.reshape(-1, X_calib.shape[-1])
    scaler.fit(X_calib_flat)
    X_calib_scaled = scaler.transform(X_calib_flat).reshape(X_calib.shape)
    # Prepare for EEGNet: (batch, window, channels) -> (batch, channels, window, 1)
    X_calib_eegnet = np.expand_dims(X_calib_scaled, -1)
    X_calib_eegnet = np.transpose(X_calib_eegnet, (0, 2, 1, 3))
    model = load_model(base_model_path)
    model.fit(X_calib_eegnet, y_calib_cat, epochs=epochs, batch_size=batch_size, verbose=1)
    model.save(out_model_path)
    joblib.dump(scaler, out_scaler_path)
    logging.info(f"Session calibration complete. Model saved to {out_model_path}, scaler saved to {out_scaler_path}.")
