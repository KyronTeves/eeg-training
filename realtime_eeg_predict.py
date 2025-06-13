"""
Real-time EEG prediction using trained models (Random Forest, XGBoost) and BrainFlow.
"""
import time
import numpy as np
import joblib
import os
import sys
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Parameters (must match training)
N_CHANNELS = 16
WINDOW_SIZE = 250

# Check for required files before proceeding
required_files = ['eeg_rf_model.pkl', 'eeg_xgb_model.pkl', 'eeg_scaler_tree.pkl', 'eeg_label_encoder.pkl']
for f in required_files:
    if not os.path.exists(f):
        print(f"Required file missing: {f}. Please ensure all models and encoders are present.")
        sys.exit(1)

# Load models, scaler, and label encoder
rf = joblib.load('eeg_rf_model.pkl')
xgb = joblib.load('eeg_xgb_model.pkl')
scaler = joblib.load('eeg_scaler_tree.pkl')  # Use the scaler used for tree models
le = joblib.load('eeg_label_encoder.pkl')

# Board parameters
params = BrainFlowInputParams()
params.serial_port = 'COM8'  # Change to your Cyton's COM port if needed
board = BoardShim(BoardIds.CYTON_DAISY_BOARD.value, params)
CHANNELS = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD.value)

def main():
    """Main function for real-time EEG prediction."""
    print("Preparing session...")
    try:
        board.prepare_session()
        board.start_stream()
    except Exception as e:
        print(f"Failed to start board session: {e}")
        sys.exit(1)

    print("Started EEG stream. Waiting for data to accumulate...")
    time.sleep(3)  # Wait for data to accumulate

    print("Select model for real-time prediction:")
    print("1: Random Forest\n2: XGBoost\n3: Both")
    model_choice = input("Enter choice (1/2/3): ").strip()
    use_rf = model_choice in ['1', '3']
    use_xgb = model_choice in ['2', '3']

    try:
        while True:
            # Get the latest WINDOW_SIZE samples for all channels
            data = board.get_current_board_data(WINDOW_SIZE)
            if data.shape[1] >= WINDOW_SIZE:
                eeg_window = data[CHANNELS, -WINDOW_SIZE:]  # shape: (N_CHANNELS, WINDOW_SIZE)
                eeg_window = eeg_window.T  # shape: (WINDOW_SIZE, N_CHANNELS)
                eeg_window_flat = eeg_window.flatten().reshape(1, -1)  # shape: (1, 4000)
                eeg_window_scaled = scaler.transform(eeg_window_flat)
                if use_rf:
                    pred_rf = rf.predict(eeg_window_scaled)
                    prob_rf = rf.predict_proba(eeg_window_scaled).max()
                    pred_label_rf = le.inverse_transform(pred_rf)[0]
                    print(f"Random Forest Prediction: {pred_label_rf} (confidence: {prob_rf:.2f})")
                if use_xgb:
                    pred_xgb = xgb.predict(eeg_window_scaled)
                    prob_xgb = xgb.predict_proba(eeg_window_scaled).max()
                    pred_label_xgb = le.inverse_transform(pred_xgb)[0]
                    print(f"XGBoost Prediction: {pred_label_xgb} (confidence: {prob_xgb:.2f})")
            else:
                print(f"Waiting for enough data... (current samples: {data.shape[1]})")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        board.stop_stream()
        board.release_session()
        print("Session closed.")

if __name__ == "__main__":
    main()
