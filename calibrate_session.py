"""
calibrate_session.py

Fine-tune a pre-trained EEGNet model and scaler for a new session 
using a small labeled calibration set.
- Loads calibration data (windowed, preprocessed, labeled)
- Fits a new scaler on calibration data
- Fine-tunes the pre-trained model for a few epochs
- Saves the session-specific model and scaler

Usage:
    python calibrate_session.py --calib_X data/calib_X.npy --calib_y data/calib_y.npy \
        --base_model models/eeg_direction_model.h5 --base_scaler models/eeg_scaler.pkl \
        --label_encoder models/eeg_label_encoder.pkl --out_model models/eeg_direction_model_session.h5 \
        --out_scaler models/eeg_scaler_session.pkl
"""

import argparse

import joblib
import numpy as np
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser(description="Session calibration for EEGNet.")
parser.add_argument("--calib_X", required=True, help="Path to calibration X (npy)")
parser.add_argument("--calib_y", required=True, help="Path to calibration y (npy)")
parser.add_argument(
    "--base_model", required=True, help="Path to pre-trained model (h5)"
)
parser.add_argument(
    "--base_scaler", required=True, help="Path to pre-trained scaler (pkl)"
)
parser.add_argument(
    "--label_encoder", required=True, help="Path to label encoder (pkl)"
)
parser.add_argument(
    "--out_model", required=True, help="Path to save session model (h5)"
)
parser.add_argument(
    "--out_scaler", required=True, help="Path to save session scaler (pkl)"
)
parser.add_argument(
    "--epochs", type=int, default=3, help="Fine-tuning epochs (default: 3)"
)
parser.add_argument(
    "--batch_size", type=int, default=16, help="Batch size (default: 16)"
)
args = parser.parse_args()

# 1. Load calibration data
X_calib = np.load(args.calib_X)  # (samples, window, channels)
y_calib = np.load(args.calib_y)  # (samples,)

# 2. Load label encoder and encode labels
le = joblib.load(args.label_encoder)
y_calib_encoded = le.transform(y_calib)
y_calib_cat = to_categorical(y_calib_encoded)

# 3. Fit a new scaler on calibration data
scaler = StandardScaler()
X_calib_flat = X_calib.reshape(-1, X_calib.shape[-1])
scaler.fit(X_calib_flat)
X_calib_scaled = scaler.transform(X_calib_flat).reshape(X_calib.shape)

# 4. Prepare for EEGNet: (batch, window, channels) -> (batch, channels, window, 1)
X_calib_eegnet = np.expand_dims(X_calib_scaled, -1)
X_calib_eegnet = np.transpose(X_calib_eegnet, (0, 2, 1, 3))

# 5. Load pre-trained model
model = load_model(args.base_model)

# 6. Fine-tune model on calibration data
model.fit(
    X_calib_eegnet,
    y_calib_cat,
    epochs=args.epochs,
    batch_size=args.batch_size,
    verbose=1,
)

# 7. Save the updated model and scaler for this session
model.save(args.out_model)
joblib.dump(scaler, args.out_scaler)

print(
    f"Session calibration complete. Model saved to {args.out_model}, scaler saved to {args.out_scaler}."
)
