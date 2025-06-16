import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model # type: ignore
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from utils import load_config, window_data

config = load_config()

N_CHANNELS = config["N_CHANNELS"]
WINDOW_SIZE = config["WINDOW_SIZE"]
STEP_SIZE = config["STEP_SIZE"]
CSV_FILE = config["OUTPUT_CSV"]
TEST_SESSION_TYPES = config["TEST_SESSION_TYPES"]
NUM_TEST_SAMPLES = config["NUM_TEST_SAMPLES"]

print(f"Loading data from {CSV_FILE} ...")
df = pd.read_csv(CSV_FILE)
test_df = df[df['session_type'].isin(TEST_SESSION_TYPES)]
print(f"Test samples: {len(test_df)}")

# Use the utility function for windowing
eeg_cols = [col for col in test_df.columns if col.startswith('ch_')]
X = test_df[eeg_cols].values
labels = test_df['label'].values
X = X.reshape(-1, N_CHANNELS)
labels = labels.reshape(-1, 1)
X_windows, y_windows = window_data(X, labels, WINDOW_SIZE, STEP_SIZE)
print(f"Test windows: {X_windows.shape}")

le = joblib.load(config["LABEL_ENCODER"])
scaler = joblib.load(config["SCALER_CNN"])
model = load_model(config["MODEL_CNN"])
rf = joblib.load(config["MODEL_RF"])
xgb = joblib.load(config["MODEL_XGB"])

X_windows_flat = X_windows.reshape(-1, N_CHANNELS)
X_windows_scaled = scaler.transform(X_windows_flat).reshape(X_windows.shape)
X_windows_flat_scaled = X_windows_scaled.reshape(X_windows.shape[0], -1)

num_samples = min(NUM_TEST_SAMPLES, X_windows.shape[0])
indices = np.random.choice(X_windows.shape[0], num_samples, replace=False)

for idx in indices:
    # Conv1D model
    sample_cnn = X_windows_scaled[idx].reshape(1, WINDOW_SIZE, N_CHANNELS)
    actual_label = y_windows[idx]
    pred_cnn = model.predict(sample_cnn)
    pred_label_cnn = le.inverse_transform([np.argmax(pred_cnn)])[0]
    # Random Forest
    sample_rf = X_windows_flat_scaled[idx].reshape(1, -1)
    pred_rf = rf.predict(sample_rf)
    pred_label_rf = le.inverse_transform(pred_rf)[0]
    # XGBoost
    pred_xgb = xgb.predict(sample_rf)
    pred_label_xgb = le.inverse_transform(pred_xgb)[0]
    print(f"Actual label:   {actual_label}")
    print(f"Conv1D Predicted label: {pred_label_cnn}")
    print(f"Random Forest Predicted label: {pred_label_rf}")
    print(f"XGBoost Predicted label: {pred_label_xgb}")
    print("-")
