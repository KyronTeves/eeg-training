"""
Test trained EEG models (Conv1D, Random Forest, XGBoost) on windowed test data.
"""
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model # type: ignore
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Parameters (must match training)
N_CHANNELS = 16
WINDOW_SIZE = 250
STEP_SIZE = 125
CSV_FILE = 'eeg_training_data.csv'
TEST_SESSION_TYPES = ['jolt', 'hybrid', 'long']  # Change as needed

def window_data(df: pd.DataFrame, window_size: int, step_size: int, n_channels: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Window the data for model testing. Returns (X, y).
    """
    X = []
    y = []
    data = df[[col for col in df.columns if col.startswith('ch_')]].values
    labels = df['label'].values
    for start in range(0, len(df) - window_size + 1, step_size):
        window = data[start:start+window_size]
        if window.shape[0] != window_size:
            continue
        window = window.reshape(window_size, n_channels)
        window_labels = labels[start:start+window_size]
        label = pd.Series(window_labels).mode()[0]
        X.append(window)
        y.append(label)
    return np.array(X), np.array(y)

def main():
    """Main function to test EEG models on windowed data."""
    print(f"Loading data from {CSV_FILE} ...")
    df = pd.read_csv(CSV_FILE)
    test_df = df[df['session_type'].isin(TEST_SESSION_TYPES)]
    print(f"Test samples: {len(test_df)}")

    print("Windowing test data ...")
    X_windows, y_windows = window_data(test_df, WINDOW_SIZE, STEP_SIZE, N_CHANNELS)
    print(f"Test windows: {X_windows.shape}")

    le = joblib.load('eeg_label_encoder.pkl')
    scaler = joblib.load('eeg_scaler.pkl')
    model = load_model('eeg_direction_model.h5')
    rf = joblib.load('eeg_rf_model.pkl')
    xgb = joblib.load('eeg_xgb_model.pkl')

    # Standardize features (fit on training data, so use scaler)
    X_windows_flat = X_windows.reshape(-1, N_CHANNELS)
    X_windows_scaled = scaler.transform(X_windows_flat).reshape(X_windows.shape)
    X_windows_flat_scaled = X_windows_scaled.reshape(X_windows.shape[0], -1)

    # Pick 10 random windows from the dataset
    num_samples = min(10, X_windows.shape[0])
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

if __name__ == "__main__":
    main()
