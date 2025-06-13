import numpy as np
import pandas as pd

# Settings
N_CHANNELS = 16  # Number of EEG channels
WINDOW_SIZE = 250  # Number of timepoints per window (e.g., 1 second at 250Hz)
STEP_SIZE = 125    # Overlap windows by 50% (optional)
RAW_CSV = 'eeg_training_data.csv'  # Input raw data file
WINDOWED_NPY = 'eeg_windowed_X.npy'  # Output windowed data (features)
WINDOWED_LABELS_NPY = 'eeg_windowed_y.npy'  # Output windowed data (labels)
SESSION_TYPES = ['pure', 'jolt', 'hybrid', 'long']  # All possible session types
USE_SESSION_TYPES = ['pure', 'jolt', 'hybrid', 'long']  # Change as needed

# Load raw data
raw_data = pd.read_csv(RAW_CSV)

# Filter by session_type if present
if 'session_type' in raw_data.columns:
    print(f"Available session types: {raw_data['session_type'].unique()}")
    raw_data = raw_data[raw_data['session_type'].isin(USE_SESSION_TYPES)]
    print(f"Using session types: {USE_SESSION_TYPES}, samples: {len(raw_data)}")

# Use only EEG channel columns (ch_*) for features
eeg_cols = [col for col in raw_data.columns if col.startswith('ch_')]
X = raw_data[eeg_cols].values
labels = raw_data['label'].values

# Reshape X to [n_samples, n_channels]
if X.shape[1] != N_CHANNELS:
    raise ValueError(f"Expected {N_CHANNELS} channels, but got {X.shape[1]} columns per sample.")
X = X.reshape(-1, N_CHANNELS)
labels = labels.reshape(-1, 1)

# Windowing
X_windows = []
y_windows = []
for start in range(0, len(X) - WINDOW_SIZE + 1, STEP_SIZE):
    window = X[start:start+WINDOW_SIZE]
    window_labels = labels[start:start+WINDOW_SIZE]
    # Use the most frequent label in the window as the label
    unique, counts = np.unique(window_labels, return_counts=True)
    window_label = unique[np.argmax(counts)]
    X_windows.append(window)
    y_windows.append(window_label)
X_windows = np.array(X_windows)  # shape: [num_windows, WINDOW_SIZE, N_CHANNELS]
y_windows = np.array(y_windows).flatten()

print(f"Windowed data shape: {X_windows.shape}, Labels shape: {y_windows.shape}")

# Save as .npy for fast loading in training
np.save(WINDOWED_NPY, X_windows)
np.save(WINDOWED_LABELS_NPY, y_windows)
print(f"Saved windowed data to {WINDOWED_NPY} and {WINDOWED_LABELS_NPY}")
