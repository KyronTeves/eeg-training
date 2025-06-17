"""
Train EEGNet, Random Forest, and XGBoost models on windowed EEG data.

- Loads windowed data from .npy files.
- Encodes labels and splits data into train/test sets.
- Standardizes features and trains models.
- Saves trained models, encoders, and scalers.
- Uses logging for status and error messages.
"""

import logging

import joblib
import numpy as np
import pandas as pd
from keras.utils import to_categorical  # type: ignore
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

from EEGModels import EEGNet
from utils import load_config

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

try:
    X_windows = np.load(config["WINDOWED_NPY"])
    y_windows = np.load(config["WINDOWED_LABELS_NPY"])
    logging.info("Loaded windowed data shape: %s, Labels shape: %s", X_windows.shape, y_windows.shape)
except FileNotFoundError:
    logging.error("Windowed data file not found.")
    raise
except (OSError, ValueError, KeyError) as e:
    logging.error("Failed to load windowed data: %s", e)
    raise

# Data validation checks
if np.isnan(X_windows).any():
    logging.error("Windowed EEG data contains NaN values.")
    raise ValueError("Windowed EEG data contains NaN values.")
if pd.isnull(y_windows).any():
    logging.error("Windowed labels contain NaN values.")
    raise ValueError("Windowed labels contain NaN values.")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y_windows)
y_cat = to_categorical(y_encoded)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_windows, y_cat, test_size=0.2, random_state=42, stratify=y_cat
)

# Standardize features (fit only on training data)
scaler = StandardScaler()
X_train_flat = X_train.reshape(-1, N_CHANNELS)
scaler.fit(X_train_flat)
X_train_scaled = scaler.transform(X_train.reshape(-1, N_CHANNELS)).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test.reshape(-1, N_CHANNELS)).reshape(X_test.shape)

# Downsample majority class (neutral) to match minority classes
unique, counts = np.unique(np.argmax(y_train, axis=1), return_counts=True)
min_count = np.min(counts)
indices_per_class = [np.nonzero(np.argmax(y_train, axis=1) == i)[0] for i in range(len(unique))]
downsampled_indices = np.concatenate([np.random.choice(idxs, min_count, replace=False) for idxs in indices_per_class])
np.random.shuffle(downsampled_indices)
X_train_bal = X_train_scaled[downsampled_indices]
y_train_bal = y_train[downsampled_indices]

# Data augmentation: add noisy/drifted/artifacted copies to balanced training set
def augment_eeg_data(X, noise_std=0.01, drift_max=0.05, artifact_prob=0.05):
    X_aug = X.copy()
    # Add Gaussian noise
    X_aug += np.random.normal(0, noise_std, X_aug.shape)
    # Add baseline drift (slow sine wave)
    drift = np.sin(np.linspace(0, np.pi, X_aug.shape[1])) * drift_max
    X_aug += drift[None, :, None]
    # Randomly zero out some windows (simulate artifacts)
    mask = np.random.rand(*X_aug.shape) < artifact_prob
    X_aug[mask] = 0
    return X_aug

X_train_aug = augment_eeg_data(X_train_bal)
y_train_aug = y_train_bal.copy()
# Concatenate original and augmented data
X_train_final = np.concatenate([X_train_bal, X_train_aug], axis=0)
y_train_final = np.concatenate([y_train_bal, y_train_aug], axis=0)

# Compute class weights for the (now balanced) training set
labels_train = np.argmax(y_train_final, axis=1)
class_weights = compute_class_weight('balanced', classes=np.unique(labels_train), y=labels_train)
class_weight_dict = dict(enumerate(class_weights))

logging.info("Class distribution after downsampling and augmentation: %s", np.bincount(labels_train))

# Prepare for EEGNet
X_train_eegnet = np.expand_dims(X_train_final, -1)
X_test_eegnet = np.expand_dims(X_test_scaled, -1)
X_train_eegnet = np.transpose(X_train_eegnet, (0, 2, 1, 3))
X_test_eegnet = np.transpose(X_test_eegnet, (0, 2, 1, 3))

# Build EEGNet model using official implementation
model = EEGNet(nb_classes=y_cat.shape[1], Chans=N_CHANNELS, Samples=WINDOW_SIZE)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_eegnet, y_train_final, epochs=30, batch_size=64, validation_split=0.2, class_weight=class_weight_dict)

# Evaluate EEGNet
_, acc = model.evaluate(X_test_eegnet, y_test)
logging.info("EEGNet Test accuracy: %.3f", acc)

# Print confusion matrix and classification report
y_pred = model.predict(X_test_eegnet)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)
logging.info("EEGNet Confusion Matrix:\n%s", confusion_matrix(y_true_labels, y_pred_labels))
logging.info("EEGNet Classification Report:\n%s", classification_report(y_true_labels, y_pred_labels, target_names=le.classes_))

# Save EEGNet model and label encoder
model.save(config["MODEL_CNN"])
joblib.dump(le, config["LABEL_ENCODER"])
joblib.dump(scaler, config["SCALER_CNN"])
np.save(config["LABEL_CLASSES_NPY"], le.classes_)

# Flatten windows for tree-based models
X_flat = X_windows.reshape(X_windows.shape[0], -1)

# Train/test split for tree-based models
X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(
    X_flat, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Standardize features (optional for trees, but keep for consistency)
scaler_tree = StandardScaler()
X_train_scaled_tree = scaler_tree.fit_transform(X_train_tree)
X_test_scaled_tree = scaler_tree.transform(X_test_tree)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled_tree, y_train_tree)
rf_pred = rf.predict(X_test_scaled_tree)
logging.info("Random Forest Results:")
logging.info("Confusion Matrix:\n%s", confusion_matrix(y_test_tree, rf_pred))
logging.info("Classification Report:\n%s", classification_report(y_test_tree, rf_pred, target_names=le.classes_))

# XGBoost
xgb = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
xgb.fit(X_train_scaled_tree, y_train_tree)
xgb_pred = xgb.predict(X_test_scaled_tree)
logging.info("XGBoost Results:")
logging.info("Confusion Matrix:\n%s", confusion_matrix(y_test_tree, xgb_pred))
logging.info("Classification Report:\n%s", classification_report(y_test_tree, xgb_pred, target_names=le.classes_))

# Save tree-based models
joblib.dump(rf, config["MODEL_RF"])
joblib.dump(xgb, config["MODEL_XGB"])
joblib.dump(le, config["LABEL_ENCODER"])
joblib.dump(scaler_tree, config["SCALER_TREE"])
