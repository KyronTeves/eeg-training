"""
Train EEGNet (deep learning), Random Forest, and XGBoost models on windowed EEG data.

- Loads windowed EEG data and labels
- Encodes and balances classes
- Applies data augmentation
- Trains:
    - EEGNet (Keras deep learning model)
    - Random Forest (scikit-learn)
    - XGBoost (xgboost)
- Saves trained models, encoders, and scalers for downstream evaluation and prediction

Input: Windowed EEG data (.npy), windowed labels (.npy)
Output: Trained model files, encoders, scalers
"""

import logging
import joblib
import numpy as np

from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

from EEGModels import EEGNet
from utils import load_config, setup_logging, check_no_nan, check_labels_valid, extract_features

setup_logging()  # Set up consistent logging to file and console

config = load_config()

N_CHANNELS = config["N_CHANNELS"]
WINDOW_SIZE = config["WINDOW_SIZE"]
SAMPLING_RATE = config["SAMPLING_RATE"]

try:
    X_windows = np.load(config["WINDOWED_NPY"])
    y_windows = np.load(config["WINDOWED_LABELS_NPY"])
    logging.info(
        "Loaded windowed data shape: %s, Labels shape: %s",
        X_windows.shape,
        y_windows.shape,
    )
except FileNotFoundError:
    logging.error("Windowed data file not found.")
    raise
except (OSError, ValueError, KeyError) as e:
    logging.error("Failed to load windowed data: %s", e)
    raise

check_no_nan(X_windows, name="Windowed EEG data")  # Validate no NaNs in windowed EEG data
check_labels_valid(y_windows, name="Windowed labels")  # Validate windowed labels

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
X_train_scaled = scaler.transform(X_train.reshape(-1, N_CHANNELS)).reshape(
    X_train.shape
)
X_test_scaled = scaler.transform(X_test.reshape(-1, N_CHANNELS)).reshape(X_test.shape)

# Downsample majority class (neutral) to match minority classes
unique, counts = np.unique(np.argmax(y_train, axis=1), return_counts=True)
min_count = np.min(counts)
indices_per_class = [
    np.nonzero(np.argmax(y_train, axis=1) == i)[0] for i in range(len(unique))
]
downsampled_indices = np.concatenate(
    [np.random.choice(idxs, min_count, replace=False) for idxs in indices_per_class]
)
np.random.shuffle(downsampled_indices)
X_train_bal = X_train_scaled[downsampled_indices]
y_train_bal = y_train[downsampled_indices]


# Data augmentation: add noisy/drifted/artifacted copies to balanced training set
def augment_eeg_data(x, noise_std=0.01, drift_max=0.05, artifact_prob=0.05):
    """
    Augment EEG data by adding realistic noise patterns and artifacts.

    This function applies multiple augmentation techniques to simulate real-world
    EEG signal variations and improve model robustness:
    1. Gaussian noise addition to simulate electrical interference
    2. Baseline drift simulation using sine waves (common in EEG)
    3. Random artifacts by zeroing values (simulates movement artifacts)

    Args:
        X (np.ndarray): Input EEG data of shape (samples, time_points, channels).
        noise_std (float): Standard deviation of Gaussian noise. Default: 0.01.
        drift_max (float): Maximum amplitude of baseline drift. Default: 0.05.
        artifact_prob (float): Probability of introducing artifacts (0-1). Default: 0.05.

    Returns:
        np.ndarray: Augmented EEG data with same shape as input.

    Note:
        Augmentation helps prevent overfitting and improves generalization to new
        sessions/users by simulating realistic EEG signal variations.
    """
    x_aug = x.copy()
    # Add Gaussian noise
    x_aug += np.random.normal(0, noise_std, x_aug.shape)
    # Add baseline drift (slow sine wave)
    drift = np.sin(np.linspace(0, np.pi, x_aug.shape[1])) * drift_max
    x_aug += drift[None, :, None]
    # Randomly zero out some windows (simulate artifacts)
    mask = np.random.rand(*x_aug.shape) < artifact_prob
    x_aug[mask] = 0
    return x_aug


X_train_aug = augment_eeg_data(X_train_bal)
y_train_aug = y_train_bal.copy()
# Concatenate original and augmented data
X_train_final = np.concatenate([X_train_bal, X_train_aug], axis=0)
y_train_final = np.concatenate([y_train_bal, y_train_aug], axis=0)

# Compute class weights for the (now balanced) training set
labels_train = np.argmax(y_train_final, axis=1)
class_weights = compute_class_weight(
    "balanced", classes=np.unique(labels_train), y=labels_train
)
class_weight_dict = dict(enumerate(class_weights))

logging.info(
    "Class distribution after downsampling and augmentation: %s",
    np.bincount(labels_train),
)

# Training context information for debugging
logging.info("Training context: %d total samples, %d classes",
            len(X_train_final), len(np.unique(y_train_final)))
unique_labels, label_counts = np.unique(y_train_final, return_counts=True)
class_dist = dict(zip(unique_labels, label_counts))
logging.info("Detailed class distribution: %s", class_dist)

# Prepare for EEGNet
X_train_eegnet = np.expand_dims(X_train_final, -1)
X_test_eegnet = np.expand_dims(X_test_scaled, -1)
X_train_eegnet = np.transpose(X_train_eegnet, (0, 2, 1, 3))
X_test_eegnet = np.transpose(X_test_eegnet, (0, 2, 1, 3))

# Early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Build EEGNet model using official implementation
# --- Hyperparameter Tuning for EEGNet ---
# As per the EEGNet paper, kernLength should be about half the sampling rate.
kernLength = SAMPLING_RATE // 2
# Experimenting with more filters as per analysis
F1 = 16
D = 4
F2 = F1 * D  # As recommended in EEGNet docs

logging.info(
    "Building EEGNet with tuned hyperparameters: kernLength=%d, F1=%d, D=%d, F2=%d",
    kernLength,
    F1,
    D,
    F2,
)

model = EEGNet(
    nb_classes=y_cat.shape[1],
    Chans=N_CHANNELS,
    Samples=WINDOW_SIZE,
    kernLength=kernLength,
    F1=F1,
    D=D,
    F2=F2,
)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(
    X_train_eegnet,
    y_train_final,
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    class_weight=class_weight_dict,
    callbacks=[early_stopping]
)

# Evaluate EEGNet
_, acc = model.evaluate(X_test_eegnet, y_test)
logging.info("EEGNet Test accuracy: %.3f", acc)

# Print confusion matrix and classification report
y_pred = model.predict(X_test_eegnet)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)
logging.info(
    "EEGNet Confusion Matrix:\n%s", confusion_matrix(y_true_labels, y_pred_labels)
)
logging.info(
    "EEGNet Classification Report:\n%s",
    classification_report(y_true_labels, y_pred_labels, target_names=le.classes_),
)

# Save EEGNet model and label encoder
model.save(config["MODEL_CNN"])
joblib.dump(le, config["LABEL_ENCODER"])
joblib.dump(scaler, config["SCALER_CNN"])
np.save(config["LABEL_CLASSES_NPY"], le.classes_)

# --- Feature Extraction for Tree-based Models ---
logging.info("Extracting features for tree-based models...")
X_features = np.array([extract_features(window, SAMPLING_RATE) for window in X_windows])
logging.info("Feature extraction complete. Feature shape: %s", X_features.shape)

# Train/test split for tree-based models
X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(
    X_features, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Standardize features
scaler_tree = StandardScaler()
X_train_scaled_tree = scaler_tree.fit_transform(X_train_tree)
X_test_scaled_tree = scaler_tree.transform(X_test_tree)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled_tree, y_train_tree)
rf_pred = rf.predict(X_test_scaled_tree)
logging.info("Random Forest Results:")
logging.info("Confusion Matrix:\n%s", confusion_matrix(y_test_tree, rf_pred))
logging.info(
    "Classification Report:\n%s",
    classification_report(y_test_tree, rf_pred, target_names=le.classes_),
)

# XGBoost
xgb = XGBClassifier(
    n_estimators=100, random_state=42, use_label_encoder=False, eval_metric="mlogloss"
)
xgb.fit(X_train_scaled_tree, y_train_tree)
xgb_pred = xgb.predict(X_test_scaled_tree)
logging.info("XGBoost Results:")
logging.info("Confusion Matrix:\n%s", confusion_matrix(y_test_tree, xgb_pred))
logging.info(
    "Classification Report:\n%s",
    classification_report(y_test_tree, xgb_pred, target_names=le.classes_),
)

# Save tree-based models
joblib.dump(rf, config["MODEL_RF"])
joblib.dump(xgb, config["MODEL_XGB"])
joblib.dump(le, config["LABEL_ENCODER"])
joblib.dump(scaler_tree, config["SCALER_TREE"])
