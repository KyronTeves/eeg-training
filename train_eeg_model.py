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
from joblib import Parallel, delayed

from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

from EEGModels import EEGNet, ShallowConvNet
from utils import (
    load_config,
    setup_logging,
    check_no_nan,
    check_labels_valid,
    extract_features,
)

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
    logging.error("Windowed data file not found. Please ensure window_eeg_data.py has been run and the config paths are correct.")
    raise
except (OSError, ValueError, KeyError) as e:
    logging.error("Failed to load windowed data: %s", e)
    raise

# All validation and windowing is handled by utility functions in utils.py
check_no_nan(
    X_windows, name="Windowed EEG data"
)  # Validate no NaNs in windowed EEG data
check_labels_valid(y_windows, name="Windowed labels")  # Validate windowed labels
# (Reminder: Any future validation/windowing logic should use utils.py)

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


def train_eegnet_model(_X_train, _y_train, _X_test, _y_test, _config, _le):
    """Train and evaluate EEGNet or ShallowConvNet."""
    X_train_eegnet = np.expand_dims(_X_train, -1)
    X_test_eegnet = np.expand_dims(_X_test, -1)
    X_train_eegnet = np.transpose(X_train_eegnet, (0, 2, 1, 3))
    X_test_eegnet = np.transpose(X_test_eegnet, (0, 2, 1, 3))

    early_stopping = EarlyStopping(
        monitor=_config["EARLY_STOPPING_MONITOR"],
        patience=_config["EARLY_STOPPING_PATIENCE"],
        restore_best_weights=True,
    )
    kernLength = _config["EEGNET_KERN_LENGTH"]
    F1 = _config["EEGNET_F1"]
    D = _config["EEGNET_D"]
    F2 = _config["EEGNET_F2"]
    models_to_train = _config.get("MODELS_TO_TRAIN", ["EEGNet", "ShallowConvNet"])
    for model_name in models_to_train:
        logging.info("=== Training %s ===", model_name)
        if model_name == "EEGNet":
            model = EEGNet(
                nb_classes=_y_train.shape[1],
                Chans=_config["N_CHANNELS"],
                Samples=_config["WINDOW_SIZE"],
                kernLength=kernLength,
                F1=F1,
                D=D,
                F2=F2,
                dropoutRate=_config["EEGNET_DROPOUT_RATE"],
                dropoutType=_config["EEGNET_DROPOUT_TYPE"],
                norm_rate=_config["EEGNET_NORM_RATE"],
            )
            model_path = _config["MODEL_EEGNET"]
        elif model_name == "ShallowConvNet":
            model = ShallowConvNet(
                nb_classes=_y_train.shape[1],
                Chans=_config["N_CHANNELS"],
                Samples=_config["WINDOW_SIZE"],
                dropoutRate=_config["EEGNET_DROPOUT_RATE"],
            )
            model_path = _config["MODEL_SHALLOW"]
        else:
            logging.warning("Unknown model: %s. Skipping.", model_name)
            continue
        model.compile(
            optimizer=_config["OPTIMIZER"],
            loss=_config["LOSS_FUNCTION"],
            metrics=["accuracy"],
        )
        model.fit(
            X_train_eegnet,
            _y_train,
            epochs=_config["EPOCHS"],
            batch_size=_config["BATCH_SIZE"],
            validation_split=_config["VALIDATION_SPLIT"],
            class_weight=None,  # Set externally if needed
            callbacks=[early_stopping],
            verbose=1,
        )
        _, acc = model.evaluate(X_test_eegnet, _y_test)
        logging.info("%s Test accuracy: %.3f", model_name, acc)
        y_pred = model.predict(X_test_eegnet)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(_y_test, axis=1)
        logging.info(
            f"{model_name} Confusion Matrix:\n%s",
            confusion_matrix(y_true_labels, y_pred_labels),
        )
        logging.info(
            f"{model_name} Classification Report:\n%s",
            classification_report(y_true_labels, y_pred_labels, target_names=_le.classes_),
        )
        model.save(model_path)
        logging.info("%s saved to %s", model_name, model_path)


def train_tree_models(_X_features, _y_encoded, _config, _le):
    """Train and evaluate Random Forest and XGBoost models."""
    X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(
        _X_features, _y_encoded, test_size=0.2, random_state=42, stratify=_y_encoded
    )
    scaler_tree = StandardScaler()
    X_train_scaled_tree = scaler_tree.fit_transform(X_train_tree)
    X_test_scaled_tree = scaler_tree.transform(X_test_tree)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled_tree, y_train_tree)
    rf_pred = rf.predict(X_test_scaled_tree)
    logging.info("Random Forest Results:")
    logging.info("Confusion Matrix:\n%s", confusion_matrix(y_test_tree, rf_pred))
    logging.info(
        "Classification Report:\n%s",
        classification_report(y_test_tree, rf_pred, target_names=_le.classes_),
    )
    xgb = XGBClassifier(
        n_estimators=100, random_state=42, use_label_encoder=False, eval_metric="mlogloss"
    )
    xgb.fit(X_train_scaled_tree, y_train_tree)
    xgb_pred = xgb.predict(X_test_scaled_tree)
    logging.info("XGBoost Results:")
    logging.info("Confusion Matrix:\n%s", confusion_matrix(y_test_tree, xgb_pred))
    logging.info(
        "Classification Report:\n%s",
        classification_report(y_test_tree, xgb_pred, target_names=_le.classes_),
    )
    joblib.dump(rf, _config["MODEL_RF"])
    joblib.dump(xgb, _config["MODEL_XGB"])
    joblib.dump(scaler_tree, _config["SCALER_TREE"])


# Data augmentation: add noisy/drifted/artifacted copies to balanced training set
X_train_aug = augment_eeg_data(X_train_bal)
y_train_aug = y_train_bal.copy()
X_train_final = np.concatenate([X_train_bal, X_train_aug], axis=0)
y_train_final = np.concatenate([y_train_bal, y_train_aug], axis=0)

labels_train = np.argmax(y_train_final, axis=1)
class_weights = compute_class_weight(
    "balanced", classes=np.unique(labels_train), y=labels_train
)
class_weight_dict = dict(enumerate(class_weights))

logging.info(
    "Class distribution after downsampling and augmentation: %s",
    np.bincount(labels_train),
)
logging.info(
    "Training context: %d total samples, %d classes",
    len(X_train_final),
    len(np.unique(y_train_final)),
)
unique_labels, label_counts = np.unique(y_train_final, return_counts=True)
class_dist = dict(zip(unique_labels, label_counts))
logging.info("Detailed class distribution: %s", class_dist)

# Train deep learning models
train_eegnet_model(X_train_final, y_train_final, X_test_scaled, y_test, config, le)

# Save shared components
joblib.dump(le, config["LABEL_ENCODER"])
joblib.dump(scaler, config["SCALER_EEGNET"])
np.save(config["LABEL_CLASSES_NPY"], le.classes_)

# --- Feature Extraction for Tree-based Models ---
logging.info("Extracting features for tree-based models...")
# Parallel feature extraction for speed
X_features = np.array(
    Parallel(n_jobs=-1, prefer="threads")(
        delayed(extract_features)(window, SAMPLING_RATE) for window in X_windows
    )
)
logging.info("Feature extraction complete. Feature shape: %s", X_features.shape)

# Train tree-based models
train_tree_models(X_features, y_encoded, config, le)
