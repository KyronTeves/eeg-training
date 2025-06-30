"""
train_eeg_model.py

Train EEGNet, ShallowConvNet, Random Forest, and XGBoost models on windowed EEG data.

Input: Windowed EEG data (.npy), windowed labels (.npy)
Process: Loads data, encodes and balances classes, applies augmentation, trains models, saves artifacts.
Output: Trained model files, encoders, scalers (for downstream evaluation and prediction)
"""

import logging
from typing import Any, Tuple

import joblib
import numpy as np
from joblib import Parallel, delayed
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from EEGModels import EEGNet, ShallowConvNet
from utils import (
    check_labels_valid,
    check_no_nan,
    extract_features,
    load_config,
    handle_errors,
    setup_logging,
)


def augment_eeg_data(
    x: np.ndarray,
    noise_std: float = 0.01,
    drift_max: float = 0.05,
    artifact_prob: float = 0.05,
) -> np.ndarray:
    """
    Augment EEG data with noise, drift, and simulated artifacts.

    Args:
        x (np.ndarray): Input EEG data, shape (n_windows, window_size, n_channels).
        noise_std (float): Standard deviation of Gaussian noise.
        drift_max (float): Maximum amplitude of baseline drift.
        artifact_prob (float): Probability of zeroing out a window.

    Returns:
        np.ndarray: Augmented EEG data.

    Input: Raw EEG window data.
    Process: Adds Gaussian noise, baseline drift, and randomly zeroes out windows.
    Output: Augmented EEG data array.
    """
    # Add Gaussian noise
    x_aug = x + np.random.randn(*x.shape) * noise_std
    # Add baseline drift (slow sine wave)
    drift = drift_max * np.sin(np.linspace(0, np.pi, x.shape[1]))
    x_aug += drift[None, :, None]
    # Randomly zero out some windows (simulate artifacts)
    mask = np.random.rand(x.shape[0]) < artifact_prob
    x_aug[mask] = 0
    return x_aug


def train_eegnet_model(
    _x_train: np.ndarray,
    _y_train: np.ndarray,
    _x_test: np.ndarray,
    _y_test: np.ndarray,
    _config: dict[str, Any],
    _le: LabelEncoder,
) -> None:
    """
    Train and evaluate EEGNet or ShallowConvNet model.

    Args:
        _X_train (np.ndarray): Training data, shape (n_samples, window, channels, 1).
        _y_train (np.ndarray): Training labels (one-hot or encoded).
        _X_test (np.ndarray): Test data, shape (n_samples, window, channels, 1).
        _y_test (np.ndarray): Test labels (one-hot or encoded).
        _config (dict): Configuration dictionary.
        _le (LabelEncoder): Label encoder.

    Returns:
        model: Trained Keras model.
        float: Test accuracy.

    Input: Preprocessed and windowed EEG data and labels.
    Process: Compiles, trains, and evaluates the model.
    Output: Trained model and test accuracy.
    """
    x_train_eegnet = np.expand_dims(_x_train, -1)
    x_test_eegnet = np.expand_dims(_x_test, -1)
    x_train_eegnet = np.transpose(x_train_eegnet, (0, 2, 1, 3))
    x_test_eegnet = np.transpose(x_test_eegnet, (0, 2, 1, 3))

    early_stopping = EarlyStopping(
        monitor=_config["EARLY_STOPPING_MONITOR"],
        patience=_config["EARLY_STOPPING_PATIENCE"],
        restore_best_weights=True,
    )
    kern_length = _config["EEGNET_KERN_LENGTH"]
    f1 = _config["EEGNET_F1"]
    d = _config["EEGNET_D"]
    f2 = _config["EEGNET_F2"]
    models_to_train = _config.get("MODELS_TO_TRAIN", ["EEGNet", "ShallowConvNet"])
    for model_name in models_to_train:
        logging.info("=== Training %s ===", model_name)
        if model_name == "EEGNet":
            model = EEGNet(
                nb_classes=_y_train.shape[1],
                Chans=_config["N_CHANNELS"],
                Samples=_config["WINDOW_SIZE"],
                kernLength=kern_length,
                F1=f1,
                D=d,
                F2=f2,
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
            x_train_eegnet,
            _y_train,
            epochs=_config["EPOCHS"],
            batch_size=_config["BATCH_SIZE"],
            validation_split=_config["VALIDATION_SPLIT"],
            class_weight=None,  # Set externally if needed
            callbacks=[early_stopping],
            verbose=1,
        )
        _, acc = model.evaluate(x_test_eegnet, _y_test)
        logging.info("%s Test accuracy: %.3f", model_name, acc)
        y_pred = model.predict(x_test_eegnet)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(_y_test, axis=1)
        logging.info(
            f"{model_name} Confusion Matrix:\n%s",
            confusion_matrix(y_true_labels, y_pred_labels),
        )
        logging.info(
            f"{model_name} Classification Report:\n%s",
            classification_report(
                y_true_labels, y_pred_labels, target_names=_le.classes_
            ),
        )
        model.save(model_path)
        logging.info("%s saved to %s", model_name, model_path)


def train_tree_models(
    _x_features: np.ndarray,
    _y_encoded: np.ndarray,
    _config: dict[str, Any],
    _le: LabelEncoder,
) -> None:
    """
    Train and evaluate Random Forest and XGBoost models.

    Args:
        _X_features (np.ndarray): Feature matrix for tree models.
        _y_encoded (np.ndarray): Encoded labels.
        _config (dict): Configuration dictionary.
        _le (LabelEncoder): Label encoder.

    Returns:
        tuple: (RandomForestClassifier, XGBClassifier, float, float)
            Trained RF and XGB models, RF accuracy, XGB accuracy.

    Input: Feature matrix and encoded labels.
    Process: Trains and evaluates Random Forest and XGBoost models.
    Output: Trained models and their test accuracies.
    """
    x_train_tree, x_test_tree, y_train_tree, y_test_tree = train_test_split(
        _x_features, _y_encoded, test_size=0.2, random_state=42, stratify=_y_encoded
    )
    scaler_tree = StandardScaler()
    x_train_scaled_tree = scaler_tree.fit_transform(x_train_tree)
    x_test_scaled_tree = scaler_tree.transform(x_test_tree)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(x_train_scaled_tree, y_train_tree)
    rf_pred = rf.predict(x_test_scaled_tree)
    logging.info("Random Forest Results:")
    logging.info("Confusion Matrix:\n%s", confusion_matrix(y_test_tree, rf_pred))
    logging.info(
        "Classification Report:\n%s",
        classification_report(y_test_tree, rf_pred, target_names=_le.classes_),
    )
    xgb = XGBClassifier(
        n_estimators=100,
        random_state=42,
        use_label_encoder=False,
        eval_metric="mlogloss",
    )
    xgb.fit(x_train_scaled_tree, y_train_tree)
    xgb_pred = xgb.predict(x_test_scaled_tree)
    logging.info("XGBoost Results:")
    logging.info("Confusion Matrix:\n%s", confusion_matrix(y_test_tree, xgb_pred))
    logging.info(
        "Classification Report:\n%s",
        classification_report(y_test_tree, xgb_pred, target_names=_le.classes_),
    )
    joblib.dump(rf, _config["MODEL_RF"])
    joblib.dump(xgb, _config["MODEL_XGB"])
    joblib.dump(scaler_tree, _config["SCALER_TREE"])


@handle_errors
def main() -> None:
    """
    Main function to orchestrate the training of EEGNet, ShallowConvNet, Random Forest, and XGBoost models
    on windowed EEG data. Handles data loading, preprocessing, augmentation, model training, and artifact saving.
    """
    setup_logging()
    config = load_config()
    X_windows, y_windows = load_windowed_data(config)
    check_no_nan(X_windows, name="Windowed EEG data")
    check_labels_valid(y_windows, name="Windowed labels")
    le, y_encoded, y_cat = encode_labels(y_windows)
    (
        X_train_final,
        y_train_final,
        X_test_scaled,
        y_test,
        scaler,
    ) = preprocess_and_augment(X_windows, y_cat, config)
    log_class_distribution(y_train_final)
    train_eegnet_model(X_train_final, y_train_final, X_test_scaled, y_test, config, le)
    joblib.dump(le, config["LABEL_ENCODER"])
    joblib.dump(scaler, config["SCALER_EEGNET"])
    np.save(config["LABEL_CLASSES_NPY"], le.classes_)
    logging.info("Extracting features for tree-based models...")
    X_features = extract_features_parallel(X_windows, config)
    logging.info("Feature extraction complete. Feature shape: %s", X_features.shape)
    train_tree_models(X_features, y_encoded, config, le)


def load_windowed_data(config: dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load windowed EEG data and corresponding labels from .npy files specified in the config.

    Args:
        config (dict): Configuration dictionary containing file paths for windowed EEG data and labels.

    Returns:
        tuple: (X_windows, y_windows) where X_windows is the EEG data and y_windows are the labels.

    Raises:
        FileNotFoundError: If the specified files are not found.
        OSError, ValueError, KeyError: If loading fails for other reasons.
    """
    try:
        X_windows = np.load(config["WINDOWED_NPY"])
        y_windows = np.load(config["WINDOWED_LABELS_NPY"])
        logging.info(
            "Loaded windowed data shape: %s, Labels shape: %s",
            X_windows.shape,
            y_windows.shape,
        )
        return X_windows, y_windows
    except FileNotFoundError:
        logging.error(
            "Windowed data file not found. Please ensure window_eeg_data.py has been run "
            "and the config paths are correct."
        )
        raise
    except (OSError, ValueError, KeyError) as e:
        logging.error("Failed to load windowed data: %s", e)
        raise


def encode_labels(y_windows: np.ndarray) -> Tuple[LabelEncoder, np.ndarray, np.ndarray]:
    """
    Encode string or categorical labels into integer and one-hot encoded formats.

    Args:
        y_windows (np.ndarray): Array of labels to encode.

    Returns:
        le (LabelEncoder): Fitted label encoder.
        y_encoded (np.ndarray): Integer-encoded labels.
        y_cat (np.ndarray): One-hot encoded labels.
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_windows)
    y_cat = to_categorical(y_encoded)
    return le, y_encoded, y_cat


def preprocess_and_augment(
    X_windows: np.ndarray, y_cat: np.ndarray, config: dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Split EEG data into train/test sets, scale, balance classes, augment, and return processed arrays.

    Args:
        X_windows (np.ndarray): Windowed EEG data.
        y_cat (np.ndarray): One-hot encoded labels.
        config (dict): Configuration dictionary.

    Returns:
        tuple: (X_train_final, y_train_final, X_test_scaled, y_test, scaler)
    """

    def balance_and_augment(X_train_scaled, y_train):
        """Balance classes by downsampling and augment the data."""
        labels = np.argmax(y_train, axis=1)
        unique, counts = np.unique(labels, return_counts=True)
        min_count = np.min(counts)
        indices_per_class = [np.nonzero(labels == i)[0] for i in range(len(unique))]
        downsampled_indices = np.concatenate(
            [
                np.random.choice(idxs, min_count, replace=False)
                for idxs in indices_per_class
            ]
        )
        np.random.shuffle(downsampled_indices)
        X_train_bal = X_train_scaled[downsampled_indices]
        y_train_bal = y_train[downsampled_indices]
        X_train_aug = augment_eeg_data(X_train_bal)
        y_train_aug = y_train_bal.copy()
        X_train_final = np.concatenate([X_train_bal, X_train_aug], axis=0)
        y_train_final = np.concatenate([y_train_bal, y_train_aug], axis=0)
        return X_train_final, y_train_final

    X_train, X_test, y_train, y_test = train_test_split(
        X_windows, y_cat, test_size=0.2, random_state=42, stratify=y_cat
    )
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, config["N_CHANNELS"])
    scaler.fit(X_train_flat)
    X_train_scaled = scaler.transform(
        X_train.reshape(-1, config["N_CHANNELS"])
    ).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, config["N_CHANNELS"])).reshape(
        X_test.shape
    )
    X_train_final, y_train_final = balance_and_augment(X_train_scaled, y_train)
    return X_train_final, y_train_final, X_test_scaled, y_test, scaler


def log_class_distribution(y_train_final: np.ndarray) -> None:
    """
    Log the class distribution of the training data after downsampling and augmentation.

    Args:
        y_train_final (np.ndarray): One-hot encoded or categorical labels for the training set.

    Logs the class counts and distribution for debugging and monitoring purposes.
    """
    labels_train = np.argmax(y_train_final, axis=1)
    logging.info(
        "Class distribution after downsampling and augmentation: %s",
        np.bincount(labels_train),
    )
    logging.info(
        "Training context: %d total samples, %d classes",
        len(y_train_final),
        len(np.unique(y_train_final)),
    )
    unique_labels, label_counts = np.unique(y_train_final, return_counts=True)
    class_dist = dict(zip(unique_labels, label_counts))
    logging.info("Detailed class distribution: %s", class_dist)


def extract_features_parallel(X_windows: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """
    Extract features from each EEG window in parallel using joblib.

    Args:
        X_windows (np.ndarray): Array of windowed EEG data.
        config (dict): Configuration dictionary containing 'SAMPLING_RATE'.

    Returns:
        np.ndarray: Array of extracted feature vectors for each window.
    """
    return np.array(
        Parallel(n_jobs=-1, prefer="threads")(
            delayed(extract_features)(window, config["SAMPLING_RATE"])
            for window in X_windows
        )
    )


if __name__ == "__main__":
    main()
