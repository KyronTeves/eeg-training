"""train_eeg_model.py.

Train EEGNet, ShallowConvNet, Random Forest, and XGBoost models on windowed EEG data.

Handles data loading, preprocessing, augmentation, model training, and artifact saving for EEG classification.

Typical usage:
    $ python train_eeg_model.py
"""

import logging
from typing import Any

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
    handle_errors,
    load_config,
    setup_logging,
)

setup_logging()
logger = logging.getLogger(__name__)


def augment_eeg_data(
    x: np.ndarray,
    noise_std: float = 0.01,
    drift_max: float = 0.05,
    artifact_prob: float = 0.05,
) -> np.ndarray:
    """Augment EEG data with noise, drift, and simulated artifacts.

    Args:
        x (np.ndarray): Input EEG data, shape (n_windows, window_size, n_channels).
        noise_std (float): Standard deviation of Gaussian noise.
        drift_max (float): Maximum amplitude of baseline drift.
        artifact_prob (float): Probability of zeroing out a window.

    Returns:
        np.ndarray: Augmented EEG data.

    """
    # Add Gaussian noise
    rng = np.random.default_rng()
    x_aug = x + rng.standard_normal(x.shape) * noise_std
    # Add baseline drift (slow sine wave)
    drift = drift_max * np.sin(np.linspace(0, np.pi, x.shape[1]))
    x_aug += drift[None, :, None]
    # Randomly zero out some windows (simulate artifacts)
    mask = rng.random(x.shape[0]) < artifact_prob
    x_aug[mask] = 0
    return x_aug


def train_eegnet_model(  # noqa: PLR0913
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    config: dict[str, Any],
    label_encoder: LabelEncoder,
) -> None:
    """Train and evaluate EEGNet or ShallowConvNet model.

    Args:
        x_train (np.ndarray): Training data, shape (n_samples, window, channels, 1).
        y_train (np.ndarray): Training labels (one-hot or encoded).
        x_test (np.ndarray): Test data, shape (n_samples, window, channels, 1).
        y_test (np.ndarray): Test labels (one-hot or encoded).
        config (dict[str, Any]): Configuration dictionary.
        label_encoder (LabelEncoder): Label encoder.

    """
    x_train_eegnet = np.expand_dims(x_train, -1)
    x_test_eegnet = np.expand_dims(x_test, -1)
    x_train_eegnet = np.transpose(x_train_eegnet, (0, 2, 1, 3))
    x_test_eegnet = np.transpose(x_test_eegnet, (0, 2, 1, 3))
    early_stopping = EarlyStopping(
        monitor=config["EARLY_STOPPING_MONITOR"],
        patience=config["EARLY_STOPPING_PATIENCE"],
        restore_best_weights=True,
    )
    kern_length = config["EEGNET_KERN_LENGTH"]
    f1 = config["EEGNET_F1"]
    d = config["EEGNET_D"]
    f2 = config["EEGNET_F2"]
    models_to_train = config.get("MODELS_TO_TRAIN", ["EEGNet", "ShallowConvNet"])
    for model_name in models_to_train:
        logger.info("=== Training %s ===", model_name)
        if model_name == "EEGNet":
            model = EEGNet(
                nb_classes=y_train.shape[1],
                Chans=config["N_CHANNELS"],
                Samples=config["WINDOW_SIZE"],
                kernLength=kern_length,
                F1=f1,
                D=d,
                F2=f2,
                dropoutRate=config["EEGNET_DROPOUT_RATE"],
                dropoutType=config["EEGNET_DROPOUT_TYPE"],
                norm_rate=config["EEGNET_NORM_RATE"],
            )
            model_path = config["MODEL_EEGNET"]
        elif model_name == "ShallowConvNet":
            model = ShallowConvNet(
                nb_classes=y_train.shape[1],
                Chans=config["N_CHANNELS"],
                Samples=config["WINDOW_SIZE"],
                dropoutRate=config["EEGNET_DROPOUT_RATE"],
            )
            model_path = config["MODEL_SHALLOW"]
        else:
            logger.warning("Unknown model: %s. Skipping.", model_name)
            continue
        model.compile(
            optimizer=config["OPTIMIZER"],
            loss=config["LOSS_FUNCTION"],
            metrics=["accuracy"],
        )
        model.fit(
            x_train_eegnet,
            y_train,
            epochs=config["EPOCHS"],
            batch_size=config["BATCH_SIZE"],
            validation_split=config["VALIDATION_SPLIT"],
            class_weight=None,  # Set externally if needed
            callbacks=[early_stopping],
            verbose=1,
        )
        _, acc = model.evaluate(x_test_eegnet, y_test)
        logger.info("%s Test accuracy: %.3f", model_name, acc)
        y_pred = model.predict(x_test_eegnet)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y_test, axis=1)
        logger.info(
            "%s Confusion Matrix:\n%s",
            model_name,
            confusion_matrix(y_true_labels, y_pred_labels),
        )
        logger.info(
            "%s Classification Report:\n%s",
            model_name,
            classification_report(
                y_true_labels, y_pred_labels, target_names=label_encoder.classes_,
            ),
        )
        model.save(model_path)
        logger.info("%s saved to %s", model_name, model_path)


def train_tree_models(
    x_features: np.ndarray,
    y_encoded: np.ndarray,
    config: dict[str, Any],
    label_encoder: LabelEncoder,
) -> None:
    """Train and evaluate Random Forest and XGBoost models.

    Args:
        x_features (np.ndarray): Feature matrix for tree models.
        y_encoded (np.ndarray): Encoded labels.
        config (dict[str, Any]): Configuration dictionary.
        label_encoder (LabelEncoder): Label encoder.

    """
    x_train_tree, x_test_tree, y_train_tree, y_test_tree = train_test_split(
        x_features, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded,
    )
    scaler_tree = StandardScaler()
    x_train_scaled_tree = scaler_tree.fit_transform(x_train_tree)
    x_test_scaled_tree = scaler_tree.transform(x_test_tree)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(x_train_scaled_tree, y_train_tree)
    rf_pred = rf.predict(x_test_scaled_tree)
    logger.info("Random Forest Results:")
    logger.info("Confusion Matrix:\n%s", confusion_matrix(y_test_tree, rf_pred))
    logger.info(
        "Classification Report:\n%s",
        classification_report(
            y_test_tree, rf_pred, target_names=label_encoder.classes_,
        ),
    )
    xgb = XGBClassifier(
        n_estimators=100,
        random_state=42,
        use_label_encoder=False,
        eval_metric="mlogloss",
    )
    xgb.fit(x_train_scaled_tree, y_train_tree)
    xgb_pred = xgb.predict(x_test_scaled_tree)
    logger.info("XGBoost Results:")
    logger.info("Confusion Matrix:\n%s", confusion_matrix(y_test_tree, xgb_pred))
    logger.info(
        "Classification Report:\n%s",
        classification_report(
            y_test_tree, xgb_pred, target_names=label_encoder.classes_,
        ),
    )
    joblib.dump(rf, config["MODEL_RF"])
    joblib.dump(xgb, config["MODEL_XGB"])
    joblib.dump(scaler_tree, config["SCALER_TREE"])


@handle_errors
def main() -> None:
    """Orchestrate the training of EEGNet, ShallowConvNet, Random Forest, and XGBoost models on windowed EEG data.

    Handle data loading, preprocessing, augmentation, model training, and artifact saving.
    """
    setup_logging()
    config = load_config()
    x_windows, y_windows = load_windowed_data(config)
    check_no_nan(x_windows, name="Windowed EEG data")
    check_labels_valid(y_windows, name="Windowed labels")
    le, y_encoded, y_cat = encode_labels(y_windows)
    (
        x_train_final,
        y_train_final,
        x_test_scaled,
        y_test,
        scaler,
    ) = preprocess_and_augment(x_windows, y_cat, config)
    log_class_distribution(y_train_final)
    train_eegnet_model(x_train_final, y_train_final, x_test_scaled, y_test, config, le)
    joblib.dump(le, config["LABEL_ENCODER"])
    joblib.dump(scaler, config["SCALER_EEGNET"])
    np.save(config["LABEL_CLASSES_NPY"], le.classes_)
    logger.info("Extracting features for tree-based models...")
    x_features = extract_features_parallel(x_windows, config)
    logger.info("Feature extraction complete. Feature shape: %s", x_features.shape)
    train_tree_models(x_features, y_encoded, config, le)


def load_windowed_data(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Load windowed EEG data and corresponding labels from .npy files specified in the config.

    Args:
        config (dict): Configuration dictionary containing file paths for windowed EEG data and labels.

    Returns:
        tuple[np.ndarray, np.ndarray]: (x_windows, y_windows) where x_windows is the EEG data and
            y_windows are the labels.

    Raises:
        FileNotFoundError: If the specified files are not found.
        OSError: If loading fails for other reasons.
        ValueError: If loading fails for other reasons.
        KeyError: If loading fails for other reasons.

    """
    try:
        x_windows = np.load(config["WINDOWED_NPY"])
        y_windows = np.load(config["WINDOWED_LABELS_NPY"])
        logger.info(
            "Loaded windowed data shape: %s, Labels shape: %s",
            x_windows.shape,
            y_windows.shape,
        )
        return x_windows, y_windows  # noqa: TRY300
    except FileNotFoundError:
        logger.exception(
            "Windowed data file not found. Please ensure window_eeg_data.py has been run "
            "and the config paths are correct.",
        )
        raise
    except (OSError, ValueError, KeyError):
        logger.exception("Failed to load windowed data.")
        raise


def encode_labels(y_windows: np.ndarray) -> tuple[LabelEncoder, np.ndarray, np.ndarray]:
    """Encode string or categorical labels into integer and one-hot encoded formats.

    Args:
        y_windows (np.ndarray): Array of labels to encode.

    Returns:
        tuple[LabelEncoder, np.ndarray, np.ndarray]:
            le: Fitted label encoder.
            y_encoded: Integer-encoded labels.
            y_cat: One-hot encoded labels.

    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_windows)
    y_cat = to_categorical(y_encoded)
    return le, y_encoded, y_cat


def preprocess_and_augment(
    x_windows: np.ndarray, y_cat: np.ndarray, config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Split EEG data into train/test sets, scales, balance classes, augment, and return processed arrays.

    Args:
        x_windows (np.ndarray): Windowed EEG data.
        y_cat (np.ndarray): One-hot encoded labels.
        config (dict): Configuration dictionary.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
            X_train_final, y_train_final, X_test_scaled, y_test, scaler

    """

    def balance_and_augment(
        x_train_scaled: np.ndarray, y_train: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Balance classes by downsampling and augmenting the data.

        Args:
            x_train_scaled (np.ndarray): Scaled training data.
            y_train (np.ndarray): One-hot encoded training labels.

        Returns:
            tuple[np.ndarray, np.ndarray]: Balanced and augmented training data and labels.

        """
        labels = np.argmax(y_train, axis=1)
        unique, counts = np.unique(labels, return_counts=True)
        min_count = np.min(counts)
        indices_per_class = [np.nonzero(labels == i)[0] for i in range(len(unique))]
        rng = np.random.default_rng()
        downsampled_indices = np.concatenate(
            [
                rng.choice(idxs, min_count, replace=False)
                for idxs in indices_per_class
            ],
        )
        rng.shuffle(downsampled_indices)
        x_train_bal = x_train_scaled[downsampled_indices]
        y_train_bal = y_train[downsampled_indices]
        x_train_aug = augment_eeg_data(x_train_bal)
        y_train_aug = y_train_bal.copy()
        x_train_final = np.concatenate([x_train_bal, x_train_aug], axis=0)
        y_train_final = np.concatenate([y_train_bal, y_train_aug], axis=0)
        return x_train_final, y_train_final

    x_train, x_test, y_train, y_test = train_test_split(
        x_windows, y_cat, test_size=0.2, random_state=42, stratify=y_cat,
    )
    scaler = StandardScaler()
    x_train_flat = x_train.reshape(-1, config["N_CHANNELS"])
    scaler.fit(x_train_flat)
    x_train_scaled = scaler.transform(
        x_train.reshape(-1, config["N_CHANNELS"]),
    ).reshape(x_train.shape)
    x_test_scaled = scaler.transform(x_test.reshape(-1, config["N_CHANNELS"])).reshape(
        x_test.shape,
    )
    x_train_final, y_train_final = balance_and_augment(x_train_scaled, y_train)
    return x_train_final, y_train_final, x_test_scaled, y_test, scaler


def log_class_distribution(y_train_final: np.ndarray) -> None:
    """Log the class distribution of the training data after downsampling and augmentation.

    Args:
        y_train_final (np.ndarray): One-hot encoded or categorical labels for the training set.

    """
    labels_train = np.argmax(y_train_final, axis=1)
    logger.info(
        "Class distribution after downsampling and augmentation: %s",
        np.bincount(labels_train),
    )
    logger.info(
        "Training context: %d total samples, %d classes",
        len(y_train_final),
        len(np.unique(y_train_final)),
    )
    unique_labels, label_counts = np.unique(y_train_final, return_counts=True)
    class_dist = dict(zip(unique_labels, label_counts))
    logger.info("Detailed class distribution: %s", class_dist)


def extract_features_parallel(
    x_windows: np.ndarray, config: dict[str, Any],
) -> np.ndarray:
    """Extract features from each EEG window in parallel using joblib.

    Args:
        x_windows (np.ndarray): Array of windowed EEG data.
        config (dict[str, Any]): Configuration dictionary containing 'SAMPLING_RATE'.

    Returns:
        np.ndarray: Array of extracted feature vectors for each window.

    """
    return np.array(
        Parallel(n_jobs=-1, prefer="threads")(
            delayed(extract_features)(window, config["SAMPLING_RATE"])
            for window in x_windows
        ),
    )


if __name__ == "__main__":
    main()
