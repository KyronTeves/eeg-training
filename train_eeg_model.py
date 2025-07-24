"""train_eeg_model.py.

Train EEGNet, ShallowConvNet, AdvancedConv1D, Random Forest, and XGBoost models on windowed EEG data.

Handles data loading, preprocessing, augmentation, model training, and artifact saving for EEG classification.

Typical usage:
    $ python train_eeg_model.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from joblib import Parallel, delayed
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model
from keras.optimizers import AdamW
from keras.optimizers.schedules import CosineDecayRestarts
from keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

from EEGModels import AdvancedConv1DNet, EEGNet, ShallowConvNet
from utils import (
    augment_eeg_data,
    check_labels_valid,
    check_no_nan,
    extract_features,
    handle_errors,
    load_config,
    setup_logging,
)

setup_logging()
logger = logging.getLogger(__name__)


def load_windowed_data(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Load windowed EEG data and corresponding labels from .npy files specified in the config.

    Args:
        config (dict): Configuration dictionary containing file paths for windowed EEG data and labels.

    Returns:
        tuple[np.ndarray, np.ndarray]: (x_windows, y_windows) where x_windows is the EEG data and
            y_windows are the labels.

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
    x_windows: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    """Extract features from each EEG window in parallel using joblib.

    Args:
        x_windows (np.ndarray): Array of windowed EEG data.
        config (dict[str, Any]): Configuration dictionary containing 'SAMPLING_RATE'.

    Returns:
        np.ndarray: Array of extracted feature vectors for each window.

    """
    # Feature extraction is CPU-bound (numpy signal processing), so prefer="processes" may be optimal.
    return np.array(
        Parallel(n_jobs=-1, prefer="processes")(
            delayed(extract_features)(window, config["SAMPLING_RATE"]) for window in x_windows
        ),
    )


def compute_class_weights(y_train_final: np.ndarray) -> dict[int, float]:
    """Compute class weights for handling class imbalance in training data.

    Args:
        y_train_final (np.ndarray): One-hot encoded or categorical labels for the training set.

    Returns:
        dict[int, float]: Dictionary mapping class indices to their computed weights.

    """
    labels_train = np.argmax(y_train_final, axis=1)
    class_weights = compute_class_weight("balanced", classes=np.unique(labels_train), y=labels_train)
    return dict(enumerate(class_weights))


def preprocess_and_augment(
    x_windows: np.ndarray,
    y_cat: np.ndarray,
    config: dict[str, Any],
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
        x_train_scaled: np.ndarray,
        y_train: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Balance classes by downsampling and augmenting the data.

        Uses the strategy specified in config['CLASS_BALANCE_STRATEGY']:
            - 'median': Downsample each class to the median class count (default, less aggressive).
            - 'min': Downsample each class to the minimum class count (more aggressive, less data loss for minority).

        Args:
            x_train_scaled (np.ndarray): Scaled training data.
            y_train (np.ndarray): One-hot encoded training labels.

        Returns:
            tuple[np.ndarray, np.ndarray]: Balanced and augmented training data and labels.

        """
        labels = np.argmax(y_train, axis=1)
        unique, counts = np.unique(labels, return_counts=True)
        strategy = config.get("CLASS_BALANCE_STRATEGY", "median")
        target_count = int(np.min(counts)) if strategy == "min" else int(np.median(counts))
        indices_per_class = [np.nonzero(labels == i)[0] for i in range(len(unique))]
        rng = np.random.default_rng()
        downsampled_indices = np.concatenate(
            [
                rng.choice(idxs, target_count, replace=False) if len(idxs) > target_count else idxs
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
        x_windows,
        y_cat,
        test_size=0.2,
        random_state=42,
        stratify=y_cat,
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


def train_eegnet_model(  # noqa: PLR0913
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    config: dict[str, Any],
    label_encoder: LabelEncoder,
    class_weight_dict: dict | None = None,
    ensemble_info: list | None = None,
) -> None:
    """Train and evaluate EEGNet or ShallowConvNet model.

    Args:
        x_train (np.ndarray): Training data, shape (n_samples, window, channels, 1).
        y_train (np.ndarray): Training labels (one-hot or encoded).
        x_test (np.ndarray): Test data, shape (n_samples, window, channels, 1).
        y_test (np.ndarray): Test labels (one-hot or encoded).
        config (dict[str, Any]): Configuration dictionary.
        label_encoder (LabelEncoder): Label encoder.
        class_weight_dict (dict | None): Dictionary mapping class indices to weights for handling class imbalance.
        ensemble_info (list | None): List to collect ensemble model information (optional).

    """
    x_train_eegnet = np.expand_dims(x_train, -1)
    x_test_eegnet = np.expand_dims(x_test, -1)
    x_train_eegnet = np.transpose(x_train_eegnet, (0, 2, 1, 3))
    x_test_eegnet = np.transpose(x_test_eegnet, (0, 2, 1, 3))
    early_stopping = EarlyStopping(
        monitor=config["EARLY_STOPPING_MONITOR"],
        patience=config["EARLY_STOPPING_PATIENCE"],
        restore_best_weights=True,
        verbose=1,
        min_delta=0.0005,
        mode="max",
    )

    reduce_lr = None
    if config.get("USE_REDUCE_LR", True):
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1,
        )

    models_to_train = config.get("MODELS_TO_TRAIN", ["EEGNet", "ShallowConvNet"])
    for model_name in models_to_train:
        logger.info("=== Training %s ===", model_name)
        if model_name == "EEGNet":
            model = EEGNet(
                nb_classes=y_train.shape[1],
                Chans=config["N_CHANNELS"],
                Samples=config["WINDOW_SIZE"],
                kernLength=config["EEGNET_KERN_LENGTH"],
                F1=config["EEGNET_F1"],
                D=config["EEGNET_D"],
                F2=config["EEGNET_F2"],
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
        # ModelCheckpoint for best weights (config-driven)
        if model_name == "EEGNet":
            checkpoint_path = config.get("MODEL_EEGNET_CHECKPOINT")
        elif model_name == "ShallowConvNet":
            checkpoint_path = config.get("MODEL_SHALLOW_CHECKPOINT")
        else:  # fallback
            checkpoint_path = "models/model_best_checkpoint.keras"
        model_checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
            save_weights_only=False,
            mode="max",
        )
        callbacks = [early_stopping, model_checkpoint]
        if reduce_lr is not None:
            callbacks.append(reduce_lr)
        model.fit(
            x_train_eegnet,
            y_train,
            epochs=config["EPOCHS"],
            batch_size=config["BATCH_SIZE"],
            validation_split=config["VALIDATION_SPLIT"],
            class_weight=class_weight_dict,
            callbacks=callbacks,
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
                y_true_labels,
                y_pred_labels,
                target_names=label_encoder.classes_,
            ),
        )
        model.save(model_path)
        logger.info("%s saved to %s", model_name, model_path)
        if ensemble_info is not None:
            ensemble_info.append(
                {
                    "name": model_name,
                    "path": model_path,
                    "type": "keras",
                    "accuracy": float(acc),
                    "weight": 1.0,
                },
            )


def train_and_save_tree_models(  # noqa: PLR0913
    x_features: np.ndarray,
    y_encoded: np.ndarray,
    label_encoder: LabelEncoder,
    rf_path: str,
    xgb_path: str,
    scaler_path: str,
    log_prefix: str = "",
    ensemble_info: list | None = None,
) -> None:
    """Train and evaluate Random Forest and XGBoost models, save models and scaler.

    Args:
        x_features (np.ndarray): Feature matrix for tree models.
        y_encoded (np.ndarray): Encoded labels.
        label_encoder (LabelEncoder): Label encoder.
        rf_path (str): Path to save Random Forest model.
        xgb_path (str): Path to save XGBoost model.
        scaler_path (str): Path to save scaler.
        log_prefix (str): Optional prefix for log messages.
        ensemble_info (list | None): List to collect ensemble model information (optional).

    """
    x_train_tree, x_test_tree, y_train_tree, y_test_tree = train_test_split(
        x_features,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )
    scaler_tree = StandardScaler()
    x_train_scaled_tree = scaler_tree.fit_transform(x_train_tree)
    x_test_scaled_tree = scaler_tree.transform(x_test_tree)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(x_train_scaled_tree, y_train_tree)
    rf_pred = rf.predict(x_test_scaled_tree)
    rf_acc = np.mean(rf_pred == y_test_tree)
    logger.info("%sRandom Forest Results:", log_prefix)
    logger.info("Confusion Matrix:\n%s", confusion_matrix(y_test_tree, rf_pred))
    logger.info(
        "Classification Report:\n%s",
        classification_report(
            y_test_tree,
            rf_pred,
            target_names=label_encoder.classes_,
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
    xgb_acc = np.mean(xgb_pred == y_test_tree)
    logger.info("%sXGBoost Results:", log_prefix)
    logger.info("Confusion Matrix:\n%s", confusion_matrix(y_test_tree, xgb_pred))
    logger.info(
        "Classification Report:\n%s",
        classification_report(
            y_test_tree,
            xgb_pred,
            target_names=label_encoder.classes_,
        ),
    )
    joblib.dump(rf, rf_path)
    joblib.dump(xgb, xgb_path)
    joblib.dump(scaler_tree, scaler_path)
    logger.info("%sTree models and scaler saved: %s, %s, %s", log_prefix, rf_path, xgb_path, scaler_path)
    if ensemble_info is not None:
        ensemble_info.append(
            {
                "name": f"{log_prefix.strip()} RandomForest",
                "path": rf_path,
                "type": "sklearn",
                "accuracy": float(rf_acc),
                "weight": 1.0,
            },
        )
        ensemble_info.append(
            {
                "name": f"{log_prefix.strip()} XGBoost",
                "path": xgb_path,
                "type": "sklearn",
                "accuracy": float(xgb_acc),
                "weight": 1.0,
            },
        )


def train_conv1d_and_tree_models(  # noqa: PLR0913
    x_train_final: np.ndarray,
    y_train_final: np.ndarray,
    x_windows: np.ndarray,
    y_encoded: np.ndarray,
    class_weight_dict: dict | None,
    config: dict[str, Any],
    le: LabelEncoder,
    ensemble_info: list | None = None,
) -> None:
    """Train Advanced Conv1D model and tree models on Conv1D features.

    Args:
        x_train_final (np.ndarray): Preprocessed training data.
        y_train_final (np.ndarray): One-hot encoded labels for the training set.
        x_windows (np.ndarray): Windowed EEG data.
        y_encoded (np.ndarray): Encoded labels for the training set.
        class_weight_dict (dict | None): Class weights for handling class imbalance.
        config (dict[str, Any]): Configuration dictionary.
        le (LabelEncoder): Label encoder for inverse transforming predictions.
        ensemble_info (list | None): List to collect ensemble model information (optional).

    """
    logger.info("=== Training Advanced Conv1D Model ===")
    window_size = config["WINDOW_SIZE"]
    n_channels = config["N_CHANNELS"]
    nb_classes = y_train_final.shape[1]
    conv1d_model = AdvancedConv1DNet(nb_classes, window_size, n_channels)
    lr_schedule = CosineDecayRestarts(
        initial_learning_rate=0.001,
        first_decay_steps=1000,
        t_mul=2.0,
        m_mul=1.0,
        alpha=0.1,
    )
    optimizer = AdamW(
        learning_rate=lr_schedule,
        weight_decay=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
    )
    conv1d_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        patience=25,
        restore_best_weights=True,
        verbose=1,
        min_delta=0.0005,
        mode="max",
    )
    model_checkpoint = ModelCheckpoint(
        config["MODEL_CONV1D_CHECKPOINT"],
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
        save_weights_only=False,
        mode="max",
    )
    # Do NOT use ReduceLROnPlateau when using a LearningRateSchedule (CosineDecayRestarts) with AdamW
    # as it will cause a TypeError. Only use ReduceLROnPlateau if using a float learning rate.
    history = conv1d_model.fit(
        x_train_final,
        y_train_final,
        epochs=300,
        batch_size=32,
        validation_split=0.25,
        class_weight=class_weight_dict,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1,
    )
    # Save the trained model
    conv1d_model.save(config["MODEL_CONV1D_SAVE_PATH_H5"])
    conv1d_model.save(config["MODEL_CONV1D_SAVE_PATH_KERAS"])
    # Save feature extractor (up to the last concatenation layer)
    feature_extractor = Model(inputs=conv1d_model.input, outputs=conv1d_model.get_layer(index=-2).output)
    feature_extractor.save(config["FEATURE_EXTRACTOR_SAVE_PATH_H5"])
    feature_extractor.save(config["FEATURE_EXTRACTOR_SAVE_PATH_KERAS"])
    logger.info("Advanced Conv1D model and feature extractor saved.")
    # Evaluate Conv1D model on training data (since test split is not passed here)
    val_acc = float(np.max(history.history["val_accuracy"])) if "val_accuracy" in history.history else None
    if ensemble_info is not None and val_acc is not None:
        ensemble_info.append(
            {
                "name": "Conv1D",
                "path": config["MODEL_CONV1D_SAVE_PATH_H5"],
                "type": "keras",
                "accuracy": val_acc,
                "weight": 1.0,
            },
        )

    # --- Conv1D Feature Extraction for Tree Models ---
    logger.info("Extracting Conv1D features for tree-based models...")
    x_conv1d_features = feature_extractor.predict(x_windows, batch_size=64, verbose=1)
    logger.info("Conv1D feature extraction complete. Feature shape: %s", x_conv1d_features.shape)
    train_and_save_tree_models(
        x_conv1d_features,
        y_encoded,
        le,
        rf_path="models/eeg_rf_model_conv1d.pkl",
        xgb_path="models/eeg_xgb_model_conv1d.pkl",
        scaler_path="models/eeg_scaler_tree_conv1d.pkl",
        log_prefix="(Conv1D features) ",
        ensemble_info=ensemble_info,
    )


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
    class_weight_dict = compute_class_weights(y_train_final)
    log_class_distribution(y_train_final)
    # --- Collect ensemble info ---
    ensemble_info = []

    train_eegnet_model(
        x_train_final,
        y_train_final,
        x_test_scaled,
        y_test,
        config,
        le,
        class_weight_dict,
        ensemble_info,
    )
    joblib.dump(le, config["LABEL_ENCODER"])
    joblib.dump(scaler, config["SCALER_EEGNET"])
    np.save(config["LABEL_CLASSES_NPY"], le.classes_)

    # --- Classic Feature Extraction for Tree Models ---
    logger.info("Extracting classic features for tree-based models...")
    x_features = extract_features_parallel(x_windows, config)
    logger.info("Classic feature extraction complete. Feature shape: %s", x_features.shape)
    train_and_save_tree_models(
        x_features,
        y_encoded,
        le,
        rf_path=config["MODEL_RF"],
        xgb_path=config["MODEL_XGB"],
        scaler_path=config["SCALER_TREE"],
        log_prefix="(Classic features) ",
        ensemble_info=ensemble_info,
    )

    # --- Advanced Conv1D Model Training and Tree Models ---
    train_conv1d_and_tree_models(
        x_train_final,
        y_train_final,
        x_windows,
        y_encoded,
        class_weight_dict,
        config,
        le,
        ensemble_info=ensemble_info,
    )

    # --- Save ensemble info as JSON ---
    ensemble_metadata = {"models": ensemble_info}
    with Path(config["ENSEMBLE_INFO_PATH"]).open("w", encoding="utf-8") as f:
        json.dump(ensemble_metadata, f, indent=2)
    logger.info("Ensemble info saved to %s", config["ENSEMBLE_INFO_PATH"])


if __name__ == "__main__":
    main()
