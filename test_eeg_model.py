"""test_eeg_model.py.

Evaluate trained EEGNet, ShallowConvNet, Random Forest, and XGBoost models on held-out EEG data windows.

Loads test data and models, applies windowing and scaling, computes predictions, and reports
evaluation metrics for all models and ensemble.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.models import load_model
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix
from umap import UMAP

from utils import (
    check_labels_valid,
    check_no_nan,
    extract_features,
    load_config,
    load_ensemble_info,
    load_models_from_ensemble_info,
    setup_logging,
    window_data,
)

if TYPE_CHECKING:
    from sklearn.calibration import LabelEncoder

setup_logging()
logger = logging.getLogger(__name__)


def load_resources(config: dict) -> tuple[dict, object, list]:
    """Load ensemble info, label encoder, and models.

    Args:
        config (dict): Configuration dictionary containing paths and parameters.

    Returns:
        tuple[dict, object, list]: Ensemble info, label encoder, and list of models.

    """
    try:
        ensemble_info = load_ensemble_info(config)
        label_encoder = joblib.load(config["LABEL_ENCODER_PATH"])
        logger.info("Loaded ensemble info from %s.", config["ENSEMBLE_INFO_PATH"])
        models = load_models_from_ensemble_info(ensemble_info)
        return ensemble_info, label_encoder, models  # noqa: TRY300
    except Exception:
        logger.exception("Failed to load ensemble info or label encoder/class list.")
        raise


def load_and_window_test_data(config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Load and window test data.

    Args:
        config (dict): Configuration dictionary containing paths and parameters.

    Returns:
        tuple[np.ndarray, np.ndarray]: Windowed EEG data and corresponding labels.

    """
    df = pd.read_csv(config["OUTPUT_CSV"])
    test_df = df[df["session_type"].isin(config["TEST_SESSION_TYPES"])]
    eeg_cols = [col for col in test_df.columns if col.startswith("ch_")]
    x = test_df[eeg_cols].to_numpy()
    labels = test_df["label"].to_numpy()
    check_no_nan(x, name="EEG data")
    check_labels_valid(labels, valid_labels=config["LABELS"], name="Labels")
    x = x.reshape(-1, config["N_CHANNELS"])
    labels = labels.reshape(-1, 1)
    x_windows, y_windows = window_data(x, labels, config["WINDOW_SIZE"], config["STEP_SIZE"])
    return x_windows, y_windows


def prepare_test_data_representations(
    x_windows: np.ndarray,
    ensemble_info: dict,
    config: dict,
) -> dict:
    """Prepare all test data representations needed for the loaded models.

    Args:
        x_windows (np.ndarray): Windowed EEG data (n_samples, window, n_channels).
        ensemble_info (dict): Ensemble info loaded from JSON.
        config (dict): Configuration dictionary containing parameters and paths.

    Returns:
        dict: Mapping from representation name to data array.

    """
    n_channels = config["N_CHANNELS"]
    # Prepare classic features
    x_classic_features = np.array([extract_features(window, config["SAMPLING_RATE"]) for window in x_windows])
    # Load and apply classic feature scaler from config
    scaler_tree_path = config["SCALER_TREE"]
    scaler_tree = joblib.load(scaler_tree_path)
    x_classic_features_scaled = scaler_tree.transform(x_classic_features)

    # Prepare scaled windows for CNNs
    scaler_eegnet = joblib.load(config["SCALER_EEGNET"])
    x_windows_flat = x_windows.reshape(-1, n_channels)
    x_windows_scaled = scaler_eegnet.transform(x_windows_flat).reshape(x_windows.shape)
    # Prepare EEGNet input shape: (batch, channels, window, 1)
    x_windows_eegnet = np.expand_dims(x_windows_scaled, -1)
    x_windows_eegnet = np.transpose(x_windows_eegnet, (0, 2, 1, 3))
    # Prepare Conv1D features if feature extractor exists
    conv1d_feature_extractor = None
    conv1d_feature_path = config["CONV1D_FEATURE_EXTRACTOR"]
    if not conv1d_feature_path:
        # Try to find in ensemble_info
        for entry in ensemble_info["models"]:
            if "conv1d_feature_extractor" in entry.get("name", "").lower() or (
                "conv1d" in entry["name"].lower() and "feature_extractor" in entry["name"].lower()
            ):
                conv1d_feature_path = entry["path"]
                break
    if not conv1d_feature_path:
        # Fallback to default
        conv1d_feature_path = "models/eeg_conv1d_feature_extractor.keras"
    try:
        conv1d_feature_extractor = load_model(conv1d_feature_path)
    except (OSError, ImportError):
        try:
            conv1d_feature_extractor = load_model("models/eeg_conv1d_feature_extractor.h5")
        except (OSError, ImportError):
            conv1d_feature_extractor = None
    x_conv1d_features = None
    if conv1d_feature_extractor is not None:
        x_conv1d_features = conv1d_feature_extractor.predict(x_windows_scaled, batch_size=32, verbose=0)
    return {
        "classic_features": x_classic_features_scaled,
        "windows_scaled": x_windows_scaled,
        "windows_eegnet": x_windows_eegnet,
        "conv1d_features": x_conv1d_features,
    }


def map_model_inputs(models: list, features: dict) -> dict:
    """Map model names to their required input features.

    Args:
        models (list): List of model metadata dictionaries.
        features (dict): Dictionary of prepared feature arrays.

    Returns:
        dict: Mapping from model names to their input feature arrays.

    """
    model_inputs = {}
    for m in models:
        name = m["name"]
        if "conv1d" in name.lower() and m["type"] == "keras":
            model_inputs[name] = features["windows_scaled"]
        elif ("shallow" in name.lower() and m["type"] == "keras") or m["type"] == "keras":
            model_inputs[name] = features["windows_eegnet"]
        elif "conv1d features" in name.lower() and features["conv1d_features"] is not None:
            model_inputs[name] = features["conv1d_features"]
        elif "classic" in name.lower():
            model_inputs[name] = features["classic_features"]
        else:
            model_inputs[name] = features["classic_features"]
    return model_inputs


def run_all_model_predictions(models: list[dict], model_inputs: dict) -> dict:
    """Run predictions for all models.

    Args:
        models (list[dict]): List of model metadata dictionaries.
        model_inputs (dict): Mapping from model names to their input feature arrays.

    Returns:
        dict: Mapping from model names to their predictions.

    """
    predictions = {}
    for m in models:
        name = m["name"]
        model = m["model"]
        x_input = model_inputs[name]
        if m["type"] == "keras":
            y_pred_prob = model.predict(x_input)
            y_pred = np.argmax(y_pred_prob, axis=1)
        else:
            y_pred = model.predict(x_input)
        predictions[name] = y_pred
    return predictions


def build_ensemble_predictions(
    predictions: dict,
    label_encoder: LabelEncoder,
    y_true_labels: np.ndarray,
) -> tuple[list, np.ndarray]:
    """Build ensemble predictions (hard voting).

    Args:
        predictions (dict): Mapping from model names to their predictions.
        label_encoder (LabelEncoder): Label encoder for inverse transforming labels.
        y_true_labels (np.ndarray): True labels for the samples.

    Returns:
        tuple[list, np.ndarray]: Ensemble predicted labels and their numeric representation.

    """
    all_pred_labels = list(predictions.values())
    pred_ensemble_labels = []
    for i in range(len(y_true_labels)):
        votes = [label_encoder.inverse_transform([pred[i]])[0] for pred in all_pred_labels]
        final_pred = Counter(votes).most_common(1)[0][0]
        pred_ensemble_labels.append(final_pred)
    pred_ensemble_numeric = label_encoder.transform(pred_ensemble_labels)
    return pred_ensemble_labels, pred_ensemble_numeric


def log_per_sample_predictions(
    y_windows: np.ndarray,
    predictions: dict,
    pred_ensemble_labels: list,
    label_encoder: LabelEncoder,
    y_true_labels: np.ndarray,
) -> None:
    """Log per-sample predictions for all models and ensemble.

    Args:
        y_windows (np.ndarray): Windowed true labels, shape (n_samples, 1) or (n_samples,).
        predictions (dict): Mapping from model names to their predicted label arrays.
        pred_ensemble_labels (list): Ensemble predicted labels.
        label_encoder (LabelEncoder): Label encoder for inverse transforming labels.
        y_true_labels (np.ndarray): True labels for the samples.

    """
    y_true_str = y_windows.ravel()
    pred_strs = {name: label_encoder.inverse_transform(pred) for name, pred in predictions.items()}
    num_samples_to_log = min(100, len(y_true_labels))
    if num_samples_to_log > 0:
        for i in range(num_samples_to_log):
            logger.info("-")
            logger.info("Actual label:   %s", y_true_str[i])
            for name, pred_str in pred_strs.items():
                match = pred_str[i] == y_true_str[i]
                logger.info("%s Predicted label: %s | Match: %s", name, pred_str[i], str(match))
            # Ensemble
            ensemble_match = pred_ensemble_labels[i] == y_true_str[i]
            logger.info("Ensemble (hard voting) label: %s | Match: %s", pred_ensemble_labels[i], str(ensemble_match))
            logger.info("-")


def evaluate_all_models_and_ensemble(
    predictions: dict,
    pred_ensemble_numeric: np.ndarray,
    y_true_labels: np.ndarray,
    label_encoder: LabelEncoder,
) -> None:
    """Evaluate all models and the ensemble.

    Args:
        predictions (dict): Mapping from model names to their predictions.
        pred_ensemble_numeric (np.ndarray): Ensemble predicted labels (numeric).
        y_true_labels (np.ndarray): True labels for the samples.
        label_encoder (LabelEncoder): Label encoder for inverse transforming labels.

    """
    n_samples = len(y_true_labels)
    for name, pred in predictions.items():
        n_correct = int(np.sum(pred == y_true_labels))
        acc = n_correct / n_samples if n_samples else 0.0
        logger.info("%s accuracy on %d test samples: %d/%d (%.2f%%)", name, n_samples, n_correct, n_samples, acc * 100)
    n_correct_ens = int(np.sum(pred_ensemble_numeric == y_true_labels))
    acc_ens = n_correct_ens / n_samples if n_samples else 0.0
    logger.info(
        "Ensemble accuracy on %d test samples: %d/%d (%.2f%%)",
        n_samples,
        n_correct_ens,
        n_samples,
        acc_ens * 100,
    )
    # Optionally, keep detailed reports below
    for name, pred in predictions.items():
        logger.info("%s Confusion Matrix:\n%s", name, confusion_matrix(y_true_labels, pred))
        logger.info(
            "%s Classification Report:\n%s",
            name,
            classification_report(y_true_labels, pred, target_names=label_encoder.classes_),
        )
    logger.info("Ensemble (Hard Voting) Confusion Matrix:\n%s", confusion_matrix(y_true_labels, pred_ensemble_numeric))
    logger.info(
        "Ensemble (Hard Voting) Classification Report:\n%s",
        classification_report(y_true_labels, pred_ensemble_numeric, target_names=label_encoder.classes_),
    )


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: LabelEncoder,
    model_name: str,
) -> None:
    """Log confusion matrix and classification report for a model.

    Args:
        y_true (np.ndarray): True label indices.
        y_pred (np.ndarray): Predicted label indices.
        label_encoder (LabelEncoder): Label encoder with class names.
        model_name (str): Name of the model being evaluated.

    """
    logger.info("--- %s Evaluation ---", model_name)
    logger.info("Confusion Matrix:\n%s", confusion_matrix(y_true, y_pred))
    logger.info(
        "Classification Report:\n%s",
        classification_report(y_true, y_pred, target_names=label_encoder.classes_),
    )


def print_class_distribution(
    df: pd.DataFrame,
    label_col: str,
    label_encoder: LabelEncoder | None = None,
    name: str = "Dataset",
) -> None:
    """Print the class distribution for a DataFrame column.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        label_col (str): The name of the column containing the labels.
        label_encoder (LabelEncoder | None): A fitted LabelEncoder instance for decoding labels. Defaults to None.
        name (str, optional): The name of the dataset. Defaults to "Dataset".

    """
    counts = Counter(df[label_col])
    if label_encoder is not None:
        # If labels are encoded, decode them for readability
        classes = label_encoder.classes_
        counts = {classes[int(k)] if str(k).isdigit() else k: v for k, v in counts.items()}
    logger.info("--- Class distribution for %s ---", name)
    for label, count in counts.items():
        logger.info("%s: %d", label, count)
    logger.info("")


def plot_tsne_features(  # noqa: PLR0913
    x_features: np.ndarray,
    y_labels: np.ndarray,
    label_encoder: LabelEncoder,
    method: str = "tsne",
    perplexity: int = 30,
    n_components: int = 2,
    random_state: int = 42,
    save_dir: str = "plots",
) -> None:
    """Plot t-SNE or UMAP of feature vectors colored by class and save to file. Supports 2D and 3D plots.

    Args:
        x_features (np.ndarray): Feature matrix, shape (n_samples, n_features).
        y_labels (np.ndarray): Array of integer class labels, shape (n_samples,).
        label_encoder (LabelEncoder): Fitted LabelEncoder instance.
        method (str, optional): 'tsne' or 'umap'. Defaults to "tsne".
        perplexity (int, optional): Perplexity for t-SNE. Defaults to 30.
        n_components (int, optional): Number of output dimensions (2 or 3). Defaults to 2.
        random_state (int, optional): Random seed. Defaults to 42.
        save_dir (str, optional): Directory to save the plot. Defaults to "plots".

    Raises:
        ValueError: If an unknown method is specified.

    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if method == "tsne":
        reducer = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state,
        )
        title = f"t-SNE ({n_components}D) of EEG Features"
        fname = f"{save_dir}/tsne_{n_components}d.png"
    elif method == "umap":
        reducer = UMAP(n_components=n_components, random_state=random_state)
        title = f"UMAP ({n_components}D) of EEG Features"
        fname = f"{save_dir}/umap_{n_components}d.png"
    else:
        msg = f"Unknown method: {method}"
        raise ValueError(msg)
    x_reduced = reducer.fit_transform(x_features)
    labels_str = label_encoder.inverse_transform(y_labels) if hasattr(label_encoder, "inverse_transform") else y_labels
    tsne_2d = 2
    tsne_3d = 3
    if n_components == tsne_2d:
        plt.figure(figsize=(8, 6))
        for label in np.unique(labels_str):
            idx = labels_str == label
            plt.scatter(
                x_reduced[idx, 0],
                x_reduced[idx, 1],
                label=label,
                alpha=0.6,
                s=20,
            )
        plt.legend()
        plt.title(title)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
        logger.info("Saved %s to %s", title, fname)
    elif n_components == tsne_3d:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        for label in np.unique(labels_str):
            idx = labels_str == label
            ax.scatter(
                x_reduced[idx, 0],
                x_reduced[idx, 1],
                x_reduced[idx, 2],
                label=label,
                alpha=0.6,
                s=20,
            )
        ax.set_title(title)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")
        ax.legend()
        plt.tight_layout()
        plt.savefig(fname)
        plt.show()  # Show interactive 3D plot
        plt.close()
        logger.info("Saved %s to %s and displayed interactively.", title, fname)
    else:
        logger.warning("n_components=%d not supported for plotting.", n_components)


def plot_feature_distributions(  # noqa: PLR0913
    x_features: np.ndarray,
    y_labels: np.ndarray,
    label_encoder: LabelEncoder,
    feature_indices: list | None = None,
    max_features: int = 5,
    save_dir: str = "plots",
) -> None:
    """Plot feature distributions for each class and save to file.

    Args:
        x_features (np.ndarray): Feature matrix, shape (n_samples, n_features).
        y_labels (np.ndarray): Array of class labels.
        label_encoder (LabelEncoder): Label encoder for inverse transforming class labels.
        feature_indices (list | None, optional): List of feature indices to plot. Defaults to None.
        max_features (int, optional): Maximum number of features to plot. Defaults to 5.
        save_dir (str, optional): Directory to save plots. Defaults to "plots".

    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    labels_str = label_encoder.inverse_transform(y_labels) if hasattr(label_encoder, "inverse_transform") else y_labels
    n_features = x_features.shape[1]
    if feature_indices is None:
        # Pick up to max_features evenly spaced features
        feature_indices = np.linspace(
            0,
            n_features - 1,
            min(max_features, n_features),
            dtype=int,
        )
    for idx in feature_indices:
        plt.figure(figsize=(7, 4))
        for label in np.unique(labels_str):
            sns.kdeplot(
                x_features[labels_str == label, idx],
                label=label,
                fill=True,
                alpha=0.3,
            )
        plt.title(f"Feature {idx} distribution by class")
        plt.xlabel(f"Feature {idx}")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        fname = f"{save_dir}/feature_{idx}_distribution.png"
        plt.savefig(fname)
        plt.close()
        logger.info("Saved feature distribution plot to %s", fname)


def main() -> None:
    """Run dynamic EEG model evaluation and ensemble testing."""
    config = load_config()

    # Step 1: Load resources
    ensemble_info, label_encoder, models = load_resources(config)

    # Step 2: Load and window test data
    x_windows, y_windows = load_and_window_test_data(config)

    # Step 3: Prepare all feature representations
    features = prepare_test_data_representations(x_windows, ensemble_info, config)

    # Step 4: Map model names to their required input features
    model_inputs = map_model_inputs(models, features)

    # Step 5: Run predictions for all models
    y_true_labels = label_encoder.transform(y_windows.ravel())
    predictions = run_all_model_predictions(models, model_inputs)

    # Step 6: Build ensemble predictions (hard voting)
    pred_ensemble_labels, pred_ensemble_numeric = build_ensemble_predictions(predictions, label_encoder, y_true_labels)

    # Step 7: Log per-sample predictions
    log_per_sample_predictions(y_windows, predictions, pred_ensemble_labels, label_encoder, y_true_labels)

    # Step 8: Evaluate all models and ensemble
    evaluate_all_models_and_ensemble(predictions, pred_ensemble_numeric, y_true_labels, label_encoder)


if __name__ == "__main__":
    main()
