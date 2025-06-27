"""
test_eeg_model.py

Evaluate trained EEGNet, ShallowConvNet, Random Forest, XGBoost models on held-out EEG data windows.

Input: Labeled EEG CSV file, trained model files
Process: Loads test data and models, applies windowing and scaling, computes predictions, reports metrics.
Output: Evaluation metrics, predictions, and logs.
"""

import logging
import os
from collections import Counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.models import load_model  # type: ignore
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix
from umap import UMAP

from utils import (
    CUSTOM_OBJECTS,
    check_labels_valid,
    check_no_nan,
    extract_features,
    load_config,
    setup_logging,
    window_data,
    log_function_call,
    handle_errors,
)


def ensemble_hard_voting(
    le,
    pred_eegnet_labels,
    pred_shallow_labels,
    pred_rf_labels,
    pred_xgb_labels,
    y_true_labels,
):
    """Perform hard voting ensemble and return predicted labels.

    Args:
        le: Label encoder for inverse transforming labels.
        pred_eegnet_labels: Predicted labels from EEGNet.
        pred_shallow_labels: Predicted labels from ShallowConvNet.
        pred_rf_labels: Predicted labels from Random Forest.
        pred_xgb_labels: Predicted labels from XGBoost.
        y_true_labels: True labels for the data.

    Returns:
        List of ensemble-predicted labels (majority vote).
    """
    pred_ensemble_labels = []
    for i in range(len(y_true_labels)):
        vote_eegnet = le.inverse_transform([pred_eegnet_labels[i]])[0]
        vote_shallow = le.inverse_transform([pred_shallow_labels[i]])[0]
        vote_rf = le.inverse_transform([pred_rf_labels[i]])[0]
        vote_xgb = le.inverse_transform([pred_xgb_labels[i]])[0]
        votes = [vote_eegnet, vote_shallow, vote_rf, vote_xgb]
        final_pred = Counter(votes).most_common(1)[0][0]
        pred_ensemble_labels.append(final_pred)
    return pred_ensemble_labels


@log_function_call
def log_sample_predictions(
    y_true_str,
    pred_eegnet_str,
    pred_shallow_str,
    pred_rf_str,
    pred_xgb_str,
    pred_ensemble_labels,
    num_samples_to_log,
):
    """Log sample predictions and accuracy for each model and the ensemble.

    Args:
        y_true_str: True labels as strings.
        pred_eegnet_str: EEGNet predicted labels as strings.
        pred_shallow_str: ShallowConvNet predicted labels as strings.
        pred_rf_str: Random Forest predicted labels as strings.
        pred_xgb_str: XGBoost predicted labels as strings.
        pred_ensemble_labels: Ensemble predicted labels as strings.
        num_samples_to_log: Number of samples to log.
    """
    eegnet_matches = 0
    shallow_matches = 0
    ensemble_matches = 0

    logging.info("--- Individual Sample Predictions ---")
    for i in range(num_samples_to_log):
        actual_label = y_true_str[i]

        eegnet_pred = pred_eegnet_str[i]
        eegnet_match = actual_label == eegnet_pred
        if eegnet_match:
            eegnet_matches += 1

        shallow_pred = pred_shallow_str[i]
        shallow_match = actual_label == shallow_pred
        if shallow_match:
            shallow_matches += 1

        ensemble_pred = pred_ensemble_labels[i]
        ensemble_match = actual_label == ensemble_pred
        if ensemble_match:
            ensemble_matches += 1

        logging.info("-")
        logging.info("Actual label:   %s", actual_label)
        logging.info(
            "EEGNet Predicted label: %s | Match: %s", eegnet_pred, eegnet_match
        )
        logging.info(
            "ShallowConvNet Predicted label: %s | Match: %s",
            shallow_pred,
            shallow_match,
        )
        logging.info("Random Forest Predicted label: %s", pred_rf_str[i])
        logging.info("XGBoost Predicted label: %s", pred_xgb_str[i])
        logging.info(
            "Ensemble (hard voting) label: %s | Match: %s",
            ensemble_pred,
            ensemble_match,
        )
        logging.info("-")

    eegnet_accuracy = eegnet_matches / num_samples_to_log
    shallow_accuracy = shallow_matches / num_samples_to_log
    ensemble_accuracy = ensemble_matches / num_samples_to_log
    logging.info(
        "EEGNet accuracy on %d test samples: %d/%d (%.2f%%)",
        num_samples_to_log,
        eegnet_matches,
        num_samples_to_log,
        eegnet_accuracy * 100,
    )
    logging.info(
        "ShallowConvNet accuracy on %d test samples: %d/%d (%.2f%%)",
        num_samples_to_log,
        shallow_matches,
        num_samples_to_log,
        shallow_accuracy * 100,
    )
    logging.info(
        "Ensemble accuracy on %d test samples: %d/%d (%.2f%%)",
        num_samples_to_log,
        ensemble_matches,
        num_samples_to_log,
        ensemble_accuracy * 100,
    )


@log_function_call
def evaluate_model(y_true, y_pred, label_encoder, model_name):
    """Log confusion matrix and classification report for a model.

    Args:
        y_true: True label indices.
        y_pred: Predicted label indices.
        label_encoder: Label encoder with class names.
        model_name: Name of the model being evaluated.
    """
    logging.info("--- %s Evaluation ---", model_name)
    logging.info("Confusion Matrix:\n%s", confusion_matrix(y_true, y_pred))
    logging.info(
        "Classification Report:\n%s",
        classification_report(y_true, y_pred, target_names=label_encoder.classes_),
    )


def print_class_distribution(df, label_col, label_encoder=None, name="Dataset"):
    """Print the class distribution for a DataFrame column."""

    counts = Counter(df[label_col])
    if label_encoder is not None:
        # If labels are encoded, decode them for readability
        classes = label_encoder.classes_
        counts = {
            classes[int(k)] if str(k).isdigit() else k: v for k, v in counts.items()
        }
    logging.info("--- Class distribution for %s ---", name)
    for label, count in counts.items():
        logging.info("%s: %d", label, count)
    logging.info("")


def plot_tsne_features(
    x_features,
    y_labels,
    label_encoder,
    method="tsne",
    perplexity=30,
    n_components=2,
    random_state=42,
    save_dir="plots",
):
    """Plot t-SNE or UMAP of feature vectors colored by class and save to file. Supports 2D and 3D plots."""

    os.makedirs(save_dir, exist_ok=True)
    if method == "tsne":
        reducer = TSNE(
            n_components=n_components, perplexity=perplexity, random_state=random_state
        )
        title = f"t-SNE ({n_components}D) of EEG Features"
        fname = f"{save_dir}/tsne_{n_components}d.png"
    elif method == "umap":
        reducer = UMAP(n_components=n_components, random_state=random_state)
        title = f"UMAP ({n_components}D) of EEG Features"
        fname = f"{save_dir}/umap_{n_components}d.png"
    else:
        raise ValueError(f"Unknown method: {method}")
    x_reduced = reducer.fit_transform(x_features)
    labels_str = (
        label_encoder.inverse_transform(y_labels)
        if hasattr(label_encoder, "inverse_transform")
        else y_labels
    )
    if n_components == 2:
        plt.figure(figsize=(8, 6))
        for label in np.unique(labels_str):
            idx = labels_str == label
            plt.scatter(
                x_reduced[idx, 0], x_reduced[idx, 1], label=label, alpha=0.6, s=20
            )
        plt.legend()
        plt.title(title)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
        logging.info("Saved %s to %s", title, fname)
    elif n_components == 3:
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
        logging.info("Saved %s to %s and displayed interactively.", title, fname)
    else:
        logging.warning("n_components=%d not supported for plotting.", n_components)


def plot_feature_distributions(
    x_features,
    y_labels,
    label_encoder,
    feature_indices=None,
    max_features=5,
    save_dir="plots",
):
    """Plot histograms for selected features across classes and save to file."""
    os.makedirs(save_dir, exist_ok=True)
    labels_str = (
        label_encoder.inverse_transform(y_labels)
        if hasattr(label_encoder, "inverse_transform")
        else y_labels
    )
    n_features = x_features.shape[1]
    if feature_indices is None:
        # Pick up to max_features evenly spaced features
        feature_indices = np.linspace(
            0, n_features - 1, min(max_features, n_features), dtype=int
        )
    for idx in feature_indices:
        plt.figure(figsize=(7, 4))
        for label in np.unique(labels_str):
            sns.kdeplot(
                x_features[labels_str == label, idx], label=label, fill=True, alpha=0.3
            )
        plt.title(f"Feature {idx} distribution by class")
        plt.xlabel(f"Feature {idx}")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        fname = f"{save_dir}/feature_{idx}_distribution.png"
        plt.savefig(fname)
        plt.close()
        logging.info("Saved feature distribution plot to %s", fname)


@handle_errors
@log_function_call
def main():
    """Main evaluation pipeline for EEG models on held-out test data windows.

    Loads test data, applies windowing and feature extraction, loads models and scalers, generates predictions,
    logs sample predictions, and reports evaluation metrics for all models and ensemble.
    """
    setup_logging()  # Set up consistent logging to file and console
    config = load_config()
    n_channels = config["N_CHANNELS"]
    window_size = config["WINDOW_SIZE"]
    step_size = config["STEP_SIZE"]
    csv_file = config["OUTPUT_CSV"]
    test_session_types = config["TEST_SESSION_TYPES"]
    sampling_rate = config["SAMPLING_RATE"]
    try:
        logging.info("Loading data from %s ...", csv_file)
        df = pd.read_csv(csv_file)
        print_class_distribution(df, "label", name="Full Dataset (all session types)")
        test_df = df[df["session_type"].isin(test_session_types)]
        logging.info("Test samples: %d", len(test_df))
        print_class_distribution(
            test_df, "label", name="Test Set (filtered session types)"
        )
    except (pd.errors.EmptyDataError, OSError, ValueError, KeyError) as e:
        logging.error("Failed to load or filter test data: %s", e)
        raise
    eeg_cols = [col for col in test_df.columns if col.startswith("ch_")]
    x = test_df[eeg_cols].values
    labels = test_df["label"].values
    check_no_nan(x, name="EEG data")
    check_labels_valid(labels, valid_labels=config["LABELS"], name="Labels")
    if x.shape[1] != n_channels:
        raise ValueError(
            f"Expected {n_channels} channels, but found {x.shape[1]} in the data."
        )
    x = x.reshape(-1, n_channels)
    labels = labels.reshape(-1, 1)
    x_windows, y_windows = window_data(x, labels, window_size, step_size)
    le = joblib.load(config["LABEL_ENCODER"])
    scaler = joblib.load(config["SCALER_EEGNET"])
    x_windows_scaled = scaler.transform(x_windows.reshape(-1, n_channels)).reshape(
        x_windows.shape
    )
    y_windows = le.transform(y_windows.ravel())
    # Feature extraction for tree models
    x_features = np.array(
        [extract_features(window, sampling_rate) for window in x_windows]
    )
    scaler_tree = joblib.load(config["SCALER_TREE"])
    x_features_scaled = scaler_tree.transform(x_features)
    # Load models
    eegnet = load_model(config["MODEL_EEGNET"])
    shallow = load_model(config["MODEL_SHALLOW"], custom_objects=CUSTOM_OBJECTS)
    rf = joblib.load(config["MODEL_RF"])
    xgb = joblib.load(config["MODEL_XGB"])
    # Predict
    pred_eegnet_labels = np.argmax(
        eegnet.predict(np.expand_dims(np.transpose(x_windows_scaled, (0, 2, 1)), -1)),
        axis=1,
    )
    pred_shallow_labels = np.argmax(
        shallow.predict(np.expand_dims(np.transpose(x_windows_scaled, (0, 2, 1)), -1)),
        axis=1,
    )
    pred_rf_labels = rf.predict(x_features_scaled)
    pred_xgb_labels = xgb.predict(x_features_scaled)
    y_true_labels = y_windows
    pred_ensemble_labels = ensemble_hard_voting(
        le,
        pred_eegnet_labels,
        pred_shallow_labels,
        pred_rf_labels,
        pred_xgb_labels,
        y_true_labels,
    )
    pred_ensemble_numeric = le.transform(pred_ensemble_labels)
    y_true_str = le.inverse_transform(y_true_labels)
    pred_eegnet_str = le.inverse_transform(pred_eegnet_labels)
    pred_shallow_str = le.inverse_transform(pred_shallow_labels)
    pred_rf_str = le.inverse_transform(pred_rf_labels)
    pred_xgb_str = le.inverse_transform(pred_xgb_labels)
    num_samples_to_log = min(100, len(y_true_labels))
    if num_samples_to_log > 0:
        log_sample_predictions(
            y_true_str,
            pred_eegnet_str,
            pred_shallow_str,
            pred_rf_str,
            pred_xgb_str,
            pred_ensemble_labels,
            num_samples_to_log,
        )
    evaluate_model(y_true_labels, pred_eegnet_labels, le, "EEGNet")
    evaluate_model(y_true_labels, pred_shallow_labels, le, "ShallowConvNet")
    evaluate_model(y_true_labels, pred_rf_labels, le, "Random Forest")
    evaluate_model(y_true_labels, pred_xgb_labels, le, "XGBoost")
    evaluate_model(y_true_labels, pred_ensemble_numeric, le, "Ensemble (Hard Voting)")


if __name__ == "__main__":
    main()
