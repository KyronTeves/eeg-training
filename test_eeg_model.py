"""
test_eeg_model.py

Evaluate trained EEGNet, ShallowConvNet, Random Forest, XGBoost models on held-out EEG data windows.

Input: Labeled EEG CSV file, trained model files
Process: Loads test data and models, applies windowing and scaling, computes predictions, reports metrics.
Output: Evaluation metrics, predictions, and logs.
"""

import logging
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import Parallel, delayed
from keras.models import load_model  # type: ignore
from sklearn.metrics import classification_report, confusion_matrix

from utils import (check_labels_valid, check_no_nan, extract_features,
                   load_config, setup_logging, window_data)


def square(x):
    """Return the element-wise square of the input tensor."""
    return tf.math.square(x)


def log(x):
    """Return the element-wise natural logarithm of the input tensor, with a lower bound for stability."""
    return tf.math.log(tf.math.maximum(x, 1e-7))


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
        test_df = df[df["session_type"].isin(test_session_types)]
        logging.info("Test samples: %d", len(test_df))
    except (pd.errors.EmptyDataError, OSError, ValueError, KeyError) as e:
        logging.error("Failed to load or filter test data: %s", e)
        raise

    eeg_cols = [col for col in test_df.columns if col.startswith("ch_")]
    x = test_df[eeg_cols].values
    labels = test_df["label"].values

    check_no_nan(x, name="EEG data")
    check_labels_valid(labels, valid_labels=config["LABELS"], name="Labels")

    x = x.reshape(-1, n_channels)
    labels = labels.reshape(-1, 1)
    x_windows, y_windows = window_data(x, labels, window_size, step_size)

    if x_windows.shape[1:] != (window_size, n_channels):
        logging.error("Windowed data shape mismatch.")
        raise ValueError("Windowed data shape mismatch.")
    if x_windows.shape[0] != y_windows.shape[0]:
        logging.error("Number of windows and labels do not match.")
        raise ValueError("Number of windows and labels do not match.")

    logging.info("Test windows: %s", x_windows.shape)

    logging.info("Extracting features for tree-based models...")
    x_features = np.array(
        Parallel(n_jobs=-1, prefer="threads")(
            delayed(extract_features)(window, sampling_rate) for window in x_windows
        )
    )
    logging.info("Feature extraction complete. Feature shape: %s", x_features.shape)

    try:
        le = joblib.load(config["LABEL_ENCODER"])
        scaler_eegnet = joblib.load(config["SCALER_EEGNET"])
        scaler_tree = joblib.load(config["SCALER_TREE"])
        model_eegnet = load_model(config["MODEL_EEGNET"])
        rf = joblib.load(config["MODEL_RF"])
        xgb = joblib.load(config["MODEL_XGB"])
        model_shallow = load_model(
            config["MODEL_SHALLOW"], custom_objects={"square": square, "log": log}
        )
        scaler_shallow = joblib.load(
            config.get("SCALER_SHALLOW", config["SCALER_EEGNET"])
        )
    except (ImportError, OSError, AttributeError) as e:
        logging.error("Failed to load models or encoders: %s", e)
        raise

    x_windows_flat = x_windows.reshape(-1, n_channels)
    x_windows_scaled_eegnet = scaler_eegnet.transform(x_windows_flat).reshape(
        x_windows.shape
    )
    x_windows_eegnet = np.expand_dims(x_windows_scaled_eegnet, -1)
    x_windows_eegnet = np.transpose(x_windows_eegnet, (0, 2, 1, 3))
    x_windows_scaled_shallow = scaler_shallow.transform(x_windows_flat).reshape(
        x_windows.shape
    )
    x_windows_shallow = np.expand_dims(x_windows_scaled_shallow, -1)
    x_windows_shallow = np.transpose(x_windows_shallow, (0, 2, 1, 3))
    x_features_scaled = scaler_tree.transform(x_features)

    logging.info("Generating predictions for all models...")
    pred_eegnet_prob = model_eegnet.predict(x_windows_eegnet)
    pred_eegnet_labels = np.argmax(pred_eegnet_prob, axis=1)
    pred_shallow_prob = model_shallow.predict(x_windows_shallow)
    pred_shallow_labels = np.argmax(pred_shallow_prob, axis=1)
    pred_rf_labels = rf.predict(x_features_scaled)
    pred_xgb_labels = xgb.predict(x_features_scaled)
    y_true_labels = le.transform(y_windows.ravel())

    pred_ensemble_labels = ensemble_hard_voting(
        le,
        pred_eegnet_labels,
        pred_shallow_labels,
        pred_rf_labels,
        pred_xgb_labels,
        y_true_labels,
    )
    pred_ensemble_numeric = le.transform(pred_ensemble_labels)

    y_true_str = y_windows.ravel()
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
