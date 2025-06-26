"""
calibrate_session.py

Fine-tune a pre-trained EEGNet model and create a session-specific scaler
    using a small labeled calibration dataset.

Input: Calibration data (windowed, preprocessed, labeled)
Process: Fits new scalers, fine-tunes models, saves session-specific artifacts.
Output: Session-specific model and scaler files for real-time use.
"""

import logging

import joblib
import numpy as np
import tensorflow as tf
from joblib import Parallel, delayed
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

from utils import (check_labels_valid, check_no_nan, extract_features,
                   load_config, setup_logging)

tf.config.run_functions_eagerly(True)  # Enable eager execution

setup_logging()


def load_artifacts(config: dict) -> tuple:
    """
    Load all necessary files for calibration (data, label encoder, models).

    Input: config (dict)
    Process: Loads .npy, .pkl, and model files for calibration
    Output: Tuple (x_calib, y_calib, le, model_eegnet, model_rf, model_xgb)
    """
    logging.info("Loading calibration artifacts...")
    try:
        x_calib = np.load(config["CALIB_X_NPY"])
        y_calib = np.load(config["CALIB_Y_NPY"])
        le = joblib.load(config["LABEL_ENCODER"])
        model_eegnet = load_model(config["MODEL_EEGNET"])
        model_rf = joblib.load(config["MODEL_RF"])
        model_xgb = joblib.load(config["MODEL_XGB"])
        logging.info("Artifacts loaded successfully.")
        return x_calib, y_calib, le, model_eegnet, model_rf, model_xgb
    except FileNotFoundError as e:
        logging.error(
            "Failed to load artifact: %s. Ensure calibration data and base models exist.",
            e,
        )
        raise


def process_and_scale_cnn_data(x_calib: np.ndarray, y_calib: np.ndarray, le) -> tuple:
    """
    Encode labels, fit scaler, and prepare data for CNN calibration.

    Input: x_calib (np.ndarray), y_calib (np.ndarray), le (LabelEncoder)
    Process: Checks data, encodes labels, fits scaler, reshapes for CNN
    Output: (x_calib_eegnet, y_calib_cat, scaler_eegnet)
    """
    logging.info("Processing and scaling CNN data...")
    check_no_nan(x_calib, name="Calibration EEG data")
    check_labels_valid(y_calib, valid_labels=le.classes_, name="Calibration labels")

    y_calib_encoded = le.transform(y_calib.ravel())
    y_calib_cat = to_categorical(y_calib_encoded)

    scaler_eegnet = StandardScaler()
    x_calib_flat = x_calib.reshape(-1, x_calib.shape[-1])
    scaler_eegnet.fit(x_calib_flat)
    x_calib_scaled = scaler_eegnet.transform(x_calib_flat).reshape(x_calib.shape)

    x_calib_eegnet = np.expand_dims(x_calib_scaled, -1)
    x_calib_eegnet = np.transpose(x_calib_eegnet, (0, 2, 1, 3))

    logging.info("CNN data processed. Scaler fitted on %d samples.", len(x_calib_flat))
    return x_calib_eegnet, y_calib_cat, scaler_eegnet


def fine_tune_cnn_model(model, x_data: np.ndarray, y_data: np.ndarray, config: dict):
    """
    Fine-tune the CNN model on calibration data.

    Input: model (Keras), x_data (np.ndarray), y_data (np.ndarray), config (dict)
    Process: Compiles and fits model, returns fine-tuned model
    Output: Fine-tuned model
    """
    logging.info("Starting CNN fine-tuning for %d epochs...", config["CALIB_EPOCHS"])

    # Recompile the model to ensure optimizer compatibility
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    model.fit(
        x_data,
        y_data,
        epochs=config["CALIB_EPOCHS"],
        batch_size=config["CALIB_BATCH_SIZE"],
        verbose=1,
    )
    logging.info("CNN fine-tuning complete.")
    return model


def calibrate_tree_models(
    x_calib: np.ndarray, y_calib: np.ndarray, le, model_rf, model_xgb, config: dict
) -> tuple:
    """
    Extract features, scale, and retrain tree-based models (RF, XGBoost).

    Input: x_calib (np.ndarray), y_calib (np.ndarray), le (LabelEncoder), model_rf, model_xgb, config (dict)
    Process: Extracts features, encodes labels, fits scaler, retrains models
    Output: (model_rf, model_xgb, scaler_tree)
    """
    logging.info("Calibrating tree-based models (RF and XGBoost)...")
    check_no_nan(x_calib, name="Calibration EEG data (tree)")
    check_labels_valid(
        y_calib, valid_labels=le.classes_, name="Calibration labels (tree)"
    )

    sampling_rate = config["SAMPLING_RATE"]
    x_features = np.array(
        Parallel(n_jobs=-1, prefer="threads")(
            delayed(extract_features)(window, sampling_rate) for window in x_calib
        )
    )
    y_encoded = le.transform(y_calib.ravel())

    scaler_tree = StandardScaler()
    x_features_scaled = scaler_tree.fit_transform(x_features)

    logging.info("Retraining Random Forest model...")
    model_rf.fit(x_features_scaled, y_encoded)
    logging.info("Retraining XGBoost model...")
    model_xgb.fit(x_features_scaled, y_encoded)

    logging.info("Tree-based model calibration complete.")
    return model_rf, model_xgb, scaler_tree


def save_session_artifacts(
    model_eegnet, model_rf, model_xgb, scaler_eegnet, scaler_tree, config: dict
):
    """
    Save all fine-tuned models and session-specific scalers to disk.

    Input: model_eegnet, model_rf, model_xgb, scaler_eegnet, scaler_tree, config (dict)
    Process: Saves models and scalers to disk
    Output: None (side effect: files written)
    """
    try:
        # CNN artifacts
        joblib.dump(scaler_eegnet, config["SCALER_EEGNET_SESSION"])
        model_eegnet.save(config["MODEL_EEGNET_SESSION"])
        logging.info("Saved session CNN model and scaler.")

        # Tree-based artifacts
        joblib.dump(scaler_tree, config["SCALER_TREE_SESSION"])
        joblib.dump(model_rf, config["MODEL_RF_SESSION"])
        joblib.dump(model_xgb, config["MODEL_XGB_SESSION"])
        logging.info("Saved session RF and XGBoost models and scaler.")

    except (OSError, AttributeError) as e:
        logging.error("Failed to save session artifacts: %s", e)
        raise


def main():
    """
    Main entry point for session calibration process.

    Input: None (uses config)
    Process: Loads artifacts, calibrates CNN and tree models, saves session artifacts
    Output: None (side effect: files written)
    """
    logging.info("--- Starting Session Calibration ---")
    config = load_config()

    try:
        x_calib, y_calib, le, model_eegnet, model_rf, model_xgb = load_artifacts(config)

        # Calibrate CNN
        x_eegnet, y_eegnet, scaler_eegnet = process_and_scale_cnn_data(
            x_calib, y_calib, le
        )
        model_eegnet_session = fine_tune_cnn_model(
            model_eegnet, x_eegnet, y_eegnet, config
        )

        # Calibrate Tree-based Models
        model_rf_session, model_xgb_session, scaler_tree_session = (
            calibrate_tree_models(x_calib, y_calib, le, model_rf, model_xgb, config)
        )

        save_session_artifacts(
            model_eegnet_session,
            model_rf_session,
            model_xgb_session,
            scaler_eegnet,
            scaler_tree_session,
            config,
        )

        logging.info("--- Session Calibration Complete ---")

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logging.error("Calibration process failed: %s", e)
        raise


if __name__ == "__main__":
    main()
