"""
Fine-tune a pre-trained EEGNet model and create a session-specific scaler
using a small labeled calibration dataset.

- Loads calibration data (windowed, preprocessed, labeled)
- Fits a new scaler on the calibration data
- Fine-tunes the pre-trained model for a few epochs
- Saves the session-specific model and scaler for real-time use
"""

import logging

import joblib
import numpy as np
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

from utils import load_config, setup_logging, extract_features

setup_logging()


def load_artifacts(config: dict) -> tuple:
    """Loads all necessary files for calibration."""
    logging.info("Loading calibration artifacts...")
    try:
        x_calib = np.load(config["CALIB_X_NPY"])
        y_calib = np.load(config["CALIB_Y_NPY"])
        le = joblib.load(config["LABEL_ENCODER"])
        model_cnn = load_model(config["MODEL_CNN"])
        model_rf = joblib.load(config["MODEL_RF"])
        model_xgb = joblib.load(config["MODEL_XGB"])
        logging.info("Artifacts loaded successfully.")
        return x_calib, y_calib, le, model_cnn, model_rf, model_xgb
    except FileNotFoundError as e:
        logging.error(
            "Failed to load artifact: %s. Ensure calibration data and base models exist.",
            e,
        )
        raise


def process_and_scale_cnn_data(
    x_calib: np.ndarray, y_calib: np.ndarray, le
) -> tuple:
    """Encodes labels, fits a new scaler, and prepares data for EEGNet."""
    logging.info("Processing and scaling CNN data...")
    y_calib_encoded = le.transform(y_calib.ravel())
    y_calib_cat = to_categorical(y_calib_encoded)

    scaler_cnn = StandardScaler()
    x_calib_flat = x_calib.reshape(-1, x_calib.shape[-1])
    scaler_cnn.fit(x_calib_flat)
    x_calib_scaled = scaler_cnn.transform(x_calib_flat).reshape(x_calib.shape)

    x_calib_eegnet = np.expand_dims(x_calib_scaled, -1)
    x_calib_eegnet = np.transpose(x_calib_eegnet, (0, 2, 1, 3))

    logging.info("CNN data processed. Scaler fitted on %d samples.", len(x_calib_flat))
    return x_calib_eegnet, y_calib_cat, scaler_cnn


def fine_tune_cnn_model(model, x_data: np.ndarray, y_data: np.ndarray, config: dict):
    """Fine-tunes the CNN model on the calibration data."""
    logging.info("Starting CNN fine-tuning for %d epochs...", config["CALIB_EPOCHS"])
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
    """Extracts features, scales, and retrains tree-based models."""
    logging.info("Calibrating tree-based models (RF and XGBoost)...")

    sampling_rate = config["SAMPLING_RATE"]
    x_features = np.array(
        [extract_features(window, sampling_rate) for window in x_calib]
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
    model_cnn, model_rf, model_xgb, scaler_cnn, scaler_tree, config: dict
):
    """Saves all fine-tuned models and session-specific scalers."""
    try:
        # CNN artifacts
        joblib.dump(scaler_cnn, config["SCALER_CNN_SESSION"])
        model_cnn.save(config["MODEL_CNN_SESSION"])
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
    """Main function to run the calibration process."""
    logging.info("--- Starting Session Calibration ---")
    config = load_config()

    try:
        x_calib, y_calib, le, model_cnn, model_rf, model_xgb = load_artifacts(config)

        # Calibrate CNN
        x_cnn, y_cnn, scaler_cnn = process_and_scale_cnn_data(x_calib, y_calib, le)
        model_cnn_session = fine_tune_cnn_model(model_cnn, x_cnn, y_cnn, config)

        # Calibrate Tree-based Models
        model_rf_session, model_xgb_session, scaler_tree_session = calibrate_tree_models(
            x_calib, y_calib, le, model_rf, model_xgb, config
        )

        save_session_artifacts(
            model_cnn_session,
            model_rf_session,
            model_xgb_session,
            scaler_cnn,
            scaler_tree_session,
            config,
        )

        logging.info("--- Session Calibration Complete ---")

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logging.error("Calibration process failed: %s", e)


if __name__ == "__main__":
    main()
