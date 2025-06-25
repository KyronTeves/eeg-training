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

from utils import load_config, setup_logging

setup_logging()


def load_artifacts(config: dict) -> tuple:
    """Loads all necessary files for calibration."""
    logging.info("Loading calibration artifacts...")
    try:
        X_calib = np.load(config["CALIB_X_NPY"])
        y_calib = np.load(config["CALIB_Y_NPY"])
        le = joblib.load(config["LABEL_ENCODER"])
        model = load_model(config["MODEL_CNN"])
        logging.info("Artifacts loaded successfully.")
        return X_calib, y_calib, le, model
    except FileNotFoundError as e:
        logging.error(
            "Failed to load artifact: %s. Ensure calibration data and base model exist.",
            e,
        )
        raise


def process_and_scale_data(
    X_calib: np.ndarray, y_calib: np.ndarray, le
) -> tuple:
    """Encodes labels, fits a new scaler, and prepares data for EEGNet."""
    logging.info("Processing and scaling calibration data...")

    # Encode labels
    y_calib_encoded = le.transform(y_calib.ravel())
    y_calib_cat = to_categorical(y_calib_encoded)

    # Fit a new scaler on the session-specific calibration data
    scaler = StandardScaler()
    X_calib_flat = X_calib.reshape(-1, X_calib.shape[-1])
    scaler.fit(X_calib_flat)
    X_calib_scaled = scaler.transform(X_calib_flat).reshape(X_calib.shape)

    # Prepare for EEGNet input shape: (batch, window, channels) -> (batch, channels, window, 1)
    X_calib_eegnet = np.expand_dims(X_calib_scaled, -1)
    X_calib_eegnet = np.transpose(X_calib_eegnet, (0, 2, 1, 3))

    logging.info("Data processed. Scaler fitted on %d samples.", len(X_calib_flat))
    return X_calib_eegnet, y_calib_cat, scaler


def fine_tune_model(model, X_data: np.ndarray, y_data: np.ndarray, config: dict):
    """Fine-tunes the model on the calibration data."""
    logging.info("Starting fine-tuning for %d epochs...", config["CALIB_EPOCHS"])
    model.fit(
        X_data,
        y_data,
        epochs=config["CALIB_EPOCHS"],
        batch_size=config["CALIB_BATCH_SIZE"],
        verbose=1,
    )
    logging.info("Fine-tuning complete.")
    return model


def save_session_artifacts(model, scaler, config: dict):
    """Saves the fine-tuned model and session-specific scaler."""
    try:
        model_path = config["MODEL_CNN_SESSION"]
        scaler_path = config["SCALER_CNN_SESSION"]
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        logging.info("Saved session model to %s", model_path)
        logging.info("Saved session scaler to %s", scaler_path)
    except (OSError, AttributeError) as e:
        logging.error("Failed to save session artifacts: %s", e)
        raise


def main():
    """Main function to run the calibration process."""
    logging.info("--- Starting Session Calibration ---")
    config = load_config()

    try:
        # Load data and base models
        X_calib, y_calib, le, model = load_artifacts(config)

        # Process data and fit a new scaler
        X_calib_processed, y_calib_processed, session_scaler = process_and_scale_data(
            X_calib, y_calib, le
        )

        # Fine-tune the model
        session_model = fine_tune_model(
            model, X_calib_processed, y_calib_processed, config
        )

        # Save the new session-specific artifacts
        save_session_artifacts(session_model, session_scaler, config)

        logging.info("--- Session Calibration Complete ---")

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logging.error("Calibration process failed: %s", e)


if __name__ == "__main__":
    main()
