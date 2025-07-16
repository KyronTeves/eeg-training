"""realtime_eeg_predict.py.

Real-time EEG direction prediction using LSL streaming from OpenBCI GUI.

This script provides an interactive menu for selecting prediction mode (EEGNet, ShallowConvNet,
Random Forest, XGBoost, or Ensemble) and allows switching between models in real time without
restarting. After each session, you can return to the menu or exit cleanly.

Typical usage:
    $ python realtime_eeg_predict.py

Features:
    - Real-time prediction using multiple models or ensemble.
    - Interactive menu for model selection and session switching.
    - Supports both weighted soft voting and hard voting for ensemble.
    - Clean exit and reconnection to LSL stream as needed.
"""
from __future__ import annotations

import logging
import time

import joblib
import numpy as np

from lsl_stream_handler import LSLStreamHandler
from prediction_pipeline import OptimizedPredictionPipeline
from utils import (
    calibrate_all_models_lsl,
    handle_errors,
    load_config,
    load_ensemble_info,
    load_models_from_ensemble_info,
    setup_logging,
)

setup_logging()
logger = logging.getLogger(__name__)


def load_realtime_resources(config: dict) -> tuple[dict, object, list]:
    """Load resources for real-time EEG prediction.

    Args:
        config (dict): Configuration dictionary containing paths and parameters.

    Returns:
        tuple[dict, object, list]: Ensemble info, label encoder, and list of models.

    """
    try:
        ensemble_info = load_ensemble_info(config)
        label_encoder = joblib.load(config["LABEL_ENCODER"])
        logger.info("Loaded ensemble info from %s.", config["ENSEMBLE_INFO_PATH"])
        models = load_models_from_ensemble_info(ensemble_info)
        return ensemble_info, label_encoder, models  # noqa: TRY300
    except Exception:
        logger.exception("Failed to load ensemble info or label encoder/class list.")
        raise


def process_prediction(
    pipeline: OptimizedPredictionPipeline,
    prediction_count: int,
    mode: str,
    models: list | None = None,
    config: dict | None = None,
) -> int:
    """Process a single prediction (single-model or ensemble) and log detailed model breakdown (dynamic version).

    Args:
        pipeline (OptimizedPredictionPipeline): The prediction pipeline.
        prediction_count (int): The current prediction count.
        mode (str): Prediction mode ('eegnet', 'shallow', 'rf', 'xgb', 'ensemble').
        models (list): List of loaded model dicts.
        config (dict): Config dict.

    Returns:
        int: Updated prediction count.

    """
    # Use dynamic system for all modes (legacy fallback removed)
    if models is not None and config is not None:
        if mode != "ensemble":
            selected_models = [m for m in models if mode.lower() in m["name"].lower()]
            if not selected_models:
                logger.error("Model '%s' not found in loaded models.", mode)
                return prediction_count
            return pipeline.predict_and_log_single_model(
                selected_models,
                config,
                prediction_count,
            )
        return pipeline.predict_and_log_ensemble(models, config, prediction_count)
    return prediction_count


def session_calibration(lsl_handler: LSLStreamHandler, config: dict) -> None:
    """Handle session calibration logic and update config with session model/scaler paths if calibration is performed.

    Args:
        lsl_handler (LSLStreamHandler): LSL handler.
        config (dict): Configuration dictionary.

    """
    session_model_path_eegnet = "models/eeg_direction_model_session.h5"
    session_scaler_path_eegnet = "models/eeg_scaler_session.pkl"
    session_model_path_shallow = "models/eeg_shallow_model_session.h5"
    session_scaler_path_shallow = "models/eeg_scaler_shallow_session.pkl"
    session_model_path_conv1d = "models/eeg_conv1d_model_session.h5"
    session_scaler_path_conv1d = "models/eeg_scaler_conv1d_session.pkl"
    session_model_path_rf_conv1d = "models/eeg_rf_model_conv1d_session.pkl"
    session_model_path_xgb_conv1d = "models/eeg_xgb_model_conv1d_session.pkl"

    user_calib = (
        input("Would you like to calibrate for this session? (Y/n): ").strip().lower()
    )
    if user_calib in ("", "y", "yes"):
        try:
            logger.info("Starting session calibration. Please follow the prompts.")
            calibrate_all_models_lsl(
                lsl_stream_handler=lsl_handler,
                config=config,
                save_dir="models",
                verbose=True,
            )
            logger.info("Session calibration complete. Using session-specific models and scalers.")
            # Update config with session-specific paths
            config["MODEL_EEGNET"] = session_model_path_eegnet
            config["SCALER_EEGNET"] = session_scaler_path_eegnet
            config["MODEL_SHALLOW"] = session_model_path_shallow
            config["SCALER_SHALLOW"] = session_scaler_path_shallow
            # AdvancedConv1D session model/scaler
            config["MODEL_CONV1D"] = session_model_path_conv1d
            config["SCALER_CONV1D"] = session_scaler_path_conv1d
            # Conv1D feature-based tree models
            config["MODEL_RF_CONV1D"] = session_model_path_rf_conv1d
            config["MODEL_XGB_CONV1D"] = session_model_path_xgb_conv1d
            config["SCALER_CONV1D_SESSION"] = session_scaler_path_conv1d
        except (FileNotFoundError, ValueError, RuntimeError):
            logger.exception("Session calibration failed. Proceeding with pre-trained models.")
    else:
        logger.info("Skipping session calibration. Using pre-trained models.")


def select_prediction_mode(ensemble_info: dict | None = None) -> str | None:
    """Prompt user to select prediction display mode dynamically from ensemble_info.

    Args:
        ensemble_info (dict, optional): Ensemble info dict. If None, loads from config.

    Returns:
        str | None: Mode string (model name, 'ensemble', or 'exit')

    """
    if ensemble_info is None:
        config = load_config()
        ensemble_info = load_ensemble_info(config)
    model_names = [m["name"] for m in ensemble_info["models"]]
    logger.info("\nChoose prediction display mode:")
    for idx, name in enumerate(model_names, 1):
        logger.info("%d. %s only", idx, name)
    logger.info("%d. Ensemble (all models)", len(model_names)+1)
    logger.info("%d. Exit", len(model_names)+2)
    while True:
        prompt = f"Enter 1-{len(model_names)+2}: "
        mode = input(prompt).strip()
        try:
            mode_idx = int(mode)
        except ValueError:
            logger.warning("Invalid selection. Please enter a number from 1 to %d.", len(model_names)+2)
            continue
        if 1 <= mode_idx <= len(model_names):
            # Return a normalized key for single-model mode (lowercase, no spaces, for matching)
            return model_names[mode_idx-1].lower().replace(" ", "")
        if mode_idx == len(model_names)+1:
            return "ensemble"
        if mode_idx == len(model_names)+2:
            return "exit"
        logger.warning("Invalid selection. Please enter a number from 1 to %d.", len(model_names)+2)


def initialize_pipeline(config_dict: dict, models_metadata: list | None = None) -> OptimizedPredictionPipeline:
    """Initialize the prediction pipeline with the correct models and scalers.

    Args:
        config_dict (dict): Configuration dictionary.
        models_metadata (list, optional): List of model metadata dicts (from ensemble_info["models"]).

    Returns:
        OptimizedPredictionPipeline: Initialized pipeline.

    """
    pipeline = OptimizedPredictionPipeline(config_dict)
    pipeline.load_optimized_models(models_metadata=models_metadata)
    return pipeline


def prediction_loop(
    lsl_handler: LSLStreamHandler,
    pipeline: OptimizedPredictionPipeline,
    mode: str,
    config_dict: dict,
    models: list | None = None,
) -> None:
    """Run the main loop for real-time EEG prediction from LSL stream.

    Args:
        lsl_handler (LSLStreamHandler): LSL handler.
        pipeline (OptimizedPredictionPipeline): Prediction pipeline.
        mode (str): Prediction mode ('eegnet', 'shallow', 'rf', 'xgb', 'ensemble').
        config_dict (dict): Configuration dictionary.
        models (list): List of loaded model dicts (for ensemble mode).

    """
    prediction_count = 0
    try:
        while True:
            window = lsl_handler.get_window(config_dict["WINDOW_SIZE"], timeout=1.0)
            if window is not None:
                for sample in window:
                    pipeline.add_sample(sample)
                if pipeline.is_ready_for_prediction():
                    prediction_count = process_prediction(
                        pipeline,
                        prediction_count,
                        mode,
                        models=models,
                        config=config_dict,
                    )
            time.sleep(0.001)
    except KeyboardInterrupt:
        logger.info("Stopping real-time prediction...")
    finally:
        lsl_handler.disconnect()
        pipeline.stop_async_prediction()
        stats = pipeline.get_performance_stats()
        logger.info("=== FINAL PERFORMANCE REPORT ===")
        logger.info("Average latency: %.1fms", stats["avg_latency_ms"])
        logger.info("Average FPS: %.1f", stats["fps"])
        logger.info("Total predictions: %d", prediction_count)


@handle_errors
def test_models_without_lsl() -> bool | None:
    """Test model loading and prediction functionality without requiring an LSL stream.

    Loads models, performs a fake prediction, and logs the results for verification.

    Returns:
        bool | None: True if test succeeds, False if it fails, or None if no prediction is made.

    """
    setup_logging()
    config = load_config()
    logger.info("Testing model loading and prediction (no LSL required)...")
    try:
        pipeline = OptimizedPredictionPipeline(config)
        pipeline.load_optimized_models()
        rng = np.random.default_rng()
        fake_window = rng.standard_normal((config["WINDOW_SIZE"], config["N_CHANNELS"])) * 0.1
        for sample in fake_window:
            pipeline.add_sample(sample)
        if pipeline.is_ready_for_prediction():
            # Use the dynamic system for test prediction
            # Load ensemble info and models for dynamic prediction
            ensemble_info = load_ensemble_info(config)
            models = load_models_from_ensemble_info(ensemble_info)
            result = pipeline.predict_realtime_dynamic(models, config)
            if result:
                label, confidence = result
                logger.info(
                    "✓ Test prediction: %s (confidence: %.3f)", label, confidence,
                )
            else:
                logger.info("Models loaded successfully but no prediction made")
        stats = pipeline.get_performance_stats()
        logger.info("✓ Performance stats: %s", stats)
        logger.info("✓ All models loaded and tested successfully!")
        return True  # noqa: TRY300
    except (FileNotFoundError, ValueError, RuntimeError):
        logger.exception("✗ Model test failed.")
        return False


def main() -> None:
    """Start real-time EEG prediction using LSL streaming."""
    setup_logging()
    config = load_config()
    logger.info("Starting LSL-based real-time EEG prediction...")
    lsl_handler = LSLStreamHandler(
        stream_name=config["LSL_STREAM_NAME"], timeout=config["LSL_TIMEOUT"],
    )
    if not lsl_handler.connect():
        logger.error(
            "Failed to connect to LSL stream. Make sure OpenBCI GUI is running with LSL streaming.",
        )
        return
    session_calibration(lsl_handler, config)
    # Load dynamic ensemble resources once for the session
    ensemble_info, _, models = load_realtime_resources(config)
    pipeline = initialize_pipeline(config, models_metadata=ensemble_info["models"])
    while True:
        mode = select_prediction_mode(ensemble_info)
        if mode == "exit":
            logger.info("Exiting real-time prediction.")
            break
        # Reconnect LSL if needed
        if not lsl_handler.connected and not lsl_handler.connect():
            logger.error("Failed to reconnect to LSL stream. Exiting.")
            break
        logger.info("=== REAL-TIME PREDICTION STARTED ===")
        logger.info("Think of different directions to control the system.")
        logger.info("Press Ctrl+C to stop.")
        try:
            prediction_loop(
                lsl_handler,
                pipeline,
                mode,
                config,
                models=models,
            )
        except KeyboardInterrupt:
            logger.info("\nPrediction stopped. Returning to menu...")
            continue


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_models_without_lsl()
    else:
        main()
