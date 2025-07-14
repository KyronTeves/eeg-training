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

import json
import logging
import os
import threading
import time
import warnings
from collections import Counter, deque
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np
from keras.models import load_model

from lsl_stream_handler import LSLStreamHandler
from utils import (
    CUSTOM_OBJECTS,
    calibrate_all_models_lsl,
    extract_features,
    handle_errors,
    load_config,
    setup_logging,
)

if TYPE_CHECKING:
    from collections.abc import Callable

setup_logging()
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings and info messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Suppress other warnings
warnings.filterwarnings("ignore", category=UserWarning, module="scipy")
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")


# Mapping for short label display
SHORT_LABELS = {
    "forward": "FWD",
    "backward": "BWD",
    "left": "LFT",
    "right": "RGT",
    "neutral": "NEU",
}


def short_label(label: str) -> str:
    """Return a short display label for a given direction label.

    Args:
        label (str): The full direction label.

    Returns:
        str: The short label (e.g., "FWD" for "forward").

    """
    return SHORT_LABELS.get(label.lower(), label[:3].upper())


# --- Dynamic resource loading (from test_eeg_model.py pattern) ---
def load_ensemble_info(config: dict) -> dict:
    """Load ensemble info from the configured path."""
    with Path(config["ENSEMBLE_INFO_PATH"]).open(encoding="utf-8") as f:
        return json.load(f)


def load_models_from_ensemble_info(ensemble_info: dict) -> list:
    """Load all models described in the ensemble info dict."""
    models = []
    for model_entry in ensemble_info["models"]:
        model_type = model_entry["type"]
        model_path = model_entry["path"]
        name = model_entry["name"]
        try:
            if model_type == "keras":
                model = load_model(model_path, custom_objects=CUSTOM_OBJECTS)
            elif model_type == "sklearn":
                model = joblib.load(model_path)
            else:
                logger.warning("Unknown model type: %s for %s", model_type, name)
                continue
            models.append(
                {"name": name, "type": model_type, "model": model, "path": model_path},
            )
            logger.info("Loaded %s model: %s from %s", model_type, name, model_path)
        except (OSError, ImportError, TypeError):
            logger.exception("Failed to load model %s from %s.", name, model_path)
    return models


def load_realtime_resources(config: dict) -> tuple[dict, object, list]:
    """Load ensemble info, label encoder, and models for real-time prediction."""
    try:
        ensemble_info = load_ensemble_info(config)
        label_encoder = joblib.load(config["LABEL_ENCODER"])
        logger.info("Loaded ensemble info from %s.", config["ENSEMBLE_INFO_PATH"])
        models = load_models_from_ensemble_info(ensemble_info)
        return ensemble_info, label_encoder, models  # noqa: TRY300
    except Exception:
        logger.exception("Failed to load ensemble info or label encoder/class list.")
        raise


class OptimizedPredictionPipeline:
    """High-performance real-time EEG prediction pipeline.

    Args:
        config_dict (dict): Configuration dictionary.

    Handles model loading, preprocessing, ensemble prediction, async prediction loop,
    performance tracking, and session calibration.

    Returns:
        Real-time predictions (label, confidence), performance stats, logs.

    """

    def __init__(self, config_dict: dict) -> None:
        """Initialize the prediction pipeline with configuration.

        Args:
            config_dict (dict): Configuration dictionary with model/scaler paths, window size, etc.

        """
        self.config = config_dict
        self.window_size = config_dict["WINDOW_SIZE"]
        self.n_channels = config_dict["N_CHANNELS"]
        self.confidence_threshold = config_dict.get("CONFIDENCE_THRESHOLD", 0.7)

        # Use buffer size multiplier from config
        buffer_multiplier = self.config.get("BUFFER_SIZE_MULTIPLIER", 2)
        self.buffer = deque(maxlen=self.window_size * buffer_multiplier)

        # Models and preprocessors
        self.models = {}
        self.scalers = {}
        self.label_encoder = None

        # Performance tracking
        self.prediction_times = deque(maxlen=100)
        self.last_prediction = None
        self.prediction_confidence = 0.0

        # Threading for async processing
        self.prediction_thread = None
        self.stop_thread = False
        self.prediction_ready = threading.Event()

        # Internal flags
        self._shape_mismatch_logged = False
        self._eegnet_error_logged = False
        self._shallow_shape_mismatch_logged = False
        self._shallow_error_logged = False

    def load_optimized_models(self) -> None:
        """Load and optimize all models (EEGNet, ShallowConvNet, RF, XGBoost) and scalers for inference."""
        logger.info("Loading and optimizing models for real-time inference...")
        try:
            # CNN Models - now trained with correct window size (125)
            self.models["eegnet"] = {"model": load_model(self.config["MODEL_EEGNET"])}
            self.models["shallow"] = {
                "model": load_model(
                    self.config["MODEL_SHALLOW"], custom_objects=CUSTOM_OBJECTS,
                ),
            }
            # Tree-based models (already fast)
            self.models["rf"] = joblib.load(self.config["MODEL_RF"])
            self.models["xgb"] = joblib.load(self.config["MODEL_XGB"])
            # Scalers and encoders
            self.scalers["eegnet"] = joblib.load(self.config["SCALER_EEGNET"])
            self.scalers["tree"] = joblib.load(self.config["SCALER_TREE"])
            self.label_encoder = joblib.load(self.config["LABEL_ENCODER"])
            self._warmup_models()
            logger.info("Model optimization complete.")
        except FileNotFoundError:
            logger.exception("Required model file not found.")
            logger.exception("Please run 'python train_eeg_model.py' first to train the models.")
            raise
        except Exception:
            logger.exception("Failed to load models.")
            raise

    def _warmup_models(self) -> None:
        """Run dummy predictions on all models to reduce first-call latency."""
        logger.info("Warming up models...")

        # Create dummy data
        rng = np.random.default_rng()
        dummy_window = rng.standard_normal((1, self.window_size, self.n_channels))

        # Create dummy features with correct feature count
        # For tree models: 16 channels * (5 band powers + 3 stats) = 128 features
        expected_features = (
            self.n_channels * 8
        )  # 5 band powers + 3 statistical features per channel
        dummy_features = rng.standard_normal((1, expected_features))

        # Warm up EEGNet
        self.predict_eegnet(dummy_window)

        # Warm up ShallowConvNet
        self.predict_shallow(dummy_window)

        # Warm up tree models
        self.models["rf"].predict(dummy_features)
        self.models["xgb"].predict(dummy_features)

        logger.info("Model warmup complete.")

    def add_sample(self, sample: np.ndarray) -> None:
        """Add a new EEG sample to the rolling buffer.

        Args:
            sample (np.ndarray): Input EEG sample of shape (n_channels,)

        Raises:
            ValueError: If the input sample has an incorrect shape.

        """
        if len(sample) != self.n_channels:
            msg = f"Expected {self.n_channels} channels, got {len(sample)}"
            raise ValueError(msg)

        self.buffer.append(sample)

    def is_ready_for_prediction(self) -> bool:
        """Check if the buffer has enough samples for a prediction window.

        Returns:
            bool: True if ready, False otherwise

        """
        return len(self.buffer) >= self.window_size

    def get_current_window(self) -> np.ndarray:
        """Return the most recent window of EEG data from the buffer.

        Returns:
            np.ndarray: Array of shape (1, window_size, n_channels)

        """
        return np.array(list(self.buffer)[-self.window_size:]).reshape(
            (1, self.window_size, self.n_channels),
        )

    def _reset_prediction_times(self) -> None:
        """Clear the prediction times buffer (for performance stats)."""
        self.prediction_times.clear()

    def predict_eegnet(self, window: np.ndarray) -> tuple[np.ndarray, float]:
        """Run EEGNet model prediction on a window of EEG data.

        Args:
            window (np.ndarray): Input window of shape (1, window_size, n_channels)

        Returns:
            tuple[np.ndarray, float]: (probabilities, confidence)

        """
        start_time = time.time()
        try:
            model = self.models["eegnet"]["model"]
            expected_shape = model.input_shape
            if expected_shape[2] != self.window_size:
                if not self._shape_mismatch_logged:
                    logger.warning(
                        "EEGNet model shape mismatch: expected %d samples, got %d. "
                        "Model needs retraining with new window size. Using zeros for now.",
                        expected_shape[2],
                        self.window_size,
                    )
                    self._shape_mismatch_logged = True
                return np.zeros(len(self.label_encoder.classes_)), 0.0
            window_scaled = (
                self.scalers["eegnet"]
                .transform(window.reshape(-1, self.n_channels))
                .reshape(window.shape)
            )
            window_input = np.expand_dims(window_scaled, -1)
            window_input = np.transpose(window_input, (0, 2, 1, 3))
            if "interpreter" in self.models["eegnet"]:
                interpreter = self.models["eegnet"]["interpreter"]
                input_details = self.models["eegnet"]["input_details"]
                output_details = self.models["eegnet"]["output_details"]
                interpreter.set_tensor(input_details[0]["index"], window_input.astype(np.float32))
                interpreter.invoke()
                predictions = interpreter.get_tensor(output_details[0]["index"])
            else:
                predictions = self.models["eegnet"]["model"].predict(window_input, verbose=0)
            confidence = np.max(predictions)
            elapsed = time.time() - start_time
            self.prediction_times.append(elapsed)
            return predictions[0], confidence
        except (ValueError, RuntimeError):
            if not self._eegnet_error_logged:
                logger.exception("EEGNet prediction failed.")
                logger.exception(
                    "EEGNet will be disabled for this session."
                    "Please retrain models with correct window size.",
                )
                self._eegnet_error_logged = True
            return np.zeros(len(self.label_encoder.classes_)), 0.0

    def predict_shallow(self, window: np.ndarray) -> tuple[np.ndarray, float]:
        """Run ShallowConvNet model prediction on a window of EEG data.

        Args:
            window (np.ndarray): Input window of shape (1, window_size, n_channels)

        Returns:
            tuple[np.ndarray, float]: (probabilities, confidence)

        """
        start_time = time.time()
        try:
            model = self.models["shallow"]["model"]
            expected_shape = model.input_shape
            if expected_shape[2] != self.window_size:
                if not self._shallow_shape_mismatch_logged:
                    logger.warning(
                        "ShallowConvNet model shape mismatch: expected %d samples, got %d. "
                        "Model needs retraining with new window size. Using zeros for now.",
                        expected_shape[2],
                        self.window_size,
                    )
                    self._shallow_shape_mismatch_logged = True
                return np.zeros(len(self.label_encoder.classes_)), 0.0
            scaler_to_use = self.scalers.get("shallow") or self.scalers["eegnet"]
            window_scaled = scaler_to_use.transform(
                window.reshape(-1, self.n_channels),
            ).reshape(window.shape)
            window_input = np.expand_dims(window_scaled, -1)
            window_input = np.transpose(window_input, (0, 2, 1, 3))
            predictions = self.models["shallow"]["model"].predict(
                window_input, verbose=0,
            )
            confidence = np.max(predictions)
            elapsed = time.time() - start_time
            self.prediction_times.append(elapsed)
            return predictions[0], confidence
        except (ValueError, RuntimeError):
            if not self._shallow_error_logged:
                logger.exception("ShallowConvNet prediction failed.")
                logger.exception(
                    "ShallowConvNet will be disabled for this session."
                    "Please retrain models with correct window size.",
                )
                self._shallow_error_logged = True
            return np.zeros(len(self.label_encoder.classes_)), 0.0

    def predict_tree_models(
        self, window: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Run Random Forest and XGBoost predictions on a window of EEG data.

        Args:
            window (np.ndarray): Input window of shape (1, window_size, n_channels)

        Returns:
            tuple[np.ndarray, np.ndarray, float]: (rf_probs, xgb_probs, confidence)

        """
        start_time = time.time()
        try:
            features = extract_features(window.squeeze(), self.config["SAMPLING_RATE"])
            features = features.reshape(1, -1)
            features_scaled = self.scalers["tree"].transform(features)
            rf_proba = self.models["rf"].predict_proba(features_scaled)[0]
            xgb_proba = self.models["xgb"].predict_proba(features_scaled)[0]
            avg_proba = (rf_proba + xgb_proba) / 2
            confidence = np.max(avg_proba)
            elapsed = time.time() - start_time
            self.prediction_times.append(elapsed)
            return rf_proba, xgb_proba, confidence  # noqa: TRY300
        except (ValueError, RuntimeError):
            logger.exception("Tree model prediction failed.")
            num_classes = len(self.label_encoder.classes_)
            return np.zeros(num_classes), np.zeros(num_classes), 0.0

    def predict_realtime(self, *, use_hard_voting: bool = False) -> tuple[str, float] | None:  # noqa: PLR0915
        """Perform ensemble prediction using all available models on the current buffer.

        Args:
            use_hard_voting (bool): If True, use hard voting (majority rule). If False, use weighted soft voting.

        Returns:
            tuple[str, float] | None: (predicted_label, confidence) or None if not enough data

        """
        if not self.is_ready_for_prediction():
            return None
        window = self.get_current_window()
        eeg_probs, _ = self.predict_eegnet(window)
        shallow_probs, _ = self.predict_shallow(window)
        rf_probs, xgb_probs, _ = self.predict_tree_models(window)

        # Logging for diagnostics
        logger.debug("EEGNet probs: %s", eeg_probs)
        logger.debug("ShallowConvNet probs: %s", shallow_probs)
        logger.debug("RandomForest probs: %s", rf_probs)
        logger.debug("XGBoost probs: %s", xgb_probs)

        # Get predicted labels for hard voting
        model_preds = []
        if not np.all(eeg_probs == 0):
            eeg_pred = np.argmax(eeg_probs)
            model_preds.append(eeg_pred)
        if not np.all(shallow_probs == 0):
            shallow_pred = np.argmax(shallow_probs)
            model_preds.append(shallow_pred)
        model_preds.append(np.argmax(rf_probs))
        model_preds.append(np.argmax(xgb_probs))
        if use_hard_voting:
            # Hard voting (majority rule)
            if not model_preds:
                return "neutral", 0.0

            vote_counts = Counter(model_preds)
            majority_pred = vote_counts.most_common(1)[0][0]
            confidence = vote_counts.most_common(1)[0][1] / len(model_preds)
            predicted_label = self.label_encoder.inverse_transform([majority_pred])[0]
            self.last_prediction = predicted_label
            self.prediction_confidence = confidence
            return predicted_label, confidence

        # Default: weighted soft voting
        available_models = []
        model_weights = []
        model_predictions = []
        if not np.all(eeg_probs == 0):
            available_models.append("EEGNet")
            model_weights.append(0.4)
            model_predictions.append(eeg_probs)
        if not np.all(shallow_probs == 0):
            available_models.append("ShallowConvNet")
            model_weights.append(0.4)
            model_predictions.append(shallow_probs)
        available_models.extend(["RandomForest", "XGBoost"])
        model_weights.extend([0.1, 0.1])
        model_predictions.extend([rf_probs, xgb_probs])
        total_weight = sum(model_weights)
        model_weights = [w / total_weight for w in model_weights]
        final_probs = np.zeros_like(rf_probs)
        for probs, weight in zip(model_predictions, model_weights):
            final_probs += probs * weight
        confidence = np.max(final_probs)
        if confidence < self.confidence_threshold:
            return "neutral", confidence
        predicted_idx = np.argmax(final_probs)
        predicted_label = self.label_encoder.inverse_transform([predicted_idx])[0]
        self.last_prediction = predicted_label
        self.prediction_confidence = confidence
        return predicted_label, confidence


    def get_performance_stats(self) -> dict:
        """Get average latency, FPS, buffer size, and last confidence for real-time predictions.

        Returns:
            dict: Stats including avg_latency_ms, fps, buffer_size, last_confidence

        """
        if not self.prediction_times:
            return {"avg_latency_ms": 0, "fps": 0}

        avg_latency = np.mean(self.prediction_times) * 1000  # Convert to ms
        fps = 1.0 / np.mean(self.prediction_times) if self.prediction_times else 0

        return {
            "avg_latency_ms": round(avg_latency, 2),
            "fps": round(fps, 1),
            "buffer_size": len(self.buffer),
            "last_confidence": round(self.prediction_confidence, 3),
        }

    def start_async_prediction(
        self, callback: Callable[[tuple[str, float]], None] | None = None,
    ) -> None:
        """Start asynchronous prediction loop in a background thread.

        Args:
            callback (Callable[[tuple[str, float]], None] | None): Callback for prediction results. Defaults to None.

        """

        def _async_prediction_loop() -> None:
            while not self.stop_thread:
                try:
                    if self.is_ready_for_prediction():
                        result = self.predict_realtime()
                        if result and callback:
                            callback(result)
                        self.prediction_ready.set()
                except RuntimeError:
                    logger.exception(
                        "Unexpected runtime error in prediction loop",
                    )

                time.sleep(0.001)  # Small delay to prevent busy waiting

        self.prediction_thread = threading.Thread(
            target=_async_prediction_loop, daemon=True,
        )
        self.prediction_thread.start()
        logger.info("Asynchronous prediction started.")

    def stop_async_prediction(self) -> None:
        """Stop the asynchronous prediction thread."""
        self.stop_thread = True
        if self.prediction_thread:
            self.prediction_thread.join(timeout=1.0)
        logger.info("Asynchronous prediction stopped.")

    def prepare_realtime_features(
        self,
        window: np.ndarray,
        ensemble_info: dict,
        config: dict,
    ) -> dict:
        """Prepare all feature representations for a single real-time EEG window.

        Args:
            window (np.ndarray): EEG window, shape (1, window_size, n_channels)
            ensemble_info (dict): Ensemble info loaded from JSON.
            config (dict): Configuration dictionary.

        Returns:
            dict: Mapping from representation name to data array (batch size 1).

        """
        n_channels = config["N_CHANNELS"]
        # Classic features
        x_classic_features = np.array([
            extract_features(window[0], config["SAMPLING_RATE"]),
        ])  # shape (1, n_features)
        scaler_tree = joblib.load(config["SCALER_TREE"])
        x_classic_features_scaled = scaler_tree.transform(x_classic_features)

        # Scaled window for CNNs
        scaler_eegnet = joblib.load(config["SCALER_EEGNET"])
        x_window_flat = window.reshape(-1, n_channels)
        x_window_scaled = scaler_eegnet.transform(x_window_flat).reshape(window.shape)
        # EEGNet input shape: (batch, channels, window, 1)
        x_window_eegnet = np.expand_dims(x_window_scaled, -1)
        x_window_eegnet = np.transpose(x_window_eegnet, (0, 2, 1, 3))

        # Conv1D features (if extractor exists)
        conv1d_feature_extractor = None
        conv1d_feature_path = config.get("CONV1D_FEATURE_EXTRACTOR")
        if not conv1d_feature_path:
            for entry in ensemble_info["models"]:
                if "conv1d_feature_extractor" in entry.get("name", "").lower() or (
                    "conv1d" in entry["name"].lower() and "feature_extractor" in entry["name"].lower()
                ):
                    conv1d_feature_path = entry["path"]
                    break
        if not conv1d_feature_path:
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
            x_conv1d_features = conv1d_feature_extractor.predict(x_window_scaled, batch_size=1, verbose=0)
        return {
            "classic_features": x_classic_features_scaled,
            "windows_scaled": x_window_scaled,
            "windows_eegnet": x_window_eegnet,
            "conv1d_features": x_conv1d_features,
        }

    def map_model_inputs_realtime(self, models: list, features: dict) -> dict:
        """Map model names to their required input features for real-time prediction.

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


    def predict_realtime_dynamic(
        self,
        models: list,
        ensemble_info: dict,
        config: dict,
        *,
        use_hard_voting: bool = True,
    ) -> tuple[str, float] | None:
        """Perform dynamic ensemble prediction using all loaded models and correct features."""
        if not self.is_ready_for_prediction():
            return None
        window = self.get_current_window()
        # Prepare all features for this window
        features = self.prepare_realtime_features(window, ensemble_info, config)
        # Map model names to their required input features
        model_inputs = self.map_model_inputs_realtime(models, features)
        # Run predictions for all models
        predictions = {}
        for m in models:
            name = m["name"]
            model = m["model"]
            x_input = model_inputs[name]
            if m["type"] == "keras":
                y_pred_prob = model.predict(x_input, verbose=0)
                y_pred = np.argmax(y_pred_prob, axis=1)
                predictions[name] = y_pred[0]
            else:
                y_pred = model.predict(x_input)
                predictions[name] = y_pred[0]
        # Hard voting ensemble
        if use_hard_voting:
            vote_counts = Counter(predictions.values())
            majority_pred = vote_counts.most_common(1)[0][0]
            confidence = vote_counts.most_common(1)[0][1] / len(predictions)
            predicted_label = self.label_encoder.inverse_transform([majority_pred])[0]
            self.last_prediction = predicted_label
            self.prediction_confidence = confidence
            return predicted_label, confidence
        # (Optional: add soft voting here if needed)
        return None


def process_prediction(  # noqa: PLR0913
    pipeline: OptimizedPredictionPipeline,
    prediction_count: int,
    models: list | None = None,
    ensemble_info: dict | None = None,
    config: dict | None = None,
    *,
    use_hard_voting: bool = False,
) -> int:
    """Process a single ensemble prediction and log detailed model breakdown (dynamic version).

    Args:
        pipeline (OptimizedPredictionPipeline): The prediction pipeline.
        prediction_count (int): The current prediction count.
        use_hard_voting (bool): Whether to use hard voting.
        models (list): List of loaded model dicts.
        ensemble_info (dict): Ensemble info dict.
        config (dict): Config dict.

    Returns:
        int: Updated prediction count.

    """
    if models is not None and ensemble_info is not None and config is not None:
        result = pipeline.predict_realtime_dynamic(models, ensemble_info, config, use_hard_voting=use_hard_voting)
    else:
        # fallback to legacy method for backward compatibility
        result = pipeline.predict_realtime(use_hard_voting=use_hard_voting)
    if result:
        predicted_label, confidence = result
        prediction_count += 1
        status = "✓" if confidence > pipeline.config["CONFIDENCE_THRESHOLD"] else "?"
        logger.info(
            "[%-4d] %s %-8s(ens:%.3f)",
            prediction_count,
            status,
            predicted_label.upper(),
            confidence,
        )
        if prediction_count % 50 == 0:
            stats = pipeline.get_performance_stats()
            logger.info(
                "Performance: %.1fms avg, %.1f FPS",
                stats["avg_latency_ms"],
                stats["fps"],
            )
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

    user_calib = (
        input("Would you like to calibrate for this session? (Y/n): ").strip().lower()
    )
    if user_calib in ("", "y", "yes"):
        try:
            logger.info("Starting session calibration. Please follow the prompts.")
            calibrate_all_models_lsl(
                lsl_stream_handler=lsl_handler,
                config_path="config.json",
                save_dir="models",
                verbose=True,
            )
            logger.info("Session calibration complete. Using session-specific models and scalers.")
            # Update config with session-specific paths
            config["MODEL_EEGNET"] = session_model_path_eegnet
            config["SCALER_EEGNET"] = session_scaler_path_eegnet
            config["MODEL_SHALLOW"] = session_model_path_shallow
            config["SCALER_SHALLOW"] = session_scaler_path_shallow
        except (FileNotFoundError, ValueError, RuntimeError):
            logger.exception("Session calibration failed. Proceeding with pre-trained models.")
    else:
        logger.info("Skipping session calibration. Using pre-trained models.")


def select_prediction_mode() -> str | None:
    """Prompt user to select prediction display mode (individual model, ensemble, or exit).

    Returns:
        str | None: Mode string ('eegnet', 'shallow', 'rf', 'xgb', 'ensemble', 'exit')

    """
    logger.info("\nChoose prediction display mode:")
    logger.info("1. EEGNet only")
    logger.info("2. ShallowConvNet only")
    logger.info("3. Random Forest only")
    logger.info("4. XGBoost only")
    logger.info("5. Ensemble (EEGNet, ShallowConvNet, Random Forest, XGBoost)")
    logger.info("6. Exit")
    while True:
        mode = input("Enter 1, 2, 3, 4, 5, or 6: ").strip()
        if mode == "1":
            return "eegnet"
        if mode == "2":
            return "shallow"
        if mode == "3":
            return "rf"
        if mode == "4":
            return "xgb"
        if mode == "5":
            return "ensemble"
        if mode == "6":
            return "exit"
        logger.warning("Invalid selection. Please enter a number from 1 to 6.")


def initialize_pipeline(config_dict: dict) -> OptimizedPredictionPipeline:
    """Initialize the prediction pipeline with the correct models and scalers.

    Args:
        config_dict (dict): Configuration dictionary.

    Returns:
        OptimizedPredictionPipeline: Initialized pipeline.

    """
    pipeline = OptimizedPredictionPipeline(config_dict)
    pipeline.load_optimized_models()
    return pipeline


def model_only_prediction(
    pipeline: OptimizedPredictionPipeline, prediction_count: int, model_name: str,
) -> int:
    """Run prediction for a single model and log the result.

    Args:
        pipeline (OptimizedPredictionPipeline): The prediction pipeline.
        prediction_count (int): Current prediction count.
        model_name (str): One of 'eegnet', 'shallow', 'rf', 'xgb'.

    Returns:
        int: Updated prediction count.

    """
    window = pipeline.get_current_window()
    if model_name == "eegnet":
        probs, confidence = pipeline.predict_eegnet(window)
    elif model_name == "shallow":
        probs, confidence = pipeline.predict_shallow(window)
    elif model_name == "rf":
        _, rf_probs, _ = pipeline.predict_tree_models(window)
        probs = rf_probs
        confidence = np.max(probs)
    elif model_name == "xgb":
        _, xgb_probs, _ = pipeline.predict_tree_models(window)
        probs = xgb_probs
        confidence = np.max(probs)
    else:
        logger.error("Unknown model: %s", model_name)
        return prediction_count
    pred_idx = np.argmax(probs)
    pred_label = pipeline.label_encoder.inverse_transform([pred_idx])[0]
    prediction_count += 1
    logger.info(
        "[%4d] %8s (conf: %.3f) [%s only]",
        prediction_count,
        pred_label.upper(),
        confidence,
        model_name.upper(),
    )
    return prediction_count


def prediction_loop(  # noqa: PLR0913
    lsl_handler: LSLStreamHandler,
    pipeline: OptimizedPredictionPipeline,
    mode: str,
    config_dict: dict,
    models: list | None = None,
    ensemble_info: dict | None = None,
    *,
    use_hard_voting: bool = False,
) -> None:
    """Run the main loop for real-time EEG prediction from LSL stream.

    Args:
        lsl_handler (LSLStreamHandler): LSL handler.
        pipeline (OptimizedPredictionPipeline): Prediction pipeline.
        mode (str): Prediction mode ('eegnet', 'shallow', 'rf', 'xgb', 'ensemble').
        config_dict (dict): Configuration dictionary.
        use_hard_voting (bool): Use hard voting for ensemble if True. Defaults to False.
        models (list): List of loaded model dicts (for ensemble mode).
        ensemble_info (dict): Ensemble info dict (for ensemble mode).

    """
    prediction_count = 0
    try:
        while True:
            window = lsl_handler.get_window(config_dict["WINDOW_SIZE"], timeout=1.0)
            if window is not None:
                for sample in window:
                    pipeline.add_sample(sample)
                if pipeline.is_ready_for_prediction():
                    if mode == "ensemble":
                        prediction_count = process_prediction(
                            pipeline,
                            prediction_count,
                            models=models,
                            ensemble_info=ensemble_info,
                            config=config_dict,
                            use_hard_voting=use_hard_voting,
                        )
                    else:
                        prediction_count = model_only_prediction(
                            pipeline, prediction_count, mode,
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
    pipeline = initialize_pipeline(config)
    # Load dynamic ensemble resources once for the session
    ensemble_info, _, models = load_realtime_resources(config)
    while True:
        mode = select_prediction_mode()
        if mode == "exit":
            logger.info("Exiting real-time prediction.")
            break
        # Reconnect LSL if needed
        if not lsl_handler.connected and not lsl_handler.connect():
            logger.error("Failed to reconnect to LSL stream. Exiting.")
            break
        use_hard_voting = False
        if mode == "ensemble":
            logger.info("\nChoose ensemble method:")
            logger.info("1. Weighted soft voting (default)")
            logger.info("2. Hard voting (majority rule, matches offline test)")
            method = input("Enter 1 or 2: ").strip()
            if method == "2":
                use_hard_voting = True
        logger.info("=== REAL-TIME PREDICTION STARTED ===")
        logger.info("Think of different directions to control the system.")
        logger.info("Press Ctrl+C to stop.")
        try:
            if mode == "ensemble":
                prediction_loop(
                    lsl_handler,
                    pipeline,
                    mode,
                    config,
                    use_hard_voting=use_hard_voting,
                    models=models,
                    ensemble_info=ensemble_info,
                )
            else:
                prediction_loop(
                    lsl_handler, pipeline, mode, config, use_hard_voting=use_hard_voting,
                )
        except KeyboardInterrupt:
            logger.info("\nPrediction stopped. Returning to menu...")
            continue


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
            result = pipeline.predict_realtime()
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


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_models_without_lsl()
    else:
        main()
