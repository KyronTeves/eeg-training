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
import os
import threading
import time
import warnings
from collections import Counter, deque
from typing import TYPE_CHECKING

import joblib
import numpy as np
from keras.models import load_model

from lsl_stream_handler import LSLStreamHandler
from utils import (
    CUSTOM_OBJECTS,
    calibrate_all_models_lsl,  # use the new unified calibration
    extract_features,
    handle_errors,  # Add error handler
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


def process_prediction(
    pipeline: OptimizedPredictionPipeline,
    prediction_count: int,
    *,
    use_hard_voting: bool = False,
) -> int:
    """Process a single ensemble prediction and log detailed model breakdown.

    Args:
        pipeline (OptimizedPredictionPipeline): The prediction pipeline.
        prediction_count (int): The current prediction count.
        use_hard_voting (bool): Whether to use hard voting.

    Returns:
        int: Updated prediction count.

    """

    def fmt_model(label: str, conf: float, *, disabled: bool) -> str:
        """Format the model prediction output for display.

        Args:
            label (str): The predicted label for the model.
            conf (float): The confidence score for the prediction.
            disabled (bool): Whether the model is disabled (e.g., due to errors).

        Returns:
            str: Formatted string for display.

        """
        if disabled:
            return "---(---)"
        return f"{short_label(label):<3}({conf:.3f})"

    result = pipeline.predict_realtime(use_hard_voting=use_hard_voting)
    if result:
        predicted_label, confidence = result
        prediction_count += 1
        status = "✓" if confidence > pipeline.config["CONFIDENCE_THRESHOLD"] else "?"

        # Get individual model predictions for detailed output
        window = pipeline.get_current_window()

        # EEGNet
        eeg_probs, eeg_confidence = pipeline.predict_eegnet(window)
        eeg_disabled = np.all(eeg_probs == 0)
        eeg_pred_idx = np.argmax(eeg_probs) if not eeg_disabled else 0
        eeg_label = (
            pipeline.label_encoder.inverse_transform([eeg_pred_idx])[0]
            if not eeg_disabled
            else "---"
        )

        # ShallowConvNet
        shallow_probs, shallow_confidence = pipeline.predict_shallow(window)
        shallow_disabled = np.all(shallow_probs == 0)
        shallow_pred_idx = np.argmax(shallow_probs) if not shallow_disabled else 0
        shallow_label = (
            pipeline.label_encoder.inverse_transform([shallow_pred_idx])[0]
            if not shallow_disabled
            else "---"
        )

        # Tree models
        rf_probs, xgb_probs, _ = pipeline.predict_tree_models(window)
        rf_pred_idx = np.argmax(rf_probs)
        xgb_pred_idx = np.argmax(xgb_probs)
        rf_label = pipeline.label_encoder.inverse_transform([rf_pred_idx])[0]
        xgb_label = pipeline.label_encoder.inverse_transform([xgb_pred_idx])[0]
        rf_conf = np.max(rf_probs)
        xgb_conf = np.max(xgb_probs)

        # Neat, aligned output
        logger.info(
            "[%-4d] %s %-8s(ens:%.3f) | EEG:%s SH:%s RF:%s XGB:%s",
            prediction_count,
            status,
            predicted_label.upper(),
            confidence,
            fmt_model(eeg_label, eeg_confidence, disabled=eeg_disabled),
            fmt_model(shallow_label, shallow_confidence, disabled=shallow_disabled),
            fmt_model(rf_label, rf_conf, disabled=False),
            fmt_model(xgb_label, xgb_conf, disabled=False),
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


def prediction_loop(
    lsl_handler: LSLStreamHandler,
    pipeline: OptimizedPredictionPipeline,
    mode: str,
    config_dict: dict,
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
                            pipeline, prediction_count, use_hard_voting=use_hard_voting,
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
