"""prediction_pipeline.py.

This module defines the `OptimizedPredictionPipeline` class for real-time EEG prediction using an ensemble of
optimized models. It provides methods for loading models, managing a rolling buffer of EEG data, extracting
features, running predictions (including dynamic ensemble voting), and tracking performance metrics.
The pipeline supports both synchronous and asynchronous prediction modes, and is designed for low-latency,
real-time applications.

"""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter, deque
from typing import Callable

import joblib
import numpy as np
from keras.models import load_model

from utils import CUSTOM_OBJECTS, extract_features, setup_logging, short_label

setup_logging()
logger = logging.getLogger(__name__)

class OptimizedPredictionPipeline:
    """Handles real-time EEG prediction using optimized models and dynamic ensemble."""

    def __init__(self, config: dict) -> None:
        """Initialize the prediction pipeline with configuration.

        Args:
            config (dict): Configuration dictionary with model/scaler paths, window size, etc.

        """
        self.config = config
        self.window_size = config["WINDOW_SIZE"]
        self.n_channels = config["N_CHANNELS"]
        self.confidence_threshold = config.get("CONFIDENCE_THRESHOLD", 0.7)

        # Use buffer size multiplier from config
        buffer_multiplier = self.config.get("BUFFER_SIZE_MULTIPLIER", 2)
        self.buffer = deque(maxlen=self.window_size * buffer_multiplier)

        # Models and preprocessors
        self.models = {}
        self.scalers = {}
        self.label_encoder = None
        self.conv1d_feature_extractor = None

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


    def load_optimized_models(self, models_metadata: list[dict] | None = None) -> None:
        """Load and optimize models for inference.

        Args:
            models_metadata (list[dict], optional): Metadata for the models to load. Defaults to None.

        """
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
            self.scalers["shallow"] = joblib.load(self.config["SCALER_SHALLOW"])
            self.label_encoder = joblib.load(self.config["LABEL_ENCODER"])

            # Conditional Conv1D feature extractor loading
            needs_conv1d = False
            if models_metadata is not None:
                for m in models_metadata:
                    if (
                        ("conv1d" in m["name"].lower() and m["type"] == "keras")
                        or ("conv1d features" in m["name"].lower())
                    ):
                        needs_conv1d = True
                        break
            self._load_conv1d_feature_extractor(needs_conv1d=needs_conv1d)

            self._warmup_models()
            logger.info("Model optimization complete.")
        except FileNotFoundError:
            logger.exception("Required model file not found.")
            logger.exception("Please run 'python train_eeg_model.py' first to train the models.")
            raise
        except Exception:
            logger.exception("Failed to load models.")
            raise

    def _load_conv1d_feature_extractor(self, *, needs_conv1d: bool) -> None:
        """Load Conv1D feature extractor if needed.

        Args:
            needs_conv1d (bool): Whether the Conv1D feature extractor is needed.

        """
        if needs_conv1d:
            conv1d_feature_path = self.config.get("CONV1D_FEATURE_EXTRACTOR")
            if not conv1d_feature_path:
                conv1d_feature_path = "models/eeg_conv1d_feature_extractor.keras"
            try:
                self.conv1d_feature_extractor = load_model(conv1d_feature_path)
                logger.info("Loaded Conv1D feature extractor from %s", conv1d_feature_path)
            except (OSError, ImportError):
                try:
                    self.conv1d_feature_extractor = load_model("models/eeg_conv1d_feature_extractor.h5")
                    logger.info("Loaded Conv1D feature extractor from fallback .h5")
                except (OSError, ImportError):
                    self.conv1d_feature_extractor = None
                    logger.info("No Conv1D feature extractor found; skipping.")
        else:
            logger.info("No Conv1D models in ensemble; skipping Conv1D feature extractor load.")

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


    def prepare_realtime_features(
        self,
        window: np.ndarray,
        config: dict,
    ) -> dict:
        """Prepare all feature representations for a single real-time EEG window.

        Args:
            window (np.ndarray): EEG window, shape (1, window_size, n_channels)
            config (dict): Configuration dictionary.

        Returns:
            dict: Mapping from representation name to data array (batch size 1).

        """
        n_channels = config["N_CHANNELS"]
        # Classic features
        x_classic_features = np.array([
            extract_features(window[0], config["SAMPLING_RATE"]),
        ])  # shape (1, n_features)
        x_classic_features_scaled = self.scalers["tree"].transform(x_classic_features)

        # Scaled window for CNNs
        x_window_flat = window.reshape(-1, n_channels)
        x_window_scaled = self.scalers["eegnet"].transform(x_window_flat).reshape(window.shape)
        # EEGNet input shape: (batch, channels, window, 1)
        x_window_eegnet = np.expand_dims(x_window_scaled, -1)
        x_window_eegnet = np.transpose(x_window_eegnet, (0, 2, 1, 3))

        # Conv1D features (if extractor exists)
        x_conv1d_features = None
        if self.conv1d_feature_extractor is not None:
            x_conv1d_features = self.conv1d_feature_extractor.predict(x_window_scaled, batch_size=1, verbose=0)
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
        config: dict,
    ) -> tuple[str, float] | None:
        """Perform dynamic ensemble prediction using all loaded models and correct features (hard voting only).

        Args:
            models (list): List of model metadata dictionaries.
            config (dict): Configuration dictionary.

        Returns:
            tuple[str, float] | None: Predicted label and confidence, or None if not ready.

        """
        if not self.is_ready_for_prediction():
            return None
        window = self.get_current_window()
        # Prepare all features for this window
        features = self.prepare_realtime_features(window, config)
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
        vote_counts = Counter(predictions.values())
        majority_pred = vote_counts.most_common(1)[0][0]
        confidence = vote_counts.most_common(1)[0][1] / len(predictions)
        predicted_label = self.label_encoder.inverse_transform([majority_pred])[0]
        self.last_prediction = predicted_label
        self.prediction_confidence = confidence
        return predicted_label, confidence


    def predict_and_log_single_model(
        self,
        selected_models: list,
        config: dict,
        prediction_count: int,
    ) -> int:
        """Predict and log for a single model (dynamic system).

        Args:
            selected_models (list): List of selected model metadata dictionaries.
            config (dict): Configuration dictionary.
            prediction_count (int): Current prediction count.

        Returns:
            int: Updated prediction count.

        """
        result = self.predict_realtime_dynamic(selected_models, config)
        if result:
            predicted_label, confidence = result
            prediction_count += 1
            logger.info(
                "[%4d] %8s (conf: %.3f) [%s only]",
                prediction_count,
                predicted_label.upper(),
                confidence,
                selected_models[0]["name"].upper(),
            )
        return prediction_count

    def predict_and_log_ensemble(self, models: list, config: dict, prediction_count: int) -> int:
        """Predict and log for ensemble mode, with per-model breakdown.

        Args:
            models (list): List of model metadata dictionaries.
            config (dict): Configuration dictionary.
            prediction_count (int): Current prediction count.

        Returns:
            int: Updated prediction count.

        """
        if not self.is_ready_for_prediction():
            return prediction_count
        window = self.get_current_window()
        features = self.prepare_realtime_features(window, config)
        model_inputs = self.map_model_inputs_realtime(models, features)
        model_results = []
        predictions = {}
        confidences = {}
        for m in models:
            name = m["name"]
            model = m["model"]
            x_input = model_inputs[name]
            if m["type"] == "keras":
                y_pred_prob = model.predict(x_input, verbose=0)
                y_pred = np.argmax(y_pred_prob, axis=1)
                pred_idx = y_pred[0]
                conf = float(np.max(y_pred_prob))
            else:
                y_pred_prob = None
                y_pred = model.predict(x_input)
                pred_idx = y_pred[0]
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(x_input)[0]
                    conf = float(np.max(proba))
                else:
                    conf = 1.0
            pred_label = self.label_encoder.inverse_transform([pred_idx])[0]
            predictions[name] = pred_idx
            confidences[name] = conf
            model_results.append(f"{name}:{short_label(pred_label)}({conf:.3f})")
        vote_counts = Counter(predictions.values())
        majority_pred = vote_counts.most_common(1)[0][0]
        confidence = vote_counts.most_common(1)[0][1] / len(predictions)
        predicted_label = self.label_encoder.inverse_transform([majority_pred])[0]
        self.last_prediction = predicted_label
        self.prediction_confidence = confidence
        prediction_count += 1
        status = "✓" if confidence > self.config["CONFIDENCE_THRESHOLD"] else "?"
        logger.info(
            "[%-4d] %s %-8s(ens:%.3f) | %s",
            prediction_count,
            status,
            predicted_label.upper(),
            confidence,
            " ".join(model_results),
        )
        return prediction_count


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
        models: list | None = None,
        config: dict | None = None,
    ) -> None:
        """Start asynchronous prediction loop in a background thread using dynamic ensemble prediction.

        Args:
            callback (Callable[[tuple[str, float]], None] | None): Callback for prediction results. Defaults to None.
            models (list): List of loaded model dicts for dynamic prediction.
            config (dict): Configuration dictionary.

        """
        if models is None or config is None:
            logger.error("Models and config must be provided for async prediction.")
            return
        def _async_prediction_loop() -> None:
            while not self.stop_thread:
                try:
                    if self.is_ready_for_prediction():
                        result = self.predict_realtime_dynamic(models, config)
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
