"""
realtime_eeg_predict.py

Real-time EEG direction prediction using LSL streaming from OpenBCI GUI.

Input: LSL stream from OpenBCI GUI (pre-filtered EEG data)
Process: Loads trained models, buffers EEG data, performs real-time ensemble prediction, logs results.
Output: Real-time predictions and confidence scores (logged and optionally displayed)
"""

import logging
import os
import threading
import time
import warnings
from collections import deque
from typing import Optional, Tuple

import joblib
import numpy as np
import tensorflow as tf
from keras.models import load_model

from lsl_stream_handler import LSLStreamHandler
from utils import (
    calibrate_all_models_lsl,  # use the new unified calibration
    extract_features,
    load_config,
    setup_logging,
)


def square(x):
    """Return the element-wise square of the input tensor."""
    return tf.math.square(x)


def log(x):
    """Return the element-wise natural logarithm of the input tensor, clipped for stability."""
    return tf.math.log(tf.clip_by_value(x, 1e-7, tf.reduce_max(x)))


# Suppress TensorFlow warnings and info messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Suppress other warnings
warnings.filterwarnings("ignore", category=UserWarning, module="scipy")
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")


setup_logging()
config = load_config()


class OptimizedPredictionPipeline:
    """
    High-performance real-time EEG prediction pipeline.

    Input: EEG samples from LSL stream (buffered in real time)
    Process: Loads and manages all models (EEGNet, ShallowConvNet, RF, XGBoost),
        preprocesses data, performs ensemble prediction, manages async prediction loop,
        tracks performance, and handles session calibration.
    Output: Real-time predictions (label, confidence), performance stats, logs.
    """

    def __init__(self, config_dict: dict):
        """
        Initialize the prediction pipeline with configuration.

        Args:
            config_dict (dict): Configuration dictionary with model/scaler paths, window size, etc.
        """
        self.config = config_dict
        self.window_size = config_dict["WINDOW_SIZE"]
        self.n_channels = config_dict["N_CHANNELS"]
        self.confidence_threshold = config_dict.get("CONFIDENCE_THRESHOLD", 0.7)

        # Use buffer size multiplier from config
        buffer_multiplier = config.get("BUFFER_SIZE_MULTIPLIER", 2)
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

    def load_optimized_models(self):
        """
        Load and optimize all models (EEGNet, ShallowConvNet, RF, XGBoost) and scalers for inference.

        Input: Model/scaler paths from config
        Process: Loads models, registers custom activations, warms up models
        Output: Models and scalers loaded into self.models and self.scalers
        """
        logging.info("Loading and optimizing models for real-time inference...")

        try:
            # CNN Models - now trained with correct window size (125)
            self.models["eegnet"] = {
                "model": load_model(self.config["MODEL_EEGNET"]),
                "optimized": False,
            }

            self.models["shallow"] = {
                "model": load_model(
                    self.config["MODEL_SHALLOW"],
                    custom_objects={"square": square, "log": log},
                ),
                "optimized": False,
            }

            # Tree-based models (already fast)
            self.models["rf"] = joblib.load(self.config["MODEL_RF"])
            self.models["xgb"] = joblib.load(self.config["MODEL_XGB"])

            # Scalers and encoders
            self.scalers["eegnet"] = joblib.load(self.config["SCALER_EEGNET"])
            self.scalers["tree"] = joblib.load(self.config["SCALER_TREE"])
            self.label_encoder = joblib.load(self.config["LABEL_ENCODER"])

            # Warm up models with dummy data
            self._warmup_models()

            logging.info("Model optimization complete.")

        except FileNotFoundError as e:
            logging.error("Required model file not found: %s", e)
            logging.error(
                "Please run 'python train_eeg_model.py' first to train the models."
            )
            raise
        except Exception as e:
            logging.error("Failed to load models: %s", e)
            raise

    def _optimize_tensorflow_model(self, model_path: str):
        """
        Attempt to optimize a Keras model for inference using TensorFlow Lite.

        Input: model_path (str) - path to Keras model file
        Process: Loads model, converts to TFLite, creates interpreter
        Output: Dict with interpreter and details, or fallback to original model
        """
        try:
            # Load original model
            if "shallow" in model_path.lower():
                model = load_model(model_path, custom_objects={"square": square})
            else:
                model = load_model(model_path)

            # Convert to TensorFlow Lite for faster inference
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

            # Optional: Use float16 quantization for even faster inference
            # converter.target_spec.supported_types = [tf.float16]

            tflite_model = converter.convert()

            # Create interpreter
            interpreter = tf.lite.Interpreter(model_content=tflite_model)
            interpreter.allocate_tensors()

            return {
                "interpreter": interpreter,
                "input_details": interpreter.get_input_details(),
                "output_details": interpreter.get_output_details(),
            }

        except (OSError, ValueError, RuntimeError) as e:
            logging.warning(
                "TensorFlow Lite optimization failed: %s. Using original model.", e
            )
            if "shallow" in model_path.lower():
                return {
                    "model": load_model(model_path, custom_objects={"square": square}),
                    "optimized": False,
                }

            return {"model": load_model(model_path), "optimized": False}

    def _warmup_models(self):
        """
        Run dummy predictions on all models to reduce first-call latency.

        Input: None (uses dummy data)
        Process: Runs one prediction on each model
        Output: None (side effect: models are warmed up)
        """
        logging.info("Warming up models...")

        # Create dummy data
        dummy_window = np.random.randn(1, self.window_size, self.n_channels)

        # Create dummy features with correct feature count
        # For tree models: 16 channels * (5 band powers + 3 stats) = 128 features
        expected_features = (
            self.n_channels * 8
        )  # 5 band powers + 3 statistical features per channel
        dummy_features = np.random.randn(1, expected_features)

        # Warm up EEGNet
        self.predict_eegnet(dummy_window)

        # Warm up ShallowConvNet
        self.predict_shallow(dummy_window)

        # Warm up tree models
        self.models["rf"].predict(dummy_features)
        self.models["xgb"].predict(dummy_features)

        logging.info("Model warmup complete.")

    def add_sample(self, sample: np.ndarray):
        """
        Add a new EEG sample to the rolling buffer.

        Input: sample (np.ndarray) - shape (n_channels,)
        Process: Appends sample to buffer
        Output: None (side effect: buffer updated)
        """
        if len(sample) != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {len(sample)}")

        self.buffer.append(sample)

    def is_ready_for_prediction(self) -> bool:
        """
        Check if the buffer has enough samples for a prediction window.

        Input: None
        Process: Compares buffer length to window size
        Output: True if ready, False otherwise
        """
        return len(self.buffer) >= self.window_size

    # Buffer management helpers
    def _get_current_window(self) -> np.ndarray:
        """
        Get the most recent window of EEG data from the buffer.

        Input: None
        Process: Slices buffer to window size
        Output: np.ndarray of shape (1, window_size, n_channels)
        """
        return np.array(list(self.buffer)[-self.window_size:]).reshape(
            (1, self.window_size, self.n_channels)
        )

    def _reset_prediction_times(self):
        """
        Clear the prediction times buffer (for performance stats).

        Input: None
        Process: Clears deque
        Output: None
        """
        self.prediction_times.clear()

    def predict_eegnet(self, window: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Run EEGNet model prediction on a window of EEG data.

        Input: window (np.ndarray) - shape (1, window_size, n_channels)
        Process: Scales, reshapes, and predicts using EEGNet (TFLite or Keras)
        Output: (probabilities, confidence)
        """
        start_time = time.time()

        try:
            # Check if model expects different window size
            if hasattr(self.models["eegnet"], "model"):
                model = self.models["eegnet"]["model"]
                expected_shape = model.input_shape

                if expected_shape[2] != self.window_size:
                    # Log the mismatch once
                    if not hasattr(self, "_shape_mismatch_logged"):
                        logging.warning(
                            "EEGNet model shape mismatch: expected %d samples, got %d. "
                            "Model needs retraining with new window size. Using zeros for now.",
                            expected_shape[2],
                            self.window_size,
                        )
                        self._shape_mismatch_logged = True

                    # Return zeros to indicate EEGNet is not working
                    return np.zeros(len(self.label_encoder.classes_)), 0.0

            # Prepare input
            window_scaled = (
                self.scalers["eegnet"]
                .transform(window.reshape(-1, self.n_channels))
                .reshape(window.shape)
            )

            window_input = np.expand_dims(window_scaled, -1)
            window_input = np.transpose(window_input, (0, 2, 1, 3))

            if "interpreter" in self.models["eegnet"]:
                # Use TensorFlow Lite for faster inference
                interpreter = self.models["eegnet"]["interpreter"]
                input_details = self.models["eegnet"]["input_details"]
                output_details = self.models["eegnet"]["output_details"]

                # Set input tensor
                interpreter.set_tensor(
                    input_details[0]["index"], window_input.astype(np.float32)
                )

                # Run inference
                interpreter.invoke()

                # Get output
                predictions = interpreter.get_tensor(output_details[0]["index"])
            else:
                # Fallback to original model
                predictions = self.models["eegnet"]["model"].predict(
                    window_input, verbose=0
                )

            confidence = np.max(predictions)
            elapsed = time.time() - start_time
            self.prediction_times.append(elapsed)

            return predictions[0], confidence

        except (ValueError, RuntimeError) as e:
            # Log error once and suppress repetitive errors
            if not hasattr(self, "_eegnet_error_logged"):
                logging.error("EEGNet prediction failed: %s", e)
                logging.error(
                    "EEGNet will be disabled for this session."
                    "Please retrain models with correct window size."
                )
                self._eegnet_error_logged = True
            return np.zeros(len(self.label_encoder.classes_)), 0.0

    def predict_shallow(self, window: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Run ShallowConvNet model prediction on a window of EEG data.

        Input: window (np.ndarray) - shape (1, window_size, n_channels)
        Process: Scales, reshapes, and predicts using ShallowConvNet
        Output: (probabilities, confidence)
        """
        start_time = time.time()

        try:
            # Check if model expects different window size
            if hasattr(self.models["shallow"], "model"):
                model = self.models["shallow"]["model"]
                expected_shape = model.input_shape

                if expected_shape[2] != self.window_size:
                    # Log the mismatch once
                    if not hasattr(self, "_shallow_shape_mismatch_logged"):
                        logging.warning(
                            "ShallowConvNet model shape mismatch: expected %d samples, got %d. "
                            "Model needs retraining with new window size. Using zeros for now.",
                            expected_shape[2],
                            self.window_size,
                        )
                        self._shallow_shape_mismatch_logged = True

                    # Return zeros to indicate ShallowConvNet is not working
                    return np.zeros(len(self.label_encoder.classes_)), 0.0

            # Prepare input (same format as EEGNet)
            window_scaled = (
                self.scalers["eegnet"]
                .transform(window.reshape(-1, self.n_channels))
                .reshape(window.shape)
            )

            window_input = np.expand_dims(window_scaled, -1)
            window_input = np.transpose(window_input, (0, 2, 1, 3))

            # Use the model directly (ShallowConvNet doesn't have TFLite optimization yet)
            predictions = self.models["shallow"]["model"].predict(
                window_input, verbose=0
            )

            confidence = np.max(predictions)
            elapsed = time.time() - start_time
            self.prediction_times.append(elapsed)

            return predictions[0], confidence

        except (ValueError, RuntimeError) as e:
            # Log error once and suppress repetitive errors
            if not hasattr(self, "_shallow_error_logged"):
                logging.error("ShallowConvNet prediction failed: %s", e)
                logging.error(
                    "ShallowConvNet will be disabled for this session."
                    "Please retrain models with correct window size."
                )
                self._shallow_error_logged = True
            return np.zeros(len(self.label_encoder.classes_)), 0.0

    def predict_tree_models(
        self, window: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run Random Forest and XGBoost predictions on a window of EEG data.

        Input: window (np.ndarray) - shape (1, window_size, n_channels)
        Process: Extracts features, scales, predicts with both models
        Output: (rf_probs, xgb_probs, confidence)
        """
        start_time = time.time()

        try:
            # Extract features for tree models
            features = extract_features(window.squeeze(), self.config["SAMPLING_RATE"])
            features = features.reshape(1, -1)

            # Scale features
            features_scaled = self.scalers["tree"].transform(features)

            # Get predictions from both tree models
            rf_proba = self.models["rf"].predict_proba(features_scaled)[0]
            xgb_proba = self.models["xgb"].predict_proba(features_scaled)[0]

            # Average probabilities for ensemble
            avg_proba = (rf_proba + xgb_proba) / 2
            confidence = np.max(avg_proba)

            elapsed = time.time() - start_time
            self.prediction_times.append(elapsed)

            return rf_proba, xgb_proba, confidence

        except (ValueError, RuntimeError) as e:
            logging.error("Tree model prediction failed: %s", e)
            num_classes = len(self.label_encoder.classes_)
            return np.zeros(num_classes), np.zeros(num_classes), 0.0

    def predict_realtime(self) -> Optional[Tuple[str, float]]:
        """
        Perform ensemble prediction using all available models on the current buffer.

        Input: None (uses buffer)
        Process: Gets predictions from all models, applies weighted ensemble, applies confidence threshold
        Output: (predicted_label, confidence) or None if not enough data
        """
        if not self.is_ready_for_prediction():
            return None
        window = self._get_current_window()
        eeg_probs, _ = self.predict_eegnet(window)
        shallow_probs, _ = self.predict_shallow(window)
        rf_probs, xgb_probs, _ = self.predict_tree_models(window)

        # Ensemble strategy: Use weighted voting based on model availability and confidence
        available_models = []
        model_weights = []
        model_predictions = []

        # Check which CNN models are working
        if not np.all(eeg_probs == 0):
            available_models.append("EEGNet")
            model_weights.append(0.4)  # Higher weight for CNN models
            model_predictions.append(eeg_probs)

        if not np.all(shallow_probs == 0):
            available_models.append("ShallowConvNet")
            model_weights.append(0.4)  # Higher weight for CNN models
            model_predictions.append(shallow_probs)

        # Tree models (always available)
        available_models.extend(["RandomForest", "XGBoost"])
        model_weights.extend([0.1, 0.1])  # Lower weights for tree models
        model_predictions.extend([rf_probs, xgb_probs])

        # Normalize weights
        total_weight = sum(model_weights)
        model_weights = [w / total_weight for w in model_weights]

        # Weighted ensemble
        final_probs = np.zeros_like(rf_probs)
        for probs, weight in zip(model_predictions, model_weights):
            final_probs += probs * weight

        confidence = np.max(final_probs)

        # Apply confidence threshold
        if confidence < self.confidence_threshold:
            return "neutral", confidence

        # Get prediction
        predicted_idx = np.argmax(final_probs)
        predicted_label = self.label_encoder.inverse_transform([predicted_idx])[0]

        self.last_prediction = predicted_label
        self.prediction_confidence = confidence

        return predicted_label, confidence

    def get_performance_stats(self) -> dict:
        """
        Get average latency, FPS, buffer size, and last confidence for real-time predictions.

        Input: None
        Process: Computes stats from prediction_times and buffer
        Output: Dict with stats
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

    def start_async_prediction(self, callback=None):
        """
        Start asynchronous prediction loop in a background thread.

        Input: Optional callback function for prediction results
        Process: Runs prediction loop, calls callback on new predictions
        Output: None (side effect: thread started)
        """

        def _async_prediction_loop():
            while not self.stop_thread:
                try:
                    if self.is_ready_for_prediction():
                        result = self.predict_realtime()
                        if result and callback:
                            callback(result)
                        self.prediction_ready.set()
                except RuntimeError as e:
                    logging.error("Unexpected runtime error in prediction loop: %s", e, exc_info=True)

                time.sleep(0.001)  # Small delay to prevent busy waiting

        self.prediction_thread = threading.Thread(
            target=_async_prediction_loop, daemon=True
        )
        self.prediction_thread.start()
        logging.info("Asynchronous prediction started.")

    def stop_async_prediction(self):
        """
        Stop the asynchronous prediction thread.

        Input: None
        Process: Signals thread to stop and joins it
        Output: None
        """
        self.stop_thread = True
        if self.prediction_thread:
            self.prediction_thread.join(timeout=1.0)
        logging.info("Asynchronous prediction stopped.")


def process_prediction(pipeline, prediction_count):
    """
    Process a single ensemble prediction and log detailed model breakdown.

    Input: pipeline (OptimizedPredictionPipeline), prediction_count (int)
    Process: Runs ensemble prediction, logs results, prints model-wise breakdown
    Output: Updated prediction_count
    """

    def fmt_model(label, conf, disabled):
        if disabled:
            return "---(---)"
        return f"{label:<3}({conf:.3f})"

    result = pipeline.predict_realtime()
    if result:
        predicted_label, confidence = result
        prediction_count += 1
        status = "✓" if confidence > config["CONFIDENCE_THRESHOLD"] else "?"

        # Get individual model predictions for detailed output
        window = np.array(list(pipeline.buffer)[-pipeline.window_size:])
        window = window.reshape((1, pipeline.window_size, pipeline.n_channels))

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
        logging.info(
            "[%-4d] %s %-8s (ens:%.3f) | EEG:%s SH:%s RF:%s XGB:%s",
            prediction_count,
            status,
            predicted_label.upper(),
            confidence,
            fmt_model(eeg_label, eeg_confidence, eeg_disabled),
            fmt_model(shallow_label, shallow_confidence, shallow_disabled),
            fmt_model(rf_label, rf_conf, False),
            fmt_model(xgb_label, xgb_conf, False),
        )

        if prediction_count % 50 == 0:
            stats = pipeline.get_performance_stats()
            logging.info(
                "Performance: %.1fms avg, %.1f FPS",
                stats["avg_latency_ms"],
                stats["fps"],
            )
    return prediction_count


def add_samples_to_buffer(pipeline, window):
    """
    Add all samples from a window to the pipeline buffer.

    Input: pipeline (OptimizedPredictionPipeline), window (np.ndarray)
    Process: Iterates over window, adds each sample
    Output: None
    """
    for sample in window:
        pipeline.add_sample(sample)


def session_calibration(lsl_handler):
    """
    Handle session calibration logic and return calibration status and model/scaler paths.

    Input: lsl_handler (LSLStreamHandler)
    Process: Prompts user, runs unified calibration, saves session models/scalers
    Output: Tuple (
        use_session_model,
        session_model_path_eegnet,
        session_scaler_path_eegnet,
        session_model_path_shallow,
        session_scaler_path_shallow,
    )
    """
    session_model_path_eegnet = "models/eeg_direction_model_session.h5"
    session_scaler_path_eegnet = "models/eeg_scaler_session.pkl"
    session_model_path_shallow = "models/eeg_shallow_model_session.h5"
    session_scaler_path_shallow = "models/eeg_scaler_shallow_session.pkl"
    use_session_model = False

    user_calib = (
        input("Would you like to calibrate for this session? (Y/n): ").strip().lower()
    )
    if user_calib in ("", "y", "yes"):
        try:
            logging.info("Starting session calibration. Please follow the prompts.")
            calibrate_all_models_lsl(
                lsl_stream_handler=lsl_handler,
                config_path="config.json",
                save_dir="models",
                verbose=True,
            )
            logging.info(
                "Session calibration complete. Using session-specific models and scalers."
            )
            use_session_model = True
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            logging.error(
                "Session calibration failed: %s. Proceeding with pre-trained models.", e
            )
            use_session_model = False
    else:
        logging.info("Skipping session calibration. Using pre-trained models.")
    return (
        use_session_model,
        session_model_path_eegnet,
        session_scaler_path_eegnet,
        session_model_path_shallow,
        session_scaler_path_shallow,
    )


def select_prediction_mode():
    """
    Prompt user to select prediction display mode (EEGNet only or ensemble).

    Input: None (user input)
    Process: Prints options, reads user input
    Output: True if ensemble mode, False if EEGNet only
    """
    print("\nChoose prediction display mode:")
    print("1. EEGNet only")
    print("2. Ensemble (EEGNet, ShallowConvNet, Random Forest, XGBoost)")
    mode = input("Enter 1 or 2: ").strip()
    return mode == "2"


def initialize_pipeline(
    config_dict,
    use_session_model,
    session_model_path_eegnet,
    session_scaler_path_eegnet,
    session_model_path_shallow,
    session_scaler_path_shallow,
):
    """
    Initialize the prediction pipeline with the correct models and scalers.

    Input: config_dict (dict), use_session_model (bool), session model/scaler paths
    Process: Loads session-specific or pre-trained models/scalers as needed
    Output: Initialized OptimizedPredictionPipeline
    """
    pipeline = OptimizedPredictionPipeline(config_dict)
    if use_session_model:
        try:
            pipeline.models["eegnet"] = {
                "model": load_model(session_model_path_eegnet),
                "optimized": False,
            }
            pipeline.scalers["eegnet"] = joblib.load(session_scaler_path_eegnet)
            pipeline.models["shallow"] = {
                "model": load_model(
                    session_model_path_shallow,
                    custom_objects={"square": square, "log": log},
                ),
                "optimized": False,
            }
            pipeline.scalers["shallow"] = joblib.load(session_scaler_path_shallow)
            logging.info(
                "Loaded session-specific EEGNet and ShallowConvNet models and scalers."
            )
        except (FileNotFoundError, ValueError) as e:
            logging.error(
                "Failed to load session-specific model/scaler: %s. Using pre-trained.",
                e,
            )
            pipeline.load_optimized_models()
    else:
        pipeline.load_optimized_models()
    return pipeline


def eegnet_only_prediction(pipeline, prediction_count):
    """
    Run EEGNet-only prediction and log the result.

    Input: pipeline (OptimizedPredictionPipeline), prediction_count (int)
    Process: Runs EEGNet prediction, logs result
    Output: Updated prediction_count
    """
    window = np.array(list(pipeline.buffer)[-pipeline.window_size:]).reshape(
        (1, pipeline.window_size, pipeline.n_channels)
    )
    result = pipeline.predict_eegnet(window)
    if result:
        probs, confidence = result
        pred_idx = np.argmax(probs)
        pred_label = pipeline.label_encoder.inverse_transform([pred_idx])[0]
        prediction_count += 1
        logging.info(
            "[%4d] %8s (conf: %.3f) [EEGNet only]",
            prediction_count,
            pred_label.upper(),
            confidence,
        )
    return prediction_count


def prediction_loop(lsl_handler, pipeline, show_ensemble, config_dict):
    """
    Main loop for real-time EEG prediction from LSL stream.

    Input:
        lsl_handler (LSLStreamHandler), pipeline (OptimizedPredictionPipeline),
        show_ensemble (bool), config_dict (dict)
    Process: Continuously collects windows, adds to buffer, runs predictions, logs results
    Output: None (side effect: predictions and logs)
    """
    prediction_count = 0
    try:
        while True:
            window = lsl_handler.get_window(config_dict["WINDOW_SIZE"], timeout=1.0)
            if window is not None:
                add_samples_to_buffer(pipeline, window)
                if pipeline.is_ready_for_prediction():
                    if show_ensemble:
                        prediction_count = process_prediction(
                            pipeline, prediction_count
                        )
                    else:
                        prediction_count = eegnet_only_prediction(
                            pipeline, prediction_count
                        )
            time.sleep(0.001)
    except KeyboardInterrupt:
        logging.info("Stopping real-time prediction...")
    finally:
        lsl_handler.disconnect()
        pipeline.stop_async_prediction()
        stats = pipeline.get_performance_stats()
        logging.info("=== FINAL PERFORMANCE REPORT ===")
        logging.info("Average latency: %.1fms", stats["avg_latency_ms"])
        logging.info("Average FPS: %.1f", stats["fps"])
        logging.info("Total predictions: %d", prediction_count)


def main():
    """
    Main entry point for real-time EEG prediction using LSL streaming and optional session calibration.

    Input: None (uses config and user input)
    Process: Connects to LSL, handles calibration, initializes pipeline, runs prediction loop
    Output: None (side effect: predictions and logs)
    """
    logging.info("Starting LSL-based real-time EEG prediction...")

    lsl_handler = LSLStreamHandler(
        stream_name=config["LSL_STREAM_NAME"], timeout=config["LSL_TIMEOUT"]
    )

    if not lsl_handler.connect():
        logging.error(
            "Failed to connect to LSL stream. Make sure OpenBCI GUI is running with LSL streaming."
        )
        return

    (
        use_session_model,
        session_model_path_eegnet,
        session_scaler_path_eegnet,
        session_model_path_shallow,
        session_scaler_path_shallow,
    ) = session_calibration(lsl_handler)
    show_ensemble = select_prediction_mode()
    pipeline = initialize_pipeline(
        config,
        use_session_model,
        session_model_path_eegnet,
        session_scaler_path_eegnet,
        session_model_path_shallow,
        session_scaler_path_shallow,
    )

    logging.info("=== REAL-TIME PREDICTION STARTED ===")
    logging.info("Think of different directions to control the system.")
    logging.info("Press Ctrl+C to stop.")

    prediction_loop(lsl_handler, pipeline, show_ensemble, config)


def test_models_without_lsl():
    """
    Test model loading and prediction logic without requiring LSL stream (for development).

    Input: None (uses config)
    Process: Loads models, runs dummy prediction, logs results
    Output: True if successful, False otherwise
    """
    logging.info("Testing model loading and prediction (no LSL required)...")

    try:
        # Initialize pipeline
        pipeline = OptimizedPredictionPipeline(config)
        pipeline.load_optimized_models()

        # Create fake EEG data
        fake_window = np.random.randn(config["WINDOW_SIZE"], config["N_CHANNELS"]) * 0.1

        # Add samples to buffer
        for sample in fake_window:
            pipeline.add_sample(sample)

        # Test prediction
        if pipeline.is_ready_for_prediction():
            result = pipeline.predict_realtime()
            if result:
                label, confidence = result
                logging.info(
                    "✅ Test prediction: %s (confidence: %.3f)", label, confidence
                )
            else:
                logging.info("✅ Models loaded successfully but no prediction made")

        # Test performance stats
        stats = pipeline.get_performance_stats()
        logging.info("✅ Performance stats: %s", stats)

        logging.info("✅ All models loaded and tested successfully!")
        return True

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logging.error("❌ Model test failed: %s", e)
        return False


if __name__ == "__main__":
    import sys

    # Check for test mode
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_models_without_lsl()
    else:
        main()
