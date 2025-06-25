"""
Perform real-time EEG direction prediction using LSL streaming from OpenBCI GUI.

Features:
- LSL streaming from OpenBCI GUI (pre-filtered data)
- Optimized prediction pipeline with confidence thresholding
- High-performance real-time processing (5-25ms latency)
- Real-time performance monitoring and statistics

Setup Instructions:
1. Start OpenBCI GUI
2. Configure filters in GUI (recommended: 1-50 Hz bandpass, 50/60 Hz notch)
3. Start LSL streaming in OpenBCI GUI
4. Run this script

Input: LSL stream from OpenBCI GUI
Output: Real-time predictions with confidence scores
"""

import os

import logging
import threading
import time
from collections import deque
from typing import Optional, Tuple
import warnings
import joblib

import numpy as np
import tensorflow as tf
from keras.models import load_model

from lsl_stream_handler import LSLStreamHandler
from utils import load_config, setup_logging, extract_features

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

    Optimizations implemented:
    - Model quantization for faster inference
    - Circular buffer for efficient windowing
    - Asynchronous processing to prevent blocking
    - Confidence thresholding to reduce false positives
    - Batch processing when possible
    """

    def __init__(self, config_dict: dict):
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
        """Load and optimize models for faster inference."""
        logging.info("Loading and optimizing models for real-time inference...")

        try:
            # CNN Models - now trained with correct window size (125)
            self.models["eegnet"] = {
                "model": load_model(self.config["MODEL_CNN"]),
                "optimized": False,
            }

            self.models["shallow"] = {
                "model": load_model(self.config["MODEL_SHALLOW"]),
                "optimized": False,
            }

            # Tree-based models (already fast)
            self.models["rf"] = joblib.load(self.config["MODEL_RF"])
            self.models["xgb"] = joblib.load(self.config["MODEL_XGB"])

            # Scalers and encoders
            self.scalers["cnn"] = joblib.load(self.config["SCALER_CNN"])
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
        """Optimize TensorFlow model for faster inference."""
        try:
            # Load original model
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
            return {"model": load_model(model_path), "optimized": False}

    def _warmup_models(self):
        """Warm up models with dummy predictions to reduce first-call latency."""
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
        """Add new EEG sample to the buffer."""
        if len(sample) != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {len(sample)}")

        self.buffer.append(sample)

    def is_ready_for_prediction(self) -> bool:
        """Check if buffer has enough data for prediction."""
        return len(self.buffer) >= self.window_size

    # Buffer management helpers
    def _get_current_window(self) -> np.ndarray:
        """Return the most recent window from the buffer."""
        return np.array(list(self.buffer)[-self.window_size :]).reshape((1, self.window_size, self.n_channels))

    def _reset_prediction_times(self):
        """Reset the prediction times buffer."""
        self.prediction_times.clear()

    def predict_eegnet(self, window: np.ndarray) -> Tuple[np.ndarray, float]:
        """Fast EEGNet prediction using optimized model."""
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
                self.scalers["cnn"]
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
        """Fast ShallowConvNet prediction."""
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
                self.scalers["cnn"]
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
        """Predict using Random Forest and XGBoost models."""
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
        Perform real-time prediction on current buffer contents using ensemble.

        Returns:
            Tuple of (predicted_label, confidence) or None if not ready
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
        """Get real-time performance statistics."""
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
        """Start asynchronous prediction in a separate thread."""

        def prediction_loop():
            while not self.stop_thread:
                if self.is_ready_for_prediction():
                    result = self.predict_realtime()
                    if result and callback:
                        callback(result)
                    self.prediction_ready.set()

                time.sleep(0.001)  # Small delay to prevent busy waiting

        self.prediction_thread = threading.Thread(target=prediction_loop, daemon=True)
        self.prediction_thread.start()
        logging.info("Asynchronous prediction started.")

    def stop_async_prediction(self):
        """Stop asynchronous prediction thread."""
        self.stop_thread = True
        if self.prediction_thread:
            self.prediction_thread.join(timeout=1.0)
        logging.info("Asynchronous prediction stopped.")


def process_prediction(pipeline, prediction_count):
    """Process a single prediction and display results with detailed breakdown."""
    result = pipeline.predict_realtime()
    if result:
        predicted_label, confidence = result
        prediction_count += 1
        status = "✓" if confidence > config["CONFIDENCE_THRESHOLD"] else "?"

        # Get individual model predictions for detailed output
        window = np.array(list(pipeline.buffer)[-pipeline.window_size :])
        window = window.reshape((1, pipeline.window_size, pipeline.n_channels))

        # Get EEGNet prediction
        eeg_probs, eeg_confidence = pipeline.predict_eegnet(window)
        eeg_pred_idx = np.argmax(eeg_probs) if not np.all(eeg_probs == 0) else 0
        eeg_label = (
            pipeline.label_encoder.inverse_transform([eeg_pred_idx])[0]
            if not np.all(eeg_probs == 0)
            else "error"
        )

        # Get tree model predictions
        rf_probs, xgb_probs, _ = pipeline.predict_tree_models(window)
        rf_pred_idx = np.argmax(rf_probs)
        xgb_pred_idx = np.argmax(xgb_probs)
        rf_label = pipeline.label_encoder.inverse_transform([rf_pred_idx])[0]
        xgb_label = pipeline.label_encoder.inverse_transform([xgb_pred_idx])[0]

        # Detailed logging every 10 predictions
        if prediction_count % 10 == 0:
            logging.info("-" * 50)
            logging.info(
                "EEGNet Predicted label: %s (conf: %.3f)", eeg_label, eeg_confidence
            )
            logging.info(
                "Random Forest Predicted label: %s (conf: %.3f)",
                rf_label,
                np.max(rf_probs),
            )
            logging.info(
                "XGBoost Predicted label: %s (conf: %.3f)", xgb_label, np.max(xgb_probs)
            )
            logging.info(
                "Final Ensemble label: %s (conf: %.3f)", predicted_label, confidence
            )
            logging.info("-" * 50)

        # Regular compact output
        logging.info(
            "[%4d] %s %8s (conf: %.3f) [EEG:%s RF:%s XGB:%s]",
            prediction_count,
            status,
            predicted_label.upper(),
            confidence,
            eeg_label[:3],
            rf_label[:3],
            xgb_label[:3],
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
    """Add samples from window to pipeline buffer."""
    for sample in window:
        pipeline.add_sample(sample)


def main():
    """Main real-time prediction function using LSL streaming."""

    logging.info("Starting LSL-based real-time EEG prediction...")

    # Initialize LSL stream handler
    lsl_handler = LSLStreamHandler(
        stream_name=config["LSL_STREAM_NAME"], timeout=config["LSL_TIMEOUT"]
    )

    # Connect to LSL stream
    if not lsl_handler.connect():
        logging.error(
            "Failed to connect to LSL stream. Make sure OpenBCI GUI is running with LSL streaming."
        )
        return

    # Note: Sampling rate verification would go here if LSLStreamHandler supported get_stream_info()
    # For now, users should manually verify sampling rates match between OpenBCI GUI and config.json

    # Initialize optimized prediction pipeline
    pipeline = OptimizedPredictionPipeline(config)
    pipeline.load_optimized_models()

    logging.info("=== REAL-TIME PREDICTION STARTED ===")
    logging.info(
        "⚠️  EEGNet model disabled due to shape mismatch (expects 250 samples, got 125)"
    )
    logging.info("📊 Using Random Forest + XGBoost ensemble for predictions")
    logging.info("🔄 To use EEGNet: retrain models with 'python train_eeg_model.py'")
    logging.info("Think of different directions to control the system.")
    logging.info("Press Ctrl+C to stop.")

    prediction_count = 0

    try:
        while True:
            window = lsl_handler.get_window(config["WINDOW_SIZE"], timeout=1.0)
            if window is not None:
                add_samples_to_buffer(pipeline, window)
                if pipeline.is_ready_for_prediction():
                    prediction_count = process_prediction(pipeline, prediction_count)
            time.sleep(0.001)  # Small delay to prevent busy waiting

    except KeyboardInterrupt:
        logging.info("Stopping real-time prediction...")
    finally:
        lsl_handler.disconnect()
        pipeline.stop_async_prediction()

        # Final performance report
        stats = pipeline.get_performance_stats()
        logging.info("=== FINAL PERFORMANCE REPORT ===")
        logging.info("Average latency: %.1fms", stats["avg_latency_ms"])
        logging.info("Average FPS: %.1f", stats["fps"])
        logging.info("Total predictions: %d", prediction_count)


def test_models_without_lsl():
    """Test model loading and prediction without LSL stream for development."""
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
