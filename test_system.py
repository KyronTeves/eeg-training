"""
Unit tests for EEG training system utility functions.

This module contains comprehensive unit tests for the EEG training pipeline,
including tests for:
- Data windowing functionality
- Configuration loading and validation
- Data validation utilities (NaN checks, label validation)
- Model training integration tests
- Pipeline integration tests for end-to-end functionality

The tests ensure the reliability and correctness of the EEG data processing
and model training workflow.
"""

import os
import subprocess  # nosec B404: subprocess is used safely with explicit argument lists and no shell=True
import sys
import tempfile
import json

import numpy as np
import pandas as pd
import pytest
from keras.models import load_model

from EEGModels import EEGNet, ShallowConvNet
from utils import check_labels_valid, check_no_nan, load_config, window_data


# Add a default timeout for all subprocess.run calls
DEFAULT_TIMEOUT = 30  # seconds


def test_window_data_shape():
    """Test that window_data returns correct shapes."""
    x = np.random.randn(1000, 16)
    y = np.random.choice(["left", "right", "neutral"], size=(1000, 1))
    x_windows, y_windows = window_data(x, y, window_size=250, step_size=125)
    assert x_windows.shape[1:] == (250, 16)
    assert x_windows.shape[0] == y_windows.shape[0]


def test_window_data_output():
    """Test window_data produces correct windowed output and label consistency."""
    x = np.random.randn(500, 8)  # 500 samples, 8 channels
    y = np.random.choice(["left", "right"], size=(500, 1))
    window_size = 100
    step_size = 50
    x_windows, y_windows = window_data(x, y, window_size, step_size)
    # Check window shape
    assert x_windows.shape[1:] == (window_size, 8)
    # Check number of windows
    expected_windows = (500 - window_size) // step_size + 1
    assert x_windows.shape[0] == expected_windows
    assert y_windows.shape[0] == expected_windows
    # Check that each window label is one of the original labels
    for label in y_windows:
        assert label in ["left", "right"]


def test_load_config_keys():
    """Test that config contains required keys."""
    config = load_config()
    required_keys = [
        "N_CHANNELS",
        "WINDOW_SIZE",
        "STEP_SIZE",
        "OUTPUT_CSV",
        "WINDOWED_NPY",
        "WINDOWED_LABELS_NPY",
        "SESSION_TYPES",
    ]
    for key in required_keys:
        assert key in config


# ----------------------
# Data Validation Tests
# ----------------------

@pytest.mark.parametrize(
    "arr,should_raise",
    [
        (np.zeros((10, 10)), False),
        (lambda: (a := np.zeros((5, 5)), a.__setitem__((0, 0), np.nan), a)[-1], True),
    ],
)
def test_check_no_nan_cases(arr, should_raise):
    if callable(arr):
        arr = arr()
    if should_raise:
        with pytest.raises(ValueError):
            check_no_nan(arr)
    else:
        check_no_nan(arr)


@pytest.mark.parametrize(
    "labels,should_raise",
    [
        (np.array(["left", "right", "neutral"]), False),
        (np.array(["left", np.nan, "right"], dtype=object), True),
        (np.array(["left", "up", "right"]), True),
    ],
)
def test_check_labels_valid_cases(labels, should_raise):
    if should_raise:
        with pytest.raises(ValueError):
            check_labels_valid(labels, valid_labels=["left", "right", "neutral"])
    else:
        check_labels_valid(labels, valid_labels=["left", "right", "neutral"])


# ----------------------
# Model Training & IO Tests
# ----------------------


@pytest.mark.parametrize(
    "model_class, model_name",
    [
        (EEGNet, "eegnet"),
        (ShallowConvNet, "shallow"),
    ],
)
def test_model_train_save_load(model_class, model_name):
    """Test that EEGNet and ShallowConvNet can be trained, saved, and loaded on dummy data."""
    x = np.random.randn(20, 16, 250, 1)  # (batch, channels, samples, 1)
    y = np.zeros((20,))
    y[:10] = 1  # Two classes
    y_cat = np.zeros((20, 2))
    y_cat[np.arange(20), y.astype(int)] = 1
    model = model_class(nb_classes=2, Chans=16, Samples=250)
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(x, y_cat, epochs=1, batch_size=4, verbose=0)
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, f"test_{model_name}_model.h5")
        model.save(model_path)
        assert os.path.exists(model_path)
        loaded = load_model(model_path)
        assert loaded is not None


# ----------------------
# Integration & Pipeline Tests
# ----------------------
class TestIntegration:
    def test_end_to_end_pipeline(self):
        """Black-box end-to-end test: CSV -> windowed .npy -> model file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Create synthetic CSV
            csv_path = os.path.join(tmpdir, "eeg.csv")
            n_samples, n_channels = 200, 4
            data = np.random.randn(n_samples, n_channels)
            labels = np.random.choice(["left", "right"], size=n_samples)
            df = pd.DataFrame(data, columns=[f"ch_{i}" for i in range(n_channels)])
            df["session_type"] = "pure"
            df["label"] = labels
            df.to_csv(csv_path, index=False)
            # 2. Create minimal config.json
            config_path = os.path.join(tmpdir, "config.json")
            config = {
                "N_CHANNELS": n_channels,
                "WINDOW_SIZE": 50,
                "STEP_SIZE": 25,
                "OUTPUT_CSV": csv_path,
                "WINDOWED_NPY": os.path.join(tmpdir, "X.npy"),
                "WINDOWED_LABELS_NPY": os.path.join(tmpdir, "y.npy"),
                "SESSION_TYPES": ["pure"],
                "USE_SESSION_TYPES": ["pure"],
                "LABELS": ["left", "right"],
                "MODEL_EEGNET": os.path.join(tmpdir, "model.h5"),
                "MODEL_SHALLOW": os.path.join(tmpdir, "shallow.h5"),
                "LABEL_ENCODER": os.path.join(tmpdir, "le.pkl"),
                "SCALER_EEGNET": os.path.join(tmpdir, "scaler.pkl"),
                "SCALER_SHALLOW": os.path.join(tmpdir, "scaler_shallow.pkl"),
                "LABEL_CLASSES_NPY": os.path.join(tmpdir, "classes.npy"),
            }
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f)
            # 3. Run windowing script
            subprocess.run(
                [sys.executable, "window_eeg_data.py"],
                env={**os.environ, "CONFIG_PATH": config_path},
                capture_output=True,
                text=True,
                check=True,
                timeout=DEFAULT_TIMEOUT,
            )  # nosec B603
            assert os.path.exists(config["WINDOWED_NPY"])
            assert os.path.exists(config["WINDOWED_LABELS_NPY"])
            # 4. Run training script
            subprocess.run(
                [sys.executable, "train_eeg_model.py"],
                env={**os.environ, "CONFIG_PATH": config_path},
                capture_output=True,
                text=True,
                check=True,
                timeout=DEFAULT_TIMEOUT,
            )  # nosec B603
            assert os.path.exists(config["MODEL_EEGNET"])
            assert os.path.exists(config["MODEL_SHALLOW"])
            assert os.path.exists(config["LABEL_ENCODER"])
            assert os.path.exists(config["SCALER_EEGNET"])
            assert os.path.exists(config["SCALER_SHALLOW"])
            # Additional: Load ShallowConvNet model and run a prediction to ensure it is functional
            from keras.models import load_model

            def square(x):
                import tensorflow as tf

                return tf.math.square(x)

            x = np.load(config["WINDOWED_NPY"])
            if x.ndim == 3:
                x = x[..., np.newaxis]
            shallow_model = load_model(
                config["MODEL_SHALLOW"], custom_objects={"square": square}
            )
            shallow_preds = shallow_model.predict(x[:5])
            assert shallow_preds.shape[0] == 5

    def test_error_handling_missing_config(self):
        """Test error handling for missing config file in windowing and training scripts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # No config file created
            env = {**os.environ, "CONFIG_PATH": os.path.join(tmpdir, "missing_config.json")}
            # Windowing script should fail
            result = subprocess.run(
                [sys.executable, "window_eeg_data.py"],
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=DEFAULT_TIMEOUT,
            )  # nosec B603
            assert result.returncode != 0
            assert (
                "No such file" in result.stderr
                or "not found" in result.stderr
                or "FileNotFoundError" in result.stderr
            )
            # Training script should fail
            result = subprocess.run(
                [sys.executable, "train_eeg_model.py"],
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=DEFAULT_TIMEOUT,
            )  # nosec B603
            assert result.returncode != 0
            assert (
                "No such file" in result.stderr
                or "not found" in result.stderr
                or "FileNotFoundError" in result.stderr
            )


def test_model_prediction_after_training():
    """Test that a trained model can be loaded and used for prediction."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Create synthetic CSV and config
        n_samples, n_channels = 100, 4
        data = np.random.randn(n_samples, n_channels)
        labels = np.random.choice(["left", "right"], size=n_samples)
        df = pd.DataFrame(data, columns=[f"ch_{i}" for i in range(n_channels)])
        df["session_type"] = "pure"
        df["label"] = labels
        csv_path = os.path.join(tmpdir, "eeg.csv")
        df.to_csv(csv_path, index=False)
        config = {
            "N_CHANNELS": n_channels,
            "WINDOW_SIZE": 50,
            "STEP_SIZE": 25,
            "OUTPUT_CSV": csv_path,
            "WINDOWED_NPY": os.path.join(tmpdir, "X.npy"),
            "WINDOWED_LABELS_NPY": os.path.join(tmpdir, "y.npy"),
            "SESSION_TYPES": ["pure"],
            "USE_SESSION_TYPES": ["pure"],
            "LABELS": ["left", "right"],
            "MODEL_EEGNET": os.path.join(tmpdir, "model.h5"),
            "LABEL_ENCODER": os.path.join(tmpdir, "le.pkl"),
            "SCALER_EEGNET": os.path.join(tmpdir, "scaler.pkl"),
            "LABEL_CLASSES_NPY": os.path.join(tmpdir, "classes.npy"),
        }
        config_path = os.path.join(tmpdir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f)
        # 2. Run windowing and training scripts
        env = {**os.environ, "CONFIG_PATH": config_path}
        subprocess.run(
            [sys.executable, "window_eeg_data.py"],
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=DEFAULT_TIMEOUT,
        )  # nosec B603
        subprocess.run(
            [sys.executable, "train_eeg_model.py"],
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=DEFAULT_TIMEOUT,
        )  # nosec B603
        # 3. Load model and run prediction
        x = np.load(config["WINDOWED_NPY"])
        model = load_model(config["MODEL_EEGNET"])
        # Model expects (batch, channels, samples, 1) or similar
        if x.ndim == 3:
            x = x[..., np.newaxis]
        preds = model.predict(x[:5])
        assert preds.shape[0] == 5


def test_windowed_npy_content():
    """Test that windowed .npy files have correct shapes and label values after windowing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _, n_channels = 120, 3
        data = np.random.randn(120, n_channels)
        labels = np.random.choice(["left", "right"], size=120)
        df = pd.DataFrame(data, columns=[f"ch_{i}" for i in range(n_channels)])
        df["session_type"] = "pure"
        df["label"] = labels
        csv_path = os.path.join(tmpdir, "eeg.csv")
        df.to_csv(csv_path, index=False)
        config = {
            "N_CHANNELS": n_channels,
            "WINDOW_SIZE": 30,
            "STEP_SIZE": 10,
            "OUTPUT_CSV": csv_path,
            "WINDOWED_NPY": os.path.join(tmpdir, "X.npy"),
            "WINDOWED_LABELS_NPY": os.path.join(tmpdir, "y.npy"),
            "SESSION_TYPES": ["pure"],
            "USE_SESSION_TYPES": ["pure"],
            "LABELS": ["left", "right"],
            "MODEL_EEGNET": os.path.join(tmpdir, "model.h5"),
            "LABEL_ENCODER": os.path.join(tmpdir, "le.pkl"),
            "SCALER_EEGNET": os.path.join(tmpdir, "scaler.pkl"),
            "LABEL_CLASSES_NPY": os.path.join(tmpdir, "classes.npy"),
        }
        config_path = os.path.join(tmpdir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f)
        env = {**os.environ, "CONFIG_PATH": config_path}
        subprocess.run(
            [sys.executable, "window_eeg_data.py"],
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=DEFAULT_TIMEOUT,
        )  # nosec B603
        x = np.load(config["WINDOWED_NPY"])
        y = np.load(config["WINDOWED_LABELS_NPY"])
        # Check shapes
        assert x.shape[1] == n_channels
        assert (
            x.shape[2] == config["WINDOW_SIZE"] or x.shape[1] == config["WINDOW_SIZE"]
        )
        assert x.shape[0] == y.shape[0]
        # Check label values
        assert set(np.unique(y)).issubset({"left", "right"})


def test_windowing_with_malformed_csv():
    """Test that window_eeg_data.py fails gracefully on malformed CSV input."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a malformed CSV (missing columns, bad delimiter)
        malformed_csv_path = os.path.join(tmpdir, "malformed.csv")
        with open(malformed_csv_path, "w", encoding="utf-8") as f:
            f.write("badly,formatted,data\n1,2\n3,4\n")
        # Minimal config
        config = {
            "N_CHANNELS": 2,
            "WINDOW_SIZE": 10,
            "STEP_SIZE": 5,
            "OUTPUT_CSV": malformed_csv_path,
            "WINDOWED_NPY": os.path.join(tmpdir, "X.npy"),
            "WINDOWED_LABELS_NPY": os.path.join(tmpdir, "y.npy"),
            "SESSION_TYPES": ["pure"],
            "USE_SESSION_TYPES": ["pure"],
            "LABELS": ["left", "right"],
            "MODEL_EEGNET": os.path.join(tmpdir, "model.h5"),
            "LABEL_ENCODER": os.path.join(tmpdir, "le.pkl"),
            "SCALER_EEGNET": os.path.join(tmpdir, "scaler.pkl"),
            "LABEL_CLASSES_NPY": os.path.join(tmpdir, "classes.npy"),
        }
        config_path = os.path.join(tmpdir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f)
        env = {**os.environ, "CONFIG_PATH": config_path}
        # Run windowing script and expect failure
        result = subprocess.run(
            [sys.executable, "window_eeg_data.py"],
            env=env,
            capture_output=True,
            text=True,
            check=False,  # Explicitly set check to avoid error if omitted
            timeout=DEFAULT_TIMEOUT,
        )  # nosec B603
        assert result.returncode != 0
        assert (
            "error" in result.stderr.lower()
            or "exception" in result.stderr.lower()
            or "traceback" in result.stderr.lower()
        )


def test_training_with_wrong_shape_npy():
    """Test that train_eeg_model.py fails gracefully on wrong-shape .npy input."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a valid config
        config = {
            "N_CHANNELS": 4,
            "WINDOW_SIZE": 10,
            "STEP_SIZE": 5,
            "OUTPUT_CSV": os.path.join(tmpdir, "dummy.csv"),
            "WINDOWED_NPY": os.path.join(tmpdir, "X.npy"),
            "WINDOWED_LABELS_NPY": os.path.join(tmpdir, "y.npy"),
            "SESSION_TYPES": ["pure"],
            "USE_SESSION_TYPES": ["pure"],
            "LABELS": ["left", "right"],
            "MODEL_EEGNET": os.path.join(tmpdir, "model.h5"),
            "LABEL_ENCODER": os.path.join(tmpdir, "le.pkl"),
            "SCALER_EEGNET": os.path.join(tmpdir, "scaler.pkl"),
            "LABEL_CLASSES_NPY": os.path.join(tmpdir, "classes.npy"),
        }
        config_path = os.path.join(tmpdir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f)
        # Write wrong-shape .npy files
        np.save(
            config["WINDOWED_NPY"], np.random.randn(5, 5, 5)
        )  # Should be (n_windows, window, channels)
        np.save(
            config["WINDOWED_LABELS_NPY"],
            np.array(["left", "right", "left", "right", "left"]),
        )
        env = {**os.environ, "CONFIG_PATH": config_path}
        # Run training script and expect failure
        result = subprocess.run(
            [sys.executable, "train_eeg_model.py"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=DEFAULT_TIMEOUT,
        )  # nosec B603
        assert result.returncode != 0
        assert (
            "error" in result.stderr.lower()
            or "exception" in result.stderr.lower()
            or "traceback" in result.stderr.lower()
        )


if __name__ == "__main__":
    pytest.main([__file__])
