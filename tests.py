"""
Unit tests for EEG training system utility functions.
"""

import os
import subprocess
import tempfile
import json

import numpy as np
import pandas as pd
import pytest
from keras.models import load_model

from EEGModels import EEGNet
from utils import check_labels_valid, check_no_nan, load_config, window_data


def test_window_data_shape():
    """Test that window_data returns correct shapes."""
    X = np.random.randn(1000, 16)
    y = np.random.choice(["left", "right", "neutral"], size=(1000, 1))
    X_windows, y_windows = window_data(X, y, window_size=250, step_size=125)
    assert X_windows.shape[1:] == (250, 16)
    assert X_windows.shape[0] == y_windows.shape[0]


def test_window_data_output():
    """Test window_data produces correct windowed output and label consistency."""
    X = np.random.randn(500, 8)  # 500 samples, 8 channels
    y = np.random.choice(["left", "right"], size=(500, 1))
    window_size = 100
    step_size = 50
    X_windows, y_windows = window_data(X, y, window_size, step_size)
    # Check window shape
    assert X_windows.shape[1:] == (window_size, 8)
    # Check number of windows
    expected_windows = (500 - window_size) // step_size + 1
    assert X_windows.shape[0] == expected_windows
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


def test_check_no_nan_pass():
    """check_no_nan should not raise for clean data."""
    arr = np.zeros((10, 10))
    check_no_nan(arr)


def test_check_no_nan_fail():
    """check_no_nan should raise ValueError for NaN data."""
    arr = np.zeros((5, 5))
    arr[0, 0] = np.nan
    with pytest.raises(ValueError):
        check_no_nan(arr)


def test_check_labels_valid_pass():
    """check_labels_valid should not raise for valid labels."""
    labels = np.array(["left", "right", "neutral"])
    check_labels_valid(labels, valid_labels=["left", "right", "neutral"])


def test_check_labels_valid_nan():
    """check_labels_valid should raise ValueError for NaN labels."""
    labels = np.array(["left", np.nan, "right"], dtype=object)
    with pytest.raises(ValueError):
        check_labels_valid(labels, valid_labels=["left", "right", "neutral"])


def test_check_labels_valid_invalid():
    """check_labels_valid should raise ValueError for invalid labels."""
    labels = np.array(["left", "up", "right"])
    with pytest.raises(ValueError):
        check_labels_valid(labels, valid_labels=["left", "right", "neutral"])


def test_eegnet_train_save_load():
    """Test that EEGNet can be trained, saved, and loaded on dummy data."""
    X = np.random.randn(20, 16, 250, 1)  # (batch, channels, samples, 1)
    y = np.zeros((20,))
    y[:10] = 1  # Two classes
    y_cat = np.zeros((20, 2))
    y_cat[np.arange(20), y.astype(int)] = 1
    model = EEGNet(nb_classes=2, Chans=16, Samples=250)
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(X, y_cat, epochs=1, batch_size=4, verbose=0)
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_eegnet_model.h5")
        model.save(model_path)
        assert os.path.exists(model_path)
        loaded = load_model(model_path)
        assert loaded is not None


def test_end_to_end_pipeline():
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
            "MODEL_CNN": os.path.join(tmpdir, "model.h5"),
            "LABEL_ENCODER": os.path.join(tmpdir, "le.pkl"),
            "SCALER_CNN": os.path.join(tmpdir, "scaler.pkl"),
            "LABEL_CLASSES_NPY": os.path.join(tmpdir, "classes.npy"),
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f)
        # 3. Run windowing script
        subprocess.run(
            ["python", "window_eeg_data.py"],
            env={**os.environ, "CONFIG_PATH": config_path},
            capture_output=True,
            text=True,
            check=True,
        )
        assert os.path.exists(config["WINDOWED_NPY"])
        assert os.path.exists(config["WINDOWED_LABELS_NPY"])
        # 4. Run training script
        subprocess.run(
            ["python", "train_eeg_model.py"],
            env={**os.environ, "CONFIG_PATH": config_path},
            capture_output=True,
            text=True,
            check=True,
        )
        assert os.path.exists(config["MODEL_CNN"])
        assert os.path.exists(config["LABEL_ENCODER"])
        assert os.path.exists(config["SCALER_CNN"])


def test_error_handling_missing_config():
    """Test error handling for missing config file in windowing and training scripts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # No config file created
        env = {**os.environ, "CONFIG_PATH": os.path.join(tmpdir, "missing_config.json")}
        # Windowing script should fail
        result = subprocess.run(
            ["python", "window_eeg_data.py"],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "No such file" in result.stderr or "not found" in result.stderr or "FileNotFoundError" in result.stderr
        # Training script should fail
        result = subprocess.run(
            ["python", "train_eeg_model.py"],
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "No such file" in result.stderr or "not found" in result.stderr or "FileNotFoundError" in result.stderr


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
            "MODEL_CNN": os.path.join(tmpdir, "model.h5"),
            "LABEL_ENCODER": os.path.join(tmpdir, "le.pkl"),
            "SCALER_CNN": os.path.join(tmpdir, "scaler.pkl"),
            "LABEL_CLASSES_NPY": os.path.join(tmpdir, "classes.npy"),
        }
        config_path = os.path.join(tmpdir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f)
        # 2. Run windowing and training scripts
        env = {**os.environ, "CONFIG_PATH": config_path}
        subprocess.run(["python", "window_eeg_data.py"], env=env, capture_output=True, text=True, check=True)
        subprocess.run(["python", "train_eeg_model.py"], env=env, capture_output=True, text=True, check=True)
        # 3. Load model and run prediction
        from keras.models import load_model
        X = np.load(config["WINDOWED_NPY"])
        model = load_model(config["MODEL_CNN"])
        # Model expects (batch, channels, samples, 1) or similar
        if X.ndim == 3:
            X = X[..., np.newaxis]
        preds = model.predict(X[:5])
        assert preds.shape[0] == 5


def test_windowed_npy_content():
    """Test that windowed .npy files have correct shapes and label values after windowing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        n_samples, n_channels = 120, 3
        data = np.random.randn(n_samples, n_channels)
        labels = np.random.choice(["left", "right"], size=n_samples)
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
            "MODEL_CNN": os.path.join(tmpdir, "model.h5"),
            "LABEL_ENCODER": os.path.join(tmpdir, "le.pkl"),
            "SCALER_CNN": os.path.join(tmpdir, "scaler.pkl"),
            "LABEL_CLASSES_NPY": os.path.join(tmpdir, "classes.npy"),
        }
        config_path = os.path.join(tmpdir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f)
        env = {**os.environ, "CONFIG_PATH": config_path}
        subprocess.run(["python", "window_eeg_data.py"], env=env, capture_output=True, text=True, check=True)
        X = np.load(config["WINDOWED_NPY"])
        y = np.load(config["WINDOWED_LABELS_NPY"])
        # Check shapes
        assert X.shape[1] == n_channels
        assert X.shape[2] == config["WINDOW_SIZE"] or X.shape[1] == config["WINDOW_SIZE"]
        assert X.shape[0] == y.shape[0]
        # Check label values
        assert set(np.unique(y)).issubset({"left", "right"})


if __name__ == "__main__":
    pytest.main([__file__])
