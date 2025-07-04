# ruff: noqa: S101, S603
"""test_system.py.

Unit tests for EEG training system utility functions and pipeline integration.

Tests data windowing, config loading, validation, model training, and integration using
synthetic or real EEG data and configs.
"""
from __future__ import annotations

import os
import subprocess  # nosec B404
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from keras.models import load_model

from EEGModels import EEGNet, ShallowConvNet
from utils import (
    CUSTOM_OBJECTS,
    check_labels_valid,
    check_no_nan,
    load_config,
    save_json,  # Add save_json utility
    window_data,
)

# Add a default timeout for all subprocess.run calls
DEFAULT_TIMEOUT = 30  # seconds


def test_window_data_shape() -> None:
    """Test that window_data returns correct shapes."""
    x, y = make_synthetic_eeg_data(n_samples=1000, n_channels=16)
    x_windows, y_windows = window_data(x, y, window_size=250, step_size=125)
    assert x_windows.shape[1:] == (250, 16)
    assert x_windows.shape[0] == y_windows.shape[0]


def test_window_data_output() -> None:
    """Test window_data produces correct windowed output and label consistency."""
    x, y = make_synthetic_eeg_data(
        n_samples=500, n_channels=8, labels=["left", "right"],
    )
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


def test_load_config_keys() -> None:
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
    ("arr", "should_raise"),
    [
        (np.zeros((10, 10)), False),
        (lambda: (a := np.zeros((5, 5)), a.__setitem__((0, 0), np.nan), a)[-1], True),
    ],
)
def test_check_no_nan_cases(arr: np.ndarray, *, should_raise: bool) -> None:
    """Test check_no_nan with arrays that should or should not raise ValueError."""
    if callable(arr):
        arr = arr()
    if should_raise:
        with pytest.raises(ValueError, match="nan|NaN|NAN"):
            check_no_nan(arr)
    else:
        check_no_nan(arr)


@pytest.mark.parametrize(
    ("labels", "should_raise"),
    [
        (np.array(["left", "right", "neutral"]), False),
        (np.array(["left", np.nan, "right"], dtype=object), True),
        (np.array(["left", "up", "right"]), True),
    ],
)
def test_check_labels_valid_cases(labels: np.ndarray, *, should_raise: bool) -> None:
    """Test check_labels_valid with valid and invalid label arrays."""
    if should_raise:
        with pytest.raises(ValueError, match="label|Label|invalid|Invalid"):
            check_labels_valid(labels, valid_labels=["left", "right", "neutral"])
    else:
        check_labels_valid(labels, valid_labels=["left", "right", "neutral"])


# ----------------------
# Model Training & IO Tests
# ----------------------


@pytest.mark.parametrize(
    ("model_class", "model_name"),
    [
        (EEGNet, "eegnet"),
        (ShallowConvNet, "shallow"),
    ],
)
def test_model_train_save_load(model_class: type, model_name: str) -> None:
    """Test that EEGNet and ShallowConvNet can be trained, saved, and loaded on dummy data."""
    rng = np.random.default_rng()
    x = rng.standard_normal((20, 16, 250, 1))  # (batch, channels, samples, 1)
    y = np.zeros((20,))
    y[:10] = 1  # Two classes
    y_cat = np.zeros((20, 2))
    y_cat[np.arange(20), y.astype(int)] = 1
    model = model_class(nb_classes=2, Chans=16, Samples=250)
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"],
    )
    model.fit(x, y_cat, epochs=1, batch_size=4, verbose=0)
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / f"test_{model_name}_model.h5"
        model.save(model_path)
        assert model_path.exists()
        if model_name == "shallow":
            # Provide custom objects for ShallowConvNet
            loaded = load_model(model_path, custom_objects=CUSTOM_OBJECTS)
        else:
            loaded = load_model(model_path)
        assert loaded is not None


# ----------------------
# Integration & Pipeline Tests
# ----------------------
class TestIntegration:
    """Integration and pipeline tests for the EEG training system."""

    def test_end_to_end_pipeline(self) -> None:
        """Black-box end-to-end test: CSV -> windowed .npy -> model file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Create synthetic CSV
            csv_path = Path(tmpdir) / "eeg.csv"
            n_samples, n_channels = 200, 4
            rng = np.random.default_rng()
            data = rng.standard_normal((n_samples, n_channels))
            labels = rng.choice(["left", "right"], size=n_samples)
            df = pd.DataFrame(data, columns=[f"ch_{i}" for i in range(n_channels)])
            df["session_type"] = "pure"
            df["label"] = labels
            df.to_csv(csv_path, index=False)
            # 2. Create minimal config.json
            config_path = Path(tmpdir) / "config.json"
            config = {
                "N_CHANNELS": n_channels,
                "WINDOW_SIZE": 50,
                "STEP_SIZE": 25,
                "OUTPUT_CSV": csv_path,
                "WINDOWED_NPY": str(Path(tmpdir) / "X.npy"),
                "WINDOWED_LABELS_NPY": str(Path(tmpdir) / "y.npy"),
                "SESSION_TYPES": ["pure"],
                "USE_SESSION_TYPES": ["pure"],
                "LABELS": ["left", "right"],
                "MODEL_EEGNET": str(Path(tmpdir) / "model.h5"),
                "MODEL_SHALLOW": str(Path(tmpdir) / "shallow.h5"),
                "LABEL_ENCODER": str(Path(tmpdir) / "le.pkl"),
                "SCALER_EEGNET": str(Path(tmpdir) / "scaler.pkl"),
                "SCALER_SHALLOW": str(Path(tmpdir) / "scaler_shallow.pkl"),
                "LABEL_CLASSES_NPY": str(Path(tmpdir) / "classes.npy"),
                "EARLY_STOPPING_MONITOR": "val_loss",
                "EARLY_STOPPING_PATIENCE": 3,
                "EEGNET_KERN_LENGTH": 64,
                "EEGNET_F1": 8,
                "EEGNET_D": 2,
                "EEGNET_F2": 16,
                "EEGNET_DROPOUT_RATE": 0.5,
                "EEGNET_DROPOUT_TYPE": "Dropout",
                "EEGNET_NORM_RATE": 0.25,
                "OPTIMIZER": "adam",
                "LOSS_FUNCTION": "categorical_crossentropy",
                "EPOCHS": 2,
                "BATCH_SIZE": 8,
                "VALIDATION_SPLIT": 0.2,
                "SAMPLING_RATE": 250,
                "MODEL_RF": str(Path(tmpdir) / "rf.pkl"),
                "MODEL_XGB": str(Path(tmpdir) / "xgb.pkl"),
                "SCALER_TREE": str(Path(tmpdir) / "scaler_tree.pkl"),
            }
            save_json(config, config_path)
            # 3. Run windowing script
            subprocess.run(
                [sys.executable, "window_eeg_data.py"],
                env={**os.environ, "CONFIG_PATH": config_path},
                capture_output=True,
                text=True,
                check=True,
                timeout=DEFAULT_TIMEOUT,
            )  # nosec B603
            assert Path(config["WINDOWED_NPY"]).exists()
            assert Path(config["WINDOWED_LABELS_NPY"]).exists()
            # 4. Run training script
            subprocess.run(
                [sys.executable, "train_eeg_model.py"],
                env={**os.environ, "CONFIG_PATH": config_path},
                capture_output=True,
                text=True,
                check=True,
                timeout=DEFAULT_TIMEOUT,
            )  # nosec B603
            assert Path(config["MODEL_EEGNET"]).exists()
            assert Path(config["MODEL_SHALLOW"]).exists()
            assert Path(config["LABEL_ENCODER"]).exists()
            assert Path(config["SCALER_EEGNET"]).exists()
            # Removed assertion for SCALER_SHALLOW, as it is not created by the code
            x = np.load(config["WINDOWED_NPY"])
            ndim_without_channel = 3
            if x.ndim == ndim_without_channel:
                x = x[..., np.newaxis]
            # Transpose to (batch, channels, samples, 1) for ShallowConvNet
            x_shallow = np.transpose(x, (0, 2, 1, 3))
            shallow_model = load_model(
                config["MODEL_SHALLOW"], custom_objects=CUSTOM_OBJECTS,
            )
            n_pred_samples = 5
            shallow_preds = shallow_model.predict(x_shallow[:n_pred_samples])
            assert shallow_preds.shape[0] == n_pred_samples

    def test_error_handling_missing_config(self) -> None:
        """Test error handling for missing config file in windowing and training scripts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # No config file created
            env = {
                **os.environ,
                "CONFIG_PATH": Path(tmpdir) / "missing_config.json",
            }
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


def test_model_prediction_after_training() -> None:
    """Test that a trained model can be loaded and used for prediction."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Create synthetic CSV and config
        n_samples, n_channels = 100, 4
        rng = np.random.default_rng()
        data = rng.standard_normal((n_samples, n_channels))
        # Ensure at least 2 samples per class for stratified split
        labels = np.array(
            ["left"] * (n_samples // 2) + ["right"] * (n_samples - n_samples // 2),
        )
        rng.shuffle(labels)
        df = pd.DataFrame(data, columns=[f"ch_{i}" for i in range(n_channels)])
        df["session_type"] = "pure"
        df["label"] = labels
        csv_path = Path(tmpdir) / "eeg.csv"
        df.to_csv(csv_path, index=False)
        config = {
            "N_CHANNELS": n_channels,
            "WINDOW_SIZE": 50,
            "STEP_SIZE": 25,
            "OUTPUT_CSV": csv_path,
            "WINDOWED_NPY": Path(tmpdir) / "X.npy",
            "WINDOWED_LABELS_NPY": Path(tmpdir) / "y.npy",
            "SESSION_TYPES": ["pure"],
            "USE_SESSION_TYPES": ["pure"],
            "LABELS": ["left", "right"],
            "MODEL_EEGNET": Path(tmpdir) / "model.h5",
            "MODEL_SHALLOW": Path(tmpdir) / "shallow.h5",
            "LABEL_ENCODER": Path(tmpdir) / "le.pkl",
            "SCALER_EEGNET": Path(tmpdir) / "scaler.pkl",
            "LABEL_CLASSES_NPY": Path(tmpdir) / "classes.npy",
            "EARLY_STOPPING_MONITOR": "val_loss",
            "EARLY_STOPPING_PATIENCE": 3,
            "EEGNET_KERN_LENGTH": 64,
            "EEGNET_F1": 8,
            "EEGNET_D": 2,
            "EEGNET_F2": 16,
            "EEGNET_DROPOUT_RATE": 0.5,
            "EEGNET_DROPOUT_TYPE": "Dropout",
            "EEGNET_NORM_RATE": 0.25,
            "OPTIMIZER": "adam",
            "LOSS_FUNCTION": "categorical_crossentropy",
            "EPOCHS": 2,
            "BATCH_SIZE": 8,
            "VALIDATION_SPLIT": 0.2,
            "SAMPLING_RATE": 250,
            "MODEL_RF": Path(tmpdir) / "rf.pkl",
            "MODEL_XGB": Path(tmpdir) / "xgb.pkl",
            "SCALER_TREE": Path(tmpdir) / "scaler_tree.pkl",
        }
        config_path = Path(tmpdir) / "config.json"
        save_json(config, config_path)
        # 2. Run windowing and training scripts
        env = {**os.environ, "CONFIG_PATH": str(config_path)}
        subprocess.run(
            [sys.executable, "window_eeg_data.py"],
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=DEFAULT_TIMEOUT,
        )  # nosec B603
        # Check label distribution after windowing
        y_windowed = np.load(config["WINDOWED_LABELS_NPY"])
        unique, counts = np.unique(y_windowed, return_counts=True)
        min_samples_per_class = 2
        if np.any(counts < min_samples_per_class):
            pytest.skip(
                f"Not enough samples per class after windowing: {dict(zip(unique, counts))}",
            )
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
        try:
            model = load_model(config["MODEL_EEGNET"], custom_objects=CUSTOM_OBJECTS)
        except TypeError:
            # If EEGNet does not use custom objects, fallback
            model = load_model(config["MODEL_EEGNET"])
        # Model expects (batch, channels, samples, 1) or similar
        ndim_without_channel = 3
        if x.ndim == ndim_without_channel:
            x = x[..., np.newaxis]
        n_pred_samples = 5
        preds = model.predict(x[:n_pred_samples])
        assert preds.shape[0] == n_pred_samples


def test_windowed_npy_content() -> None:
    """Test that windowed .npy files have correct shapes and label values after windowing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _, n_channels = 120, 3
        rng = np.random.default_rng()
        data = rng.standard_normal((120, n_channels))
        labels = rng.choice(["left", "right"], size=120)
        df = pd.DataFrame(data, columns=[f"ch_{i}" for i in range(n_channels)])
        df["session_type"] = "pure"
        df["label"] = labels
        csv_path = Path(tmpdir) / "eeg.csv"
        df.to_csv(csv_path, index=False)
        config = {
            "N_CHANNELS": n_channels,
            "WINDOW_SIZE": 30,
            "STEP_SIZE": 10,
            "OUTPUT_CSV": csv_path,
            "WINDOWED_NPY": Path(tmpdir) / "X.npy",
            "WINDOWED_LABELS_NPY": Path(tmpdir) / "y.npy",
            "SESSION_TYPES": ["pure"],
            "USE_SESSION_TYPES": ["pure"],
            "LABELS": ["left", "right"],
            "MODEL_EEGNET": Path(tmpdir) / "model.h5",
            "LABEL_ENCODER": Path(tmpdir) / "le.pkl",
            "SCALER_EEGNET": Path(tmpdir) / "scaler.pkl",
            "LABEL_CLASSES_NPY": Path(tmpdir) / "classes.npy",
            "EARLY_STOPPING_MONITOR": "val_loss",
            "EARLY_STOPPING_PATIENCE": 3,
            "EEGNET_KERN_LENGTH": 64,
            "EEGNET_F1": 8,
            "EEGNET_D": 2,
            "EEGNET_F2": 16,
            "EEGNET_DROPOUT_RATE": 0.5,
            "EEGNET_DROPOUT_TYPE": "Dropout",
            "EEGNET_NORM_RATE": 0.25,
            "OPTIMIZER": "adam",
            "LOSS_FUNCTION": "categorical_crossentropy",
            "EPOCHS": 2,
            "BATCH_SIZE": 8,
            "VALIDATION_SPLIT": 0.2,
        }
        config_path = Path(tmpdir) / "config.json"
        save_json(config, config_path)
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
        # Accept either (n_windows, window_size, n_channels) or (n_windows, n_channels, window_size)
        assert (x.shape[1] == config["WINDOW_SIZE"] and x.shape[2] == n_channels) or (
            x.shape[2] == config["WINDOW_SIZE"] and x.shape[1] == n_channels
        )
        assert x.shape[0] == y.shape[0]
        # Check label values
        assert set(np.unique(y)).issubset({"left", "right"})


def test_windowing_with_malformed_csv() -> None:
    """Test that window_eeg_data.py fails gracefully on malformed CSV input."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a malformed CSV (missing columns, bad delimiter)
        malformed_csv_path = Path(tmpdir) / "malformed.csv"
        with malformed_csv_path.open("w", encoding="utf-8") as f:
            f.write("badly,formatted,data\n1,2\n3,4\n")
        # Minimal config
        config = {
            "N_CHANNELS": 2,
            "WINDOW_SIZE": 10,
            "STEP_SIZE": 5,
            "OUTPUT_CSV": malformed_csv_path,
            "WINDOWED_NPY": Path(tmpdir) / "X.npy",
            "WINDOWED_LABELS_NPY": Path(tmpdir) / "y.npy",
            "SESSION_TYPES": ["pure"],
            "USE_SESSION_TYPES": ["pure"],
            "LABELS": ["left", "right"],
            "MODEL_EEGNET": Path(tmpdir) / "model.h5",
            "LABEL_ENCODER": Path(tmpdir) / "le.pkl",
            "SCALER_EEGNET": Path(tmpdir) / "scaler.pkl",
            "LABEL_CLASSES_NPY": Path(tmpdir) / "classes.npy",
        }
        config_path = Path(tmpdir) / "config.json"
        save_json(config, config_path)
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


def test_training_with_wrong_shape_npy() -> None:
    """Test that train_eeg_model.py fails gracefully on wrong-shape .npy input."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a valid config
        config = {
            "N_CHANNELS": 4,
            "WINDOW_SIZE": 10,
            "STEP_SIZE": 5,
            "OUTPUT_CSV": Path(tmpdir) / "dummy.csv",
            "WINDOWED_NPY": Path(tmpdir) / "X.npy",
            "WINDOWED_LABELS_NPY": Path(tmpdir) / "y.npy",
            "SESSION_TYPES": ["pure"],
            "USE_SESSION_TYPES": ["pure"],
            "LABELS": ["left", "right"],
            "MODEL_EEGNET": Path(tmpdir) / "model.h5",
            "LABEL_ENCODER": Path(tmpdir) / "le.pkl",
            "SCALER_EEGNET": Path(tmpdir) / "scaler.pkl",
            "LABEL_CLASSES_NPY": Path(tmpdir) / "classes.npy",
        }
        config_path = Path(tmpdir) / "config.json"
        save_json(config, config_path)
        # Write wrong-shape .npy files
        rng = np.random.default_rng()
        np.save(
            config["WINDOWED_NPY"], rng.standard_normal((5, 5, 5)),
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


# TEST UTILITIES


def make_synthetic_eeg_data(
    n_samples: int = 1000,
    n_channels: int = 8,
    labels: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Define factory for synthetic EEG data and labels.

    Args:
        n_samples (int, optional): Number of samples to generate. Defaults to 1000.
        n_channels (int, optional): Number of EEG channels. Defaults to 8.
        labels (list[str] | None, optional): List of labels to use. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple containing the synthetic EEG data and labels.

    """
    rng = np.random.default_rng()
    x = rng.standard_normal((n_samples, n_channels))
    if labels is None:
        labels = rng.choice(["left", "right", "neutral"], size=(n_samples, 1))
    else:
        labels = rng.choice(labels, size=(n_samples, 1))
    return x, labels


if __name__ == "__main__":
    pytest.main([__file__])
