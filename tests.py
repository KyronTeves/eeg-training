"""
Unit tests for EEG training system utility functions.
"""

import numpy as np
import pytest
from utils import window_data, load_config, check_no_nan, check_labels_valid
import tempfile
import os
from keras.models import load_model
from EEGModels import EEGNet

def test_window_data_shape():
    """Test that window_data returns correct shapes."""
    X = np.random.randn(1000, 16)
    y = np.random.choice(['left', 'right', 'neutral'], size=(1000, 1))
    X_windows, y_windows = window_data(X, y, window_size=250, step_size=125)
    assert X_windows.shape[1:] == (250, 16)
    assert X_windows.shape[0] == y_windows.shape[0]

def test_window_data_output():
    """Test window_data produces correct windowed output and label consistency."""
    X = np.random.randn(500, 8)  # 500 samples, 8 channels
    y = np.random.choice(['left', 'right'], size=(500, 1))
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
        assert label in ['left', 'right']

def test_load_config_keys():
    """Test that config contains required keys."""
    config = load_config()
    required_keys = [
        "N_CHANNELS", "WINDOW_SIZE", "STEP_SIZE", "OUTPUT_CSV",
        "WINDOWED_NPY", "WINDOWED_LABELS_NPY", "SESSION_TYPES"
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
    labels = np.array(['left', 'right', 'neutral'])
    check_labels_valid(labels, valid_labels=['left', 'right', 'neutral'])

def test_check_labels_valid_nan():
    """check_labels_valid should raise ValueError for NaN labels."""
    labels = np.array(['left', np.nan, 'right'], dtype=object)
    with pytest.raises(ValueError):
        check_labels_valid(labels, valid_labels=['left', 'right', 'neutral'])

def test_check_labels_valid_invalid():
    """check_labels_valid should raise ValueError for invalid labels."""
    labels = np.array(['left', 'up', 'right'])
    with pytest.raises(ValueError):
        check_labels_valid(labels, valid_labels=['left', 'right', 'neutral'])

def test_eegnet_train_save_load():
    """Test that EEGNet can be trained, saved, and loaded on dummy data."""
    X = np.random.randn(20, 16, 250, 1)  # (batch, channels, samples, 1)
    y = np.zeros((20,))
    y[:10] = 1  # Two classes
    y_cat = np.zeros((20, 2))
    y_cat[np.arange(20), y.astype(int)] = 1
    model = EEGNet(nb_classes=2, Chans=16, Samples=250)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X, y_cat, epochs=1, batch_size=4, verbose=0)
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_eegnet_model.h5")
        model.save(model_path)
        assert os.path.exists(model_path)
        loaded = load_model(model_path)
        assert loaded is not None

if __name__ == "__main__":
    pytest.main([__file__])
