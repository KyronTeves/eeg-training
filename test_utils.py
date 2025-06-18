"""
Unit tests for EEG training system utility functions.
"""

import numpy as np
import pytest
from utils import window_data, load_config, check_no_nan, check_labels_valid

def test_window_data_shape():
    """Test that window_data returns correct shapes."""
    X = np.random.randn(1000, 16)
    y = np.random.choice(['left', 'right', 'neutral'], size=(1000, 1))
    X_windows, y_windows = window_data(X, y, window_size=250, step_size=125)
    assert X_windows.shape[1:] == (250, 16)
    assert X_windows.shape[0] == y_windows.shape[0]

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

if __name__ == "__main__":
    pytest.main([__file__])
