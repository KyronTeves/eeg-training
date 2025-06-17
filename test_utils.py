"""
Unit tests for EEG training system utility functions.
"""

import numpy as np
import pytest
from utils import window_data, load_config

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

if __name__ == "__main__":
    pytest.main([__file__])
