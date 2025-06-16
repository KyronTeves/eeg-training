import json
import numpy as np
import pandas as pd
from typing import Tuple, List

def load_config(path: str = 'config.json') -> dict:
    """Load configuration from a JSON file."""
    with open(path, 'r') as f:
        return json.load(f)

def window_data(
    data: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    step_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment data and labels into overlapping windows.
    Returns (X_windows, y_windows)
    """
    X_windows = []
    y_windows = []
    for start in range(0, len(data) - window_size + 1, step_size):
        window = data[start:start+window_size]
        window_labels = labels[start:start+window_size]
        # Use the most frequent label in the window as the label
        unique, counts = np.unique(window_labels, return_counts=True)
        window_label = unique[np.argmax(counts)]
        X_windows.append(window)
        y_windows.append(window_label)
    return np.array(X_windows), np.array(y_windows)
