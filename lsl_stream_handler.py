"""
lsl_stream_handler.py

LSL Stream Handler for receiving pre-filtered EEG data from OpenBCI GUI.

Input: LSL stream from OpenBCI GUI
Process: Connects to LSL, receives and buffers EEG data, provides windowed access
Output: EEG data windows for downstream processing
"""

import logging
import time
from typing import Optional, Tuple

import numpy as np
from pylsl import StreamInlet, resolve_byprop

from utils import setup_logging

setup_logging()


class LSLStreamHandler:
    """
    Handles LSL stream connection and data collection from OpenBCI GUI.

    Input: LSL stream name (str), timeout (float)
    Process: Connects to LSL, receives and buffers EEG data, provides windowed access
    Output: Methods for chunk/window retrieval, connection management
    """

    def __init__(self, stream_name: str = "OpenBCIGUI", timeout: float = 10.0):
        """
        Initialize LSL stream handler.

        Input: stream_name (str), timeout (float)
        Process: Sets up handler attributes
        Output: LSLStreamHandler instance
        """
        self.stream_name = stream_name
        self.timeout = timeout
        self.inlet: Optional[StreamInlet] = None
        self.sample_rate: Optional[float] = None
        self.n_channels: Optional[int] = None

    def connect(self) -> bool:
        """
        Connect to the LSL stream from OpenBCI GUI.

        Input: None (uses self.stream_name, self.timeout)
        Process: Resolves and connects to LSL stream
        Output: True if successful, False otherwise
        """
        try:
            logging.info("Looking for LSL stream: %s", self.stream_name)
            streams = resolve_byprop("name", self.stream_name, timeout=self.timeout)

            if not streams:
                logging.error("No LSL stream found with name: %s", self.stream_name)
                return False

            # Connect to the first matching stream
            self.inlet = StreamInlet(streams[0])
            stream_info = streams[0]

            self.sample_rate = stream_info.nominal_srate()
            self.n_channels = stream_info.channel_count()

            logging.info("Connected to LSL stream:")
            logging.info("  - Sample rate: %s Hz", self.sample_rate)
            logging.info("  - Channels: %s", self.n_channels)
            logging.info("  - Data type: %s", stream_info.channel_format())

            return True

        except (RuntimeError, ValueError, ImportError) as e:
            logging.error("Failed to connect to LSL stream: %s", e)
            return False

    def get_chunk(self, max_samples: int = 250) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a chunk of data from the LSL stream.

        Input: max_samples (int)
        Process: Pulls chunk from LSL inlet
        Output: (data, timestamps) arrays
        """
        if self.inlet is None:
            raise RuntimeError("LSL stream not connected. Call connect() first.")

        try:
            # Pull chunk of data with timestamps
            chunk, timestamps = self.inlet.pull_chunk(max_samples=max_samples)

            if chunk:
                data = np.array(chunk)  # Shape: (samples, channels)
                timestamps = np.array(timestamps)
                return data, timestamps
            else:
                # Return empty arrays if no data available
                return np.array([]).reshape(0, self.n_channels), np.array([])

        except (RuntimeError, ValueError, ImportError) as e:
            logging.error("Error pulling data from LSL stream: %s", e)
            return np.array([]).reshape(0, self.n_channels), np.array([])

    def get_window(
        self, window_size: int, timeout: float = 2.0
    ) -> Optional[np.ndarray]:
        """
        Collect exactly window_size samples from the stream.

        Input: window_size (int), timeout (float)
        Process: Collects samples until window_size or timeout
        Output: np.ndarray (window_size, channels) or None
        """
        if self.inlet is None:
            raise RuntimeError("LSL stream not connected. Call connect() first.")

        collected_data = []
        start_time = time.time()

        while len(collected_data) < window_size:
            if time.time() - start_time > timeout:
                logging.warning(
                    "Timeout collecting window. Got %d/%d samples",
                    len(collected_data), window_size
                )
                return None

            chunk, _ = self.get_chunk(max_samples=window_size - len(collected_data))
            if len(chunk) > 0:
                collected_data.extend(chunk)

            time.sleep(0.001)  # Small delay to prevent busy waiting

        # Return exactly window_size samples
        return np.array(collected_data[:window_size])

    def disconnect(self):
        """
        Disconnect from the LSL stream.

        Input: None
        Process: Closes inlet and cleans up
        Output: None
        """
        if self.inlet:
            self.inlet.close_stream()
            self.inlet = None
            logging.info("Disconnected from LSL stream")


# Update requirements.txt to include:
# pylsl>=1.16.0
