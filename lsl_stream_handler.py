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

from utils import setup_logging, log_function_call

setup_logging()


class LSLStreamHandler:
    """
    Handles LSL stream connection and data collection from OpenBCI GUI.
    """

    def __init__(self, stream_name: str = "OpenBCIGUI", timeout: float = 10.0) -> None:
        """
        Initialize LSL stream handler.

        Args:
            stream_name (str): Name of the LSL stream to connect to.
            timeout (float): Timeout in seconds for stream resolution.
        """
        self.stream_name: str = stream_name
        self.timeout: float = timeout
        self.inlet: Optional[StreamInlet] = None
        self.sample_rate: Optional[float] = None
        self.n_channels: Optional[int] = None

    @log_function_call
    def connect(self) -> bool:
        """
        Connect to the LSL stream from OpenBCI GUI.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            logging.info("Looking for LSL stream: %s", self.stream_name)
            streams = resolve_byprop("name", self.stream_name, timeout=self.timeout)
            if not streams:
                logging.error("No LSL stream found with name: %s", self.stream_name)
                return False
            self.inlet = StreamInlet(streams[0])
            stream_info = streams[0]
            self.sample_rate = stream_info.nominal_srate()
            self.n_channels = stream_info.channel_count()
            logging.info(
                "Connected to LSL stream: %s | Sample rate: %s Hz | Channels: %s | Data type: %s",
                self.stream_name,
                self.sample_rate,
                self.n_channels,
                stream_info.channel_format(),
            )
            return True
        except (RuntimeError, ValueError, ImportError) as e:
            logging.error("Failed to connect to LSL stream: %s", e)
            return False

    @log_function_call
    def get_chunk(self, max_samples: int = 250) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a chunk of data from the LSL stream.

        Args:
            max_samples (int): Maximum number of samples to retrieve.
        Returns:
            Tuple[np.ndarray, np.ndarray]: (data, timestamps) arrays.
        """
        if self.inlet is None:
            raise RuntimeError("LSL stream not connected. Call connect() first.")
        try:
            chunk, timestamps = self.inlet.pull_chunk(max_samples=max_samples)
            if chunk:
                data = np.array(chunk)
                timestamps = np.array(timestamps)
                return data, timestamps
            return np.empty((0, self.n_channels or 0)), np.empty((0,))
        except (RuntimeError, ValueError, ImportError) as e:
            logging.error("Error pulling data from LSL stream: %s", e)
            return np.empty((0, self.n_channels or 0)), np.empty((0,))

    @log_function_call
    def get_window(
        self, window_size: int, timeout: float = 2.0
    ) -> Optional[np.ndarray]:
        """
        Collect exactly window_size samples from the stream.

        Args:
            window_size (int): Number of samples to collect.
            timeout (float): Timeout in seconds.
        Returns:
            Optional[np.ndarray]: Array of shape (window_size, channels) or None if timeout.
        """
        if self.inlet is None:
            raise RuntimeError("LSL stream not connected. Call connect() first.")
        collected_data = []
        start_time = time.time()
        while len(collected_data) < window_size:
            if time.time() - start_time > timeout:
                logging.warning(
                    "Timeout collecting window. Got %d/%d samples",
                    len(collected_data),
                    window_size,
                )
                return None
            chunk, _ = self.get_chunk(max_samples=window_size - len(collected_data))
            if len(chunk) > 0:
                collected_data.extend(chunk)
            time.sleep(0.001)
        return np.array(collected_data[:window_size])

    @log_function_call
    def disconnect(self) -> None:
        """
        Disconnect from the LSL stream.
        """
        if self.inlet:
            self.inlet.close_stream()
            self.inlet = None
            logging.info("Disconnected from LSL stream")

# pylsl>=1.16.0 required in requirements.txt
