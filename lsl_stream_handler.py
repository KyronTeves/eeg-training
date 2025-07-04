"""lsl_stream_handler.py.

LSL Stream Handler for receiving pre-filtered EEG data from OpenBCI GUI.

Connects to LSL, receives and buffers EEG data, and provides windowed access for downstream processing.
"""
from __future__ import annotations

import logging
import time

import numpy as np
from pylsl import StreamInlet, resolve_byprop

from utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class LSLStreamHandler:
    """Handles LSL stream connection and data collection from OpenBCI GUI."""

    def __init__(self, stream_name: str = "obci_eeg1", timeout: float = 10.0) -> None:
        """Initialize the LSL stream handler.

        Args:
            stream_name (str, optional): Name of the LSL stream to connect to. Defaults to "obci_eeg1".
            timeout (float, optional): Timeout in seconds for stream resolution. Defaults to 10.0.

        """
        self.stream_name: str = stream_name
        self.timeout: float = timeout
        self.inlet: StreamInlet | None = None
        self.sample_rate: float | None = None
        self.n_channels: int | None = None
        self._connected: bool = False

    @property
    def connected(self) -> bool:
        """Check if the LSL stream is connected.

        Returns:
            bool: True if connected, False otherwise.

        """
        return self._connected and self.inlet is not None

    def connect(self) -> bool:
        """Connect to the LSL stream from OpenBCI GUI.

        Returns:
            bool: True if connected successfully, False otherwise.

        """
        try:
            logger.info("Looking for LSL stream: %s", self.stream_name)
            streams = resolve_byprop("name", self.stream_name, timeout=self.timeout)
            if not streams:
                logger.error("No LSL stream found with name: %s", self.stream_name)
                return False
            self.inlet = StreamInlet(streams[0])
            stream_info = streams[0]
            self.sample_rate = stream_info.nominal_srate()
            self.n_channels = stream_info.channel_count()
            logger.info(
                "Connected to LSL stream: %s | Sample rate: %s Hz | Channels: %s | Data type: %s",
                self.stream_name,
                self.sample_rate,
                self.n_channels,
                stream_info.channel_format(),
            )
            self._connected = True
            return True  # noqa: TRY300
        except (RuntimeError, ValueError, ImportError):
            logger.exception("Failed to connect to LSL stream.")
            self._connected = False
            return False

    def get_chunk(self, max_samples: int = 250) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve a chunk of data from the LSL stream.

        Args:
            max_samples (int, optional): Maximum number of samples to retrieve. Defaults to 250.

        Raises:
            RuntimeError: If there is an error pulling data from the LSL stream.

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple of (data, timestamps) arrays.

        """
        if self.inlet is None:
            msg = "LSL stream not connected. Call connect() first."
            raise RuntimeError(msg)
        try:
            chunk, timestamps = self.inlet.pull_chunk(max_samples=max_samples)
            if chunk:
                data = np.array(chunk)
                timestamps = np.array(timestamps)
                return data, timestamps
            return np.empty((0, self.n_channels or 0)), np.empty((0,))
        except (RuntimeError, ValueError, ImportError):
            logger.exception("Error pulling data from LSL stream.")
            return np.empty((0, self.n_channels or 0)), np.empty((0,))

    def get_window(
        self, window_size: int, timeout: float = 2.0,
    ) -> np.ndarray | None:
        """Collect exactly window_size samples from the stream.

        Args:
            window_size (int): Number of samples to collect for the window.
            timeout (float, optional): Maximum time in seconds to wait for collecting the window. Defaults to 2.0.

        Raises:
            RuntimeError: If the LSL stream is not connected.

        Returns:
            np.ndarray | None: Array of shape (window_size, channels) or None if timeout.

        """
        if self.inlet is None:
            msg = "LSL stream not connected. Call connect() first."
            raise RuntimeError(msg)
        collected_data = []
        start_time = time.time()
        while len(collected_data) < window_size:
            if time.time() - start_time > timeout:
                logger.warning(
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

    def disconnect(self) -> None:
        """Disconnect from the LSL stream."""
        if self.inlet:
            self.inlet.close_stream()
            self.inlet = None
            self._connected = False
            logger.info("Disconnected from LSL stream")


# pylsl>=1.16.0 required in requirements.txt
