"""collect_data.py.

Collect EEG data from an LSL stream (OpenBCI GUI), label it, and save to CSV for model training.

This script manages LSL connection, session/label prompting, data collection, and CSV writing for
supervised EEG experiments.

Typical usage:
    $ python collect_data.py
"""

import csv
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from lsl_stream_handler import LSLStreamHandler
from utils import (
    check_labels_valid,
    check_no_nan,
    handle_errors,
    load_config,
    setup_logging,
)

setup_logging()
logger = logging.getLogger(__name__)


def get_session_phases(
    session_type: str, label: str, config: dict,
) -> list[tuple[int, str]]:
    """Get the list of (duration, label) phases for a given session type.

    Args:
        session_type (str): The type of the session (e.g., 'pure', 'jolt').
        label (str): The label for the phase.
        config (dict): The configuration dictionary.

    Returns:
        list[tuple[int, str]]: A list of (duration, label) tuples for the session phases.

    """
    return {
        "pure": [(config["TRIAL_DURATION"], label)],
        "jolt": [(5, "neutral"), (1, label), (5, "neutral")],
        "hybrid": [(5, "neutral"), (5, label)],
        "long": [(config["LONG_DURATION"], label)],
        "test": [(2, label)],
    }.get(session_type, [])


def collect_phase_data(
    lsl_handler: LSLStreamHandler,
    phase_duration: int,
    phase_label: str,
    label: str,
    config: dict,
) -> tuple[list[Any], list[float]]:
    """Collect EEG data and timestamps for a single phase.

    Args:
        lsl_handler (LSLStreamHandler): LSL handler instance.
        phase_duration (int): Duration of the phase in seconds.
        phase_label (str): Label for the phase.
        label (str): Label for the trial.
        config (dict): Configuration dictionary.

    Returns:
        tuple[list[Any], list[float]]: (phase_data, phase_timestamps)

    """
    logger.info("Phase: Think '%s' for %d seconds.", phase_label, phase_duration)
    start_time = time.time()
    phase_data = []
    phase_timestamps = []
    while time.time() - start_time < phase_duration:
        chunk, chunk_timestamps = lsl_handler.get_chunk(max_samples=50)
        if len(chunk) > 0:
            phase_data.extend(chunk)
            phase_timestamps.extend(chunk_timestamps)
        time.sleep(0.001)
    if phase_data:
        phase_data_np = np.array(phase_data)
        try:
            check_no_nan(phase_data_np, name="Phase EEG data")
            check_labels_valid(
                [label], valid_labels=config["LABELS"], name="Phase label",
            )
        except ValueError:
            logger.exception("Data validation failed for phase '%s'.", phase_label)
            return [], []
    return phase_data, phase_timestamps


def write_phase_to_csv(  # noqa: PLR0913
    phase_data: list,
    phase_timestamps: list,
    session_type: str,
    trial_num: int,
    phase_label: str,
    label: str,
    output_writer: csv.writer,
    config: dict,
) -> tuple[list, list]:
    """Write phase EEG data and metadata to CSV, returning new rows and timestamps.

    Args:
        phase_data (list): EEG data samples.
        phase_timestamps (list): Timestamps for samples.
        session_type (str): Session type.
        trial_num (int): Trial number.
        phase_label (str): Phase label.
        label (str): Label for the trial.
        output_writer (csv.writer): CSV writer.
        config (dict): Configuration dictionary.

    Returns:
        tuple[list, list]: (rows, timestamps) written for this phase.

    """
    rows = []
    timestamps = []
    if phase_data:
        for i, sample in enumerate(phase_data):
            if len(sample) == config["N_CHANNELS"]:
                timestamp = (
                    phase_timestamps[i] if i < len(phase_timestamps) else time.time()
                )
                row = [
                    datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc).isoformat(),
                    timestamp,
                    session_type,
                    trial_num,
                    phase_label,
                    label,
                ] + (sample.tolist() if hasattr(sample, "tolist") else list(sample))
                output_writer.writerow(row)
                rows.append(row)
                timestamps.append(timestamp)
    else:
        logger.warning("No data collected for phase '%s'", phase_label)
    return rows, timestamps


def collect_trial_eeg_lsl(  # noqa: PLR0913
    lsl_handler: LSLStreamHandler,
    session_type: str,
    label: str,
    trial_num: int,
    output_writer: csv.writer,
    config: dict,
) -> tuple[int, list[float]]:
    """Collect EEG data for a single trial using LSL streaming.

    Args:
        lsl_handler (LSLStreamHandler): LSL handler instance.
        session_type (str): Session type.
        label (str): Label for the trial.
        trial_num (int): Trial number.
        output_writer (csv.writer): CSV writer.
        config (dict): Configuration dictionary.

    Returns:
        tuple[int, list[float]]: (number of rows written, list of timestamps)

    """
    all_rows = []
    all_timestamps = []
    session_phases = get_session_phases(session_type, label, config)
    for phase_duration, phase_label in session_phases:
        phase_data, phase_timestamps = collect_phase_data(
            lsl_handler, phase_duration, phase_label, label, config,
        )
        rows, timestamps = write_phase_to_csv(
            phase_data,
            phase_timestamps,
            session_type,
            trial_num,
            phase_label,
            label,
            output_writer,
            config,
        )
        all_rows.extend(rows)
        all_timestamps.extend(timestamps)
    return len(all_rows), all_timestamps


def run_trials_for_label(  # noqa: PLR0913
    lsl_handler: LSLStreamHandler,
    session_type: str,
    label: str,
    writer: csv.writer,
    total_rows: int,
    config: dict,
) -> int:
    """Run all trials for a given label and updates total_rows.

    Args:
        lsl_handler (LSLStreamHandler): LSL handler instance.
        session_type (str): Session type.
        label (str): Label for the trial.
        writer (csv.writer): CSV writer.
        total_rows (int): Current total rows written.
        config (dict): Configuration dictionary.

    Returns:
        int: Updated total_rows.

    """
    for trial in range(config["TRIALS_PER_LABEL"]):
        logger.info(
            "\nTrial %d/%d for '%s'", trial + 1, config["TRIALS_PER_LABEL"], label,
        )
        if trial > 0:
            logger.info("30-second break between trials...")
            time.sleep(30)
        input(f"Press Enter when ready to start trial {trial + 1} for '{label}'...")
        try:
            rows_written, _ = collect_trial_eeg_lsl(
                lsl_handler, session_type, label, trial + 1, writer, config,
            )
            total_rows += rows_written
            logger.info("Trial %d complete. Rows written: %d", trial + 1, rows_written)
        except KeyboardInterrupt:
            logger.info("Data collection interrupted by user.")
            break
        except (OSError, ValueError, RuntimeError):
            logger.exception("Error during trial %d.", trial + 1)
            continue
    return total_rows


@handle_errors
def main() -> None:
    """Collect EEG data from LSL and save to CSV.

    Load config, connect to LSL, prompt for session/label, collect and write data, and log results.
    """
    setup_logging()
    config = load_config()
    lsl_handler = LSLStreamHandler(
        stream_name=config["LSL_STREAM_NAME"], timeout=config["LSL_TIMEOUT"],
    )
    if not lsl_handler.connect():
        logger.error("Failed to connect to LSL stream.")
        logger.error("Make sure OpenBCI GUI is running with LSL streaming enabled.")
        return
    logger.info(
        "Connected to LSL stream with %d channels at %d Hz",
        lsl_handler.n_channels,
        lsl_handler.sample_rate,
    )
    if lsl_handler.n_channels != config["N_CHANNELS"]:
        logger.warning(
            "Channel count mismatch: Expected %d, got %d",
            config["N_CHANNELS"],
            lsl_handler.n_channels,
        )
        response = input("Continue anyway? (y/n): ")
        if response.lower() != "y":
            return
    logger.info("Available session types: %s", config["SESSION_TYPES"])
    session_type = input("Enter session type: ").strip()
    if session_type not in config["SESSION_TYPES"]:
        logger.error("Invalid session type: %s", session_type)
        return
    logger.info("Available labels: %s", config["LABELS"])
    with Path(config["OUTPUT_CSV"]).open("w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "timestamp_iso",
            "timestamp",
            "session_type",
            "trial_num",
            "phase_label",
            "label",
        ] + [f"ch_{i}" for i in range(config["N_CHANNELS"])]
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)
        logger.info("Starting data collection. Output file: %s", config["OUTPUT_CSV"])
        logger.info("Session type: %s", session_type)
        logger.info("Trials per label: %d", config["TRIALS_PER_LABEL"])
        total_rows = 0
        for label in config["LABELS"]:
            logger.info("\n=== Collecting data for label: %s ===", label)
            total_rows = run_trials_for_label(
                lsl_handler, session_type, label, writer, total_rows, config,
            )
        logger.info("\n=== Data Collection Complete ===")
        logger.info("Total rows written: %d", total_rows)
        logger.info("Output file: %s", config["OUTPUT_CSV"])
    lsl_handler.disconnect()


if __name__ == "__main__":
    main()
