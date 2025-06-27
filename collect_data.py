"""
collect_data.py

Collect EEG data from LSL stream (OpenBCI GUI), label it, and save to CSV for model training.

Input: LSL stream from OpenBCI GUI (pre-filtered)
Process: Collects, labels, and validates EEG data, writes to CSV.
Output: Labeled CSV file for model training.
"""

import csv
import logging
import time
from datetime import datetime

import numpy as np

from lsl_stream_handler import LSLStreamHandler
from utils import check_labels_valid, check_no_nan, load_config, setup_logging


def get_session_phases(
    session_type: str, label: str, config: dict
) -> list[tuple[int, str]]:
    """
    Get the list of (duration, label) phases for a given session type.

    Input: session_type (str), label (str), config (dict)
    Process: Looks up phase structure for session type
    Output: List of (duration, label) tuples
    """
    return {
        "pure": [(config["TRIAL_DURATION"], label)],
        "jolt": [(5, "neutral"), (1, label), (5, "neutral")],
        "hybrid": [(5, "neutral"), (5, label)],
        "long": [(config["LONG_DURATION"], label)],
        "test": [(2, label)],  # Quick 2-second test trials
    }.get(session_type, [])


def collect_phase_data(
    lsl_handler: LSLStreamHandler,
    phase_duration: int,
    phase_label: str,
    label: str,
    config: dict,
) -> tuple[list, list]:
    """
    Collect EEG data and timestamps for a single phase.

    Input: lsl_handler (LSLStreamHandler), phase_duration (int), phase_label (str), label (str), config (dict)
    Process: Collects data from LSL for phase duration
    Output: (phase_data, phase_timestamps)
    """
    logging.info("Phase: Think '%s' for %d seconds.", phase_label, phase_duration)
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
                [label], valid_labels=config["LABELS"], name="Phase label"
            )
        except ValueError as e:
            logging.error("Data validation failed for phase '%s': %s", phase_label, e)
            return [], []
    return phase_data, phase_timestamps


def write_phase_to_csv(
    phase_data: list,
    phase_timestamps: list,
    session_type: str,
    trial_num: int,
    phase_label: str,
    label: str,
    output_writer: csv.writer,
    rows: list,
    timestamps: list,
    config: dict,
):
    """
    Write phase EEG data and metadata to CSV, updating row/timestamp lists.

    Input: phase_data (list), phase_timestamps (list), session_type (str), trial_num (int),
        phase_label (str), label (str), output_writer (csv.writer), rows (list), timestamps (list), config (dict)
    Process: Validates and writes each sample to CSV
    Output: None (side effect: CSV written, lists updated)
    """
    if phase_data:
        phase_data = np.array(phase_data)
        for i, sample in enumerate(phase_data):
            if len(sample) == config["N_CHANNELS"]:
                timestamp = (
                    phase_timestamps[i] if i < len(phase_timestamps) else time.time()
                )
                row = [
                    datetime.fromtimestamp(timestamp).isoformat(),
                    timestamp,
                    session_type,
                    trial_num,
                    phase_label,
                    label,
                ] + sample.tolist()
                output_writer.writerow(row)
                rows.append(row)
                timestamps.append(timestamp)
    else:
        logging.warning("No data collected for phase '%s'", phase_label)


def collect_trial_eeg_lsl(
    lsl_handler: LSLStreamHandler,
    session_type: str,
    label: str,
    trial_num: int,
    output_writer: csv.writer,
    config: dict,
) -> tuple[int, list[float]]:
    """
    Collect EEG data for a single trial using LSL streaming.

    Input: lsl_handler (LSLStreamHandler), session_type (str), label (str),
        trial_num (int), output_writer (csv.writer), config (dict)
    Process: Runs all phases for the trial, writes to CSV
    Output: (number of rows written, list of timestamps)
    """
    rows = []
    timestamps = []
    session_phases = get_session_phases(session_type, label, config)
    for phase_duration, phase_label in session_phases:
        phase_data, phase_timestamps = collect_phase_data(
            lsl_handler, phase_duration, phase_label, label, config
        )
        write_phase_to_csv(
            phase_data,
            phase_timestamps,
            session_type,
            trial_num,
            phase_label,
            label,
            output_writer,
            rows,
            timestamps,
            config,
        )
    return len(rows), timestamps


def run_trials_for_label(lsl_handler, session_type, label, writer, total_rows, config):
    """
    Run all trials for a given label and update total_rows.

    Input: lsl_handler (LSLStreamHandler), session_type (str), label (str),
        writer (csv.writer), total_rows (int), config (dict)
    Process: Loops over trials, collects and writes data
    Output: Updated total_rows
    """
    for trial in range(config["TRIALS_PER_LABEL"]):
        logging.info(
            "\nTrial %d/%d for '%s'", trial + 1, config["TRIALS_PER_LABEL"], label
        )

        # Pause between trials
        if trial > 0:
            logging.info("30-second break between trials...")
            time.sleep(30)

        input(f"Press Enter when ready to start trial {trial + 1} for '{label}'...")

        # Collect trial data
        try:
            rows_written, _ = collect_trial_eeg_lsl(
                lsl_handler, session_type, label, trial + 1, writer, config
            )
            total_rows += rows_written
            logging.info("Trial %d complete. Rows written: %d", trial + 1, rows_written)

        except KeyboardInterrupt:
            logging.info("Data collection interrupted by user.")
            break
        except (IOError, ValueError, RuntimeError) as e:
            logging.error("Error during trial %d: %s", trial + 1, e)
            continue
    return total_rows


def main():
    """
    Main entry point for collecting EEG data from LSL and saving to CSV.

    Input: None (uses config and user input)
    Process: Connects to LSL, collects data for all labels/trials, writes CSV
    Output: None (side effect: CSV written)
    """

    setup_logging()
    config = load_config()

    # Initialize LSL handler
    lsl_handler = LSLStreamHandler(
        stream_name=config["LSL_STREAM_NAME"], timeout=config["LSL_TIMEOUT"]
    )

    # Connect to LSL stream
    if not lsl_handler.connect():
        logging.error("Failed to connect to LSL stream.")
        logging.error("Make sure OpenBCI GUI is running with LSL streaming enabled.")
        return

    logging.info(
        "Connected to LSL stream with %d channels at %d Hz",
        lsl_handler.n_channels,
        lsl_handler.sample_rate,
    )

    # Verify channel count matches configuration
    if lsl_handler.n_channels != config["N_CHANNELS"]:
        logging.warning(
            "Channel count mismatch: Expected %d, got %d",
            config["N_CHANNELS"],
            lsl_handler.n_channels,
        )
        response = input("Continue anyway? (y/n): ")
        if response.lower() != "y":
            return

    # Get session information
    print(f"Available session types: {config['SESSION_TYPES']}")
    session_type = input("Enter session type: ").strip()
    if session_type not in config["SESSION_TYPES"]:
        logging.error("Invalid session type: %s", session_type)
        return

    print(f"Available labels: {config['LABELS']}")

    # Create output CSV
    with open(config["OUTPUT_CSV"], "w", newline="", encoding="utf-8") as csvfile:
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

        logging.info("Starting data collection. Output file: %s", config["OUTPUT_CSV"])
        logging.info("Session type: %s", session_type)
        logging.info("Trials per label: %d", config["TRIALS_PER_LABEL"])

        total_rows = 0

        # Collect data for each label
        for label in config["LABELS"]:
            logging.info("\n=== Collecting data for label: %s ===", label)
            total_rows = run_trials_for_label(
                lsl_handler, session_type, label, writer, total_rows, config
            )

        logging.info("\n=== Data Collection Complete ===")
        logging.info("Total rows written: %d", total_rows)
        logging.info("Output file: %s", config["OUTPUT_CSV"])

    # Disconnect from LSL
    lsl_handler.disconnect()


if __name__ == "__main__":
    main()
