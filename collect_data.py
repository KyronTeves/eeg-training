"""
Collect EEG data from LSL stream (OpenBCI GUI), label it, and save to CSV for model training.

This script collects pre-filtered data from OpenBCI GUI via LSL streaming,
ensuring optimal compatibility with the real-time prediction pipeline.

Setup Instructions:
1. Start OpenBCI GUI
2. Connect to your board and verify signal quality
3. Configure filters (recommended: 1-50 Hz bandpass, 50/60 Hz notch)
4. Start LSL streaming in OpenBCI GUI
5. Run this script

Input: LSL stream from OpenBCI GUI (pre-filtered)
Output: Labeled CSV file for model training
"""

import csv
import logging
import time
from datetime import datetime

import numpy as np

from lsl_stream_handler import LSLStreamHandler
from utils import load_config, setup_logging, check_no_nan, check_labels_valid

setup_logging()
config = load_config()

# Configuration
LABELS = config["LABELS"]
SESSION_TYPES = config["SESSION_TYPES"]
TRIAL_DURATION = config["TRIAL_DURATION"]
JOLT_TOTAL_DURATION = config["JOLT_TOTAL_DURATION"]
HYBRID_TOTAL_DURATION = config["HYBRID_TOTAL_DURATION"]
LONG_DURATION = config["LONG_DURATION"]
TRIALS_PER_LABEL = config["TRIALS_PER_LABEL"]
OUTPUT_CSV = config["OUTPUT_CSV"]
N_CHANNELS = config["N_CHANNELS"]
SAMPLING_RATE = config["SAMPLING_RATE"]


def get_session_phases(session_type: str, label: str) -> list[tuple[int, str]]:
    """
    Get the phases for a given session type.

    Args:
        session_type: The type of session.
        label: The label for the trial.

    Returns:
        A list of tuples, where each tuple contains the duration and label for a phase.
    """
    return {
        "pure": [(TRIAL_DURATION, label)],
        "jolt": [(5, "neutral"), (1, label), (5, "neutral")],
        "hybrid": [(5, "neutral"), (5, label)],
        "long": [(LONG_DURATION, label)],
        "test": [(2, label)],  # Quick 2-second test trials
    }.get(session_type, [])


def collect_phase_data(
    lsl_handler: LSLStreamHandler, phase_duration: int, phase_label: str
) -> tuple[list, list]:
    """
    Collects EEG data and timestamps for a single phase.
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
):
    """
    Writes phase data to CSV and updates rows and timestamps lists.
    Performs data validation using utils.py before writing.
    """

    if phase_data:
        phase_data = np.array(phase_data)
        try:
            check_no_nan(phase_data, name="Phase EEG data")
            check_labels_valid([label], valid_labels=LABELS, name="Phase label")
        except ValueError as e:
            logging.error("Data validation failed for phase '%s': %s", phase_label, e)
            return
        logging.info(
            "Collected %d samples for phase '%s'", len(phase_data), phase_label
        )
        for i, sample in enumerate(phase_data):
            if len(sample) == N_CHANNELS:
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
) -> tuple[int, list[float]]:
    """
    Collect EEG data for a single trial using LSL streaming.

    Args:
        lsl_handler: LSL stream handler instance.
        session_type: Type of session (e.g., 'pure', 'jolt').
        label: Label for the trial (e.g., 'left', 'right').
        trial_num: Trial number.
        output_writer: CSV writer object.

    Returns:
        Tuple of (number of rows written, list of timestamps).
    """
    rows = []
    timestamps = []
    session_phases = get_session_phases(session_type, label)
    for phase_duration, phase_label in session_phases:
        phase_data, phase_timestamps = collect_phase_data(
            lsl_handler, phase_duration, phase_label
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
        )
    return len(rows), timestamps


def run_trials_for_label(lsl_handler, session_type, label, writer, total_rows):
    """Run all trials for a given label and update total_rows."""
    for trial in range(TRIALS_PER_LABEL):
        logging.info("\nTrial %d/%d for '%s'", trial + 1, TRIALS_PER_LABEL, label)

        # Pause between trials
        if trial > 0:
            logging.info("30-second break between trials...")
            time.sleep(30)

        input(f"Press Enter when ready to start trial {trial + 1} for '{label}'...")

        # Collect trial data
        try:
            rows_written, _ = collect_trial_eeg_lsl(
                lsl_handler, session_type, label, trial + 1, writer
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
    """Main data collection function using LSL streaming."""

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
    if lsl_handler.n_channels != N_CHANNELS:
        logging.warning(
            "Channel count mismatch: Expected %d, got %d",
            N_CHANNELS,
            lsl_handler.n_channels,
        )
        response = input("Continue anyway? (y/n): ")
        if response.lower() != "y":
            return

    # Get session information
    print(f"Available session types: {SESSION_TYPES}")
    session_type = input("Enter session type: ").strip()
    if session_type not in SESSION_TYPES:
        logging.error("Invalid session type: %s", session_type)
        return

    print(f"Available labels: {LABELS}")

    # Create output CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "timestamp_iso",
            "timestamp",
            "session_type",
            "trial_num",
            "phase_label",
            "label",
        ] + [f"ch_{i}" for i in range(N_CHANNELS)]

        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)

        logging.info("Starting data collection. Output file: %s", OUTPUT_CSV)
        logging.info("Session type: %s", session_type)
        logging.info("Trials per label: %d", TRIALS_PER_LABEL)

        total_rows = 0

        # Collect data for each label
        for label in LABELS:
            logging.info("\n=== Collecting data for label: %s ===", label)
            total_rows = run_trials_for_label(
                lsl_handler, session_type, label, writer, total_rows
            )

        logging.info("\n=== Data Collection Complete ===")
        logging.info("Total rows written: %d", total_rows)
        logging.info("Output file: %s", OUTPUT_CSV)

    # Disconnect from LSL
    lsl_handler.disconnect()


if __name__ == "__main__":
    main()
