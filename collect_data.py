"""
Collect EEG data from a BrainFlow-compatible board (e.g., Cyton Daisy), label it, and save to
CSV for supervised model training.

- Prompts the user for session type and label
- Collects multi-phase EEG trials
- Writes raw EEG channel data with metadata to a CSV file for downstream processing and model
  development

Input: Live EEG data stream from board
Output: Labeled CSV file for training
"""

import csv
import json
import logging
import os
import time
from datetime import datetime

from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams

from utils import load_config, setup_logging

setup_logging()

config = load_config()

# --- Configuration ---
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
BOARD_ID = BoardIds.CYTON_DAISY_BOARD.value


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


def collect_trial_eeg(
    board: BoardShim,
    eeg_channels: list,
    session_type: str,
    label: str,
    trial_num: int,
    output_writer: csv.writer,
) -> tuple[int, list[float]]:
    """
    Collect EEG data for a single trial, consisting of one or more phases.

    Args:
        board: BrainFlow BoardShim instance.
        eeg_channels: List of EEG channel indices.
        session_type: Type of session (e.g., 'pure', 'jolt').
        label: Label for the trial (e.g., 'left', 'right').
        trial_num: Trial number.
        output_writer: CSV writer object.

    Returns:
        Tuple of (number of rows written, list of timestamps).
    """
    _ = trial_num  # Suppress unused variable warning

    rows = []
    timestamps = []
    session_phases = get_session_phases(session_type, label)

    for phase_duration, phase_label in session_phases:
        logging.info("Thinking '%s' for %d seconds.", phase_label, phase_duration)
        board.get_board_data()  # Clear buffer
        board.insert_marker(1)
        time.sleep(phase_duration)
        data = board.get_board_data()

        # Use the actual number of samples returned by the board
        n_samples = data.shape[1]
        if n_samples > 0:
            for i in range(n_samples):
                row = [data[ch][i] for ch in eeg_channels] + [
                    session_type,
                    phase_label,
                ]
                rows.append(row)
                timestamps.append(time.time())

    for row in rows:
        output_writer.writerow(row)

    return len(rows), timestamps


def get_user_input(prompt: str, valid_options: list[str]) -> str:
    """
    Prompt the user for input and validate it against a list of valid options.

    Args:
        prompt: The message to display to the user.
        valid_options: A list of valid string inputs.

    Returns:
        The validated user input.
    """
    while True:
        user_input = input(prompt).strip().lower()
        if user_input in valid_options:
            return user_input
        logging.warning(
            "Invalid input. Please choose from: %s", ", ".join(valid_options)
        )


def main():
    """
    Main entry point for EEG data collection. Initializes the board, handles user input,
    manages data collection trials, and writes data and metadata to CSV/JSON files.
    Handles errors related to board connection, file I/O, and user interruptions.
    """
    try:
        params = BrainFlowInputParams()
        params.serial_port = config["COM_PORT"]
        board = BoardShim(BOARD_ID, params)
        board.prepare_session()
        board.start_stream()
        eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
    except (OSError, ValueError, RuntimeError) as e:
        logging.error("Failed to start board session: %s", e)
        return

    try:
        file_exists = os.path.isfile(OUTPUT_CSV)
        with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists or os.stat(OUTPUT_CSV).st_size == 0:
                header = [f"ch_{ch}" for ch in eeg_channels] + [
                    "session_type",
                    "label",
                ]
                writer.writerow(header)

            session_type = get_user_input(
                f"Enter session type ({', '.join(SESSION_TYPES)}): ", SESSION_TYPES
            )
            label = get_user_input(
                f"Enter direction label ({', '.join(LABELS)}): ", LABELS
            )

            n_trials = 1 if session_type in ["long", "test"] else TRIALS_PER_LABEL

            for trial in range(n_trials):
                logging.info(
                    "\nGet ready for '%s' - Trial %d/%d (%s)...",
                    label,
                    trial + 1,
                    n_trials,
                    session_type,
                )
                for sec in range(3, 0, -1):
                    print(f"Starting in {sec}...", end="\r", flush=True)
                    time.sleep(1)
                print(" " * 20, end="\r")

                rows_written, _ = collect_trial_eeg(
                    board, eeg_channels, session_type, label, trial, writer
                )

                # Save metadata for this specific trial
                meta = {
                    "session_type": session_type,
                    "label": label,
                    "trial_num": trial + 1,
                    "n_trials": n_trials,
                    "rows_written": rows_written,
                    "timestamp_utc": datetime.now(datetime.timezone.utc).isoformat(),
                }
                meta_filename = (
                    f"meta_{session_type}_{label}_{trial + 1}_{int(time.time())}.json"
                )
                try:
                    with open(
                        os.path.join("data", "metadata", meta_filename),
                        "w",
                        encoding="utf-8",
                    ) as metaf:
                        json.dump(meta, metaf, indent=2)
                except (OSError, json.JSONDecodeError) as e:
                    logging.error(
                        "Failed to save metadata file %s: %s", meta_filename, e
                    )

                logging.info(
                    "Trial %d for '%s' (%s) complete. Wrote %d rows.",
                    trial + 1,
                    label,
                    session_type,
                    rows_written,
                )

    except (OSError, ValueError, RuntimeError) as e:
        logging.error("An error occurred during data collection: %s", e)
    finally:
        try:
            board.stop_stream()
            board.release_session()
            logging.info("Board session released.")
        except (OSError, RuntimeError) as e:
            logging.error("Error releasing board session: %s", e)


if __name__ == "__main__":
    main()
