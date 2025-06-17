"""
Collect EEG data from a BrainFlow-compatible board (e.g., Cyton Daisy),
label it, and save to CSV for training.

- Prompts user for session type and label.
- Collects data in trials, with configurable session phases.
- Saves both raw data and metadata.
- Uses logging for status and error messages.
"""

import csv
import json  # do not remove please
import logging
import os
import time

from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams

from utils import load_config

# Configure logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("eeg_training.log", mode="a"),
    ],
)

config = load_config()

LABELS = config["LABELS"]
SESSION_TYPES = config["SESSION_TYPES"]
TRIAL_DURATION = config["TRIAL_DURATION"]
JOLT_TOTAL_DURATION = config["JOLT_TOTAL_DURATION"]
HYBRID_TOTAL_DURATION = config["HYBRID_TOTAL_DURATION"]
LONG_DURATION = config["LONG_DURATION"]
TRIALS_PER_LABEL = config["TRIALS_PER_LABEL"]
OUTPUT_CSV = config["OUTPUT_CSV"]
N_CHANNELS = config["N_CHANNELS"]


def collect_eeg(
    board: BoardShim,
    eeg_channels: list,
    session_type: str,
    label: str,
    trial_num: int,
    output_writer: csv.writer,
) -> tuple[int, list[float]]:
    """
    Collect EEG data for a given session type and label, write to CSV.

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
    _ = trial_num  # Dummy assignment to suppress unused variable warning

    def run_phase(phase_duration, phase_label):
        logging.info("Think '%s' for %d seconds.", phase_label, phase_duration)
        board.get_board_data()  # Clear buffer
        board.insert_marker(1)
        start_time = time.time()
        while time.time() - start_time < phase_duration:
            time.sleep(0.1)
        data = board.get_board_data()
        n_samples = int(phase_duration * sampling_rate)
        for i in range(-n_samples, 0):
            if abs(i) <= data.shape[1]:
                row = [data[ch][i] for ch in eeg_channels] + [session_type, phase_label]
                rows.append(row)
                timestamps.append(time.time())

    sampling_rate = BoardShim.get_sampling_rate(BoardIds.CYTON_DAISY_BOARD.value)
    rows = []
    timestamps = []

    session_phases = {
        "pure": [(TRIAL_DURATION, label)],
        "jolt": [(5, "neutral"), (1, label), (5, "neutral")],
        "hybrid": [(5, "neutral"), (5, label)],
        "long": [(LONG_DURATION, label)],
    }

    for phase_duration, phase_label in session_phases.get(session_type, []):
        run_phase(phase_duration, phase_label)

    for row in rows:
        output_writer.writerow(row)
    return len(rows), timestamps


def main():
    """
    Main entry point for EEG data collection. Initializes the board, handles user input,
    manages data collection trials, and writes data and metadata to CSV/JSON files.
    Handles errors related to board connection, file I/O, and user interruptions.
    """
    try:
        params = BrainFlowInputParams()
        params.serial_port = config["COM_PORT"]  # Use config value
        board = BoardShim(BoardIds.CYTON_DAISY_BOARD.value, params)
        board.prepare_session()
        board.start_stream()
        eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD.value)
    except FileNotFoundError as fnf:
        logging.error("Could not find BrainFlow board or driver: %s", fnf)
        return
    except (OSError, ValueError, KeyError) as e:
        logging.error("Failed to start board session: %s", e)
        return

    try:
        file_exists = os.path.isfile(OUTPUT_CSV)
        with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            header = [f"ch_{ch}" for ch in eeg_channels] + ["session_type", "label"]
            if not file_exists or os.stat(OUTPUT_CSV).st_size == 0:
                writer.writerow(header)
            logging.info("Session types: pure, jolt, hybrid, long")
            session_type = input("Enter session type: ").strip().lower()
            while session_type not in SESSION_TYPES:
                session_type = (
                    input(f"Invalid. Enter session type {SESSION_TYPES}: ")
                    .strip()
                    .lower()
                )
            logging.info("Available labels: %s", LABELS)
            label = input("Enter direction label: ").strip().lower()
            while label not in LABELS:
                label = input(f"Invalid. Enter label {LABELS}: ").strip().lower()
            if session_type == "long":
                n_trials = 1
            else:
                n_trials = TRIALS_PER_LABEL
            meta = {
                "session_type": session_type,
                "label": label,
                "n_trials": n_trials,
                "timestamps": [],
                "rows_written": 0,
            }
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
                logging.info("Collecting data for '%s' (%s)...", label, session_type)
                rows_written, timestamps = collect_eeg(
                    board, eeg_channels, session_type, label, trial, writer
                )
                meta["rows_written"] += rows_written
                meta["timestamps"].extend(timestamps)
                logging.info(
                    "Trial %d for '%s' (%s) complete.", trial + 1, label, session_type
                )
            # Save metadata
            meta_filename = f"meta_{session_type}_{label}_{int(time.time())}.json"
            try:
                with open(meta_filename, "w", encoding="utf-8") as metaf:
                    json.dump(meta, metaf, indent=2)
            except (OSError, json.JSONDecodeError) as e:
                logging.error("Failed to save metadata file %s: %s", meta_filename, e)
    except (OSError, ValueError, csv.Error, KeyError) as e:
        logging.error("Error during data collection or file writing: %s", e)
        return
    finally:
        try:
            board.stop_stream()
            board.release_session()
        except (OSError, AttributeError) as e:
            logging.error("Error releasing board session: %s", e)


if __name__ == "__main__":
    main()
