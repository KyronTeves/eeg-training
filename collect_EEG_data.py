import time
import csv
import os
import json
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter

# Settings
LABELS = ['forward', 'backward', 'left', 'right', 'neutral']
SESSION_TYPES = ['pure', 'jolt', 'hybrid', 'long']
TRIAL_DURATION = 10  # seconds per trial (for pure/hybrid)
JOLT_TOTAL_DURATION = 11  # 5s neutral, 1s direction, 5s neutral
HYBRID_TOTAL_DURATION = 10  # 5s neutral, 5s direction
LONG_DURATION = 180  # 3 minutes for long session
TRIALS_PER_LABEL = 30
OUTPUT_CSV = 'eeg_training_data.csv'


def collect_eeg(
    board: BoardShim,
    eeg_channels: list,
    session_type: str,
    label: str,
    trial_num: int,
    output_writer: csv.writer
) -> tuple[int, list[float]]:
    """
    Collect EEG data for a given session type and label, write to CSV.
    Returns: (number of rows written, list of timestamps)
    """
    def run_phase(phase_duration, phase_label):
        print(f"Think '{phase_label}' for {phase_duration} seconds.")
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
        'pure': [(TRIAL_DURATION, label)],
        'jolt': [(5, 'neutral'), (1, label), (5, 'neutral')],
        'hybrid': [(5, 'neutral'), (5, label)],
        'long': [(LONG_DURATION, label)]
    }

    for phase_duration, phase_label in session_phases.get(session_type, []):
        run_phase(phase_duration, phase_label)

    for row in rows:
        output_writer.writerow(row)
    return len(rows), timestamps


def main():
    params = BrainFlowInputParams()
    params.serial_port = 'COM8'  # Change to your Cyton's COM port

    board = BoardShim(BoardIds.CYTON_DAISY_BOARD.value, params)
    board.prepare_session()
    board.start_stream()
    eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD.value)

    file_exists = os.path.isfile(OUTPUT_CSV)
    with open(OUTPUT_CSV, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = [f'ch_{ch}' for ch in eeg_channels] + ['session_type', 'label']
        if not file_exists or os.stat(OUTPUT_CSV).st_size == 0:
            writer.writerow(header)
        print("Session types: pure, jolt, hybrid, long")
        session_type = input("Enter session type: ").strip().lower()
        while session_type not in SESSION_TYPES:
            session_type = input(f"Invalid. Enter session type {SESSION_TYPES}: ").strip().lower()
        print(f"Available labels: {LABELS}")
        label = input("Enter direction label: ").strip().lower()
        while label not in LABELS:
            label = input(f"Invalid. Enter label {LABELS}: ").strip().lower()
        if session_type == 'long':
            n_trials = 1
        else:
            n_trials = TRIALS_PER_LABEL
        meta = {
            'session_type': session_type,
            'label': label,
            'n_trials': n_trials,
            'timestamps': [],
            'rows_written': 0
        }
        for trial in range(n_trials):
            print(f"\nGet ready for '{label}' - Trial {trial+1}/{n_trials} ({session_type})...")
            for sec in range(3, 0, -1):
                print(f"Starting in {sec}...", end='\r', flush=True)
                time.sleep(1)
            print(" " * 20, end='\r')
            print(f"Collecting data for '{label}' ({session_type})...")
            rows_written, timestamps = collect_eeg(board, eeg_channels, session_type, label, trial, writer)
            meta['rows_written'] += rows_written
            meta['timestamps'].extend(timestamps)
            print(f"Trial {trial+1} for '{label}' ({session_type}) complete.")
        # Save metadata
        meta_filename = f"meta_{session_type}_{label}_{int(time.time())}.json"
        with open(meta_filename, 'w') as metaf:
            json.dump(meta, metaf, indent=2)

    board.stop_stream()
    board.release_session()
    print(f"\nData collection complete. Saved to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
