# EEG Training System

This project provides a complete pipeline for collecting, processing, training, testing, and performing real-time prediction on EEG data using a Cyton Daisy board.

## Project Structure

- `collect_EEG_data.py` — Collects labeled EEG data and saves to CSV.
- `window_eeg_data.py` — Segments raw EEG data into overlapping windows for model training.
- `train_eeg_model.py` — Trains Conv1D, Random Forest, and XGBoost models on windowed data.
- `test_eeg_model.py` — Tests trained models on held-out data.
- `realtime_eeg_predict.py` — Performs real-time EEG prediction using trained models.
- `utils.py` — Utility functions for config loading and windowing.
- `config.json` — Centralized configuration for all scripts.
- `.gitignore` — Excludes large data/model files from version control.

## Setup

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

2. **Configure parameters:**
   - Edit `config.json` to set COM port, window sizes, file names, etc.

3. **Collect data:**
   ```sh
   python collect_EEG_data.py
   ```

4. **Window data:**
   ```sh
   python window_eeg_data.py
   ```

5. **Train models:**
   ```sh
   python train_eeg_model.py
   ```

6. **Test models:**
   ```sh
   python test_eeg_model.py
   ```

7. **Real-time prediction:**
   ```sh
   python realtime_eeg_predict.py
   ```

## Notes
- All configuration is centralized in `config.json`.
- Large data and model files are excluded from git via `.gitignore`.
- Logging and error handling are implemented for reliability.
- Utility functions are in `utils.py` for code reuse.

## Requirements
- Python 3.8+
- See `requirements.txt` for package list
