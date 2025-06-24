# EEG Direction Classification & Real-Time Prediction

A complete pipeline for collecting, processing, training, evaluating, and performing real-time
prediction of EEG-based direction commands using deep learning (EEGNet) and ensemble methods.

## Features
- Collect and label raw EEG data from BrainFlow-compatible boards
- Segment and window EEG data for supervised learning
- Train EEGNet (deep learning), Random Forest, and XGBoost models
- Data augmentation and class balancing
- Evaluate models and ensemble predictions on held-out data
- Real-time direction prediction from live EEG streams
- Robust logging and data validation utilities
- Modular, configuration-driven workflow

## Project Structure
```
.
├── collect_data.py         # Collect and label raw EEG data from hardware
├── window_eeg_data.py      # Segment raw EEG CSV into overlapping windows
├── train_eeg_model.py      # Train EEGNet, Random Forest, XGBoost on windowed data
├── test_eeg_model.py       # Evaluate trained models and ensembles
├── realtime_eeg_predict.py # Real-time prediction from live EEG data
├── utils.py                # Config loading, windowing, logging, validation utilities
├── EEGModels.py            # Model architectures (EEGNet, etc.)
├── config.json             # Centralized configuration for all scripts
├── requirements.txt        # Python dependencies
├── test_utils.py           # Unit tests for utility functions
├── data/                   # Data files (CSV, .npy)
├── models/                 # Trained models, encoders, scalers
└── ...
```

## Setup & Quickstart

1. **Clone the repository:**
   ```sh
   git clone https://github.com/KyronTeves/eeg-training/tree/test-coverage-improvements
   cd eeg-training
   ```
2. **Create and activate a virtual environment (recommended):**
   ```sh
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # or
   source venv/bin/activate  # On macOS/Linux
   ```
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
4. **Edit `config.json`** to set COM port, window sizes, file names, etc.
5. **Run the pipeline:**
   ```sh
   python collect_data.py           # Collect and label EEG data
   python window_eeg_data.py        # Segment and window data
   python train_eeg_model.py        # Train all models
   python test_eeg_model.py         # Evaluate models and ensemble
   python realtime_eeg_predict.py   # Real-time prediction from live EEG
   ```

- All scripts use configuration from `config.json`.
- Data and models are saved in the `data/` and `models/` directories.
- See below for more detailed example usage and troubleshooting tips.

## Example Usage

Collect and label EEG data:
```sh
python collect_data.py
```

Segment and window data for model training:
```sh
python window_eeg_data.py
```

Train all models (EEGNet, Random Forest, XGBoost):
```sh
python train_eeg_model.py
```

Evaluate models and ensemble on test data:
```sh
python test_eeg_model.py
```

Run real-time prediction from live EEG stream:
```sh
python realtime_eeg_predict.py
```

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

- Follow PEP8 and Pylint guidelines (100 character lines, docstrings, etc.)
- Add or update docstrings for new functions or scripts
- Add tests for new utility functions in `test_utils.py`

## License

See the LICENSE.TXT file for details. Portions of this project are released under Creative Commons Zero 1.0 (CC0) and Apache 2.0, as described in `EEGModels.py`.

## References

- [EEGNet: A Compact Convolutional Neural Network for EEG-based Brain–Computer Interfaces](https://doi.org/10.1088/1741-2552/aace8c)
- [BrainFlow](https://brainflow.org/)
- [scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [Keras](https://keras.io/)
