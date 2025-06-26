# EEG Direction Classification & Real-Time Prediction

> **⚠️ WORK IN PROGRESS**: This system is under active development. Features may be incomplete, unstable, or subject to change. Use for research and development purposes only. Real-time prediction and model integration are experimental and not guaranteed to be robust.

## Overview

A modular pipeline for collecting, processing, training, evaluating, and performing real-time prediction of EEG-based direction commands using LSL streaming and multiple machine learning models.

---

## Features
- **LSL Streaming**: Collect EEG data from OpenBCI GUI (pre-filtered)
- **Multiple Models**: EEGNet, ShallowConvNet, Random Forest, XGBoost
- **Optimized Pipeline**: Modular, configuration-driven, and supports session calibration
- **Comprehensive Logging**: Real-time performance stats and error reporting
- **Data Augmentation**: Class balancing and noise injection (optional)
- **Ensemble Methods**: Weighted voting across all models
- **Session Calibration**: Optional per-user fine-tuning

## Project Structure
```
.
├── collect_data.py         # Collect EEG data via LSL streaming from OpenBCI GUI
├── window_eeg_data.py      # Segment raw EEG CSV into overlapping windows
├── train_eeg_model.py      # Train EEGNet, ShallowConvNet, RF, XGBoost models
├── test_eeg_model.py       # Evaluate trained models and ensembles
├── realtime_eeg_predict.py # Real-time prediction with LSL streaming & optimized pipeline
├── calibrate_session.py    # Fine-tune models for individual sessions
├── lsl_stream_handler.py   # LSL streaming interface for OpenBCI GUI
├── utils.py                # Config loading, windowing, logging, validation utilities
├── EEGModels.py            # Model architectures (EEGNet, ShallowConvNet, etc.)
├── config.json             # Centralized configuration for all scripts
├── requirements.txt        # Python dependencies (includes pylsl)
├── test_system.py          # Unit tests for utility functions
├── data/                   # Data files (CSV, .npy)
├── models/                 # Trained models, encoders, scalers
└── ...
```

## Setup & Quickstart

### Prerequisites
- Python 3.8+
- OpenBCI Cyton board (or compatible EEG device)
- OpenBCI GUI (for LSL streaming)

### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/KyronTeves/eeg-training.git
   cd eeg-training
   ```
2. Create and activate a virtual environment:
   ```sh
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # or
   source venv/bin/activate  # On macOS/Linux
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

### LSL Streaming Setup
- Start OpenBCI GUI and connect to your board
- Configure filters (e.g., 1-50 Hz bandpass, 50/60 Hz notch)
- Enable LSL streaming in the GUI (stream name: "OpenBCIGUI")

### Configure System
- Edit `config.json` to match your hardware and preferences

### Run Pipeline
```sh
python collect_data.py          # Collect data
python window_eeg_data.py       # Segment and window data
python train_eeg_model.py       # Train models
python test_eeg_model.py        # Evaluate models
python realtime_eeg_predict.py  # Real-time prediction (experimental)
```

## Development Status

| Component              | Status         | Notes                                  |
|-----------------------|----------------|----------------------------------------|
| Data Collection       | ✅ Working     | LSL streaming integration complete     |
| Data Processing       | ✅ Working     | Windowing and feature extraction       |
| Model Training        | ✅ Working     | Multiple models supported              |
| Model Evaluation      | ✅ Working     | Testing and validation implemented     |
| Real-time Prediction  | 🚧 In Progress| Experimental, not production-ready     |
| Performance Optimization | 🚧 Ongoing  | Pipeline improvements in development   |

- All scripts use centralized configuration (`config.json`)
- Data and models are saved in `data/` and `models/` directories
- Real-time prediction and ensemble logic are experimental

## Real-time Output Example
```
[  42] ✓ LEFT     (ens:0.569) | EEG:lef(0.443) SH:lef(0.512) RF:lef(0.510) XGB:lef(0.766)
```
- `✓` = High confidence prediction
- `?` = Low confidence (neutral or uncertain)
- Model confidences shown for each model

## Troubleshooting
- Ensure OpenBCI GUI is running and LSL streaming is enabled
- Check that `pylsl` is installed and stream name matches config
- For model errors, retrain using `train_eeg_model.py` and check data quality
- For performance issues, adjust window size, batch size, or confidence threshold in config

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss.
- Follow PEP8 and Pylint guidelines
- Add or update docstrings for new functions
- Add tests for new utilities in `test_system.py`
- Update README for new features or changes

## License
See LICENSE.TXT for details. Portions of this project are released under Creative Commons Zero 1.0 (CC0) and Apache 2.0, as described in `EEGModels.py`.

## References
- [EEGNet: A Compact Convolutional Neural Network for EEG-based Brain–Computer Interfaces](https://doi.org/10.1088/1741-2552/aace8c)
- [Deep Learning with Convolutional Neural Networks for EEG Decoding and Visualization](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730)
- [Lab Streaming Layer (LSL)](https://labstreaminglayer.readthedocs.io/)
- [OpenBCI Documentation](https://docs.openbci.com/)
- [scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [Keras](https://keras.io/)
