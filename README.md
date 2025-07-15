# EEG Direction Classification & Real-Time Prediction

[![Python 3.10.11](https://img.shields.io/badge/python-3.10.11-blue.svg)](https://www.python.org/downloads/release/python-31011/)

> **⚠️ WORK IN PROGRESS:** This system is under active development. Features may be incomplete or unstable. Use for research and development only. Real-time prediction and model integration are experimental.

---

## Overview

Modular pipeline for collecting, processing, training, evaluating, and performing real-time prediction of EEG-based direction commands using LSL streaming and multiple machine learning models.

---

## Table of Contents

- [EEG Direction Classification \& Real-Time Prediction](#eeg-direction-classification--real-time-prediction)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Project Structure](#project-structure)
  - [Setup \& Quickstart](#setup--quickstart)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [LSL Streaming Setup](#lsl-streaming-setup)
    - [Configure System](#configure-system)
    - [Run Pipeline](#run-pipeline)
  - [Development Status](#development-status)
  - [Real-time Output Example](#real-time-output-example)
  - [Troubleshooting](#troubleshooting)
  - [References](#references)

## Features

- **LSL Streaming:** Collect EEG data from OpenBCI GUI (pre-filtered)
- **Multiple Models:** EEGNet, ShallowConvNet, Random Forest, XGBoost, and advanced Conv1D neural network
- **Optimized Pipeline:** Modular, config-driven, supports session calibration
- **Comprehensive Logging:** Real-time stats and error reporting
- **Data Augmentation:** Class balancing, noise injection (optional)
- **Ensemble Methods:** Ensemble (hard voting) across models; all trained models are tracked in `models/ensemble_info.json`
- **Session Calibration:** Optional per-user fine-tuning
- **Experimental Real-Time Prediction:** Online ensemble prediction (experimental)

## Project Structure

```text
.
├── collect_data.py         # Collect EEG data via LSL streaming
├── window_eeg_data.py      # Segment raw EEG into overlapping windows
├── train_eeg_model.py      # Train EEGNet, ShallowConvNet, RF, XGBoost, Conv1D
├── test_eeg_model.py       # Evaluate trained models and ensembles
├── realtime_eeg_predict.py # Real-time prediction (experimental)
├── lsl_stream_handler.py   # LSL streaming interface
├── utils.py                # Config, windowing, logging, validation
├── EEGModels.py            # Model architectures
├── config.json             # Centralized configuration
├── requirements.txt        # Python dependencies

├── data/                   # Data files
├── models/                 # Trained models, encoders, scalers, ensemble_info.json
└── ...
```

## Setup & Quickstart

### Prerequisites

- Python 3.10.11
- OpenBCI Cyton board (or compatible EEG device)
- OpenBCI GUI (for LSL streaming)

### Installation

1. **Clone the repository:**

   ```sh
   git clone https://github.com/KyronTeves/eeg-training.git
   cd eeg-training
   ```

2. **Create and activate a virtual environment:**

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

### LSL Streaming Setup

1. Start OpenBCI GUI and connect to your board
2. Configure filters (e.g., 1-50 Hz bandpass, 50/60 Hz notch)
3. Enable LSL streaming (default stream: `obci_eeg1`, use TimeSeriesFilt)

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

| Component                | Status         | Notes                                |
|--------------------------|----------------|--------------------------------------|
| Data Collection          | ✅ Working     | LSL streaming integration complete   |
| Data Processing          | ✅ Working     | Windowing, feature extraction        |
| Model Training           | ✅ Working     | Multiple models supported            |
| Model Evaluation         | ✅ Working     | Testing and validation implemented   |
| Real-time Prediction     | 🚧 In Progress | Experimental, not production-ready   |
| Performance Optimization | 🚧 Ongoing     | Pipeline improvements in development |

- All scripts use centralized configuration (`config.json`).
- Data and models are saved in `data/` and `models/`.
- Real-time prediction and ensemble logic are experimental; session calibration is interactive in `realtime_eeg_predict.py`.

## Real-time Output Example

```text
[  42] ✓ LEFT     (ens:0.569) | EEG:lef(0.443) SH:lef(0.512) RF:lef(0.510) XGB:lef(0.766)
```

- `✓` = High confidence prediction
- `?` = Low confidence (neutral/uncertain)
- Model confidences shown for each model

## Troubleshooting

- Ensure OpenBCI GUI is running and LSL streaming is enabled
- Check that `pylsl` is installed and stream name matches config
- For model errors, retrain using `train_eeg_model.py` and check data quality
- For performance issues, adjust window size, batch size, or confidence threshold in config

## References

- [EEGNet: A Compact Convolutional Neural Network for EEG-based Brain–Computer Interfaces](https://doi.org/10.1088/1741-2552/aace8c)
- [Deep Learning with Convolutional Neural Networks for EEG Decoding and Visualization](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730)
- [Lab Streaming Layer (LSL)](https://labstreaminglayer.readthedocs.io/)
- [OpenBCI Documentation](https://docs.openbci.com/)
- [scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [Keras](https://keras.io/)
