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
├── calibrate_session.py    # Fine-tune models for individual sessions
├── utils.py                # Config loading, windowing, logging, validation utilities
├── EEGModels.py            # Model architectures (EEGNet, etc.)
├── config.json             # Centralized configuration for all scripts
├── requirements.txt        # Python dependencies
├── test_system.py          # Unit tests for utility functions
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

Run session calibration before real-time prediction (optional but recommended):
```sh
python calibrate_session.py --calib_X data/calib_X.npy --calib_y data/calib_y.npy \
    --base_model models/eeg_direction_model.h5 --out_model models/eeg_direction_model_session.h5
```

## Configuration

The system uses a centralized configuration file (`config.json`) to manage all parameters:

### Key Configuration Parameters:
- **`COM_PORT`**: Serial port for EEG hardware communication
- **`LABELS`**: List of direction labels for classification ["forward", "backward", "left", "right", "neutral"]
- **`SESSION_TYPES`**: Types of data collection sessions ["pure", "jolt", "hybrid", "long"]
- **`WINDOW_SIZE`**: Size of EEG data windows for model training (default: 250 samples)
- **`STEP_SIZE`**: Step size for overlapping windows (default: 125 samples)
- **`N_CHANNELS`**: Number of EEG channels to use (default: 16)
- **`TRIAL_DURATION`**: Duration of each data collection trial in seconds
- **Model paths**: File paths for saving trained models, scalers, and encoders

Edit `config.json` before running the pipeline to match your hardware setup and requirements.

## Data Requirements

### EEG Hardware:
- **Compatible Boards**: BrainFlow-supported EEG devices (e.g., OpenBCI Cyton Daisy)
- **Channels**: 16-channel EEG setup (configurable in `config.json`)
- **Sampling Rate**: 250 Hz (standard for most EEG boards)

### Data Collection:
- **Session Types**: Pure (stationary), Jolt (with movement), Hybrid (mixed), Long (extended)
- **Labels**: 5 direction classes - forward, backward, left, right, neutral
- **Duration**: Configurable trial durations (default: 10 seconds per trial)
- **Quantity**: 30 trials per label recommended for good model performance

### File Structure:
```
data/
├── eeg_training_data.csv       # Raw labeled EEG data
├── eeg_windowed_X.npy         # Windowed feature data
└── eeg_windowed_y.npy         # Windowed labels

models/
├── eeg_direction_model.h5      # Trained EEGNet model
├── eeg_rf_model.pkl           # Random Forest model
├── eeg_xgb_model.pkl          # XGBoost model
├── eeg_label_encoder.pkl       # Label encoding mappings
└── eeg_scaler*.pkl            # Feature scaling parameters
```

## Troubleshooting

### Common Issues:

**EEG Hardware Connection:**
- Ensure the correct COM port is specified in `config.json`
- Check that the EEG device is properly connected and powered
- Verify BrainFlow compatibility with your EEG hardware

**Data Collection:**
- If data collection fails, check hardware connections and board initialization
- Ensure sufficient disk space for CSV data files
- Verify that the specified number of channels matches your hardware

**Model Training:**
- If training fails with memory errors, reduce batch size or window size
- Ensure windowed data files exist before running `train_eeg_model.py`
- Check that all required packages are installed: `pip install -r requirements.txt`

**Real-time Prediction:**
- Verify all model files exist in the `models/` directory
- Ensure proper session calibration before real-time prediction
- Check TensorFlow/Keras version compatibility if prediction fails

**File Not Found Errors:**
- Run scripts in order: `collect_data.py` → `window_eeg_data.py` → `train_eeg_model.py`
- Check that `data/` and `models/` directories exist and have proper permissions

## Model Performance & Evaluation

The system trains and evaluates three complementary models:

### Models:
1. **EEGNet** - Deep learning CNN specialized for EEG classification
2. **Random Forest** - Tree-based ensemble for robust predictions  
3. **XGBoost** - Gradient boosting for high accuracy

### Evaluation Methods:
- **Individual Model Accuracy**: Performance of each model separately
- **Hard Voting Ensemble**: Majority vote across all three models
- **Rule-based Ensemble**: Prioritizes EEGNet for directional commands, other models for neutral

### Metrics Reported:
- Classification accuracy on held-out test data
- Confusion matrices for detailed performance analysis
- Per-class precision, recall, and F1-scores
- Real-time prediction latency and throughput

Run `python test_eeg_model.py` to evaluate trained models on test data.

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

- Follow PEP8 and Pylint guidelines (100 character lines, docstrings, etc.)
- Add or update docstrings for new functions or scripts
- Add tests for new utility functions in `test_system.py`

## License

See the LICENSE.TXT file for details. Portions of this project are released under Creative Commons Zero 1.0 (CC0) and Apache 2.0, as described in `EEGModels.py`.

## References

- [EEGNet: A Compact Convolutional Neural Network for EEG-based Brain–Computer Interfaces](https://doi.org/10.1088/1741-2552/aace8c)
- [BrainFlow](https://brainflow.org/)
- [scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [Keras](https://keras.io/)
