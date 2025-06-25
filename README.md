# EEG Direction Classification & Real-Time Prediction

> **⚠️ WORK IN PROGRESS**: This system is currently under active development and is not yet working as intended. Features may be incomplete, unstable, or subject to significant changes. Use for research and development purposes only.

## 🚀 EEG Classification System

**EEG-based direction classification system with LSL streaming:**

### ✅ Current Features:
- **LSL Streaming**: Integration with OpenBCI GUI for pre-filtered data
- **Multiple Models**: EEGNet, ShallowConvNet, Random Forest, and XGBoost
- **Real-time Pipeline**: Optimized prediction pipeline (in development)
- **Configuration-Driven**: Centralized parameter management via config.json
- **Comprehensive Logging**: Built-in monitoring and validation

### 🔧 Development Status:
- **Data Collection**: ✅ Functional via LSL streaming
- **Model Training**: ✅ Multiple architectures supported  
- **Real-time Prediction**: 🚧 Under development
- **Performance Optimization**: 🚧 Work in progress
- **Documentation**: 🚧 Being updated

---

A complete pipeline for collecting, processing, training, evaluating, and performing real-time
prediction of EEG-based direction commands using LSL streaming and advanced neural networks.

## Features
- **LSL Streaming**: Collect from OpenBCI GUI with professional-grade filtering
- **Advanced Models**: EEGNet, ShallowConvNet, Random Forest, and XGBoost
- **Pre-filtered Data**: Use OpenBCI GUI's professional filtering for better signal quality
- **Optimized Pipeline**: High-performance real-time prediction with confidence thresholding
- **Data Augmentation**: Automatic class balancing and realistic noise injection
- **Ensemble Methods**: Hard voting and rule-based ensemble predictions
- **Real-time Monitoring**: Performance stats, latency tracking, and confidence scoring
- **Session Calibration**: Fine-tune models for individual users and sessions
- **Robust Validation**: Comprehensive logging and data quality checks
- **Modular Design**: Configuration-driven workflow with centralized settings

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
- OpenBCI GUI (required for LSL streaming)

### Installation

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

### LSL Streaming Setup

#### OpenBCI GUI Setup
1. **Start OpenBCI GUI** and connect to your Cyton board
2. **Configure Filters** (go to "Filtering" tab):
   - **Bandpass Filter**: 1-50 Hz (motor imagery frequency range)
   - **Notch Filter**: 50 Hz or 60 Hz (remove power line noise)
   - **High Pass**: 1 Hz (remove DC drift)
   - **Low Pass**: 50 Hz (remove high-frequency artifacts)
3. **Start LSL Streaming** (go to "Networking" tab):
   - Enable "Stream Data via LSL"
   - Set stream name to "OpenBCIGUI"
   - Click "Start LSL Stream"

#### Configure System
Update your `config.json`:
```json
{
  "LSL_STREAM_NAME": "OpenBCIGUI",
  "LSL_TIMEOUT": 10.0,
  "CONFIDENCE_THRESHOLD": 0.7,
  "MODELS_TO_TRAIN": ["EEGNet", "ShallowConvNet"]
}
```

#### Run Pipeline
```sh
python collect_data.py          # Collect pre-filtered data from OpenBCI GUI
python window_eeg_data.py       # Segment and window data
python train_eeg_model.py       # Train models (EEGNet + ShallowConvNet)
python test_eeg_model.py        # Evaluate models and ensemble
python realtime_eeg_predict.py  # Real-time prediction with optimization
```

### Current Development Status

| Component | Status | Notes |
|-----------|--------|-------|
| **Data Collection** | ✅ Working | LSL streaming integration complete |
| **Data Processing** | ✅ Working | Windowing and feature extraction functional |
| **Model Training** | ✅ Working | Multiple models supported and configurable |
| **Model Evaluation** | ✅ Working | Comprehensive testing and validation |
| **Real-time Prediction** | 🚧 In Progress | Basic functionality present, optimization ongoing |
| **Performance Optimization** | 🚧 In Progress | Pipeline improvements under development |

- All scripts use configuration from `config.json`.
- Data and models are saved in the `data/` and `models/` directories.
- **LSL streaming**: Provides better data quality through pre-filtering
- See troubleshooting section below for common issues and solutions.

## Real-time Performance Monitoring

The system provides real-time performance feedback:

```
[  50] ✓ FORWARD  (conf: 0.834)
[  51] ? NEUTRAL  (conf: 0.432)
[  52] ✓ LEFT     (conf: 0.891)
Performance: 12.3ms avg, 81.2 FPS
```

- `✓` = High confidence prediction (above threshold)
- `?` = Low confidence (likely neutral state)
- Performance stats displayed every 50 predictions

## Example Usage

### Data Collection
```sh
# Start OpenBCI GUI with LSL streaming enabled
python collect_data.py
```

Segment and window data for model training:
```sh
python window_eeg_data.py
```

Train multiple models with configurable parameters:
```sh
python train_eeg_model.py
```

Example training output:
```
=== Training Results ===
EEGNet Test accuracy: 0.75-0.85 (typical range)
ShallowConvNet Test accuracy: 0.72-0.88 (varies by user)
Random Forest accuracy: 0.70-0.82 (feature-dependent)
XGBoost accuracy: 0.73-0.85 (consistent performer)
```

Evaluate models and compare performance:
```sh
python test_eeg_model.py
```

Run real-time prediction (⚠️ under development):
```sh
python realtime_eeg_predict.py
```

Optional session calibration for individual users:
```sh
python calibrate_session.py
```

## Configuration

The system uses a centralized configuration file (`config.json`) to manage all parameters. The configuration is organized into logical sections:

### Core Configuration Sections:

#### **LSL Streaming & Hardware:**
- **`LSL_STREAM_NAME`**: Name of LSL stream (default: "OpenBCIGUI")
- **`LSL_TIMEOUT`**: Connection timeout for LSL streams (default: 10.0)
- **`SAMPLING_RATE`**: EEG sampling rate in Hz (default: 250)
- **`N_CHANNELS`**: Number of EEG channels (default: 16)

#### **Data Collection:**
- **`LABELS`**: Direction labels ["forward", "backward", "left", "right", "neutral"]
- **`SESSION_TYPES`**: Collection sessions ["pure", "jolt", "hybrid", "long"]
- **`TRIAL_DURATION`**: Duration per trial in seconds (default: 10)
- **`TRIALS_PER_LABEL`**: Number of trials per label (default: 30)

#### **Data Processing:**
- **`WINDOW_SIZE`**: EEG data window size in samples (default: 250)
- **`STEP_SIZE`**: Step size for overlapping windows (default: 125)
- **`USE_SESSION_TYPES`**: Sessions to use for training
- **`TEST_SESSION_TYPES`**: Sessions to use for testing

#### **Model Training:**
- **`MODELS_TO_TRAIN`**: Models to train ["EEGNet", "ShallowConvNet"]
- **`EPOCHS`**: Training epochs (default: 100)
- **`BATCH_SIZE`**: Training batch size (default: 64)
- **`VALIDATION_SPLIT`**: Validation data fraction (default: 0.2)
- **`OPTIMIZER`**: Training optimizer (default: "adam")
- **`LOSS_FUNCTION`**: Loss function (default: "categorical_crossentropy")

#### **EEGNet Hyperparameters:**
- **`EEGNET_F1`**: Temporal filters (default: 16)
- **`EEGNET_D`**: Spatial filters per temporal filter (default: 2)
- **`EEGNET_F2`**: Pointwise filters (default: 32)
- **`EEGNET_KERN_LENGTH`**: Temporal kernel length (default: 125)
- **`EEGNET_DROPOUT_RATE`**: Dropout rate (default: 0.5)

#### **Session Calibration:**
- **`CALIB_EPOCHS`**: Fine-tuning epochs (default: 5)
- **`CALIB_BATCH_SIZE`**: Calibration batch size (default: 32)
- **`CALIB_X_NPY`**, **`CALIB_Y_NPY`**: Calibration data file paths

#### **Real-time Prediction:**
- **`CONFIDENCE_THRESHOLD`**: Minimum confidence for predictions (default: 0.7)
- **`USE_OPTIMIZED_PIPELINE`**: Enable optimization features (default: true)
- **`ENABLE_MODEL_QUANTIZATION`**: Enable model quantization (default: true)

#### **File Paths:**
- **Data Files**: `OUTPUT_CSV`, `WINDOWED_NPY`, `WINDOWED_LABELS_NPY`
- **Base Models**: `MODEL_CNN`, `MODEL_SHALLOW`, `MODEL_RF`, `MODEL_XGB`
- **Session Models**: `MODEL_CNN_SESSION`, `MODEL_RF_SESSION`, `MODEL_XGB_SESSION`
- **Scalers & Encoders**: `SCALER_CNN`, `SCALER_TREE`, `LABEL_ENCODER`, etc.

### Configuration Example:
```json
{
  "LSL_STREAM_NAME": "OpenBCIGUI",
  "CONFIDENCE_THRESHOLD": 0.7,
  "MODELS_TO_TRAIN": ["EEGNet", "ShallowConvNet"],
  "EPOCHS": 100,
  "BATCH_SIZE": 64,
  "EEGNET_F1": 16,
  "EEGNET_D": 2
}
```

Edit `config.json` before running the pipeline to match your hardware setup and requirements.

## Data Requirements

### EEG Hardware:
- **Compatible Boards**: OpenBCI Cyton and other LSL-compatible EEG devices
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
├── eeg_training_data_lsl.csv   # EEG data (pre-filtered via LSL)
├── eeg_windowed_X.npy         # Windowed feature data
├── eeg_windowed_y.npy         # Windowed labels
├── calib_X.npy                # Calibration data (optional)
└── calib_y.npy                # Calibration labels (optional)

models/
├── eeg_direction_model.h5      # Trained EEGNet model
├── eeg_shallow_model.h5        # Trained ShallowConvNet model
├── eeg_rf_model.pkl           # Random Forest model
├── eeg_xgb_model.pkl          # XGBoost model
├── eeg_label_encoder.pkl       # Label encoding mappings
├── eeg_scaler*.pkl            # Feature scaling parameters
└── *_session.*                # Session-specific models (optional)
```

## Troubleshooting

### Common Issues:

**LSL Stream Connection:**
- Ensure OpenBCI GUI is running with LSL streaming enabled
- Check that stream name matches config (`"LSL_STREAM_NAME": "OpenBCIGUI"`)
- Verify `pylsl` is installed: `pip install pylsl`
- Confirm filters are configured in OpenBCI GUI before streaming

**EEG Hardware Connection:**
- Ensure OpenBCI GUI is connected to your EEG device
- Check that LSL streaming is properly configured and active
- Verify hardware compatibility with OpenBCI GUI
- Confirm all electrodes have good contact

**Performance Issues:**
- **Slow predictions**: Ensure optimized pipeline is enabled in config
- **High latency**: Verify LSL streaming is working properly
- **CPU usage**: Close unnecessary applications, enable OpenBCI GUI filtering
- **Memory errors**: Reduce batch size or window size in config

**Data Collection:**
- **Data quality issues**: Retrain models if switching data sources
- **Missing files**: Run `collect_data.py` with LSL enabled before windowing and training
- **Channel mismatch**: Verify N_CHANNELS matches your hardware setup
- **Insufficient data**: Collect 30+ trials per label for good performance

**Model Training:**
- **Low accuracy**: Try ShallowConvNet instead of EEGNet for motor imagery
- **Training fails**: Check that windowed data files exist
- **Version errors**: Ensure TensorFlow/Keras compatibility: `pip install -r requirements.txt`
- **Memory issues**: Reduce batch size in training script

**Real-time Prediction:**
- **False positives**: Increase confidence threshold (0.5 → 0.8)
- **Model not found**: Verify all model files exist in `models/` directory  
- **Poor electrode contact**: Check signal quality in OpenBCI GUI
- **Inconsistent results**: Run session calibration for user-specific tuning

## Model Performance & Evaluation

The system trains and evaluates multiple complementary models:

### Available Models:
1. **EEGNet** - Compact CNN specialized for EEG classification
2. **ShallowConvNet** - CNN architecture designed for motor imagery tasks  
3. **Random Forest** - Tree-based ensemble for robust predictions
4. **XGBoost** - Gradient boosting for feature-based classification

### Development Status:
- **Model Training**: ✅ All models implemented and configurable
- **Performance Evaluation**: ✅ Comprehensive testing framework
- **Real-time Integration**: 🚧 Under development
- **Optimization Pipeline**: 🚧 Work in progress

### Target Performance:
- **Classification Accuracy**: 70-85% (varies by user and data quality)
- **Real-time Latency**: Target <50ms (currently in development)
- **Confidence Filtering**: Reduces false positives on neutral states
- **Model Comparison**: Framework for selecting best model per user

### Evaluation Methods:
- **Individual Model Accuracy**: Performance of each model separately
- **Cross-validation**: Robust performance estimation across data splits
- **Confusion Matrices**: Detailed per-class performance analysis
- **Real-time Metrics**: Latency, confidence scores, and prediction stability (in development)

### Key Metrics:
- **Accuracy**: Overall classification performance
- **Precision/Recall**: Per-class performance (important for unbalanced data)
- **Confidence Scores**: Prediction reliability (helps reduce false positives)
- **Inference Time**: Important for real-time BCI applications

Run `python test_eeg_model.py` to evaluate trained models on test data.

## Migration from Previous Versions

If you have existing training data from previous versions:

### Option 1: Fresh Start (Recommended)
```bash
# 1. Collect new LSL data (pre-filtered, better quality)
python collect_data.py

# 2. Process and train on new data  
python window_eeg_data.py
python train_eeg_model.py

# 3. Enjoy improved performance
python realtime_eeg_predict.py
```

### Option 2: Keep Existing Data
- Continue using your existing data files
- Compare performance between old and new approaches
- Gradually migrate to LSL method for better results

### Why LSL?
- **Filter Quality**: Professional-grade filtering from OpenBCI GUI
- **Signal Characteristics**: Consistent amplitude ranges and frequency content
- **Performance**: Training data matches real-time prediction data exactly

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

- Follow PEP8 and Pylint guidelines (100 character lines, docstrings, etc.)
- Add or update docstrings for new functions or scripts
- Add tests for new utility functions in `test_system.py`
- Update README for new features or breaking changes

## License

See the LICENSE.TXT file for details. Portions of this project are released under Creative Commons Zero 1.0 (CC0) and Apache 2.0, as described in `EEGModels.py`.

## References

- [EEGNet: A Compact Convolutional Neural Network for EEG-based Brain–Computer Interfaces](https://doi.org/10.1088/1741-2552/aace8c)
- [Deep Learning with Convolutional Neural Networks for EEG Decoding and Visualization](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730) (ShallowConvNet)
- [Lab Streaming Layer (LSL)](https://labstreaminglayer.readthedocs.io/)
- [OpenBCI Documentation](https://docs.openbci.com/)
- [scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [Keras](https://keras.io/)
