import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv1D, Activation, MaxPooling1D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load configuration from config.json
with open('config.json', 'r') as f:
    config = json.load(f)

N_CHANNELS = config["N_CHANNELS"]
WINDOW_SIZE = config["WINDOW_SIZE"]

# Load windowed data
X_windows = np.load(config["WINDOWED_NPY"])
y_windows = np.load(config["WINDOWED_LABELS_NPY"])

print(f"Loaded windowed data shape: {X_windows.shape}, Labels shape: {y_windows.shape}")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y_windows)
y_cat = to_categorical(y_encoded)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_windows, y_cat, test_size=0.2, random_state=42, stratify=y_cat
)

# Standardize features (fit only on training data)
scaler = StandardScaler()
X_train_flat = X_train.reshape(-1, N_CHANNELS)
scaler.fit(X_train_flat)
X_train_scaled = scaler.transform(X_train.reshape(-1, N_CHANNELS)).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test.reshape(-1, N_CHANNELS)).reshape(X_test.shape)

print(f"Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}")

# Build Conv1D model (channels last)
model = Sequential([
    Conv1D(64, kernel_size=3, input_shape=(WINDOW_SIZE, N_CHANNELS)),
    Activation('relu'),
    Conv1D(64, kernel_size=2),
    Activation('relu'),
    MaxPooling1D(pool_size=2),
    Conv1D(64, kernel_size=2),
    Activation('relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(y_cat.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train_scaled, y_train, epochs=30, batch_size=64, validation_split=0.2)

# Evaluate model
_, acc = model.evaluate(X_test_scaled, y_test)
print(f"Test accuracy: {acc:.3f}")

# Print confusion matrix and classification report
y_pred = model.predict(X_test_scaled)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)
print(confusion_matrix(y_true_labels, y_pred_labels))
print(classification_report(y_true_labels, y_pred_labels, target_names=le.classes_))

# Save model and label encoder
model.save('eeg_direction_model.h5')
joblib.dump(le, config["LABEL_ENCODER"])
joblib.dump(scaler, config["SCALER_CNN"])
np.save(config["LABEL_CLASSES_NPY"], le.classes_)

# Flatten windows for tree-based models
X_flat = X_windows.reshape(X_windows.shape[0], -1)

# Train/test split for tree-based models
X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(
    X_flat, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Standardize features (optional for trees, but keep for consistency)
scaler_tree = StandardScaler()
X_train_scaled_tree = scaler_tree.fit_transform(X_train_tree)
X_test_scaled_tree = scaler_tree.transform(X_test_tree)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled_tree, y_train_tree)
rf_pred = rf.predict(X_test_scaled_tree)
print("Random Forest Results:")
print(confusion_matrix(y_test_tree, rf_pred))
print(classification_report(y_test_tree, rf_pred, target_names=le.classes_))

# XGBoost
xgb = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
xgb.fit(X_train_scaled_tree, y_train_tree)
xgb_pred = xgb.predict(X_test_scaled_tree)
print("XGBoost Results:")
print(confusion_matrix(y_test_tree, xgb_pred))
print(classification_report(y_test_tree, xgb_pred, target_names=le.classes_))

# Save tree-based models
joblib.dump(rf, config["MODEL_RF"])
joblib.dump(xgb, config["MODEL_XGB"])
joblib.dump(le, config["LABEL_ENCODER"])
joblib.dump(scaler_tree, config["SCALER_TREE"])