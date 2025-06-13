import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Activation, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Parameters for windowed data
N_CHANNELS = 16
WINDOW_SIZE = 250

# Load windowed data
X_windows = np.load('eeg_windowed_X.npy')  # shape: (num_windows, WINDOW_SIZE, N_CHANNELS)
y_windows = np.load('eeg_windowed_y.npy')  # shape: (num_windows,)

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
history = model.fit(X_train_scaled, y_train, epochs=30, batch_size=64, validation_split=0.2)

# Evaluate model
loss, acc = model.evaluate(X_test_scaled, y_test)
print(f"Test accuracy: {acc:.3f}")

# Print confusion matrix and classification report
y_pred = model.predict(X_test_scaled)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)
print(confusion_matrix(y_true_labels, y_pred_labels))
print(classification_report(y_true_labels, y_pred_labels, target_names=le.classes_))

# Save model and label encoder
model.save('eeg_direction_model.h5')
joblib.dump(le, 'eeg_label_encoder.pkl')
joblib.dump(scaler, 'eeg_scaler.pkl')
np.save('eeg_label_classes.npy', le.classes_)

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
joblib.dump(rf, 'eeg_rf_model.pkl')
joblib.dump(xgb, 'eeg_xgb_model.pkl')
joblib.dump(le, 'eeg_label_encoder.pkl')
joblib.dump(scaler_tree, 'eeg_scaler_tree.pkl')