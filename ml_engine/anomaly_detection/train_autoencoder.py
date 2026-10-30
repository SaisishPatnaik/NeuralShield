
import os
import joblib
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# =====================================================
# NEURALSHIELD - AUTOENCODER ANOMALY DETECTION
# =====================================================

print("\n========== NEURALSHIELD AUTOENCODER ==========")

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Paths
DATA_PATH = "data/processed/prepared_data.joblib"
MODEL_DIR = "models/anomaly_detection"

os.makedirs(MODEL_DIR, exist_ok=True)

# =====================================================
# 1. LOAD PROCESSED DATA
# =====================================================

print("\nLoading processed dataset...")

data = joblib.load(DATA_PATH)

X_train = data["X_train"]
X_val = data["X_val"]

y_train = np.asarray(data["y_binary_train"]).reshape(-1)
y_val = np.asarray(data["y_binary_val"]).reshape(-1)

print("Training dataset:", X_train.shape)
print("Validation dataset:", X_val.shape)

# =====================================================
# 2. CONVERT SPARSE MATRICES IF REQUIRED
# =====================================================

def to_dense_float32(X):
    if hasattr(X, "toarray"):
        X = X.toarray()

    return np.asarray(X, dtype=np.float32)


# =====================================================
# 3. SELECT BENIGN TRAFFIC ONLY
# =====================================================

print("\nSelecting benign traffic...")

X_train_normal = X_train[y_train == 0]
X_val_normal = X_val[y_val == 0]

X_train_normal = to_dense_float32(X_train_normal)
X_val_normal = to_dense_float32(X_val_normal)

print("Normal training records:", len(X_train_normal))
print("Normal validation records:", len(X_val_normal))

if len(X_train_normal) == 0 or len(X_val_normal) == 0:
    raise ValueError("No benign training or validation records found.")

# =====================================================
# 4. BUILD AUTOENCODER
# =====================================================

input_dim = X_train_normal.shape[1]

print("\nInput features:", input_dim)

input_layer = Input(shape=(input_dim,))

# Encoder
encoded = Dense(128, activation="relu")(input_layer)
encoded = Dropout(0.1)(encoded)
encoded = Dense(64, activation="relu")(encoded)
encoded = Dense(32, activation="relu")(encoded)

# Decoder
decoded = Dense(64, activation="relu")(encoded)
decoded = Dense(128, activation="relu")(decoded)

output_layer = Dense(
    input_dim,
    activation="linear"
)(decoded)

autoencoder = Model(
    inputs=input_layer,
    outputs=output_layer
)

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)

autoencoder.summary()

# =====================================================
# 5. TRAIN AUTOENCODER
# =====================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

print("\nTraining Autoencoder...")

history = autoencoder.fit(
    X_train_normal,
    X_train_normal,
    epochs=50,
    batch_size=256,
    validation_split=0.15,
    callbacks=[early_stop],
    shuffle=True,
    verbose=1
)

print("\nTraining completed.")

# =====================================================
# 6. CALCULATE RECONSTRUCTION ERRORS
# =====================================================

print("\nCalculating reconstruction errors...")

reconstructed = autoencoder.predict(
    X_val_normal,
    batch_size=256,
    verbose=1
)

normal_errors = np.mean(
    np.square(X_val_normal - reconstructed),
    axis=1
)

normal_errors = np.asarray(
    normal_errors
).reshape(-1)

print("Reconstruction error shape:", normal_errors.shape)

# =====================================================
# 7. DETERMINE ANOMALY THRESHOLD
# =====================================================

# 95th percentile of benign validation errors
threshold = float(
    np.percentile(normal_errors, 95)
)

print("\n========== ANOMALY THRESHOLD ==========")

print("95th percentile threshold:", threshold)
print("Mean reconstruction error:", np.mean(normal_errors))
print("Maximum reconstruction error:", np.max(normal_errors))

# =====================================================
# 8. SAVE MODEL AND THRESHOLD
# =====================================================

model_path = os.path.join(
    MODEL_DIR,
    "autoencoder.keras"
)

threshold_path = os.path.join(
    MODEL_DIR,
    "threshold.joblib"
)

history_path = os.path.join(
    MODEL_DIR,
    "training_history.joblib"
)

autoencoder.save(model_path)

joblib.dump(
    threshold,
    threshold_path
)

joblib.dump(
    history.history,
    history_path
)

# =====================================================
# 9. FINAL REPORT
# =====================================================

print("\n========== NEURALSHIELD AUTOENCODER REPORT ==========")

print("Training records:", len(X_train_normal))
print("Validation records:", len(X_val_normal))
print("Input features:", input_dim)
print("Epochs completed:", len(history.history["loss"]))
print("Final training loss:", history.history["loss"][-1])
print("Final validation loss:", history.history["val_loss"][-1])
print("Anomaly threshold:", threshold)

print("\nSaved files:")
print("1.", model_path)
print("2.", threshold_path)
print("3.", history_path)

print("\n========== AUTOENCODER TRAINING COMPLETE ==========")
