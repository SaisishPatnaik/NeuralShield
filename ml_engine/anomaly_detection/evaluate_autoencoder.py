
import os
import joblib
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt

print("\n========== NEURALSHIELD AUTOENCODER EVALUATION ==========")

# Load processed data
data = joblib.load("data/processed/prepared_data.joblib")

X_val = data["X_val"]
y_val = np.asarray(data["y_binary_val"]).reshape(-1)

X_test = data["X_test"]
y_test = np.asarray(data["y_binary_test"]).reshape(-1)

# Convert sparse matrices if necessary
def to_dense(X):
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)

X_val = to_dense(X_val)
X_test = to_dense(X_test)

# Load Autoencoder
model = load_model(
    "models/anomaly_detection/autoencoder.keras"
)

threshold = joblib.load(
    "models/anomaly_detection/threshold.joblib"
)

print("\nAnomaly threshold:", threshold)

# Calculate reconstruction errors
print("\nCalculating validation errors...")

val_reconstructed = model.predict(
    X_val,
    batch_size=256,
    verbose=1
)

val_errors = np.mean(
    np.square(X_val - val_reconstructed),
    axis=1
)

print("\nCalculating test errors...")

test_reconstructed = model.predict(
    X_test,
    batch_size=256,
    verbose=1
)

test_errors = np.mean(
    np.square(X_test - test_reconstructed),
    axis=1
)

# Classify anomalies
val_predictions = (val_errors > threshold).astype(int)
test_predictions = (test_errors > threshold).astype(int)

# Validation evaluation
print("\n========== VALIDATION RESULTS ==========")

print("Accuracy:", accuracy_score(y_val, val_predictions))
print(classification_report(
    y_val,
    val_predictions,
    target_names=["Benign", "Malicious"],
    zero_division=0
))

# Test evaluation
print("\n========== TEST RESULTS ==========")

print("Accuracy:", accuracy_score(y_test, test_predictions))

print("\nClassification Report:")

print(classification_report(
    y_test,
    test_predictions,
    target_names=["Benign", "Malicious"],
    zero_division=0
))

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
)

print("\nConfusion Matrix:")
print(cm)

# Save evaluation results
os.makedirs("models/anomaly_detection/evaluation", exist_ok=True)

pd.DataFrame(
    cm,
    index=["Actual Benign", "Actual Malicious"],
    columns=["Predicted Benign", "Predicted Malicious"]
).to_csv(
    "models/anomaly_detection/evaluation/confusion_matrix.csv"
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Benign", "Malicious"]
)

disp.plot(cmap="Blues", values_format="d")

plt.title("NeuralShield Autoencoder Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "models/anomaly_detection/evaluation/confusion_matrix.png",
    dpi=300
)

plt.show()

print("\n========== AUTOENCODER EVALUATION COMPLETE ==========")
