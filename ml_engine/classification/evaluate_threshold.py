
import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("\n========== NEURALSHIELD OPTIMIZED EVALUATION ==========")

data = joblib.load("data/processed/prepared_data.joblib")
model = joblib.load("models/xgboost.joblib")
threshold = joblib.load("models/optimal_threshold.joblib")

X_test = data["X_test"]
y_test = data["y_binary_test"]

# Predict probabilities
probabilities = model.predict_proba(X_test)[:, 1]

# Compare default and optimized thresholds
for current_threshold in [0.50, threshold]:

    predictions = (
        probabilities >= current_threshold
    ).astype(int)

    print(f"\n========== THRESHOLD: {current_threshold:.2f} ==========")

    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Benign", "Malicious"],
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

print("\n========== EVALUATION COMPLETE ==========")
