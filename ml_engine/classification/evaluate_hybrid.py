
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from hybrid_detector import NeuralShieldDetector


print("\n========== NEURALSHIELD HYBRID EVALUATION ==========")

# Load test dataset
test_data = pd.read_csv(
    "data/raw/UNSW_NB15_testing-set.csv"
)

y_true = test_data["label"].values

# Initialize detector
detector = NeuralShieldDetector()

# Predict entire test dataset
print("\nRunning hybrid predictions...")

results = detector.predict(test_data)

y_binary = results["binary_prediction"].values
y_anomaly = results["anomaly_prediction"].values
y_hybrid = results["hybrid_prediction"].values

# Evaluation function
def evaluate_model(name, predictions):

    print(f"\n========== {name} ==========")

    print("Accuracy:",
          round(accuracy_score(y_true, predictions), 4))

    print("Precision:",
          round(precision_score(y_true, predictions, zero_division=0), 4))

    print("Recall:",
          round(recall_score(y_true, predictions, zero_division=0), 4))

    print("F1 Score:",
          round(f1_score(y_true, predictions, zero_division=0), 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, predictions))

# Evaluate individual models
evaluate_model("BINARY XGBOOST", y_binary)

evaluate_model("AUTOENCODER", y_anomaly)

evaluate_model("HYBRID DETECTOR", y_hybrid)

# Save predictions
results["actual_label"] = y_true

results.to_csv(
    "models/hybrid_evaluation_results.csv",
    index=False
)

print("\nResults saved to models/hybrid_evaluation_results.csv")

print("\n========== EVALUATION COMPLETE ==========")
