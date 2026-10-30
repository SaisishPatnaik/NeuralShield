
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

print("\n========== NEURALSHIELD THRESHOLD OPTIMIZATION ==========")

data = joblib.load("data/processed/prepared_data.joblib")
model = joblib.load("models/xgboost.joblib")

X_val = data["X_val"]
y_val = data["y_binary_val"]

# Get malicious-class probabilities
probabilities = model.predict_proba(X_val)[:, 1]

results = []

# Test different classification thresholds
for threshold in np.arange(0.30, 0.81, 0.02):

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val, predictions, labels=[0, 1]
    ).ravel()

    malicious_recall = recall_score(
        y_val, predictions, zero_division=0
    )

    benign_recall = tn / (tn + fp) if (tn + fp) else 0

    precision = precision_score(
        y_val, predictions, zero_division=0
    )

    f1 = f1_score(
        y_val, predictions, zero_division=0
    )

    results.append({
        "Threshold": round(float(threshold), 2),
        "Accuracy": (tn + tp) / (tn + fp + fn + tp),
        "Malicious Recall": malicious_recall,
        "Benign Recall": benign_recall,
        "Precision": precision,
        "F1 Score": f1,
        "False Positives": fp,
        "False Negatives": fn
    })

df = pd.DataFrame(results)

print("\n========== THRESHOLD RESULTS ==========")
print(df.round(4).to_string(index=False))

# Preserve at least 95% malicious recall
eligible = df[df["Malicious Recall"] >= 0.95]

if eligible.empty:
    print("\nNo threshold meets the 95% malicious recall target.")
else:
    best = eligible.sort_values(
        ["Benign Recall", "F1 Score"],
        ascending=False
    ).iloc[0]

    print("\n========== BEST THRESHOLD ==========")
    print(best.to_string())

    joblib.dump(
        float(best["Threshold"]),
        "models/optimal_threshold.joblib"
    )

    print("\nSaved: models/optimal_threshold.joblib")

print("\n========== OPTIMIZATION COMPLETE ==========")
