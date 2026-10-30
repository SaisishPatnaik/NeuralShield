
import os
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt

print("\n========== NEURALSHIELD FINAL EVALUATION ==========")

data = joblib.load("data/processed/prepared_data.joblib")
model = joblib.load("models/xgboost.joblib")

X_test = data["X_test"]
y_test = data["y_binary_test"]

print("Evaluating XGBoost on test data...")

predictions = model.predict(X_test)

print("\n========== TEST RESULTS ==========")
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")

print("\n========== CLASSIFICATION REPORT ==========")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["Benign", "Malicious"],
        zero_division=0
    )
)

cm = confusion_matrix(y_test, predictions)

print("\n========== CONFUSION MATRIX ==========")
print(cm)

os.makedirs("models/evaluation", exist_ok=True)

pd.DataFrame(
    cm,
    index=["Actual Benign", "Actual Malicious"],
    columns=["Predicted Benign", "Predicted Malicious"]
).to_csv("models/evaluation/confusion_matrix.csv")

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Benign", "Malicious"]
)

disp.plot(cmap="Blues", values_format="d")
plt.title("NeuralShield - XGBoost Confusion Matrix")
plt.tight_layout()
plt.savefig("models/evaluation/confusion_matrix.png", dpi=300)
plt.show()

print("\nEvaluation files saved in models/evaluation/")
print("\n========== EVALUATION COMPLETE ==========")
