
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("\n========== NEURALSHIELD MULTICLASS TEST ==========")

data = joblib.load("data/processed/prepared_data.joblib")

model = joblib.load(
    "models/multiclass/xgboost_moderate.joblib"
)

encoder = joblib.load(
    "models/multiclass/label_encoder.joblib"
)

X_test = data["X_test"]

y_test = encoder.transform(data["y_multi_test"])

predictions = model.predict(X_test).astype(int)

print("\nTest Accuracy:")
print(accuracy_score(y_test, predictions))

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        labels=range(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\n========== TEST EVALUATION COMPLETE ==========")
