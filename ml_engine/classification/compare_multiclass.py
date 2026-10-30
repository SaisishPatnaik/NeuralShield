
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("\n========== NEURALSHIELD MODEL COMPARISON ==========")

data = joblib.load("data/processed/prepared_data.joblib")

X_test = data["X_test"]

encoder = joblib.load(
    "models/multiclass/label_encoder.joblib"
)

y_test = encoder.transform(data["y_multi_test"])

models = {
    "Original": "models/multiclass/xgboost_multiclass.joblib",
    "Weighted": "models/multiclass/xgboost_weighted.joblib",
    "Moderate": "models/multiclass/xgboost_moderate.joblib"
}

results = []

for name, path in models.items():

    model = joblib.load(path)
    predictions = model.predict(X_test).astype(int)

    print(f"\n========== {name.upper()} MODEL ==========")

    print("Accuracy:", accuracy_score(y_test, predictions))

    print(
        classification_report(
            y_test,
            predictions,
            labels=range(len(encoder.classes_)),
            target_names=encoder.classes_,
            zero_division=0
        )
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions)
    })

print("\n========== FINAL COMPARISON ==========")
print(pd.DataFrame(results).to_string(index=False))

print("\n========== COMPARISON COMPLETE ==========")
