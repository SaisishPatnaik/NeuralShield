
import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier

print("\n========== NEURALSHIELD WEIGHTED MULTICLASS ==========")

data = joblib.load("data/processed/prepared_data.joblib")

X_train = data["X_train"]
X_val = data["X_val"]

encoder = LabelEncoder()

y_train = encoder.fit_transform(data["y_multi_train"])
y_val = encoder.transform(data["y_multi_val"])

# Calculate balanced sample weights
weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

print("\nClass weights:")
for category, weight in zip(
    encoder.classes_,
    compute_sample_weight(
        class_weight="balanced",
        y=encoder.transform(encoder.classes_)
    )
):
    print(f"{category}: {weight:.3f}")

model = XGBClassifier(
    objective="multi:softprob",
    num_class=len(encoder.classes_),
    n_estimators=300,
    max_depth=6,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

print("\nTraining weighted model...")

start = time.time()

model.fit(
    X_train,
    y_train,
    sample_weight=weights
)

print(f"Training time: {time.time() - start:.2f} seconds")

predictions = model.predict(X_val).astype(int)

print("\n========== WEIGHTED MODEL RESULTS ==========")

print(f"Accuracy: {accuracy_score(y_val, predictions):.4f}")

print(
    classification_report(
        y_val,
        predictions,
        labels=range(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0
    )
)

os.makedirs("models/multiclass", exist_ok=True)

joblib.dump(
    model,
    "models/multiclass/xgboost_weighted.joblib"
)

print("\nWeighted model saved successfully.")
