
import os
import time
import joblib
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier

print("\n========== NEURALSHIELD MULTICLASS TRAINING ==========")

data = joblib.load("data/processed/prepared_data.joblib")

X_train = data["X_train"]
X_val = data["X_val"]

y_train_raw = data["y_multi_train"]
y_val_raw = data["y_multi_val"]

# Encode attack categories into numeric labels
encoder = LabelEncoder()

y_train = encoder.fit_transform(y_train_raw)
y_val = encoder.transform(y_val_raw)

print("\nAttack categories:")
for i, category in enumerate(encoder.classes_):
    print(f"{i}: {category}")

model = XGBClassifier(
    objective="multi:softprob",
    num_class=len(encoder.classes_),
    n_estimators=250,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

print("\nTraining XGBoost multiclass model...")

start = time.time()
model.fit(X_train, y_train)
training_time = time.time() - start

print(f"\nTraining time: {training_time:.2f} seconds")

predictions = model.predict(X_val).astype(int)

print("\n========== VALIDATION RESULTS ==========")

print(f"Accuracy: {accuracy_score(y_val, predictions):.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_val,
        predictions,
        labels=range(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, predictions))

os.makedirs("models/multiclass", exist_ok=True)

joblib.dump(
    model,
    "models/multiclass/xgboost_multiclass.joblib"
)

joblib.dump(
    encoder,
    "models/multiclass/label_encoder.joblib"
)

print("\nSaved multiclass model and label encoder.")
print("\n========== MULTICLASS TRAINING COMPLETE ==========")
