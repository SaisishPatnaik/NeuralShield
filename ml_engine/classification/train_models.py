
import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier

DATA_PATH = "data/processed/prepared_data.joblib"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("\n========== NEURALSHIELD MODEL TRAINING ==========")

data = joblib.load(DATA_PATH)

X_train = data["X_train"]
X_val = data["X_val"]
y_train = data["y_binary_train"]
y_val = data["y_binary_val"]

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
}

results = []

for name, model in models.items():
    print(f"\nTraining {name}...")

    start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start

    predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(y_val, predictions, zero_division=0)
    recall = recall_score(y_val, predictions, zero_division=0)
    f1 = f1_score(y_val, predictions, zero_division=0)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Training Time (s)": training_time
    })

    filename = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(model, os.path.join(MODEL_DIR, filename))

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Training time: {training_time:.2f} seconds")

print("\n========== MODEL COMPARISON ==========")

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

results_df.to_csv(
    os.path.join(MODEL_DIR, "model_comparison.csv"),
    index=False
)

print("\n========== TRAINING COMPLETED ==========")
print("Models saved in:", MODEL_DIR)
