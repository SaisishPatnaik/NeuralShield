
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


# ==========================================
# NEURALSHIELD - DATA PREPROCESSING
# ==========================================

TRAIN_PATH = "data/raw/UNSW_NB15_training-set.csv"
TEST_PATH = "data/raw/UNSW_NB15_testing-set.csv"

OUTPUT_DIR = "data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n========== NEURALSHIELD PREPROCESSING ==========")


# 1. LOAD DATASETS

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Training:", train_df.shape)
print("Testing:", test_df.shape)


# 2. CLEAN TARGET LABELS

train_df["attack_cat"] = train_df["attack_cat"].str.strip()
test_df["attack_cat"] = test_df["attack_cat"].str.strip()

if train_df[["label", "attack_cat"]].isnull().any().any():
    raise ValueError("Missing training target labels detected.")

if test_df[["label", "attack_cat"]].isnull().any().any():
    raise ValueError("Missing testing target labels detected.")


# 3. SEPARATE FEATURES AND TARGETS

# Remove ID and target columns to prevent data leakage.

drop_columns = ["id", "label", "attack_cat"]

X = train_df.drop(columns=drop_columns)
X_test = test_df.drop(columns=drop_columns)

y_binary = train_df["label"].astype(int)
y_multiclass = train_df["attack_cat"]

y_binary_test = test_df["label"].astype(int)
y_multiclass_test = test_df["attack_cat"]


# Replace infinite values with missing values.

X = X.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)


# 4. IDENTIFY FEATURE TYPES

categorical_features = ["proto", "service", "state"]

numeric_features = [
    col for col in X.columns
    if col not in categorical_features
]

print("\nNumeric features:", len(numeric_features))
print("Categorical features:", len(categorical_features))


# 5. CREATE PREPROCESSING PIPELINES

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# 6. TRAIN / VALIDATION SPLIT

indices = np.arange(len(X))

train_indices, val_indices = train_test_split(
    indices,
    test_size=0.20,
    random_state=42,
    stratify=y_multiclass
)

X_train_raw = X.iloc[train_indices]
X_val_raw = X.iloc[val_indices]

print("\nTraining records:", len(train_indices))
print("Validation records:", len(val_indices))


# 7. FIT ONLY ON TRAINING DATA

print("\nFitting preprocessing pipeline...")

X_train = preprocessor.fit_transform(X_train_raw)

X_val = preprocessor.transform(X_val_raw)

X_test_processed = preprocessor.transform(X_test)


# 8. PREPARE TARGETS

y_binary_train = y_binary.iloc[train_indices].to_numpy()
y_binary_val = y_binary.iloc[val_indices].to_numpy()

y_multi_train = y_multiclass.iloc[train_indices].to_numpy()
y_multi_val = y_multiclass.iloc[val_indices].to_numpy()

y_binary_test = y_binary_test.to_numpy()
y_multi_test = y_multiclass_test.to_numpy()


# 9. SAVE PROCESSED DATA

processed_data = {
    "X_train": X_train,
    "X_val": X_val,
    "X_test": X_test_processed,

    "y_binary_train": y_binary_train,
    "y_binary_val": y_binary_val,
    "y_binary_test": y_binary_test,

    "y_multi_train": y_multi_train,
    "y_multi_val": y_multi_val,
    "y_multi_test": y_multi_test,

    "feature_names": preprocessor.get_feature_names_out()
}

joblib.dump(
    processed_data,
    os.path.join(OUTPUT_DIR, "prepared_data.joblib")
)

joblib.dump(
    preprocessor,
    os.path.join(OUTPUT_DIR, "preprocessor.joblib")
)


# 10. FINAL REPORT

print("\n========== PREPROCESSING REPORT ==========")

print("Training matrix:", X_train.shape)
print("Validation matrix:", X_val.shape)
print("Testing matrix:", X_test_processed.shape)

print("\nBinary training distribution:")
print(pd.Series(y_binary_train).value_counts())

print("\nMulticlass training distribution:")
print(pd.Series(y_multi_train).value_counts())

print("\nSaved files:")
print("1. data/processed/prepared_data.joblib")
print("2. data/processed/preprocessor.joblib")

print("\n========== PREPROCESSING COMPLETE ==========")
