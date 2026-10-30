
import pandas as pd
import numpy as np
import os

print("\n========== NEURALSHIELD DATA EXPLORATION ==========\n")

# Dataset paths
TRAIN_PATH = "data/raw/UNSW_NB15_training-set.csv"
TEST_PATH = "data/raw/UNSW_NB15_testing-set.csv"

# Check files
if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(TRAIN_PATH)

if not os.path.exists(TEST_PATH):
    raise FileNotFoundError(TEST_PATH)

# Load datasets
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

# 1. Dataset dimensions
print("TRAINING SHAPE:", train.shape)
print("TESTING SHAPE:", test.shape)

# 2. Dataset columns
print("\nCOLUMNS:")
print(train.columns.tolist())

# 3. Dataset preview
print("\nFIRST 5 RECORDS:")
print(train.head())

# 4. Data types
print("\nDATA TYPES:")
print(train.dtypes)

# 5. Attack distribution
print("\nATTACK DISTRIBUTION:")
print(train["attack_cat"].value_counts(dropna=False))

# 6. Binary classification distribution
print("\nBINARY LABEL DISTRIBUTION:")
print(train["label"].value_counts(dropna=False))

# 7. Missing values
print("\nMISSING VALUES:")
print(train.isnull().sum().sort_values(ascending=False))

# 8. Duplicate records
print("\nDUPLICATE RECORDS:", train.duplicated().sum())

# 9. Statistical summary
print("\nSTATISTICAL SUMMARY:")
print(train.describe(include="all").T)

print("\n========== EXPLORATION COMPLETE ==========")
