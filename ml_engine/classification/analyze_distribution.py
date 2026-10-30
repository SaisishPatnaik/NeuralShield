
import pandas as pd
import numpy as np

TRAIN_PATH = "data/raw/UNSW_NB15_training-set.csv"
TEST_PATH = "data/raw/UNSW_NB15_testing-set.csv"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print("\n========== NEURALSHIELD DISTRIBUTION ANALYSIS ==========")

for name, df in [("TRAINING", train), ("TESTING", test)]:
    print(f"\n========== {name} DATA ==========")

    print("\nBinary distribution:")
    print(df["label"].value_counts(normalize=True).mul(100).round(2))

    print("\nAttack category distribution:")
    print(df["attack_cat"].value_counts(normalize=True).mul(100).round(2))

    print("\nProtocol distribution (top 10):")
    print(df["proto"].value_counts(normalize=True).head(10).mul(100).round(2))

print("\n========== NUMERIC FEATURE COMPARISON ==========")

numeric_cols = train.select_dtypes(include=np.number).columns
numeric_cols = [
    col for col in numeric_cols
    if col not in ["id", "label"]
]

comparison = pd.DataFrame({
    "Train Mean": train[numeric_cols].mean(),
    "Test Mean": test[numeric_cols].mean(),
    "Train Std": train[numeric_cols].std(),
    "Test Std": test[numeric_cols].std()
})

comparison["Mean Difference"] = (
    comparison["Test Mean"] - comparison["Train Mean"]
).abs()

comparison["Relative Difference"] = (
    comparison["Mean Difference"] /
    comparison["Train Std"].replace(0, np.nan)
)

print(
    comparison.sort_values(
        "Relative Difference", ascending=False
    ).head(20).round(3)
)

print("\n========== ANALYSIS COMPLETE ==========")
