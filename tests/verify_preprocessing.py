
import joblib

data = joblib.load("data/processed/prepared_data.joblib")

print("\n========== NEURALSHIELD DATA VERIFICATION ==========")

for key, value in data.items():
    if hasattr(value, "shape"):
        print(f"{key}: {value.shape}")
    else:
        print(f"{key}: {len(value)}")

print("\nPreprocessing artifacts loaded successfully!")
