
import os
import joblib
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model


class NeuralShieldDetector:

    def __init__(self):

        print("Initializing NeuralShield Hybrid Detector...")

        # Load preprocessing pipeline
        self.preprocessor = joblib.load(
            "models/preprocessor.joblib"
        )

        # Load binary classifier
        self.binary_model = joblib.load(
            "models/xgboost.joblib"
        )

        # Load multiclass classifier
        self.multiclass_model = joblib.load(
            "models/multiclass/xgboost_multiclass.joblib"
        )

        self.label_encoder = joblib.load(
            "models/multiclass/label_encoder.joblib"
        )

        # Load optimized binary threshold
        self.binary_threshold = joblib.load(
            "models/optimal_threshold.joblib"
        )

        # Load autoencoder
        self.autoencoder = load_model(
            "models/anomaly_detection/autoencoder.keras"
        )

        self.anomaly_threshold = joblib.load(
            "models/anomaly_detection/threshold.joblib"
        )

        print("All models loaded successfully!")

    @staticmethod
    def to_dense(X):

        if hasattr(X, "toarray"):
            X = X.toarray()

        return np.asarray(X, dtype=np.float32)

    def predict(self, traffic_data):

        if not isinstance(traffic_data, pd.DataFrame):
            traffic_data = pd.DataFrame(traffic_data)

        # Remove metadata and target columns if supplied
        X = traffic_data.drop(
            columns=["id", "label", "attack_cat"],
            errors="ignore"
        )

        # Apply the same preprocessing used during training
        X_processed = self.preprocessor.transform(X)

        # Convert to dense format for TensorFlow
        X_dense = self.to_dense(X_processed)

        # 1. Binary classification
        malicious_probability = self.binary_model.predict_proba(
            X_processed
        )[:, 1]

        binary_predictions = (
            malicious_probability >= self.binary_threshold
        ).astype(int)

        # 2. Multiclass attack classification
        multiclass_encoded = self.multiclass_model.predict(
            X_processed
        )

        attack_categories = self.label_encoder.inverse_transform(
            multiclass_encoded.astype(int)
        )

        # 3. Autoencoder anomaly detection
        reconstructed = self.autoencoder.predict(
            X_dense,
            verbose=0
        )

        reconstruction_errors = np.mean(
            np.square(X_dense - reconstructed),
            axis=1
        )

        anomaly_predictions = (
            reconstruction_errors > self.anomaly_threshold
        ).astype(int)

        # 4. Hybrid decision
        hybrid_predictions = (
            (binary_predictions == 1) |
            (anomaly_predictions == 1)
        ).astype(int)

        # Generate results
        results = pd.DataFrame({

            "malicious_probability": malicious_probability,

            "binary_prediction": binary_predictions,

            "predicted_attack": attack_categories,

            "reconstruction_error": reconstruction_errors,

            "anomaly_prediction": anomaly_predictions,

            "hybrid_prediction": hybrid_predictions

        })

        results["final_status"] = np.where(
            results["hybrid_prediction"] == 1,
            "SUSPICIOUS",
            "BENIGN"
        )

        return results


if __name__ == "__main__":

    print("\n========== NEURALSHIELD ==========")

    detector = NeuralShieldDetector()

    # Load sample network traffic from the test dataset
    test_data = pd.read_csv(
        "data/raw/UNSW_NB15_testing-set.csv"
    )

    # Test the first 10 records
    sample = test_data.head(10)

    results = detector.predict(sample)

    print("\n========== HYBRID DETECTION RESULTS ==========")

    print(results.to_string(index=False))

    print("\n========== SUMMARY ==========")

    print(
        "Suspicious records:",
        (results["hybrid_prediction"] == 1).sum()
    )

    print(
        "Benign records:",
        (results["hybrid_prediction"] == 0).sum()
    )

    print("\nNeuralShield detection completed!")
