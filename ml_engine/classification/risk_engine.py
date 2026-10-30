
import numpy as np
import pandas as pd



def calculate_risk(results):

    results = results.copy()

    risk_scores = []
    risk_levels = []

    for _, row in results.iterrows():

        probability = row["malicious_probability"]
        anomaly = row["anomaly_prediction"]
        binary = row["binary_prediction"]

        score = probability * 70

        if anomaly == 1:
            score += 30

        score = min(score, 100)

        if binary == 1 and anomaly == 1:
            level = "CRITICAL"

        elif probability >= 0.60:
            level = "HIGH"

        elif anomaly == 1 or probability >= 0.40:
            level = "MEDIUM"

        else:
            level = "LOW"

        risk_scores.append(round(score, 2))
        risk_levels.append(level)

    results["risk_score"] = risk_scores
    results["risk_level"] = risk_levels

    return results



if __name__ == "__main__":

    from hybrid_detector import NeuralShieldDetector

    print("\n========== NEURALSHIELD RISK ENGINE ==========")

    detector = NeuralShieldDetector()

    test_data = pd.read_csv(
        "data/raw/UNSW_NB15_testing-set.csv"
    ).head(20)

    predictions = detector.predict(test_data)

    risk_results = calculate_risk(predictions)

    print("\nRisk Analysis:")
    print(
        risk_results[
            [
                "malicious_probability",
                "predicted_attack",
                "anomaly_prediction",
                "risk_score",
                "risk_level"
            ]
        ].to_string(index=False)
    )

    print("\nRisk Distribution:")
    print(risk_results["risk_level"].value_counts())

    print("\nRisk engine completed!")
