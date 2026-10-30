
import json
import pandas as pd
import requests

API_URL = "http://127.0.0.1:8000/detection/analyze"

# Load one real network traffic record
dataset = pd.read_csv(
    "data/raw/UNSW_NB15_testing-set.csv"
)

# Convert the first record into JSON-compatible Python data
traffic_record = json.loads(
    dataset.head(1).to_json(orient="records")
)[0]

print("\nSending network traffic to NeuralShield API...")

try:
    response = requests.post(
        API_URL,
        json={"traffic": traffic_record},
        timeout=120
    )

    print("\nHTTP Status:", response.status_code)

    response.raise_for_status()

    result = response.json()

    print("\n========== NEURALSHIELD API RESPONSE ==========")

    for key, value in result.items():
        print(f"{key}: {value}")

    print("\nAPI TEST SUCCESSFUL!")

except requests.exceptions.RequestException as error:
    print("\nAPI REQUEST FAILED:", error)

    if getattr(error, "response", None) is not None:
        print(error.response.text)
