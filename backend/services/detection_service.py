
from datetime import datetime, timezone

from backend.database.mongodb import (
    detections_collection,
    alerts_collection
)


def save_detection(result):

    # Store detection results
    detection_record = {
        "malicious_probability": float(
            result["malicious_probability"]
        ),
        "binary_prediction": int(
            result["binary_prediction"]
        ),
        "predicted_attack": str(
            result["predicted_attack"]
        ),
        "reconstruction_error": float(
            result["reconstruction_error"]
        ),
        "anomaly_prediction": int(
            result["anomaly_prediction"]
        ),
        "hybrid_prediction": int(
            result["hybrid_prediction"]
        ),
        "final_status": str(
            result["final_status"]
        ),
        "risk_score": float(
            result["risk_score"]
        ),
        "risk_level": str(
            result["risk_level"]
        ),
        "timestamp": datetime.now(timezone.utc)
    }

    # Insert into detections collection
    inserted = detections_collection.insert_one(
        detection_record
    )

    detection_id = str(inserted.inserted_id)

    # Create an alert for high-risk traffic
    if detection_record["risk_level"] in [
        "HIGH",
        "CRITICAL"
    ]:

        alert_record = {
            "detection_id": detection_id,
            "risk_level": detection_record["risk_level"],
            "predicted_attack": detection_record["predicted_attack"],
            "risk_score": detection_record["risk_score"],
            "status": "OPEN",
            "created_at": datetime.now(timezone.utc)
        }

        alerts_collection.insert_one(alert_record)

    return {
        "detection_id": detection_id,
        "stored": True,
        "alert_created": (
            detection_record["risk_level"] in ["HIGH", "CRITICAL"]
        )
    }
