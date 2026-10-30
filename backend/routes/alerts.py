
from fastapi import APIRouter, HTTPException
from bson import ObjectId

from backend.database.mongodb import (
    alerts_collection,
    detections_collection
)

router = APIRouter(
    prefix="/alerts",
    tags=["Alerts"]
)


def serialize_document(document):
    document["_id"] = str(document["_id"])
    return document


@router.get("/")
def get_alerts():

    alerts = list(
        alerts_collection.find().sort("created_at", -1).limit(100)
    )

    return {
        "total": len(alerts),
        "alerts": [
            serialize_document(alert)
            for alert in alerts
        ]
    }


@router.get("/detections")
def get_detections():

    detections = list(
        detections_collection.find().sort("timestamp", -1).limit(100)
    )

    return {
        "total": len(detections),
        "detections": [
            serialize_document(item)
            for item in detections
        ]
    }


@router.patch("/{alert_id}/resolve")
def resolve_alert(alert_id: str):

    if not ObjectId.is_valid(alert_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid alert ID"
        )

    result = alerts_collection.update_one(
        {"_id": ObjectId(alert_id)},
        {
            "$set": {
                "status": "RESOLVED"
            }
        }
    )

    if result.matched_count == 0:
        raise HTTPException(
            status_code=404,
            detail="Alert not found"
        )

    return {
        "message": "Alert resolved successfully",
        "alert_id": alert_id
    }
