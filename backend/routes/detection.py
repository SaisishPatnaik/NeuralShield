
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import sys
import os

# Add project root to Python path
ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ml_engine.classification.hybrid_detector import (
    NeuralShieldDetector
)

from ml_engine.classification.risk_engine import (
    calculate_risk
)

from backend.services.detection_service import save_detection

router = APIRouter(
    prefix="/detection",
    tags=["Detection"]
)

detector = None


def get_detector():

    global detector

    if detector is None:
        detector = NeuralShieldDetector()

    return detector


class TrafficRequest(BaseModel):
    traffic: dict


@router.post("/analyze")
def analyze_traffic(request: TrafficRequest):

    try:
        model = get_detector()

        traffic_df = pd.DataFrame([request.traffic])

        predictions = model.predict(traffic_df)

        results = calculate_risk(predictions)

        result = results.to_dict(orient="records")[0]

        # Save detection and create alert if required
        storage_result = save_detection(result)

        # Return prediction and database information
        result["database"] = storage_result

        return result

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
