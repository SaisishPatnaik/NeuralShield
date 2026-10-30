
from fastapi import FastAPI
from datetime import datetime

from backend.routes.alerts import router as alerts_router
from backend.routes.monitor import router as monitor_router


app = FastAPI(
    title="NeuralShield API",
    description="AI-powered Network Intrusion Detection System",
    version="1.0.0"
)

app.include_router(alerts_router)
app.include_router(monitor_router)

@app.get("/")
def home():
    return {
        "project": "NeuralShield",
        "message": "Network Security API is running",
        "status": "online"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }
