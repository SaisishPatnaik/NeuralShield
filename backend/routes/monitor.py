
import asyncio

from fastapi import APIRouter, HTTPException, WebSocket
from pydantic import BaseModel
from starlette.websockets import WebSocketDisconnect

from capture_engine.monitor import network_monitor


router = APIRouter(tags=["Network Monitoring"])


class MonitorRequest(BaseModel):
    interface: str | None = None


@router.post("/monitor/start")
def start_monitoring(request: MonitorRequest):

    if network_monitor.running:
        return {
            "message": "Monitoring is already running",
            "status": network_monitor.get_status()
        }

    started = network_monitor.start(
        interface=request.interface
    )

    if not started:
        raise HTTPException(
            status_code=400,
            detail="Could not start monitoring"
        )

    return {
        "message": "Network monitoring started",
        "status": network_monitor.get_status()
    }


@router.post("/monitor/stop")
def stop_monitoring():

    stopped = network_monitor.stop()

    return {
        "message": (
            "Network monitoring stopped"
            if stopped
            else "Monitoring was not running"
        ),
        "status": network_monitor.get_status()
    }


@router.get("/monitor/status")
def monitor_status():

    return network_monitor.get_status()


@router.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):

    await websocket.accept()

    try:

        while True:

            event = await asyncio.to_thread(
                network_monitor.get_event,
                1
            )

            if event is not None:
                await websocket.send_json(event)

            else:
                await websocket.send_json({
                    "type": "heartbeat",
                    "status": (
                        "running"
                        if network_monitor.running
                        else "stopped"
                    )
                })

    except WebSocketDisconnect:
        pass

    except Exception:
        try:
            await websocket.close()
        except Exception:
            pass
