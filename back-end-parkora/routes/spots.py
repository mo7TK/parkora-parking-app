import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from socket_manager import manager

router = APIRouter()

# In-memory state — holds the latest statuses received from detect.py
latest_state: dict = {"spots": []}


class SpotStatus(BaseModel):
    id: int
    status: str  # "free" or "occupied"


class SpotsUpdate(BaseModel):
    spots: list[SpotStatus]


@router.post("/update-spots")
async def update_spots(data: SpotsUpdate):
    global latest_state
    latest_state = {"spots": [spot.dict() for spot in data.spots]}
    await manager.broadcast(json.dumps(latest_state))
    return {
        "received": len(data.spots),
        "clients_notified": len(manager.active_connections),
    }


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)

    if latest_state["spots"]:
        await websocket.send_text(json.dumps(latest_state))

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)