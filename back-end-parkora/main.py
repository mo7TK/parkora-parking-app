from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import json

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI()

# ── In-memory state ───────────────────────────────────────────────────────────
# This is the single source of truth for the prototype.
# It holds the latest status of every spot.
# Format: { "spots": [{"id": 1, "status": "free"}, ...] }
latest_state: dict = {"spots": []}


# ── WebSocket connection manager ──────────────────────────────────────────────
class ConnectionManager:
    """
    Keeps track of all active WebSocket clients (phone connections).
    When a phone opens the mini map screen it connects here.
    When it closes the screen it disconnects.
    """

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()                        # complete the handshake
        self.active_connections.append(websocket)
        print(f"Client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print(f"Client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        """Send the same message to every connected client."""
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()


# ── Pydantic models ───────────────────────────────────────────────────────────
# Pydantic validates the shape of incoming JSON automatically.
# If detect.py sends wrong data, FastAPI returns a clear error instead of crashing.

class SpotStatus(BaseModel):
    id: int
    status: str   # "free" or "occupied"

class SpotsUpdate(BaseModel):
    spots: list[SpotStatus]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Health check — open this in your browser to confirm the server is running."""
    return {"message": "Parkora backend is running"}


@app.post("/update-spots")
async def update_spots(data: SpotsUpdate):
    """
    Called by detect.py every second.
    1. Saves the new statuses into latest_state
    2. Broadcasts the update to all connected WebSocket clients
    """
    global latest_state

    # Convert pydantic models back to plain dicts for JSON serialization
    latest_state = {"spots": [spot.dict() for spot in data.spots]}

    # Broadcast to every connected phone
    await manager.broadcast(json.dumps(latest_state))

    return {"received": len(data.spots), "clients_notified": len(manager.active_connections)}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    The mobile app connects here and stays connected.
    As soon as it connects, we send the current state immediately
    so the screen doesn't start empty.
    Then we just keep the connection alive and wait.
    detect.py → POST /update-spots → broadcast() → arrives here on the phone.
    """
    await manager.connect(websocket)

    # Send current state immediately on connect so the app isn't blank
    if latest_state["spots"]:
        await websocket.send_text(json.dumps(latest_state))

    try:
        while True:
            # We don't expect messages from the phone in this prototype,
            # but we must keep this loop running to maintain the connection.
            # receive_text() just waits — if the client disconnects it raises
            # WebSocketDisconnect which we catch below.
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
