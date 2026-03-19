import cv2
import json
import time
import numpy as np
import requests
from ultralytics import YOLO

# ── Configuration ────────────────────────────────────────────────────────────
VIDEO_PATH   = "parking.mp4"
SPOTS_FILE   = "spots.json"
BACKEND_URL  = "http://127.0.0.1:8000/update-spots"
SEND_EVERY   = 1.0   # seconds between each POST to the backend
CONFIDENCE   = 0.35  # YOLO confidence threshold (0 to 1)

# ── Performance tuning ───────────────────────────────────────────────────────
INFER_EVERY  = 3     # run YOLO only on every Nth frame (2–5 is a good range)
INFER_WIDTH  = 640   # resize frame to this width before inference (None = original)
# ─────────────────────────────────────────────────────────────────────────────

# YOLO is trained on the COCO dataset.
# These are the class IDs that correspond to vehicles in that dataset:
#   2 = car   3 = motorcycle   5 = bus   7 = truck
VEHICLE_CLASSES = {2, 3, 5, 7}
# ─────────────────────────────────────────────────────────────────────────────


def load_spots(path):
    """Load polygon coordinates from spots.json"""
    with open(path, "r") as f:
        raw = json.load(f)
    # Convert to numpy arrays once — OpenCV needs them in this format
    return [np.array(polygon, dtype=np.int32) for polygon in raw]


def is_vehicle_in_spot(spot_polygon, detection_box):
    """
    Decide if a detected vehicle occupies a parking spot.

    Strategy: we test two points from the bounding box —
      1. The bottom-center  (where the car touches the ground)
      2. The center         (middle of the car body)

    If EITHER point is inside the polygon, the spot is occupied.
    Using two points makes detection more robust — if a car is partially
    outside the polygon but clearly parked there, we still catch it.

    cv2.pointPolygonTest returns:
       > 0  →  point is INSIDE the polygon
       = 0  →  point is ON the edge
       < 0  →  point is OUTSIDE
    """
    x1, y1, x2, y2 = detection_box

    center_x      = int((x1 + x2) / 2)
    center_y      = int((y1 + y2) / 2)
    bottom_center = (center_x, int(y2))
    center        = (center_x, center_y)

    inside_bottom = cv2.pointPolygonTest(spot_polygon, bottom_center, False) >= 0
    inside_center = cv2.pointPolygonTest(spot_polygon, center,        False) >= 0

    return inside_bottom or inside_center


def compute_statuses(spots, vehicle_boxes):
    """
    For each spot, check if any detected vehicle is inside it.
    Returns a list like: ["occupied", "free", "free", "occupied", ...]
    """
    statuses = []
    for spot in spots:
        occupied = any(is_vehicle_in_spot(spot, box) for box in vehicle_boxes)
        statuses.append("occupied" if occupied else "free")
    return statuses


def draw_frame(frame, spots, statuses):
    """Draw colored polygon overlays and a summary counter on the frame."""
    for i, (spot, status) in enumerate(zip(spots, statuses)):
        color = (0, 200, 0) if status == "free" else (0, 0, 220)

        # Semi-transparent fill
        overlay = frame.copy()
        cv2.fillPoly(overlay, [spot], color)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        # Solid border
        cv2.polylines(frame, [spot], isClosed=True, color=color, thickness=2)

        # Spot number at centroid
        cx = int(spot[:, 0].mean())
        cy = int(spot[:, 1].mean())
        cv2.putText(frame, str(i + 1), (cx - 8, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Summary bar at top
    free_count     = statuses.count("free")
    occupied_count = statuses.count("occupied")
    cv2.rectangle(frame, (0, 0), (320, 40), (0, 0, 0), -1)
    cv2.putText(frame,
                f"Free: {free_count}   Occupied: {occupied_count}",
                (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)


def send_to_backend(statuses):
    """
    Send the current spot statuses to the FastAPI backend via HTTP POST.
    The payload is a JSON object like:
      { "spots": [{"id": 1, "status": "free"}, {"id": 2, "status": "occupied"}, ...] }
    """
    payload = {
        "spots": [
            {"id": i + 1, "status": status}
            for i, status in enumerate(statuses)
        ]
    }
    try:
        response = requests.post(BACKEND_URL, json=payload, timeout=1)
        print(f"[{time.strftime('%H:%M:%S')}] Sent → {response.status_code} | "
              f"Free: {statuses.count('free')}  Occupied: {statuses.count('occupied')}")
    except requests.exceptions.ConnectionError:
        # Backend is not running yet — just warn, don't crash
        print(f"[{time.strftime('%H:%M:%S')}] Backend not reachable, skipping send.")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] Send error: {e}")


# ── Main ─────────────────────────────────────────────────────────────────────

print("Loading spots...")
spots = load_spots(SPOTS_FILE)
print(f"  {len(spots)} spots loaded from {SPOTS_FILE}")

print("Loading YOLO model (downloads ~6MB on first run)...")
model = YOLO("yolov8n.pt")   # 'n' = nano, the smallest and fastest variant
print("  YOLO ready.")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ERROR: Could not open '{VIDEO_PATH}'")
    exit(1)

last_send_time = 0
frame_count    = 0
statuses       = ["free"] * len(spots)   # cached — reused on skipped frames

print("\nDetection running. Press Q in the window to stop.\n")

while True:
    ret, frame = cap.read()

    if not ret:
        # Video ended — loop back to the beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    frame_count += 1

    # ── Run YOLO only every INFER_EVERY frames ────────────────────────────
    if frame_count % INFER_EVERY == 0:
        # Optionally downscale for faster inference, scale boxes back afterward
        if INFER_WIDTH:
            h, w  = frame.shape[:2]
            scale = INFER_WIDTH / w
            infer_frame = cv2.resize(frame, (INFER_WIDTH, int(h * scale)))
        else:
            infer_frame = frame
            scale       = 1.0

        results = model(infer_frame, verbose=False)[0]

        vehicle_boxes = []
        for box in results.boxes:
            cls_id     = int(box.cls[0])
            confidence = float(box.conf[0])
            if cls_id in VEHICLE_CLASSES and confidence >= CONFIDENCE:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if scale != 1.0:
                    x1, y1, x2, y2 = (int(x1 / scale), int(y1 / scale),
                                       int(x2 / scale), int(y2 / scale))
                vehicle_boxes.append((x1, y1, x2, y2))

        statuses = compute_statuses(spots, vehicle_boxes)
    # On skipped frames, `statuses` keeps its previous value — nearly free.

    # ── Draw the result on the frame ──────────────────────────────────────
    draw_frame(frame, spots, statuses)
    cv2.imshow("Parking Detection", frame)

    # ── Send to backend every SEND_EVERY seconds ──────────────────────────
    now = time.time()
    if now - last_send_time >= SEND_EVERY:
        send_to_backend(statuses)
        last_send_time = now

    # Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Detection stopped.")