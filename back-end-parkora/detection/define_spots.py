import cv2
import json
import numpy as np

# ── Configuration ────────────────────────────────────────────────────────────
VIDEO_PATH = "parking.mp4"   # place your video file next to this script
SPOTS_FILE = "spots.json"    # where the spot coordinates will be saved
NUM_SPOTS  = 12              # how many spots you want to define
# ─────────────────────────────────────────────────────────────────────────────

spots           = []   # list of confirmed polygons, each is [(x,y),(x,y),(x,y),(x,y)]
current_clicks  = []   # corners clicked so far for the spot being drawn
original_frame  = None # the raw first frame, never modified
display_frame   = None # what we show on screen (redrawn after every action)


def redraw():
    """Rebuild display_frame from scratch on top of the original frame."""
    global display_frame
    img = original_frame.copy()

    # Draw all confirmed spots
    for idx, polygon in enumerate(spots):
        pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(img, [pts], (0, 255, 0))          # green semi-fill
        overlay = original_frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 200, 0))
        cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
        cv2.polylines(img, [pts], isClosed=True, color=(0, 200, 0), thickness=2)
        # Spot number at centroid
        cx = int(sum(p[0] for p in polygon) / len(polygon))
        cy = int(sum(p[1] for p in polygon) / len(polygon))
        cv2.putText(img, str(idx + 1), (cx - 8, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Draw corners clicked so far for the current spot (in red)
    for pt in current_clicks:
        cv2.circle(img, pt, 5, (0, 0, 255), -1)
    if len(current_clicks) > 1:
        for i in range(len(current_clicks) - 1):
            cv2.line(img, current_clicks[i], current_clicks[i + 1], (0, 0, 255), 1)
        if len(current_clicks) == 4:
            # Close the shape preview
            cv2.line(img, current_clicks[3], current_clicks[0], (0, 0, 255), 1)

    # Instructions overlay
    cv2.rectangle(img, (0, 0), (480, 95), (0, 0, 0), -1)
    cv2.putText(img, f"Spots confirmed: {len(spots)} / {NUM_SPOTS}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    cv2.putText(img, f"Current spot clicks: {len(current_clicks)} / 4",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 1)
    cv2.putText(img, "R = undo last click/spot   Q = quit & save",
                (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    display_frame = img


def on_mouse_click(event, x, y, flags, param):
    """Called by OpenCV every time the user clicks inside the window."""
    global current_clicks, spots

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # Ignore clicks if all spots are already defined
    if len(spots) == NUM_SPOTS:
        return

    current_clicks.append((x, y))

    # Once we have 4 corners, the spot is complete
    if len(current_clicks) == 4:
        spots.append(current_clicks.copy())
        current_clicks = []
        print(f"  Spot {len(spots)} confirmed.")
        if len(spots) == NUM_SPOTS:
            save_spots()

    redraw()


def save_spots():
    with open(SPOTS_FILE, "w") as f:
        json.dump(spots, f, indent=2)
    print(f"\nAll {NUM_SPOTS} spots saved to {SPOTS_FILE}")
    print("You can now close this window and run detect.py")


# ── Main ─────────────────────────────────────────────────────────────────────

# Read only the very first frame of the video
cap = cv2.VideoCapture(VIDEO_PATH)
ret, original_frame = cap.read()
cap.release()

if not ret:
    print(f"ERROR: Could not open '{VIDEO_PATH}'.")
    print("Make sure the video file is placed inside the detection/ folder.")
    exit(1)

print(f"Video loaded. Frame size: {original_frame.shape[1]}x{original_frame.shape[0]}")
print(f"Click 4 corners for each of the {NUM_SPOTS} parking spots.")
print("  R = undo last action")
print("  Q = quit and save whatever is defined so far")

redraw()
cv2.namedWindow("Define Parking Spots")
cv2.setMouseCallback("Define Parking Spots", on_mouse_click)

while True:
    cv2.imshow("Define Parking Spots", display_frame)
    key = cv2.waitKey(20) & 0xFF

    if key == ord('r'):
        # Undo: remove last click, or last confirmed spot if no clicks in progress
        if current_clicks:
            current_clicks.pop()
            print("  Last click removed.")
        elif spots:
            spots.pop()
            print(f"  Spot {len(spots) + 1} removed.")
        redraw()

    elif key == ord('q'):
        if spots:
            save_spots()
        else:
            print("No spots defined, nothing saved.")
        break

    # Auto-close when all spots are done
    if len(spots) == NUM_SPOTS:
        cv2.waitKey(1500)  # short pause so user can see the final result
        break

cv2.destroyAllWindows()
