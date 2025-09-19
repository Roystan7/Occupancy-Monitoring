import cv2
import time
from datetime import datetime
from collections import defaultdict
import numpy as np
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials, firestore
import webbrowser
import os

# ----------------------------- Config -----------------------------
CAMERA_INDEX = 1
TARGET_WIDTH = 960

WEIGHTS = "yolov10s.pt"
TRACKER_CFG = "bytetrack.yaml"

CONF_THRES = 0.60
IOU_THRES  = 0.45
MIN_BOX_W  = 40
MIN_BOX_H  = 70

LINE_POS_X_REL   = 0.40
LINE_TOL_PX      = 20
DEBOUNCE_SEC     = 1.0
MAX_STAY_LOST_SEC = 2.0
WARM_START_INSIDE = True

SERVICE_ACCOUNT_JSON = "firebase_key.json"
ENTRIES_COLLECTION   = "entries"
OCC_COLLECTION       = "occupancy"
META_COUNTER_DOC     = ("metadata", "counters")

OCC_UPDATE_INTERVAL_SEC = 0.25
OPEN_FIREBASE_CONSOLE = True
FIREBASE_CONSOLE_URL = "https://console.firebase.google.com/u/0/project/occupancy-monitoring-26bc2/firestore/databases/-default-/data/~2Fentries~2F1?view=panel-view&query=1%7CLIM%7C3%2F100"

OUTPUT_DIR = "synopsis_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROXIMITY_THRESHOLD = 50  # pixels to avoid counting same person twice

# ----------------------------- Firebase -----------------------------
def init_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(SERVICE_ACCOUNT_JSON)
        firebase_admin.initialize_app(cred)
    db = firestore.client()

    meta_col, counter_doc_id = META_COUNTER_DOC
    counter_ref = db.collection(meta_col).document(counter_doc_id)

    doc = counter_ref.get()
    if doc.exists:
        counter_val = int(doc.to_dict().get("entry_counter", 1))
    else:
        counter_val = 1
        counter_ref.set({"entry_counter": counter_val})
        print(f"[FIREBASE] Created {meta_col}/{counter_doc_id} with entry_counter={counter_val}")

    print(f"[FIREBASE] Counter starts at {counter_val}")
    return db, counter_ref, counter_val

def write_event_sync(db, counter_ref, counter_state, event_type, track_id):
    ts = datetime.now().isoformat()
    doc_id = str(counter_state['val'])
    db.collection(ENTRIES_COLLECTION).document(doc_id).set({
        'track_id': int(track_id),
        'event': event_type,
        'timestamp': ts
    })
    counter_state['val'] += 1
    counter_ref.set({'entry_counter': counter_state['val']})
    print(f"[FIREBASE] {event_type.upper()} stored (doc {doc_id}) for ID {track_id} at {ts}")

def write_occupancy_sync(db, count):
    db.collection(OCC_COLLECTION).document('live').set({
        'count': int(count),
        'last_updated': datetime.now().isoformat()
    })

# ----------------------------- Helpers -----------------------------
def draw_arrow(frame, cx, y, direction="right", color=(0,0,255)):
    if direction=="right":
        cv2.arrowedLine(frame, (cx-20, y), (cx+20, y), color, 3, tipLength=0.5)
    else:
        cv2.arrowedLine(frame, (cx+20, y), (cx-20, y), color, 3, tipLength=0.5)

def is_nearby(cx, people_inside, prev_cx):
    return any(abs(cx - prev_cx[other]) < PROXIMITY_THRESHOLD for other in people_inside)

# ----------------------------- Main -----------------------------
def main():
    db, counter_ref, entry_counter = init_firebase()
    counter_state = {'val': entry_counter}

    if OPEN_FIREBASE_CONSOLE:
        try:
            webbrowser.open(FIREBASE_CONSOLE_URL)
        except Exception as e:
            print(f"[WARN] Could not open Firebase console: {e}")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    summary_path = os.path.join(OUTPUT_DIR, "summary.avi")
    summary_writer = None

    model = YOLO(WEIGHTS)

    # ---------------- State ----------------
    people_inside = set()
    prev_cx = {}
    last_event_time = defaultdict(lambda: 0.0)
    last_seen = defaultdict(lambda: time.time())
    LINE_POS = LINE_POS_X_REL
    last_occ_push = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Camera frame not read; exiting loop.")
            break

        if TARGET_WIDTH and frame.shape[1] > TARGET_WIDTH:
            scale = TARGET_WIDTH / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        h, w = frame.shape[:2]

        if summary_writer is None:
            summary_writer = cv2.VideoWriter(summary_path, fourcc, FPS, (w, h), True)

        line_x = int(w * LINE_POS)
        cv2.line(frame, (line_x, 0), (line_x, h), (255, 0, 0), 2)

        results = model.track(
            frame,
            persist=True,
            imgsz=640,
            conf=CONF_THRES,
            iou=IOU_THRES,
            classes=[0],
            tracker=TRACKER_CFG,
            verbose=False
        )

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                if box.id is None:
                    continue
                tid = int(box.id.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                now = time.time()
                last_seen[tid] = now

                if tid not in prev_cx:
                    prev_cx[tid] = cx
                    continue

                # ENTRY
                if prev_cx[tid] < line_x <= cx + LINE_TOL_PX:
                    if tid not in people_inside and not is_nearby(cx, people_inside, prev_cx):
                        if now - last_event_time[tid] > DEBOUNCE_SEC:
                            write_event_sync(db, counter_ref, counter_state, "entry", tid)
                            people_inside.add(tid)
                            last_event_time[tid] = now
                            draw_arrow(frame, cx, cy, direction="right", color=(0,255,0))

                # EXIT
                elif prev_cx[tid] > line_x >= cx - LINE_TOL_PX:
                    # allow spatial proximity for exit as well
                    nearby_inside = [p for p in people_inside if abs(cx - prev_cx[p]) < PROXIMITY_THRESHOLD]
                    if tid in people_inside or nearby_inside:
                        if now - last_event_time[tid] > DEBOUNCE_SEC:
                            write_event_sync(db, counter_ref, counter_state, "exit", tid)
                            people_inside.discard(tid)
                            for p in nearby_inside:
                                people_inside.discard(p)
                            last_event_time[tid] = now
                            draw_arrow(frame, cx, cy, direction="left", color=(0,0,255))

                prev_cx[tid] = cx

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {tid}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # remove lost IDs
        lost_ids = [tid for tid, t in last_seen.items() if time.time() - t > MAX_STAY_LOST_SEC]
        for tid in lost_ids:
            people_inside.discard(tid)
            prev_cx.pop(tid, None)
            last_event_time.pop(tid, None)
            last_seen.pop(tid, None)

        # occupancy update
        if time.time() - last_occ_push >= OCC_UPDATE_INTERVAL_SEC:
            write_occupancy_sync(db, len(people_inside))
            last_occ_push = time.time()

        # HUD
        cv2.putText(frame, f"People Inside: {len(people_inside)}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Recording
        if len(people_inside) > 0:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, f"Recording @ {timestamp}", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            summary_writer.write(frame)

        cv2.imshow("Occupancy Tracking + Synopsis", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    summary_writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print(f"[SYSTEM] Summary video saved at: {summary_path}")

if __name__ == "__main__":
    main()
