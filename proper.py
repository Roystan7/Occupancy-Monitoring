import cv2
import time
from datetime import datetime
from collections import defaultdict
import numpy as np
from ultralytics import YOLO

import firebase_admin
from firebase_admin import credentials, firestore
import webbrowser

# ----------------------------- Config -----------------------------
CAMERA_INDEX = 1
TARGET_WIDTH = 960

WEIGHTS = "yolov8n.pt"
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
META_COUNTER_DOC     = ("metadata", "counters")  # collection, doc

OCC_UPDATE_INTERVAL_SEC = 0.25
OPEN_FIREBASE_CONSOLE = True  # Changed from False to True
FIREBASE_CONSOLE_URL = "https://console.firebase.google.com/u/0/project/occupancy-monitoring-26bc2/firestore/databases/-default-/data/~2Fentries~2F1?view=panel-view&query=1%7CLIM%7C3%2F100"
# ------------------------------------------------------------------


def init_firebase():
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(SERVICE_ACCOUNT_JSON)
            firebase_admin.initialize_app(cred)
        db = firestore.client()
    except Exception as e:
        raise RuntimeError(f"Firebase init failed: {e}")

    meta_col, counter_doc_id = META_COUNTER_DOC
    counter_ref = db.collection(meta_col).document(counter_doc_id)

    doc = counter_ref.get()
    if doc.exists:
        d = doc.to_dict() or {}
        counter_val = int(d.get("entry_counter", 1))
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


def main():
    db, counter_ref, entry_counter = init_firebase()
    counter_state = {'val': entry_counter}

    if OPEN_FIREBASE_CONSOLE:
        try:
            webbrowser.open(FIREBASE_CONSOLE_URL)
        except Exception as e:
            print(f"[WARN] Could not open Firebase console: {e}")

    model = YOLO(WEIGHTS)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

    # State
    people_inside = set()                              # tids currently inside
    last_event_time = defaultdict(lambda: datetime.min)  # per-tid debounce
    last_seen_time = {}                                # per-tid last seen
    prev_cx = {}                                       # per-tid previous centroid x

    # visit state per ID: entered flag + last_side ('L' or 'R')
    id_state = defaultdict(lambda: {'entered': False, 'last_side': None})

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
        line_x = int(w * LINE_POS_X_REL)

        # Draw counting line
        cv2.line(frame, (line_x, 0), (line_x, h), (255, 0, 0), 2)

        # Run tracking (person class only)
        results = model.track(
            source=frame,
            persist=True,
            conf=CONF_THRES,
            iou=IOU_THRES,
            classes=[0],
            tracker=TRACKER_CFG,
            stream=True,
            verbose=False
        )

        rects = []       # (x1,y1,x2,y2,tid,conf)
        active_tids = set()

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy  = r.boxes.xyxy.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            ids   = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else None
            clses = r.boxes.cls.cpu().numpy().astype(int)
            if ids is None:
                continue
            for (x1, y1, x2, y2), conf, cls_id, tid in zip(xyxy, confs, clses, ids):
                if cls_id != 0 or conf < CONF_THRES:
                    continue
                if (x2 - x1) < MIN_BOX_W or (y2 - y1) < MIN_BOX_H:
                    continue
                rects.append((x1, y1, x2, y2, tid, float(conf)))
                active_tids.add(tid)
                last_seen_time[tid] = datetime.now()

        # Handle crossings with side tracking and robust warm start
        for (x1, y1, x2, y2, tid, conf) in rects:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Determine current side relative to the line with tolerance
            if cx <= line_x - LINE_TOL_PX:
                cur_side = 'L'
            elif cx >= line_x + LINE_TOL_PX:
                cur_side = 'R'
            else:
                cur_side = id_state[tid]['last_side']  # keep previous side if within band

            # Draw dot + ID
            color = (0, 255, 0) if id_state[tid]['entered'] else (0, 0, 255)
            cv2.circle(frame, (cx, cy), 4, color, -1)
            cv2.putText(frame, f"ID {tid}", (cx - 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            now = datetime.now()

            # Initialize on first sight
            if id_state[tid]['last_side'] is None:
                id_state[tid]['last_side'] = cur_side
                prev_cx[tid] = cx
                if WARM_START_INSIDE and cur_side == 'R' and not id_state[tid]['entered']:
                    id_state[tid]['entered'] = True
                    people_inside.add(tid)
                    # allow immediate exit by setting last_event_time sufficiently in the past
                    last_event_time[tid] = datetime.min
                    print(f"[WARM START] ID {tid} marked inside (side=R)")
                continue

            # Side transition detection (robust to missed single-frame straddle)
            prev_side = id_state[tid]['last_side']

            # Left -> Right transition = ENTRY
            if prev_side == 'L' and cur_side == 'R' and not id_state[tid]['entered']:
                if (now - last_event_time[tid]).total_seconds() > DEBOUNCE_SEC:
                    id_state[tid]['entered'] = True
                    people_inside.add(tid)
                    write_event_sync(db, counter_ref, counter_state, 'entry', tid)
                    last_event_time[tid] = now

            # Right -> Left transition = EXIT
            elif prev_side == 'R' and cur_side == 'L' and id_state[tid]['entered']:
                if (now - last_event_time[tid]).total_seconds() > DEBOUNCE_SEC:
                    id_state[tid]['entered'] = False
                    if tid in people_inside:
                        people_inside.discard(tid)
                    write_event_sync(db, counter_ref, counter_state, 'exit', tid)
                    last_event_time[tid] = now

            # Update side and previous x
            id_state[tid]['last_side'] = cur_side
            prev_cx[tid] = cx

        # Drop stale IDs and log exits for tracks that are lost from view
        now = datetime.now()
        tids_to_remove = []
        for tid in list(id_state.keys()):
            if tid not in active_tids:
                last_t = last_seen_time.get(tid, datetime.min)
                if (now - last_t).total_seconds() > MAX_STAY_LOST_SEC:
                    if id_state[tid]['entered']:
                        if tid in people_inside:
                            people_inside.discard(tid)
                        write_event_sync(db, counter_ref, counter_state, 'exit', tid)
                    
                    tids_to_remove.append(tid)

        # Clean up state for lost tracks
        for tid in tids_to_remove:
            id_state.pop(tid, None)
            last_event_time.pop(tid, None)
            last_seen_time.pop(tid, None)
            prev_cx.pop(tid, None)

        # Push occupancy rate-limited
        tnow = time.time()
        if tnow - last_occ_push >= OCC_UPDATE_INTERVAL_SEC:
            write_occupancy_sync(db, len(people_inside))
            last_occ_push = tnow

        # HUD
        cv2.putText(frame, f"People Inside: {len(people_inside)}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Occupancy Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()