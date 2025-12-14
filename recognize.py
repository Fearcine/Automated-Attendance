# recognize.py — clean, smooth, multi-face attendance

import time
import cv2
import numpy as np
from deepface import DeepFace
from mtcnn import MTCNN
from scipy.spatial.distance import cosine
from db import load_people, mark_attendance, mark_excel_attendance


def run_recognition():
    # ===== CONFIG =====
    FRAME_SKIP = 5          # recognize every N frames
    MAX_FACES = 4
    SIMILARITY_THRESHOLD = 0.45
    MARK_COOLDOWN = 60      # seconds
    SCALE = 0.5             # downscale for speed
    # ==================

    people = load_people()
    if not people:
        print("No registered faces found. Run register.py first.")
        return

    detector = MTCNN()
    facenet_model = DeepFace.build_model("Facenet")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found.")
        return

    frame_count = 0
    seen = {}   # person_id -> last_marked_time
    last_boxes = []  # cached boxes for smooth drawing

    def extract_faces(img):
        out = []
        res = detector.detect_faces(img)
        if not res:
            return out
        for i, r in enumerate(res[:MAX_FACES]):
            x, y, w, h = r["box"]
            x, y = max(0, x), max(0, y)
            crop = img[y:y+h, x:x+w]
            out.append((x, y, w, h, crop))
        return out

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        do_process = (frame_count % FRAME_SKIP == 0)

        labels = []

        if do_process:
            small = cv2.resize(frame, None, fx=SCALE, fy=SCALE)
            faces = extract_faces(small)
            last_boxes = []

            for (x, y, w, h, face) in faces:
                if face.size == 0:
                    continue

                rep = DeepFace.represent(
                 face,
                  model_name="Facenet",
                  model=facenet_model,
                 enforce_detection=False
)

                # normalize all possible return types
                if isinstance(rep, list):
                    if len(rep) > 0 and isinstance(rep[0], dict):
                        emb = rep[0]["embedding"]
                    else:
                        emb = rep
                elif isinstance(rep, dict):
                    emb = rep["embedding"]
                else:
                    emb = rep

                emb = np.array(emb, dtype=float)



                emb = np.array(emb, dtype=float)

                best = (None, 1.0)
                for pid, name, dbemb in people:
                    d = cosine(emb, np.array(dbemb, dtype=float))
                    if d < best[1]:
                        best = ((pid, name), d)

                # scale boxes back
                sx, sy, sw, sh = int(x/SCALE), int(y/SCALE), int(w/SCALE), int(h/SCALE)

                if best[0] and best[1] < SIMILARITY_THRESHOLD:
                    pid, name = best[0]
                    labels.append((sx, sy, sw, sh, name, (0, 255, 0)))

                    now = time.time()
                    if pid not in seen or (now - seen[pid]) > MARK_COOLDOWN:
                        mark_attendance(pid, name)
                        mark_excel_attendance(name)
                        seen[pid] = now
                else:
                    labels.append((sx, sy, sw, sh, "Unknown", (0, 0, 255)))

                last_boxes = labels
        else:
            labels = last_boxes

        # draw
        for x, y, w, h, text, color in labels:
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, max(y-10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

