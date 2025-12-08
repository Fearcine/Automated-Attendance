import cv2, time
import numpy as np
from deepface import DeepFace
from mtcnn import MTCNN
from db import load_people, mark_attendance
from scipy.spatial.distance import cosine

detector = MTCNN()

def extract_face(img):
    res = detector.detect_faces(img)
    if not res:
        return None
    x, y, w, h = res[0]['box']
    x, y = max(0, x), max(0, y)
    return img[y:y+h, x:x+w]

def get_embedding(face):
    reps = DeepFace.represent(face, model_name="Facenet", enforce_detection=False)
    # DeepFace can return:
    # - [ { "embedding": [...] } ]
    # - [0.1, 0.2, ...]  (direct vector)
    if isinstance(reps, list):
        if len(reps) > 0 and isinstance(reps[0], dict) and "embedding" in reps[0]:
            return reps[0]["embedding"]
        else:
            return reps  # already list of floats
    if isinstance(reps, dict) and "embedding" in reps:
        return reps["embedding"]
    return reps

people = load_people()       # list of (id, name, embedding)
if not people:
    print("No registered faces found in DB. Run register.py first.")
    exit()

threshold = 0.45             # adjust if needed
cap = cv2.VideoCapture(0)
seen = {}                    # person_id -> last_marked_time

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face = extract_face(frame)
    if face is not None:
        emb = get_embedding(face)

        # make sure emb is vector
        if not isinstance(emb, (list, tuple, np.ndarray)):
            emb = [emb]

        best = (None, 1.0)  # (person, dist)

        for pid, name, dbemb in people:
            d = cosine(emb, dbemb)
            if d < best[1]:
                best = ((pid, name), d)

        if best[0] and best[1] < threshold:
            pid, name = best[0]
            cv2.putText(
                frame,
                f"{name} {best[1]:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            now = time.time()
            if pid not in seen or now - seen[pid] > 60:
                mark_attendance(pid, name)
                seen[pid] = now
        else:
            cv2.putText(
                frame,
                "Unknown",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
