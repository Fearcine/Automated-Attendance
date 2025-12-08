import cv2
from deepface import DeepFace
from mtcnn import MTCNN
from db import init_db, save_person

# init db + detector
init_db()
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
            return reps  # already a list of floats
    if isinstance(reps, dict) and "embedding" in reps:
        return reps["embedding"]
    return reps

name = input("Name to register: ").strip()
cap = cv2.VideoCapture(0)
print("Look at the camera. Press SPACE to capture. ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Register - press SPACE", frame)
    k = cv2.waitKey(1)

    if k % 256 == 27:      # ESC
        break

    if k % 256 == 32:      # SPACE
        face = extract_face(frame)
        if face is None:
            print("No face found. Try again.")
            continue

        emb = get_embedding(face)
        save_person(name, emb)
        print(f"Saved {name}.")
        break

cap.release()
cv2.destroyAllWindows()
