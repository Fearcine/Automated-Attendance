# register.py — clean face registration (GUI-safe)

import cv2
import numpy as np
from deepface import DeepFace
from mtcnn import MTCNN
from db import init_db, save_person


def run_register():
    init_db()
    detector = MTCNN()

    name = input("Enter name to register: ").strip()
    if not name:
        print("Invalid name.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found.")
        return

    print("Look at camera. Press S to save. ESC to cancel.")

    def extract_face(img):
        res = detector.detect_faces(img)
        if not res:
            return None
        x, y, w, h = res[0]["box"]
        x, y = max(0, x), max(0, y)
        return img[y:y+h, x:x+w]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Register Face (S = Save, ESC = Exit)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break

        if key == ord("s"):
            face = extract_face(frame)
            if face is None or face.size == 0:
                print("No face detected. Try again.")
                continue

            rep = DeepFace.represent(
                face,
                model_name="Facenet",
                enforce_detection=False
            )

            if isinstance(rep, list):
                if isinstance(rep[0], dict):
                    emb = rep[0]["embedding"]
                else:
                    emb = rep
            elif isinstance(rep, dict):
                emb = rep["embedding"]
            else:
                emb = rep

            emb = np.array(emb, dtype=float)
            save_person(name, emb.tolist())
            print(f"Saved face for {name}.")
            break

    cap.release()
    cv2.destroyAllWindows()

