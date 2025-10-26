import cv2
import mediapipe as mp
import numpy as np
import fitz
from PIL import Image
import sys

if len(sys.argv) < 2:
    print("Usage: python eyereader_pdf.py your_book.pdf")
    sys.exit(1)

pdf_path = sys.argv[1]
doc = fitz.open(pdf_path)

# Об’єднуємо всі сторінки в одне велике зображення
pages_img = []
for page in doc:
    pix = page.get_pixmap(alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    np_img = np.array(img)
    np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    pages_img.append(np_img)

pdf_img = np.vstack(pages_img)  # вертикально склеюємо всі сторінки

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

neutral_iris_y = None
text_pos = 0
paused = False

cap = cv2.VideoCapture(0)
window_w, window_h = 800, 600

def get_iris_y(landmarks):
    left_iris = landmarks[468]
    right_iris = landmarks[473]
    return (left_iris.y + right_iris.y) / 2

def eyes_closed(landmarks):
    return (landmarks[145].y - landmarks[159].y) < 0.01

scroll_multiplier = 250  # швидкість скролу, можна трохи збільшити

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    if res.multi_face_landmarks:
        for face_landmarks in res.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            iris_y = get_iris_y(landmarks)

            for i in [468, 473]:
                pt = landmarks[i]
                cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 2, (0, 255, 0), -1)

            if neutral_iris_y is None:
                neutral_iris_y = iris_y

            paused = eyes_closed(landmarks)

            if not paused:
                delta = iris_y - neutral_iris_y
                if abs(delta) > 0.002:
                    text_pos += delta * scroll_multiplier
                    text_pos = max(0, min(pdf_img.shape[0] - window_h, text_pos))

    y_start = int(text_pos)
    y_end = y_start + window_h
    pdf_window = pdf_img[y_start:y_end, 0:window_w].copy()

    cam_small = cv2.resize(frame, (200, 150))
    pdf_window[0:150, 0:200] = cam_small

    cv2.imshow("Eye Reader PDF Mode", pdf_window)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
