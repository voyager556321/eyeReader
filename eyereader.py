import cv2
import mediapipe as mp
import time
import numpy as np

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Parameters
scroll_speed = 0.3  # base scroll speed
neutral_iris_y = None
text_pos = 0
paused = False

# Create dummy text for demonstration
text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 50
text_lines = text.split(" ")
font = cv2.FONT_HERSHEY_SIMPLEX

# Camera setup
cap = cv2.VideoCapture(0)

def draw_text_window(frame, pos):
    h, w, _ = frame.shape
    overlay = np.ones_like(frame) * 255
    line_height = 25
    start_idx = int(pos)
    end_idx = min(start_idx + 20, len(text_lines))
    y = 50
    for i in range(start_idx, end_idx):
        cv2.putText(overlay, text_lines[i], (40, y), font, 0.7, (0, 0, 0), 2)
        y += line_height
    blended = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)
    return blended

def get_iris_y(landmarks, w, h):
    left_iris = landmarks[468]
    right_iris = landmarks[473]
    avg_y = (left_iris.y + right_iris.y) / 2
    return avg_y

def eyes_closed(landmarks):
    # eye aspect ratio to detect blink
    left_top = landmarks[159].y
    left_bottom = landmarks[145].y
    ratio = (left_bottom - left_top)
    return ratio < 0.01  # tweak if too sensitive

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
            iris_y = get_iris_y(landmarks, w, h)

            # Draw iris dots
            for i in [468, 473]:
                pt = landmarks[i]
                cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 2, (0, 255, 0), -1)

            # Calibration (set neutral when user looks straight)
            if neutral_iris_y is None:
                neutral_iris_y = iris_y

            # Eye close detection
            if eyes_closed(landmarks):
                paused = True
                cv2.putText(frame, "PAUSED (eyes closed)", (30, 50), font, 0.7, (0, 0, 255), 2)
            else:
                paused = False

            if not paused:
                delta = iris_y - neutral_iris_y
                if abs(delta) > 0.005:  # small dead zone
                    text_pos += delta * 100 * scroll_speed
                    text_pos = max(0, min(len(text_lines) - 20, text_pos))

    output = draw_text_window(frame, text_pos)
    cv2.imshow("Eye Reader Focus Mode", output)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
