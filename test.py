import cv2
import mediapipe as mp
import pyautogui
import subprocess
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Reference vertical position for neutral gaze
neutral_y = None
threshold = 2  # pixels difference to trigger a swipe
cooldown_frames = 10
counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Left eye landmarks (iris center points)
            left_eye = [face_landmarks.landmark[i] for i in [474, 475, 476, 477]]
            left_center_y = sum([p.y for p in left_eye]) / len(left_eye)

            if neutral_y is None:
                neutral_y = left_center_y

            diff = left_center_y - neutral_y

            if counter == 0:
                if diff > 0.02:  # look down
                    pyautogui.press('pgdn')
                    print("Eye move DOWN detected → Page Down")
                    counter = cooldown_frames
                elif diff < -0.02:  # look up
                    pyautogui.press('pgup')
                    print("Eye move UP detected → Page Up")
                    counter = cooldown_frames

            if counter > 0:
                counter -= 1

    cv2.imshow("Eye Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

