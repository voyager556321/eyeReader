import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

# --- Ініціалізація Mediapipe ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Індекси ключових точок ---
LEFT_IRIS = [474, 475, 476, 477]  # координати райдужки лівого ока
LEFT_EYE_LID_TOP = 159
LEFT_EYE_LID_BOTTOM = 145

# --- Стан програми ---
state = {
    'threshold_up': None,
    'threshold_down': None,
    'cooldown_frames': 10,
    'counter': 0
}

# --- Відкриваємо камеру ---
cap = cv2.VideoCapture(0)

def get_relative_iris_y(landmarks):
    iris_y = sum([landmarks[i].y for i in LEFT_IRIS]) / len(LEFT_IRIS)
    top = landmarks[LEFT_EYE_LID_TOP].y
    bottom = landmarks[LEFT_EYE_LID_BOTTOM].y
    return (iris_y - top) / (bottom - top)

def swipe(direction):
    if direction == 'down':
        pyautogui.press('pgdn')
        print("Swipe DOWN")
    elif direction == 'up':
        pyautogui.press('pgup')
        print("Swipe UP")

# --- Калібрування ---
def calibrate():
    points = ['CENTER', 'UP', 'DOWN']
    iris_data = {'CENTER': [], 'UP': [], 'DOWN': []}

    for p in points:
        print(f"Сфокусуйтесь на точку: {p}. Натисніть 's' щоб зафіксувати")
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                iris_rel_y = get_relative_iris_y(landmarks)
                cv2.putText(frame, f"Точка: {p}, Iris Y: {iris_rel_y:.3f}", (50,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Calibration", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('s'):
                iris_data[p].append(iris_rel_y)
                print(f"Зафіксовано для {p}: {iris_rel_y:.3f}")
                break
            elif key & 0xFF == ord('q'):
                exit()

    cv2.destroyWindow("Calibration")

    # Обчислюємо середнє для порогів
    state['threshold_up'] = np.mean(iris_data['UP']) - np.mean(iris_data['CENTER'])
    state['threshold_down'] = np.mean(iris_data['CENTER']) - np.mean(iris_data['DOWN'])
    print(f"Калібрування завершено. Поріг UP={state['threshold_up']:.3f}, DOWN={state['threshold_down']:.3f}")

# --- Функція для визначення напрямку ---
def check_gaze(iris_y):
    if iris_y - center_y > state['threshold_up']:
        return 'up'
    elif center_y - iris_y > state['threshold_down']:
        return 'down'
    return None

# --- Основний цикл ---
calibrate()
center_y = (state['threshold_up'] + state['threshold_down']) / 2  # приблизне центр

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        iris_rel_y = get_relative_iris_y(landmarks)

        if state['counter'] == 0:
            direction = check_gaze(iris_rel_y)
            if direction:
                swipe(direction)
                state['counter'] = state['cooldown_frames']

        if state['counter'] > 0:
            state['counter'] -= 1

        cv2.putText(frame, f"Iris Y: {iris_rel_y:.3f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Eye Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
