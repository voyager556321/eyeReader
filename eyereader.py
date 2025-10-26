import cv2
import mediapipe as mp
import pyautogui
import time

# Ініціалізація Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# Індекс точок лівого ока для трекінгу райдужки
LEFT_EYE_INDICES = [474, 475, 476, 477]

# Стан програми
state = {
    'neutral_y': None,       # нейтральна позиція погляду
    'threshold': 0.02,       # поріг зміщення для PageUp/PageDown
    'cooldown_frames': 10,   # кадри перед наступним swipe
    'counter': 0             # лічильник cooldown
}

# Відкриваємо камеру
cap = cv2.VideoCapture(0)

def eye_center_y(landmarks, indices):
    """Обчислює вертикальний центр ока по заданих індексах точок"""
    return sum(landmarks[i].y for i in indices) / len(indices)

def check_gaze(y, neutral, threshold=0.02):
    """Визначає напрямок погляду: 'up', 'down' або None"""
    diff = y - neutral
    if diff > threshold:
        return 'down'
    elif diff < -threshold:
        return 'up'
    return None

def swipe(direction):
    """Виконує Page Up/Down залежно від напрямку погляду"""
    if direction == 'down':
        pyautogui.press('pgdn')
        print("Eye move DOWN detected → Page Down")
    elif direction == 'up':
        pyautogui.press('pgup')
        print("Eye move UP detected → Page Up")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            left_center_y = eye_center_y(landmarks, LEFT_EYE_INDICES)

            # Встановлюємо нейтральну позицію на першому кадрі
            if state['neutral_y'] is None:
                state['neutral_y'] = left_center_y

            # Перевірка напрямку погляду
            if state['counter'] == 0:
                direction = check_gaze(left_center_y, state['neutral_y'], state['threshold'])
                if direction:
                    swipe(direction)
                    state['counter'] = state['cooldown_frames']

            # Зменшення лічильника cooldown
            if state['counter'] > 0:
                state['counter'] -= 1

    # Відображення кадру
    cv2.imshow("Eye Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Звільняємо ресурси
cap.release()
cv2.destroyAllWindows()
