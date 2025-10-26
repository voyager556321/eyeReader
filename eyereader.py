import cv2
import mediapipe as mp
import pyautogui
import time

# Ініціалізація Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Індекси ключових точок
LEFT_IRIS = [474, 475, 476, 477]  # координати райдужки лівого ока
LEFT_EYE_TOP = 386
LEFT_EYE_BOTTOM = 374
LEFT_EYE_LID_TOP = 159
LEFT_EYE_LID_BOTTOM = 145

# Стан програми
state = {
    'neutral_y': None,
    'threshold': 0.02,
    'cooldown_frames': 10,
    'counter': 0
}

# Відкриваємо камеру
cap = cv2.VideoCapture(0)

def get_relative_iris_y(landmarks):
    # координати райдужки
    iris_y = sum([landmarks[i].y for i in LEFT_IRIS]) / len(LEFT_IRIS)
    # верхня та нижня повіка
    top = landmarks[LEFT_EYE_LID_TOP].y
    bottom = landmarks[LEFT_EYE_LID_BOTTOM].y
    # нормалізована відносна позиція райдужки
    return (iris_y - top) / (bottom - top)

def eyes_closed(landmarks):
    # простий показник закриття очей: відстань між верхньою та нижньою повікою
    eye_height = landmarks[LEFT_EYE_LID_BOTTOM].y - landmarks[LEFT_EYE_LID_TOP].y
    return eye_height < 0.02

def check_gaze(rel_y, neutral, threshold):
    diff = rel_y - neutral
    if diff > threshold:
        return 'up'
    elif diff < -threshold:
        return 'down'
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

            # Малюємо всі крапки обличчя
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

            # Малюємо райдужку лівого ока
            for idx in LEFT_IRIS:
                x = int(landmarks[idx].x * frame.shape[1])
                y = int(landmarks[idx].y * frame.shape[0])
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            if eyes_closed(landmarks):
                cv2.putText(frame, "Eyes closed → swipe paused", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                continue

            # Відносна позиція райдужки
            iris_rel_y = get_relative_iris_y(landmarks)

            # Встановлюємо нейтральну позицію на початку
            if state['neutral_y'] is None:
                state['neutral_y'] = iris_rel_y

            if state['counter'] == 0:
                direction = check_gaze(iris_rel_y, state['neutral_y'], state['threshold'])
                if direction:
                    swipe(direction)
                    print(f"Eye move {direction.upper()} detected → Swipe {direction}")
                    state['counter'] = state['cooldown_frames']

            if state['counter'] > 0:
                state['counter'] -= 1

            # Показуємо відносну позицію райдужки на екрані
            cv2.putText(frame, f"Iris Y: {iris_rel_y:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Eye Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Звільняємо ресурси
cap.release()
cv2.destroyAllWindows()
