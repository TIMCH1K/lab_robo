import cv2
import time

# Загрузка каскадов Хаара
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Открываем веб-камеру
cap = cv2.VideoCapture(0)

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Начало обработки текущего кадра
    start_time = time.time()

    # Преобразуем в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Поиск лиц на полном кадре
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(60, 60)  # минимальный размер лица
    )

    # Флаги для состояния "улыбка" и "глаза"
    user_smiling = False
    eyes_open = False

    for (x, y, w, h) in faces:
        # Рисуем прямоугольник вокруг лица
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Зона интереса (ROI) для распознавания глаз и улыбки — внутри лица
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Ищем глаза
        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )
        # Если найдено 2 глаза — считаем, что глаза "открыты"
        if len(eyes) >= 2:
            eyes_open = True

        # Отрисуем прямоугольники вокруг глаз (для наглядности)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        # Ищем улыбку
        smiles = smile_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.7,
            minNeighbors=20,  # число, чувствительное к ложным срабатываниям
            minSize=(25, 25)
        )
        # Если найдена хотя бы одна "улыбка", считаем, что пользователь улыбается
        if len(smiles) > 0:
            user_smiling = True

        # Для наглядности также отрисуем рамку вокруг улыбки
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 255), 2)

    # Если распознали лицо, но нет улыбки
    if len(faces) > 0 and not user_smiling:
        cv2.putText(frame, "Ulibnis", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Если распознали лицо, но не нашли 2 глаза
    if len(faces) > 0 and not eyes_open:
        cv2.putText(frame, "Otkroi glaza", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Вычисление FPS
    current_time = time.time()
    time_diff = current_time - prev_time
    if time_diff > 0:
        fps = 1.0 / time_diff
    else:
        fps = 0

    prev_time = current_time

    # Выводим значение FPS в левом верхнем углу
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Frame', frame)

    # Нажмите ESC, чтобы выйти
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
