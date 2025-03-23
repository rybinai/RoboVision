import cv2
import numpy as np

def init_object_detection(cap):
    # Захват первого кадра (фон)
    ret, background = cap.read()
    background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    background_gray = cv2.GaussianBlur(background_gray, (21, 21), 0)
    return background_gray

def generate_object_frames(cap, background_gray):
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Предобработка текущего кадра
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Вычисление разницы между текущим кадром и фоном
        diff = cv2.absdiff(background_gray, gray)
        _, diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # Морфологические операции для улучшения маски
        kernel = np.ones((5, 5), np.uint8)
        diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)  # Удаление шумов
        diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, kernel)  # Заполнение дыр

        # Поиск контуров для рисования боксов
        contours, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Отрисовка ограничивающих боксов на реальном кадре
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Игнорирование мелких объектов
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  

        # Создание комбинированного изображения (реальный кадр + разница)
        diff_colored = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)  # Преобразование разницы в цветное изображение
        combined_frame = np.hstack((frame, diff_colored))  # Объединение двух изображений горизонтально

        # Кодировка в JPEG
        _, buffer = cv2.imencode('.jpg', combined_frame)
        frame_bytes = buffer.tobytes()

        # Отправка кадра в поток
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')