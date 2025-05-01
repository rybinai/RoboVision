import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, cap):
        #cap: объект видеозахвата OpenCV
        
        self.cap = cap
        self.background_gray = self._capture_background()
        
    def _capture_background(self):
        #return: обработанное фоновое изображение в градациях серого

        ret, background = self.cap.read()
        if not ret:
            raise RuntimeError("Камера не доступна")
            
        background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        background_gray = cv2.GaussianBlur(background_gray, (21, 21), 0)
        return background_gray
    
    def _preprocess_frame(self, frame):
        #frame: входной кадр
        #return: обработанное изображение в градациях серого

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        return gray
    
    def _process_difference(self, current_gray):
        #current_gray: текущий кадр в градациях серого
        #return: бинарная маска различий
        
        diff = cv2.absdiff(self.background_gray, current_gray)
        _, diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Морфологические операции
        kernel = np.ones((5, 5), np.uint8)
        diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)
        diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, kernel)
        return diff
    
    def _draw_contours(self, frame, diff):
        #frame: исходный кадр
        #diff: маска различий
        #return: кадр с нарисованными контурами

        contours, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return frame
    
    def generate_frames(self):
        #yield: байты кадра в формате JPEG

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Обработка кадра
            current_gray = self._preprocess_frame(frame)
            diff = self._process_difference(current_gray)
            processed_frame = self._draw_contours(frame.copy(), diff)
            
            # Создание комбинированного изображения
            diff_colored = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
            combined_frame = np.hstack((processed_frame, diff_colored))
            
            # Кодировка в JPEG
            _, buffer = cv2.imencode('.jpg', combined_frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Функции для совместимости с существующим кодом
def init_object_detection(cap):
    detector = ObjectDetector(cap)
    return detector

def generate_object_frames(cap, detector):
    yield from detector.generate_frames()