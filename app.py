from flask import Flask, Response, render_template
import cv2
from modules.human_detection import init_human_detection, generate_human_frames
from modules.object_detection import init_object_detection, generate_object_frames
from modules.human_segmentation import init_human_segmentation, generate_human_segmentation_frames

app = Flask(__name__)

# Инициализация камеры
cap = cv2.VideoCapture(0)
if not cap.isOpened():
        raise RuntimeError("Камера не доступна")

# Инициализация модуля распознавания людей
human_model = init_human_detection()

# Инициализация модуля распознавания посторонних объектов
object_detector = init_object_detection(cap)

# Инициализация модели сегментации людей
segmentation_model = init_human_segmentation()

# Главная страница
@app.route('/')
def index():
    return render_template('index.html')

# Страница для распознавания объектов
@app.route('/object_detection')
def object_detection():
    return render_template('object_detection.html')

# Страница для распознавания людей
@app.route('/human_detection')
def human_detection():
    return render_template('human_detection.html')

# Страница для сегментации людей (заглушка)
@app.route('/human_segmentation')
def human_segmentation():
    return render_template('human_segmentation.html')

# Видеопоток для распознавания объектов
@app.route('/video_feed_object')
def video_feed_object():
    return Response(generate_object_frames(cap, object_detector), mimetype='multipart/x-mixed-replace; boundary=frame')

# Видеопоток для распознавания людей
@app.route('/video_feed_human')
def video_feed_human():
    return Response(generate_human_frames(cap, human_model), mimetype='multipart/x-mixed-replace; boundary=frame')

# Видеопоток для сегментации людей
@app.route('/video_feed_human_segmentation')
def video_feed_human_segmentation():
    return Response(generate_human_segmentation_frames(cap, segmentation_model), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4999, debug=False)