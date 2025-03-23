import cv2
from ultralytics import YOLO
import random

def init_human_detection():
    # Инициализация YOLO модели
    model = YOLO('models/people.pt')
    model.fuse()
    return model

def generate_human_frames(cap, model):
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Обработка кадра YOLO
        results = model.track(frame, iou=0.4, conf=0.5, persist=True, imgsz=608, verbose=False, tracker="models/botsort.yaml")

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, id in zip(boxes, ids):
                random.seed(int(id))
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(frame, f"Id {id}", (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Кодировка в JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Поток
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')