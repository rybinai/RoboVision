import cv2
import torch
import numpy as np
from ultralytics import YOLO

def init_human_segmentation():
    # Загрузка модели сегментации YOLO
    model = YOLO("models/yolo11n-seg.pt")
    return model

def generate_human_segmentation_frames(cap, model):

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Получение сегментации
        results = model(frame, task="segment", show=False, show_labels=False, show_boxes=False)

        for result in results:
            if result.masks is not None:
                mask = result.masks.data.cpu().numpy()  # Преобразуем маски в numpy
                
                for m in mask:
                    resized_mask = cv2.resize(m, (frame.shape[1], frame.shape[0]))  # Изменяем размер маски
                    mask_binary = (resized_mask > 0.5).astype(np.uint8) * 255  # Преобразуем в бинарную маску

                    # Создаем цветную маску (зеленую)
                    color_mask = np.zeros_like(frame, dtype=np.uint8)
                    color_mask[:] = (0, 255, 0)  # Зеленый цвет

                    # Применяем битовую маску
                    masked_color = cv2.bitwise_and(color_mask, color_mask, mask=mask_binary)

                    # Накладываем с прозрачностью 50%
                    frame = cv2.addWeighted(frame, 1, masked_color, 0.5, 0)

        # Кодируем в JPEG
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")