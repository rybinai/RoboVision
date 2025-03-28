import cv2
import torch
import numpy as np
from ultralytics import YOLO

class HumanSegmentation:
    def __init__(self, model_path='models/yolo11n-seg.pt'):
        self.model = YOLO(model_path)
        self.inference_params = {
            'task': 'segment',
            'show': False,
            'show_labels': False,
            'show_boxes': False,
            'verbose': False  # Отключаем вывод информации в терминал
        }

    def process_frame(self, frame):
        results = self.model(frame, **self.inference_params)
        
        for result in results:
            if result.masks is not None:
                mask = result.masks.data.cpu().numpy()
                
                for m in mask:
                    resized_mask = cv2.resize(m, (frame.shape[1], frame.shape[0]))
                    mask_binary = (resized_mask > 0.5).astype(np.uint8) * 255
                    
                    color_mask = np.zeros_like(frame, dtype=np.uint8)
                    color_mask[:] = (0, 255, 0)  # Зеленый цвет
                    masked_color = cv2.bitwise_and(color_mask, color_mask, mask=mask_binary)
                    
                    frame = cv2.addWeighted(frame, 1, masked_color, 0.5, 0)
        
        return frame
    
    def generate_frames(self, video_capture):
        while True:
            success, frame = video_capture.read()
            if not success:
                break
            
            processed_frame = self.process_frame(frame)
            
            _, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def init_human_segmentation():
    return HumanSegmentation()

def generate_human_segmentation_frames(cap, model):
    return model.generate_frames(cap)
