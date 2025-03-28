import cv2
from ultralytics import YOLO
import random

class HumanDetector:
    def __init__(self, model_path='models/people.pt', tracker_config="models/botsort.yaml"):
        self.model = YOLO(model_path)
        self.model.fuse()
        self.tracker_config = tracker_config
        self.tracking_params = {
            'iou': 0.4,
            'conf': 0.5,
            'persist': True,
            'imgsz': 608,
            'verbose': False
        }
    
    def process_frame(self, frame):
        results = self.model.track(frame, tracker=self.tracker_config, **self.tracking_params)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            for box, id in zip(boxes, ids):
                color = self._get_color_for_id(id)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(frame, f"Id {id}", (box[0], box[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        return frame
    
    def _get_color_for_id(self, obj_id):
        random.seed(int(obj_id))
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
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

def init_human_detection():
    return HumanDetector()

def generate_human_frames(cap, model):
    return model.generate_frames(cap)