import unittest
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../modules')))

with patch('ultralytics.YOLO'):
    from human_detection import HumanDetector
    from object_detection import ObjectDetector
    from human_segmentation import HumanSegmentation

class TestHumanSegmentation(unittest.TestCase):
    def setUp(self):
        self.model = HumanSegmentation()
        self.model.model = MagicMock()
        self.test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(self.test_frame, (100, 100), (200, 300), (255, 255, 255), -1)

    # Проверяет корректность обработки кадра с сегментацией человека
    def test_process_frame(self):
        mock_result = MagicMock()
        mock_mask = np.zeros((1, 160, 160), dtype=np.float32)
        mock_mask[0, 50:100, 50:100] = 1.0
        mock_result.masks.data.cpu.return_value.numpy.return_value = mock_mask
        self.model.model.return_value = [mock_result]

        processed_frame = self.model.process_frame(self.test_frame)
        self.assertEqual(processed_frame.shape, self.test_frame.shape)
        self.assertFalse(np.array_equal(processed_frame, self.test_frame))

    # Проверяет генерацию кадров видеопотока с сегментацией
    def test_generate_frames(self):
        mock_cap = MagicMock()
        mock_cap.read.side_effect = [
            (True, self.test_frame),
            (False, None)
        ]
        
        frames = list(self.model.generate_frames(mock_cap))
        self.assertGreater(len(frames), 0)
        self.assertIn(b'Content-Type: image/jpeg', frames[0])

class TestObjectDetector(unittest.TestCase):
    def setUp(self):
        self.mock_cap = MagicMock()
        self.mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        self.detector = ObjectDetector(self.mock_cap)
        self.test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.test_gray = np.zeros((480, 640), dtype=np.uint8)

    # Проверяет корректность захвата фонового изображения
    def test_capture_background(self):
        background = self.detector._capture_background()
        self.assertEqual(background.shape, (480, 640))
        self.assertEqual(background.dtype, np.uint8)

    # Проверяет преобразование кадра в оттенки серого с размытием
    def test_preprocess_frame(self):
        processed = self.detector._preprocess_frame(self.test_frame)
        self.assertEqual(processed.shape, (480, 640))
        self.assertEqual(processed.dtype, np.uint8)

    # Проверяет обнаружение различий между текущим кадром и фоном
    def test_process_difference(self):
        current_gray = np.zeros((480, 640), dtype=np.uint8)
        current_gray[100:150, 100:150] = 50
        diff = self.detector._process_difference(current_gray)
        self.assertEqual(diff.shape, (480, 640))
        self.assertTrue(np.any(diff > 0))

    # Проверяет отрисовку контуров обнаруженных объектов
    def test_draw_contours(self):
        diff = np.zeros((480, 640), dtype=np.uint8)
        diff[100:150, 100:150] = 255
        processed = self.detector._draw_contours(self.test_frame.copy(), diff)
        self.assertFalse(np.array_equal(processed, self.test_frame))

    # Проверяет генерацию кадров с обнаруженными объектами
    def test_generate_frames(self):
        self.mock_cap.read.side_effect = [
            (True, self.test_frame),
            (False, None)
        ]
        frames = list(self.detector.generate_frames())
        self.assertGreater(len(frames), 0)
        self.assertIn(b'Content-Type: image/jpeg', frames[0])

class TestHumanDetector(unittest.TestCase):
    def setUp(self):
        self.detector = HumanDetector()
        self.detector.model = MagicMock()
        self.test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        self.mock_result = MagicMock()
        self.mock_boxes = MagicMock()
        self.mock_result.boxes = self.mock_boxes

        self.mock_boxes.id = MagicMock()
        self.mock_boxes.id.cpu.return_value.numpy.return_value = np.array([1], dtype=int)
        self.mock_boxes.xyxy = MagicMock()
        self.mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[100, 100, 200, 200]], dtype=float)
        
        self.detector.model.track.return_value = [self.mock_result]

    # Проверяет обработку кадра с обнаружением и трекингом людей
    def test_process_frame(self):
        with patch.object(self.detector, '_get_color_for_id', return_value=(0, 255, 0)):
            processed = self.detector.process_frame(self.test_frame.copy())
            green_pixels = np.all(processed[100:200, 100:200] == [0, 255, 0], axis=-1)
            self.assertTrue(np.any(green_pixels))
            self.assertFalse(np.array_equal(processed, self.test_frame))

    # Проверяет генерацию уникальных цветов для разных ID
    def test_get_color_for_id(self):
        color1 = self.detector._get_color_for_id(1)
        color2 = self.detector._get_color_for_id(2)
        self.assertEqual(len(color1), 3)
        self.assertNotEqual(color1, color2)

    # Проверяет генерацию кадров с обнаруженными людьми
    def test_generate_frames(self):
        mock_cap = MagicMock()
        mock_cap.read.side_effect = [
            (True, self.test_frame),
            (False, None)
        ]
        frames = list(self.detector.generate_frames(mock_cap))
        self.assertGreater(len(frames), 0)
        self.assertIn(b'Content-Type: image/jpeg', frames[0])

if __name__ == '__main__':
    unittest.main()