import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Добавляем пути к модулям проекта
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../modules')))

class TestAppRoutes(unittest.TestCase):
    def setUp(self):
        # Создаем моки для всех зависимостей
        self.mocks = {
            'cv2.VideoCapture': MagicMock(),
            'modules.human_detection.init_human_detection': MagicMock(),
            'modules.object_detection.init_object_detection': MagicMock(),
            'modules.human_segmentation.init_human_segmentation': MagicMock()
        }
        
        # Применяем моки
        self.patchers = [
            patch(name, mock) for name, mock in self.mocks.items()
        ]
        
        for patcher in self.patchers:
            patcher.start()
        
        from app import app
        self.app = app
        self.client = app.test_client()

    def tearDown(self):
        for patcher in self.patchers:
            patcher.stop()

    #Тестирование маршрута видео-потока для обнаружения объектов.
    def test_video_feed_object_route(self):
        mock_generator = MagicMock(return_value=[b'fake_frame'])
        with patch('app.object_detector.generate_frames', mock_generator):
            response = self.client.get('/video_feed_object')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.mimetype, 'multipart/x-mixed-replace')
    #Тестирование маршрута видео-потока для обнаружения людей.
    def test_video_feed_human_route(self):
        mock_generator = MagicMock(return_value=[b'fake_frame'])
        with patch('app.human_model.generate_frames', mock_generator):
            response = self.client.get('/video_feed_human')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.mimetype, 'multipart/x-mixed-replace')
    #Тестирование маршрута видео-потока для сегментирования людей.
    def test_video_feed_segmentation_route(self):
        mock_generator = MagicMock(return_value=[b'fake_frame'])
        with patch('app.segmentation_model.generate_frames', mock_generator):
            response = self.client.get('/video_feed_human_segmentation')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.mimetype, 'multipart/x-mixed-replace')

if __name__ == '__main__':
    unittest.main()