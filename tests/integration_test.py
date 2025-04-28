import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import app

class FlaskAppTestCase(unittest.TestCase):
    def setUp(self):
        # Создаем тестовый клиент Flask
        self.app = app.test_client()
        self.app.testing = True

    def test_index_route(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<html', response.data)

    def test_object_detection_route(self):
        response = self.app.get('/object_detection')
        self.assertEqual(response.status_code, 200)

    def test_human_detection_route(self):
        response = self.app.get('/human_detection')
        self.assertEqual(response.status_code, 200)

    def test_human_segmentation_route(self):
        response = self.app.get('/human_segmentation')
        self.assertEqual(response.status_code, 200)

    def test_video_feed_object(self):
        response = self.app.get('/video_feed_object')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, 'multipart/x-mixed-replace')

    def test_video_feed_human(self):
        response = self.app.get('/video_feed_human')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, 'multipart/x-mixed-replace')

    def test_video_feed_human_segmentation(self):
        response = self.app.get('/video_feed_human_segmentation')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, 'multipart/x-mixed-replace')

if __name__ == '__main__':
    unittest.main()