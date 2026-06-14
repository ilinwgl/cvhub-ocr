import pytest
from src.models import RecognitionResult
from src.recognizer.paddle_recognizer import PaddleRecognizer
from src.config.config_loader import get_paddleocr_config


def test_recognizer_loads_model():
    recognizer_config_data = get_paddleocr_config()["recognizer"]
    recognizer = PaddleRecognizer(recognizer_config_data)
    recognizer.load_model()
    assert recognizer.recognizer is not None


def test_recognizer_detects(rec_test_image):
    recognizer_config_data = get_paddleocr_config()["recognizer"]
    recognizer = PaddleRecognizer(recognizer_config_data)
    recognizer.load_model()
    results = recognizer.recognize(rec_test_image)
    print(results)
    assert isinstance(results, RecognitionResult|None)