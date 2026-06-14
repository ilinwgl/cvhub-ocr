import pytest
from src.detector.paddle_detector import PaddleDetector
from src.config.config_loader import get_paddleocr_config


def test_detector_loads_model():
    detector_config_data = get_paddleocr_config()["detector"]
    detector = PaddleDetector(detector_config_data)
    detector.load_model()
    assert detector.detector is not None


def test_detector_detects(det_test_image):
    detector_config_data = get_paddleocr_config()["detector"]
    detector = PaddleDetector(detector_config_data)
    detector.load_model()
    results = detector.detect(det_test_image)
    print(results)
    assert isinstance(results, list)