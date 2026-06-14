import pytest
from pathlib import Path
from src.config.config_loader import load_config, get_paddleocr_config

def test_load_config_returns_dict():
    config = load_config()
    assert isinstance(config, dict)


def test_load_config_has_paddleocr_key():
    config = load_config()
    assert "paddleocr" in config


def test_get_paddleocr_config_returns_dict():
    config = get_paddleocr_config()
    assert isinstance(config, dict)


def test_get_paddleocr_config_has_detector():
    config = get_paddleocr_config()
    assert "detector" in config


def test_get_paddleocr_config_has_recognizer():
    config = get_paddleocr_config()
    assert "recognizer" in config


def test_detector_config_has_required_keys():
    config = get_paddleocr_config()
    detector = config["detector"]
    assert "model_name" in detector
    assert "model_dir" in detector
    assert "device" in detector


def test_recognizer_config_has_required_keys():
    config = get_paddleocr_config()
    recognizer = config["recognizer"]
    assert "model_name" in recognizer
    assert "model_dir" in recognizer
    assert "device" in recognizer


def test_detector_model_dir_exists():
    config = get_paddleocr_config()
    model_dir = Path(config["detector"]["model_dir"])
    assert model_dir.exists(), f"Detector model dir not found: {model_dir}"


def test_recognizer_model_dir_exists():
    config = get_paddleocr_config()
    model_dir = Path(config["recognizer"]["model_dir"])
    assert model_dir.exists(), f"Recognizer model dir not found: {model_dir}"