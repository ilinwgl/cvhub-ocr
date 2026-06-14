import pytest
import numpy as np
import cv2


@pytest.fixture
def det_test_image():
    return cv2.imread("tests/assets/images/det_test.png")

@pytest.fixture
def rec_test_image():
    return cv2.imread("tests/assets/images/rec_test.png")