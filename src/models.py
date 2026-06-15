from dataclasses import dataclass
from enum import Enum

@dataclass
class DetectionResult:
    bbox: list[list[int]]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    confidence: float

@dataclass
class RecognitionResult:
    text: str
    confidence: float

@dataclass
class OCRResult:
    det_result: DetectionResult
    rec_result: RecognitionResult

class EngineType(str, Enum):
    """Supported OCR engine types."""
    PADDLE_OCR = "PADDLE_OCR"
