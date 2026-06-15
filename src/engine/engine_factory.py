from src.engine.base_engine import BaseEngine
from src.models import EngineType


class EngineFactory:
    @staticmethod
    def create_engine(engine_type: EngineType, config_data: dict) -> BaseEngine:
        if engine_type == EngineType.PADDLE_OCR:
            from src.engine.paddle_engine import PaddleOCREngine
            engine = PaddleOCREngine()
            engine.build(config_data)
            return engine
        
        raise ValueError(f"Unsupported engine type: {engine_type.value}")