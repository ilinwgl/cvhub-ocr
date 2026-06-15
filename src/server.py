"""
OCR gRPC Server

Startup sequence:
    1. load config
    2. build engine once (detector + recognizer)
    3. serve requests: decode bytes → preprocess → engine.run → response
"""

import time
import logging
from concurrent import futures

import cv2
import grpc
import numpy as np

from proto import ocr_pb2, ocr_pb2_grpc
from src.config.config_loader import get_paddleocr_config
from src.preprocess.image_processor import Preprocessor
from src.engine.engine_factory import EngineFactory    # returns a concrete BaseEngine
from src.models import EngineType, OCRResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def _decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes → BGR numpy array."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image bytes")
    return img


def _preprocess(image: np.ndarray) -> np.ndarray:
    """Run preprocessing pipeline and return result array."""
    return (
        Preprocessor(image)
        .gaussian_denoise(kernel=3)
        .unsharp_mask()
        .result()
    )


def _ocr_result_to_proto(result: OCRResult) -> ocr_pb2.OCRResult: # type: ignore
    """Convert dataclass OCRResult → proto OCRResult message."""
    points = [
        ocr_pb2.Point(x=pt[0], y=pt[1]) # type: ignore
        for pt in result.det_result.bbox
    ]
    bbox = ocr_pb2.BoundingBox(points=points) # type: ignore

    det = ocr_pb2.DetectionResult( # type: ignore
        bbox=bbox,
        confidence=result.det_result.confidence,
    )
    rec = ocr_pb2.RecognitionResult( # type: ignore
        text=result.rec_result.text,
        confidence=result.rec_result.confidence,
    )
    return ocr_pb2.OCRResult(det_result=det, rec_result=rec) # type: ignore


def _process_single(image_bytes: bytes, engine) -> tuple[list[OCRResult], float]:
    """
    Full pipeline for one image.

    Returns:
        (ocr_results, elapsed_ms)
    """
    t0 = time.perf_counter()
    image = _decode_image(image_bytes)
    image = _preprocess(image)
    results = engine.run(image)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return results, elapsed_ms


# ------------------------------------------------------------------ #
#  Servicer
# ------------------------------------------------------------------ #

class OCRServicer(ocr_pb2_grpc.OCRServiceServicer): # type: ignore
    def __init__(self, engine) -> None:
        self._engine = engine

    def Recognize(
        self,
        request: ocr_pb2.OCRRequest, # type: ignore
        context: grpc.ServicerContext,
    ) -> ocr_pb2.OCRResponse: # type: ignore
        """Handle single-image OCR request."""
        try:
            results, elapsed_ms = _process_single(request.image, self._engine)
            proto_results = [_ocr_result_to_proto(r) for r in results]
            return ocr_pb2.OCRResponse( # type: ignore
                results=proto_results,
                processing_time_ms=elapsed_ms,
                error="",
            )
        except Exception as e:
            logger.exception("Recognize failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ocr_pb2.OCRResponse(error=str(e)) # type: ignore

    def BatchRecognize(
        self,
        request: ocr_pb2.BatchOCRRequest, # type: ignore
        context: grpc.ServicerContext,
    ) -> ocr_pb2.BatchOCRResponse: # type: ignore
        """Handle batch OCR request (one response per image)."""
        responses = []
        for req in request.requests:
            try:
                results, elapsed_ms = _process_single(req.image, self._engine)
                proto_results = [_ocr_result_to_proto(r) for r in results]
                responses.append(ocr_pb2.OCRResponse( # type: ignore
                    results=proto_results,
                    processing_time_ms=elapsed_ms,
                    error="",
                ))
            except Exception as e:
                logger.exception("BatchRecognize failed on one image")
                responses.append(ocr_pb2.OCRResponse(error=str(e))) # type: ignore
        return ocr_pb2.BatchOCRResponse(responses=responses) # type: ignore


# ------------------------------------------------------------------ #
#  Entry point
# ------------------------------------------------------------------ #

def serve(host: str = "0.0.0.0", port: int = 50051, max_workers: int = 4) -> None:
    logger.info(f"Loading default config ({EngineType.PADDLE_OCR.value})...")
    default_config = get_paddleocr_config()

    logger.info(f"Building default engine ({EngineType.PADDLE_OCR.value})...")
    default_engine = EngineFactory.create(EngineType.PADDLE_OCR, default_config)

    logger.info(f"Starting gRPC server on {host}:{port}")
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    ocr_pb2_grpc.add_OCRServiceServicer_to_server(OCRServicer(default_engine), grpc_server) # type: ignore
    grpc_server.add_insecure_port(f"{host}:{port}")
    grpc_server.start()

    logger.info("Server ready.")
    try:
        grpc_server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        grpc_server.stop(grace=5)


if __name__ == "__main__":
    serve()