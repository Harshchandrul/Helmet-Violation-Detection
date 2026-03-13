from src.exception import CustomException
from src.logger import logging
from src.config import OUTPUT_WIDTH, OUTPUT_HEIGHT
import sys
import cv2


def resize_frame(frame, width: int, height: int):
    if frame is None:
        raise CustomException("Got empty frame in resize_frame()", sys)

    try:
        return cv2.resize(frame, (width, height))
    except Exception as e:
        logging.error("Failed to resize frame")
        raise CustomException(e, sys)


def transform_frame(frame, width: int = OUTPUT_WIDTH, height: int = OUTPUT_HEIGHT):
    """
    Main preprocessing function used by the pipeline.
    Keep it minimal: resize only (for now).
    """
    return resize_frame(frame, width, height)
