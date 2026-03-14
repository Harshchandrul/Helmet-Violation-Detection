from src.exception import CustomException
from src.logger import logging
from src.config import TRACKER_MODEL  # add this to config: TRACKER_MODEL = "yolo11m.pt"
import sys
from ultralytics import YOLO


class PersonMotorcycleTracker:
    """
    Uses pretrained YOLOv11m only for detecting and tracking persons and motorcycles.
    Helmet classification is done elsewhere (postprocess/drawing) with models/yolo/best.pt.
    """

    def __init__(self, model_path: str = None, device=None):
        self.model_path = model_path or TRACKER_MODEL
        self.device = device
        self.model = self._load_model()

    def _load_model(self):
        try:
            logging.info(f"Loading tracker model: {self.model_path}")
            model = YOLO(self.model_path)
            # Moving model from device to gpu
            if self.device is not None:
                model.to(self.device)
            logging.info("Tracker model loaded successfully")
            return model
        except Exception as e:
            logging.error("Failed to load tracker model")
            raise CustomException(e, sys)

    def track(self, frame):
        if frame is None:
            raise CustomException(
                "Received empty frame in PersonMotorcycleTracker.track()", sys
            )
        try:
            results = self.model.track(
                frame,
                imgsz=960,
                tracker="bytetrack.yaml",
                persist=True,
                conf=0.15,
                iou=0.5,
                classes=[0, 3],
            )
            return results
        except Exception as e:
            logging.error("Failed during YOLO track()")
            raise CustomException(e, sys)
