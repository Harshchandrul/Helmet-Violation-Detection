from src.exception import CustomException
from src.logger import logging
from src.config import (
    VIDEO_PATH,
    OUTPUT_VIDEO_PATH,
    OUTPUT_WIDTH,
    OUTPUT_HEIGHT,
    FRAME_SKIP,
)
import cv2
import os
import sys


class VideoReader:
    def __init__(self, video_path=None):
        self.video_path = video_path if video_path else VIDEO_PATH
        self.cap = None
        self.fps = None

        logging.info(f"Video path: {self.video_path}")

    def open_video_file(self):
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                logging.error("Failed to open video file")
                raise CustomException("Failed to open video file", sys)

            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0:
                self.fps = 25
                logging.info(f"FPS not found, using default: {self.fps}")

            return self

        except Exception as e:
            logging.error("Failed to open video capture")
            raise CustomException(e, sys)

    def read_frame(self):
        ret, frame = self.cap.read()
        return (ret, frame)

    def is_opened(self):
        return self.cap is not None and self.cap.isOpened()

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logging.info("Video file released")

    def get_fps(self):
        return self.fps

    def __del__(self):
        if hasattr(self, "cap") and self.cap is not None:
            self.release()

    def __enter__(self):
        self.open_video_file()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()


class OutputVideoWriter:
    def __init__(self, output_video_path=None):
        self.output_video_path = (
            output_video_path if output_video_path else OUTPUT_VIDEO_PATH
        )
        self.out = None
        self.fps = None
        self.width = None
        self.height = None

        logging.info(f"Output video path: {self.output_video_path}")

    def open_writer(self, output_path=None, fps=None, width=None, height=None):
        self.output_path = output_path if output_path else self.output_video_path
        self.width = width if width else OUTPUT_WIDTH
        self.height = height if height else OUTPUT_HEIGHT

        if fps is None:
            raise ValueError("fps must be provided (e.g. from VideoReader.get_fps())")

        self.fps = fps
        out_fps = self.fps / FRAME_SKIP  # we write every FRAME_SKIP-th frame

        try:
            output_dir = os.path.dirname(self.output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.out = cv2.VideoWriter(
                self.output_path,
                fourcc,
                out_fps,
                (self.width, self.height),
            )

            if not self.out.isOpened():
                logging.error("Failed to create output video writer")
                raise CustomException("Failed to create output video writer", sys)

            logging.info(
                f"Output writer opened: {self.output_path} @ {out_fps:.1f} fps, {self.width}x{self.height}"
            )
            return self

        except Exception as e:
            logging.error("Failed to open output video writer")
            raise CustomException(e, sys)

    def write_frame(self, frame):
        if self.out is None:
            raise CustomException(
                "Output writer not opened. Call open_writer() first.", sys
            )
        self.out.write(frame)

    def release(self):
        if self.out is not None:
            self.out.release()
            self.out = None
            logging.info("Output video writer released")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
