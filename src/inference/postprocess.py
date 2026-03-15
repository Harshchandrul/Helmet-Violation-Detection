# from notebooks.main import data
from src.config import (
    OVERLAP_THRESHOLD,
    EARLY_FRAMES,
    FRAME_SKIP,
    HELMET_MODEL_PATH,
    RIDER_COAST_FRAMES,
    RIDER_TIMEOUT,
    N_FRAMES,
)
from src.logger import logging
from src.exception import CustomException
from collections import defaultdict, deque
from ultralytics import YOLO
import sys, os
import cv2
import json


def overlap(box_a, box_b):  # box_a -> person_box | box_b -> motorcycle_box
    """
    Compute overlap ratio between two boxes based on intersection area
    over the smaller box area.
    box = (x1, y1, x2, y2)
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter_area / float(min(area_a, area_b))


class RiderHelmetPostProcessor:
    """
    Maintains cross-frame state to:
    1) Associate persons with motorcycles (identify riders).
    2) Run helmet classifier on rider head crops.
    3) Decide helmet/no_helmet status and draw labels/boxes.
    """

    def __init__(self):
        # Association state
        self.association_history = defaultdict(
            lambda: deque(maxlen=10)
        )  # (pid, mid) -> deque[bool]
        self.first_seen = {}  # pid -> frame_count
        self.riders = (
            {}
        )  # pid -> {"bike": mid, "last_seen": frame_count, "last_person_box", "last_bike_box"}
        # Helmet state
        self.helmet_history = defaultdict(lambda: deque(maxlen=7))  # pid -> deque[bool]
        self.helmet_status = {}  # pid -> "helmet" | "no_helmet" | "unknown"
        self.violation_logged = set()  # pids already logged to disk

        # Helmet model (trained classifier on head crops)
        try:
            logging.info(f"Loading helmet model from: {HELMET_MODEL_PATH}")
            self.helmet_model = YOLO(HELMET_MODEL_PATH)
            logging.info("Helmet model loaded successfully")
        except Exception as e:
            logging.error("Failed to load helmet model")
            raise CustomException(e, sys)

    # ------------------------------------------------------------------ #
    #   Parsing YOLO results
    # ------------------------------------------------------------------ #

    def _parse_results(self, results):
        """
        Convert Ultralytics results into simple dicts:
            persons: pid -> (x1, y1, x2, y2)
            motorcycles: mid -> (x1, y1, x2, y2)
        """
        r = results[0]
        if r.boxes.id is None:
            return {}, {}
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        track_ids = r.boxes.id.cpu().numpy().astype(int)
        persons = {}
        motorcycles = {}
        for box, cls_id, track_id in zip(boxes, classes, track_ids):
            x1, y1, x2, y2 = box.astype(int)
            if cls_id == 0:  # person
                persons[track_id] = (x1, y1, x2, y2)
            elif cls_id == 3:  # motorcycle
                motorcycles[track_id] = (x1, y1, x2, y2)
        return persons, motorcycles

    # ------------------------------------------------------------------ #
    #   Rider association
    # ------------------------------------------------------------------ #
    def _lock(self, pid, mid, frame_count, persons, motorcycles):
        self.riders[pid] = {
            "bike": mid,
            "last_seen": frame_count,
            "last_person_box": persons[pid],
            "last_bike_box": motorcycles[mid],
        }

    def _expire_riders(self, frame_count):
        """
        Remove riders who have not been seen for too long.
        """
        expired = []
        for pid, data in self.riders.items():
            if frame_count - data["last_seen"] > RIDER_COAST_FRAMES * FRAME_SKIP:
                expired.append(pid)
        for pid in expired:
            del self.riders[pid]
            self.helmet_history.pop(pid, None)
            self.helmet_status.pop(pid, None)

    def _update_riders(self, persons, motorcycles, frame_count):
        """
        Update rider associations for this frame.
        """
        # For each person, try to find a matching motorcycle
        for pid, pbox in persons.items():

            # If already a rider, just update last_seen and skip association
            if pid in self.riders:
                self.riders[pid]["last_seen"] = frame_count
                continue
            px1, py1, px2, py2 = pbox
            ph = py2 - py1
            person_lower = (px1, int(py1 + ph * 0.5), px2, py2)

            # Track first appearance
            if pid not in self.first_seen:
                self.first_seen[pid] = frame_count

            # Try to associate with each motorcycle
            for mid, mbox in motorcycles.items():
                mx1, my1, mx2, my2 = mbox
                mh = my2 - my1
                motorcycle_upper = (mx1, my1, mx2, int(my1 + mh * 0.5))

                ov = overlap(person_lower, motorcycle_upper)
                key = (pid, mid)
                ov_okay = ov > OVERLAP_THRESHOLD

                # Centers
                pcx = (px1 + px2) // 2
                pcy = (py1 + py2) // 2
                mcx = (mx1 + mx2) // 2
                mcy = (my1 + my2) // 2

                # Heuristics from notebook
                vertical_ok = py2 < my2 + 0.2 * mh
                horizontal_ok = abs(pcx - mcx) < 0.4 * (mx2 - mx1)

                evidence = sum([ov_okay, vertical_ok, horizontal_ok])

                # Only track positive evidence
                if evidence < 2:
                    continue

                self.association_history[key].append(True)

                # Fast provisional lock (early frames)
                if frame_count - self.first_seen[pid] <= EARLY_FRAMES:
                    if evidence >= 2 and ov > 0.05:
                        self._lock(pid, mid, frame_count, persons, motorcycles)
                        break

                # Normal lock: enough positive history
                if sum(self.association_history[key]) >= N_FRAMES:
                    self._lock(pid, mid, frame_count, persons, motorcycles)
                    break

        # Cleanup old riders
        self._expire_riders(frame_count)

    # ------------------------------------------------------------------ #
    #   Helmet classification + labels
    # ------------------------------------------------------------------ #
    def _run_helmet_on_riders(self, frame, persons, frame_count, annotated_frame):
        """
        For each rider, crop head region, run helmet model, update state
        and draw labels on annotated_frame.
        """
        for rider_id in self.riders:
            if rider_id not in persons:
                continue
            px1, py1, px2, py2 = persons[rider_id]
            person_height = py2 - py1
            head_y2 = int(py1 + 0.35 * person_height)
            head_crop = frame[py1:head_y2, px1:px2]
            if head_crop.size == 0:
                continue
            try:
                helmet_results = self.helmet_model.predict(
                    head_crop,
                    conf=0.1,
                    imgsz=256,
                    verbose=False,
                )
            except Exception as e:
                logging.error("Helmet model prediction failed")
                raise CustomException(e, sys)
            has_helmet = False
            if len(helmet_results[0].boxes) > 0:
                for cls_tensor in helmet_results[0].boxes.cls:
                    if int(cls_tensor) == 0:  # class 0 = helmet
                        has_helmet = True
                        break
            self.helmet_history[rider_id].append(has_helmet)
            # Initialize status if not present
            if rider_id not in self.helmet_status:
                self.helmet_status[rider_id] = "unknown"

            history = self.helmet_history[rider_id]
            ones = sum(history)
            # UNKNOWN → decide after a few frames
            if self.helmet_status[rider_id] == "unknown":
                if len(history) >= 2:
                    if ones >= 2:
                        self.helmet_status[rider_id] = "helmet"
                    else:
                        self.helmet_status[rider_id] = "no_helmet"

            # HELMET → keep as helmet (never downgrade)
            elif self.helmet_status[rider_id] == "helmet":
                pass

            # NO_HELMET → allow upgrade if strong evidence of helmet later
            elif self.helmet_status[rider_id] == "no_helmet":
                if ones >= 5:
                    self.helmet_status[rider_id] = "helmet"

            # Violation logging (optional, similar to notebook)
            if (
                self.helmet_status[rider_id] == "no_helmet"
                and rider_id not in self.violation_logged
            ):
                rider_dir = os.path.join("violations", f"rider_{rider_id}")
                os.makedirs(rider_dir, exist_ok=True)
                # Save full frame
                cv2.imwrite(os.path.join(rider_dir, "frame.jpg"), frame)
                # Save rider crop
                rider_crop = frame[py1:py2, px1:px2]
                cv2.imwrite(os.path.join(rider_dir, "rider.jpg"), rider_crop)
                # Save head crop
                cv2.imwrite(os.path.join(rider_dir, "head.jpg"), head_crop)
                meta = {
                    "rider_id": int(rider_id),
                    "frame": int(frame_count),
                    "helmet": "no_helmet",
                    "bbox": [int(px1), int(py1), int(px2), int(py2)],
                }
                with open(os.path.join(rider_dir, "meta.json"), "w") as f:
                    json.dump(meta, f, indent=2)
                self.violation_logged.add(rider_id)
            # Draw label on annotated frame
            label = self.helmet_status.get(rider_id, "unknown")
            color = (0, 255, 0) if label == "helmet" else (0, 0, 255)
            cv2.putText(
                annotated_frame,
                label.upper(),
                (px1, max(py1 - 10, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

    # ------------------------------------------------------------------ #
    #   Drawing rider boxes
    # ------------------------------------------------------------------ #
    def _draw_rider_boxes(self, persons, annotated_frame):
        for pid, data in self.riders.items():
            if pid not in persons:
                continue
            px1, py1, px2, py2 = persons[pid]
            cv2.rectangle(
                annotated_frame,
                (px1, py1),
                (px2, py2),
                (0, 0, 255),
                2,
            )

    # ------------------------------------------------------------------ #
    #   Public API
    # ------------------------------------------------------------------ #
    def process(self, results, frame, frame_count):
        """
        Main entrypoint called from app.py.
        Args:
            results: Ultralytics model.track(...) results list
            frame:   current frame (after resize)
            frame_count: global frame counter s ,
        Returns:
            annotated_frame: frame with tracking, rider boxes, helmet labels
        """
        if frame is None:
            raise CustomException("Empty frame passed to postprocess", sys)
        persons, motorcycles = self._parse_results(results)
        # Base annotated frame from YOLO (already has boxes/tracks)
        annotated_frame = results[0].plot()
        # Update rider associations
        self._update_riders(persons, motorcycles, frame_count)
        # Helmet classification + labels
        self._run_helmet_on_riders(frame, persons, frame_count, annotated_frame)
        # Draw rider boxes
        self._draw_rider_boxes(persons, annotated_frame)
        return annotated_frame
