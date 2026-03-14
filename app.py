from src.ingestion.video_reader import VideoReader, OutputVideoWriter
from src.preprocessing.frame_transform import transform_frame
from src.config import FRAME_SKIP
from src.inference.detector import PersonMotorcycleTracker


detector = PersonMotorcycleTracker()
reader = VideoReader()

with reader:
    writer = OutputVideoWriter()
    writer.open_writer(fps=reader.get_fps())

    frame_count = 0
    while reader.is_opened():
        ret, frame = reader.read_frame()
        if not ret:
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        # ... resize, detect, annotate ...
        frame = transform_frame(frame)
        results = detector.track(frame)

        # TODO: pass results to postprocess + drawing
        # annotated_frame = ...

        # writer.write_frame(frame)
        # writer.release()
