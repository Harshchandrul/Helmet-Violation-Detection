

def run(video_path=None, output_path=None):

    from src.ingestion.video_reader import VideoReader, OutputVideoWriter
    from src.preprocessing.frame_transform import transform_frame
    from src.config import FRAME_SKIP
    from src.inference.detector import PersonMotorcycleTracker
    from src.inference.postprocess import RiderHelmetPostProcessor
    import os

    # Project root = directory containing application.py / app.py
    root = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(root, "assets", "videos", "Traffic_video_1.mp4")
    # Then pass to VideoReader:
    reader = VideoReader(video_path=video_path)
    detector = PersonMotorcycleTracker()
    postprocessor = RiderHelmetPostProcessor()

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
            annotated_frame = postprocessor.process(results, frame, frame_count)

            writer.write_frame(annotated_frame)

    # writer.release()
