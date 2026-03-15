# So this config file will contain all the configurations for the project like the path of the data, the model path , constants, etc.


DATA_PATH = "data/dataset.yaml"
OVERLAP_THRESHOLD = 0.2
FRAME_SKIP = 5
N_FRAMES = 5
RIDER_TIMEOUT = 3
EARLY_FRAMES = 3
OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080
VIDEO_PATH = "assets/videos/Traffic_video_1.mp4"
OUTPUT_VIDEO_PATH = "assets/videos/output_video.mp4"
TRACKER_MODEL = "yolo11m.pt"
HELMET_MODEL_PATH = "models/yolo/best.pt"
RIDER_COAST_FRAMES = 60
