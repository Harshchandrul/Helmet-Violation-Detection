# So this config file will contain all the configurations for the project like the path of the data, the model path , constants, etc.


DATA_PATH = "data/dataset.yaml"
# i have to put the model in the model folder so that i can use it later for inference and training.
MODEL_PATH = "models/yolov8n.pt"
OVERLAP_THRESHOLD = 0.2
FRAME_SKIP = 5
N_FRAMES = 5
RIDER_TIMEOUT = 3
EARLY_FRAMES = 3
FRAME_SKIP = 5
OUTPUT_WIDTH = 960
OUTPUT_HEIGHT = 540
VIDEO_PATH = "../assets/videos/Traffic_video_1.mp4"
OUTPUT_VIDEO_PATH = "../assets/videos/output_video.mp4"
