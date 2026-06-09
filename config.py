import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(PROJECT_DIR, "images")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

# Rectified board size in pixels
BOARD_SIZE = 800
# Red border width as fraction of board (crop after warp)
BORDER_MARGIN = 0.05

# Red border HSV ranges (OpenCV H: 0-180)
RED_LOWER1 = (0, 70, 50)
RED_UPPER1 = (10, 255, 255)
RED_LOWER2 = (170, 70, 50)
RED_UPPER2 = (180, 255, 255)

CAMERA_INDEX = 0
